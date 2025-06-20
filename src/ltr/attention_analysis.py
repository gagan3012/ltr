import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from baukit import TraceDict
import logging


def analyze_attention_patterns(
    model,
    tokenizer,
    prompt: str,
    target_heads: Optional[List[Tuple[int, int]]] = None,
    concepts: Optional[List[str]] = None,
) -> Dict:
    """
    Analyzes attention patterns in the model for a given prompt.

    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        prompt: Input text to analyze
        target_heads: List of (layer, head) tuples to analyze. If None, analyzes all heads
        concepts: List of concepts to track in attention patterns

    Returns:
        Dict containing attention analysis results
    """
    # Get model architecture information
    model_type = (
        model.config.model_type.lower() if hasattr(model.config, "model_type") else ""
    )

    # Configure layer and attention patterns based on model type
    if "llama" in model_type or "mistral" in model_type or "qwen" in model_type:
        # For decoder-only models like LLaMA, Mistral, Qwen
        attn_pattern = "model.layers.{}.self_attn"
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
    elif "gpt-neox" in model_type or "gpt_neox" in model_type:
        attn_pattern = "gpt_neox.layers.{}.attention"
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
    elif "gpt2" in model_type:
        attn_pattern = "transformer.h.{}.attn"
        n_layers = model.config.n_layer
        n_heads = model.config.n_head
    elif "falcon" in model_type:
        attn_pattern = "transformer.h.{}.self_attention"
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
    else:
        # Default pattern as fallback
        attn_pattern = "model.layers.{}.self_attn"
        n_layers = getattr(model.config, "num_hidden_layers", 12)
        n_heads = getattr(model.config, "num_attention_heads", 12)

    # Create attention layer patterns for tracing
    attn_patterns = [attn_pattern.format(layer) for layer in range(n_layers)]

    # If no target heads are specified, analyze a subset for efficiency
    if target_heads is None:
        # Sample every 4th layer and every 2nd head for efficiency
        target_heads = [
            (l, h)
            for l in range(0, n_layers, max(1, n_layers // 8))
            for h in range(0, n_heads, max(1, n_heads // 4))
        ]

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_tokens = inputs.input_ids.shape[1]

    # Prepare results structure
    results = {
        "prompt": prompt,
        "tokens": tokenizer.convert_ids_to_tokens(inputs.input_ids[0]),
        "token_strings": tokenizer.tokenize(prompt),
        "attention_maps": {},
        "head_importance": {},
        "concept_attention": {},
    }

    # Extract attention using forward hooks
    attention_weights = {}

    def attention_hook(module, input, output, layer_idx):
        """Hook to capture attention weights during forward pass"""
        try:
            # Handle different model architectures
            if hasattr(module, "attn_weights") and module.attn_weights is not None:
                # Some models store attention weights in attn_weights
                attn = module.attn_weights
            elif hasattr(output, "attentions") and output.attentions is not None:
                # Some models return attention in output
                attn = output.attentions
            elif len(output) > 1 and isinstance(output[1], torch.Tensor):
                # Attention weights as second element of tuple
                attn = output[1]
            else:
                # For models that don't expose attention weights directly,
                # we'll compute attention from Q, K matrices
                attn = compute_attention_from_qk(module, input[0])

            if attn is not None:
                # Store attention weights for this layer
                attention_weights[layer_idx] = attn.detach().cpu()

        except Exception as e:
            logging.warning(f"Could not extract attention for layer {layer_idx}: {e}")
            # Create dummy attention (uniform distribution)
            attention_weights[layer_idx] = (
                torch.ones(1, n_heads, n_tokens, n_tokens) / n_tokens
            )

    # Register hooks for attention extraction
    hooks = []
    for layer_idx, layer_pattern in enumerate(attn_patterns):
        try:
            # Navigate to the attention module
            module = model
            for attr in layer_pattern.split("."):
                module = getattr(module, attr)

            # Register hook
            hook = module.register_forward_hook(
                lambda mod, inp, out, idx=layer_idx: attention_hook(mod, inp, out, idx)
            )
            hooks.append(hook)

        except AttributeError as e:
            logging.warning(f"Could not register hook for {layer_pattern}: {e}")
            continue

    # Run forward pass to collect attention
    try:
        with torch.no_grad():
            _ = model(**inputs, output_attentions=True)
    except:
        # If output_attentions is not supported, run without it
        with torch.no_grad():
            _ = model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Process collected attention weights
    for layer_idx, attn_tensor in attention_weights.items():
        if attn_tensor is None:
            continue

        # Ensure tensor has the right shape [batch, heads, seq_len, seq_len]
        if attn_tensor.dim() == 3:
            attn_tensor = attn_tensor.unsqueeze(0)  # Add batch dimension

        batch_size, num_heads, seq_len, _ = attn_tensor.shape

        # Process each head
        for head_idx in range(min(num_heads, n_heads)):
            if (layer_idx, head_idx) not in target_heads:
                continue

            # Extract attention pattern for this head
            attn_pattern = attn_tensor[0, head_idx].numpy()

            if attn_pattern.shape[0] != n_tokens or attn_pattern.shape[1] != n_tokens:
                # Resize if necessary
                attn_pattern = np.eye(n_tokens)
                logging.warning(
                    f"Attention shape mismatch for layer {layer_idx}, head {head_idx}"
                )

            results["attention_maps"][(layer_idx, head_idx)] = attn_pattern

            # Calculate attention entropy as a measure of head focus
            attn_entropy = -np.sum(attn_pattern * np.log(attn_pattern + 1e-10), axis=1)
            results["head_importance"][(layer_idx, head_idx)] = 1.0 - (
                attn_entropy / np.log(n_tokens)
            )

    # If concepts are provided, analyze attention to these concepts
    if concepts and len(concepts) > 0:
        for concept in concepts:
            # Find token positions for this concept
            concept_positions = find_concept_positions(tokenizer, prompt, concept)

            if concept_positions:
                results["concept_attention"][concept] = {}

                # For each attention head, calculate average attention to concept tokens
                for (layer, head), attn_map in results["attention_maps"].items():
                    if len(concept_positions) > 0 and attn_map.shape[1] > max(
                        concept_positions
                    ):
                        concept_attention = np.mean(
                            [np.mean(attn_map[:, pos]) for pos in concept_positions]
                        )
                        results["concept_attention"][concept][(layer, head)] = (
                            concept_attention
                        )

    return results


def compute_attention_from_qk(attention_module, hidden_states):
    """
    Compute attention weights from Q and K matrices for models that don't expose them directly.
    """
    try:
        # This is a simplified computation - actual implementation depends on model architecture
        if hasattr(attention_module, "q_proj") and hasattr(attention_module, "k_proj"):
            # For LLaMA-style models
            bsz, seq_len, hidden_size = hidden_states.shape

            # Project to Q and K
            query_states = attention_module.q_proj(hidden_states)
            key_states = attention_module.k_proj(hidden_states)

            # Reshape for multi-head attention
            num_heads = getattr(attention_module, "num_heads", 12)
            head_dim = hidden_size // num_heads

            query_states = query_states.view(
                bsz, seq_len, num_heads, head_dim
            ).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, num_heads, head_dim).transpose(
                1, 2
            )

            # Compute attention scores
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (
                head_dim**0.5
            )
            attn_weights = torch.softmax(attn_weights, dim=-1)

            return attn_weights

        elif hasattr(attention_module, "query") and hasattr(attention_module, "key"):
            # For other model architectures
            bsz, seq_len, hidden_size = hidden_states.shape

            query_states = attention_module.query(hidden_states)
            key_states = attention_module.key(hidden_states)

            # Simplified attention computation
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
            attn_weights = torch.softmax(attn_weights / (hidden_size**0.5), dim=-1)

            # Add head dimension if missing
            if attn_weights.dim() == 3:
                attn_weights = attn_weights.unsqueeze(1)

            return attn_weights
        else:
            return None

    except Exception as e:
        logging.error(f"Error computing attention from Q/K: {e}")
        return None


def extract_attention_pattern(model, layer, head, inputs, output):
    """
    Extract attention pattern for a specific head.
    Implementation depends on model architecture.
    """
    # This function is now deprecated in favor of the hook-based approach above
    # Keeping for backward compatibility

    model_type = (
        model.config.model_type.lower() if hasattr(model.config, "model_type") else ""
    )
    n_tokens = inputs.input_ids.shape[1]

    try:
        # Try to extract from cached attention weights
        if hasattr(model, "_attention_cache") and layer in model._attention_cache:
            attn_weights = model._attention_cache[layer]
            if attn_weights.shape[1] > head:
                return attn_weights[0, head].detach().cpu().numpy()

        # Fallback: create identity matrix
        logging.warning(
            f"Could not extract attention pattern for layer {layer}, head {head}. Using identity matrix."
        )
        return np.eye(n_tokens)

    except Exception as e:
        logging.error(f"Error extracting attention pattern: {e}")
        return np.eye(n_tokens)


def find_concept_positions(tokenizer, prompt, concept):
    """
    Find token positions corresponding to a concept in the prompt.
    Enhanced version with better matching.
    """
    # Tokenize the prompt and concept
    prompt_tokens = tokenizer.tokenize(prompt)
    concept_tokens = tokenizer.tokenize(concept)

    if not concept_tokens:
        return []

    concept_len = len(concept_tokens)
    positions = []

    # Exact matching
    for i in range(len(prompt_tokens) - concept_len + 1):
        if prompt_tokens[i : i + concept_len] == concept_tokens:
            positions.extend(range(i, i + concept_len))

    # Fuzzy matching for subword tokens
    if not positions:
        concept_str = concept.lower()
        for i, token in enumerate(prompt_tokens):
            token_str = token.lower().replace("▁", "").replace("Ġ", "")
            if concept_str in token_str or token_str in concept_str:
                if len(token_str) > 2:  # Avoid very short matches
                    positions.append(i)

    return positions


def ablate_attention_patterns(
    model,
    tokenizer,
    prompt: str,
    target_heads: List[Tuple[int, int]],
    ablation_factor: float = 0.0,
) -> Dict:
    """
    Ablates specified attention heads to measure their importance.
    Enhanced version with better architecture support.
    """
    model_type = (
        model.config.model_type.lower() if hasattr(model.config, "model_type") else ""
    )

    # Configure layer patterns based on model type
    if "llama" in model_type or "mistral" in model_type or "qwen" in model_type:
        attn_pattern = "model.layers.{}.self_attn.o_proj"
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
    elif "gpt-neox" in model_type or "gpt_neox" in model_type:
        attn_pattern = "gpt_neox.layers.{}.attention.dense"
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
    elif "gpt2" in model_type:
        attn_pattern = "transformer.h.{}.attn.c_proj"
        n_layers = model.config.n_layer
        n_heads = model.config.n_head
    elif "falcon" in model_type:
        attn_pattern = "transformer.h.{}.self_attention.dense"
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
    else:
        attn_pattern = "model.layers.{}.self_attn.o_proj"
        n_layers = getattr(model.config, "num_hidden_layers", 12)
        n_heads = getattr(model.config, "num_attention_heads", 12)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_tokens = inputs.input_ids[0]
    token_strings = tokenizer.convert_ids_to_tokens(input_tokens)

    # Get baseline output
    with torch.no_grad():
        baseline_output = model(**inputs)
    baseline_logits = baseline_output.logits.detach().cpu()

    # Create a mapping of layer to heads for targeted ablation
    layer_to_heads = {}
    for layer, head in target_heads:
        if layer not in layer_to_heads:
            layer_to_heads[layer] = []
        layer_to_heads[layer].append(head)

    def ablation_hook(module, input, output, layer, heads):
        """Hook to ablate specific attention heads."""
        try:
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Get head dimension
            hidden_size = hidden_states.shape[-1]
            head_size = hidden_size // n_heads

            # Apply ablation to specified heads
            for head_idx in heads:
                if head_idx < n_heads:
                    start_idx = head_idx * head_size
                    end_idx = (head_idx + 1) * head_size

                    if ablation_factor == 0.0:
                        # Complete ablation
                        hidden_states[:, :, start_idx:end_idx] = 0
                    else:
                        # Partial ablation
                        hidden_states[:, :, start_idx:end_idx] *= ablation_factor

            # Return the modified output in the same format
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states

        except Exception as e:
            logging.error(f"Error in ablation hook: {e}")
            return output

    # Apply hooks for ablation
    hooks = []
    for layer, heads in layer_to_heads.items():
        try:
            layer_pattern = attn_pattern.format(layer)
            # Get the module from the model
            module = model
            for name in layer_pattern.split("."):
                if hasattr(module, name):
                    module = getattr(module, name)
                else:
                    logging.warning(f"Module {name} not found in {layer_pattern}")
                    break
            else:
                # Register the hook if we successfully found the module
                hook = module.register_forward_hook(
                    lambda mod, inp, out, layer=layer, heads=heads: ablation_hook(
                        mod, inp, out, layer, heads
                    )
                )
                hooks.append(hook)
        except Exception as e:
            logging.error(f"Error registering hook for layer {layer}: {e}")
            continue

    # Run the model with ablation
    with torch.no_grad():
        ablated_output = model(**inputs)
    ablated_logits = ablated_output.logits.detach().cpu()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Calculate KL divergence between baseline and ablated outputs
    baseline_probs = torch.softmax(baseline_logits, dim=-1)
    ablated_probs = torch.softmax(ablated_logits, dim=-1)
    kl_div = torch.sum(
        baseline_probs
        * (torch.log(baseline_probs + 1e-10) - torch.log(ablated_probs + 1e-10)),
        dim=-1,
    )

    # Get top tokens affected by ablation
    token_changes = []
    n_tokens = len(token_strings)

    for pos in range(min(n_tokens - 1, baseline_logits.shape[1] - 1)):
        baseline_top_tokens = torch.topk(baseline_logits[0, pos], k=5)
        ablated_top_tokens = torch.topk(ablated_logits[0, pos], k=5)

        baseline_top = [
            (tokenizer.decode([id.item()]).strip(), prob.item())
            for id, prob in zip(
                baseline_top_tokens.indices,
                torch.softmax(baseline_top_tokens.values, dim=-1),
            )
        ]
        ablated_top = [
            (tokenizer.decode([id.item()]).strip(), prob.item())
            for id, prob in zip(
                ablated_top_tokens.indices,
                torch.softmax(ablated_top_tokens.values, dim=-1),
            )
        ]

        token_changes.append(
            {
                "position": pos,
                "token": token_strings[pos] if pos < len(token_strings) else "<unk>",
                "baseline_top": baseline_top,
                "ablated_top": ablated_top,
                "kl_div": kl_div[0, pos].item() if pos < kl_div.shape[1] else 0.0,
            }
        )

    results = {
        "prompt": prompt,
        "tokens": token_strings,
        "target_heads": target_heads,
        "ablation_factor": ablation_factor,
        "token_changes": token_changes,
        "avg_kl_div": kl_div.mean().item(),
        "total_effect": (baseline_logits - ablated_logits).abs().mean().item(),
    }

    return results
