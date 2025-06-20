import numpy as np
import torch
import gc
from typing import Dict, List, Optional, Tuple, Union
from baukit import TraceDict
import logging


def perform_patching_control(
    model, 
    tokenizer, 
    clean_prompt: str,
    corrupted_prompt: str,
    target_layers: Optional[List[int]] = None,
    target_positions: Optional[List[int]] = None,
    patchtype: str = "resid"
) -> Dict:
    """
    Performs controlled patching experiments between clean and corrupted prompts.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        clean_prompt: Clean input prompt
        corrupted_prompt: Corrupted version of the prompt
        target_layers: List of layers to patch. If None, uses all layers
        target_positions: List of token positions to patch. If None, uses all positions
        patchtype: Type of activation to patch:
            - "resid": Residual stream after layernorm
            - "mlp": MLP output
            - "attn": Attention output
        
    Returns:
        Dict containing patching control results
    """
    # Get model architecture information
    model_type = model.config.model_type.lower() if hasattr(model.config, "model_type") else ""
    
    # Configure patterns based on model type and patchtype
    if "llama" in model_type or "mistral" in model_type or "qwen" in model_type:
        if patchtype == "resid":
            pattern = "model.layers.{}.input_layernorm"
        elif patchtype == "mlp":
            pattern = "model.layers.{}.mlp"
        elif patchtype == "attn":
            pattern = "model.layers.{}.self_attn"
        n_layers = model.config.num_hidden_layers
    elif "gpt-neox" in model_type or "gpt_neox" in model_type:
        if patchtype == "resid":
            pattern = "gpt_neox.layers.{}.input_layernorm"
        elif patchtype == "mlp":
            pattern = "gpt_neox.layers.{}.mlp"
        elif patchtype == "attn":
            pattern = "gpt_neox.layers.{}.attention"
        n_layers = model.config.num_hidden_layers
    elif "gpt2" in model_type:
        if patchtype == "resid":
            pattern = "transformer.h.{}.ln_1"
        elif patchtype == "mlp":
            pattern = "transformer.h.{}.mlp"
        elif patchtype == "attn":
            pattern = "transformer.h.{}.attn"
        n_layers = model.config.n_layer
    else:
        # Default pattern as fallback
        if patchtype == "resid":
            pattern = "model.layers.{}.input_layernorm"
        elif patchtype == "mlp":
            pattern = "model.layers.{}.mlp"
        elif patchtype == "attn":
            pattern = "model.layers.{}.attention"
        n_layers = getattr(model.config, "num_hidden_layers", 12)
    
    # Use all layers if not specified
    if target_layers is None:
        target_layers = list(range(n_layers))
    
    # Tokenize both prompts
    clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(model.device)
    corrupted_inputs = tokenizer(corrupted_prompt, return_tensors="pt").to(model.device)
    
    clean_tokens = tokenizer.convert_ids_to_tokens(clean_inputs.input_ids[0])
    corrupted_tokens = tokenizer.convert_ids_to_tokens(corrupted_inputs.input_ids[0])
    
    n_clean_tokens = len(clean_tokens)
    n_corrupted_tokens = len(corrupted_tokens)
    
    # Use all positions if not specified
    if target_positions is None:
        max_pos = min(n_clean_tokens, n_corrupted_tokens) - 1
        target_positions = list(range(max_pos))
    
    # Create patterns for tracing
    layer_patterns = [pattern.format(layer) for layer in target_layers]
    
    # Run model for baseline outputs
    with torch.no_grad():
        clean_outputs = model(**clean_inputs)
        corrupted_outputs = model(**corrupted_inputs)
    
    clean_logits = clean_outputs.logits.detach()
    corrupted_logits = corrupted_outputs.logits.detach()
    
    # Get output positions (last token position)
    clean_output_pos = clean_inputs.input_ids.shape[1] - 1
    corrupted_output_pos = corrupted_inputs.input_ids.shape[1] - 1
    
    # Get baseline outputs for the final position
    clean_output = clean_logits[0, clean_output_pos]
    corrupted_output = corrupted_logits[0, corrupted_output_pos]
    
    # Get top tokens for clean and corrupted outputs
    clean_top_k = torch.topk(clean_output, k=5)
    corrupted_top_k = torch.topk(corrupted_output, k=5)
    
    clean_top_tokens = [(tokenizer.decode([token.item()]).strip(), prob.item()) 
                        for token, prob in zip(clean_top_k.indices, torch.softmax(clean_top_k.values, dim=-1))]
    corrupted_top_tokens = [(tokenizer.decode([token.item()]).strip(), prob.item()) 
                           for token, prob in zip(corrupted_top_k.indices, torch.softmax(corrupted_top_k.values, dim=-1))]
    
    results = {
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "clean_tokens": clean_tokens,
        "corrupted_tokens": corrupted_tokens,
        "clean_top_tokens": clean_top_tokens,
        "corrupted_top_tokens": corrupted_top_tokens,
        "patching_results": {},
        "layer_effects": {}
    }
    
    # For each layer, trace through the model and gather activations
    with TraceDict(model, layer_patterns) as clean_traces:
        _ = model(**clean_inputs)
        
        clean_activations = {pattern: clean_traces[pattern].detach().clone() for pattern in layer_patterns}
    
    # For each layer and position, patch from clean to corrupted and measure effect
    for layer_idx, layer in enumerate(target_layers):
        layer_pattern = layer_patterns[layer_idx]
        layer_effects = []
        
        for pos in target_positions:
            if pos >= min(n_clean_tokens, n_corrupted_tokens):
                continue
            
            # Create a hook for patching
            def patch_activation_hook(module, input_tensors, output, cached_activation, position):
                patched_output = output.clone()
                
                # Patch based on the type of output
                if isinstance(output, tuple):
                    # Some modules return tuples, handle accordingly
                    if len(output) > 0 and isinstance(output[0], torch.Tensor):
                        patched_output_tensor = output[0].clone()
                        patched_output_tensor[0, position] = cached_activation[0, position]
                        patched_output = (patched_output_tensor,) + output[1:]
                else:
                    # Single tensor output
                    patched_output[0, position] = cached_activation[0, position]
                
                return patched_output
            
            # Apply the hook
            hook = None
            try:
                # Get the module for this layer
                module = model
                for name in layer_pattern.split('.'):
                    if not hasattr(module, name):
                        logging.warning(f"Module {name} not found in {layer_pattern}")
                        break
                    module = getattr(module, name)
                
                # Register the hook
                hook = module.register_forward_hook(
                    lambda mod, inp, out, act=clean_activations[layer_pattern], position=pos: 
                    patch_activation_hook(mod, inp, out, act, position)
                )
                
                # Run the model with the patching hook
                patched_outputs = model(**corrupted_inputs)
                patched_logits = patched_outputs.logits.detach()
                
                # Get output for the final position
                patched_final = patched_logits[0, corrupted_output_pos]
                
                # Calculate similarity to clean and corrupted baselines
                clean_sim = torch.nn.functional.cosine_similarity(
                    clean_output.unsqueeze(0), patched_final.unsqueeze(0))[0].item()
                corrupted_sim = torch.nn.functional.cosine_similarity(
                    corrupted_output.unsqueeze(0), patched_final.unsqueeze(0))[0].item()
                
                # Calculate normalized effect (0 = unchanged, 1 = fully patched to clean)
                effect = (clean_sim - corrupted_sim) / (2 - corrupted_sim - clean_sim + 1e-10)
                
                layer_effects.append({
                    "position": pos,
                    "token": corrupted_tokens[pos],
                    "clean_similarity": clean_sim,
                    "corrupted_similarity": corrupted_sim,
                    "effect": effect
                })
                
            except Exception as e:
                logging.error(f"Error patching at layer {layer}, position {pos}: {e}")
            
            finally:
                # Remove the hook
                if hook:
                    hook.remove()
        
        # Sort positions by effect magnitude
        layer_effects.sort(key=lambda x: abs(x["effect"]), reverse=True)
        results["patching_results"][layer] = layer_effects
        
        # Calculate average effect per layer
        if layer_effects:
            results["layer_effects"][layer] = sum(item["effect"] for item in layer_effects) / len(layer_effects)
        else:
            results["layer_effects"][layer] = 0.0
    
    # Clean up to save memory
    del clean_activations, clean_logits, corrupted_logits
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def pairwise_patching(
    model,
    tokenizer,
    prompt_a: str,
    prompt_b: str,
    target_layers: Optional[List[int]] = None,
    key_positions: Optional[List[int]] = None,
    patchtype: str = "resid"
) -> Dict:
    """
    Performs pairwise patching between two prompts to identify key differences.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        prompt_a: First input prompt
        prompt_b: Second input prompt
        target_layers: List of layers to patch. If None, uses all layers
        key_positions: List of specific positions to patch. If None, uses all positions
        patchtype: Type of activation to patch ("resid", "mlp", "attn")
        
    Returns:
        Dict containing pairwise patching results
    """
    # This is a wrapper around perform_patching_control
    result_a_to_b = perform_patching_control(
        model=model,
        tokenizer=tokenizer,
        clean_prompt=prompt_a,
        corrupted_prompt=prompt_b,
        target_layers=target_layers,
        target_positions=key_positions,
        patchtype=patchtype
    )
    
    result_b_to_a = perform_patching_control(
        model=model,
        tokenizer=tokenizer,
        clean_prompt=prompt_b,
        corrupted_prompt=prompt_a,
        target_layers=target_layers,
        target_positions=key_positions,
        patchtype=patchtype
    )
    
    # Compute bidirectional effects
    bidirectional_results = {
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "a_to_b": result_a_to_b,
        "b_to_a": result_b_to_a,
        "bidirectional_effects": {}
    }
    
    # Compute bidirectional layer effects (average of both directions)
    if target_layers is None:
        # Extract the layers from the results
        layers_a = set(result_a_to_b["layer_effects"].keys())
        layers_b = set(result_b_to_a["layer_effects"].keys())
        target_layers = sorted(layers_a.union(layers_b))
    
    for layer in target_layers:
        if (layer in result_a_to_b["layer_effects"] and
            layer in result_b_to_a["layer_effects"]):
            
            effect_a_to_b = result_a_to_b["layer_effects"][layer]
            effect_b_to_a = result_b_to_a["layer_effects"][layer]
            
            # Compute bidirectional effect
            bidirectional_effect = (abs(effect_a_to_b) + abs(effect_b_to_a)) / 2
            
            bidirectional_results["bidirectional_effects"][layer] = {
                "a_to_b": effect_a_to_b,
                "b_to_a": effect_b_to_a,
                "bidirectional": bidirectional_effect
            }
    
    return bidirectional_results
