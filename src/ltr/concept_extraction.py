import numpy as np
import torch
from typing import List, Dict
from baukit import TraceDict
import logging


def get_layer_pattern_and_count(model):
    """
    Determine the appropriate layer pattern and count for different model architectures.
    This function returns the residual stream pattern rather than layernorm.
    """
    model_type = (
        model.config.model_type.lower() if hasattr(model.config, "model_type") else ""
    )

    # Configure layer patterns for popular model types - focusing on residual stream
    if "llama" in model_type:
        # For LLaMA, we want the residual output after the entire block
        layer_pattern = "model.layers.{}"  # Main layer container
        n_layers = model.config.num_hidden_layers
    elif "qwen" in model_type:
        layer_pattern = "model.layers.{}"
        n_layers = model.config.num_hidden_layers
    elif "mistral" in model_type:
        layer_pattern = "model.layers.{}"
        n_layers = model.config.num_hidden_layers
    elif "gpt-neox" in model_type or "gpt_neox" in model_type:
        layer_pattern = "gpt_neox.layers.{}"
        n_layers = model.config.num_hidden_layers
    elif "gpt2" in model_type:
        layer_pattern = "transformer.h.{}"
        n_layers = model.config.n_layer
    elif "falcon" in model_type:
        layer_pattern = "transformer.h.{}"
        n_layers = model.config.num_hidden_layers
    elif "bloom" in model_type:
        layer_pattern = "transformer.h.{}"
        n_layers = model.config.n_layer
    elif "opt" in model_type:
        layer_pattern = "model.decoder.layers.{}"
        n_layers = model.config.num_hidden_layers
    else:
        # Default pattern for transformers
        layer_pattern = "model.layers.{}"
        n_layers = getattr(
            model.config,
            "n_layer",
            getattr(
                model.config, "num_hidden_layers", getattr(model.config, "n_layers", 12)
            ),
        )
        logging.warning(
            f"Unknown model type '{model_type}'. Using default layer pattern '{layer_pattern}' and {n_layers} layers."
        )

    return layer_pattern, n_layers


def extract_concept_activations(
    model,
    tokenizer,
    prompt: str,
    intermediate_concepts: List[str],
    final_concepts: List[str],
    logit_threshold: float = 0.001,
) -> Dict:
    """
    Extract evidence of concept activations across all layers and positions.

    Parameters:
    -----------
    model : PreTrainedModel
        The transformer model to analyze
    tokenizer : PreTrainedTokenizer
        The tokenizer corresponding to the model
    prompt : str
        The input text prompt
    intermediate_concepts : List[str]
        Concepts that may appear during reasoning
    final_concepts : List[str]
        Concepts that represent final answers
    logit_threshold : float
        Minimum activation threshold to consider

    Returns:
    --------
    Dict
        Detailed information about concept activations
    """
    # For backward compatibility with existing code that passes a HookedTransformer
    if hasattr(model, "to_str_tokens") and hasattr(model, "run_with_cache"):
        return _extract_with_transformerlens(
            model, prompt, intermediate_concepts, final_concepts, logit_threshold
        )

    # Determine layer pattern and count for the model
    layer_pattern, n_layers = get_layer_pattern_and_count(model)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    n_tokens = len(tokens)
    all_concepts = intermediate_concepts + final_concepts

    # Create result structure
    results = {
        "prompt": prompt,
        "tokens": tokens,
        "intermediate_concepts": intermediate_concepts,
        "final_concepts": final_concepts,
        "activations": {concept: [] for concept in all_concepts},
        "activation_grid": {
            concept: np.zeros((n_layers, n_tokens - 1)) for concept in all_concepts
        },
    }

    # Get token IDs for all concepts
    concept_token_ids = {}
    for concept in all_concepts:
        try:
            # For better compatibility with different tokenizers and to match TransformerLens
            # We'll try both with space prefix and without
            tokens_with_space = tokenizer.encode(
                " " + concept, add_special_tokens=False
            )
            tokens_without_space = tokenizer.encode(concept, add_special_tokens=False)

            # Choose the one that gives a single token, prefer with space like TransformerLens
            if len(tokens_with_space) == 1:
                concept_token_ids[concept] = tokens_with_space[0]
            elif len(tokens_without_space) == 1:
                concept_token_ids[concept] = tokens_without_space[0]
            else:
                logging.warning(
                    f"Concept '{concept}' maps to multiple tokens. Using first token."
                )
                concept_token_ids[concept] = tokens_with_space[0]
        except Exception as e:
            logging.warning(f"Failed to encode concept '{concept}': {e}")
            continue

    # Set up traces for residual outputs of each layer
    # This is more equivalent to TransformerLens hook_resid_post
    trace_layers = [layer_pattern.format(i) for i in range(n_layers)]

    # Run the model with tracing
    with torch.no_grad():
        with TraceDict(
            model, trace_layers, retain_input=False, retain_output=True
        ) as traces:
            outputs = model(**inputs, output_hidden_states=True)

            # Get the output projection matrix - equivalent to W_U in TransformerLens
            if hasattr(model, "lm_head"):
                output_weights = model.lm_head.weight
            elif hasattr(model, "cls"):
                output_weights = model.cls.predictions.decoder.weight
            else:
                # Last resort - try to get it from the output embeddings
                if hasattr(model, "get_output_embeddings"):
                    output_weights = model.get_output_embeddings().weight
                else:
                    raise ValueError(
                        "Could not locate output projection matrix. This model architecture may not be supported."
                    )

            # Process each layer's hidden states
            # Use model hidden states if available (more reliable than traced layers)
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states
                for layer in range(n_layers):
                    layer_output = hidden_states[
                        layer + 1
                    ]  # +1 to skip the embedding layer

                    # Start from position 1 to skip the first token (same as TransformerLens)
                    for pos in range(1, n_tokens):
                        # Get the residual vector for this position - matches TransformerLens
                        residual = layer_output[0, pos, :]

                        # Project to the vocabulary space - same as TransformerLens
                        projected_logits = residual @ output_weights.T

                        # Check activation for each concept
                        for concept, concept_id in concept_token_ids.items():
                            concept_score = projected_logits[concept_id].item()

                            # Store in activation grid
                            results["activation_grid"][concept][layer, pos - 1] = (
                                concept_score
                            )

                            # Store activations above threshold
                            if concept_score > logit_threshold:
                                results["activations"][concept].append(
                                    {
                                        "layer": layer,
                                        "position": pos - 1,
                                        "probability": concept_score,
                                        "context_token": tokens[pos],
                                    }
                                )
            else:
                # Fall back to traced layers if hidden_states not available
                for layer in range(n_layers):
                    layer_name = layer_pattern.format(layer)

                    # Skip if the layer wasn't traced
                    if layer_name not in traces:
                        logging.warning(
                            f"Layer '{layer_name}' wasn't traced. Skipping."
                        )
                        continue

                    # Get the layer output
                    layer_output = traces[layer_name].output

                    # Standardize shape handling to match TransformerLens
                    # We want shape to be [batch, sequence, hidden_dim]
                    if len(layer_output.shape) == 3:  # [batch, seq_len, hidden_dim]
                        # This is the expected shape
                        pass
                    elif len(layer_output.shape) == 2:  # [seq_len, hidden_dim]
                        # Add batch dimension
                        layer_output = layer_output.unsqueeze(0)
                    else:
                        logging.warning(
                            f"Unexpected layer output shape: {layer_output.shape}. Trying to adapt."
                        )
                        if (
                            hasattr(outputs, "hidden_states")
                            and outputs.hidden_states is not None
                        ):
                            # Use hidden states as fallback
                            layer_output = outputs.hidden_states[layer + 1]

                    # Start from position 1 to skip the first token (same as TransformerLens)
                    for pos in range(1, n_tokens):
                        try:
                            # Always access as [0, pos, :] to match TransformerLens
                            residual = layer_output[0, pos, :]

                            # Project to the vocabulary space - same as TransformerLens
                            projected_logits = residual @ output_weights.T

                            # Check activation for each concept
                            for concept, concept_id in concept_token_ids.items():
                                concept_score = projected_logits[concept_id].item()

                                # Store in activation grid
                                results["activation_grid"][concept][layer, pos - 1] = (
                                    concept_score
                                )

                                # Store activations above threshold
                                if concept_score > logit_threshold:
                                    results["activations"][concept].append(
                                        {
                                            "layer": layer,
                                            "position": pos - 1,
                                            "probability": concept_score,
                                            "context_token": tokens[pos],
                                        }
                                    )
                        except IndexError as e:
                            logging.warning(
                                f"IndexError at layer {layer}, position {pos}: {e}"
                            )
                            continue

    # Calculate maximum probabilities per layer
    results["layer_max_probs"] = {}
    for concept in all_concepts:
        if concept in concept_token_ids:
            layer_maxes = np.max(results["activation_grid"][concept], axis=1)
            results["layer_max_probs"][concept] = layer_maxes
        else:
            # Handle concepts that couldn't be encoded
            results["layer_max_probs"][concept] = np.zeros(n_layers)

    return results
