import numpy as np
import torch
import gc
from typing import Dict, List, Optional, Tuple, Union
from baukit import TraceDict
import logging


def perform_backpatching(
    model, 
    tokenizer, 
    prompt_a: str,
    prompt_b: str,
    target_positions: Optional[List[int]] = None,
    target_layers: Optional[List[int]] = None,
    trace_concepts: Optional[List[str]] = None
) -> Dict:
    """
    Performs backpatching intervention between two prompts.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        prompt_a: First input prompt
        prompt_b: Second input prompt
        target_positions: List of token positions to analyze. If None, uses all positions
        target_layers: List of layers to analyze. If None, uses all layers
        trace_concepts: List of concepts to trace during backpatching
        
    Returns:
        Dict containing backpatching intervention results
    """
    # Get model architecture information
    model_type = model.config.model_type.lower() if hasattr(model.config, "model_type") else ""
    
    # Configure residual stream patterns based on model type
    if "llama" in model_type or "mistral" in model_type or "qwen" in model_type:
        # For decoder-only models like LLaMA, Mistral, Qwen
        resid_pattern = "model.layers.{}.input_layernorm"
        n_layers = model.config.num_hidden_layers
    elif "gpt-neox" in model_type or "gpt_neox" in model_type:
        resid_pattern = "gpt_neox.layers.{}.input_layernorm"
        n_layers = model.config.num_hidden_layers
    elif "gpt2" in model_type:
        resid_pattern = "transformer.h.{}.ln_1"
        n_layers = model.config.n_layer
    else:
        # Default pattern as fallback
        resid_pattern = "model.layers.{}.input_layernorm"
        n_layers = getattr(model.config, "num_hidden_layers", 12)
    
    # Use all layers if not specified
    if target_layers is None:
        target_layers = list(range(n_layers))
    
    # Tokenize both prompts
    inputs_a = tokenizer(prompt_a, return_tensors="pt").to(model.device)
    inputs_b = tokenizer(prompt_b, return_tensors="pt").to(model.device)
    
    tokens_a = tokenizer.convert_ids_to_tokens(inputs_a.input_ids[0])
    tokens_b = tokenizer.convert_ids_to_tokens(inputs_b.input_ids[0])
    
    n_tokens_a = len(tokens_a)
    n_tokens_b = len(tokens_b)
    
    # Use all positions if not specified, up to the minimum length of both prompts
    if target_positions is None:
        max_pos = min(n_tokens_a, n_tokens_b) - 1
        target_positions = list(range(max_pos))
    
    # Create residual stream patterns for tracing
    resid_patterns = [resid_pattern.format(layer) for layer in target_layers]
    
    # Trace through both prompts to get activations
    with TraceDict(model, resid_patterns) as traces_a:
        outputs_a = model(**inputs_a)
        logits_a = outputs_a.logits.detach().cpu()
        
        # Store activations from prompt A
        cached_activations_a = {pattern: traces_a[pattern].detach().clone() for pattern in resid_patterns}
    
    # Calculate concept activations if specified
    concept_results = {}
    if trace_concepts and len(trace_concepts) > 0:
        from ltr.concept_extraction import extract_concept_activations
        concept_results["a"] = extract_concept_activations(model, tokenizer, prompt_a, intermediate_concepts=trace_concepts, final_concepts=[])
        concept_results["b"] = extract_concept_activations(model, tokenizer, prompt_b, intermediate_concepts=trace_concepts, final_concepts=[])
    
    results = {
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "tokens_a": tokens_a,
        "tokens_b": tokens_b,
        "intervention_results": [],
        "concept_results": concept_results
    }
    
    # For each layer, patch activations from prompt_a to prompt_b
    for layer_idx, layer in enumerate(target_layers):
        layer_results = {
            "layer": layer,
            "position_effects": []
        }
        
        layer_pattern = resid_patterns[layer_idx]
        
        # For each position, patch and measure effect
        for pos in target_positions:
            # Skip invalid positions
            if pos >= min(n_tokens_a, n_tokens_b):
                continue
            
            # Define a hook to patch this position
            def patch_position_hook(module, input_tensors, output, cached_activation, position):
                # Create a patched version of the output
                patched_output = output.clone()
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
                    lambda mod, inp, out, act=cached_activations_a[layer_pattern], position=pos: 
                    patch_position_hook(mod, inp, out, act, position)
                )
                
                # Run the model with the patching hook
                outputs_patched = model(**inputs_b)
                patched_logits = outputs_patched.logits.detach().cpu()
                
                # Calculate effect of patching
                positions_affected = []
                
                # Get the next token predictions for each position
                for effect_pos in range(min(n_tokens_a, n_tokens_b) - 1):
                    # Skip if this is too close to the end of either sequence
                    if effect_pos + 1 >= n_tokens_a or effect_pos + 1 >= n_tokens_b:
                        continue
                    
                    # Get the logits for the next token in each sequence
                    with torch.no_grad():
                        # Get original outputs for prompt B if we haven't already
                        if 'outputs_b' not in locals():
                            outputs_b = model(**inputs_b)
                            logits_b = outputs_b.logits.detach().cpu()
                        
                        # Get the next token in each sequence
                        next_token_a = inputs_a.input_ids[0, effect_pos + 1].item()
                        next_token_b = inputs_b.input_ids[0, effect_pos + 1].item()
                        
                        # Calculate probabilities
                        orig_probs_a = torch.softmax(logits_a[0, effect_pos], dim=-1)
                        orig_probs_b = torch.softmax(logits_b[0, effect_pos], dim=-1)
                        patched_probs = torch.softmax(patched_logits[0, effect_pos], dim=-1)
                        
                        # Calculate log probabilities
                        orig_logprob_a = torch.log(orig_probs_a[next_token_a] + 1e-10).item()
                        orig_logprob_b = torch.log(orig_probs_b[next_token_b] + 1e-10).item()
                        patched_logprob_b = torch.log(patched_probs[next_token_b] + 1e-10).item()
                        
                        # Calculate effect of patching
                        effect = patched_logprob_b - orig_logprob_b
                        
                        # Store effect if significant
                        if abs(effect) > 0.01:
                            positions_affected.append({
                                "position": effect_pos,
                                "token": tokens_b[effect_pos],
                                "next_token": tokens_b[effect_pos + 1],
                                "orig_logprob": orig_logprob_b,
                                "patched_logprob": patched_logprob_b,
                                "effect": effect
                            })
                
                # Add to results if there were any effects
                if positions_affected:
                    layer_results["position_effects"].append({
                        "patched_position": pos,
                        "patched_token": tokens_b[pos],
                        "affected_positions": positions_affected
                    })
                
            except Exception as e:
                logging.error(f"Error during patching at layer {layer}, position {pos}: {e}")
            
            finally:
                # Remove the hook
                if hook:
                    hook.remove()
        
        # Add layer results if there were any effects
        if layer_results["position_effects"]:
            results["intervention_results"].append(layer_results)
    
    # Clean up to save memory
    del cached_activations_a
    if 'logits_a' in locals(): del logits_a
    if 'logits_b' in locals(): del logits_b
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results
