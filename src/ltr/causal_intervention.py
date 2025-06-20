import numpy as np
import torch
import gc
from typing import List, Dict, Optional
from baukit import TraceDict
from ltr.concept_extraction import get_layer_pattern_and_count

def perform_causal_intervention(model, tokenizer, prompt: str,
                                concepts: List[str],
                                target_positions: Optional[List[int]] = None,
                                patch_positions: Optional[List[int]] = None) -> Dict:
    """
    Perform causal interventions to analyze concept dependencies.
    
    Parameters:
    -----------
    model : PreTrainedModel
        The transformer model to analyze
    tokenizer : PreTrainedTokenizer
        The tokenizer corresponding to the model
    prompt : str
        The input text prompt
    concepts : List[str]
        Concepts to trace
    target_positions : Optional[List[int]]
        Token positions to target for intervention
    patch_positions : Optional[List[int]]
        Token positions to patch during intervention
        
    Returns:
    --------
    Dict
        Intervention results including token importance scores
    """
    # For backward compatibility with existing code that passes a HookedTransformer
    if hasattr(model, 'to_str_tokens') and hasattr(model, 'run_with_cache'):
        return _causal_intervention_with_transformerlens(model, prompt, concepts, target_positions, patch_positions)
    
    # Determine layer pattern and count for the model
    layer_pattern, n_layers = get_layer_pattern_and_count(model)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    n_tokens = len(tokens)
    
    if target_positions is None:
        target_positions = list(range(1, n_tokens - 1))  # Skip first token

    if patch_positions is None:
        patch_positions = list(range(n_tokens))

    # Prepare results structure
    results = {
        "prompt": prompt,
        "tokens": tokens,
        "concepts": concepts,
        "intervention_grids": {c: {} for c in concepts},
        "token_importance": {c: [] for c in concepts}
    }
    
    # Get token IDs for concepts
    concept_token_ids = {}
    for concept in concepts:
        try:
            concept_tokens = tokenizer.encode(" " + concept, add_special_tokens=False)
            if len(concept_tokens) == 1:  # Only use single token concepts
                concept_token_ids[concept] = concept_tokens[0]
            else:
                print(f"Concept '{concept}' maps to multiple tokens. Using first token.")
                concept_token_ids[concept] = concept_tokens[0]
        except Exception as e:
            print(f"Failed to encode concept '{concept}': {e}")
            concept_token_ids[concept] = -1
    
    # Set up traces for last layer
    final_layer = layer_pattern.format(n_layers - 1)
    
    # Get clean model predictions
    with torch.no_grad():
        with TraceDict(model, [final_layer]) as traces:
            outputs = model(**inputs)
            
        # Get the output projection matrix
        if hasattr(model, "lm_head"):
            output_weights = model.lm_head.weight
        elif hasattr(model, "cls"):
            output_weights = model.cls.predictions.decoder.weight
        else:
            raise ValueError("Could not locate output projection matrix.")
            
        # Get clean predictions for final token
        final_pos = n_tokens - 1
        final_layer_output = traces[final_layer].output[0]  # [batch, seq_len, hidden_dim]
        final_residual = final_layer_output[0, final_pos, :]
        clean_logits = final_residual @ output_weights.T
        
        # Calculate clean probabilities for each concept
        clean_probs = {}
        for concept, concept_id in concept_token_ids.items():
            if concept_id != -1:
                clean_probs[concept] = clean_logits[concept_id].item()
            else:
                clean_probs[concept] = 0.0
    
    # Define token replacements for interventions
    replacements = {
        " Dallas": " Chicago", 
        " plus": " minus", 
        " antagonist": " protagonist",
        " true": " false",
        " Texas": " Illinois",
        " right": " wrong"
    }
    
    del outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    # Intervene on each target position
    for pos in target_positions:
        if pos < 1 or pos >= n_tokens:  # Skip invalid positions
            continue
            
        token_to_replace = tokens[pos]
        
        # Skip common tokens that don't carry much semantic meaning
        if token_to_replace.strip().lower() in [".", ",", "?", "!", ":", ";", "the", "a", "an", "of", "to", "in", "is", "and"]:
            continue
            
        # Try to find a replacement
        replaced_token = None
        for orig, repl in replacements.items():
            if orig.lower() in token_to_replace.lower():
                replaced_token = token_to_replace.replace(orig, repl)
                break
                
        if replaced_token is None:
            continue
            
        # Create corrupted prompt
        corrupted_text = tokenizer.decode(inputs.input_ids[0][:pos]) + replaced_token + tokenizer.decode(inputs.input_ids[0][pos+1:])
        corrupted_inputs = tokenizer(corrupted_text, return_tensors="pt").to(model.device)
        
        # Run corrupted model
        with torch.no_grad():
            with TraceDict(model, [final_layer]) as traces:
                _ = model(**corrupted_inputs)
                
            corrupted_layer_output = traces[final_layer].output[0] 
            corrupted_final_residual = corrupted_layer_output[0, final_pos, :]
            corrupted_logits = corrupted_final_residual @ output_weights.T
            
            # Calculate corrupted probabilities
            for concept, concept_id in concept_token_ids.items():
                if concept_id == -1:
                    continue
                    
                corrupted_prob = corrupted_logits[concept_id].item()
                
                # Calculate impact of intervention
                impact = clean_probs[concept] - corrupted_prob
                
                # Store results
                results["token_importance"][concept].append({
                    "token": token_to_replace,
                    "position": pos,
                    "clean_prob": clean_probs[concept],
                    "corrupted_prob": corrupted_prob,
                    "impact": impact
                })
                
                # Sort results by impact magnitude
                results["token_importance"][concept].sort(key=lambda x: abs(x["impact"]), reverse=True)
                
    return results

def _causal_intervention_with_transformerlens(model, prompt, concepts, target_positions=None, patch_positions=None):
    """Legacy function to maintain backward compatibility with TransformerLens models"""
    tokens = model.to_str_tokens(prompt)
    n_tokens = len(tokens)
    n_layers = model.cfg.n_layers

    if target_positions is None:
        target_positions = list(range(n_tokens - 1))

    if patch_positions is None:
        patch_positions = list(range(n_tokens))

    results = {
        "prompt": prompt,
        "tokens": tokens,
        "concepts": concepts,
        "intervention_grids": {c: {} for c in concepts},
        "token_importance": {c: [] for c in concepts}
    }

    clean_logits, clean_cache = model.run_with_cache(prompt)

    concept_ids = []
    for concept in concepts:
        try:
            concept_ids.append(model.to_single_token(concept))
        except Exception:
            print(f"Warning: Could not convert '{concept}' to a single token")
            concept_ids.append(-1)

    final_pos = n_tokens - 1

    clean_probs = {}
    for concept, concept_id in zip(concepts, concept_ids):
        if concept_id != -1:
            clean_probs[concept] = clean_logits[0, final_pos, concept_id].item()
        else:
            clean_probs[concept] = 0.0

    replacements = {
        " Dallas": " Chicago", 
        " plus": " minus", 
        " antagonist": " protagonist",
        " true": " false",
        " Texas": " Illinois",
        " right": " wrong"
    }

    del clean_logits
    torch.cuda.empty_cache()
    gc.collect()

    for pos in target_positions:
        if pos < 1 or pos >= n_tokens:  # Skip invalid positions
            continue
            
        token_to_replace = tokens[pos]
        
        # Skip common tokens that don't carry much semantic meaning
        if token_to_replace.strip().lower() in [".", ",", "?", "!", ":", ";", "the", "a", "an", "of", "to", "in", "is", "and"]:
            continue

        # Try to find a replacement for the token
        replaced_token = None
        for orig, repl in replacements.items():
            if orig.lower() in token_to_replace.lower():
                replaced_token = token_to_replace.replace(orig, repl)
                break
                
        if replaced_token is None:
            continue
            
        # Create corrupted prompt by replacing the token
        corrupted_prompt = prompt[:model.to_offsets(prompt)[pos][0]] + replaced_token + prompt[model.to_offsets(prompt)[pos][1]:]
        
        # Run the model on the corrupted prompt
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_prompt)
        
        # Calculate corrupted probabilities for each concept
        for concept_idx, (concept, concept_id) in enumerate(zip(concepts, concept_ids)):
            if concept_id == -1:
                continue
                
            corrupted_prob = corrupted_logits[0, final_pos, concept_id].item()
            
            # Calculate impact of intervention
            impact = clean_probs[concept] - corrupted_prob
            
            # Store results
            results["token_importance"][concept].append({
                "token": token_to_replace,
                "position": pos,
                "clean_prob": clean_probs[concept],
                "corrupted_prob": corrupted_prob,
                "impact": impact
            })
            
            # Now create the intervention grid for each layer and position
            if pos not in results["intervention_grids"][concept]:
                results["intervention_grids"][concept][pos] = np.zeros((n_layers, n_tokens))
                
            # Perform causal interventions across layers and positions
            for layer in range(n_layers):
                for patch_pos in patch_positions:
                    if patch_pos >= n_tokens or patch_pos < 0:
                        continue
                        
                    # Skip certain combinations to save computation
                    if patch_pos != pos and layer < n_layers // 2:
                        continue
                        
                    # Create a copy of the corrupted cache for patching
                    patched_cache = dict(corrupted_cache)
                    
                    # Patch the clean activation at this layer and position
                    hook_name = f"blocks.{layer}.hook_resid_post"
                    if hook_name in clean_cache and hook_name in patched_cache:
                        if patched_cache[hook_name].shape == clean_cache[hook_name].shape:
                            patched_cache[hook_name][0, patch_pos] = clean_cache[hook_name][0, patch_pos]
                              # Run the model with the patched cache
                            with torch.no_grad():
                                try:
                                    # We need to get the tokenized version of the corrupted prompt
                                    corrupted_tokens = model.to_tokens(corrupted_prompt)
                                    patched_logits = model.forward(corrupted_tokens, return_type="logits", past_key_values=patched_cache)
                                    
                                    # Calculate the effect of patching
                                    if concept_id != -1:
                                        patched_prob = patched_logits[0, final_pos, concept_id].item()
                                        effect = patched_prob - corrupted_prob
                                        
                                        # Store the effect in the intervention grid
                                        results["intervention_grids"][concept][pos][layer, patch_pos] = effect
                                except Exception as e:
                                    # Skip in case of errors during patching
                                    print(f"Error patching at layer {layer}, position {patch_pos}: {e}")
                                    continue
            
        # Clean up to save memory
        del corrupted_logits, corrupted_cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Sort results by impact magnitude for better readability
        for concept in concepts:
            if results["token_importance"][concept]:
                results["token_importance"][concept].sort(key=lambda x: abs(x["impact"]), reverse=True)
        
    return results
