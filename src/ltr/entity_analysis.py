import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from baukit import TraceDict
import logging
from ltr.concept_extraction import extract_concept_activations


def analyze_causal_entities(
    model, 
    tokenizer, 
    prompt: str,
    target_entities: List[str],
    target_layers: Optional[List[int]] = None
) -> Dict:
    """
    Performs causal entity analysis to identify entity influences.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        prompt: Input prompt to analyze
        target_entities: List of entities to analyze
        target_layers: List of layers to analyze. If None, uses all layers
        
    Returns:
        Dict containing causal entity analysis results
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
    elif "olmo" in model_type:
        resid_pattern = "model.layers.{}.post_attention_layernorm"
        n_layers = model.config.num_hidden_layers
    else:
        # Default pattern as fallback
        resid_pattern = "model.layers.{}.input_layernorm"
        n_layers = getattr(model.config, "num_hidden_layers", 12)
    
    # Use all layers if not specified
    if target_layers is None:
        target_layers = list(range(n_layers))
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    n_tokens = len(tokens)
    
    # Extract concept activations for entities
    concept_results = extract_concept_activations(
        model=model, 
        tokenizer=tokenizer, 
        prompt=prompt, 
        intermediate_concepts=target_entities, 
        final_concepts=[]
    )
    
    # Track entity occurrences in the prompt
    entity_positions = {}
    for entity in target_entities:
        entity_positions[entity] = []
        entity_tokens = tokenizer.tokenize(entity)
        entity_len = len(entity_tokens)
        
        for i in range(len(tokens) - entity_len + 1):
            if tokens[i:i+entity_len] == entity_tokens:
                entity_positions[entity].append((i, i+entity_len))
    
    results = {
        "prompt": prompt,
        "tokens": tokens,
        "target_entities": target_entities,
        "entity_positions": entity_positions,
        "concept_activations": concept_results,
        "entity_influences": {}
    }
    
    # Create residual stream patterns for tracing
    resid_patterns = [resid_pattern.format(layer) for layer in target_layers]
    
    # Run baseline model to get original logits
    with torch.no_grad():
        baseline_outputs = model(**inputs)
    baseline_logits = baseline_outputs.logits.detach().cpu()
    
    # For each entity, measure influence by ablating its activations
    for entity in target_entities:
        if entity not in entity_positions or not entity_positions[entity]:
            continue
        
        entity_results = []
        
        for start_pos, end_pos in entity_positions[entity]:
            influence_data = {
                "start_pos": start_pos,
                "end_pos": end_pos,
                "entity_span": tokens[start_pos:end_pos],
                "layer_influence": {}
            }
            
            # For each target layer, measure influence by ablating entity information
            for layer_idx, layer in enumerate(target_layers):
                layer_pattern = resid_patterns[layer_idx]
                
                # Define ablation hook
                def ablate_entity_hook(module, input_tensors, output, start, end):
                    # Create a patched version of the output
                    patched_output = output.clone()
                    
                    # Get stats for entity positions
                    entity_mean = output[0, start:end].mean(dim=0)
                    entity_std = output[0, start:end].std(dim=0)
                    
                    # Replace with Gaussian noise with same mean and std
                    noise = torch.randn_like(output[0, start:end]) * entity_std + entity_mean
                    
                    # Apply the noise to the entity positions
                    patched_output[0, start:end] = noise
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
                        lambda mod, inp, out, s=start_pos, e=end_pos: ablate_entity_hook(mod, inp, out, s, e)
                    )
                    
                    # Run the model with the ablation hook
                    ablated_outputs = model(**inputs)
                    ablated_logits = ablated_outputs.logits.detach().cpu()
                    
                    # Calculate influence on each position
                    position_influences = []
                    
                    for pos in range(n_tokens - 1):
                        # Skip positions within or immediately after the entity span
                        if start_pos <= pos <= end_pos:
                            continue
                        
                        # Get next token
                        next_token_id = input_ids[pos + 1].item()
                        
                        # Calculate probabilities
                        baseline_probs = torch.softmax(baseline_logits[0, pos], dim=-1)
                        ablated_probs = torch.softmax(ablated_logits[0, pos], dim=-1)
                        
                        baseline_prob = baseline_probs[next_token_id].item()
                        ablated_prob = ablated_probs[next_token_id].item()
                        
                        # Calculate influence
                        influence = baseline_prob - ablated_prob
                        
                        # Only record significant influences
                        if abs(influence) > 0.01:
                            position_influences.append({
                                "position": pos,
                                "token": tokens[pos],
                                "next_token": tokens[pos + 1] if pos + 1 < n_tokens else None,
                                "baseline_prob": baseline_prob,
                                "ablated_prob": ablated_prob,
                                "influence": influence
                            })
                    
                    # Sort by influence magnitude
                    position_influences.sort(key=lambda x: abs(x["influence"]), reverse=True)
                    
                    # Calculate overall influence for this layer
                    avg_influence = sum(abs(item["influence"]) for item in position_influences) / len(position_influences) if position_influences else 0
                    
                    influence_data["layer_influence"][layer] = {
                        "position_influences": position_influences,
                        "avg_influence": avg_influence
                    }
                    
                except Exception as e:
                    logging.error(f"Error ablating entity at layer {layer}: {e}")
                
                finally:
                    # Remove the hook
                    if hook:
                        hook.remove()
            
            entity_results.append(influence_data)
        
        results["entity_influences"][entity] = entity_results
    
    return results


def extract_entity_representations(
    model,
    tokenizer,
    entities: List[str],
    context_templates: Optional[List[str]] = None,
    target_layer: Optional[int] = None
) -> Dict:
    """
    Extracts neural representations of entities in different contexts.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        entities: List of entities to extract representations for
        context_templates: List of context templates (e.g., "{entity} is a")
        target_layer: Specific layer to extract representations from (defaults to last layer)
        
    Returns:
        Dict containing entity representations
    """
    # Get model architecture information
    model_type = model.config.model_type.lower() if hasattr(model.config, "model_type") else ""
    
    # Configure residual stream patterns based on model type
    if "llama" in model_type or "mistral" in model_type or "qwen" in model_type:
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
    
    # Use last layer if not specified
    if target_layer is None:
        target_layer = n_layers - 1
    
    # Use a simple context if none provided
    if context_templates is None:
        context_templates = ["{entity}"]
    
    # Create the layer pattern to trace
    layer_pattern = resid_pattern.format(target_layer)
    
    results = {
        "entities": entities,
        "contexts": context_templates,
        "target_layer": target_layer,
        "representations": {}
    }
    
    # Process each entity
    for entity in entities:
        entity_results = {}
        
        # Process each context template
        for context_template in context_templates:
            # Create prompt with entity
            prompt = context_template.format(entity=entity)
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            
            # Find the entity position in the tokens
            entity_tokens = tokenizer.tokenize(entity)
            entity_len = len(entity_tokens)
            entity_pos = None
            
            for i in range(len(tokens) - entity_len + 1):
                if tokens[i:i+entity_len] == entity_tokens:
                    entity_pos = (i, i+entity_len)
                    break
            
            if entity_pos is None:
                logging.warning(f"Entity '{entity}' not found in tokenized prompt")
                continue
            
            # Run the model and trace activations
            with TraceDict(model, [layer_pattern]) as traces:
                _ = model(**inputs)

                if layer_pattern in traces:
                    # Extract the activations properly from the Trace object
                    trace_output = traces[layer_pattern].output

                    # Handle different output formats
                    if isinstance(trace_output, tuple):
                        # If the output is a tuple, take the first element (hidden states)
                        activations = trace_output[0]
                    else:
                        # If it's already a tensor
                        activations = trace_output

                    # Extract the activations for the entity
                    start_pos, end_pos = entity_pos
                    entity_activations = (
                        activations[0, start_pos:end_pos].detach().cpu()
                    )

                    # Average over the entity tokens
                    entity_representation = entity_activations.mean(dim=0).numpy()

                    # Store in results
                    entity_results[context_template] = {
                        "representation": entity_representation,
                        "tokens": tokens[start_pos:end_pos],
                    }
                else:
                    logging.warning(
                        f"Layer pattern '{layer_pattern}' not found in traces"
                    )

        results["representations"][entity] = entity_results

    return results


def compare_entity_representations(
    entity_representations: Dict,
    method: str = "cosine"
) -> Dict:
    """
    Compares entity representations to measure similarities.
    
    Args:
        entity_representations: Output from extract_entity_representations
        method: Similarity method ("cosine", "euclidean")
        
    Returns:
        Dict containing similarity scores between entities
    """
    entities = entity_representations["entities"]
    contexts = entity_representations["contexts"]
    representations = entity_representations["representations"]
    
    results = {
        "entities": entities,
        "contexts": contexts,
        "similarity_method": method,
        "similarity_matrices": {}
    }
    
    # For each context, compute a similarity matrix
    for context in contexts:
        # Create matrix of shape (n_entities, n_entities)
        n_entities = len(entities)
        similarity_matrix = np.zeros((n_entities, n_entities))
        
        # Compute similarities
        for i, entity_i in enumerate(entities):
            if entity_i not in representations or context not in representations[entity_i]:
                continue
                
            rep_i = representations[entity_i][context]["representation"]
            
            for j, entity_j in enumerate(entities):
                if entity_j not in representations or context not in representations[entity_j]:
                    continue
                    
                rep_j = representations[entity_j][context]["representation"]
                
                # Calculate similarity
                if method == "cosine":
                    # Cosine similarity
                    similarity = np.dot(rep_i, rep_j) / (np.linalg.norm(rep_i) * np.linalg.norm(rep_j))
                elif method == "euclidean":
                    # Euclidean distance (converted to similarity)
                    distance = np.linalg.norm(rep_i - rep_j)
                    similarity = 1.0 / (1.0 + distance)  # Convert to similarity
                else:
                    similarity = 0.0
                
                similarity_matrix[i, j] = similarity
        
        results["similarity_matrices"][context] = similarity_matrix
    
    return results
