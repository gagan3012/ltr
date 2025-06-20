import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from baukit import TraceDict
import logging


def logit_lens_analysis(
    model, 
    tokenizer, 
    prompt: str,
    target_layers: Optional[List[int]] = None,
    target_positions: Optional[List[int]] = None,
    top_k: int = 5
) -> Dict:
    """
    Performs logit lens analysis on model activations.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        prompt: Input text to analyze
        target_layers: List of layers to analyze. If None, uses all layers.
        target_positions: List of token positions to analyze. If None, uses all positions.
        top_k: Number of top tokens to return at each layer
        
    Returns:
        Dict containing logit lens analysis results
    """
    # Get model architecture information
    model_type = model.config.model_type.lower() if hasattr(model.config, "model_type") else ""
    
    # Configure residual stream patterns based on model type
    if "llama" in model_type or "mistral" in model_type or "qwen" in model_type:
        resid_pattern = "model.layers.{}.input_layernorm"
        n_layers = model.config.num_hidden_layers
        embedding_size = model.config.hidden_size
    elif "gpt-neox" in model_type or "gpt_neox" in model_type:
        resid_pattern = "gpt_neox.layers.{}.input_layernorm"
        n_layers = model.config.num_hidden_layers
        embedding_size = model.config.hidden_size
    elif "gpt2" in model_type:
        resid_pattern = "transformer.h.{}.ln_1"
        n_layers = model.config.n_layer
        embedding_size = model.config.n_embd
    elif "olmo" in model_type: 
        resid_pattern = "model.layers.{}.post_attention_layernorm"
        n_layers = model.config.num_hidden_layers
        embedding_size = model.config.hidden_size
    else:
        # Default pattern as fallback
        resid_pattern = "model.layers.{}.input_layernorm"
        n_layers = getattr(model.config, "num_hidden_layers", 12)
        embedding_size = getattr(model.config, "hidden_size", 768)
    
    # Use all layers if not specified
    if target_layers is None:
        target_layers = list(range(n_layers))
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    token_ids = inputs.input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    n_tokens = len(tokens)
    
    # Use all positions if not specified
    if target_positions is None:
        target_positions = list(range(n_tokens - 1))  # Exclude last token for next-token prediction
    
    # Create residual stream patterns for tracing
    resid_patterns = [resid_pattern.format(layer) for layer in target_layers]
    
    # Get the lm_head or output embedding matrix
    if hasattr(model, "lm_head"):
        unembed = model.lm_head
    elif hasattr(model, "get_output_embeddings"):
        unembed = model.get_output_embeddings()
    else:
        logging.warning("Could not find output embedding matrix, using approximation")
        # Approximate using the word embedding matrix
        if hasattr(model, "get_input_embeddings"):
            unembed = model.get_input_embeddings()
        else:
            logging.error("Could not find embedding matrix for logit lens")
            return {"error": "Could not find embedding matrix for logit lens"}
    
    results = {
        "prompt": prompt,
        "tokens": tokens,
        "layer_results": {}
    }
    
    # Trace through the model to get activations at each layer
    with TraceDict(model, resid_patterns) as traces:
        _ = model(**inputs)
        
        # For each layer, project residual stream to vocabulary space
        for layer_idx, layer in enumerate(target_layers):
            layer_pattern = resid_patterns[layer_idx]
            
            if layer_pattern not in traces:
                logging.warning(f"Layer pattern {layer_pattern} not found in traces")
                continue
            
            layer_results = []
            layer_activations = traces[layer_pattern].output
            
            # For each position, project to vocabulary space
            for pos in target_positions:
                if pos >= n_tokens:
                    continue
                
                try:
                    # Get residual activations at this layer and position
                    resid = layer_activations[0, pos].detach()
                    
                    # Project to vocabulary space using the unembedding matrix
                    if isinstance(unembed, torch.nn.Linear):
                        # If it's a linear layer, use its forward method
                        logits = unembed(resid.unsqueeze(0)).squeeze(0)
                    else:
                        # Otherwise, try matrix multiplication with the weight
                        logits = resid @ unembed.weight.t()
                    
                    # Convert to float32 for softmax stability
                    logits = logits.to(torch.float32)
                    
                    # Get top tokens
                    top_tokens = torch.topk(logits, k=top_k)
                    
                    # Convert token ids to strings
                    top_token_strs = [(tokenizer.decode([token.item()]).strip(), prob.item()) 
                                    for token, prob in zip(top_tokens.indices, torch.softmax(top_tokens.values, dim=-1))]
                    
                    layer_results.append({
                        "position": pos,
                        "token": tokens[pos],
                        "top_tokens": top_token_strs
                    })
                    
                except Exception as e:
                    logging.error(f"Error in logit lens at layer {layer}, position {pos}: {e}")
            
            results["layer_results"][layer] = layer_results
    
    return results


def trace_token_evolution(
    model, 
    tokenizer, 
    prompt: str,
    target_tokens: List[str],
    start_layer: int = 0
) -> Dict:
    """
    Traces the evolution of specific tokens through model layers.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        prompt: Input text to analyze
        target_tokens: List of tokens to trace through layers
        start_layer: Layer to start tracing from
        
    Returns:
        Dict containing token evolution traces
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
    elif "olmo" in model_type:
        resid_pattern = "model.layers.{}.post_attention_layernorm"
        n_layers = model.config.num_hidden_layers
        embedding_size = model.config.hidden_size
    else:
        # Default pattern as fallback
        resid_pattern = "model.layers.{}.input_layernorm"
        n_layers = getattr(model.config, "num_hidden_layers", 12)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    token_ids = inputs.input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    n_tokens = len(tokens)
    
    # Create residual stream patterns for tracing
    resid_patterns = [resid_pattern.format(layer) for layer in range(start_layer, n_layers)]
    
    # Get the lm_head or output embedding matrix
    if hasattr(model, "lm_head"):
        unembed = model.lm_head
    elif hasattr(model, "get_output_embeddings"):
        unembed = model.get_output_embeddings()
    else:
        logging.warning("Could not find output embedding matrix, using approximation")
        # Approximate using the word embedding matrix
        if hasattr(model, "get_input_embeddings"):
            unembed = model.get_input_embeddings()
        else:
            logging.error("Could not find embedding matrix for token evolution tracing")
            return {"error": "Could not find embedding matrix for token evolution tracing"}
    
    # Get token IDs for target tokens
    target_token_ids = []
    for token in target_tokens:
        token_input = tokenizer(token, return_tensors="pt", add_special_tokens=False).input_ids
        if token_input.numel() > 0:
            target_token_ids.append(token_input[0, 0].item())
        else:
            logging.warning(f"Could not find token ID for {token}")
            target_token_ids.append(-1)
    
    results = {
        "prompt": prompt,
        "tokens": tokens,
        "target_tokens": target_tokens,
        "token_evolution": {token: {} for token in target_tokens}
    }
    
    # Trace through the model to get activations at each layer
    with TraceDict(model, resid_patterns) as traces:
        _ = model(**inputs)
        
        # For each position in the sequence
        for pos in range(n_tokens - 1):  # Exclude last token
            # For each layer
            for layer_idx, layer in enumerate(range(start_layer, n_layers)):
                layer_pattern = resid_patterns[layer_idx]
                
                if layer_pattern not in traces:
                    logging.warning(f"Layer pattern {layer_pattern} not found in traces")
                    continue
                
                try:
                    # Get residual activations at this layer and position
                    resid = traces[layer_pattern].output[0, pos].detach()
                    
                    # Project to vocabulary space using the unembedding matrix
                    if isinstance(unembed, torch.nn.Linear):
                        # If it's a linear layer, use its forward method
                        logits = unembed(resid.unsqueeze(0)).squeeze(0)
                    else:
                        # Otherwise, try matrix multiplication with the weight
                        logits = resid @ unembed.weight.t()
                    
                    # Convert to float32 for softmax stability
                    logits = logits.to(torch.float32)
                    
                    # Get probabilities
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Record probabilities for target tokens
                    for token, token_id in zip(target_tokens, target_token_ids):
                        if token_id != -1:
                            if pos not in results["token_evolution"][token]:
                                results["token_evolution"][token][pos] = {}
                            
                            results["token_evolution"][token][pos][layer] = probs[token_id].item()
                    
                except Exception as e:
                    logging.error(f"Error in token evolution at layer {layer}, position {pos}: {e}")
    
    return results
