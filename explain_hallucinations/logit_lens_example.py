"""
Example demonstrating the logit lens analysis module.

This example shows how to analyze intermediate representations using the logit lens technique.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from ltr.logit_lens import logit_lens_analysis, trace_token_evolution
import matplotlib.pyplot as plt
import numpy as np

def logit_lens_example():
    """
    Example showing how to use logit lens to analyze intermediate representations.
    """
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "gpt2"  # Or any HuggingFace model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Define prompt and target layers
    prompt = "The city of Austin is the capital of the state of Texas in the United States."
    target_layers = [0, 3, 6, 9, 11]  # Analyze specific layers
    
    # Perform logit lens analysis
    print(f"Performing logit lens analysis for prompt: {prompt}")
    lens_results = logit_lens_analysis(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_layers=target_layers,
        top_k=5  # Return top 5 tokens at each layer
    )
    
    # Print results
    print("\nLogit lens analysis results:")
    print(f"Prompt: {lens_results['prompt']}")
    print(f"Tokens: {lens_results['tokens']}")
    
    # Select an interesting position to analyze (e.g., before "Texas")
    target_position = None
    for i, token in enumerate(lens_results['tokens']):
        if "Texas" in token:
            target_position = i - 1  # Position before "Texas"
            break
    
    if target_position is None:
        target_position = 5  # Default position if "Texas" not found
    
    print(f"\nAnalyzing position {target_position} (token: '{lens_results['tokens'][target_position]}')")
    
    # Print top tokens at each layer for the target position
    print("\nTop tokens by layer:")
    for layer in target_layers:
        if layer in lens_results['layer_results']:
            for pos_result in lens_results['layer_results'][layer]:
                if pos_result['position'] == target_position:
                    print(f"  Layer {layer}:")
                    for token_text, prob in pos_result['top_tokens']:
                        print(f"    {token_text}: {prob:.4f}")
    
    # Trace specific token evolution
    print("\nTracing token evolution through layers...")
    target_tokens = ["Texas", "California", "New York"]
    
    evolution_results = trace_token_evolution(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_tokens=target_tokens
    )
    
    # Plot token probability evolution
    plt.figure(figsize=(10, 6))
    
    # For each target token, plot its probability evolution at a specific position
    target_pos = target_position
    for token in target_tokens:
        if token in evolution_results['token_evolution'] and target_pos in evolution_results['token_evolution'][token]:
            # Extract layer -> probability mapping
            layers = []
            probs = []
            for layer, prob in sorted(evolution_results['token_evolution'][token][target_pos].items()):
                layers.append(layer)
                probs.append(prob)
            
            if layers and probs:  # Only plot if we have data
                plt.plot(layers, probs, label=token, marker='o')
    
    plt.title(f"Token Probability Evolution at Position {target_pos}")
    plt.xlabel("Layer")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("token_evolution.png")
    print("\nSaved token evolution plot to 'token_evolution.png'")

if __name__ == "__main__":
    logit_lens_example()
