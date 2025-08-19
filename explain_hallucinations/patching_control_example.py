"""
Example demonstrating patching control techniques.

This example shows how to perform controlled patching experiments between prompts.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from ltr.patching_control import perform_patching_control, pairwise_patching
import matplotlib.pyplot as plt
import numpy as np

def patching_control_example():
    """
    Example showing how to use patching control to analyze model interpretability.
    """
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "gpt2"  # Or any HuggingFace model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Define clean and corrupted prompts
    clean_prompt = "The Eiffel Tower is located in Paris, France."
    corrupted_prompt = "The Eiffel Tower is located in Rome, Italy."
    
    # Perform patching control analysis
    print(f"Performing patching control analysis between:")
    print(f"  Clean: {clean_prompt}")
    print(f"  Corrupted: {corrupted_prompt}")
    
    patch_results = perform_patching_control(
        model=model,
        tokenizer=tokenizer,
        clean_prompt=clean_prompt,
        corrupted_prompt=corrupted_prompt,
        target_layers=[0, 2, 4, 6, 8, 10],  # Analyze specific layers
        patchtype="resid"  # Patch the residual stream
    )
    
    # Print results
    print("\nPatching control results:")
    print(f"Clean top tokens: {patch_results['clean_top_tokens']}")
    print(f"Corrupted top tokens: {patch_results['corrupted_top_tokens']}")
    
    # Print layer effects
    print("\nLayer effects:")
    for layer, effect in sorted(patch_results['layer_effects'].items()):
        print(f"  Layer {layer}: {effect:.4f}")
    
    # Find most influential positions for a specific layer
    target_layer = max(patch_results['layer_effects'].items(), key=lambda x: abs(x[1]))[0]
    print(f"\nMost influential positions in layer {target_layer}:")
    
    if target_layer in patch_results['patching_results']:
        for pos_result in patch_results['patching_results'][target_layer][:3]:  # Top 3 positions
            print(f"  Position {pos_result['position']} (token: '{pos_result['token']}')")
            print(f"    Clean similarity: {pos_result['clean_similarity']:.4f}")
            print(f"    Corrupted similarity: {pos_result['corrupted_similarity']:.4f}")
            print(f"    Effect: {pos_result['effect']:.4f}")
    
    # Plot layer effects
    plt.figure(figsize=(10, 6))
    
    layers = sorted(patch_results['layer_effects'].keys())
    effects = [patch_results['layer_effects'][layer] for layer in layers]
    
    plt.bar(layers, effects)
    plt.title("Layer Effects (Patching Clean → Corrupted)")
    plt.xlabel("Layer")
    plt.ylabel("Effect Magnitude")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("patching_layer_effects.png")
    print("\nSaved patching layer effects to 'patching_layer_effects.png'")
    
    # Pairwise patching example
    print("\nPerforming pairwise patching analysis...")
    alternative_prompt = "The Eiffel Tower is located in Lyon, France."
    
    pairwise_results = pairwise_patching(
        model=model,
        tokenizer=tokenizer,
        prompt_a=clean_prompt,
        prompt_b=alternative_prompt,
        target_layers=[0, 2, 4, 6, 8, 10]
    )
    
    # Print bidirectional effects
    print("\nBidirectional effects:")
    for layer, effect_data in sorted(pairwise_results['bidirectional_effects'].items()):
        print(f"  Layer {layer}:")
        print(f"    A → B: {effect_data['a_to_b']:.4f}")
        print(f"    B → A: {effect_data['b_to_a']:.4f}")
        print(f"    Bidirectional: {effect_data['bidirectional']:.4f}")

if __name__ == "__main__":
    patching_control_example()
