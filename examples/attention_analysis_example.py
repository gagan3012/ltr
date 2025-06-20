"""
Example demonstrating the attention analysis module.

This example shows how to analyze attention patterns in a language model.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from ltr.attention_analysis import analyze_attention_patterns, ablate_attention_patterns
import matplotlib.pyplot as plt
import numpy as np

def attention_analysis_example():
    """
    Example showing how to analyze attention patterns in a model.
    """
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "gpt2"  # Or any HuggingFace model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Define prompt and concepts
    prompt = "The capital of France is Paris, which is known for the Eiffel Tower."
    concepts = ["France", "Paris", "Eiffel Tower"]
    
    # Analyze attention patterns
    print(f"Analyzing attention patterns for prompt: {prompt}")
    attention_results = analyze_attention_patterns(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        concepts=concepts
    )
    
    # Print results
    print("\nAttention analysis results:")
    print(f"Prompt: {attention_results['prompt']}")
    print(f"Tokens: {attention_results['tokens']}")
    
    # Print attention to concepts
    print("\nConcept attention:")
    for concept, attn_data in attention_results['concept_attention'].items():
        print(f"  {concept}:")
        # Get top 3 heads with highest attention to this concept
        top_heads = sorted(attn_data.items(), key=lambda x: x[1], reverse=True)[:3]
        for (layer, head), score in top_heads:
            print(f"    Layer {layer}, Head {head}: {score:.4f}")
    
    # Plot attention heatmap for a specific head
    if attention_results['attention_maps']:
        # Select a head with high attention to a concept
        layer, head = list(attention_results['attention_maps'].keys())[0]
        attn_map = attention_results['attention_maps'][(layer, head)]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(attn_map, cmap='viridis')
        plt.title(f"Attention Matrix for Layer {layer}, Head {head}")
        plt.xlabel("Token Position (Key)")
        plt.ylabel("Token Position (Query)")
        plt.colorbar(label="Attention Weight")
        
        # Add token labels
        tokens = attention_results['tokens']
        if len(tokens) <= 20:  # Only add labels if there aren't too many tokens
            plt.xticks(np.arange(len(tokens)), tokens, rotation=90)
            plt.yticks(np.arange(len(tokens)), tokens)
        
        plt.tight_layout()
        plt.savefig("attention_heatmap.png")
        print("\nSaved attention heatmap to 'attention_heatmap.png'")

    # Ablation example
    print("\nPerforming attention head ablation...")
    # Choose a few important heads to ablate
    target_heads = [(0, 0), (5, 7), (8, 4)]  # Example head selections
    
    ablation_results = ablate_attention_patterns(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_heads=target_heads,
        ablation_factor=0.0  # Complete ablation
    )
    
    print(f"Average KL divergence after ablation: {ablation_results['avg_kl_div']:.4f}")
    
    # Print top affected tokens
    print("\nTop affected tokens after ablation:")
    top_affected = sorted(ablation_results['token_changes'], 
                         key=lambda x: x['kl_div'], reverse=True)[:3]
    
    for item in top_affected:
        print(f"  Token: '{item['token']}', KL divergence: {item['kl_div']:.4f}")
        print(f"    Baseline top predictions: {item['baseline_top'][:2]}")
        print(f"    Ablated top predictions: {item['ablated_top'][:2]}")
        print()

if __name__ == "__main__":
    attention_analysis_example()
