"""
Example demonstrating backpatching interventions.

This example shows how to perform backpatching interventions between prompts.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from ltr.backpatching import perform_backpatching
import matplotlib.pyplot as plt
import numpy as np

def backpatching_example():
    """
    Example showing how to use backpatching for model interpretability.
    """
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "gpt2"  # Or any HuggingFace model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Define two prompts that differ at a specific point
    prompt_a = "The best way to learn a programming language is to practice coding."
    prompt_b = "The best way to learn a programming language is to read documentation."
    
    # Define concepts to trace
    trace_concepts = ["practice", "coding", "read", "documentation"]
    
    # Perform backpatching analysis
    print(f"Performing backpatching between:")
    print(f"  Prompt A: {prompt_a}")
    print(f"  Prompt B: {prompt_b}")
    
    backpatch_results = perform_backpatching(
        model=model,
        tokenizer=tokenizer,
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        target_layers=[0, 2, 4, 6, 8, 10],  # Analyze specific layers
        trace_concepts=trace_concepts
    )
    
    # Print results
    print("\nBackpatching results:")
    print(f"Prompt A: {backpatch_results['prompt_a']}")
    print(f"Prompt B: {backpatch_results['prompt_b']}")
    
    # Print tokens
    print("\nTokens A:", backpatch_results['tokens_a'])
    print("Tokens B:", backpatch_results['tokens_b'])
    
    # Print concept activations if available
    if backpatch_results['concept_results']:
        print("\nConcept activations in Prompt A:")
        for concept, activations in backpatch_results['concept_results']['a']['activations'].items():
            if 'presence' in activations:
                presence = [round(p, 2) for p in activations['presence']]
                print(f"  {concept}: {presence}")
        
        print("\nConcept activations in Prompt B:")
        for concept, activations in backpatch_results['concept_results']['b']['activations'].items():
            if 'presence' in activations:
                presence = [round(p, 2) for p in activations['presence']]
                print(f"  {concept}: {presence}")
    
    # Print intervention results
    print("\nIntervention results by layer:")
    
    # Create lists to store data for plotting
    all_layers = []
    avg_effects = []
    
    for layer_result in backpatch_results['intervention_results']:
        layer = layer_result['layer']
        all_layers.append(layer)
        
        # Calculate average effect magnitude for this layer
        effects = []
        for pos_effect in layer_result['position_effects']:
            for affected in pos_effect['affected_positions']:
                effects.append(abs(affected['effect']))
        
        avg_effect = sum(effects) / len(effects) if effects else 0
        avg_effects.append(avg_effect)
        
        print(f"  Layer {layer}:")
        print(f"    Positions patched: {len(layer_result['position_effects'])}")
        print(f"    Average effect: {avg_effect:.4f}")
        
        # Print top position effect
        if layer_result['position_effects']:
            top_pos = layer_result['position_effects'][0]
            print(f"    Top patched position: {top_pos['patched_position']} (token: '{top_pos['patched_token']}')")
            
            # Print top affected positions
            if top_pos['affected_positions']:
                print(f"    Top affected positions:")
                for i, affected in enumerate(sorted(top_pos['affected_positions'], key=lambda x: abs(x['effect']), reverse=True)[:2]):
                    print(f"      Position {affected['position']} (token: '{affected['token']}')")
                    print(f"        Original prob: {affected['orig_logprob']:.4f}")
                    print(f"        Patched prob: {affected['patched_logprob']:.4f}")
                    print(f"        Effect: {affected['effect']:.4f}")
    
    # Plot average effect by layer
    plt.figure(figsize=(10, 6))
    plt.bar(all_layers, avg_effects)
    plt.title("Average Effect Magnitude by Layer")
    plt.xlabel("Layer")
    plt.ylabel("Average Effect Magnitude")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("backpatching_effects.png")
    print("\nSaved backpatching effects to 'backpatching_effects.png'")
    
    # Find most influential patch
    max_effect = 0
    max_layer = None
    max_pos = None
    max_affected_pos = None
    
    for layer_result in backpatch_results['intervention_results']:
        for pos_effect in layer_result['position_effects']:
            for affected in pos_effect['affected_positions']:
                if abs(affected['effect']) > max_effect:
                    max_effect = abs(affected['effect'])
                    max_layer = layer_result['layer']
                    max_pos = pos_effect['patched_position']
                    max_affected_pos = affected['position']
    
    if max_layer is not None:
        print(f"\nMost influential patch:")
        print(f"  Layer: {max_layer}")
        print(f"  Patched position: {max_pos} (token: '{backpatch_results['tokens_b'][max_pos]}')")
        print(f"  Affected position: {max_affected_pos} (token: '{backpatch_results['tokens_b'][max_affected_pos]}')")
        print(f"  Effect magnitude: {max_effect:.4f}")

if __name__ == "__main__":
    backpatching_example()
