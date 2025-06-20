"""
Example demonstrating entity analysis techniques.

This example shows how to analyze causal effects of entities and compare their representations.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from ltr.entity_analysis import analyze_causal_entities, extract_entity_representations, compare_entity_representations
import matplotlib.pyplot as plt
import numpy as np

def entity_analysis_example():
    """
    Example showing how to analyze entities in a model.
    """
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "gpt2"  # Or any HuggingFace model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Define prompt and target entities
    prompt = "Albert Einstein developed the theory of relativity. Marie Curie discovered radium and polonium."
    target_entities = ["Einstein", "Curie", "relativity", "radium"]
    
    # Perform causal entity analysis
    print(f"Performing causal entity analysis for prompt: {prompt}")
    entity_results = analyze_causal_entities(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_entities=target_entities,
        target_layers=[0, 5, 11]  # Analyze specific layers
    )
    
    # Print results
    print("\nCausal entity analysis results:")
    print(f"Prompt: {entity_results['prompt']}")
    print(f"Target entities: {entity_results['target_entities']}")
    
    # Print entity positions
    print("\nEntity positions in prompt:")
    for entity, positions in entity_results['entity_positions'].items():
        position_str = [f"{start}-{end}" for start, end in positions]
        print(f"  {entity}: {position_str}")
    
    # Print entity influences
    print("\nEntity influences:")
    for entity, influences in entity_results['entity_influences'].items():
        print(f"  {entity}:")
        
        # For each occurrence of the entity
        for i, influence_data in enumerate(influences):
            print(f"    Occurrence {i+1} (tokens: {influence_data['entity_span']}):")
            
            # Print top influenced layers
            top_layers = sorted(
                [(layer, data['avg_influence']) for layer, data in influence_data['layer_influence'].items()],
                key=lambda x: x[1],
                reverse=True
            )[:2]  # Top 2 layers
            
            for layer, avg_influence in top_layers:
                print(f"      Layer {layer}: avg_influence={avg_influence:.4f}")
                
                # Print top influenced positions
                if layer in influence_data['layer_influence']:
                    top_positions = influence_data['layer_influence'][layer]['position_influences'][:2]
                    for pos_data in top_positions:
                        print(f"        Position {pos_data['position']} ('{pos_data['token']}'): influence={pos_data['influence']:.4f}")
    
    # Entity representation analysis
    print("\nExtracting entity representations...")
    context_templates = [
        "{entity}",
        "{entity} is a",
        "The work of {entity}"
    ]
    
    entities = ["Einstein", "Curie", "Newton", "Darwin"]
    
    rep_results = extract_entity_representations(
        model=model,
        tokenizer=tokenizer,
        entities=entities,
        context_templates=context_templates,
        target_layer=5  # Use mid-layer representations
    )
    
    # Compare entity representations
    print("\nComparing entity representations...")
    comparison_results = compare_entity_representations(
        rep_results,
        method="cosine"
    )
    
    # Print similarity matrix for the first context
    context = context_templates[0]
    if context in comparison_results['similarity_matrices']:
        sim_matrix = comparison_results['similarity_matrices'][context]
        print(f"\nSimilarity matrix for context '{context}':")
        
        # Header row with entity names
        print(f"{'Entity':<10}", end='')
        for entity in entities:
            print(f"{entity:<10}", end='')
        print()
        
        # Matrix rows
        for i, entity_i in enumerate(entities):
            print(f"{entity_i:<10}", end='')
            for j, _ in enumerate(entities):
                print(f"{sim_matrix[i, j]:.2f}{'':<6}", end='')
            print()
    
    # Plot similarity heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix, cmap='viridis')
    plt.title(f"Entity Similarity Matrix (Context: '{context}')")
    plt.xticks(np.arange(len(entities)), entities)
    plt.yticks(np.arange(len(entities)), entities)
    plt.colorbar(label="Cosine Similarity")
    plt.tight_layout()
    plt.savefig("entity_similarity.png")
    print("\nSaved entity similarity heatmap to 'entity_similarity.png'")

if __name__ == "__main__":
    entity_analysis_example()
