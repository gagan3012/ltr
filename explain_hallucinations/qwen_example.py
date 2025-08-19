"""
Example usage with Qwen model to demonstrate compatibility with various HuggingFace models
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ltr.concept_extraction import extract_concept_activations
from ltr.reasoning_analysis import analyze_reasoning_paths
from ltr.visualization import plot_concept_activation_heatmap, animate_reasoning_flow
import matplotlib.pyplot as plt
from IPython.display import display, HTML

def analyze_with_qwen():
    """Analyze a geographical reasoning task using Qwen model"""
    # Load Qwen model and tokenizer
    model_name = "Qwen/Qwen-1.5-0.5B"  # A smaller Qwen model for quick loading
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define prompt and concepts for geographical reasoning
    prompt = "Fact: Dallas exists in the state whose capital is"
    intermediate_concepts = ["Dallas", "Texas"]
    final_concepts = ["Austin", "capital"]
    
    print(f"Running concept extraction on model: {model_name}")
    print(f"Model type: {model.config.model_type}")
    
    # Extract concept activations
    activations = extract_concept_activations(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        intermediate_concepts=intermediate_concepts,
        final_concepts=final_concepts
    )
    
    # Visualize concept activations
    print("Generating concept activation heatmap...")
    heatmap = plot_concept_activation_heatmap(activations)
    display(heatmap)
    
    # Analyze reasoning paths
    potential_paths = [
        ["Dallas", "Texas", "Austin"],
        ["Dallas", "capital", "Austin"]
    ]
    
    print("Analyzing reasoning paths...")
    path_results = analyze_reasoning_paths(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        potential_paths=potential_paths
    )
    
    # Calculate number of layers (for visualization)
    n_layers = len(activations["layer_max_probs"][intermediate_concepts[0]])
    
    # Visualize reasoning flow
    print("Generating reasoning flow animation...")
    flow_anim = animate_reasoning_flow(
        path_results=path_results,
        tokens=activations["tokens"],
        model_layers=n_layers
    )
    
    # Display animation in notebook or save as GIF
    display(HTML(flow_anim.to_jshtml()))
    
    # Return results for further analysis
    return {
        "activations": activations,
        "path_results": path_results,
        "model_info": {
            "name": model_name,
            "type": model.config.model_type,
            "n_layers": n_layers
        }
    }

if __name__ == "__main__":
    results = analyze_with_qwen()
    print(f"Best reasoning path: {' → '.join(results['path_results']['best_path'])}")
    print(f"Path score: {results['path_results']['best_path_score']:.4f}")
    
    # Print model information
    print("\nModel information:")
    for key, value in results["model_info"].items():
        print(f"  {key}: {value}")
