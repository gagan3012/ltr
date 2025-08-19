"""
Example usage of the LLM Reasoning Tracer with HuggingFace models
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ltr.concept_extraction import extract_concept_activations
from ltr.reasoning_analysis import analyze_reasoning_paths
from ltr.causal_intervention import perform_causal_intervention
from ltr.visualization import (
    plot_concept_activation_heatmap,
    animate_concept_activation_diagonal,
    animate_reasoning_flow,
    plot_layer_position_intervention,
    save_animation,
)
import matplotlib.pyplot as plt
from IPython.display import display

def analyze_math_problem():
    """Example analysis of a math problem using a HuggingFace model"""
    # Load model and tokenizer
    model_name = "gpt2"  # Replace with your preferred model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define prompt and concepts
    prompt = "To calculate 6 × 5, I multiply 6 by 5 to get 30."
    intermediate_concepts = ["multiply", "multiplying", "multiplication"]
    final_concepts = ["30", "thirty", "result"]
    
    # Extract concept activations
    activations = extract_concept_activations(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        intermediate_concepts=intermediate_concepts,
        final_concepts=final_concepts
    )
    
    # Visualize concept activations
    heatmap = plot_concept_activation_heatmap(activations)
    display(heatmap)
    
    # Analyze reasoning paths
    potential_paths = [
        ["multiply", "30"],
        ["multiplying", "thirty"]
    ]
    
    path_results = analyze_reasoning_paths(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        potential_paths=potential_paths
    )
    
    # Visualize reasoning flow
    flow_anim = animate_reasoning_flow(
        path_results=path_results,
        tokens=activations["tokens"],
        model_layers=len(activations["layer_max_probs"][intermediate_concepts[0]])
    )
    
    # Perform causal intervention
    intervention_results = perform_causal_intervention(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        concepts=["30", "multiply"]
    )
    
    # Visualize intervention results
    intervention_plot = plot_layer_position_intervention(intervention_results)
    display(intervention_plot)
    
    return {
        "activations": activations,
        "path_results": path_results,
        "intervention_results": intervention_results
    }

if __name__ == "__main__":
    results = analyze_math_problem()
    # Save animation as GIF
    save_animation(
        path_results=results["path_results"],
        tokens=results["activations"]["tokens"],
        model_layers=len(results["activations"]["layer_max_probs"][results["activations"]["intermediate_concepts"][0]]),
        output_path="math_reasoning_flow.gif",
        format="gif"
    )
