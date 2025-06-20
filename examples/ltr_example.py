# Example usage of the ltr package
from transformers import AutoModelForCausalLM, AutoTokenizer
from ltr.concept_extraction import extract_concept_activations
from ltr.reasoning_analysis import analyze_reasoning_paths
from ltr.causal_intervention import perform_causal_intervention
from ltr.visualization import plot_concept_activations, plot_causal_intervention_heatmap
import matplotlib.pyplot as plt

def calculate_simple_math_example():
    """
    Example showing how to trace mathematical reasoning in a model.
    """
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "gpt2"  # Or any HuggingFace model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Define prompt and concepts
    prompt = "To calculate 6 times 5, I multiply 6 by 5 to get 30."
    intermediate_concepts = ["multiply", "multiplication"]
    final_concepts = ["30", "thirty"]

    # Extract and analyze concept activations
    print(f"Extracting concept activations for prompt: {prompt}")
    activations = extract_concept_activations(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        intermediate_concepts=intermediate_concepts,
        final_concepts=final_concepts
    )
    
    # Visualize concept activations
    print("Plotting concept activations...")
    fig = plot_concept_activations(activations)
    plt.savefig("math_concept_activations.png")
    plt.close(fig)
    
    # Analyze potential reasoning paths
    potential_paths = [
        ["multiply", "30"],
        ["multiplication", "30"]
    ]
    
    print("Analyzing reasoning paths...")
    path_results = analyze_reasoning_paths(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        potential_paths=potential_paths
    )
    
    # Print reasoning path results
    best_path = path_results["best_path"]
    best_score = path_results["best_path_score"]
    print(f"Best reasoning path: {' -> '.join(best_path)} (Score: {best_score:.3f})")
    
    # Perform causal intervention
    print("Performing causal intervention...")
    intervention_results = perform_causal_intervention(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        concepts=["30"],
        target_positions=[3]  # Position of "multiply"
    )
    
    # Visualize causal intervention
    if intervention_results["intervention_grids"]["30"]:
        fig = plot_causal_intervention_heatmap(intervention_results, "30")
        plt.savefig("math_causal_intervention.png")
        plt.close(fig)
    
    print("Analysis complete! Check the saved visualization files.")

if __name__ == "__main__":
    calculate_simple_math_example()
