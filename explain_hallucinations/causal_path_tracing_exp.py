import torch
from ltr.concept_extraction import extract_concept_activations
from ltr.reasoning_analysis import analyze_reasoning_paths
from ltr.visualization import plot_concept_activation_heatmap, animate_reasoning_flow
import matplotlib.pyplot as plt
from IPython.display import display, HTML

from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_model(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    print("Loading model and tokenizer...")
    # model_name = "meta-llama/Llama-3.2-1B-Instruct"  # A smaller Qwen model for quick loading
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto"
    )

    return model, tokenizer


model, tokenizer = setup_model()

def generate_respnse(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def analyze_reasoning(
    model,
    tokenizer,
    model_name,
    prompt,
    intermediate_concepts,
    final_concepts,
    potential_paths,
):
    print(f"Running concept extraction on model: {model_name}")
    print(f"Model type: {model.config.model_type}")

    # Extract concept activations
    activations = extract_concept_activations(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        intermediate_concepts=intermediate_concepts,
        final_concepts=final_concepts,
    )

    # Visualize concept activations
    print("Generating concept activation heatmap...")
    heatmap = plot_concept_activation_heatmap(activations)
    display(heatmap)

    print("Analyzing reasoning paths...")
    path_results = analyze_reasoning_paths(
        model=model, tokenizer=tokenizer, prompt=prompt, potential_paths=potential_paths
    )

    # Calculate number of layers (for visualization)
    n_layers = len(activations["layer_max_probs"][intermediate_concepts[0]])

    # Visualize reasoning flow
    print("Generating reasoning flow animation...")
    flow_anim = animate_reasoning_flow(
        path_results=path_results, tokens=activations["tokens"], model_layers=n_layers
    )

    # Display animation in notebook or save as GIF
    display(HTML(flow_anim.to_jshtml()))

    print("Model response")
    print("Prompt: ", prompt)
    response = generate_respnse(model, tokenizer, prompt)
    print("Response: ", response)

    # Return results for further analysis
    return {
        "activations": activations,
        "path_results": path_results,
        "model_info": {
            "name": model_name,
            "type": model.config.model_type,
            "n_layers": n_layers,
        },
    }


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # Define prompt and concepts for geographical reasoning
    prompt = "I am at a concert. I see a bass. Is it an fish? Answer in yes or no: "
    potential_paths = [
        ["concert", "bass", "instrument", "Yes"],
        ["concert", "bass", "instrument", "No"],
        ["concert", "bass", "No"],
        ["concert", "bass", "Yes"],
        ["concert", "instrument", "No"],
        ["concert", "instrument", "Yes"],
        ["concert", "bass", "fish", "No"],
        ["concert", "bass", "fish", "Yes"],
        ["concert", "fish", "No"],
        ["concert", "fish", "Yes"],
    ]
    intermediate_concepts = ["concert", "bass", "instrument", "fish"]
    final_concepts = ["Yes", "No"]

    results = analyze_reasoning(
        model,
        tokenizer,
        model_name,
        prompt,
        intermediate_concepts,
        final_concepts,
        potential_paths,
    )

    # Print results
    print("Results:")
    print(f"Best reasoning path: {' → '.join(results['path_results']['best_path'])}")
    print(f"Path score: {results['path_results']['best_path_score']:.4f}")

    # Print model information
    print("\nModel information:")
    for key, value in results["model_info"].items():
        print(f"  {key}: {value}")

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # Define prompt and concepts for geographical reasoning
    prompt = (
        "I am in a forest. I see a trunk. Is it a part of a tree? Answer in yes or no: "
    )
    potential_paths = [
        ["forest", "cars", "trunk", "Yes"],
        ["forest", "cars", "trunk", "No"],
        ["forest", "cars", "No"],
        ["forest", "cars", "Yes"],
        ["forest", "trunk", "No"],
        ["forest", "trunk", "Yes"],
        ["forest", "cars", "tree", "No"],
        ["forest", "cars", "tree", "Yes"],
        ["forest", "tree", "No"],
        ["forest", "tree", "Yes"],
        ["forest", "cars", "trunk", "tree", "No"],
        ["forest", "cars", "trunk", "tree", "Yes"],
    ]

    intermediate_concepts = ["forest", "cars", "trunk", "tree"]
    final_concepts = ["Yes", "No"]

    results = analyze_reasoning(
        model,
        tokenizer,
        model_name,
        prompt,
        intermediate_concepts,
        final_concepts,
        potential_paths,
    )

    # Print results
    print("Results:")
    print(f"Best reasoning path: {' → '.join(results['path_results']['best_path'])}")
    print(f"Path score: {results['path_results']['best_path_score']:.4f}")

    # Print model information
    print("\nModel information:")
    for key, value in results["model_info"].items():
        print(f"  {key}: {value}")
