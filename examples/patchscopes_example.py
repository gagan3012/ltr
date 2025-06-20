import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ltr.patchscopes import (
    perform_patchscope_analysis,
    analyze_entity_trajectories,
    analyze_llm_hallucinations_with_patchscopes,
)
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def setup_model_and_tokenizer(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    """Setup model and tokenizer for analysis."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    return model, tokenizer


def example_1_basic_hallucination_detection():
    """Example 1: Basic hallucination detection with entity tracking."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Hallucination Detection")
    print("=" * 60)

    model, tokenizer = setup_model_and_tokenizer()

    # Prompt that might lead to hallucination
    prompt = "The capital of Mars is"

    # Entities we expect might appear (some correct, some hallucinated)
    entities_to_track = [
        "Earth",
        "Mars",
        "New",
        "York",
        "London",
        "Paris",
        "Tokyo",
        "fictional",
        "colony",
        "red",
        "planet",
    ]

    print(f"Analyzing prompt: '{prompt}'")
    print(f"Tracking entities: {entities_to_track}")

    # Perform analysis
    results = perform_patchscope_analysis(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_entities=entities_to_track,
        max_tokens=20,
        target_layers=[0, 4, 8, 12]
        if hasattr(model.config, "num_hidden_layers")
        else [0, 2, 4, 6],
        explanation_prompts=[
            "Is this factually accurate?",
            "What concept is being processed?",
            "Is this about a real place?",
            "What is the confidence level?",
        ],
    )

    print(f"\nGenerated text: {results['generated_text']}")
    print(f"Total generation steps: {results['summary']['total_generation_steps']}")

    # Display entity trajectories
    print("\nEntity Probability Trajectories:")
    for entity, trajectory in results["entity_traces"].items():
        if trajectory:  # Only show entities that appeared
            avg_prob = sum(step["probability"] for step in trajectory) / len(trajectory)
            max_prob = max(step["probability"] for step in trajectory)
            print(
                f"  {entity}: avg={avg_prob:.4f}, max={max_prob:.4f}, steps={len(trajectory)}"
            )

    # Show key insights
    print(f"\nKey Insights: {results['summary']['key_insights']}")

    return results


def example_2_comparative_analysis():
    """Example 2: Compare factual vs potentially hallucinated prompts."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Comparative Analysis - Factual vs Hallucinated")
    print("=" * 60)

    model, tokenizer = setup_model_and_tokenizer()

    prompts = {
        "factual": "The capital of France is",
        "potentially_hallucinated": "The capital of Atlantis is",
        "mixed": "Einstein discovered relativity in",
    }

    entities = {
        "factual": ["Paris", "France", "city", "European"],
        "potentially_hallucinated": [
            "Atlantis",
            "fictional",
            "underwater",
            "mythical",
            "Poseidon",
        ],
        "mixed": ["Einstein", "relativity", "1905", "Newton", "gravity", "Princeton"],
    }

    comparative_results = {}

    for prompt_type, prompt in prompts.items():
        print(f"\nAnalyzing {prompt_type} prompt: '{prompt}'")

        results = perform_patchscope_analysis(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_entities=entities[prompt_type],
            max_tokens=15,
            target_layers=[0, 6, 12]
            if hasattr(model.config, "num_hidden_layers")
            else [0, 3, 6],
            explanation_prompts=[
                "Is this historically accurate?",
                "Is this about a real entity?",
                "What is the factual basis?",
            ],
        )

        comparative_results[prompt_type] = results

        print(f"  Generated: {results['generated_text']}")
        print(f"  Steps: {results['summary']['total_generation_steps']}")

        # Show most probable entities
        if results["summary"]["average_entity_probabilities"]:
            top_entity = max(
                results["summary"]["average_entity_probabilities"].items(),
                key=lambda x: x[1],
            )
            print(f"  Top entity: {top_entity[0]} (avg prob: {top_entity[1]:.4f})")

    return comparative_results


def example_3_layer_by_layer_analysis():
    """Example 3: Detailed layer-by-layer analysis of hallucination emergence."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Layer-by-Layer Hallucination Analysis")
    print("=" * 60)

    model, tokenizer = setup_model_and_tokenizer()

    # Known hallucination prompt
    prompt = "The first person to walk on Venus was"
    entities = [
        "Venus",
        "Neil",
        "Armstrong",
        "Buzz",
        "Aldrin",
        "astronaut",
        "impossible",
        "surface",
    ]

    print(f"Analyzing prompt: '{prompt}'")

    # Analyze with many layers for detailed view
    n_layers = getattr(model.config, "num_hidden_layers", 12)
    target_layers = list(
        range(0, n_layers, max(1, n_layers // 8))
    )  # Sample 8 layers evenly

    results = perform_patchscope_analysis(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_entities=entities,
        max_tokens=25,
        target_layers=target_layers,
        window_size=3,
        explanation_prompts=[
            "Is this physically possible?",
            "What planet is being discussed?",
            "Is this about space exploration?",
            "What is the factual accuracy?",
        ],
    )

    print(f"Generated text: {results['generated_text']}")

    # Analyze layer-specific patterns
    print("\nLayer-by-Layer Analysis:")
    for step_idx, step in enumerate(results["generation_trace"][:5]):  # First 5 steps
        print(f"\nStep {step_idx}: Token '{step['next_token_info']['token']}'")

        for layer_idx, layer_data in step["layer_activations"].items():
            if "activation_norm" in layer_data:
                norm = layer_data["activation_norm"]
                mean = layer_data["activation_mean"]
                print(f"  Layer {layer_idx}: norm={norm:.2f}, mean={mean:.3f}")

        # Show entity probabilities for this step
        if step["entity_probabilities"]:
            top_entities = sorted(
                step["entity_probabilities"].items(), key=lambda x: x[1], reverse=True
            )[:3]
            print(f"  Top entities: {top_entities}")

        # Show explanations if available
        if step["explanations"]:
            for layer_idx, layer_explanations in step["explanations"].items():
                for expl in layer_explanations[:2]:  # First 2 explanations
                    print(f"    Layer {layer_idx}: {expl['explanation']}")

    return results


def example_4_hallucination_intervention():
    """Example 4: Using patchscopes for hallucination intervention analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Hallucination Intervention Analysis")
    print("=" * 60)

    model, tokenizer = setup_model_and_tokenizer()

    # Test different versions of a potentially problematic prompt
    base_prompt = "The author of Harry Potter"

    prompt_variations = {
        "ambiguous": base_prompt,
        "guided_correct": base_prompt + " is J.K. Rowling, who",
        "guided_incorrect": base_prompt + " is Stephen King, who",
    }

    entities = [
        "Rowling",
        "Stephen",
        "King",
        "author",
        "Harry",
        "Potter",
        "books",
        "series",
    ]

    intervention_results = {}

    for variation_name, prompt in prompt_variations.items():
        print(f"\nAnalyzing variation '{variation_name}': '{prompt}'")

        results = analyze_llm_hallucinations_with_patchscopes(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            entities_of_interest=entities,
            max_tokens=20,
            target_layers=[0, 4, 8]
            if hasattr(model.config, "num_hidden_layers")
            else [0, 2, 4],
        )

        intervention_results[variation_name] = results

        print(f"  Generated: {results['generated_text']}")

        # Calculate entity confidence scores
        entity_confidence = {}
        for entity, trajectory in results["entity_traces"].items():
            if trajectory:
                # Calculate confidence as max probability achieved
                max_prob = max(step["probability"] for step in trajectory)
                entity_confidence[entity] = max_prob

        print("  Entity confidence scores:")
        for entity, confidence in sorted(
            entity_confidence.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"    {entity}: {confidence:.4f}")

    return intervention_results


def visualize_results(results, title="Patchscope Analysis"):
    """Create visualizations of the analysis results."""
    print(f"\nCreating visualizations for: {title}")

    # Entity trajectory plot
    if results["entity_traces"]:
        plt.figure(figsize=(12, 8))

        for entity, trajectory in results["entity_traces"].items():
            if trajectory:
                steps = [step["step"] for step in trajectory]
                probs = [step["probability"] for step in trajectory]
                plt.plot(steps, probs, marker="o", label=entity, linewidth=2)

        plt.xlabel("Generation Step")
        plt.ylabel("Entity Probability")
        plt.title(f"{title} - Entity Trajectories")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Attention patterns heatmap (if available)
    attention_data = []
    for step in results["generation_trace"]:
        for layer_idx, layer_data in step["layer_activations"].items():
            if "attention" in layer_data and layer_data["attention"]:
                attention_data.append(
                    {
                        "step": step["step"],
                        "layer": layer_idx,
                        "attention_sum": sum(layer_data["attention"]),
                    }
                )

    if attention_data:
        df = pd.DataFrame(attention_data)
        pivot_df = df.pivot(index="layer", columns="step", values="attention_sum")

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_df, cmap="viridis", cbar_kws={"label": "Attention Sum"})
        plt.title(f"{title} - Attention Patterns by Layer and Step")
        plt.ylabel("Layer")
        plt.xlabel("Generation Step")
        plt.tight_layout()
        plt.show()


def main():
    """Run all examples and create visualizations."""
    print("LTR Patchscopes Hallucination Analysis Examples")
    print("=" * 60)

    # Run examples
    try:
        results1 = example_1_basic_hallucination_detection()
        visualize_results(results1, "Basic Hallucination Detection")

        results2 = example_2_comparative_analysis()
        for prompt_type, results in results2.items():
            visualize_results(results, f"Comparative Analysis - {prompt_type}")

        results3 = example_3_layer_by_layer_analysis()
        visualize_results(results3, "Layer-by-Layer Analysis")

        results4 = example_4_hallucination_intervention()
        for variation, results in results4.items():
            visualize_results(results, f"Intervention Analysis - {variation}")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install torch transformers matplotlib seaborn pandas")


def save_results_to_json(results, filename):
    """Save analysis results to JSON file."""

    # Convert tensors and other non-serializable objects
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    serializable_results = convert_for_json(results)

    with open(filename, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()
