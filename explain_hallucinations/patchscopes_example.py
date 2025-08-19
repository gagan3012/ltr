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


def visualize_results(results, title="Patchscope Analysis", prompt="", save_path=None):
    """Create enhanced visualizations of the analysis results.

    Args:
        results: The analysis results dictionary
        title: The title for the visualization
        prompt: The original prompt used to generate the results
        save_path: Optional path to save figures as PNG files
    """
    print(f"\nCreating enhanced visualizations for: {title}")

    # Set a consistent, color-blind friendly palette
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # 1. Enhanced Entity Trajectory Plot
    if results["entity_traces"]:
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot a significance threshold
        ax.axhline(
            y=0.1,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Significance threshold",
        )

        # Sort entities by max probability for better legend ordering
        entity_max_probs = {}
        for entity, trajectory in results["entity_traces"].items():
            if trajectory:
                entity_max_probs[entity] = max(
                    step["probability"] for step in trajectory
                )

        # Plot entities in order of maximum probability
        for i, (entity, _) in enumerate(
            sorted(entity_max_probs.items(), key=lambda x: x[1], reverse=True)
        ):
            trajectory = results["entity_traces"][entity]
            if trajectory:
                steps = [step["step"] for step in trajectory]
                probs = [step["probability"] for step in trajectory]
                color = colors[i % len(colors)]
                line = ax.plot(
                    steps,
                    probs,
                    marker="o",
                    label=entity,
                    linewidth=3,
                    markersize=8,
                    color=color,
                    alpha=0.8,
                )

                # Annotate maximum probability point
                max_idx = probs.index(max(probs))
                ax.annotate(
                    f"{probs[max_idx]:.3f}",
                    xy=(steps[max_idx], probs[max_idx]),
                    xytext=(10, 5),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )

                # Annotate final probability
                if steps[-1] == max(steps):
                    ax.annotate(
                        f"{entity}: {probs[-1]:.3f}",
                        xy=(steps[-1], probs[-1]),
                        xytext=(15, 0),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    )

        # Add context information
        generation = results.get("generated_text", "")
        context_text = f'Prompt: "{prompt}"\nGeneration: "{generation}"'
        ax.text(
            0.01,
            0.01,
            context_text,
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", alpha=0.8),
        )

        # Enhance grid and formatting
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_xlabel("Generation Step", fontsize=14, fontweight="bold")
        ax.set_ylabel("Entity Probability", fontsize=14, fontweight="bold")
        ax.set_title(
            f"{title} - Entity Probability Trajectories",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Add generated tokens as x-tick labels if available
        if results.get("generation_trace"):
            token_labels = []
            for step_data in results["generation_trace"]:
                if (
                    "next_token_info" in step_data
                    and "token" in step_data["next_token_info"]
                ):
                    token = step_data["next_token_info"]["token"]
                    token_labels.append(f"{step_data['step']}: '{token}'")
                else:
                    token_labels.append(str(step_data.get("step", "")))

            # Only use token labels if we have a reasonable number
            if len(token_labels) <= 25:
                plt.xticks(
                    range(len(token_labels)), token_labels, rotation=45, ha="right"
                )

        # Better legend placement and formatting
        legend = ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=12,
            title="Tracked Entities",
            title_fontsize=13,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        # Add summary statistics as text
        stats_text = "Summary Statistics:\n"
        for entity, prob_max in sorted(
            entity_max_probs.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            stats_text += f"- {entity}: max={prob_max:.4f}\n"

        # Add key insights if available
        if "summary" in results and "key_insights" in results["summary"]:
            stats_text += "\nKey Insights:\n"
            insights = results["summary"]["key_insights"]
            if isinstance(insights, list):
                for insight in insights[:3]:  # Top 3 insights
                    stats_text += f"- {insight}\n"
            else:
                stats_text += f"- {insights}\n"

        # Add the stats text box
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.4)
        ax.text(
            0.01,
            0.99,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            va="top",
            bbox=props,
        )

        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            fig_path = f"{save_path}_entity_trajectories.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Entity trajectory plot saved to: {fig_path}")

        plt.show()

    # 2. Enhanced Attention Patterns Heatmap
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

        fig, ax = plt.subplots(figsize=(16, 8))

        # Create enhanced heatmap
        heatmap = sns.heatmap(
            pivot_df,
            cmap="viridis",
            cbar_kws={"label": "Attention Sum", "shrink": 0.8},
            annot=True,  # Show values in cells
            fmt=".2f",  # Format for annotations
            linewidths=0.5,
            ax=ax,
        )

        # Get generated tokens if available for x-axis labels
        token_labels = []
        if results.get("generation_trace"):
            for i in range(len(results["generation_trace"])):
                if i < len(results["generation_trace"]):
                    step_data = results["generation_trace"][i]
                    if (
                        "next_token_info" in step_data
                        and "token" in step_data["next_token_info"]
                    ):
                        token = step_data["next_token_info"]["token"]
                        token_labels.append(f"{i}: '{token}'")

        if token_labels:
            ax.set_xticklabels(token_labels, rotation=45, ha="right")

        # Enhance the plot
        ax.set_title(
            f"{title} - Attention Patterns by Layer and Generation Step",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_ylabel("Model Layer", fontsize=14, fontweight="bold")
        ax.set_xlabel("Generation Step", fontsize=14, fontweight="bold")

        # Add context information
        generation = results.get("generated_text", "")
        context_text = f'Prompt: "{prompt}"\nGeneration: "{generation}"'
        ax.text(
            0.01,
            0.01,
            context_text,
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        )

        # Highlight high attention regions
        if not pivot_df.empty:
            max_attention = pivot_df.max().max()
            threshold = max_attention * 0.8

            # Find coordinates of high attention points
            for layer in pivot_df.index:
                for step in pivot_df.columns:
                    val = pivot_df.loc[layer, step]
                    if val >= threshold:
                        ax.add_patch(
                            plt.Rectangle(
                                (step - 0.5, layer - 0.5),
                                1,
                                1,
                                fill=False,
                                edgecolor="red",
                                linewidth=2,
                            )
                        )

        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            fig_path = f"{save_path}_attention_heatmap.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Attention heatmap saved to: {fig_path}")

        plt.show()

    # 3. NEW: Entity Probability Bar Chart (final state)
    if results["entity_traces"]:
        # Get final probabilities for each entity
        final_probs = {}
        for entity, trajectory in results["entity_traces"].items():
            if trajectory:
                final_probs[entity] = trajectory[-1]["probability"]

        if final_probs:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Sort entities by final probability
            sorted_entities = sorted(
                final_probs.items(), key=lambda x: x[1], reverse=True
            )
            entities = [item[0] for item in sorted_entities]
            probabilities = [item[1] for item in sorted_entities]

            # Create bar chart with custom colors
            bars = ax.bar(entities, probabilities, color=colors[: len(entities)])

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

            # Add a threshold line
            ax.axhline(
                y=0.1,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Significance threshold",
            )

            # Enhance the plot
            ax.set_xlabel("Entity", fontsize=14, fontweight="bold")
            ax.set_ylabel("Final Probability", fontsize=14, fontweight="bold")
            ax.set_title(
                f"{title} - Final Entity Probabilities", fontsize=16, fontweight="bold"
            )
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            # Add context
            generation = results.get("generated_text", "")
            context_text = f'Prompt: "{prompt}"\nGeneration: "{generation}"'
            ax.text(
                0.01,
                0.01,
                context_text,
                transform=ax.transAxes,
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", alpha=0.8),
            )

            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # Save figure if path provided
            if save_path:
                fig_path = f"{save_path}_final_probabilities.png"
                plt.savefig(fig_path, dpi=300, bbox_inches="tight")
                print(f"Final probabilities chart saved to: {fig_path}")

            plt.show()

    # 4. NEW: Layer Activation Norms Visualization
    layer_activations = {}
    for step in results["generation_trace"]:
        step_idx = step["step"]
        for layer_idx, layer_data in step["layer_activations"].items():
            if "activation_norm" in layer_data:
                if layer_idx not in layer_activations:
                    layer_activations[layer_idx] = []
                while len(layer_activations[layer_idx]) < step_idx:
                    layer_activations[layer_idx].append(None)
                layer_activations[layer_idx].append(layer_data["activation_norm"])

    if layer_activations:
        fig, ax = plt.subplots(figsize=(14, 8))

        for i, (layer_idx, norms) in enumerate(sorted(layer_activations.items())):
            # Filter out None values and create corresponding steps
            valid_norms = [n for n in norms if n is not None]
            valid_steps = [idx for idx, n in enumerate(norms) if n is not None]

            color = colors[i % len(colors)]
            ax.plot(
                valid_steps,
                valid_norms,
                marker="o",
                label=f"Layer {layer_idx}",
                linewidth=2.5,
                markersize=7,
                color=color,
                alpha=0.8,
            )

        # Enhance the plot
        ax.set_xlabel("Generation Step", fontsize=14, fontweight="bold")
        ax.set_ylabel("Activation Norm", fontsize=14, fontweight="bold")
        ax.set_title(
            f"{title} - Layer Activation Norms", fontsize=16, fontweight="bold"
        )
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        # Better legend
        legend = ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=12,
            title="Model Layers",
            title_fontsize=13,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        # Add generated tokens as x-tick labels if available
        if results.get("generation_trace") and len(results["generation_trace"]) <= 20:
            token_labels = []
            for step_data in results["generation_trace"]:
                if (
                    "next_token_info" in step_data
                    and "token" in step_data["next_token_info"]
                ):
                    token = step_data["next_token_info"]["token"]
                    token_labels.append(f"{step_data['step']}: '{token}'")
                else:
                    token_labels.append(str(step_data.get("step", "")))

            plt.xticks(range(len(token_labels)), token_labels, rotation=45, ha="right")

        # Add context
        generation = results.get("generated_text", "")
        context_text = f'Prompt: "{prompt}"\nGeneration: "{generation}"'
        ax.text(
            0.01,
            0.01,
            context_text,
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="whitesmoke", alpha=0.8),
        )

        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            fig_path = f"{save_path}_layer_activations.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Layer activations plot saved to: {fig_path}")

        plt.show()


def main():
    """Run all examples and create visualizations."""
    print("LTR Patchscopes Hallucination Analysis Examples")
    print("=" * 60)
    import os

    viz_dir = "visualization_results"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Run examples
    try:
        results1 = example_1_basic_hallucination_detection()
        prompt1 = "The capital of Mars is"
        visualize_results(
            results1,
            "Basic Hallucination Detection",
            prompt1,
            save_path=os.path.join(viz_dir, "example1"),
        )

        results2 = example_2_comparative_analysis()
        prompts2 = {
            "factual": "The capital of France is",
            "potentially_hallucinated": "The capital of Atlantis is",
            "mixed": "Einstein discovered relativity in",
        }
        for prompt_type, results in results2.items():
            visualize_results(
                results,
                f"Comparative Analysis - {prompt_type}",
                prompts2[prompt_type],
                save_path=os.path.join(viz_dir, f"example2_{prompt_type}"),
            )

        results3 = example_3_layer_by_layer_analysis()
        prompt3 = "The first person to walk on Venus was"
        visualize_results(
            results3,
            "Layer-by-Layer Analysis",
            prompt3,
            save_path=os.path.join(viz_dir, "example3"),
        )

        results4 = example_4_hallucination_intervention()
        prompts4 = {
            "ambiguous": "The author of Harry Potter",
            "guided_correct": "The author of Harry Potter is J.K. Rowling, who",
            "guided_incorrect": "The author of Harry Potter is Stephen King, who",
        }
        for variation, results in results4.items():
            visualize_results(
                results,
                f"Intervention Analysis - {variation}",
                prompts4[variation],
                save_path=os.path.join(viz_dir, f"example4_{variation}"),
            )

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
