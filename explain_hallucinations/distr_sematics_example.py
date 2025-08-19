import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import dendrogram, linkage
from IPython.display import HTML

# Import LTR tools
from ltr.concept_extraction import extract_concept_activations
from ltr.logit_lens import logit_lens_analysis, trace_token_evolution
from ltr.entity_analysis import analyze_causal_entities
from ltr.visualization import (
    plot_concept_activations,
    animate_concept_activation_diagonal,
    animate_concept_evolution,
)


def setup_model(model_name="gpt2-medium"):
    """Set up the model for analysis."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, tokenizer


def analyze_distributional_semantics(
    model, tokenizer, prompt, concepts, related_concepts=None
):
    """
    Analyzes the distributional semantics of concepts across model layers.

    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        prompt: The input text for analysis
        concepts: List of primary concepts to track
        related_concepts: Optional list of related concepts to compare against primary concepts
    """
    print(f"Analyzing distributional semantics for prompt: {prompt}")
    print(f"Tracking concepts: {concepts}")

    # 1. Extract concept activations across all layers
    concept_results = extract_concept_activations(
        model,
        tokenizer,
        prompt,
        intermediate_concepts=concepts,
        final_concepts=related_concepts or [],
    )

    # 2. Use logit lens to see how concepts evolve through the network
    logit_results = logit_lens_analysis(model, tokenizer, prompt, top_k=10)

    # 3. Track the evolution of specific concepts through all layers
    if related_concepts:
        concept_evolution = trace_token_evolution(
            model, tokenizer, prompt, target_tokens=concepts + related_concepts
        )

    # 4. Analyze causal influence of entities
    entity_influences = analyze_causal_entities(
        model, tokenizer, prompt, target_entities=concepts
    )

    # Return the full results for visualization
    results = {
        "concept_results": concept_results,
        "logit_results": logit_results,
        "entity_influences": entity_influences,
    }

    if related_concepts:
        results["concept_evolution"] = concept_evolution

    return results


def plot_layer_semantic_heatmap(
    results, concepts, title="Layer-wise Semantic Activation", figsize=(14, 10)
):
    """
    Creates an enhanced heatmap visualization of concept activations across layers.

    Args:
        results: Results from analyze_distributional_semantics
        concepts: List of concepts to plot
        title: Title for the plot
        figsize: Figure size as tuple
    """
    concept_results = results["concept_results"]
    n_layers = len(next(iter(concept_results["layer_max_probs"].values())))

    # Create a DataFrame with layer data
    layer_data = []
    for concept in concepts:
        if concept in concept_results["layer_max_probs"]:
            for layer, prob in enumerate(concept_results["layer_max_probs"][concept]):
                layer_data.append(
                    {"Layer": layer, "Concept": concept, "Activation": prob}
                )

    df = pd.DataFrame(layer_data)

    # Calculate normalized activations for better visualization
    pivot_df = df.pivot(index="Concept", columns="Layer", values="Activation")

    # Normalize by row (concept) for better comparison
    normalized_df = pivot_df.div(pivot_df.max(axis=1), axis=0)

    # Create a custom colormap for better visualization
    colors = ["#f0f0f0", "#d4e6f1", "#85c1e9", "#3498db", "#1a5276"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom_blues", colors, N=n_bins)

    plt.figure(figsize=figsize)

    # Plot the heatmap with improved styling
    ax = sns.heatmap(
        normalized_df,
        cmap=cmap,
        linewidths=0.1,
        linecolor="gray",
        cbar_kws={"label": "Normalized Activation", "shrink": 0.8},
    )

    # Enhance the plot styling
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Model Layer", fontsize=14, labelpad=10)
    plt.ylabel("Concept", fontsize=14, labelpad=10)

    # Show layer groupings (early, middle, late)
    layer_thirds = n_layers // 3
    plt.axvline(x=layer_thirds, color="gray", linestyle="--", alpha=0.7)
    plt.axvline(x=2 * layer_thirds, color="gray", linestyle="--", alpha=0.7)

    # Add layer group labels
    plt.text(layer_thirds / 2, -0.8, "Early Layers", ha="center", fontsize=12)
    plt.text(
        layer_thirds + layer_thirds / 2, -0.8, "Middle Layers", ha="center", fontsize=12
    )
    plt.text(
        2 * layer_thirds + (n_layers - 2 * layer_thirds) / 2,
        -0.8,
        "Late Layers",
        ha="center",
        fontsize=12,
    )

    # Rotate x-axis labels for better readability
    plt.xticks(np.arange(0, n_layers, 4), fontsize=10, rotation=0)
    plt.yticks(fontsize=11)

    plt.tight_layout()
    plt.savefig("layer_semantic_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    return normalized_df


def plot_concept_semantic_flow(results, concepts, figsize=(15, 10)):
    """
    Creates a flowing visualization of how concepts evolve through the model.
    """
    concept_results = results["concept_results"]

    # Get tokens from results
    tokens = concept_results["tokens"]

    # Create animated visualization using the existing function
    for concept in concepts:
        if concept in concept_results["activation_grid"]:
            print(f"Creating semantic flow animation for concept: {concept}")
            ani = animate_concept_activation_diagonal(
                concept_results,
                selected_concepts=[concept],
                compression_factor=2,
                figsize=figsize,
                interval=80,
            )

            # For non-notebook environments, save as video
            try:
                # Try to save as HTML for notebook environments
                with open(f"semantic_flow_{concept}.html", "w") as f:
                    f.write(ani.to_html5_video())
            except:
                # If that fails, save as mp4
                try:
                    ani.save(
                        f"semantic_flow_{concept}.mp4", writer="ffmpeg", fps=10, dpi=200
                    )
                except:
                    print(
                        "Could not save animation. Consider displaying in a notebook environment."
                    )


def create_concept_embedding_map(
    results, concepts, layer_indices=None, figsize=(14, 12)
):
    """
    Creates a 2D map of concept embeddings across selected layers.

    Args:
        results: Results from analyze_distributional_semantics
        concepts: List of concepts to visualize
        layer_indices: List of layer indices to visualize (if None, uses evenly spaced layers)
        figsize: Figure size as tuple
    """
    concept_results = results["concept_results"]
    activation_grid = concept_results["activation_grid"]

    n_layers = next(iter(activation_grid.values())).shape[0]
    if layer_indices is None:
        # Choose evenly spaced layers for visualization
        n_layers_to_show = min(6, n_layers)
        layer_indices = np.linspace(0, n_layers - 1, n_layers_to_show, dtype=int)

    # Collect embeddings for each concept at each selected layer
    layer_embeddings = {}
    for layer_idx in layer_indices:
        embeddings = []
        labels = []

        for concept in concepts:
            if concept in activation_grid:
                # Get activation pattern across positions for this concept at this layer
                activation_pattern = activation_grid[concept][layer_idx]

                if np.sum(activation_pattern) > 0:  # Only include if there's activation
                    embeddings.append(activation_pattern)
                    labels.append(concept)

        if embeddings:
            layer_embeddings[layer_idx] = (np.vstack(embeddings), labels)

    # Create multi-panel visualization
    n_layers_available = len(layer_embeddings)
    if n_layers_available == 0:
        print("No layers with sufficient activations found for visualization")
        return

    n_cols = min(3, n_layers_available)
    n_rows = (n_layers_available + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Custom color palette for concept categories
    concept_categories = {c: i % 10 for i, c in enumerate(concepts)}
    palette = sns.color_palette("tab10", n_colors=10)

    layer_idx_to_plot_idx = {
        layer_idx: i for i, layer_idx in enumerate(sorted(layer_embeddings.keys()))
    }

    # For each layer, create a 2D embedding visualization
    for layer_idx, (embeddings, labels) in layer_embeddings.items():
        plot_idx = layer_idx_to_plot_idx[layer_idx]
        ax = axes[plot_idx]

        # Apply dimensionality reduction
        if embeddings.shape[0] >= 3:
            try:
                # Try t-SNE first
                tsne = TSNE(
                    n_components=2,
                    perplexity=min(5, embeddings.shape[0] - 1),
                    random_state=42,
                    init="pca",
                    learning_rate="auto",
                )
                transformed = tsne.fit_transform(embeddings)
                reduction_method = "t-SNE"
            except:
                # Fall back to PCA if t-SNE fails
                pca = PCA(n_components=2)
                transformed = pca.fit_transform(embeddings)
                reduction_method = "PCA"
        else:
            # Use PCA for very small datasets
            pca = PCA(n_components=2)
            transformed = pca.fit_transform(embeddings)
            reduction_method = "PCA"

        # Plot points
        for i, (x, y) in enumerate(transformed):
            label = labels[i]
            color_idx = concept_categories[label]
            ax.scatter(
                x,
                y,
                s=100,
                color=palette[color_idx],
                alpha=0.7,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.text(
                x,
                y,
                label,
                fontsize=9,
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
            )

        # Customize plot
        ax.set_title(
            f"Layer {layer_idx} Concept Space ({reduction_method})", fontsize=12
        )
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for i in range(n_layers_available, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Concept Semantic Space by Layer", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("concept_embedding_map.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_semantic_clustering(results, concepts, figsize=(14, 10)):
    """
    Creates a hierarchical clustering visualization of concepts based on their activation patterns.

    Args:
        results: Results from analyze_distributional_semantics
        concepts: List of concepts to visualize
        figsize: Figure size as tuple
    """
    concept_results = results["concept_results"]
    activation_grid = concept_results["activation_grid"]

    # Collect activation vectors for each concept (flattened across layers and positions)
    concept_vectors = {}
    for concept in concepts:
        if concept in activation_grid:
            # Flatten the 2D activation grid to a 1D vector
            concept_vectors[concept] = activation_grid[concept].flatten()

    # Skip if no valid concepts
    if not concept_vectors:
        print("No valid concepts for clustering")
        return

    # Create distance matrix
    valid_concepts = list(concept_vectors.keys())
    n_concepts = len(valid_concepts)
    distance_matrix = np.zeros((n_concepts, n_concepts))

    for i, c1 in enumerate(valid_concepts):
        for j, c2 in enumerate(valid_concepts):
            if i > j:  # Only compute lower triangle
                v1 = concept_vectors[c1]
                v2 = concept_vectors[c2]
                # Use cosine distance
                distance_matrix[i, j] = cosine(v1, v2)
                distance_matrix[j, i] = distance_matrix[i, j]

    # Perform hierarchical clustering
    condensed_dist = distance_matrix[np.triu_indices(n_concepts, k=1)]
    Z = linkage(condensed_dist, method="ward")

    # Create the plot
    plt.figure(figsize=figsize)

    # Plot dendrogram
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=4, rowspan=2)
    dendrogram(Z, labels=valid_concepts, leaf_rotation=90, leaf_font_size=10, ax=ax1)
    ax1.set_title(
        "Hierarchical Clustering of Concepts by Activation Patterns", fontsize=16
    )
    ax1.set_ylabel("Distance", fontsize=12)

    # Plot heatmap of distance matrix
    ax2 = plt.subplot2grid((3, 4), (2, 0), colspan=4, rowspan=1)
    im = ax2.imshow(distance_matrix, cmap="YlGnBu_r", aspect="auto")
    ax2.set_xticks(range(n_concepts))
    ax2.set_xticklabels(valid_concepts, rotation=45, ha="right", fontsize=10)
    ax2.set_yticks(range(n_concepts))
    ax2.set_yticklabels(valid_concepts, fontsize=10)
    ax2.set_title("Concept Distance Matrix", fontsize=14)

    # Add colorbar
    plt.colorbar(im, ax=ax2, label="Cosine Distance", shrink=0.7)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig("concept_clustering.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_animated_radar_chart(results, concepts, n_layers=6, figsize=(12, 10)):
    """
    Creates an animated radar chart showing how concept representations evolve across layers.

    Args:
        results: Results from analyze_distributional_semantics
        concepts: List of concepts to visualize
        n_layers: Number of layers to include in the animation
        figsize: Figure size as tuple
    """
    concept_results = results["concept_results"]
    layer_max_probs = concept_results["layer_max_probs"]

    # Get total number of layers
    total_layers = len(next(iter(layer_max_probs.values())))

    # Select layers evenly spaced throughout the model
    layer_indices = np.linspace(0, total_layers - 1, n_layers, dtype=int)

    # Set up the figure for animation
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, polar=True)

    # Categories (concepts) for the radar chart
    categories = [c for c in concepts if c in layer_max_probs]
    if not categories:
        print("No valid concepts for radar chart")
        return

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Function to update the radar chart for each layer
    def update(i):
        ax.clear()

        # Get layer index
        layer_idx = layer_indices[i % len(layer_indices)]

        # Get values for this layer
        values = [layer_max_probs[c][layer_idx] for c in categories]
        values += values[:1]  # Close the loop

        # Plot values
        ax.plot(
            angles, values, linewidth=2, linestyle="solid", label=f"Layer {layer_idx}"
        )
        ax.fill(angles, values, alpha=0.25)

        # Customize plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="gray", size=9)
        ax.set_ylim(0, 1)

        plt.title(f"Concept Activation Pattern at Layer {layer_idx}", size=16, y=1.1)

        # Add legend
        ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        return [ax]

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(layer_indices), interval=800, blit=True
    )

    # Save animation
    try:
        ani.save("concept_radar_animation.mp4", writer="ffmpeg", fps=1)
    except:
        print(
            "Could not save animation. Consider displaying in a notebook environment."
        )

    plt.close()


def plot_layer_specialization(results, figsize=(14, 8)):
    """
    Visualizes how different layers specialize in different semantic aspects.

    Args:
        results: Results from analyze_distributional_semantics
        figsize: Figure size as tuple
    """
    concept_results = results["concept_results"]
    logit_results = results["logit_results"]

    # Extract layer results
    layer_results = logit_results["layer_results"]
    n_layers = len(layer_results)

    # Calculate diversity metrics for each layer
    layer_metrics = []

    for layer_idx, positions in layer_results.items():
        # Only consider layers with position data
        if positions:
            # Calculate entropy of token distributions
            entropies = []
            top1_probs = []

            for pos_data in positions:
                top_tokens = pos_data["top_tokens"]
                if top_tokens:
                    # Extract probabilities
                    probs = [prob for _, prob in top_tokens]

                    # Calculate entropy
                    entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
                    entropies.append(entropy)

                    # Extract top token probability
                    top1_probs.append(probs[0])

            # Calculate average metrics for this layer
            if entropies:
                avg_entropy = np.mean(entropies)
                avg_top1_prob = np.mean(top1_probs)

                layer_metrics.append(
                    {
                        "layer": layer_idx,
                        "avg_entropy": avg_entropy,
                        "avg_top1_prob": avg_top1_prob,
                    }
                )

    # Convert to DataFrame
    df = pd.DataFrame(layer_metrics)

    # Plot
    plt.figure(figsize=figsize)

    # Create two subplots
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    # Plot entropy
    sns.lineplot(x="layer", y="avg_entropy", data=df, marker="o", ax=ax1)
    ax1.set_title("Token Distribution Entropy by Layer", fontsize=14)
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Average Entropy", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Highlight layer regions
    n_layers = df["layer"].max() + 1
    third = n_layers // 3
    ax1.axvspan(0, third - 1, alpha=0.2, color="blue", label="Early Layers")
    ax1.axvspan(third, 2 * third - 1, alpha=0.2, color="green", label="Middle Layers")
    ax1.axvspan(2 * third, n_layers, alpha=0.2, color="red", label="Late Layers")
    ax1.legend()

    # Plot top-1 probability
    sns.lineplot(
        x="layer", y="avg_top1_prob", data=df, marker="o", color="orange", ax=ax2
    )
    ax2.set_title("Top Token Probability by Layer", fontsize=14)
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Average Top-1 Probability", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Highlight layer regions
    ax2.axvspan(0, third - 1, alpha=0.2, color="blue", label="Early Layers")
    ax2.axvspan(third, 2 * third - 1, alpha=0.2, color="green", label="Middle Layers")
    ax2.axvspan(2 * third, n_layers, alpha=0.2, color="red", label="Late Layers")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("layer_specialization.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_full_analysis():
    """Run a comprehensive analysis with multiple examples."""
    model, tokenizer = setup_model()

    print("\n=== Example 1: Concrete vs. Abstract Concepts ===")
    prompt = "The tree in the garden provides shade during hot summer days. It's an oak tree that has stood there for centuries."
    concepts = ["tree", "garden", "shade", "oak", "centuries"]
    results_concrete = analyze_distributional_semantics(
        model, tokenizer, prompt, concepts
    )

    # Create visualizations
    plot_layer_semantic_heatmap(
        results_concrete, concepts, title="Concrete Concept Activation Across Layers"
    )
    plot_concept_semantic_flow(results_concrete, concepts)
    create_concept_embedding_map(results_concrete, concepts)
    plot_semantic_clustering(results_concrete, concepts)
    create_animated_radar_chart(results_concrete, concepts)
    plot_layer_specialization(results_concrete)

    print("\n=== Example 2: Semantic Relationships ===")
    prompt = "In linguistics, the relationship between similar words like 'happy' and 'joyful' is called synonymy, while the relationship between opposite words like 'fast' and 'slow' is called antonymy."
    concepts = ["synonymy", "antonymy"]
    related_concepts = ["happy", "joyful", "fast", "slow"]
    results_relationships = analyze_distributional_semantics(
        model, tokenizer, prompt, concepts, related_concepts
    )

    # Create visualizations
    plot_layer_semantic_heatmap(
        results_relationships,
        concepts + related_concepts,
        title="Semantic Relationship Concept Activation",
    )
    create_concept_embedding_map(results_relationships, concepts + related_concepts)
    plot_semantic_clustering(results_relationships, concepts + related_concepts)

    print("\n=== Example 3: Abstract Reasoning ===")
    prompt = "In mathematics, infinity refers to something without any bound or larger than any natural number. In philosophy, the concept of infinity is considered both profound and paradoxical."
    concepts = [
        "infinity",
        "mathematics",
        "philosophy",
        "natural number",
        "paradoxical",
    ]
    results_abstract = analyze_distributional_semantics(
        model, tokenizer, prompt, concepts
    )

    # Create visualizations
    plot_layer_semantic_heatmap(
        results_abstract, concepts, title="Abstract Concept Activation Across Layers"
    )
    create_concept_embedding_map(results_abstract, concepts)
    create_animated_radar_chart(results_abstract, concepts)

    print("\nAnalysis complete! Visualizations saved to current directory.")


if __name__ == "__main__":
    run_full_analysis()
