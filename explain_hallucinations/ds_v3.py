import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from IPython.display import display, HTML
import os
import matplotlib.animation as animation
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.ndimage import gaussian_filter
from baukit import TraceDict

# Import LTR functions
from ltr.concept_extraction import (
    extract_concept_activations,
    get_layer_pattern_and_count,
)
from ltr.visualization import (
    plot_concept_activation_heatmap,
    animate_concept_activation_diagonal,
)


def analyze_concept_distributions(
    model,
    tokenizer,
    prompt: str,
    concepts: List[str],
    output_dir: Optional[str] = None,
    figsize=(14, 8),
    max_new_tokens: int = 250,
    num_layers_to_show: int = 5,
    custom_layer_indices: Optional[List[int]] = None,
) -> Tuple[str, Dict]:
    """
    Analyze distributional semantics of concepts across model layers.

    Parameters:
    -----------
    model : transformers model
        The language model to analyze
    tokenizer : transformers tokenizer
        Tokenizer for the model
    prompt : str
        Input prompt to process
    concepts : List[str]
        List of concepts to track through the model
    output_dir : Optional[str]
        Directory to save visualizations (if None, won't save)
    figsize : tuple
        Default figure size
    max_new_tokens : int
        Maximum number of new tokens to generate

    Returns:
    --------
    Tuple[str, Dict]
        Generated completion and results dictionary with visualizations
    """
    # Generate completion from the model
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Get the generated text
    generated_ids = outputs.sequences[0][inputs.input_ids.shape[1] :]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Full text (prompt + completion)
    full_text = prompt + completion

    # Extract concept activations
    concept_results = extract_concept_activations(
        model, tokenizer, full_text, intermediate_concepts=concepts, final_concepts=[]
    )

    # Create visualizations
    results = {
        "prompt": prompt,
        "completion": completion,
        "full_text": full_text,
        "concept_results": concept_results,
        "visualizations": {},
    }

    # 1. Standard heatmap visualization
    print("1. Heatmap Viz")
    fig_heatmap = plot_concept_activation_heatmap(
        concept_results,
        selected_concepts=concepts,
        figsize_per_concept=(figsize[0], figsize[1] / len(concepts)),
    )
    results["visualizations"]["heatmap"] = fig_heatmap

    # 2. Distributional semantics visualization
    print("2. Distributional Viz")
    fig_dist = visualize_concept_distributional_semantics(
        concept_results,
        concepts,
        figsize=figsize,
        num_layers_to_show=num_layers_to_show,
        custom_layer_indices=custom_layer_indices,
    )
    results["visualizations"]["distributional"] = fig_dist

    # 3. Layer comparison visualization
    print("3. Layer Comparison Viz")
    fig_layers = visualize_concept_layer_comparison(
        concept_results, concepts, figsize=figsize
    )
    results["visualizations"]["layer_comparison"] = fig_layers

    # 4. Semantic connections visualization
    print("4. Semantic Connections Viz")
    if len(concepts) >= 2:
        fig_connections = visualize_concept_semantic_connections(
            concept_results, concepts, figsize=figsize
        )
        results["visualizations"]["semantic_connections"] = fig_connections

    # 5. Create animated visualizations
    print("5. Animated Viz")
    for concept in concepts:
        animation = animate_concept_activation_diagonal(
            concept_results,
            selected_concepts=[concept],
            figsize=(figsize[0], figsize[1] / 2),
        )
        results["visualizations"][f"animation_{concept}"] = animation

    # 6. Network graph visualization
    print("6. Network Graph Viz")
    if len(concepts) >= 2:
        fig_network = visualize_concept_network_graph(
            concept_results,
            concepts,
            figsize=figsize,
            sample_layers=num_layers_to_show,
        )
        results["visualizations"]["network_graph"] = fig_network

    # Save visualizations if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_heatmap.savefig(
            os.path.join(output_dir, "concept_heatmap.png"),
            bbox_inches="tight",
            dpi=300,
        )
        fig_dist.savefig(
            os.path.join(output_dir, "concept_distribution.png"),
            bbox_inches="tight",
            dpi=300,
        )
        fig_layers.savefig(
            os.path.join(output_dir, "layer_comparison.png"),
            bbox_inches="tight",
            dpi=300,
        )
        if len(concepts) >= 2:
            fig_connections.savefig(
                os.path.join(output_dir, "semantic_connections.png"),
                bbox_inches="tight",
                dpi=300,
            )

    return completion, results


def visualize_concept_distributional_semantics(
    concept_results: Dict,
    selected_concepts: Optional[List[str]] = None,
    num_layers_to_show: int = 5,
    custom_layer_indices: Optional[List[int]] = None,
    figsize=(14, 10),
) -> plt.Figure:
    """
    Visualize how concept distributional semantics change across layers.

    Parameters:
    -----------
    concept_results : Dict
        Results from extract_concept_activations
    selected_concepts : Optional[List[str]]
        Specific concepts to visualize
    num_layers_to_show : int
        Number of layers to show (evenly distributed)
    figsize : tuple
        Figure size

    Returns:
    --------
    plt.Figure
        Figure with visualizations
    """
    all_concepts = (
        concept_results["intermediate_concepts"] + concept_results["final_concepts"]
    )
    if selected_concepts is None:
        selected_concepts = all_concepts
    else:
        selected_concepts = [c for c in selected_concepts if c in all_concepts]

    if not selected_concepts:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No valid concepts to display",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        return fig

    # Get token information
    tokens = concept_results["tokens"][1:]  # Skip first token (usually BOS)
    n_tokens = len(tokens)

    # Get layer information
    grid = concept_results["activation_grid"][selected_concepts[0]]
    n_layers = grid.shape[0]

    # Select layers to visualize (evenly distributed)
    if custom_layer_indices is not None:
        # Use custom indices but ensure they're valid
        layer_indices = [idx for idx in custom_layer_indices if 0 <= idx < n_layers]
        # If no valid indices, fall back to default
        if not layer_indices:
            layer_indices = np.linspace(0, n_layers - 1, num_layers_to_show, dtype=int)
    else:
        # Original approach - evenly spaced layers
        if num_layers_to_show > n_layers:
            num_layers_to_show = n_layers
        layer_indices = np.linspace(0, n_layers - 1, num_layers_to_show, dtype=int)

    # Update number of layers to match actual number being displayed
    num_layers_to_show = len(layer_indices)

    layer_indices = np.linspace(0, n_layers - 1, num_layers_to_show, dtype=int)

    # Create figure
    n_concepts = len(selected_concepts)
    fig, axes = plt.subplots(
        num_layers_to_show, 1, figsize=figsize, sharex=True, squeeze=False
    )

    # Create color palette
    concept_colors = sns.color_palette("husl", n_concepts)

    # For each selected layer
    for i, layer_idx in enumerate(layer_indices):
        ax = axes[i, 0]

        # Plot distribution for each concept at this layer
        for j, concept in enumerate(selected_concepts):
            # Get activation at this layer across tokens
            grid = concept_results["activation_grid"][concept]
            activations = grid[layer_idx, :]

            # Plot as distribution
            sns.kdeplot(
                activations,
                ax=ax,
                label=concept,
                color=concept_colors[j],
                fill=True,
                alpha=0.3,
            )

            # Mark top activation positions
            top_k = 3
            top_indices = np.argsort(activations)[-top_k:]
            top_values = activations[top_indices]

            for idx, val in zip(top_indices, top_values):
                if idx < len(tokens):
                    ax.scatter(
                        val,
                        0.01,  # small y-value for visibility on distribution
                        color=concept_colors[j],
                        s=50,
                        alpha=0.8,
                        marker="^",
                        zorder=5,
                    )
                    ax.annotate(
                        tokens[idx],
                        (val, 0.02),
                        fontsize=8,
                        color=concept_colors[j],
                        ha="center",
                        va="bottom",
                        rotation=45,
                    )

        ax.set_ylabel(f"Layer {layer_idx}\nDensity", fontsize=10)

        if i == 0:
            ax.legend(title="Concepts", bbox_to_anchor=(1.01, 1), loc="upper left")

        if i == num_layers_to_show - 1:
            ax.set_xlabel("Activation Strength", fontsize=12)

    plt.suptitle(
        "Distributional Semantics of Concepts Across Layers", fontsize=16, y=0.98
    )
    plt.subplots_adjust(hspace=0.3, right=0.85)

    return fig


import networkx as nx
from matplotlib.cm import get_cmap


def visualize_concept_network_graph(
    concept_results: Dict,
    selected_concepts: Optional[List[str]] = None,
    sample_layers: int = 4,
    correlation_threshold: float = 0.3,
    figsize=(16, 14),
) -> plt.Figure:
    """
    Visualize network graph showing relationships between concepts across layers.

    Parameters:
    -----------
    concept_results : Dict
        Results from extract_concept_activations
    selected_concepts : Optional[List[str]]
        Specific concepts to visualize
    sample_layers : int
        Number of sample layers to visualize
    correlation_threshold : float
        Minimum correlation to show an edge in the graph
    figsize : tuple
        Figure size

    Returns:
    --------
    plt.Figure
        Figure with network graph visualizations
    """
    all_concepts = (
        concept_results["intermediate_concepts"] + concept_results["final_concepts"]
    )
    if selected_concepts is None:
        selected_concepts = all_concepts
    else:
        selected_concepts = [c for c in selected_concepts if c in all_concepts]

    if not selected_concepts or len(selected_concepts) < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "Need at least 2 valid concepts to display network",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        return fig

    # Get grid dimensions
    grid = concept_results["activation_grid"][selected_concepts[0]]
    n_layers, n_tokens = grid.shape

    # Select layers to sample
    if sample_layers > n_layers:
        sample_layers = n_layers

    layer_indices = np.linspace(0, n_layers - 1, sample_layers, dtype=int)

    # Create figure
    fig, axes = plt.subplots(sample_layers, 1, figsize=figsize, squeeze=False)

    # Get tokens
    tokens = concept_results["tokens"][1:]

    # Create colormap for edge weights
    edge_cmap = get_cmap("coolwarm")

    # For each selected layer
    for i, layer_idx in enumerate(layer_indices):
        ax = axes[i, 0]

        # Create graph
        G = nx.Graph()

        # Add nodes (concepts)
        for concept in selected_concepts:
            G.add_node(concept)

        # Get activation vectors for each concept at this layer
        activation_vectors = {}
        for concept in selected_concepts:
            grid = concept_results["activation_grid"][concept]
            activation_vectors[concept] = grid[layer_idx, :]

        # Compute correlations and add edges
        max_corr = 0.1  # For edge width scaling
        for c1_idx, concept1 in enumerate(selected_concepts):
            for c2_idx, concept2 in enumerate(
                selected_concepts[c1_idx + 1 :], c1_idx + 1
            ):
                # Calculate correlation between activation patterns
                corr = np.corrcoef(
                    activation_vectors[concept1], activation_vectors[concept2]
                )[0, 1]

                # Only add edge if correlation is significant
                if abs(corr) >= correlation_threshold:
                    G.add_edge(concept1, concept2, weight=abs(corr), correlation=corr)
                    max_corr = max(max_corr, abs(corr))

        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=1.5, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=1000,
            node_color="lightblue",
            edgecolors="black",
            alpha=0.7,
        )

        # Draw node labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")

        # Draw edges with color based on positive/negative correlation
        # and width based on correlation strength
        for u, v, d in G.edges(data=True):
            corr = d["correlation"]
            # Normalize correlation value to color range (-1 to 1) -> (0 to 1)
            color_val = (corr + 1) / 2
            # Draw edge
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                edgelist=[(u, v)],
                width=5 * abs(corr) / max_corr,
                alpha=0.7,
                edge_color=[edge_cmap(color_val)],
            )

        # Add colorbar for edge colors
        sm = plt.cm.ScalarMappable(cmap=edge_cmap)
        sm.set_array(np.linspace(-1, 1, 100))
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7)
        cbar.set_label("Correlation", fontsize=10)

        ax.set_title(f"Layer {layer_idx}: Concept Relationship Network", fontsize=12)
        ax.axis("off")

    plt.suptitle("Semantic Concept Network Across Layers", fontsize=16, y=0.98)
    plt.subplots_adjust(hspace=0.3)

    return fig


def visualize_concept_layer_comparison(
    concept_results: Dict,
    selected_concepts: Optional[List[str]] = None,
    figsize=(14, 8),
) -> plt.Figure:
    """
    Visualize how concepts are distributed across layers.

    Parameters:
    -----------
    concept_results : Dict
        Results from extract_concept_activations
    selected_concepts : Optional[List[str]]
        Specific concepts to visualize
    figsize : tuple
        Figure size

    Returns:
    --------
    plt.Figure
        Figure with visualizations
    """
    all_concepts = (
        concept_results["intermediate_concepts"] + concept_results["final_concepts"]
    )
    if selected_concepts is None:
        selected_concepts = all_concepts
    else:
        selected_concepts = [c for c in selected_concepts if c in all_concepts]

    if not selected_concepts:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No valid concepts to display",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        return fig

    # Get grid dimensions
    grid = concept_results["activation_grid"][selected_concepts[0]]
    n_layers, n_tokens = grid.shape

    # Create figure
    fig, axes = plt.subplots(
        1, len(selected_concepts), figsize=figsize, sharey=True, squeeze=False
    )

    # For each concept
    for i, concept in enumerate(selected_concepts):
        ax = axes[0, i]

        # Get layer-wise maximum activations
        layer_max = concept_results["layer_max_probs"].get(concept, np.zeros(n_layers))

        # Get layer-wise mean activations
        grid = concept_results["activation_grid"][concept]
        layer_mean = np.mean(grid, axis=1)

        # Plot
        ax.plot(
            layer_max,
            np.arange(n_layers),
            color="darkred",
            label="Max",
            marker="o",
            markersize=4,
            linestyle="-",
            linewidth=2,
            alpha=0.8,
        )
        ax.plot(
            layer_mean,
            np.arange(n_layers),
            color="navy",
            label="Mean",
            marker="s",
            markersize=3,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

        ax.set_title(concept, fontsize=12)
        ax.set_xlabel("Activation Strength", fontsize=10)

        if i == 0:
            ax.set_ylabel("Layer", fontsize=12)
            ax.legend()

        # Invert y-axis to have early layers at the top
        ax.invert_yaxis()

        # Add grid
        ax.grid(alpha=0.3, linestyle=":")

    plt.suptitle("Layer-wise Concept Activation Distribution", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


def visualize_concept_semantic_connections(
    concept_results: Dict,
    selected_concepts: Optional[List[str]] = None,
    sample_layers: int = 4,
    figsize=(16, 12),
) -> plt.Figure:
    """
    Visualize semantic connections between concepts across layers

    Parameters:
    -----------
    concept_results : Dict
        Results from extract_concept_activations
    selected_concepts : Optional[List[str]]
        Specific concepts to visualize
    sample_layers : int
        Number of sample layers to visualize
    figsize : tuple
        Figure size

    Returns:
    --------
    plt.Figure
        Figure with visualizations
    """
    all_concepts = (
        concept_results["intermediate_concepts"] + concept_results["final_concepts"]
    )
    if selected_concepts is None:
        selected_concepts = all_concepts
    else:
        selected_concepts = [c for c in selected_concepts if c in all_concepts]

    if not selected_concepts or len(selected_concepts) < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "Need at least 2 valid concepts to display connections",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        return fig

    # Get grid dimensions
    grid = concept_results["activation_grid"][selected_concepts[0]]
    n_layers, n_tokens = grid.shape

    # Select layers to sample
    if sample_layers > n_layers:
        sample_layers = n_layers

    layer_indices = np.linspace(0, n_layers - 1, sample_layers, dtype=int)

    # Create figure
    fig, axes = plt.subplots(sample_layers, 1, figsize=figsize, squeeze=False)

    # Get tokens
    tokens = concept_results["tokens"][1:]

    # For each selected layer
    for i, layer_idx in enumerate(layer_indices):
        ax = axes[i, 0]

        # Create correlation matrix between concepts
        corr_matrix = np.zeros((len(selected_concepts), len(selected_concepts)))

        # Get activation vectors for each concept at this layer
        activation_vectors = {}
        for c_idx, concept in enumerate(selected_concepts):
            grid = concept_results["activation_grid"][concept]
            activation_vectors[concept] = grid[layer_idx, :]

        # Compute correlations
        for c1_idx, concept1 in enumerate(selected_concepts):
            for c2_idx, concept2 in enumerate(selected_concepts):
                if c1_idx == c2_idx:
                    corr_matrix[c1_idx, c2_idx] = 1.0
                else:
                    # Calculate correlation between activation patterns
                    corr = np.corrcoef(
                        activation_vectors[concept1], activation_vectors[concept2]
                    )[0, 1]
                    corr_matrix[c1_idx, c2_idx] = corr

        # Plot correlation matrix as heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax,
            xticklabels=selected_concepts,
            yticklabels=selected_concepts,
        )

        ax.set_title(f"Layer {layer_idx}: Concept Activation Correlations", fontsize=12)

    plt.suptitle(
        "Semantic Connections Between Concepts Across Layers", fontsize=16, y=0.98
    )
    plt.subplots_adjust(hspace=0.4)

    return fig


def main():
    """Demo function to show example usage"""
    # Load model and tokenizer (you can use any model compatible with extract_concept_activations)
    model_name = "Qwen/Qwen3-0.6B"  # Can use larger models for better results
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Example prompt
    prompt = "John is going fishing, so he walks over to the bank. John realizes that he doesn't have any cash on him. Can he make an ATM transaction at this bank? Answer in yes or no:"
    concepts = ["bank", "river", "money", "finance", "water"]

    n_layers = model.config.num_hidden_layers
    print(f"Model has {n_layers} layers.")

    # Analyze concept distributions
    completion, results = analyze_concept_distributions(
        model,
        tokenizer,
        prompt,
        concepts,
        output_dir="concept_visualizations",
        figsize=(20, 12),
        max_new_tokens=250,
        num_layers_to_show=5,
        custom_layer_indices=range(
            0, n_layers, max(1, n_layers // 14)
        ),  # Show every nth layer
    )

    # Print completion
    print(f"Prompt: {prompt}")
    print(f"Completion: {completion}")

    # Display visualizations
    for viz_name, viz in results["visualizations"].items():
        if not viz_name.startswith("animation"):
            display(viz)

    # Display animations
    for viz_name, viz in results["visualizations"].items():
        if viz_name.startswith("animation"):
            display(viz)


if __name__ == "__main__":
    main()
