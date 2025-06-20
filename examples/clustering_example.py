import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import TraceDict
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import os
import sys
import pandas as pd
import seaborn as sns
from pathlib import Path

# Import LTR components
import ltr
from ltr import (
    extract_concept_activations,
    logit_lens_analysis,
    analyze_model_behavior,
    analyze_factuality,
    perform_causal_intervention,
    SubsequenceAnalyzer,
    analyze_hallucination_subsequences,
    plot_concept_activations,
    plot_logit_lens_heatmap,
)


class HallucinationClusterAnalyzer:
    """
    Advanced analyzer for embedding clustering patterns in hallucinated vs factual content.
    Leverages the LTR library for comprehensive mechanistic analysis.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_model()
        self.hallucination_examples = []
        self.factual_examples = []

        # Initialize LTR subsequence analyzer
        self.subsequence_analyzer = SubsequenceAnalyzer(
            self.model, self.tokenizer, device=self.device
        )

    def setup_model(self):
        """Initialize model and tokenizer with proper configuration"""
        print(f"Loading model {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Get model architecture info
        self.model_config = self.model.config
        self.n_layers = getattr(
            self.model_config,
            "num_hidden_layers",
            getattr(self.model_config, "n_layer", 12),
        )

    def extract_multilayer_representations(
        self, texts: List[str], layer_subset: Optional[List[int]] = None
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Extract representations from multiple layers using LTR's tracing capabilities.

        Args:
            texts: List of text inputs
            layer_subset: Which layers to extract from (None = all layers)

        Returns:
            Dict mapping text to layer representations
        """
        if layer_subset is None:
            layer_subset = list(range(0, self.n_layers, max(1, self.n_layers // 8)))

        # Determine layer pattern based on model architecture
        model_type = getattr(self.model_config, "model_type", "").lower()

        if "llama" in model_type or "qwen" in model_type:
            layer_pattern = "model.layers.{}.input_layernorm"
        elif "gpt2" in model_type:
            layer_pattern = "transformer.h.{}.ln_1"
        elif "gpt-neox" in model_type:
            layer_pattern = "gpt_neox.layers.{}.input_layernorm"
        else:
            layer_pattern = "model.layers.{}.input_layernorm"

        layer_names = [layer_pattern.format(i) for i in layer_subset]

        representations = {}

        print("Extracting multilayer representations...")
        for text in tqdm(texts):
            text_representations = {}

            # Tokenize input
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            # Extract layer representations using baukit
            with torch.no_grad():
                with TraceDict(self.model, layer_names) as traces:
                    _ = self.model(**inputs)

                    for layer_idx, layer_name in zip(layer_subset, layer_names):
                        if layer_name in traces:
                            # Get layer activations
                            activations = traces[layer_name].output
                            if isinstance(activations, tuple):
                                activations = activations[0]

                            # Pool across sequence length (mean of non-padding tokens)
                            attention_mask = inputs.attention_mask
                            masked_activations = activations * attention_mask.unsqueeze(
                                -1
                            )
                            pooled = masked_activations.sum(dim=1) / attention_mask.sum(
                                dim=1, keepdim=True
                            )

                            text_representations[layer_idx] = (
                                pooled.squeeze().cpu().numpy()
                            )

            representations[text] = text_representations

        return representations

    def analyze_hallucination_subsequences(self, examples: List[Dict]) -> Dict:
        """Use LTR's subsequence analysis to identify hallucination-causing patterns"""

        subsequence_results = {}

        print("Analyzing hallucination-causing subsequences...")

        for i, example in enumerate(tqdm(examples[:10])):  # Limit for efficiency
            if example["type"] != "hallucination":
                continue

            prompt = example["prompt"]
            hallucinated_content = example.get("hallucinated_units", [])

            if not hallucinated_content:
                continue

            # Extract hallucinated terms
            for unit_type, content in hallucinated_content:
                try:
                    # Analyze subsequences that lead to this hallucination
                    subseq_analysis = analyze_hallucination_subsequences(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        prompt=prompt,
                        target_string=content,
                        num_perturbations=50,
                        perturbation_rate=0.15,
                    )

                    subsequence_results[f"example_{i}_{content}"] = {
                        "prompt": prompt,
                        "target": content,
                        "analysis": subseq_analysis,
                    }

                except Exception as e:
                    print(f"Error analyzing subsequences for {content}: {e}")
                    continue

        return subsequence_results

    def perform_causal_analysis(self, examples: List[Dict]) -> Dict:
        """Use LTR's causal intervention to understand hallucination mechanisms"""

        causal_results = {}

        print("Performing causal interventions...")

        for i, example in enumerate(tqdm(examples[:5])):  # Limit for efficiency
            if example["type"] != "hallucination":
                continue

            prompt = example["prompt"]
            response = example["response"]

            # Extract key concepts from the response
            words = response.split()
            concepts = [word.strip(".,!?()[]") for word in words if len(word) > 3][:3]

            if not concepts:
                continue

            try:
                # Perform causal intervention analysis
                intervention_results = perform_causal_intervention(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    concepts=concepts,
                )

                causal_results[f"example_{i}"] = {
                    "prompt": prompt,
                    "response": response,
                    "concepts": concepts,
                    "intervention_results": intervention_results,
                }

            except Exception as e:
                print(f"Error in causal analysis for example {i}: {e}")
                continue

        return causal_results

    def analyze_logit_evolution(self, examples: List[Dict]) -> Dict:
        """Analyze how logits evolve differently for hallucinations vs facts"""

        logit_results = {}

        print("Analyzing logit lens evolution...")

        for example_type in ["hallucination", "factual"]:
            type_examples = [ex for ex in examples if ex["type"] == example_type][:5]

            type_logit_results = []

            for example in tqdm(type_examples, desc=f"Processing {example_type}"):
                full_text = example["prompt"] + " " + example["response"]

                try:
                    # Use LTR's logit lens analysis
                    logit_analysis = logit_lens_analysis(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        prompt=full_text,
                        target_layers=list(
                            range(0, self.n_layers, max(1, self.n_layers // 6))
                        ),
                        top_k=5,
                    )

                    type_logit_results.append(logit_analysis)

                except Exception as e:
                    print(f"Error in logit analysis: {e}")
                    continue

            logit_results[example_type] = type_logit_results

        return logit_results

    def cluster_analysis_with_metrics(self, representations: Dict) -> Dict:
        """Enhanced clustering analysis with multiple metrics"""

        clustering_results = {}

        # Extract layer indices
        sample_text = next(iter(representations.keys()))
        layer_indices = list(representations[sample_text].keys())

        for layer_idx in layer_indices:
            print(f"Analyzing clustering for layer {layer_idx}...")

            # Collect embeddings and labels
            embeddings = []
            labels = []
            text_info = []

            for text, layer_reprs in representations.items():
                if layer_idx in layer_reprs:
                    embeddings.append(layer_reprs[layer_idx])

                    # Determine label based on text content or metadata
                    if any(
                        "hallucination" in str(ex)
                        for ex in self.hallucination_examples
                        if ex["prompt"] in text or ex["response"] in text
                    ):
                        labels.append("hallucination")
                    else:
                        labels.append("factual")

                    text_info.append(text[:100])  # Store text snippet

            if len(embeddings) < 4:  # Need minimum samples for clustering
                continue

            embeddings = np.array(embeddings)
            labels = np.array(labels)

            # Dimensionality reduction
            if embeddings.shape[1] > 50:
                pca = PCA(n_components=50, random_state=42)
                reduced_embeddings = pca.fit_transform(embeddings)
                explained_variance = pca.explained_variance_ratio_.sum()
            else:
                reduced_embeddings = embeddings
                explained_variance = 1.0

            # K-means clustering
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(reduced_embeddings)

            # Calculate clustering metrics
            silhouette = silhouette_score(reduced_embeddings, cluster_labels)

            # Calculate purity (how well clusters separate hallucination vs factual)
            hall_mask = labels == "hallucination"
            cluster_0_purity = np.mean(hall_mask[cluster_labels == 0])
            cluster_1_purity = np.mean(hall_mask[cluster_labels == 1])

            # Overall purity score
            purity_score = max(
                abs(cluster_0_purity - 0.5) + abs(cluster_1_purity - 0.5),
                abs((1 - cluster_0_purity) - 0.5) + abs((1 - cluster_1_purity) - 0.5),
            )

            # t-SNE for visualization
            if len(embeddings) > 5:
                perplexity = min(30, len(embeddings) - 1)
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                tsne_embeddings = tsne.fit_transform(reduced_embeddings)
            else:
                tsne_embeddings = reduced_embeddings[:, :2]

            clustering_results[layer_idx] = {
                "embeddings": reduced_embeddings,
                "tsne_embeddings": tsne_embeddings,
                "labels": labels,
                "cluster_labels": cluster_labels,
                "silhouette_score": silhouette,
                "purity_score": purity_score,
                "explained_variance": explained_variance,
                "text_info": text_info,
                "cluster_centers": kmeans.cluster_centers_,
            }

        return clustering_results

    def visualize_comprehensive_analysis(
        self, clustering_results: Dict, logit_results: Dict, output_dir: str
    ):
        """Create comprehensive visualizations of the analysis"""

        os.makedirs(output_dir, exist_ok=True)

        # 1. Clustering evolution across layers
        self.plot_clustering_evolution(clustering_results, output_dir)

        # 2. Detailed metrics comparison
        self.plot_clustering_metrics(clustering_results, output_dir)

        # 3. Logit lens heatmaps
        self.plot_logit_comparisons(logit_results, output_dir)

        # 4. Individual layer detailed plots
        self.plot_detailed_layer_analysis(clustering_results, output_dir)

    def plot_clustering_evolution(self, clustering_results: Dict, output_dir: str):
        """Plot how clustering patterns evolve across layers"""

        layers = sorted(clustering_results.keys())
        n_layers = len(layers)

        # Create subplot grid
        n_cols = min(4, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_layers == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        colors = {"hallucination": "#e74c3c", "factual": "#27ae60"}

        for i, layer_idx in enumerate(layers):
            if i >= len(axes):
                break

            ax = axes[i]
            result = clustering_results[layer_idx]

            # Plot t-SNE embeddings colored by true labels
            for label_type in ["hallucination", "factual"]:
                mask = result["labels"] == label_type
                if np.any(mask):
                    ax.scatter(
                        result["tsne_embeddings"][mask, 0],
                        result["tsne_embeddings"][mask, 1],
                        c=colors[label_type],
                        label=f"{label_type.capitalize()}",
                        alpha=0.7,
                        s=60,
                        edgecolors="black",
                        linewidth=0.5,
                    )

            # Add cluster centers
            ax.scatter(
                result["cluster_centers"][:, 0]
                if result["cluster_centers"].shape[1] >= 2
                else [0, 1],
                result["cluster_centers"][:, 1]
                if result["cluster_centers"].shape[1] >= 2
                else [0, 1],
                marker="x",
                s=200,
                c="black",
                linewidth=3,
                label="Cluster Centers",
            )

            ax.set_title(
                f"Layer {layer_idx}\n"
                f"Purity: {result['purity_score']:.3f}\n"
                f"Silhouette: {result['silhouette_score']:.3f}",
                fontsize=10,
            )
            ax.legend(fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        # Remove empty subplots
        for i in range(len(layers), len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(
            f"Clustering Evolution Across Layers - {self.model_name}",
            fontsize=14,
            y=0.98,
        )
        plt.tight_layout()

        output_path = (
            Path(output_dir)
            / f"{self.model_name.replace('/', '_')}_clustering_evolution.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Clustering evolution saved to {output_path}")
        plt.show()

    def plot_clustering_metrics(self, clustering_results: Dict, output_dir: str):
        """Plot clustering quality metrics across layers"""

        layers = sorted(clustering_results.keys())
        purity_scores = [clustering_results[l]["purity_score"] for l in layers]
        silhouette_scores = [clustering_results[l]["silhouette_score"] for l in layers]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Purity scores
        ax1.plot(
            layers, purity_scores, "o-", linewidth=2, markersize=8, color="#3498db"
        )
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Cluster Purity Score")
        ax1.set_title("Cluster Purity Across Layers")
        ax1.grid(True, alpha=0.3)

        # Highlight best layer
        best_purity_idx = np.argmax(purity_scores)
        ax1.annotate(
            f"Best: Layer {layers[best_purity_idx]}\n{purity_scores[best_purity_idx]:.3f}",
            xy=(layers[best_purity_idx], purity_scores[best_purity_idx]),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
            arrowprops=dict(arrowstyle="->"),
        )

        # Silhouette scores
        ax2.plot(
            layers, silhouette_scores, "o-", linewidth=2, markersize=8, color="#e74c3c"
        )
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Clustering Quality Across Layers")
        ax2.grid(True, alpha=0.3)

        # Highlight best layer
        best_silhouette_idx = np.argmax(silhouette_scores)
        ax2.annotate(
            f"Best: Layer {layers[best_silhouette_idx]}\n{silhouette_scores[best_silhouette_idx]:.3f}",
            xy=(layers[best_silhouette_idx], silhouette_scores[best_silhouette_idx]),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
            arrowprops=dict(arrowstyle="->"),
        )

        plt.tight_layout()

        output_path = (
            Path(output_dir)
            / f"{self.model_name.replace('/', '_')}_clustering_metrics.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Clustering metrics saved to {output_path}")
        plt.show()

    def plot_logit_comparisons(self, logit_results: Dict, output_dir: str):
        """Plot logit lens comparisons between hallucination and factual examples"""

        if not logit_results:
            print("No logit results to plot")
            return

        try:
            # Use LTR's built-in logit lens visualization
            for example_type, type_results in logit_results.items():
                if not type_results:
                    continue

                # Take the first example for visualization
                sample_result = type_results[0]

                fig = plot_logit_lens_heatmap(
                    sample_result,
                    figsize=(12, 8),
                    show_top_tokens=True,
                    top_k_display=3,
                )

                plt.suptitle(
                    f"Logit Lens Analysis - {example_type.capitalize()}", fontsize=14
                )

                output_path = (
                    Path(output_dir)
                    / f"{self.model_name.replace('/', '_')}_logit_lens_{example_type}.png"
                )
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                print(f"Logit lens for {example_type} saved to {output_path}")
                plt.show()

        except Exception as e:
            print(f"Error plotting logit comparisons: {e}")

    def plot_detailed_layer_analysis(self, clustering_results: Dict, output_dir: str):
        """Create detailed analysis plots for the best performing layers"""

        # Find the layers with highest purity scores
        layer_purities = [
            (layer, result["purity_score"])
            for layer, result in clustering_results.items()
        ]
        layer_purities.sort(key=lambda x: x[1], reverse=True)

        # Plot detailed analysis for top 2 layers
        for i, (layer_idx, purity) in enumerate(layer_purities[:2]):
            result = clustering_results[layer_idx]

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

            colors = {"hallucination": "#e74c3c", "factual": "#27ae60"}

            # 1. t-SNE plot with true labels
            for label_type in ["hallucination", "factual"]:
                mask = result["labels"] == label_type
                if np.any(mask):
                    ax1.scatter(
                        result["tsne_embeddings"][mask, 0],
                        result["tsne_embeddings"][mask, 1],
                        c=colors[label_type],
                        label=label_type.capitalize(),
                        alpha=0.7,
                        s=60,
                    )
            ax1.set_title("True Labels")
            ax1.legend()

            # 2. t-SNE plot with cluster labels
            cluster_colors = ["#3498db", "#f39c12"]
            for cluster_id in range(2):
                mask = result["cluster_labels"] == cluster_id
                if np.any(mask):
                    ax2.scatter(
                        result["tsne_embeddings"][mask, 0],
                        result["tsne_embeddings"][mask, 1],
                        c=cluster_colors[cluster_id],
                        label=f"Cluster {cluster_id}",
                        alpha=0.7,
                        s=60,
                    )
            ax2.set_title("K-means Clusters")
            ax2.legend()

            # 3. Confusion matrix style analysis
            confusion_data = np.zeros((2, 2))
            for true_label in ["hallucination", "factual"]:
                for cluster_id in range(2):
                    mask = (result["labels"] == true_label) & (
                        result["cluster_labels"] == cluster_id
                    )
                    confusion_data[
                        0 if true_label == "hallucination" else 1, cluster_id
                    ] = np.sum(mask)

            sns.heatmap(
                confusion_data,
                annot=True,
                fmt="g",
                xticklabels=["Cluster 0", "Cluster 1"],
                yticklabels=["Hallucination", "Factual"],
                ax=ax3,
                cmap="Blues",
            )
            ax3.set_title("Cluster Assignment Matrix")

            # 4. Distance distribution
            from scipy.spatial.distance import cdist

            hall_embeddings = result["embeddings"][result["labels"] == "hallucination"]
            fact_embeddings = result["embeddings"][result["labels"] == "factual"]

            if len(hall_embeddings) > 0 and len(fact_embeddings) > 0:
                # Intra-class distances
                hall_distances = cdist(hall_embeddings, hall_embeddings).flatten()
                fact_distances = cdist(fact_embeddings, fact_embeddings).flatten()

                # Inter-class distances
                inter_distances = cdist(hall_embeddings, fact_embeddings).flatten()

                ax4.hist(
                    hall_distances,
                    alpha=0.5,
                    label="Intra-hallucination",
                    bins=20,
                    color="#e74c3c",
                )
                ax4.hist(
                    fact_distances,
                    alpha=0.5,
                    label="Intra-factual",
                    bins=20,
                    color="#27ae60",
                )
                ax4.hist(
                    inter_distances,
                    alpha=0.5,
                    label="Inter-class",
                    bins=20,
                    color="#3498db",
                )
                ax4.set_xlabel("Distance")
                ax4.set_ylabel("Frequency")
                ax4.set_title("Distance Distributions")
                ax4.legend()

            plt.suptitle(
                f"Detailed Analysis - Layer {layer_idx} (Purity: {purity:.3f})",
                fontsize=14,
            )
            plt.tight_layout()

            output_path = (
                Path(output_dir)
                / f"{self.model_name.replace('/', '_')}_detailed_layer_{layer_idx}.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Detailed layer analysis saved to {output_path}")
            plt.show()

    def create_synthetic_examples(self):
        """Create synthetic examples using better datasets and generation"""

        print("Creating synthetic examples...")

        try:
            # Load factual examples
            factual_df = pd.read_parquet(
                "hf://datasets/hirundo-io/HaluEval-correct-test/data/test-00000-of-00001.parquet"
            ).head(50)  # Reduced for efficiency

            # Load hallucination examples
            hallucination_df = pd.read_parquet(
                "hf://datasets/hirundo-io/HaluEval-hallucinated-test/data/test-00000-of-00001.parquet"
            ).head(50)

            # Process factual examples
            for _, row in tqdm(
                factual_df.iterrows(), desc="Processing factual examples"
            ):
                prompt = row["question"] + " Answer:"
                answer = row["answer"]

                self.factual_examples.append(
                    {
                        "prompt": prompt,
                        "response": answer,
                        "hallucinated_units": [],
                        "type": "factual",
                    }
                )

            # Process hallucination examples
            for _, row in tqdm(
                hallucination_df.iterrows(), desc="Processing hallucination examples"
            ):
                prompt = row["question"] + " Answer:"
                answer = row["answer"]

                # Mark the answer as potentially hallucinated
                self.hallucination_examples.append(
                    {
                        "prompt": prompt,
                        "response": answer,
                        "hallucinated_units": [
                            (
                                "fabricated_fact",
                                answer.split()[-1] if answer.split() else "unknown",
                            )
                        ],
                        "type": "hallucination",
                    }
                )

        except Exception as e:
            print(f"Error loading datasets: {e}")
            print("Creating minimal synthetic examples...")

            # Fallback synthetic examples
            self.factual_examples = [
                {
                    "prompt": "What is the capital of France?",
                    "response": "Paris",
                    "hallucinated_units": [],
                    "type": "factual",
                },
                {
                    "prompt": "What is 2 + 2?",
                    "response": "4",
                    "hallucinated_units": [],
                    "type": "factual",
                },
            ]

            self.hallucination_examples = [
                {
                    "prompt": "What is the capital of Mars?",
                    "response": "New Marsopolis",
                    "hallucinated_units": [("fabricated_place", "Marsopolis")],
                    "type": "hallucination",
                },
                {
                    "prompt": "Who invented the quantum telephone?",
                    "response": "Dr. Einstein invented it in 1955",
                    "hallucinated_units": [
                        ("fabricated_invention", "quantum telephone")
                    ],
                    "type": "hallucination",
                },
            ]

    def run_comprehensive_analysis(
        self, output_dir: str = "ltr_hallucination_analysis"
    ):
        """Run the complete LTR-powered hallucination analysis"""

        print("Starting comprehensive hallucination analysis with LTR...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load or create examples
        if not self.hallucination_examples and not self.factual_examples:
            self.create_synthetic_examples()

        all_examples = self.hallucination_examples + self.factual_examples
        print(
            f"Analyzing {len(self.hallucination_examples)} hallucination and {len(self.factual_examples)} factual examples"
        )

        # Extract all text for representation analysis
        all_texts = [ex["prompt"] + " " + ex["response"] for ex in all_examples]

        # 1. Extract multilayer representations
        print("\n=== Phase 1: Extracting multilayer representations ===")
        representations = self.extract_multilayer_representations(all_texts)

        # 2. Clustering analysis
        print("\n=== Phase 2: Clustering analysis ===")
        clustering_results = self.cluster_analysis_with_metrics(representations)

        # 3. Logit lens analysis
        print("\n=== Phase 3: Logit lens analysis ===")
        logit_results = self.analyze_logit_evolution(all_examples)

        # 4. Subsequence analysis
        print("\n=== Phase 4: Subsequence analysis ===")
        subsequence_results = self.analyze_hallucination_subsequences(all_examples)

        # 5. Causal intervention analysis
        print("\n=== Phase 5: Causal intervention analysis ===")
        causal_results = self.perform_causal_analysis(all_examples)

        # 6. Comprehensive visualization
        print("\n=== Phase 6: Creating visualizations ===")
        self.visualize_comprehensive_analysis(
            clustering_results, logit_results, output_dir
        )

        # 7. Save detailed results
        results_summary = {
            "model_name": self.model_name,
            "n_hallucination_examples": len(self.hallucination_examples),
            "n_factual_examples": len(self.factual_examples),
            "clustering_summary": {
                layer: {
                    "purity_score": result["purity_score"],
                    "silhouette_score": result["silhouette_score"],
                }
                for layer, result in clustering_results.items()
            },
            "best_separation_layer": max(
                clustering_results.keys(),
                key=lambda l: clustering_results[l]["purity_score"],
            ),
            "subsequence_analysis_count": len(subsequence_results),
            "causal_analysis_count": len(causal_results),
        }

        # Save summary
        import json

        with open(Path(output_dir) / "analysis_summary.json", "w") as f:
            json.dump(results_summary, f, indent=2)

        print(f"\n=== Analysis Complete ===")
        print(f"Results saved to: {output_dir}")
        print(f"Best separation at layer: {results_summary['best_separation_layer']}")
        print(
            f"Best purity score: {clustering_results[results_summary['best_separation_layer']]['purity_score']:.3f}"
        )

        return {
            "clustering_results": clustering_results,
            "logit_results": logit_results,
            "subsequence_results": subsequence_results,
            "causal_results": causal_results,
            "summary": results_summary,
        }


def main():
    """Main function to run the enhanced hallucination clustering analysis"""

    # Initialize analyzer with a smaller model for testing
    analyzer = HallucinationClusterAnalyzer(model_name="Qwen/Qwen2.5-0.5B-Instruct")

    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(
        output_dir="ltr_hallucination_analysis"
    )

    print("\nAnalysis completed successfully!")
    return results


if __name__ == "__main__":
    main()
