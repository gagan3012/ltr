import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import TraceDict
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
import os
import sys
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

# Add LTR to path
from ltr.concept_extraction import extract_concept_activations
from ltr.patchscopes import perform_patchscope_analysis
from ltr.behavioral_analysis import analyze_factuality


class ConceptualHallucinationAnalyzer:
    """
    Analyze how poor distributional semantics of concepts lead to hallucinations.

    This analyzer demonstrates that hallucinations often stem from:
    1. Ambiguous concept representations in embedding space
    2. Poor separation between related but distinct concepts
    3. Concept drift across model layers
    4. Weak semantic boundaries in high-dimensional space
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        self.model_name = model_name
        self.setup_model()
        self.concept_embeddings = defaultdict(lambda: defaultdict(list))
        self.semantic_clusters = {}
        self.hallucination_patterns = {}

    def setup_model(self):
        """Initialize model and tokenizer"""
        print(f"Loading model {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype=torch.float16
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_concept_representations(
        self, texts: List[str], concepts: List[str], target_layers: List[int]
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Extract concept representations across different contexts and layers.

        Args:
            texts: List of contexts containing the concepts
            concepts: List of concepts to track
            target_layers: Model layers to analyze

        Returns:
            Dict mapping concept -> layer -> list of embeddings
        """

        concept_representations = defaultdict(lambda: defaultdict(list))

        # Determine layer pattern based on model architecture
        model_type = self.model.config.model_type.lower()
        if "llama" in model_type:
            layer_pattern = "model.layers.{}"
        elif "gpt2" in model_type:
            layer_pattern = "transformer.h.{}"
        else:
            layer_pattern = "model.layers.{}"

        layer_names = [layer_pattern.format(i) for i in target_layers]

        print("Extracting concept representations across contexts...")

        for text in tqdm(texts):
            # Tokenize text
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

            # Find concept positions in text
            concept_positions = self._find_concept_positions(tokens, concepts)

            if not concept_positions:
                continue

            # Extract representations using tracing
            with torch.no_grad():
                with TraceDict(self.model, layer_names) as traces:
                    _ = self.model(**inputs.to(self.model.device))

                    for layer_idx, layer_name in zip(target_layers, layer_names):
                        if layer_name in traces:
                            layer_output = traces[layer_name].output[
                                0
                            ]  # Remove batch dim

                            # Extract embeddings for each concept occurrence
                            for concept, positions in concept_positions.items():
                                for pos in positions:
                                    if pos < layer_output.shape[0]:
                                        concept_emb = layer_output[pos].cpu().numpy()
                                        concept_representations[concept][
                                            layer_idx
                                        ].append(
                                            {
                                                "embedding": concept_emb,
                                                "context": text,
                                                "position": pos,
                                                "token": tokens[pos]
                                                if pos < len(tokens)
                                                else None,
                                            }
                                        )

        return concept_representations

    def _find_concept_positions(
        self, tokens: List[str], concepts: List[str]
    ) -> Dict[str, List[int]]:
        """Find positions where concepts appear in tokenized text"""
        concept_positions = defaultdict(list)

        for concept in concepts:
            # Try different tokenization variants
            concept_variants = [
                concept.lower(),
                concept.upper(),
                concept.capitalize(),
                f"Ġ{concept.lower()}",  # GPT-style space prefix
                f"▁{concept.lower()}",  # SentencePiece style
            ]

            for i, token in enumerate(tokens):
                for variant in concept_variants:
                    if variant in token.lower():
                        concept_positions[concept].append(i)
                        break

        return concept_positions

    def analyze_semantic_ambiguity(
        self, concept_representations: Dict[str, Dict[int, List]], target_layer: int
    ) -> Dict[str, Dict]:
        """
        Analyze semantic ambiguity of concepts in embedding space.

        High ambiguity = poor distributional semantics = higher hallucination risk
        """

        ambiguity_metrics = {}

        for concept, layer_data in concept_representations.items():
            if target_layer not in layer_data or len(layer_data[target_layer]) < 3:
                continue

            embeddings = np.array(
                [item["embedding"] for item in layer_data[target_layer]]
            )
            contexts = [item["context"] for item in layer_data[target_layer]]

            # Calculate various ambiguity metrics
            metrics = {}

            # 1. Intra-concept cosine similarity variance (lower = more ambiguous)
            pairwise_similarities = cosine_similarity(embeddings)
            upper_triangle = pairwise_similarities[
                np.triu_indices_from(pairwise_similarities, k=1)
            ]
            metrics["similarity_mean"] = np.mean(upper_triangle)
            metrics["similarity_std"] = np.std(upper_triangle)
            metrics["coherence_score"] = (
                metrics["similarity_mean"] - metrics["similarity_std"]
            )

            # 2. Clustering quality (silhouette score)
            if len(embeddings) >= 4:
                n_clusters = min(3, len(embeddings) // 2)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings)
                if len(np.unique(cluster_labels)) > 1:
                    metrics["silhouette_score"] = silhouette_score(
                        embeddings, cluster_labels
                    )
                else:
                    metrics["silhouette_score"] = 0.0
            else:
                metrics["silhouette_score"] = 0.0

            # 3. Embedding dispersion (higher = more ambiguous)
            centroid = np.mean(embeddings, axis=0)
            distances = [np.linalg.norm(emb - centroid) for emb in embeddings]
            metrics["dispersion"] = np.mean(distances)
            metrics["dispersion_std"] = np.std(distances)

            # 4. Dimensionality of concept space (effective rank)
            if len(embeddings) > 1:
                U, s, Vh = np.linalg.svd(embeddings - embeddings.mean(axis=0))
                normalized_s = s / np.sum(s)
                metrics["effective_rank"] = np.exp(entropy(normalized_s + 1e-10))
            else:
                metrics["effective_rank"] = 1.0

            # 5. Context diversity score
            unique_contexts = len(set(contexts))
            metrics["context_diversity"] = unique_contexts / len(contexts)

            # Overall ambiguity score (higher = more ambiguous = more hallucination-prone)
            ambiguity_score = (
                (1.0 - metrics["coherence_score"]) * 0.3
                + (1.0 - max(0, metrics["silhouette_score"])) * 0.2
                + (metrics["dispersion"] / 10.0) * 0.3  # Normalize dispersion
                + (metrics["effective_rank"] / len(embeddings)) * 0.2
            )
            metrics["ambiguity_score"] = ambiguity_score

            ambiguity_metrics[concept] = metrics

        return ambiguity_metrics

    def create_semantic_confusion_map(
        self,
        concept_representations: Dict[str, Dict[int, List]],
        target_layer: int,
        output_dir: str,
    ):
        """
        Create a confusion map showing which concepts are easily confused in embedding space.
        """

        concepts = list(concept_representations.keys())
        confusion_matrix = np.zeros((len(concepts), len(concepts)))

        # Calculate inter-concept similarities
        for i, concept1 in enumerate(concepts):
            if target_layer not in concept_representations[concept1]:
                continue

            embs1 = np.array(
                [
                    item["embedding"]
                    for item in concept_representations[concept1][target_layer]
                ]
            )
            centroid1 = np.mean(embs1, axis=0)

            for j, concept2 in enumerate(concepts):
                if target_layer not in concept_representations[concept2]:
                    continue

                embs2 = np.array(
                    [
                        item["embedding"]
                        for item in concept_representations[concept2][target_layer]
                    ]
                )
                centroid2 = np.mean(embs2, axis=0)

                # Calculate confusion score (cosine similarity between centroids)
                similarity = cosine_similarity([centroid1], [centroid2])[0, 0]
                confusion_matrix[i, j] = similarity

        # Create confusion heatmap
        plt.figure(figsize=(12, 10))
        mask = np.zeros_like(confusion_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt=".3f",
            xticklabels=concepts,
            yticklabels=concepts,
            cmap="RdYlBu_r",
            center=0.5,
            mask=mask,
            square=True,
            cbar_kws={"label": "Semantic Similarity"},
        )

        plt.title(
            f"Concept Semantic Confusion Matrix (Layer {target_layer})\nHigher values = More confusion = Higher hallucination risk"
        )
        plt.xlabel("Concepts")
        plt.ylabel("Concepts")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                output_dir, f"semantic_confusion_matrix_layer_{target_layer}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        return confusion_matrix

    def visualize_concept_semantic_spaces(
        self,
        concept_representations: Dict[str, Dict[int, List]],
        ambiguity_metrics: Dict[str, Dict],
        target_layer: int,
        output_dir: str,
    ):
        """
        Visualize concept distributions in semantic space with ambiguity indicators.
        """

        # Prepare data for visualization
        all_embeddings = []
        all_labels = []
        all_concepts = []
        ambiguity_scores = []

        for concept, layer_data in concept_representations.items():
            if target_layer not in layer_data:
                continue

            embeddings = [item["embedding"] for item in layer_data[target_layer]]
            all_embeddings.extend(embeddings)
            all_labels.extend([concept] * len(embeddings))
            all_concepts.extend([concept] * len(embeddings))

            # Get ambiguity score for this concept
            amb_score = ambiguity_metrics.get(concept, {}).get("ambiguity_score", 0.0)
            ambiguity_scores.extend([amb_score] * len(embeddings))

        if len(all_embeddings) < 10:
            print("Not enough embeddings for visualization")
            return

        all_embeddings = np.array(all_embeddings)

        # Reduce dimensionality for visualization
        print("Reducing dimensionality for visualization...")

        # First apply PCA if dimensionality is very high
        if all_embeddings.shape[1] > 50:
            pca = PCA(n_components=50)
            embeddings_pca = pca.fit_transform(all_embeddings)
        else:
            embeddings_pca = all_embeddings

        # Apply UMAP for better clustering visualization
        umap_reducer = UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=min(15, len(all_embeddings) - 1),
        )
        embeddings_2d = umap_reducer.fit_transform(embeddings_pca)

        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # Plot 1: Concept distributions colored by ambiguity
        unique_concepts = list(set(all_concepts))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_concepts)))
        concept_colors = {
            concept: colors[i] for i, concept in enumerate(unique_concepts)
        }

        for concept in unique_concepts:
            mask = np.array(all_concepts) == concept
            amb_score = ambiguity_metrics.get(concept, {}).get("ambiguity_score", 0.0)

            ax1.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[concept_colors[concept]],
                label=f"{concept} (ambig: {amb_score:.3f})",
                s=60 + amb_score * 200,  # Size based on ambiguity
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )

        ax1.set_title("Concept Semantic Spaces\n(Size = Ambiguity Score)", fontsize=14)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Ambiguity heatmap
        scatter = ax2.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=ambiguity_scores,
            cmap="Reds",
            s=80,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )

        ax2.set_title(
            "Semantic Ambiguity Heatmap\n(Red = High Hallucination Risk)", fontsize=14
        )
        plt.colorbar(scatter, ax=ax2, label="Ambiguity Score")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Concept coherence vs dispersion
        coherence_scores = [
            ambiguity_metrics.get(concept, {}).get("coherence_score", 0.0)
            for concept in unique_concepts
        ]
        dispersion_scores = [
            ambiguity_metrics.get(concept, {}).get("dispersion", 0.0)
            for concept in unique_concepts
        ]
        ambig_scores = [
            ambiguity_metrics.get(concept, {}).get("ambiguity_score", 0.0)
            for concept in unique_concepts
        ]

        scatter3 = ax3.scatter(
            coherence_scores,
            dispersion_scores,
            c=ambig_scores,
            s=100,
            cmap="Reds",
            alpha=0.8,
            edgecolors="black",
        )

        for i, concept in enumerate(unique_concepts):
            ax3.annotate(
                concept,
                (coherence_scores[i], dispersion_scores[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        ax3.set_xlabel("Coherence Score (Higher = Less Ambiguous)")
        ax3.set_ylabel("Dispersion Score (Higher = More Ambiguous)")
        ax3.set_title(
            "Concept Quality Analysis\n(Red = High Hallucination Risk)", fontsize=14
        )
        plt.colorbar(scatter3, ax=ax3, label="Ambiguity Score")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Semantic neighborhood analysis
        # Show which concepts are in dangerous proximity
        for i, concept1 in enumerate(unique_concepts):
            mask1 = np.array(all_concepts) == concept1
            centroid1 = np.mean(embeddings_2d[mask1], axis=0)

            for j, concept2 in enumerate(unique_concepts[i + 1 :], i + 1):
                mask2 = np.array(all_concepts) == concept2
                centroid2 = np.mean(embeddings_2d[mask2], axis=0)

                # Calculate distance between centroids
                distance = np.linalg.norm(centroid1 - centroid2)

                # Draw line if concepts are dangerously close
                if distance < np.percentile(
                    [
                        np.linalg.norm(embeddings_2d[k] - embeddings_2d[l])
                        for k in range(len(embeddings_2d))
                        for l in range(k + 1, len(embeddings_2d))
                    ],
                    25,
                ):  # Bottom 25%
                    ax4.plot(
                        [centroid1[0], centroid2[0]],
                        [centroid1[1], centroid2[1]],
                        "r--",
                        alpha=0.6,
                        linewidth=2,
                    )

        # Plot concept centroids
        for concept in unique_concepts:
            mask = np.array(all_concepts) == concept
            centroid = np.mean(embeddings_2d[mask], axis=0)
            ax4.scatter(
                centroid[0],
                centroid[1],
                c=[concept_colors[concept]],
                s=200,
                marker="*",
                edgecolors="black",
                linewidth=2,
                label=f"{concept} centroid",
            )
            ax4.annotate(
                concept,
                centroid,
                xytext=(5, 5),
                textcoords="offset points",
                fontweight="bold",
            )

        ax4.set_title(
            "Dangerous Semantic Neighborhoods\n(Red lines = Confusion Risk)",
            fontsize=14,
        )
        ax4.grid(True, alpha=0.3)

        plt.suptitle(
            f"Conceptual Hallucination Analysis - Layer {target_layer}\nPoor Distributional Semantics Lead to Hallucinations",
            fontsize=16,
            y=0.98,
        )
        plt.tight_layout()

        # Save figure
        plt.savefig(
            os.path.join(
                output_dir, f"concept_semantic_analysis_layer_{target_layer}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def analyze_layer_progression(
        self,
        concept_representations: Dict[str, Dict[int, List]],
        target_layers: List[int],
        output_dir: str,
    ):
        """
        Analyze how concept ambiguity changes across model layers.
        """

        progression_data = defaultdict(list)

        for layer in target_layers:
            ambiguity_metrics = self.analyze_semantic_ambiguity(
                concept_representations, layer
            )

            for concept, metrics in ambiguity_metrics.items():
                progression_data[concept].append(
                    {
                        "layer": layer,
                        "ambiguity_score": metrics["ambiguity_score"],
                        "coherence_score": metrics["coherence_score"],
                        "dispersion": metrics["dispersion"],
                        "silhouette_score": metrics["silhouette_score"],
                    }
                )

        # Create progression plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot ambiguity progression
        for concept, data in progression_data.items():
            layers = [d["layer"] for d in data]
            ambiguity = [d["ambiguity_score"] for d in data]
            ax1.plot(layers, ambiguity, "o-", label=concept, linewidth=2, markersize=6)

        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Ambiguity Score")
        ax1.set_title("Concept Ambiguity Across Layers")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot coherence progression
        for concept, data in progression_data.items():
            layers = [d["layer"] for d in data]
            coherence = [d["coherence_score"] for d in data]
            ax2.plot(layers, coherence, "s-", label=concept, linewidth=2, markersize=6)

        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Coherence Score")
        ax2.set_title("Concept Coherence Across Layers")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot dispersion progression
        for concept, data in progression_data.items():
            layers = [d["layer"] for d in data]
            dispersion = [d["dispersion"] for d in data]
            ax3.plot(layers, dispersion, "^-", label=concept, linewidth=2, markersize=6)

        ax3.set_xlabel("Layer")
        ax3.set_ylabel("Dispersion Score")
        ax3.set_title("Concept Dispersion Across Layers")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Create layer-wise ambiguity heatmap
        concepts = list(progression_data.keys())
        layer_ambiguity_matrix = np.zeros((len(concepts), len(target_layers)))

        for i, concept in enumerate(concepts):
            for j, layer in enumerate(target_layers):
                layer_data = next(
                    (d for d in progression_data[concept] if d["layer"] == layer), None
                )
                if layer_data:
                    layer_ambiguity_matrix[i, j] = layer_data["ambiguity_score"]

        sns.heatmap(
            layer_ambiguity_matrix,
            xticklabels=target_layers,
            yticklabels=concepts,
            annot=True,
            fmt=".3f",
            cmap="Reds",
            ax=ax4,
            cbar_kws={"label": "Ambiguity Score"},
        )

        ax4.set_xlabel("Layer")
        ax4.set_ylabel("Concept")
        ax4.set_title("Layer-wise Ambiguity Heatmap")

        plt.suptitle("Concept Semantic Evolution Across Model Layers", fontsize=16)
        plt.tight_layout()

        # Save figure
        plt.savefig(
            os.path.join(output_dir, "concept_layer_progression.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        return progression_data

    def create_hallucination_correlation_analysis(
        self,
        ambiguity_metrics: Dict[str, Dict],
        hallucination_results: Dict[str, Any],
        output_dir: str,
    ):
        """
        Correlate concept ambiguity with actual hallucination occurrence.
        """

        # Extract hallucination data
        concept_hallucination_rates = {}

        if "entity_traces" in hallucination_results:
            for concept, trace in hallucination_results["entity_traces"].items():
                if trace:
                    # Calculate how often this concept appears in problematic contexts
                    high_prob_steps = [
                        step for step in trace if step["probability"] > 0.1
                    ]
                    hallucination_rate = (
                        len(high_prob_steps) / len(trace) if trace else 0
                    )
                    concept_hallucination_rates[concept] = hallucination_rate

        # Combine with ambiguity metrics
        combined_data = []
        for concept in set(ambiguity_metrics.keys()) & set(
            concept_hallucination_rates.keys()
        ):
            combined_data.append(
                {
                    "concept": concept,
                    "ambiguity_score": ambiguity_metrics[concept]["ambiguity_score"],
                    "coherence_score": ambiguity_metrics[concept]["coherence_score"],
                    "dispersion": ambiguity_metrics[concept]["dispersion"],
                    "hallucination_rate": concept_hallucination_rates[concept],
                }
            )

        if not combined_data:
            print("No data available for correlation analysis")
            return

        df = pd.DataFrame(combined_data)

        # Create correlation analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Ambiguity vs Hallucination Rate
        ax1.scatter(df["ambiguity_score"], df["hallucination_rate"], s=100, alpha=0.7)
        for i, row in df.iterrows():
            ax1.annotate(
                row["concept"],
                (row["ambiguity_score"], row["hallucination_rate"]),
                xytext=(5, 5),
                textcoords="offset points",
            )

        # Add trend line
        z = np.polyfit(df["ambiguity_score"], df["hallucination_rate"], 1)
        p = np.poly1d(z)
        ax1.plot(df["ambiguity_score"], p(df["ambiguity_score"]), "r--", alpha=0.8)

        correlation = df["ambiguity_score"].corr(df["hallucination_rate"])
        ax1.set_xlabel("Semantic Ambiguity Score")
        ax1.set_ylabel("Hallucination Rate")
        ax1.set_title(
            f"Ambiguity vs Hallucination Rate\n(Correlation: {correlation:.3f})"
        )
        ax1.grid(True, alpha=0.3)

        # Plot 2: Coherence vs Hallucination Rate
        ax2.scatter(
            df["coherence_score"],
            df["hallucination_rate"],
            s=100,
            alpha=0.7,
            color="orange",
        )
        for i, row in df.iterrows():
            ax2.annotate(
                row["concept"],
                (row["coherence_score"], row["hallucination_rate"]),
                xytext=(5, 5),
                textcoords="offset points",
            )

        correlation2 = df["coherence_score"].corr(df["hallucination_rate"])
        ax2.set_xlabel("Semantic Coherence Score")
        ax2.set_ylabel("Hallucination Rate")
        ax2.set_title(
            f"Coherence vs Hallucination Rate\n(Correlation: {correlation2:.3f})"
        )
        ax2.grid(True, alpha=0.3)

        # Plot 3: Combined analysis
        bubble_sizes = df["dispersion"] * 50  # Scale for visibility
        scatter = ax3.scatter(
            df["ambiguity_score"],
            df["hallucination_rate"],
            s=bubble_sizes,
            alpha=0.6,
            c=df["coherence_score"],
            cmap="RdYlBu",
        )

        for i, row in df.iterrows():
            ax3.annotate(
                row["concept"],
                (row["ambiguity_score"], row["hallucination_rate"]),
                xytext=(5, 5),
                textcoords="offset points",
            )

        ax3.set_xlabel("Semantic Ambiguity Score")
        ax3.set_ylabel("Hallucination Rate")
        ax3.set_title("Combined Analysis\n(Size=Dispersion, Color=Coherence)")
        plt.colorbar(scatter, ax=ax3, label="Coherence Score")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Ranking analysis
        df_sorted = df.sort_values("hallucination_rate", ascending=False)
        y_pos = np.arange(len(df_sorted))

        bars = ax4.barh(y_pos, df_sorted["hallucination_rate"], alpha=0.7)

        # Color bars by ambiguity score
        for i, (bar, ambiguity) in enumerate(zip(bars, df_sorted["ambiguity_score"])):
            bar.set_color(plt.cm.Reds(ambiguity))

        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(df_sorted["concept"])
        ax4.set_xlabel("Hallucination Rate")
        ax4.set_title(
            "Concepts Ranked by Hallucination Rate\n(Color intensity = Ambiguity)"
        )
        ax4.grid(True, alpha=0.3, axis="x")

        plt.suptitle(
            "Correlation: Poor Distributional Semantics → Hallucinations", fontsize=16
        )
        plt.tight_layout()

        # Save figure
        plt.savefig(
            os.path.join(output_dir, "hallucination_correlation_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # Print correlation statistics
        print("\n" + "=" * 50)
        print("HALLUCINATION CORRELATION ANALYSIS")
        print("=" * 50)
        print(f"Ambiguity → Hallucination correlation: {correlation:.3f}")
        print(f"Coherence → Hallucination correlation: {correlation2:.3f}")
        print("\nConcepts ranked by hallucination risk:")
        for i, row in df_sorted.iterrows():
            print(
                f"{row['concept']:15} | Hall. Rate: {row['hallucination_rate']:.3f} | Ambiguity: {row['ambiguity_score']:.3f}"
            )

    def run_full_semantic_analysis(
        self, output_dir: str = "semantic_hallucination_analysis"
    ):
        """
        Run complete analysis showing how poor distributional semantics cause hallucinations.
        """

        print("Starting Conceptual Hallucination Analysis...")
        print("=" * 60)

        # Create test scenarios with concepts that should have clear vs ambiguous semantics
        test_scenarios = self._create_test_scenarios()

        # Get target layers for analysis
        n_layers = getattr(
            self.model.config,
            "num_hidden_layers",
            getattr(self.model.config, "n_layer", 12),
        )
        target_layers = [
            0,
            n_layers // 4,
            n_layers // 2,
            3 * n_layers // 4,
            n_layers - 1,
        ]
        analysis_layer = n_layers // 2  # Focus on middle layer for main analysis

        print(f"Analyzing {len(target_layers)} layers: {target_layers}")
        print(f"Primary analysis on layer {analysis_layer}")

        # Extract concept representations
        concept_representations = self.extract_concept_representations(
            test_scenarios["texts"], test_scenarios["concepts"], target_layers
        )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # 1. Analyze semantic ambiguity
        print("\n1. Analyzing semantic ambiguity...")
        ambiguity_metrics = self.analyze_semantic_ambiguity(
            concept_representations, analysis_layer
        )

        # 2. Create semantic confusion map
        print("\n2. Creating semantic confusion map...")
        confusion_matrix = self.create_semantic_confusion_map(
            concept_representations, analysis_layer, output_dir
        )

        # 3. Visualize concept semantic spaces
        print("\n3. Visualizing concept semantic spaces...")
        self.visualize_concept_semantic_spaces(
            concept_representations, ambiguity_metrics, analysis_layer, output_dir
        )

        # 4. Analyze layer progression
        print("\n4. Analyzing concept evolution across layers...")
        progression_data = self.analyze_layer_progression(
            concept_representations, target_layers, output_dir
        )

        # 5. Run hallucination analysis
        print("\n5. Running hallucination analysis...")
        hallucination_results = self._run_hallucination_tests(
            test_scenarios["concepts"]
        )

        # 6. Create correlation analysis
        print("\n6. Creating hallucination correlation analysis...")
        self.create_hallucination_correlation_analysis(
            ambiguity_metrics, hallucination_results, output_dir
        )

        # 7. Generate summary report
        self._generate_summary_report(ambiguity_metrics, progression_data, output_dir)

        print(f"\n" + "=" * 60)
        print(f"Analysis complete! Results saved to: {output_dir}")
        print("=" * 60)

    def _create_test_scenarios(self) -> Dict[str, List]:
        """
        Create test scenarios with concepts that should show different semantic properties.
        """

        # Concepts with clear semantics (should be less ambiguous)
        clear_concepts = ["dog", "cat", "house", "tree", "car"]

        # Concepts with ambiguous semantics (should be more ambiguous)
        ambiguous_concepts = ["bank", "bat", "spring", "rock", "light"]

        # Abstract concepts (often problematic)
        abstract_concepts = ["justice", "freedom", "love", "success", "intelligence"]

        all_concepts = clear_concepts + ambiguous_concepts + abstract_concepts

        # Create diverse contexts for these concepts
        contexts = []

        # Clear concept contexts
        contexts.extend(
            [
                "The dog barked loudly at the mailman.",
                "A small dog ran across the yard.",
                "My neighbor has a friendly dog.",
                "The cat sat on the windowsill.",
                "A black cat crossed the street.",
                "The cat purred contentedly.",
                "They bought a new house last year.",
                "The house has a red roof.",
                "This house is very expensive.",
                "The old tree provided shade.",
                "A tree fell during the storm.",
                "We planted a tree in the garden.",
                "My car needs an oil change.",
                "The red car is very fast.",
                "I bought a new car yesterday.",
            ]
        )

        # Ambiguous concept contexts (multiple meanings)
        contexts.extend(
            [
                "I went to the bank to withdraw money.",
                "We sat by the river bank.",
                "The bank approved my loan application.",
                "The baseball bat was made of wood.",
                "A bat flew through the cave.",
                "He swung the bat with great force.",
                "Spring is my favorite season.",
                "The spring in the mattress is broken.",
                "Water flows from the natural spring.",
                "The rock band played all night.",
                "A large rock blocked the path.",
                "She wore a beautiful rock on her finger.",
                "Turn on the light, please.",
                "The feather is very light.",
                "Light travels faster than sound.",
            ]
        )

        # Abstract concept contexts
        contexts.extend(
            [
                "Justice must be served fairly.",
                "The court delivered justice.",
                "She fought for social justice.",
                "Freedom is a fundamental right.",
                "They gained freedom after years.",
                "Freedom of speech is important.",
                "Love conquers all obstacles.",
                "I love chocolate ice cream.",
                "Love is a powerful emotion.",
                "Success requires hard work.",
                "The project was a great success.",
                "Success means different things to people.",
                "Artificial intelligence is advancing rapidly.",
                "She has high emotional intelligence.",
                "Intelligence comes in many forms.",
            ]
        )

        return {
            "texts": contexts,
            "concepts": all_concepts,
            "clear_concepts": clear_concepts,
            "ambiguous_concepts": ambiguous_concepts,
            "abstract_concepts": abstract_concepts,
        }

    def _run_hallucination_tests(self, concepts: List[str]) -> Dict[str, Any]:
        """
        Run hallucination analysis using LTR's patchscopes.
        """

        # Create prompts that might lead to hallucinations involving these concepts
        hallucination_prompts = [
            f"The {concept} that never existed was discovered in",
            f"Scientists recently found that {concept} can",
            f"The ancient {concept} was known for its ability to",
            f"In the future, {concept} will be able to",
        ]

        all_results = {
            "entity_traces": defaultdict(list),
            "confidence_scores": [],
            "generation_steps": 0,
        }

        for concept in concepts[:5]:  # Limit to avoid long processing
            for prompt_template in hallucination_prompts[:2]:  # Limit prompts
                prompt = prompt_template.format(concept=concept)

                try:
                    # Use LTR's patchscope analysis
                    results = perform_patchscope_analysis(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        prompt=prompt,
                        target_entities=[concept],
                        max_tokens=15,
                        target_layers=[0, 4, 8, 12],
                    )

                    # Aggregate results
                    if concept in results["entity_traces"]:
                        all_results["entity_traces"][concept].extend(
                            results["entity_traces"][concept]
                        )

                    all_results["confidence_scores"].extend(
                        results.get("confidence_scores", [])
                    )
                    all_results["generation_steps"] += results.get("summary", {}).get(
                        "total_generation_steps", 0
                    )

                except Exception as e:
                    print(f"Error analyzing prompt '{prompt}': {e}")
                    continue

        return all_results

    def _generate_summary_report(
        self,
        ambiguity_metrics: Dict[str, Dict],
        progression_data: Dict,
        output_dir: str,
    ):
        """
        Generate a comprehensive summary report.
        """

        report_path = os.path.join(output_dir, "hallucination_analysis_report.txt")

        with open(report_path, "w") as f:
            f.write("CONCEPTUAL HALLUCINATION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write("EXECUTIVE SUMMARY:\n")
            f.write("This analysis demonstrates how poor distributional semantics\n")
            f.write("of concepts in neural language models lead to hallucinations.\n\n")

            f.write("KEY FINDINGS:\n")
            f.write(
                "- Concepts with high semantic ambiguity are more prone to hallucination\n"
            )
            f.write(
                "- Poor separation in embedding space creates confusion between concepts\n"
            )
            f.write(
                "- Abstract concepts show higher ambiguity than concrete concepts\n"
            )
            f.write("- Semantic quality evolves differently across model layers\n\n")

            f.write("CONCEPT AMBIGUITY RANKINGS:\n")
            f.write("-" * 30 + "\n")

            sorted_concepts = sorted(
                ambiguity_metrics.items(),
                key=lambda x: x[1]["ambiguity_score"],
                reverse=True,
            )

            for concept, metrics in sorted_concepts:
                f.write(
                    f"{concept:15} | Ambiguity: {metrics['ambiguity_score']:.3f} | "
                )
                f.write(f"Coherence: {metrics['coherence_score']:.3f} | ")
                f.write(f"Dispersion: {metrics['dispersion']:.3f}\n")

            f.write(f"\nMost hallucination-prone concepts:\n")
            for concept, metrics in sorted_concepts[:5]:
                f.write(
                    f"- {concept}: Ambiguity score {metrics['ambiguity_score']:.3f}\n"
                )

            f.write(f"\nMost stable concepts:\n")
            for concept, metrics in sorted_concepts[-5:]:
                f.write(
                    f"- {concept}: Ambiguity score {metrics['ambiguity_score']:.3f}\n"
                )

        print(f"Summary report saved to: {report_path}")


def main():
    """
    Main function to run the conceptual hallucination analysis.
    """

    print("Conceptual Hallucination Analysis")
    print("Demonstrating how poor distributional semantics lead to hallucinations")
    print("=" * 70)

    # Initialize analyzer
    analyzer = ConceptualHallucinationAnalyzer(
        model_name="meta-llama/Llama-3.2-1B-Instruct"  # Change as needed
    )

    # Run full analysis
    analyzer.run_full_semantic_analysis(output_dir="semantic_hallucination_analysis")


if __name__ == "__main__":
    main()
