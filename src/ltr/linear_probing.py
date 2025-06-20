import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Tuple, Optional, Callable, Iterator
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from matplotlib.lines import Line2D
from tqdm import tqdm
import os
import json
from dataclasses import dataclass, field


@dataclass
class ProbeConfig:
    """Configuration for linear probe analysis"""

    model_name: str
    dataset_name: str = "custom"
    dataset_path: Optional[str] = None
    cache_dir: Optional[str] = None
    layer_norm_name: str = "post_attention_layernorm"
    batch_size: int = 8
    test_size: float = 0.3
    random_state: int = 42
    classifier: str = "LR"  # 'LR' or 'RF'
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    quantization: int = 32  # 8, 16, or 32 bits
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])
    # Dataset specific config
    text_col: str = None  # Auto-detect if None
    label_col: str = None  # Auto-detect if None
    prompt_template: str = None  # Optional template for text formatting


class ModelWrapper:
    """Extract hidden states from transformer models"""

    def __init__(self, config: ProbeConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model_name = config.model_name

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, cache_dir=config.cache_dir
        )

        if config.quantization == 32:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name, cache_dir=config.cache_dir
            )
        elif config.quantization == 16:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                cache_dir=config.cache_dir,
                torch_dtype=torch.bfloat16,
            )
        elif config.quantization == 8:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                cache_dir=config.cache_dir,
                quantization_config=quantization_config,
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Determine total layers
        self.num_layers = self.model.config.num_hidden_layers

        # Create layer names
        self.layer_names = [
            f"model.layers.{i}.{config.layer_norm_name}" for i in range(self.num_layers)
        ]

    def format_input(self, text: str) -> str:
        """Format input text using prompt template if provided"""
        if self.config.prompt_template:
            return self.config.prompt_template.format(text=text)
        return text

    def get_hidden_states(self, text: str) -> torch.Tensor:
        """Extract hidden states from all layers for a given text."""
        try:
            from baukit import TraceDict
        except ImportError:
            raise ImportError(
                "baukit is required for tracing. Install with: pip install git+https://github.com/davidbau/baukit"
            )

        formatted_text = self.format_input(text)

        with torch.no_grad():
            # Prepare input
            inputs = self.tokenizer(formatted_text, return_tensors="pt").to(self.device)

            # Trace through the model
            with TraceDict(self.model, self.layer_names) as tr:
                _ = self.model(**inputs)["logits"]

            # Collect hidden states from all layers
            hidden_states = torch.stack([tr[ln].output[0] for ln in self.layer_names])

            # Get the last token's representation for each layer
            last_token_states = hidden_states[:, 0, -1, :]  # [num_layers, hidden_size]

            return last_token_states.cpu()

    def batch_hidden_states(
        self, texts: List[str]
    ) -> Iterator[Tuple[List[str], torch.Tensor]]:
        """Process texts in batches and yield hidden states"""
        batch_size = self.config.batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_states = []

            for text in batch_texts:
                try:
                    states = self.get_hidden_states(text)
                    batch_states.append(states)
                except Exception as e:
                    print(f"Error processing text: {text[:50]}...")
                    print(f"Error: {e}")
                    # Return a tensor of zeros as fallback
                    zeros = torch.zeros(
                        (self.num_layers, self.model.config.hidden_size)
                    )
                    batch_states.append(zeros)

            if batch_states:
                yield batch_texts, torch.stack(batch_states)


class DatasetLoader:
    """Load and prepare datasets for linear probing"""

    @staticmethod
    def detect_columns(dataset) -> Tuple[str, str]:
        """Detect text and label columns"""
        # Common text column names
        text_candidates = [
            "text",
            "Text",
            "question",
            "Question",
            "content",
            "Content",
            "prompt",
            "Prompt",
        ]
        # Common label column names
        label_candidates = [
            "label",
            "Label",
            "answer",
            "Answer",
            "class",
            "Class",
            "target",
            "Target",
        ]

        text_col = None
        for candidate in text_candidates:
            if candidate in dataset.column_names:
                text_col = candidate
                break

        label_col = None
        for candidate in label_candidates:
            if candidate in dataset.column_names:
                label_col = candidate
                break

        if not text_col:
            raise ValueError(
                f"Could not detect text column. Available columns: {dataset.column_names}"
            )
        if not label_col:
            raise ValueError(
                f"Could not detect label column. Available columns: {dataset.column_names}"
            )

        return text_col, label_col

    @staticmethod
    def load_dataset(config: ProbeConfig) -> Tuple[List[str], List[str], List[int]]:
        """
        Load dataset and return texts and labels.

        Returns:
            texts: List of text strings
            labels: List of label strings
            int_labels: List of integer labels
        """
        # Load the dataset
        if config.dataset_name == "custom" and config.dataset_path:
            # Load custom dataset
            if config.dataset_path.endswith(".csv"):
                dataset = load_dataset("csv", data_files=config.dataset_path)["train"]
            elif config.dataset_path.endswith(".json"):
                dataset = load_dataset("json", data_files=config.dataset_path)["train"]
            elif config.dataset_path.endswith(".parquet"):
                dataset = load_dataset("parquet", data_files=config.dataset_path)[
                    "train"
                ]
            else:
                raise ValueError(f"Unsupported file format: {config.dataset_path}")
        else:
            # Load from Hugging Face datasets
            try:
                dataset = load_dataset(config.dataset_name)["train"]
            except:
                # Try with main split if train is not available
                dataset = load_dataset(config.dataset_name)["main"]

        # Detect column names if not provided
        text_col = config.text_col
        label_col = config.label_col

        if not text_col or not label_col:
            detected_text_col, detected_label_col = DatasetLoader.detect_columns(
                dataset
            )
            text_col = text_col or detected_text_col
            label_col = label_col or detected_label_col

        # Extract texts and labels
        texts = []
        labels = []

        print(f"Using text column: {text_col}, label column: {label_col}")

        for sample in dataset:
            # Handle potential nested structures
            text = sample[text_col]
            label = sample[label_col]

            # Handle potentially nested JSON structures
            if isinstance(text, dict) and "text" in text:
                text = text["text"]
            if isinstance(label, dict) and "text" in label:
                label = label["text"]

            # Handle list of strings (e.g., first element)
            if isinstance(text, list) and len(text) > 0 and isinstance(text[0], str):
                text = text[0]
            if isinstance(label, list) and len(label) > 0 and isinstance(label[0], str):
                label = label[0]

            texts.append(text)
            labels.append(label)

        # Convert string labels to integers
        unique_labels = sorted(set(labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        int_labels = [label_map[label] for label in labels]

        # Print dataset summary
        print(f"Loaded {len(texts)} examples")
        print(
            f"Label distribution: {dict(zip(unique_labels, [int_labels.count(i) for i in range(len(unique_labels))]))}"
        )

        return texts, labels, int_labels


class LinearProbeAnalyzer:
    """Analyze representations using linear probes"""

    def __init__(self, config: ProbeConfig):
        self.config = config
        self.model_wrapper = ModelWrapper(config)
        self.results = {"accuracy": [], "f1": [], "auc": []}

    def fit_and_evaluate(self, X_train, X_test, y_train, y_test) -> Dict[str, float]:
        """Train classifier and evaluate on test set"""
        # Initialize classifier
        if self.config.classifier == "LR":
            clf = LogisticRegression(
                random_state=self.config.random_state, max_iter=1000
            )
        elif self.config.classifier == "RF":
            clf = RandomForestClassifier(random_state=self.config.random_state)
        else:
            raise ValueError(f"Unsupported classifier: {self.config.classifier}")

        # Fit classifier
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred = clf.predict(X_test)

        # Calculate metrics
        metrics = {}
        if "accuracy" in self.config.metrics:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
        if "f1" in self.config.metrics:
            metrics["f1"] = f1_score(
                y_test, y_pred, average="binary" if len(set(y_test)) == 2 else "macro"
            )
        if "auc" in self.config.metrics:
            try:
                y_prob = clf.predict_proba(X_test)
                if len(set(y_test)) == 2:  # Binary classification
                    metrics["auc"] = roc_auc_score(y_test, y_prob[:, 1])
                else:  # Multi-class
                    metrics["auc"] = roc_auc_score(y_test, y_prob, multi_class="ovo")
            except:
                # If AUC calculation fails, set to NaN
                metrics["auc"] = float("nan")

        return metrics

    def analyze(self):
        """Run linear probe analysis across all layers"""
        # Load dataset
        texts, string_labels, int_labels = DatasetLoader.load_dataset(self.config)

        # Extract representations for each text in batches
        all_layer_representations = []

        print(f"Extracting representations from {len(texts)} examples...")
        for _, batch_hidden_states in tqdm(
            self.model_wrapper.batch_hidden_states(texts),
            total=(len(texts) + self.config.batch_size - 1) // self.config.batch_size,
        ):
            # batch_hidden_states shape: [batch_size, num_layers, hidden_size]
            for states in batch_hidden_states:
                all_layer_representations.append(states)

        # Ensure we have the right number of representations
        assert len(all_layer_representations) == len(texts), (
            f"Representation count ({len(all_layer_representations)}) doesn't match text count ({len(texts)})"
        )

        # Analyze each layer
        print(f"Analyzing {self.model_wrapper.num_layers} layers...")
        for layer_idx in tqdm(range(self.model_wrapper.num_layers)):
            # Get representations for this layer
            layer_reps = torch.stack(
                [rep[layer_idx] for rep in all_layer_representations]
            )
            X = layer_reps.numpy()
            y = np.array(int_labels)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y,  # Ensure class balance
            )

            # Evaluate
            metrics = self.fit_and_evaluate(X_train, X_test, y_train, y_test)

            # Store results
            for metric_name, value in metrics.items():
                self.results[metric_name].append(value)

            print(
                f"Layer {layer_idx}: "
                + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            )

        return self.results

    def save_results(self, filepath: str):
        """Save results to a JSON file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        # Save results
        with open(filepath, "w") as f:
            json.dump(self.results, f)

        print(f"Results saved to {filepath}")


class ProbeVisualizer:
    """Visualize linear probe results"""

    @staticmethod
    def load_results(filepath: str) -> Dict[str, List[float]]:
        """Load results from a JSON file"""
        with open(filepath, "r") as f:
            return json.load(f)

    @staticmethod
    def plot_metric_curves(
        results_list: List[Dict[str, List[float]]],
        model_names: List[str],
        metric: str = "accuracy",
        target_value: float = 0.8,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (4, 3),
        dpi: int = 300,
    ):
        """
        Plot metric curves across layers with target crossing points.

        Args:
            results_list: List of results dictionaries from different models
            model_names: List of model names corresponding to results_list
            metric: Which metric to plot ('accuracy', 'f1', 'auc')
            target_value: Target value to highlight (e.g., 0.8 for 80% accuracy)
            save_path: Path to save the figure
            title: Plot title
            figsize: Figure size
            dpi: Figure DPI
        """
        # Set up theme for publication-quality plots
        sns.set_theme(
            context="paper",
            style="whitegrid",
            font_scale=1.1,
            rc={
                "grid.linestyle": "--",
                "grid.alpha": 0.35,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            },
        )

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
        palette = sns.color_palette("colorblind", len(results_list))

        # Plot each model's curve
        tcp_points = []

        for i, results in enumerate(results_list):
            values = np.asarray(results[metric])
            x_vals = np.linspace(0, 100, len(values))

            # Clean up model name
            label = model_names[i]
            for prefix in ["-Instruct", "Qwen/", "Qwen2.5-", "gpt-"]:
                label = label.replace(prefix, "")

            # Plot curve
            ax.plot(
                x_vals,
                values,
                lw=1.4,
                marker="o",
                markersize=4.5,
                mec=palette[i],
                markeredgewidth=0.8,
                color=palette[i],
                label=label,
            )

            # Add vertical line at target crossing
            idx = np.argmax(values >= target_value)
            if idx < len(values) and values[idx] >= target_value:
                x_t = x_vals[idx]
                tcp_points.append((x_t, model_names[i]))
                ax.axvline(x_t, ls=":", lw=1.0, color=palette[i])
                ax.annotate(
                    f"{x_t:.0f}%",
                    xy=(x_t, target_value),
                    xytext=(0, -3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

        # Set up axes
        ax.set(
            xlabel="Layer progression (%)",
            ylabel=metric.capitalize(),
            xlim=(0, 100),
            ylim=(0, 1.02),
            xticks=[0, 25, 50, 75, 100],
            yticks=np.linspace(0, 1.0, 6),
        )
        ax.tick_params(axis="both", which="major", length=3)

        # Add TCP to legend
        handles, labels = ax.get_legend_handles_labels()
        tcp_proxy = Line2D([0], [0], ls=":", lw=1.0, color="gray", label="TCP")
        handles.append(tcp_proxy)
        labels.append("TCP")

        ax.legend(
            handles,
            labels,
            frameon=False,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=7,
            handlelength=1.4,
        )

        # Add title if provided
        if title:
            ax.set_title(title, fontsize=11)

        # Save figure
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, format="pdf", bbox_inches="tight")
            print(f"Figure saved to {save_path}")

            # Also save a PNG version for easy viewing
            png_path = save_path.replace(".pdf", ".png")
            fig.savefig(png_path, format="png", bbox_inches="tight", dpi=dpi)
            print(f"PNG preview saved to {png_path}")

        # Print TCP points
        if tcp_points:
            print("\nTemporal Compensation Points (TCP):")
            for x_t, model in tcp_points:
                print(f"  {model}: {x_t:.2f}% layer depth")

        plt.show()
        return fig


def run_analysis(config: ProbeConfig) -> Dict[str, List[float]]:
    """Helper function to run end-to-end analysis"""
    analyzer = LinearProbeAnalyzer(config)
    results = analyzer.analyze()

    # Create output filename
    model_id = config.model_name.replace("/", "_")
    dataset_id = (
        config.dataset_name
        if config.dataset_name != "custom"
        else os.path.basename(config.dataset_path).split(".")[0]
    )
    output_path = f"{model_id}_{dataset_id}_probe_results.json"

    # Save results
    analyzer.save_results(output_path)

    return results


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run linear probe analysis")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--dataset", type=str, default="custom", help="Dataset name or path"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="Path to custom dataset"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="LR",
        choices=["LR", "RF"],
        help="Classifier type",
    )
    parser.add_argument(
        "--quantization",
        type=int,
        default=32,
        choices=[8, 16, 32],
        help="Model quantization",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument(
        "--target", type=float, default=0.8, help="Target accuracy for TCP"
    )

    args = parser.parse_args()

    # Create config
    config = ProbeConfig(
        model_name=args.model,
        dataset_name=args.dataset if args.dataset != "custom" else "custom",
        dataset_path=args.dataset_path,
        classifier=args.classifier,
        quantization=args.quantization,
        batch_size=args.batch_size,
    )

    # Run analysis
    results = run_analysis(config)

    # Visualize results
    output_path = (
        args.output
        or f"{args.model.replace('/', '_')}_{args.dataset}_probe_results.pdf"
    )
    ProbeVisualizer.plot_metric_curves(
        [results],
        [args.model],
        metric="accuracy",
        target_value=args.target,
        save_path=output_path,
        title=f"Linear Probe Analysis: {args.model} on {args.dataset}",
    )
