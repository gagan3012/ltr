"""
Distributional Semantics Tracing (DST)

A unified approach to explain why language models hallucinate by:
1. Computing concept-importance scores via causal-path contributions
2. Generating patched representations to isolate feature drift
3. Extracting spurious input spans responsible for residual associations

Uses baukit.tracedict for activation tracking and intervention.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

from baukit import TraceDict
import torch.nn.functional as F


@dataclass
class DSTResult:
    """Container for DST analysis results"""

    concept_importance: Dict[int, float]  # Layer-wise importance scores
    patched_representations: Dict[str, Any]
    spurious_spans: List[Dict]
    semantic_drift_trajectory: Dict
    intervention_results: Optional[Dict] = None


class DistributionalSemanticsTracer:
    """
    Implements Distributional Semantics Tracing using baukit.tracedict to explain
    hallucination mechanisms by tracking semantic drift through model layers.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "auto",
        layer_prefix: str = "model.layers.",
        batch_size: int = 1,
    ):
        """
        Initialize the DST tracer with model and configuration.

        Args:
            model: The language model to analyze
            tokenizer: Tokenizer for the model
            device: Device to run computations on
            layer_prefix: Prefix for accessing layers in the model
            batch_size: Batch size for processing
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = (
            device
            if device != "auto"
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model.to(self.device)
        self.batch_size = batch_size
        self.layer_prefix = layer_prefix

        # Infer number of layers based on model structure
        self.n_layers = self._detect_num_layers()

        # Track layer names for baukit tracing
        self.layer_names = [
            f"{self.layer_prefix}{i}.output" for i in range(self.n_layers)
        ]
        self.mlp_names = [
            f"{self.layer_prefix}{i}.mlp.output" for i in range(self.n_layers)
        ]
        self.attention_names = [
            f"{self.layer_prefix}{i}.self_attn.output" for i in range(self.n_layers)
        ]

    def _detect_num_layers(self) -> int:
        """Detect number of layers in the model"""
        # This is a simple implementation, might need adjustment for different model architectures
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "num_hidden_layers"):
                return self.model.config.num_hidden_layers
            elif hasattr(self.model.config, "n_layer"):
                return self.model.config.n_layer

        # Fallback method: try to find layers by common patterns
        i = 0
        while hasattr(self.model, f"layer_{i}") or hasattr(self.model, f"h.{i}"):
            i += 1
        if i > 0:
            return i
        return 12  # Default assumption if detection fails

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to token IDs"""
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        return tokens

    def _model_forward(self, tokens, trace_layers=None):
        """Run model forward pass with optional layer tracing"""
        if trace_layers:
            with TraceDict(self.model, trace_layers) as ret:
                outputs = self.model(**tokens)
                return outputs, ret
        else:
            outputs = self.model(**tokens)
            return outputs, None

    def extract_concept_subspace(
        self, concept_examples: List[str], n_components: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Extract concept subspace from examples using PCA on activations.

        Args:
            concept_examples: Example strings representing the concept
            n_components: Number of principal components to extract

        Returns:
            Dictionary mapping layer names to concept subspace components
        """
        # Collect activations for concept examples
        all_activations = defaultdict(list)

        for example in concept_examples:
            tokens = self._encode_text(example)

            # Trace through all layers
            with TraceDict(self.model, self.layer_names) as traces:
                _ = self.model(**tokens)

                # Store activations
                for layer_name in self.layer_names:
                    # Get activations and reshape if needed
                    layer_act = traces[layer_name].output
                    if len(layer_act.shape) > 2:
                        # For sequence models, average across sequence dimension
                        layer_act = layer_act.mean(dim=1)
                    all_activations[layer_name].append(layer_act.detach().cpu())

        # Perform PCA to extract concept subspace for each layer
        concept_subspaces = {}

        for layer_name, activations in all_activations.items():
            # Stack activations
            act_stack = torch.cat(activations, dim=0)

            # Center the data
            act_mean = torch.mean(act_stack, dim=0, keepdim=True)
            act_centered = act_stack - act_mean

            # Compute covariance matrix
            cov = torch.mm(act_centered.T, act_centered) / (act_centered.shape[0] - 1)

            # Compute eigenvectors (concept directions)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)

            # Sort by eigenvalues (descending) and take top n_components
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            top_vectors = eigenvectors[:, sorted_indices[:n_components]]

            concept_subspaces[layer_name] = {
                "directions": top_vectors,
                "center": act_mean.squeeze(0),
            }

        return concept_subspaces

    def compute_concept_importance(
        self,
        prompt: str,
        concept_examples: List[str],
        target_token: Optional[str] = None,
        n_components: int = 10,
    ) -> Dict[str, float]:
        """
        Stage 1: Compute concept-importance scores via causal-path contributions.

        Args:
            prompt: Input prompt to analyze
            concept_examples: Example strings representing the concept
            target_token: Specific token to focus on (if None, uses last token)
            n_components: Number of components in concept subspace

        Returns:
            Dictionary mapping layer names to importance scores
        """
        # Extract concept subspace from examples
        concept_subspaces = self.extract_concept_subspace(
            concept_examples=concept_examples, n_components=n_components
        )

        # Tokenize input prompt
        tokens = self._encode_text(prompt)

        # Get target token position
        if target_token:
            target_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
            target_pos = (tokens.input_ids == target_id).nonzero()
            if len(target_pos) == 0:
                # If not found, use the last token
                target_pos = tokens.input_ids.shape[1] - 1
            else:
                target_pos = target_pos[0, 1].item()
        else:
            # Default to last token
            target_pos = tokens.input_ids.shape[1] - 1

        # Get baseline logits without intervention
        baseline_output, _ = self._model_forward(tokens)
        baseline_logits = baseline_output.logits[:, target_pos, :]

        # Compute causal importance by patching each layer's concept subspace
        importance_scores = {}

        for layer_idx, layer_name in enumerate(self.layer_names):
            # Skip if we don't have concept subspace for this layer
            if layer_name not in concept_subspaces:
                continue

            # Get concept directions and center
            directions = concept_subspaces[layer_name]["directions"].to(self.device)
            center = concept_subspaces[layer_name]["center"].to(self.device)

            # Define intervention function for this layer
            def intervention_fn(activations):
                # Project activations onto concept subspace
                flat_acts = activations.view(-1, activations.shape[-1])

                # Center activations
                centered_acts = flat_acts - center

                # Project onto concept directions
                proj = torch.mm(centered_acts, directions)

                # Zero out the projection (ablation)
                modified_acts = centered_acts - torch.mm(proj, directions.T)

                # Restore center
                modified_acts = modified_acts + center

                return modified_acts.view(activations.shape)

            # Run model with intervention at this layer
            with TraceDict(
                self.model, [layer_name], edit_output=intervention_fn
            ) as traces:
                modified_output = self.model(**tokens)
                modified_logits = modified_output.logits[:, target_pos, :]

            # Compute importance as logit difference
            logit_diff = F.kl_div(
                F.log_softmax(modified_logits, dim=-1),
                F.softmax(baseline_logits, dim=-1),
                reduction="sum",
            ).item()

            importance_scores[layer_name] = logit_diff

        return importance_scores

    def generate_patched_representations(
        self,
        prompt: str,
        factual_prompt: str,
        concept_importance: Dict[str, float],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Stage 2: Generate patched representations by replacing concept subspaces.

        Args:
            prompt: Input prompt containing potential hallucination
            factual_prompt: Ground truth prompt for the concept
            concept_importance: Importance scores from stage 1
            threshold: Threshold for selecting important dimensions

        Returns:
            Dictionary of patched representations and patching results
        """
        # Sort layers by importance score and select those above threshold
        important_layers = {
            k: v for k, v in concept_importance.items() if v > threshold
        }
        sorted_layers = sorted(
            important_layers.keys(), key=lambda k: important_layers[k], reverse=True
        )

        if not sorted_layers:
            return {"error": "No layers exceeded importance threshold"}

        # Get baseline activations for both prompts
        halluc_tokens = self._encode_text(prompt)
        factual_tokens = self._encode_text(factual_prompt)

        # Get baseline activations for factual prompt
        with TraceDict(self.model, sorted_layers) as factual_traces:
            _ = self.model(**factual_tokens)
            factual_activations = {
                k: v.output.detach().clone() for k, v in factual_traces.items()
            }

        # Create patches for each important layer
        patches = {}
        patched_logits = {}
        patched_outputs = {}

        for layer_name in sorted_layers:
            # Define patching function for this layer
            def patch_fn(activations):
                # Copy activations from factual run, but maintain batch size and sequence length
                shape_prefix = activations.shape[:-1]
                patched = factual_activations[layer_name].clone()

                # Handle sequence length differences
                if activations.shape[1] != patched.shape[1]:
                    # Use the min length
                    min_len = min(activations.shape[1], patched.shape[1])
                    # Only patch the overlapping part
                    activations[:, :min_len, :] = patched[:, :min_len, :]
                    return activations

                return patched

            # Run model with patch at this layer
            with TraceDict(self.model, [layer_name], edit_output=patch_fn) as traces:
                output = self.model(**halluc_tokens)

                # Store results
                patches[layer_name] = traces[layer_name].output.detach().clone()
                patched_logits[layer_name] = output.logits.detach().clone()
                patched_outputs[layer_name] = self.tokenizer.decode(
                    output.logits[0, -1].argmax().item(), skip_special_tokens=True
                )

        return {
            "patches": patches,
            "patched_logits": patched_logits,
            "patched_outputs": patched_outputs,
            "important_layers": sorted_layers,
            "importance_scores": important_layers,
        }

    def identify_spurious_spans(
        self,
        prompt: str,
        target_string: str,
        num_perturbations: int = 100,
        window_sizes: List[int] = [3, 5, 7],
    ) -> List[Dict]:
        """
        Stage 3: Extract spurious input spans through subsequence analysis.

        Args:
            prompt: Input prompt to analyze
            target_string: Hallucinated output string to trace
            num_perturbations: Number of perturbations to generate
            window_sizes: List of window sizes for span analysis

        Returns:
            List of spurious spans with their impact scores
        """
        # Tokenize prompt and get target token ID
        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        target_ids = self.tokenizer(target_string, add_special_tokens=False).input_ids

        # Get baseline output probability for the target
        baseline_output, _ = self._model_forward(tokens)
        target_probs = []

        # Get probability for each token in the target sequence
        for i, target_id in enumerate(target_ids):
            if i >= baseline_output.logits.size(1):
                break
            target_prob = F.softmax(baseline_output.logits[0, i], dim=-1)[
                target_id
            ].item()
            target_probs.append(target_prob)

        baseline_prob = np.mean(target_probs) if target_probs else 0.0

        # Tokenize into words for span analysis
        words = prompt.split()
        word_spans = []

        # Generate spans of different window sizes
        for window_size in window_sizes:
            for i in range(len(words) - window_size + 1):
                span_text = " ".join(words[i : i + window_size])
                word_spans.append(
                    {
                        "text": span_text,
                        "start_idx": i,
                        "end_idx": i + window_size,
                    }
                )

        # Score each span by masking and measuring impact
        for span in tqdm(word_spans, desc="Analyzing spans"):
            # Create perturbed prompt with span masked
            perturbed_words = words.copy()
            for i in range(span["start_idx"], span["end_idx"]):
                perturbed_words[i] = "[MASK]"
            perturbed_prompt = " ".join(perturbed_words)

            # Get output probability for target with span masked
            perturbed_tokens = self._encode_text(perturbed_prompt)
            perturbed_output, _ = self._model_forward(perturbed_tokens)

            # Calculate target probability with masked span
            perturbed_probs = []
            for i, target_id in enumerate(target_ids):
                if i >= perturbed_output.logits.size(1):
                    break
                prob = F.softmax(perturbed_output.logits[0, i], dim=-1)[
                    target_id
                ].item()
                perturbed_probs.append(prob)

            perturbed_prob = np.mean(perturbed_probs) if perturbed_probs else 0.0

            # Score is how much masking reduces target probability
            span["score"] = baseline_prob - perturbed_prob

        # Sort spans by score and return the most influential ones
        word_spans.sort(key=lambda x: x["score"], reverse=True)

        # Return top spans that increase probability of hallucination
        return [span for span in word_spans if span["score"] > 0][:10]

    def visualize_semantic_drift(
        self, prompt: str, factual_prompt: str, concept_importance: Dict[str, float]
    ) -> Dict:
        """
        Visualize the semantic drift trajectory that leads to hallucination.

        Args:
            prompt: Input prompt containing potential hallucination
            factual_prompt: Ground truth prompt for the concept
            concept_importance: Importance scores by layer

        Returns:
            Visualization data
        """
        # Encode both prompts
        halluc_tokens = self._encode_text(prompt)
        factual_tokens = self._encode_text(factual_prompt)

        # Get activations across all layers for both prompts
        with TraceDict(self.model, self.layer_names) as halluc_traces:
            _ = self.model(**halluc_tokens)
            halluc_activations = {
                k: v.output.detach() for k, v in halluc_traces.items()
            }

        with TraceDict(self.model, self.layer_names) as factual_traces:
            _ = self.model(**factual_tokens)
            factual_activations = {
                k: v.output.detach() for k, v in factual_traces.items()
            }

        # Calculate drift magnitudes across layers
        drift_magnitudes = {}
        layer_indices = []
        drift_values = []

        for i, layer_name in enumerate(self.layer_names):
            if layer_name in halluc_activations and layer_name in factual_activations:
                halluc_act = halluc_activations[layer_name]
                factual_act = factual_activations[layer_name]

                # Use the min sequence length
                min_len = min(halluc_act.shape[1], factual_act.shape[1])
                halluc_act = halluc_act[:, :min_len, :]
                factual_act = factual_act[:, :min_len, :]

                # Calculate semantic drift as cosine similarity
                halluc_flat = halluc_act.reshape(-1, halluc_act.shape[-1])
                factual_flat = factual_act.reshape(-1, factual_act.shape[-1])

                # Normalize vectors
                halluc_norm = F.normalize(halluc_flat, p=2, dim=1)
                factual_norm = F.normalize(factual_flat, p=2, dim=1)

                # Cosine similarity (higher means more similar)
                cos_sim = torch.mean(torch.sum(halluc_norm * factual_norm, dim=1))

                # Convert to distance (1 - similarity)
                drift = 1 - cos_sim.item()

                drift_magnitudes[layer_name] = drift
                layer_indices.append(i)
                drift_values.append(drift)

        # Create visualization data
        plt.figure(figsize=(10, 6))
        plt.plot(layer_indices, drift_values, "o-", linewidth=2)
        plt.xlabel("Model Layers")
        plt.ylabel("Semantic Drift Magnitude")
        plt.title("Semantic Drift Trajectory")
        plt.grid(True)

        # Highlight important layers based on concept importance
        if concept_importance:
            # Normalize importance scores
            max_importance = max(concept_importance.values())
            for layer_name, importance in concept_importance.items():
                if layer_name in self.layer_names:
                    layer_idx = self.layer_names.index(layer_name)
                    marker_size = (importance / max_importance) * 100 + 50
                    plt.plot(
                        layer_idx, drift_values[layer_idx], "ro", markersize=marker_size
                    )

        plt.tight_layout()

        # Save figure to temporary file or convert to base64 for display
        vis_file = "semantic_drift_trajectory.png"
        plt.savefig(vis_file)
        plt.close()

        return {
            "layer_indices": layer_indices,
            "drift_values": drift_values,
            "drift_magnitudes": drift_magnitudes,
            "visualization_file": vis_file,
        }

    def run_targeted_intervention(
        self,
        prompt: str,
        spurious_spans: List[Dict],
        patched_representations: Dict[str, Any],
    ) -> Dict:
        """
        Run targeted interventions to arrest semantic drift.

        Args:
            prompt: Original prompt
            spurious_spans: Identified spurious spans
            patched_representations: Patched representations from stage 2

        Returns:
            Results of intervention experiments
        """
        results = {}

        # 1. Remove spurious spans and test
        if spurious_spans:
            words = prompt.split()
            fixed_words = words.copy()

            # Remove top spurious span
            top_span = spurious_spans[0]
            for i in range(top_span["start_idx"], top_span["end_idx"]):
                if i < len(fixed_words):
                    fixed_words[i] = ""

            # Remove empty strings
            fixed_words = [w for w in fixed_words if w]
            fixed_prompt = " ".join(fixed_words)

            # Test fixed prompt
            fixed_tokens = self._encode_text(fixed_prompt)
            fixed_output, _ = self._model_forward(fixed_tokens)

            # Decode prediction
            next_token_id = fixed_output.logits[0, -1].argmax().item()
            next_token = self.tokenizer.decode([next_token_id])

            results["span_removal"] = {
                "removed_span": top_span["text"],
                "fixed_prompt": fixed_prompt,
                "predicted_next": next_token,
            }

        # 2. Apply patches at critical layers
        if "important_layers" in patched_representations:
            important_layers = patched_representations["important_layers"]
            if important_layers:
                # Pick most important layer
                critical_layer = important_layers[0]

                # Define patching function for critical layer
                def critical_patch_fn(activations):
                    return patched_representations["patches"][critical_layer]

                # Apply patch during inference
                tokens = self._encode_text(prompt)
                with TraceDict(
                    self.model, [critical_layer], edit_output=critical_patch_fn
                ) as traces:
                    output = self.model(**tokens)
                    next_token_id = output.logits[0, -1].argmax().item()
                    next_token = self.tokenizer.decode([next_token_id])

                results["critical_layer_patching"] = {
                    "layer": critical_layer,
                    "predicted_next": next_token,
                }

        return results

    def run_analysis(
        self,
        prompt: str,
        factual_prompt: str,
        concept_examples: List[str],
        hallucinated_output: str,
        run_intervention: bool = False,
    ) -> DSTResult:
        """
        Run the complete DST analysis pipeline.

        Args:
            prompt: Input prompt containing potential hallucination
            factual_prompt: Ground truth prompt for the concept
            concept_examples: Examples of the concept
            hallucinated_output: The hallucinated output to analyze
            run_intervention: Whether to run intervention experiments

        Returns:
            DSTResult object containing all analysis results
        """
        print("Stage 1: Computing concept importance...")
        concept_importance = self.compute_concept_importance(
            prompt=prompt, concept_examples=concept_examples
        )

        print("Stage 2: Generating patched representations...")
        patched_representations = self.generate_patched_representations(
            prompt=prompt,
            factual_prompt=factual_prompt,
            concept_importance=concept_importance,
        )

        print("Stage 3: Identifying spurious spans...")
        spurious_spans = self.identify_spurious_spans(
            prompt=prompt, target_string=hallucinated_output
        )

        print("Visualizing semantic drift...")
        semantic_drift = self.visualize_semantic_drift(
            prompt=prompt,
            factual_prompt=factual_prompt,
            concept_importance=concept_importance,
        )

        # Optionally run interventions
        intervention_results = None
        if run_intervention:
            print("Running targeted interventions...")
            intervention_results = self.run_targeted_intervention(
                prompt=prompt,
                spurious_spans=spurious_spans,
                patched_representations=patched_representations,
            )

        # Return comprehensive results
        return DSTResult(
            concept_importance=concept_importance,
            patched_representations=patched_representations,
            spurious_spans=spurious_spans,
            semantic_drift_trajectory=semantic_drift,
            intervention_results=intervention_results,
        )

if __name__ == "__main__":
    # Example usage
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tracer = DistributionalSemanticsTracer(model, tokenizer)

    prompt = "The capital of France is Paris, but it is also known for its Eiffel Tower."
    factual_prompt = "The capital of France is Paris."
    concept_examples = ["Paris", "Eiffel Tower", "France"]
    hallucinated_output = "The capital of France is Berlin."

    result = tracer.run_analysis(
        prompt=prompt,
        factual_prompt=factual_prompt,
        concept_examples=concept_examples,
        hallucinated_output=hallucinated_output,
        run_intervention=True,
    )

    print(result)
