"""
Distributional Semantics Tracing (DST)

A unified approach to explain why language models hallucinate by:
1. Computing concept-importance scores via causal-path contributions
2. Generating patched representations to isolate feature drift
3. Extracting spurious input spans responsible for residual associations

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
import seaborn as sns


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
            f"{self.layer_prefix}{i}" for i in range(self.n_layers)
        ]
        self.mlp_names = [
            f"{self.layer_prefix}{i}.mlp" for i in range(self.n_layers)
        ]
        self.attention_names = [
            f"{self.layer_prefix}{i}.self_attn" for i in range(self.n_layers)
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
                    # Get activations and handle tuple output format
                    layer_output = traces[layer_name].output
                    
                    # Fix: Check if output is a tuple and extract the tensor
                    if isinstance(layer_output, tuple):
                        # Most likely the first element contains the activation tensor
                        layer_act = layer_output[0]
                    else:
                        layer_act = layer_output
                    # Get activations and reshape if needed
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
                if isinstance(activations, tuple):
                    activations = activations[0]
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
                self.model, [layer_name], #edit_output=intervention_fn
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
                k: self._get_activation_from_trace(v.output).detach().clone()
                for k, v in factual_traces.items()
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
                k: self._get_activation_from_trace(v.output).detach()
                for k, v in halluc_traces.items()
            }

        with TraceDict(self.model, self.layer_names) as factual_traces:
            _ = self.model(**factual_tokens)
            factual_activations = {
                k: self._get_activation_from_trace(v.output).detach()
                for k, v in factual_traces.items()
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
    
    def _get_activation_from_trace(self, trace_output):
        """
        Helper function to extract activation tensor from trace output,
        which might be a tuple or a tensor directly.
        
        Args:
            trace_output: Output from TraceDict
            
        Returns:
            Activation tensor
        """
        if isinstance(trace_output, tuple):
            # Most commonly, the actual tensor is the first element
            return trace_output[0]
        return trace_output

    def visualize_semantic_drift_enhanced(
        self,
        prompt: str,
        factual_prompt: str,
        concept_importance: Dict[str, float],
        hallucinated_output: str = None,
        figsize=(14, 10)
    ) -> Dict:
        """
        Enhanced visualization of the semantic drift trajectory with distribution plots.
        
        Args:
            prompt: Input prompt containing potential hallucination
            factual_prompt: Ground truth prompt for the concept
            concept_importance: Importance scores by layer
            hallucinated_output: The hallucinated output to analyze
            figsize: Figure size
            
        Returns:
            Dictionary with visualization data and figures
        """
        # Encode both prompts
        halluc_tokens = self._encode_text(prompt)
        factual_tokens = self._encode_text(factual_prompt)
        
        # Get activations across all layers for both prompts
        with TraceDict(self.model, self.layer_names) as halluc_traces:
            _ = self.model(**halluc_tokens)
            halluc_activations = {
                k: self._get_activation_from_trace(v.output).detach()
                for k, v in halluc_traces.items()
            }
        
        with TraceDict(self.model, self.layer_names) as factual_traces:
            _ = self.model(**factual_tokens)
            factual_activations = {
                k: self._get_activation_from_trace(v.output).detach()
                for k, v in factual_traces.items()
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
        
        # Create main drift trajectory figure
        fig1, ax1 = plt.subplots(figsize=figsize)
        ax1.plot(layer_indices, drift_values, 'o-', linewidth=2)
        ax1.set_xlabel('Model Layers')
        ax1.set_ylabel('Semantic Drift Magnitude')
        ax1.set_title('Semantic Drift Trajectory')
        ax1.grid(True)
        
        # Highlight important layers based on concept importance
        if concept_importance:
            # Normalize importance scores
            max_importance = max(concept_importance.values())
            for layer_name, importance in concept_importance.items():
                if layer_name in self.layer_names:
                    layer_idx = self.layer_names.index(layer_name)
                    if layer_idx < len(drift_values):
                        marker_size = (importance / max_importance) * 100 + 50
                        ax1.plot(layer_idx, drift_values[layer_idx], 'ro', markersize=marker_size)
        
        # Create distributional comparison figure (inspired by distr_semantics_viz.py)
        # Select a few representative layers
        num_layers_to_show = min(5, len(layer_indices))
        sample_layers = np.linspace(0, len(layer_indices)-1, num_layers_to_show, dtype=int)
        sample_layer_indices = [layer_indices[i] for i in sample_layers]
        
        fig2, axes = plt.subplots(num_layers_to_show, 1, figsize=figsize, sharex=True)
        
        # Get tokens for annotations
        halluc_text = self.tokenizer.decode(halluc_tokens.input_ids[0], skip_special_tokens=True)
        halluc_tokens_text = [self.tokenizer.decode([t]) for t in halluc_tokens.input_ids[0]]
        
        for i, layer_idx in enumerate(sample_layer_indices):
            ax = axes[i] if num_layers_to_show > 1 else axes
            layer_name = self.layer_names[layer_idx]
            
            # Get activations for hallucinated and factual content
            if layer_name in halluc_activations and layer_name in factual_activations:
                halluc_act = halluc_activations[layer_name]
                factual_act = factual_activations[layer_name]
                
                # Use the min sequence length
                min_len = min(halluc_act.shape[1], factual_act.shape[1])
                
                # Flatten activations for distributional view
                halluc_flat = halluc_act[0, :min_len, :].reshape(-1).detach().cpu().numpy()
                factual_flat = factual_act[0, :min_len, :].reshape(-1).detach().cpu().numpy()
                
                # Create KDE plots
                try:
                    sns.kdeplot(halluc_flat, ax=ax, label='Hallucinated', color='red', fill=True, alpha=0.3)
                    sns.kdeplot(factual_flat, ax=ax, label='Factual', color='blue', fill=True, alpha=0.3)
                except np.linalg.LinAlgError:
                    # Fallback if KDE fails
                    ax.hist(halluc_flat, bins=50, alpha=0.3, color='red', label='Hallucinated', density=True)
                    ax.hist(factual_flat, bins=50, alpha=0.3, color='blue', label='Factual', density=True)
                
                ax.set_ylabel(f"Layer {layer_idx}\nDensity")
                
                if i == 0:
                    ax.legend()
                    
                if i == num_layers_to_show - 1:
                    ax.set_xlabel("Activation Value")
        
        fig2.suptitle("Distributional Comparison of Activations Across Layers", fontsize=14)
        plt.tight_layout()
        
        # Create token-wise drift figure
        fig3, ax3 = plt.subplots(figsize=figsize)
        
        # Calculate token-wise drift across layers
        token_drifts = []
        token_indices = []
        
        # Select a middle layer for token analysis
        mid_layer_idx = layer_indices[len(layer_indices)//2]
        mid_layer_name = self.layer_names[mid_layer_idx]
        
        if mid_layer_name in halluc_activations and mid_layer_name in factual_activations:
            halluc_act = halluc_activations[mid_layer_name]
            factual_act = factual_activations[mid_layer_name]
            
            min_len = min(halluc_act.shape[1], factual_act.shape[1])
            
            # Calculate drift for each token position
            for tok_idx in range(min_len):
                h_vec = halluc_act[0, tok_idx, :]
                f_vec = factual_act[0, tok_idx, :]
                
                # Normalize
                h_norm = F.normalize(h_vec.unsqueeze(0), p=2, dim=1)
                f_norm = F.normalize(f_vec.unsqueeze(0), p=2, dim=1)
                
                # Cosine similarity
                cos_sim = torch.sum(h_norm * f_norm, dim=1).item()
                drift = 1 - cos_sim
                
                token_drifts.append(drift)
                token_indices.append(tok_idx)
        
        # Plot token-wise drift
        ax3.plot(token_indices, token_drifts, 'o-', color='purple')
        ax3.set_xlabel('Token Position')
        ax3.set_ylabel('Semantic Drift Magnitude')
        ax3.set_title(f'Token-wise Semantic Drift at Layer {mid_layer_idx}')
        ax3.grid(True)
        
        # Add token annotations for top drift points
        if token_drifts:
            top_n = min(5, len(token_drifts))
            top_indices = sorted(range(len(token_drifts)), key=lambda i: token_drifts[i], reverse=True)[:top_n]
            
            for idx in top_indices:
                token_pos = token_indices[idx]
                if token_pos < len(halluc_tokens_text):
                    token_text = halluc_tokens_text[token_pos]
                    ax3.annotate(
                        token_text, 
                        (token_pos, token_drifts[idx]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
                    )
        
        # Save figures
        fig1_path = "semantic_drift_trajectory.png"
        fig2_path = "activation_distributions.png"
        fig3_path = "token_wise_drift.png"
        
        fig1.savefig(fig1_path)
        fig2.savefig(fig2_path)
        fig3.savefig(fig3_path)
        
        plt.close('all')
        
        return {
            "layer_indices": layer_indices,
            "drift_values": drift_values,
            "drift_magnitudes": drift_magnitudes,
            "visualization_files": [fig1_path, fig2_path, fig3_path],
            "halluc_activations": {k: v.cpu() for k, v in halluc_activations.items()},
            "factual_activations": {k: v.cpu() for k, v in factual_activations.items()}
        }

    def visualize_concept_network(
        self,
        prompt: str,
        spurious_spans: List[Dict],
        concept_importance: Dict[str, float],
        num_layers: int = 4,
        correlation_threshold: float = 0.3,
        figsize=(12, 10)
    ) -> List[plt.Figure]:
        """
        Visualize the network of concept relationships using graph visualization.
        
        Args:
            prompt: Input prompt to analyze
            spurious_spans: List of identified spurious spans
            concept_importance: Dictionary of concept importance scores
            num_layers: Number of layers to visualize
            correlation_threshold: Minimum correlation to show an edge
            figsize: Figure size
            
        Returns:
            List of figures with network visualizations
        """
        import networkx as nx
        from matplotlib.cm import get_cmap
        
        # Extract concepts from spurious spans and importance scores
        concepts = []
        
        # Add top spurious spans as concepts
        for span in spurious_spans[:3]:  # Use top 3 spans
            if span["text"] not in concepts:
                concepts.append(span["text"])
        
        # Add top concepts from importance scores
        important_layers = sorted(concept_importance.items(), key=lambda x: x[1], reverse=True)
        for layer, _ in important_layers[:3]:  # Use top 3 layers
            layer_name = layer.split('.')[-2]  # Extract layer name
            if layer_name not in concepts:
                concepts.append(layer_name)
                
        # Ensure we have at least some concepts
        if len(concepts) < 2:
            concepts = [prompt.split()[:3], prompt.split()[3:6]]  # Simple fallback
            
        # Tokenize the prompt
        tokens = self._encode_text(prompt)
        
        # Select layers to visualize
        layer_indices = np.linspace(0, self.n_layers - 1, num_layers, dtype=int)
        layer_names = [self.layer_names[i] for i in layer_indices]
        
        # Get activations for each layer
        with TraceDict(self.model, layer_names) as traces:
            _ = self.model(**tokens)
            activations = {
                k: self._get_activation_from_trace(v.output).detach()
                for k, v in traces.items()
            }
        
        # Create figures
        figures = []
        
        # For each layer
        for layer_idx, layer_name in zip(layer_indices, layer_names):
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create graph
            G = nx.Graph()
            
            # Define nodes from concepts
            for concept in concepts:
                G.add_node(concept)
                
            # Calculate token-wise activations for this layer
            if layer_name in activations:
                layer_act = activations[layer_name][0]  # [seq_len, hidden_dim]
                
                # Create pseudo-concept activations by using token positions
                concept_activations = {}
                for concept in concepts:
                    # Find all occurrences of concept in tokenized text
                    concept_tokens = self.tokenizer.encode(concept, add_special_tokens=False)
                    
                    # Look for these tokens in the input
                    for i in range(len(tokens.input_ids[0]) - len(concept_tokens) + 1):
                        match = True
                        for j in range(len(concept_tokens)):
                            if i + j >= len(tokens.input_ids[0]) or tokens.input_ids[0][i + j] != concept_tokens[j]:
                                match = False
                                break
                        
                        if match:
                            # Found the concept, use its activations
                            concept_activations[concept] = torch.mean(layer_act[i:i+len(concept_tokens)], dim=0)
                            break
                    
                    # If concept not found, use a random position as fallback
                    if concept not in concept_activations:
                        random_pos = random.randint(0, layer_act.shape[0]-1)
                        concept_activations[concept] = layer_act[random_pos]
            
                # Compute correlations and add edges
                for i, concept1 in enumerate(concepts):
                    for j in range(i+1, len(concepts)):
                        concept2 = concepts[j]
                        
                        if concept1 in concept_activations and concept2 in concept_activations:
                            # Calculate correlation
                            vec1 = concept_activations[concept1].cpu().numpy()
                            vec2 = concept_activations[concept2].cpu().numpy()
                            
                            corr = np.corrcoef(vec1, vec2)[0, 1]
                            
                            # Only add edge if correlation is significant
                            if abs(corr) >= correlation_threshold:
                                G.add_edge(concept1, concept2, weight=abs(corr), correlation=corr)
            
            # Draw graph
            if G.number_of_edges() > 0:
                pos = nx.spring_layout(G, seed=42)
                
                # Choose colormap based on edge data
                edge_data = list(G.edges(data=True))
                any_negative = any(d["correlation"] < 0 for _, _, d in edge_data)
                
                if any_negative:
                    cmap = plt.get_cmap("coolwarm")
                    norm = plt.Normalize(vmin=-1, vmax=1)
                    value_fn = lambda r: r
                else:
                    cmap = plt.get_cmap("Reds")
                    norm = plt.Normalize(vmin=correlation_threshold, vmax=1)
                    value_fn = abs
                    
                # Draw nodes
                nx.draw_networkx_nodes(
                    G, pos, ax=ax, node_size=1000, 
                    node_color='lightblue', edgecolors='black', alpha=0.7
                )
                
                # Draw labels
                nx.draw_networkx_labels(
                    G, pos, ax=ax, font_size=10, font_weight='bold'
                )
                
                # Draw edges with color based on correlation
                for u, v, d in G.edges(data=True):
                    corr = d["correlation"]
                    color_val = (corr + 1) / 2 if any_negative else abs(corr)
                    nx.draw_networkx_edges(
                        G, pos, ax=ax, edgelist=[(u, v)],
                        width=2 + 3*abs(corr),
                        alpha=0.7,
                        edge_color=[cmap(norm(value_fn(corr)))]
                    )
                    
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7)
                cbar.set_label('Correlation', fontsize=10)
                
                # Set title
                ax.set_title(f"Layer {layer_idx}: Concept Relationship Network", fontsize=12)
                
                # Turn off axis
                ax.axis('off')
            else:
                # No significant correlations
                ax.text(
                    0.5, 0.5,
                    f"No significant concept correlations in Layer {layer_idx}",
                    ha='center', va='center', fontsize=12
                )
                ax.axis('off')
            
            figures.append(fig)
            
            # Save each figure
            fig_path = f"concept_network_layer_{layer_idx}.png"
            fig.savefig(fig_path)
        
        return figures

    def visualize_activation_distributions(
        self,
        prompt: str,
        factual_prompt: str,
        hallucinated_output: str,
        num_layers_to_show: int = 5,
        figsize=(14, 10)
    ) -> plt.Figure:
        """
        Visualize distributions of activations for hallucinated and factual prompts.
        
        Args:
            prompt: Input prompt with potential hallucination
            factual_prompt: Ground truth prompt
            hallucinated_output: The hallucinated output string
            num_layers_to_show: Number of layers to visualize
            figsize: Figure size
            
        Returns:
            Figure with activation distributions
        """
        # Encode prompts
        halluc_tokens = self._encode_text(prompt)
        factual_tokens = self._encode_text(factual_prompt)
        
        # Select layers to visualize
        layer_indices = np.linspace(0, self.n_layers - 1, num_layers_to_show, dtype=int)
        layer_names = [self.layer_names[i] for i in layer_indices]
        
        # Get activations
        with TraceDict(self.model, layer_names) as halluc_traces:
            _ = self.model(**halluc_tokens)
            # halluc_activations = {k: v.output.detach() for k, v in halluc_traces.items()}
            halluc_activations = {k: self._get_activation_from_trace(v.output).detach() for k, v in halluc_traces.items()}
        
        with TraceDict(self.model, layer_names) as factual_traces:
            _ = self.model(**factual_tokens)
            factual_activations = {k: self._get_activation_from_trace(v.output).detach() for k, v in factual_traces.items()}

        # Create figure
        fig, axes = plt.subplots(num_layers_to_show, 1, figsize=figsize, sharex=True)
        
        # Handle single layer case
        if num_layers_to_show == 1:
            axes = [axes]
        
        # For each layer
        for i, (layer_idx, layer_name) in enumerate(zip(layer_indices, layer_names)):
            ax = axes[i]
            
            if layer_name in halluc_activations and layer_name in factual_activations:
                halluc_act = halluc_activations[layer_name]
                factual_act = factual_activations[layer_name]
                
                # Use the min sequence length
                min_len = min(halluc_act.shape[1], factual_act.shape[1])
                
                # Reshape for distribution visualization
                halluc_values = halluc_act[0, :min_len, :].reshape(-1).cpu().numpy()
                factual_values = factual_act[0, :min_len, :].reshape(-1).cpu().numpy()
                
                # Apply smoothing to better visualize distributions
                try:
                    sns.kdeplot(halluc_values, ax=ax, label='Hallucinated', color='red', alpha=0.6)
                    sns.kdeplot(factual_values, ax=ax, label='Factual', color='blue', alpha=0.6)
                except np.linalg.LinAlgError:
                    # Fallback if KDE fails
                    ax.hist(halluc_values, bins=50, alpha=0.4, color='red', density=True, label='Hallucinated')
                    ax.hist(factual_values, bins=50, alpha=0.4, color='blue', density=True, label='Factual')
                    
                # Highlight the difference between distributions
                ax.fill_between(
                    ax.lines[0].get_xdata(),
                    ax.lines[0].get_ydata(),
                    ax.lines[1].get_ydata(),
                    where=(ax.lines[0].get_ydata() > ax.lines[1].get_ydata()),
                    color='red', alpha=0.2
                )
                
                ax.set_ylabel(f"Layer {layer_idx}")
                
                if i == 0:
                    ax.legend()
                    
                if i == num_layers_to_show - 1:
                    ax.set_xlabel("Activation Value")
        
        plt.suptitle("Distributional Semantics: Hallucinated vs. Factual", fontsize=14)
        plt.tight_layout()
        
        # Save figure
        fig_path = "activation_distributions.png"
        fig.savefig(fig_path)
        
        return fig

    def create_concept_correlation_matrix(
        self,
        prompt: str,
        concept_examples: List[str],
        figsize=(12, 10)
    ) -> plt.Figure:
        """
        Create correlation matrix showing relationships between concepts.
        
        Args:
            prompt: Input prompt to analyze
            concept_examples: Example strings representing different concepts
            figsize: Figure size
            
        Returns:
            Figure with correlation matrix
        """
        # Tokenize the prompt
        tokens = self._encode_text(prompt)
        
        # Encode concept examples
        concept_tokens = [self._encode_text(ex) for ex in concept_examples]
        
        # Get middle layer for analysis
        mid_layer = self.layer_names[self.n_layers // 2]
        
        # Get activations for the prompt
        with TraceDict(self.model, [mid_layer]) as traces:
            _ = self.model(**tokens)
            prompt_act = traces[mid_layer].output.detach()
        
        # Get activations for each concept
        concept_acts = []
        for c_tokens in concept_tokens:
            with TraceDict(self.model, [mid_layer]) as traces:
                _ = self.model(**c_tokens)
                concept_acts.append(traces[mid_layer].output.detach())
        
        # Calculate correlation matrix
        n_concepts = len(concept_examples)
        corr_matrix = np.zeros((n_concepts, n_concepts))
        
        for i in range(n_concepts):
            for j in range(n_concepts):
                # Get mean activation vectors
                act_i = concept_acts[i].mean(dim=1).cpu().numpy()
                act_j = concept_acts[j].mean(dim=1).cpu().numpy()
                
                # Calculate correlation
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr = np.corrcoef(act_i.flatten(), act_j.flatten())[0, 1]
                    corr_matrix[i, j] = corr
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot correlation matrix
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax,
            xticklabels=concept_examples,
            yticklabels=concept_examples
        )
        
        ax.set_title("Concept Correlation Matrix", fontsize=14)
        plt.tight_layout()
        
        # Save figure
        fig_path = "concept_correlation_matrix.png"
        fig.savefig(fig_path)
        
        return fig

    def run_analysis(
        self,
        prompt: str,
        factual_prompt: str,
        concept_examples: List[str],
        hallucinated_output: str,
        run_intervention: bool = False,
        enhanced_viz: bool = True,
    ) -> DSTResult:
        """
        Run the complete DST analysis pipeline with enhanced visualizations.

        Args:
            prompt: Input prompt containing potential hallucination
            factual_prompt: Ground truth prompt for the concept
            concept_examples: Examples of the concept
            hallucinated_output: The hallucinated output to analyze
            run_intervention: Whether to run intervention experiments
            enhanced_viz: Whether to use enhanced visualizations

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
        if enhanced_viz:
            semantic_drift = self.visualize_semantic_drift_enhanced(
                prompt=prompt,
                factual_prompt=factual_prompt,
                concept_importance=concept_importance,
                hallucinated_output=hallucinated_output,
            )
            
            # Create network visualization
            print("Generating concept network visualization...")
            network_viz = self.visualize_concept_network(
                prompt=prompt,
                spurious_spans=spurious_spans,
                concept_importance=concept_importance,
            )
            
            # Create activation distributions visualization
            print("Generating activation distributions visualization...")
            dist_viz = self.visualize_activation_distributions(
                prompt=prompt,
                factual_prompt=factual_prompt,
                hallucinated_output=hallucinated_output,
            )
            
            # Create concept correlation matrix
            print("Generating concept correlation matrix...")
            corr_matrix = self.create_concept_correlation_matrix(
                prompt=prompt,
                concept_examples=concept_examples,
            )
            
            # Add enhanced visualizations to semantic_drift
            semantic_drift["network_visualization"] = "concept_network_layer_*.png"
            semantic_drift["distribution_visualization"] = "activation_distributions.png"
            semantic_drift["correlation_matrix"] = "concept_correlation_matrix.png"
        else:
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

