import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from baukit import TraceDict
import logging
from functools import partial
import re


class PatchscopeAnalyzer:
    """
    Main class for performing patchscope analysis similar to Racing_Thoughts implementation.
    """

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or model.device
        self.model_type = self._get_model_type()
        self.layer_patterns = self._configure_layer_patterns()

    def _get_model_type(self):
        """Determine model architecture type."""
        return (
            self.model.config.model_type.lower()
            if hasattr(self.model.config, "model_type")
            else ""
        )

    def _configure_layer_patterns(self):
        """Configure layer patterns based on model architecture."""
        if (
            "llama" in self.model_type
            or "mistral" in self.model_type
            or "qwen" in self.model_type
        ):
            return {
                "attention": "model.layers.{}.self_attn",
                "mlp": "model.layers.{}.mlp",
                "residual": "model.layers.{}",
                "n_layers": self.model.config.num_hidden_layers,
            }
        elif "gpt-neox" in self.model_type or "gpt_neox" in self.model_type:
            return {
                "attention": "gpt_neox.layers.{}.attention",
                "mlp": "gpt_neox.layers.{}.mlp",
                "residual": "gpt_neox.layers.{}",
                "n_layers": self.model.config.num_hidden_layers,
            }
        elif "gpt2" in self.model_type:
            return {
                "attention": "transformer.h.{}.attn",
                "mlp": "transformer.h.{}.mlp",
                "residual": "transformer.h.{}",
                "n_layers": self.model.config.n_layer,
            }
        else:
            return {
                "attention": "model.layers.{}.attention",
                "mlp": "model.layers.{}.mlp",
                "residual": "model.layers.{}",
                "n_layers": getattr(self.model.config, "num_hidden_layers", 12),
            }

    def get_layer_names(self, component_type="attention", layers=None):
        """Get layer names for tracing."""
        if layers is None:
            layers = range(self.layer_patterns["n_layers"])

        pattern = self.layer_patterns.get(
            component_type, self.layer_patterns["attention"]
        )
        return [pattern.format(layer) for layer in layers]


def get_entity_token_ids(tokenizer, entities):
    """
    More robust entity tokenization that handles multi-token entities and variations.
    """
    entity_mappings = {}

    for entity in entities:
        # Try different tokenization approaches
        variations = [
            entity,
            entity.lower(),
            entity.upper(),
            entity.capitalize(),
            f" {entity}",  # With leading space
            f"{entity.lower()}",
            f"{entity.upper()}",
        ]

        found_tokens = []
        for variation in variations:
            try:
                tokens = tokenizer(variation, add_special_tokens=False).input_ids
                if tokens:
                    found_tokens.extend(tokens)
            except:
                continue

        # Remove duplicates and take most common patterns
        unique_tokens = list(set(found_tokens))
        entity_mappings[entity] = unique_tokens  # Take up to 3 variations

        print(f"Entity '{entity}' mapped to token IDs: {unique_tokens}")

    return entity_mappings


def calculate_entity_probabilities(probs, entity_mappings):
    """
    Calculate entity probabilities using multiple token variations.
    """
    entity_probs = {}

    for entity, token_ids in entity_mappings.items():
        if not token_ids:
            entity_probs[entity] = 0.0
            continue

        # Sum probabilities across all token variations for this entity
        total_prob = 0.0
        valid_tokens = 0

        for token_id in token_ids:
            if 0 <= token_id < len(probs):
                total_prob += float(probs[token_id])
                valid_tokens += 1

        # Average probability across valid tokens
        entity_probs[entity] = total_prob / max(1, valid_tokens)

    return entity_probs


def perform_patchscope_analysis(
    model,
    tokenizer,
    prompt: str,
    target_layers: Optional[List[int]] = None,
    explanation_prompts: Optional[List[str]] = None,
    max_tokens: int = 50,
    window_size: int = 5,
    target_entities: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Perform comprehensive patchscope analysis with improved entity tracking.
    """

    analyzer = PatchscopeAnalyzer(model, tokenizer)

    # Set default parameters
    if target_layers is None:
        # Sample layers more strategically
        n_layers = analyzer.layer_patterns["n_layers"]
        target_layers = [
            0,
            n_layers // 4,
            n_layers // 2,
            3 * n_layers // 4,
            n_layers - 1,
        ]
        target_layers = [l for l in target_layers if l < n_layers]

    if explanation_prompts is None:
        explanation_prompts = [
            "What concept is the model processing?",
            "What is the model's confidence in this prediction?",
            "Is this factually correct?",
            "What reasoning led to this output?",
        ]

    if target_entities is None:
        target_entities = []

    print(f"Using target layers: {target_layers}")

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids[0]

    results = {
        "prompt": prompt,
        "input_tokens": tokenizer.convert_ids_to_tokens(input_ids),
        "layer_explanations": [],
        "entity_traces": {entity: [] for entity in target_entities},
        "attention_patterns": {},
        "generation_trace": [],
        "intervention_effects": {},
        "metadata": {
            "model_type": analyzer.model_type,
            "n_layers": analyzer.layer_patterns["n_layers"],
            "target_layers": target_layers,
        },
        "hallucination_indicators": {},
        "confidence_scores": [],
    }

    # Improved entity tracking
    entity_mappings = get_entity_token_ids(tokenizer, target_entities)

    # Generate and analyze token by token
    current_input_ids = input_ids.clone()

    for generation_step in range(max_tokens):
        step_results = {
            "step": generation_step,
            "input_length": len(current_input_ids),
            "layer_activations": {},
            "attention_weights": {},
            "entity_probabilities": {},
            "next_token_info": {},
            "explanations": {},
            "hallucination_signals": {},
        }

        try:
            # Forward pass without tracing first (for better performance)
            with torch.no_grad():
                outputs = model(current_input_ids.unsqueeze(0))
                logits = outputs.logits[0, -1]

                # Get next token
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.argmax(probs).item()
                next_token = tokenizer.decode([next_token_id]).strip()

                # Calculate confidence metrics
                top_probs, top_indices = torch.topk(probs, 5)
                confidence_score = float(top_probs[0])
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

                step_results["next_token_info"] = {
                    "token_id": next_token_id,
                    "token": next_token,
                    "probability": confidence_score,
                    "entropy": entropy,
                    "top_5_tokens": [
                        {
                            "token": tokenizer.decode([int(tid)]).strip(),
                            "token_id": int(tid),
                            "probability": float(prob),
                        }
                        for tid, prob in zip(top_indices, top_probs)
                    ],
                }

                # Improved entity probability calculation
                entity_probs = calculate_entity_probabilities(probs, entity_mappings)
                step_results["entity_probabilities"] = entity_probs

                # Update entity traces
                for entity, prob in entity_probs.items():
                    if prob > 1e-6:  # Only record significant probabilities
                        results["entity_traces"][entity].append(
                            {
                                "step": generation_step,
                                "position": len(current_input_ids),
                                "probability": prob,
                                "token_generated": next_token,
                            }
                        )

                # Hallucination detection signals
                hallucination_signals = {
                    "low_confidence": confidence_score < 0.3,
                    "high_entropy": entropy > 5.0,
                    "unexpected_entity": any(
                        prob > 0.1 for prob in entity_probs.values()
                    ),
                    "token_repetition": next_token
                    in [
                        tokenizer.decode([current_input_ids[i]])
                        for i in range(
                            max(0, len(current_input_ids) - 5), len(current_input_ids)
                        )
                    ],
                }
                step_results["hallucination_signals"] = hallucination_signals

                # Store confidence score
                results["confidence_scores"].append(confidence_score)

            # Layer analysis (every few steps to reduce overhead)
            if generation_step % 3 == 0:
                layer_data = analyze_layer_activations(
                    model, current_input_ids, target_layers, analyzer
                )
                step_results["layer_activations"] = layer_data

        except Exception as e:
            logging.error(f"Error in generation step {generation_step}: {e}")
            step_results["error"] = str(e)

        results["generation_trace"].append(step_results)

        # Update input for next iteration
        current_input_ids = torch.cat(
            [current_input_ids, torch.tensor([next_token_id], device=model.device)]
        )

        # Stop conditions
        if next_token_id == tokenizer.eos_token_id:
            break

    # Final generation text
    results["generated_text"] = tokenizer.decode(
        current_input_ids, skip_special_tokens=True
    )

    # Enhanced post-processing
    results = add_enhanced_summary_statistics(results)
    results = detect_hallucination_patterns(results)

    return results


def analyze_layer_activations(model, input_ids, target_layers, analyzer):
    """
    Analyze layer activations without full tracing for better performance.
    """
    layer_data = {}

    # Get residual patterns for target layers
    residual_patterns = analyzer.get_layer_names("residual", target_layers)

    try:
        with TraceDict(model, residual_patterns) as traces:
            # Forward pass
            _ = model(input_ids.unsqueeze(0))

            for i, layer_idx in enumerate(target_layers):
                pattern = residual_patterns[i]
                if pattern in traces:
                    try:
                        activation = traces[pattern].output
                        if isinstance(activation, tuple):
                            activation = activation[0]

                        # Analyze last token activation
                        last_token_activation = activation[0, -1].detach().cpu()

                        layer_data[layer_idx] = {
                            "activation_norm": float(torch.norm(last_token_activation)),
                            "activation_mean": float(torch.mean(last_token_activation)),
                            "activation_std": float(torch.std(last_token_activation)),
                            "activation_max": float(torch.max(last_token_activation)),
                            "activation_min": float(torch.min(last_token_activation)),
                            "sparsity": float(
                                torch.sum(torch.abs(last_token_activation) < 0.1)
                                / len(last_token_activation)
                            ),
                        }
                    except Exception as e:
                        logging.warning(f"Error processing layer {layer_idx}: {e}")
                        layer_data[layer_idx] = {"error": str(e)}

    except Exception as e:
        logging.error(f"Error in layer analysis: {e}")

    return layer_data


def add_enhanced_summary_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Add enhanced summary statistics with better insights."""
    summary = {
        "total_generation_steps": len(results["generation_trace"]),
        "entities_tracked": list(results["entity_traces"].keys()),
        "layers_analyzed": results["metadata"]["target_layers"],
        "average_entity_probabilities": {},
        "max_entity_probabilities": {},
        "entity_appearance_counts": {},
        "attention_summary": {},
        "key_insights": [],
        "hallucination_indicators": {},
        "confidence_statistics": {},
    }

    # Enhanced entity analysis
    for entity, trace in results["entity_traces"].items():
        if trace:
            probs = [step["probability"] for step in trace]
            summary["average_entity_probabilities"][entity] = np.mean(probs)
            summary["max_entity_probabilities"][entity] = np.max(probs)
            summary["entity_appearance_counts"][entity] = len(trace)

    # Confidence statistics
    if results["confidence_scores"]:
        summary["confidence_statistics"] = {
            "mean_confidence": np.mean(results["confidence_scores"]),
            "min_confidence": np.min(results["confidence_scores"]),
            "max_confidence": np.max(results["confidence_scores"]),
            "confidence_std": np.std(results["confidence_scores"]),
        }

    # Hallucination indicators
    hallucination_counts = {}
    for step in results["generation_trace"]:
        if "hallucination_signals" in step:
            for signal, value in step["hallucination_signals"].items():
                if signal not in hallucination_counts:
                    hallucination_counts[signal] = 0
                if value:
                    hallucination_counts[signal] += 1

    summary["hallucination_indicators"] = hallucination_counts

    # Generate insights
    insights = []

    if summary["average_entity_probabilities"]:
        max_entity = max(
            summary["average_entity_probabilities"].items(), key=lambda x: x[1]
        )
        insights.append(
            f"Highest average entity probability: {max_entity[0]} ({max_entity[1]:.4f})"
        )

        # Find entities with suspicious patterns
        for entity, avg_prob in summary["average_entity_probabilities"].items():
            max_prob = summary["max_entity_probabilities"][entity]
            if max_prob > 0.1 and avg_prob < 0.05:
                insights.append(
                    f"Entity '{entity}' shows spike pattern (max: {max_prob:.4f}, avg: {avg_prob:.4f})"
                )

    if summary["confidence_statistics"]:
        mean_conf = summary["confidence_statistics"]["mean_confidence"]
        if mean_conf < 0.3:
            insights.append(f"Low overall confidence detected (mean: {mean_conf:.3f})")
        elif mean_conf > 0.8:
            insights.append(f"High confidence generation (mean: {mean_conf:.3f})")

    if hallucination_counts:
        total_steps = summary["total_generation_steps"]
        for signal, count in hallucination_counts.items():
            if count > total_steps * 0.3:  # More than 30% of steps
                insights.append(
                    f"Frequent {signal} detected ({count}/{total_steps} steps)"
                )

    summary["key_insights"] = insights
    results["summary"] = summary
    return results


def detect_hallucination_patterns(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect specific hallucination patterns in the generation.
    """
    patterns = {
        "factual_inconsistency": False,
        "entity_confusion": False,
        "confidence_drops": [],
        "anomalous_activations": [],
    }

    # Check for confidence drops
    if results["confidence_scores"]:
        for i, score in enumerate(results["confidence_scores"]):
            if i > 0 and score < results["confidence_scores"][i - 1] * 0.5:
                patterns["confidence_drops"].append(i)

    # Check for entity confusion
    entity_traces = results["entity_traces"]
    if len(entity_traces) >= 2:
        # Look for competing entities
        for step in results["generation_trace"]:
            if "entity_probabilities" in step:
                high_prob_entities = [
                    entity
                    for entity, prob in step["entity_probabilities"].items()
                    if prob > 0.05
                ]
                if len(high_prob_entities) > 1:
                    patterns["entity_confusion"] = True
                    break

    # Check for anomalous layer activations
    for step in results["generation_trace"]:
        if "layer_activations" in step:
            for layer_idx, layer_data in step["layer_activations"].items():
                if isinstance(layer_data, dict) and "activation_norm" in layer_data:
                    norm = layer_data["activation_norm"]
                    if norm > 20.0 or norm < 0.1:  # Very high or very low
                        patterns["anomalous_activations"].append(
                            {"step": step["step"], "layer": layer_idx, "norm": norm}
                        )

    results["hallucination_patterns"] = patterns
    return results


# Keep the existing functions but with improvements
def extract_attention_weights(
    attn_output, window_size: int, model_type: str
) -> List[float]:
    """Extract and process attention weights from model output."""
    try:
        if isinstance(attn_output, tuple):
            if len(attn_output) > 1 and attn_output[1] is not None:
                attn_weights = attn_output[1]
            else:
                return []
        else:
            return []

        if attn_weights is None or len(attn_weights.shape) < 3:
            return []

        # Process attention weights based on shape
        if len(attn_weights.shape) == 4:  # [batch, heads, seq_len, seq_len]
            attn = attn_weights[0, :, -1, -window_size:].mean(dim=0)
        elif len(attn_weights.shape) == 3:  # [heads, seq_len, seq_len]
            attn = attn_weights[:, -1, -window_size:].mean(dim=0)
        else:
            attn = attn_weights[-1, -window_size:]

        return attn.detach().cpu().numpy().tolist()

    except Exception as e:
        logging.warning(f"Error processing attention weights: {e}")
        return []


def perform_layer_explanations(
    model, tokenizer, traces, target_layers, explanation_prompts, analyzer
) -> Dict[str, Any]:
    """Perform patchscope self-explanations using traced activations."""
    explanations = {}

    for layer_idx in target_layers:
        layer_explanations = []
        residual_pattern = analyzer.get_layer_names("residual", [layer_idx])[0]

        if residual_pattern in traces:
            try:
                activation = traces[residual_pattern].output
                if isinstance(activation, tuple):
                    activation = activation[0]

                for prompt in explanation_prompts:
                    explanation = generate_explanation_from_activation(
                        model, tokenizer, activation, prompt
                    )
                    layer_explanations.append(
                        {"prompt": prompt, "explanation": explanation}
                    )

                explanations[layer_idx] = layer_explanations

            except Exception as e:
                logging.warning(
                    f"Could not process layer {layer_idx} for explanations: {e}"
                )

    return explanations


def generate_explanation_from_activation(
    model, tokenizer, activation, explanation_prompt
):
    """Generate explanation by analyzing activation patterns."""
    try:
        if isinstance(activation, tuple):
            activation = activation[0]

        last_token_act = activation[0, -1]

        # More sophisticated analysis
        norm = float(torch.norm(last_token_act))
        mean = float(torch.mean(last_token_act))
        std = float(torch.std(last_token_act))
        sparsity = float(
            torch.sum(torch.abs(last_token_act) < 0.1) / len(last_token_act)
        )

        # Generate contextual explanations
        if "factually correct" in explanation_prompt.lower():
            if norm < 2.0:
                return "Low activation suggests uncertain or speculative content"
            elif norm > 15.0:
                return "Very high activation may indicate confident but potentially incorrect assertion"
            else:
                return "Moderate activation suggests normal factual processing"

        elif "confidence" in explanation_prompt.lower():
            if std < 1.0:
                return f"Low variance (std={std:.2f}) suggests high confidence"
            elif std > 3.0:
                return f"High variance (std={std:.2f}) suggests uncertainty"
            else:
                return f"Moderate variance (std={std:.2f}) suggests normal confidence"

        elif "concept" in explanation_prompt.lower():
            if sparsity > 0.8:
                return f"High sparsity ({sparsity:.2f}) suggests specific concept activation"
            elif sparsity < 0.3:
                return (
                    f"Low sparsity ({sparsity:.2f}) suggests broad concept activation"
                )
            else:
                return f"Moderate sparsity ({sparsity:.2f}) suggests balanced concept processing"

        else:
            return f"Activation pattern: norm={norm:.2f}, mean={mean:.3f}, sparsity={sparsity:.2f}"

    except Exception as e:
        return f"Analysis error: {e}"


def analyze_entity_trajectories(
    model, tokenizer, prompt: str, entities: List[str], max_tokens: int = 30
) -> Dict[str, Any]:
    """Analyze entity probability trajectories during generation."""
    return perform_patchscope_analysis(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_entities=entities,
        max_tokens=max_tokens,
        target_layers=None,
        explanation_prompts=["What entity is being considered?"],
    )


def analyze_llm_hallucinations_with_patchscopes(
    model,
    tokenizer,
    prompt: str,
    suspected_hallucination: Optional[str] = None,
    entities_of_interest: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Main interface for hallucination analysis using patchscopes."""
    return perform_patchscope_analysis(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_entities=entities_of_interest or [],
        explanation_prompts=[
            "Is this information factually correct?",
            "What reasoning led to this prediction?",
            "What is the model's confidence level?",
            "Are there any inconsistencies in the reasoning?",
        ],
        **kwargs,
    )
