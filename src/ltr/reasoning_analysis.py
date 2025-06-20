import numpy as np
from typing import List, Dict, Tuple
from ltr.concept_extraction import extract_concept_activations
from ltr.causal_intervention import perform_causal_intervention

def analyze_reasoning_paths(
    model,
    tokenizer,
    prompt: str,
    potential_paths: List[List[str]],
    concept_threshold: float = 0.2,
    use_causal_analysis: bool = True,
    use_attention_analysis: bool = True,
) -> Dict:
    """
    Analyze potential reasoning paths using both layer and position information.

    Parameters:
    -----------
    model : PreTrainedModel
        The transformer model to analyze
    tokenizer : PreTrainedTokenizer
        The tokenizer corresponding to the model
    prompt : str
        The input text prompt
    potential_paths : List[List[str]]
        List of possible reasoning paths, where each path is a list of concepts
    concept_threshold : float
        Threshold for concept activation significance
    use_causal_analysis : bool
        Whether to use causal intervention to improve path scoring
    use_attention_analysis : bool
        Whether to analyze attention patterns between concepts

    Returns:
    --------
    Dict
        Analysis results including best path and path scores
    """
    # For backward compatibility with existing code that passes a HookedTransformer
    if hasattr(model, "to_str_tokens") and hasattr(model, "run_with_cache"):
        return _analyze_reasoning_paths_with_transformerlens(
            model,
            prompt,
            potential_paths,
            concept_threshold,
            use_causal_analysis,
            use_attention_analysis,
        )

    all_concepts = set(c for path in potential_paths for c in path)
    results = extract_concept_activations(
        model,
        tokenizer,
        prompt,
        intermediate_concepts=list(all_concepts),
        final_concepts=[],
    )

    # Initialize results structure
    path_results = {
        "prompt": prompt,
        "potential_paths": potential_paths,
        "path_scores": [],
        "path_details": [],
        "concept_results": results,
        "causal_strengths": {},
        "attention_patterns": {},
    }

    # Perform causal analysis if requested
    causal_strengths = {}
    if use_causal_analysis and len(all_concepts) > 1:
        try:
            # Get token positions for causal intervention
            token_positions = _get_concept_token_positions(results)
            if token_positions:
                # Perform causal interventions between concepts
                causal_results = perform_causal_intervention(
                    model,
                    tokenizer,
                    prompt,
                    list(all_concepts),
                    target_positions=list(token_positions.values()),
                    patch_positions=list(token_positions.values()),
                )

                # Extract causal strengths between concepts
                causal_strengths = _extract_causal_relationships(
                    causal_results, token_positions, list(all_concepts)
                )
                path_results["causal_strengths"] = causal_strengths
        except Exception as e:
            print(f"Causal analysis failed: {e}")

    # Analyze attention patterns if requested
    attention_patterns = {}
    if use_attention_analysis:
        try:
            # Extract attention patterns between concepts
            attention_patterns = _extract_attention_patterns(
                model, tokenizer, prompt, list(all_concepts)
            )
            path_results["attention_patterns"] = attention_patterns
        except Exception as e:
            print(f"Attention analysis failed: {e}")

    # Analyze each path
    for path in potential_paths:
        # Check if all concepts in the path are found
        concepts_found = [
            concept for concept in path if results["activations"].get(concept)
        ]

        if len(concepts_found) < len(path):
            missing = set(path) - set(concepts_found)
            path_results["path_scores"].append(
                {
                    "path": path,
                    "score": 0.0,
                    "complete": False,
                    "missing_concepts": list(missing),
                }
            )
            continue

        # Find peak activations for each concept
        concept_peaks = []
        for concept in path:
            if not results["activations"].get(concept):
                continue

            peak = max(results["activations"][concept], key=lambda x: x["probability"])
            concept_peaks.append(
                {
                    "concept": concept,
                    "position": peak["position"],
                    "peak_layer": peak["layer"],
                    "peak_prob": peak["probability"],
                    "token": peak["context_token"],
                }
            )

        if not concept_peaks:
            continue

        # Calculate enhanced path score
        path_score, score_details = _calculate_enhanced_path_score(
            path, concept_peaks, concept_threshold, causal_strengths, attention_patterns
        )

        path_results["path_scores"].append(
            {
                "path": path,
                "score": path_score,
                "complete": True,
                "score_details": score_details,
            }
        )

        path_results["path_details"].append(
            {"path": path, "concept_peaks": concept_peaks}
        )

    # Sort paths by score
    path_results["path_scores"].sort(key=lambda x: x["score"], reverse=True)

    # Determine best path
    if path_results["path_scores"]:
        path_results["best_path"] = path_results["path_scores"][0]["path"]
        path_results["best_path_score"] = path_results["path_scores"][0]["score"]
        path_results["best_path_details"] = path_results["path_scores"][0].get(
            "score_details", {}
        )
    else:
        path_results["best_path"] = []
        path_results["best_path_score"] = 0.0
        path_results["best_path_details"] = {}

    return path_results


def _calculate_enhanced_path_score(
    path, concept_peaks, concept_threshold, causal_strengths={}, attention_patterns={}
) -> Tuple[float, Dict]:
    """
    Calculate an enhanced path score using multiple factors:
    1. Position ordering (weighted higher)
    2. Layer ordering
    3. Activation strength
    4. Causal relationships between concepts
    5. Attention patterns
    6. Consistency of activations
    7. Final answer confidence
    """
    score_details = {}

    # Check position and layer ordering
    position_order = all(
        concept_peaks[i]["position"] <= concept_peaks[i + 1]["position"]
        for i in range(len(concept_peaks) - 1)
    )
    layer_order = all(
        concept_peaks[i]["peak_layer"] <= concept_peaks[i + 1]["peak_layer"]
        for i in range(len(concept_peaks) - 1)
    )

    # Position ordering is weighted higher (0.7) than layer ordering (0.3)
    order_score = (0.7 if position_order else 0.0) + (0.3 if layer_order else 0.0)
    score_details["order_score"] = order_score

    # Calculate activation strength score
    avg_prob = sum(peak["peak_prob"] for peak in concept_peaks) / len(concept_peaks)
    activation_score = min(avg_prob / concept_threshold, 1.0)
    score_details["activation_score"] = activation_score

    # NEW: Calculate final concept strength separately
    # This is crucial for distinguishing between contradictory answers
    print(concept_peaks)
    if len(concept_peaks) > 0:
        final_concept = concept_peaks[-1]
        final_concept_score = final_concept["peak_prob"]

        # Examine if this is a YES/NO or similar contradictory answer
        final_concept_name = final_concept["concept"].upper()
        is_conclusion = final_concept_name in [
            "YES",
            "NO",
            "TRUE",
            "FALSE",
            "CORRECT",
            "INCORRECT",
        ]

        # Store these new metrics
        score_details["final_concept_score"] = final_concept_score
        score_details["is_conclusion"] = is_conclusion
    else:
        final_concept_score = 0.0
        is_conclusion = False

    # Evaluate position gaps - penalize large gaps between consecutive concepts
    position_gaps = []
    for i in range(len(concept_peaks) - 1):
        gap = concept_peaks[i + 1]["position"] - concept_peaks[i]["position"]
        position_gaps.append(gap)

    avg_gap = sum(position_gaps) / len(position_gaps) if position_gaps else 0
    gap_penalty = max(0, 1.0 - (0.1 * avg_gap))  # Penalty increases with gap size
    score_details["gap_penalty"] = gap_penalty

    # Evaluate causal relationships if available
    causal_score = 0.0
    causal_confidence = 0.0
    if causal_strengths:
        causal_links = []
        causal_coverage = 0
        expected_links = len(path) - 1  # Number of expected links in a complete path

        # Check for direct links (adjacent concepts)
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            if key in causal_strengths:
                # Apply position-based weighting - later links are more important
                link_weight = (
                    0.5 + (0.5 * (i / (len(path) - 2))) if len(path) > 2 else 1.0
                )
                causal_links.append(causal_strengths[key] * link_weight)
                causal_coverage += 1

        # If we're missing direct links, check for transitive relationships (A→C through A→B→C)
        if causal_coverage < expected_links:
            for i in range(len(path) - 2):
                for j in range(
                    i + 2, min(i + 4, len(path))
                ):  # Look ahead up to 3 steps
                    key = (path[i], path[j])
                    if key in causal_strengths:
                        # Discount transitive links by distance
                        discount = 1.0 / (j - i)
                        causal_links.append(causal_strengths[key] * discount)
                        # Only count as partial coverage
                        causal_coverage += discount * 0.5

        # Calculate coverage ratio to estimate confidence
        causal_confidence = (
            causal_coverage / expected_links if expected_links > 0 else 0
        )

        if causal_links:
            # Use weighted harmonic mean instead of arithmetic mean
            # This ensures all links need to be strong for a high score
            weights = np.linspace(
                0.5, 1.0, len(causal_links)
            )  # Later links have higher weight
            causal_score = (
                np.average(causal_links, weights=weights)
                if len(causal_links) > 0
                else 0
            )

            # Apply a penalty for incomplete causal chains
            causal_score = causal_score * (0.5 + 0.5 * causal_confidence)

        score_details["causal_score"] = causal_score
        score_details["causal_confidence"] = causal_confidence
        score_details["causal_coverage"] = f"{causal_coverage}/{expected_links} links"

    # Evaluate attention patterns with improved methodology
    attention_score = 0.0
    attention_confidence = 0.0
    if attention_patterns:
        attention_links = []
        attention_coverage = 0
        expected_attention_links = len(path) - 1

        # Primary analysis: check consecutive concepts
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            if key in attention_patterns:
                # Apply stronger weights to key transitions
                # Early concepts that influence final answer get higher weight
                importance_weight = 1.0
                if i == 0 or i == len(path) - 2:  # First or last transition
                    importance_weight = 1.5

                attention_links.append(attention_patterns[key] * importance_weight)
                attention_coverage += 1

        # Secondary analysis: check for attention to early concepts from late concepts
        if len(path) > 2:
            # Check if final concept attends to earlier concepts (sign of good reasoning)
            final_concept = path[-1]
            for i in range(len(path) - 2):
                key = (path[i], final_concept)
                if key in attention_patterns:
                    # Weight by distance - closer concepts should have stronger attention
                    distance_weight = 0.5 * (1.0 / (len(path) - i - 1))
                    attention_links.append(attention_patterns[key] * distance_weight)
                    attention_coverage += (
                        0.25  # Partial credit for non-adjacent attention
                    )

        # Calculate confidence based on coverage
        attention_confidence = (
            attention_coverage / expected_attention_links
            if expected_attention_links > 0
            else 0
        )

        if attention_links:
            # Use higher percentile instead of mean to reward paths with some very strong attention
            if len(attention_links) >= 3:
                # Use 75th percentile for paths with multiple links to reward strong attention
                attention_score = np.percentile(attention_links, 75)
            else:
                # Use mean for paths with few links
                attention_score = sum(attention_links) / len(attention_links)

            # Apply confidence scaling
            attention_score = attention_score * (0.7 + 0.3 * attention_confidence)

        score_details["attention_score"] = attention_score
        score_details["attention_confidence"] = attention_confidence
        score_details["attention_coverage"] = (
            f"{attention_coverage}/{expected_attention_links} links"
        )

    # Combine all factors with appropriate weights
    base_weight = 0.4
    causal_weight = 0.2 if causal_strengths else 0
    attn_weight = 0.15 if attention_patterns else 0
    gap_weight = 0.05
    final_concept_weight = 0.2

    # Normalize weights to sum to 1
    total_weight = (
        base_weight
        + base_weight
        + causal_weight
        + attn_weight
        + gap_weight
        + final_concept_weight
    )
    base_weight /= total_weight
    causal_weight /= total_weight
    attn_weight /= total_weight
    gap_weight /= total_weight
    final_concept_weight /= total_weight

    # Calculate final score with the new component
    final_score = (
        base_weight * order_score
        + base_weight * activation_score
        + causal_weight * causal_score
        + attn_weight * attention_score
        + gap_weight * gap_penalty
        + final_concept_weight * final_concept_score
    )

    score_details["final_score"] = final_score

    # Add explanation for scoring
    factors = []
    if order_score > 0:
        factors.append(f"concept ordering ({order_score:.2f})")
    if activation_score > 0:
        factors.append(f"activation strength ({activation_score:.2f})")
    if causal_score > 0:
        factors.append(f"causal influence ({causal_score:.2f})")
    if attention_score > 0:
        factors.append(f"attention pattern ({attention_score:.2f})")
    if final_concept_score > 0:
        factors.append(f"final concept confidence ({final_concept_score:.2f})")

    score_details["explanation"] = (
        f"Path {path} scored {final_score:.2f} based on " + ", ".join(factors)
    )

    print(f"Path {path} scored {final_score} based on " + ", ".join(factors))

    return final_score, score_details


def _get_concept_token_positions(concept_results):
    """Extract the position of each concept's peak activation"""
    token_positions = {}

    for concept, activations in concept_results["activations"].items():
        if activations:
            # Find the peak activation for this concept
            peak_activation = max(activations, key=lambda x: x["probability"])
            # Store the position - add 1 to account for first token in actual input
            token_positions[concept] = peak_activation["position"] + 1

    return token_positions


def _extract_causal_relationships(causal_results, token_positions, all_concepts):
    """Extract causal strengths between pairs of concepts"""
    concept_pairs = {}

    # Create mapping from position to concept
    pos_to_concept = {pos: concept for concept, pos in token_positions.items()}

    # For each target concept, find how much it's affected by other concepts
    for concept in all_concepts:
        if concept not in causal_results["token_importance"]:
            continue

        for impact_info in causal_results["token_importance"][concept]:
            position = impact_info["position"]
            if position in pos_to_concept:
                source_concept = pos_to_concept[position]
                # We only care about the impact between distinct concepts
                if source_concept != concept:
                    # Normalize the impact to range [0, 1]
                    impact = impact_info["impact"]
                    normalized_impact = min(1.0, max(0.0, abs(impact)))
                    concept_pairs[(source_concept, concept)] = normalized_impact

    return concept_pairs


def _extract_attention_patterns(model, tokenizer, prompt, concepts):
    """Analyze attention patterns between concepts"""
    attention_patterns = {}

    # This is a placeholder for a more complex attention analysis
    # Ideally, we would get the attention weights and analyze how strongly
    # the model attends to earlier concepts when generating later ones

    try:
        # Only attempt this for models with attention mask and outputs
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs, output_attentions=True)

        if hasattr(outputs, "attentions") and outputs.attentions:
            attentions = outputs.attentions  # This is a tuple of attention tensors

            # Get token ids for concepts
            concept_tokens = {}
            for concept in concepts:
                try:
                    tokens = tokenizer.encode(" " + concept, add_special_tokens=False)
                    if tokens:
                        concept_tokens[concept] = tokens[0]
                except Exception:
                    pass

            # Find positions where concepts appear in the input
            input_ids = inputs.input_ids[0].tolist()
            concept_positions = {}
            for concept, token_id in concept_tokens.items():
                if token_id in input_ids:
                    # Find all positions (might appear multiple times)
                    positions = [i for i, t in enumerate(input_ids) if t == token_id]
                    if positions:
                        concept_positions[concept] = positions

            # Calculate attention flows between concept pairs
            for i, concept1 in enumerate(concepts):
                for j, concept2 in enumerate(concepts):
                    if i >= j:  # Skip self-attention and reversed pairs
                        continue

                    if concept1 in concept_positions and concept2 in concept_positions:
                        pos1 = concept_positions[concept1][0]  # Use first occurrence
                        pos2 = concept_positions[concept2][0]

                        # Calculate average attention from concept2 to concept1 across layers
                        # This measures how much the model focuses on concept1 when generating concept2
                        avg_attention = 0.0
                        count = 0

                        for layer_attns in attentions:
                            # layer_attns shape: [batch, heads, seq_len, seq_len]
                            if pos2 < layer_attns.size(
                                2
                            ):  # Make sure position is valid
                                # Average across attention heads
                                attn_score = layer_attns[0, :, pos2, pos1].mean().item()
                                avg_attention += attn_score
                                count += 1

                        if count > 0:
                            avg_attention /= count
                            attention_patterns[(concept1, concept2)] = avg_attention
    except Exception as e:
        print(f"Error analyzing attention patterns: {e}")

    return attention_patterns


def _analyze_reasoning_paths_with_transformerlens(
    model,
    prompt,
    potential_paths,
    concept_threshold=0.2,
    use_causal_analysis=True,
    use_attention_analysis=True,
):
    """Legacy function to maintain backward compatibility with TransformerLens models"""
    all_concepts = set(c for path in potential_paths for c in path)
    results = extract_concept_activations(
        model, prompt, intermediate_concepts=list(all_concepts), final_concepts=[]
    )

    # Initialize results structure
    path_results = {
        "prompt": prompt,
        "potential_paths": potential_paths,
        "path_scores": [],
        "path_details": [],
        "concept_results": results,
        "causal_strengths": {},
        "attention_patterns": {},
    }

    # Perform causal analysis if requested and TransformerLens has the functionality
    causal_strengths = {}
    if use_causal_analysis and len(all_concepts) > 1:
        try:
            # Get token positions for causal intervention
            token_positions = _get_concept_token_positions(results)
            if token_positions:
                # Perform causal interventions between concepts
                causal_results = perform_causal_intervention(
                    model,
                    prompt,
                    list(all_concepts),
                    target_positions=list(token_positions.values()),
                    patch_positions=list(token_positions.values()),
                )

                # Extract causal strengths between concepts
                causal_strengths = _extract_causal_relationships(
                    causal_results, token_positions, list(all_concepts)
                )
                path_results["causal_strengths"] = causal_strengths
        except Exception as e:
            print(f"Causal analysis failed with TransformerLens: {e}")

    # Analyze attention patterns if requested
    attention_patterns = {}
    if use_attention_analysis:
        try:
            # Run the model to get attention patterns
            tokens = model.to_tokens(prompt)
            _, cache = model.run_with_cache(tokens)

            # Get token ids for concepts
            concept_tokens = {}
            for concept in all_concepts:
                try:
                    concept_tokens[concept] = model.to_single_token(" " + concept)
                except Exception:
                    pass

            # Find positions where concepts appear in the input
            input_ids = tokens[0].tolist()
            concept_positions = {}
            for concept, token_id in concept_tokens.items():
                if token_id in input_ids:
                    positions = [i for i, t in enumerate(input_ids) if t == token_id]
                    if positions:
                        concept_positions[concept] = positions

            # Calculate attention flows between concept pairs
            for i, concept1 in enumerate(all_concepts):
                for j, concept2 in enumerate(all_concepts):
                    if i >= j:  # Skip self-attention and reversed pairs
                        continue

                    if concept1 in concept_positions and concept2 in concept_positions:
                        pos1 = concept_positions[concept1][0]  # Use first occurrence
                        pos2 = concept_positions[concept2][0]

                        # Calculate average attention from concept2 to concept1 across layers
                        avg_attention = 0.0
                        count = 0

                        for layer in range(model.cfg.n_layers):
                            if f"blocks.{layer}.attn.hook_attn" in cache:
                                attn = cache[f"blocks.{layer}.attn.hook_attn"]
                                if pos2 < attn.size(2):  # Make sure position is valid
                                    # Average across attention heads
                                    attn_score = attn[0, :, pos2, pos1].mean().item()
                                    avg_attention += attn_score
                                    count += 1

                        if count > 0:
                            avg_attention /= count
                            attention_patterns[(concept1, concept2)] = avg_attention

            path_results["attention_patterns"] = attention_patterns
        except Exception as e:
            print(f"Attention analysis failed with TransformerLens: {e}")

    # Analyze each path
    for path in potential_paths:
        # Check if all concepts in the path are found
        concepts_found = [
            concept for concept in path if results["activations"].get(concept)
        ]

        if len(concepts_found) < len(path):
            missing = set(path) - set(concepts_found)
            path_results["path_scores"].append(
                {
                    "path": path,
                    "score": 0.0,
                    "complete": False,
                    "missing_concepts": list(missing),
                }
            )
            continue

        # Find peak activations for each concept
        concept_peaks = []
        for concept in path:
            if not results["activations"].get(concept):
                continue

            peak = max(results["activations"][concept], key=lambda x: x["probability"])
            concept_peaks.append(
                {
                    "concept": concept,
                    "position": peak["position"],
                    "peak_layer": peak["layer"],
                    "peak_prob": peak["probability"],
                    "token": peak.get("context_token", ""),
                }
            )

        if not concept_peaks:
            continue

        # Calculate enhanced path score
        path_score, score_details = _calculate_enhanced_path_score(
            path, concept_peaks, concept_threshold, causal_strengths, attention_patterns
        )

        path_results["path_scores"].append(
            {
                "path": path,
                "score": path_score,
                "complete": True,
                "score_details": score_details,
            }
        )

        path_results["path_details"].append(
            {"path": path, "concept_peaks": concept_peaks}
        )

    # Sort paths by score
    path_results["path_scores"].sort(key=lambda x: x["score"], reverse=True)

    # Determine best path
    if path_results["path_scores"]:
        path_results["best_path"] = path_results["path_scores"][0]["path"]
        path_results["best_path_score"] = path_results["path_scores"][0]["score"]
        path_results["best_path_details"] = path_results["path_scores"][0].get(
            "score_details", {}
        )
    else:
        path_results["best_path"] = []
        path_results["best_path_score"] = 0.0
        path_results["best_path_details"] = {}

    return path_results