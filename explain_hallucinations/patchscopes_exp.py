import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

# Import from LTR library
from ltr.patchscopes import (
    perform_patchscope_analysis,
    analyze_entity_trajectories,
    analyze_llm_hallucinations_with_patchscopes,
)
from ltr.subsequence_analysis import (
    SubsequenceAnalyzer,
    analyze_hallucination_subsequences,
)
from ltr.visualization import plot_token_evolution_curves



def setup_model(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    print("Loading model and tokenizer...")
    # model_name = "meta-llama/Llama-3.2-1B-Instruct"  # A smaller Qwen model for quick loading
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto"
    )

    return model, tokenizer


model, tokenizer = setup_model()


def create_entity_resolution_table(
    model, tokenizer, prompt: str, entities: List[str], generate_length: int = 30
) -> pd.DataFrame:
    """
    Create a table illustrating entity resolution through layers.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        entities: List of entities to track
        generate_length: Maximum generation length

    Returns:
        DataFrame showing entity resolution across layers
    """
    # Get entity trajectory analysis
    entity_results = analyze_entity_trajectories(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        entities=entities,
        max_tokens=generate_length,
    )

    # Extract generations and entity traces
    generations = entity_results["generations"]
    entity_traces = entity_results["entity_traces"]

    # Create table rows
    rows = []

    # Process each generation step
    for i, gen in enumerate(generations):
        explanation = ""

        # Determine which entity is most active at this step
        if i < len(entity_traces):
            top_entity = max(entity_traces[i].items(), key=lambda x: x[1])[0]
            confidence = max(entity_traces[i].values())
            explanation = f"References '{top_entity}' with {confidence:.2f} confidence"

        rows.append(
            {
                "Layer": i + 1,
                "Generation": gen,
                "Explanation": explanation,
                "Context Integration": f"{min(100, (i + 1) * 10)}%",
            }
        )

    return pd.DataFrame(rows)


def visualize_hallucinations(
    df: pd.DataFrame, model, tokenizer
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Visualize why LLMs hallucinate using the provided dataframe.

    Args:
        df: DataFrame containing examples of hallucination cases
        model: The language model
        tokenizer: The tokenizer

    Returns:
        Two matplotlib figures explaining hallucinations
    """
    # Prepare data for analysis
    analyzer = SubsequenceAnalyzer(model, tokenizer)

    # Process a few examples to show context-dependent entity resolution
    results = []
    for _, row in df.head(4).iterrows():
        prompt = row["prompt"]
        cue = row["cue"]
        actual = row["verbalization"]
        counterfactual = row["counterfactual_verbalization"]

        # Analyze why the model might hallucinate for this example
        analysis = analyze_llm_hallucinations_with_patchscopes(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            suspected_hallucination=actual if not row["label"] else None,
            entities_of_interest=[actual, counterfactual],
        )

        results.append(
            {
                "prompt": prompt,
                "cue": cue,
                "correct_answer": actual if row["label"] else counterfactual,
                "hallucinated_answer": counterfactual if row["label"] else actual,
                "confidence": analysis.get("confidence_scores", [0])[0],
                "entity_confusion": analysis.get("hallucination_patterns", {}).get(
                    "entity_confusion", False
                ),
            }
        )

    # Create first figure: Context-dependent entity resolution
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    results_df = pd.DataFrame(results)

    sns.barplot(x="cue", y="confidence", hue="prompt", data=results_df, ax=ax1)
    ax1.set_title("Context-Dependent Entity Resolution")
    ax1.set_ylabel("Model Confidence")
    ax1.set_xlabel("Ambiguous Entity")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Create second figure: Subsequence influence on hallucination
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    # Select a problematic example where the model hallucinated
    example = df[df["label"] == False].iloc[0]

    # Analyze the subsequences that led to hallucination
    subseq_results = analyze_hallucination_subsequences(
        model=model,
        tokenizer=tokenizer,
        prompt=example["prompt"],
        target_string=example["verbalization"],
        num_perturbations=50,
    )
    print(subseq_results)

    # Extract subsequence influence scores
    if "subsequence_scores" in subseq_results:
        subseqs = list(subseq_results["subsequence_scores"].keys())
        scores = list(subseq_results["subsequence_scores"].values())

        # Sort by influence
        sorted_indices = np.argsort(scores)[::-1]
        top_subseqs = [subseqs[i] for i in sorted_indices[:5]]
        top_scores = [scores[i] for i in sorted_indices[:5]]

        sns.barplot(x=top_scores, y=top_subseqs, ax=ax2)
        ax2.set_title("Subsequences That Trigger Hallucinations")
        ax2.set_xlabel("Influence Score")
        ax2.set_ylabel("Subsequence")
    else:
        ax2.text(
            0.5,
            0.5,
            "Insufficient data for subsequence analysis",
            horizontalalignment="center",
            verticalalignment="center",
        )

    plt.tight_layout()

    return fig1, fig2


def analyze_open_ended_interpretations(
    model, tokenizer, df: pd.DataFrame, target_layers: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Analyze how the model's interpretation of polysemous entities evolves across layers.

    Implements the open-ended patchscope technique inspired by SelfIE that extracts
    interpretations of entity representations from different layers.

    Args:
        model: The language model
        tokenizer: The tokenizer
        df: DataFrame with (prompt, cue, label, verbalization, counterfactual_verbalization)
        target_layers: Specific layers to analyze (if None, will use even layers)

    Returns:
        DataFrame showing interpretations across layers
    """
    # Determine target layers if not provided
    if target_layers is None:
        if hasattr(model.config, "n_layer"):
            n_layers = model.config.n_layer
        elif hasattr(model.config, "num_hidden_layers"):
            n_layers = model.config.num_hidden_layers
        else:
            n_layers = 12  # Default fallback

        target_layers = list(range(0, n_layers, 2))

    results = []

    for _, row in df.iterrows():
        cue = row["cue"]
        prompt = row["prompt"]
        label = row["label"]
        true_sense = (
            row["verbalization"] if label else row["counterfactual_verbalization"]
        )

        # For each layer, generate an interpretation
        for layer in target_layers:
            # Create the interpretation prompt
            interpretation_prompt = f"Tell me about {cue}"

            # Use patchscopes to get layer-specific analysis
            analysis = perform_patchscope_analysis(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_layers=[layer],
                explanation_prompts=[f"What does the word '{cue}' refer to?"],
                target_entities=[cue],
                max_tokens=15,
            )

            # Extract the interpretation
            if "layer_explanations" in analysis and analysis["layer_explanations"]:
                explanation = analysis["layer_explanations"][0].get("explanation", "")
                interpretation = (
                    f"Sure! In this context, the word refers to {explanation}"
                )
            else:
                # Fallback if patchscope doesn't provide explanations
                input_ids = tokenizer(
                    f"{prompt} The word {cue} refers to", return_tensors="pt"
                ).input_ids

                with torch.no_grad():
                    input_ids = input_ids.to(model.device)
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=150,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                interpretation = tokenizer.decode(
                    outputs[0][input_ids.shape[1] :], skip_special_tokens=True
                )
                interpretation = (
                    f"Sure! In this context, the word refers to {interpretation}"
                )

            # Auto-determine if interpretation matches correct sense
            # In practice, we'd use a separate LLM to score this as mentioned in the paper
            correct_interpretation = true_sense.lower() in interpretation.lower()

            print(f"Interpretation for '{cue}' in layer {layer}:")
            print(interpretation)

            results.append(
                {
                    "Word": cue,
                    "Sense": true_sense,
                    "Layer": layer,
                    "Interpretation": interpretation,
                    "Has_Distractor": "distractor" in prompt.lower(),
                    "Correct_Answer": label,
                    "Correct_Interpretation": correct_interpretation,
                }
            )

    return pd.DataFrame(results)


def display_interpretation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format interpretation results into a nice table."""
    # Create a summary table like the example provided
    table_data = []

    for _, row in df.iterrows():
        # Bold the relevant part of the interpretation that indicates the sense
        interpretation = row["Interpretation"]

        # Add to table
        table_data.append(
            {
                "Word": row["Word"],
                "Sense": row["Sense"],
                "Layer": row["Layer"],
                "Interpretation": interpretation,
            }
        )

    return pd.DataFrame(table_data)


def analyze_contextualization_accuracy(
    interpretation_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Analyze cumulative accuracy of word sense interpretation across layers.

    Args:
        interpretation_df: DataFrame with interpretations across layers

    Returns:
        Dictionary with accuracy metrics and plots
    """
    # Group by word and layer
    layers = sorted(interpretation_df["Layer"].unique())

    # Initialize tracking for cumulative accuracy
    cum_acc_no_distractor = []
    cum_acc_with_distractor = []
    cum_acc_correct = []
    cum_acc_incorrect = []

    # For each layer, calculate cumulative accuracy
    for layer in layers:
        # No distractor vs with distractor
        no_distractor_df = interpretation_df[
            (~interpretation_df["Has_Distractor"])
            & (interpretation_df["Layer"] <= layer)
        ]
        with_distractor_df = interpretation_df[
            (interpretation_df["Has_Distractor"])
            & (interpretation_df["Layer"] <= layer)
        ]

        # Calculate if correct interpretation has been achieved by this layer
        no_distractor_correct = no_distractor_df.groupby("Word")[
            "Correct_Interpretation"
        ].any()
        with_distractor_correct = with_distractor_df.groupby("Word")[
            "Correct_Interpretation"
        ].any()

        cum_acc_no_distractor.append(
            no_distractor_correct.mean() if not no_distractor_correct.empty else 0
        )
        cum_acc_with_distractor.append(
            with_distractor_correct.mean() if not with_distractor_correct.empty else 0
        )

        # Correct answer vs incorrect answer (for cases with distractors)
        correct_cases_df = interpretation_df[
            (interpretation_df["Has_Distractor"])
            & (interpretation_df["Correct_Answer"])
            & (interpretation_df["Layer"] <= layer)
        ]
        incorrect_cases_df = interpretation_df[
            (interpretation_df["Has_Distractor"])
            & (~interpretation_df["Correct_Answer"])
            & (interpretation_df["Layer"] <= layer)
        ]

        correct_cases_correct = correct_cases_df.groupby("Word")[
            "Correct_Interpretation"
        ].any()
        incorrect_cases_correct = incorrect_cases_df.groupby("Word")[
            "Correct_Interpretation"
        ].any()

        cum_acc_correct.append(
            correct_cases_correct.mean() if not correct_cases_correct.empty else 0
        )
        cum_acc_incorrect.append(
            incorrect_cases_correct.mean() if not incorrect_cases_correct.empty else 0
        )

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot distractor vs. no distractor
    ax1.plot(layers, cum_acc_no_distractor, label="No Distractor")
    ax1.plot(layers, cum_acc_with_distractor, label="With Distractor")
    ax1.set_title("Effect of Distractors on Contextualization")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Cumulative Accuracy")
    ax1.legend()

    # Plot correct vs. incorrect
    ax2.plot(layers, cum_acc_correct, label="Correct Answer")
    ax2.plot(layers, cum_acc_incorrect, label="Incorrect Answer")
    ax2.set_title("Contextualization in Correct vs. Incorrect Cases")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Cumulative Accuracy")
    ax2.legend()

    return {
        "figure": fig,
        "cumulative_accuracy": {
            "no_distractor": cum_acc_no_distractor,
            "with_distractor": cum_acc_with_distractor,
            "correct": cum_acc_correct,
            "incorrect": cum_acc_incorrect,
        },
    }


def analyze_entity_contextualization(
    model, tokenizer, df: pd.DataFrame
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Analyze how entities get contextualized across model layers.

    Args:
        model: The language model
        tokenizer: The tokenizer
        df: DataFrame with examples

    Returns:
        Tuple of (interpretation table, cumulative accuracy figure)
    """
    # Generate layer-by-layer interpretations
    interpretations = analyze_open_ended_interpretations(model, tokenizer, df)

    # Format into a display table
    table = display_interpretation_table(interpretations)

    # Analyze contextualization accuracy
    accuracy_results = analyze_contextualization_accuracy(interpretations)

    print("Entity Contextualization Analysis:")
    print(table)

    return table, accuracy_results["figure"]


def main():
    # df = pd.DataFrame(
    #     data,
    #     columns=[
    #         "cue",
    #         "prompt",
    #         "stringlengths",
    #         "label",
    #         "verbalization",
    #         "counterfactual_verbalization",
    #     ],
    # )

    df = pd.read_parquet(
        "hf://datasets/gagan3012/HallData/data/train-00000-of-00001.parquet"
    ).tail(2)

    # Load model and tokenizer
    # model, tokenizer = load_model_and_tokenizer(model_name="meta-llama/Llama-3.2-1B-Instruct")

    # NEW: Analyze entity contextualization
    interp_table, interp_fig = analyze_entity_contextualization(model, tokenizer, df)

    print("Entity Contextualization Interpretations:")
    print(interp_table)

    print(interp_table["Interpretation"])

    # Save interpretation figure
    interp_fig.savefig("entity_contextualization.png")

    # Create hallucination visualizations
    fig1, fig2 = visualize_hallucinations(df, model, tokenizer)

    # Save figures
    fig1.savefig("context_dependent_entity_resolution.png")
    fig2.savefig("hallucination_triggers.png")

    print(
        "Visualizations saved to 'context_dependent_entity_resolution.png', 'hallucination_triggers.png', and 'entity_contextualization.png'"
    )


if __name__ == "__main__":
    main()
