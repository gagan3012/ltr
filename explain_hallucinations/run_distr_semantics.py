# Example usage
from transformers import AutoModelForCausalLM, AutoTokenizer
from ltr.dst import DistributionalSemanticsTracer

# Load model and tokenizer
model_name = "gpt2-medium"  # or any model you want to analyze
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the DST tracer
tracer = DistributionalSemanticsTracer(model, tokenizer)

# Run analysis with enhanced visualizations
result = tracer.run_analysis(
    prompt="I am at a concert. I see a bass. Is it a fish? Answer in yes or no:",
    factual_prompt="I am at a concert. I see a bass. It is a musical instrument.",
    concept_examples=["instrument", "guitar", "music", "fish", "animal", "concert"],
    hallucinated_output="yes",  # The incorrect answer (it's not a fish at a concert)
    run_intervention=True,
    enhanced_viz=True,  # Enable enhanced visualizations
)

print("Analysis complete!")
print(f"Generated visualizations:")
for viz_file in result.semantic_drift_trajectory.get("visualization_files", []):
    print(f" - {viz_file}")

# Display information about the hallucination
print("\nHallucination Analysis:")
print(f"Top spurious spans related to 'yes':")
for span in result.spurious_spans[:3]:
    print(f" - '{span['text']}' (score: {span['score']:.4f})")

if result.intervention_results:
    print("\nIntervention Results:")
    if "span_removal" in result.intervention_results:
        print(
            f"After removing '{result.intervention_results['span_removal']['removed_span']}':"
        )
        print(
            f" - New prediction: {result.intervention_results['span_removal']['predicted_next']}"
        )

    if "critical_layer_patching" in result.intervention_results:
        print(
            f"After patching critical layer {result.intervention_results['critical_layer_patching']['layer']}:"
        )
        print(
            f" - New prediction: {result.intervention_results['critical_layer_patching']['predicted_next']}"
        )
