# Reproducibility Guide: Distributional Semantics Tracing (DST) Experiments

This guide provides instructions for reproducing the experiments described in the thesis using the Distributional Semantics Tracing (DST) framework. All code examples are based on the existing `ltr` codebase.

## Setup

### Installation

```bash
# Install the ltr package
uv pip install git+https://github.com/gagan3012/ltr.git
```

## Qualitative Comparison of Explainability Methods

To reproduce the qualitative analysis of the five representative hallucination cases:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from ltr.dst import DistributionalSemanticsTracer

# Load model
model_name = "Qwen/Qwen3-0.6B"  # Can be replaced with "google/gemma2-2b", "google/gemma2-9b", or "allenai/OLMo2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize DST tracer
tracer = DistributionalSemanticsTracer(model, tokenizer)

# Example 1: Australia case
result_australia = tracer.run_analysis(
    prompt="Answer based on the information provided here. Forget everything you know about geography. The capital city of Australia was just renamed from Canberra to Dublin. Is the capital city of Australia named Dublin?",
    factual_prompt="The capital city of Australia is Canberra.",
    concept_examples=["Australia", "capital", "Canberra", "Dublin", "renamed"],
    hallucinated_output="Canberra",
    run_intervention=True,
    enhanced_viz=True
)

# Example 2: Bank ambiguity case
result_bank = tracer.run_analysis(
    prompt="John is going fishing, so he walks over to the river bank. Can he withdraw money at the bank?",
    factual_prompt="A river bank is a geographical feature, not a financial institution.",
    concept_examples=["bank", "river", "money", "withdraw", "fishing"],
    hallucinated_output="Yes",
    run_intervention=True,
    enhanced_viz=True
)

# Example 3: Bass ambiguity case
result_bass = tracer.run_analysis(
    prompt="Please answer succinctly. I am at a concert. I see a bass. Is it a fish?",
    factual_prompt="A bass at a concert is a musical instrument, not a fish.",
    concept_examples=["bass", "concert", "instrument", "fish", "music"],
    hallucinated_output="Yes",
    run_intervention=True,
    enhanced_viz=True
)

# Example 4: Java ambiguity case
result_java = tracer.run_analysis(
    prompt="Please answer succinctly. I am coding. I see some java. Is it a programming language?",
    factual_prompt="Java in a coding context is a programming language.",
    concept_examples=["java", "coding", "programming", "language", "coffee"],
    hallucinated_output="coffee",
    run_intervention=True,
    enhanced_viz=True
)

# Example 5: Physician case
result_physician = tracer.run_analysis(
    prompt="Please answer succinctly. The physician is somebody's grandmother. Is the physician a woman?",
    factual_prompt="A grandmother is a woman.",
    concept_examples=["physician", "grandmother", "woman", "doctor", "gender"],
    hallucinated_output="No",
    run_intervention=True,
    enhanced_viz=True
)

# Results will include:
# - Concept importance scores across layers
# - Semantic drift trajectory visualizations
# - Spurious spans responsible for hallucinations
# - Concept network visualizations
# - Activation distribution comparisons
```

## Pinpointing the Onset of Hallucination

To identify critical layers in the hallucination process (prediction onset, semantic inversion point, and commitment layer):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from ltr.dst import DistributionalSemanticsTracer
import matplotlib.pyplot as plt
import numpy as np

# Load model
model_name = "google/gemma2-2b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize tracer
tracer = DistributionalSemanticsTracer(model, tokenizer)

# Example: Bass ambiguity case
prompt = "Please answer succinctly. I am at a concert. I see a bass. Is it a fish?"

# Define correct and incorrect reasoning pathways
correct_pathways = [
    ["concert", "bass", "instrument"],
    ["music", "bass", "instrument"]
]

incorrect_pathways = [
    ["bass", "fish", "animal"],
    ["water", "bass", "fish"]
]

# Calculate DSS scores for each layer
n_layers = tracer.n_layers
dss_scores = []

for layer_idx in range(n_layers):
    dss = tracer.compute_dss(
        prompt=prompt,
        correct_pathways=correct_pathways,
        incorrect_pathways=incorrect_pathways,
        layer_idx=layer_idx
    )
    dss_scores.append(dss)

# Plot DSS across layers to identify critical points
plt.figure(figsize=(10, 6))
plt.plot(range(n_layers), dss_scores, 'o-', linewidth=2)
plt.xlabel('Model Layers')
plt.ylabel('Distributional Semantics Strength (DSS)')
plt.title('Layer-wise DSS Analysis')
plt.grid(True)

# Find critical layers
prediction_onset = np.argmax(np.array(dss_scores) < 0.6)  # DSS starts dropping
semantic_inversion = np.argmax(np.array(dss_scores) < 0.5)  # DSS crosses below 0.5
commitment_layer = np.argmin(dss_scores)  # DSS reaches minimum

# Mark critical layers
plt.scatter(prediction_onset, dss_scores[prediction_onset], color='green', s=100, label='Prediction Onset')
plt.scatter(semantic_inversion, dss_scores[semantic_inversion], color='yellow', s=100, label='Semantic Inversion')
plt.scatter(commitment_layer, dss_scores[commitment_layer], color='red', s=100, label='Commitment Layer')
plt.legend()
plt.savefig("hallucination_onset_analysis.png")
plt.close()

print(f"Critical layers identified:")
print(f"- Prediction Onset Layer: {prediction_onset}")
print(f"- Semantic Inversion Point: {semantic_inversion}")
print(f"- Commitment Layer: {commitment_layer}")

# Generate network visualizations for these critical layers
tracer.visualize_concept_network(
    prompt=prompt,
    spurious_spans=[],
    concept_importance={},
    concept_examples=["concert", "bass", "instrument", "fish", "music", "water", "animal"],
    num_layers=3,  # Only visualize the critical layers
    figsize=(12, 10)
)
```

## Analyzing Intrinsic Causes

To analyze the associative vs. contextual pathways and validate the hypothesis:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from ltr.dst import DistributionalSemanticsTracer
import matplotlib.pyplot as plt
import pandas as pd

# Load model
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize tracer
tracer = DistributionalSemanticsTracer(model, tokenizer)

# 1. Reasoning Shortcut Hijack Example
shortcut_prompt = "The bass player in the jazz band caught a large bass at the lake yesterday."
shortcut_result = tracer.run_analysis(
    prompt=shortcut_prompt,
    factual_prompt="Bass can refer to a musical instrument and a type of fish.",
    concept_examples=["bass", "player", "instrument", "fish", "lake", "jazz"],
    hallucinated_output="fish",
    run_intervention=True,
    enhanced_viz=True
)

# Get concept correlations to visualize associative vs contextual pathways
shortcut_matrix = tracer.create_concept_correlation_matrix(
    prompt=shortcut_prompt,
    concept_examples=["bass", "player", "instrument", "fish", "lake", "jazz"]
)

# 2. Analogical Collapse Example
collapse_prompt = "John is going fishing, so he walks over to the river bank. Can he withdraw money at the bank?"
collapse_result = tracer.run_analysis(
    prompt=collapse_prompt,
    factual_prompt="A river bank is a geographical feature, not a financial institution.",
    concept_examples=["bank", "river", "money", "withdraw", "fishing"],
    hallucinated_output="Yes",
    run_intervention=True,
    enhanced_viz=True
)

# Get concept correlations to visualize associative vs contextual pathways
collapse_matrix = tracer.create_concept_correlation_matrix(
    prompt=collapse_prompt,
    concept_examples=["bank", "river", "money", "withdraw", "fishing"]
)

# 3. Large-scale correlation study simulation
# Note: This is a simplified example - for full reproduction, you'd need to process the entire HALoGEN dataset

# Define example prompts
example_prompts = [
    "I am at a concert. I see a bass. Is it a fish?",
    "John is going fishing, so he walks over to the river bank. Can he withdraw money at the bank?",
    "The physician is somebody's grandmother. Is the physician a woman?",
    "I am coding. I see some java. Is it a programming language?"
]

# Define corresponding ground truths
ground_truths = [False, False, True, True]

# Calculate DSS scores for each example
dss_scores = []
for prompt in example_prompts:
    # Define appropriate pathways for each prompt
    # (This would need to be adjusted for each specific example)
    correct_pathways = [["concert", "bass", "instrument"]]
    incorrect_pathways = [["bass", "fish", "animal"]]
    
    # Calculate DSS
    dss = tracer.compute_dss(
        prompt=prompt,
        correct_pathways=correct_pathways,
        incorrect_pathways=incorrect_pathways
    )
    dss_scores.append(dss)

# Create dataframe with results
df = pd.DataFrame({
    'prompt': example_prompts,
    'hallucination': [not truth for truth in ground_truths],
    'dss': dss_scores
})

# Plot correlation between DSS and hallucination rate
plt.figure(figsize=(10, 6))
plt.scatter(df['dss'], df['hallucination'].astype(int), alpha=0.7)
plt.xlabel('Distributional Semantics Strength (DSS)')
plt.ylabel('Hallucination (1=Yes, 0=No)')
plt.title('Correlation between DSS and Hallucination')
plt.grid(True)
plt.savefig("dss_hallucination_correlation.png")
plt.close()

# Calculate correlation
correlation = df['dss'].corr(df['hallucination'])
print(f"Correlation between DSS and hallucination: {correlation}")
```

## Quantitative Evaluation of Explanation Faithfulness

For a simple quantitative evaluation using the DST framework:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from ltr.dst import DistributionalSemanticsTracer
import pandas as pd

# Load model
model_name = "google/gemma2-2b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize tracer
tracer = DistributionalSemanticsTracer(model, tokenizer)

# Define a sample of prompts for evaluation
# (In a full study, you would use all 11,000 HALoGEN prompts and 750 Racing Thoughts prompts)
sample_prompts = [
    "Please answer succinctly. I am at a concert. I see a bass. Is it a fish?",
    "John is going fishing, so he walks over to the river bank. Can he withdraw money at the bank?",
    "Please answer succinctly. I am coding. I see some java. Is it a programming language?",
    "Please answer succinctly. The physician is somebody's grandmother. Is the physician a woman?"
]

# Define factual versions of prompts
factual_prompts = [
    "A bass at a concert is a musical instrument, not a fish.",
    "A river bank is a geographical feature, not a financial institution.",
    "Java in a coding context is a programming language.",
    "A grandmother is a woman."
]

# Define hallucinated outputs for each prompt
hallucinated_outputs = [
    "Yes",
    "Yes",
    "No",
    "No"
]

# Initialize results collection
results = []

# Run DST analysis on each prompt
for i, (prompt, factual, hallucinated) in enumerate(zip(sample_prompts, factual_prompts, hallucinated_outputs)):
    print(f"Processing prompt {i+1}/{len(sample_prompts)}")
    
    # Run DST analysis
    result = tracer.run_analysis(
        prompt=prompt,
        factual_prompt=factual,
        concept_examples=["concert", "bass", "instrument", "fish", "music"],  # Would need customization per prompt
        hallucinated_output=hallucinated,
        run_intervention=True,
        enhanced_viz=False  # Disable visualizations for batch processing
    )
    
    # Extract relevant metrics for faithfulness evaluation
    
    # 1. Concept importance - measure of causal faithfulness
    max_importance = max(result.concept_importance.values()) if result.concept_importance else 0
    
    # 2. Spurious spans - measure of attribution faithfulness
    attribution_score = len(result.spurious_spans) / 5 if result.spurious_spans else 0
    
    # 3. Intervention success - measure of counterfactual faithfulness
    intervention_success = 0
    if result.intervention_results and 'critical_layer_patching' in result.intervention_results:
        if result.intervention_results['critical_layer_patching']['predicted_next'] != hallucinated:
            intervention_success = 1
    
    # Composite faithfulness score (simple average of three components)
    faithfulness_score = (max_importance + attribution_score + intervention_success) / 3
    
    # Store results
    results.append({
        'prompt': prompt,
        'causal_faithfulness': max_importance,
        'attribution_faithfulness': attribution_score,
        'counterfactual_faithfulness': intervention_success,
        'composite_faithfulness': faithfulness_score
    })

# Create summary dataframe
results_df = pd.DataFrame(results)
print(results_df)

# Calculate average faithfulness
avg_faithfulness = results_df['composite_faithfulness'].mean()
print(f"Average faithfulness score: {avg_faithfulness:.4f}")
```

