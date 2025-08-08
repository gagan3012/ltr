# Distributional Semantics Tracing (DST) for LLM Hallucination Analysis

This repository contains the implementation for the paper investigating how Large Language Models (LLMs) generate hallucinations through intrinsic failure modes. Our work introduces **Distributional Semantics Tracing (DST)**, a unified framework that maps internal semantic failures leading to hallucinations.

## 🎯 Overview

The codebase addresses three key research questions:

1. **How to reliably trace internal semantic failures** causing hallucinations
2. **When hallucinations become inevitable** during computation (commitment layer identification)  
3. **What underlying mechanisms** cause these failures (fast associative vs slow contextual pathways)

## 🚀 Quick Start

### Installation

```bash
uv pip install ltr-llm
```

### Basic DST Analysis

```python
from ltr.dst import DistributionalSemanticsTracer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Initialize DST
tracer = DistributionalSemanticsTracer(model, tokenizer)

# Run analysis
result = tracer.run_analysis(
    prompt="The first person to walk on Venus was",
    factual_prompt="Venus has never been walked on by humans",
    concept_examples=["Venus", "Neil Armstrong", "impossible", "surface"],
    hallucinated_output="Neil Armstrong",
    run_intervention=True
)
```

## 🔬 Reproducing Paper Results

### 1. Commitment Layer Identification

Identify when hallucinations become inevitable:

```python
from explain_hallucinations.training_data_distr_semantics import ConceptualHallucinationAnalyzer

analyzer = ConceptualHallucinationAnalyzer("meta-llama/Llama-3.2-1B-Instruct")
analyzer.run_full_semantic_analysis(
    output_dir="commitment_layer_analysis",
    selected_layer=None  # Analyzes all layers
)
```

### 2. Semantic Ambiguity Analysis

Analyze how poor distributional semantics lead to hallucinations:

```python
from examples.ds_v2 import ConceptualHallucinationAnalyzer

analyzer = ConceptualHallucinationAnalyzer()

# Analyze specific concepts with ambiguous meanings
analyzer.analyze_prompt_with_concepts(
    prompt="I went to the bank to get money, but sat by the river bank instead.",
    target_concepts=["bank", "river", "money", "water"],
    max_texts=5000,
    output_dir="semantic_ambiguity_analysis"
)
```

### 3. Causal Path Tracing

Trace reasoning paths leading to hallucinations:

```python
from explain_hallucinations.causal_path_tracing_exp import analyze_reasoning

# Analyze reasoning for ambiguous contexts
results = analyze_reasoning(
    model, tokenizer, model_name,
    prompt="I am at a concert. I see a bass. Is it a fish?",
    intermediate_concepts=["concert", "bass", "instrument", "fish"],
    final_concepts=["Yes", "No"],
    potential_paths=[
        ["concert", "bass", "instrument", "No"],
        ["concert", "bass", "fish", "Yes"]
    ]
)
```

### 4. Patchscopes Analysis

Investigate layer-by-layer hallucination emergence:

```python
from ltr.patchscopes import perform_patchscope_analysis

results = perform_patchscope_analysis(
    model=model,
    tokenizer=tokenizer,
    prompt="The capital of Mars is",
    target_entities=["Mars", "capital", "impossible"],
    target_layers=list(range(0, 24, 4)),  # Sample layers
    explanation_prompts=[
        "Is this factually correct?",
        "What planet is being discussed?",
        "Is this physically possible?"
    ]
)
```

### 5. Clustering Analysis

Analyze embedding patterns in hallucinated vs factual content:

```python
from examples.clustering_example import HallucinationClusterAnalyzer

cluster_analyzer = HallucinationClusterAnalyzer("Qwen/Qwen2.5-0.5B-Instruct")
cluster_analyzer.run_comprehensive_analysis(
    output_dir="clustering_analysis"
)
```

## 📊 Key Experiments

### Experiment 1: Fast vs Slow Pathways

Demonstrates the conflict between associative and contextual reasoning:

```python
# Run the main DST analysis
python explain_hallucinations/run_distr_semantics.py
```

### Experiment 2: Reasoning Shortcut Hijacks

Shows how models take reasoning shortcuts leading to hallucinations:

```python
from examples.hall_test import ConceptualHallucinationAnalyzer

analyzer = ConceptualHallucinationAnalyzer()
analyzer.run_full_semantic_analysis("reasoning_shortcuts_analysis")
```

### Experiment 3: Analogical Collapse

Investigates how semantic similarity causes incorrect analogies:

```python
from explain_hallucinations.distr_semantics_viz import analyze_concept_distributions

completion, results = analyze_concept_distributions(
    model, tokenizer,
    prompt="John cena is going fishing, so he walks over to the river. He has come to the bank with his son, who wants to learn fishing. He has run out of cash. Can he withdraw money at the ATM at this bank?",
    concepts=["bank", "river", "money", "finance", "water"],
    output_dir="analogical_collapse_analysis"
)
```

## 📈 Visualizations and Analysis

### Generate Concept Heatmaps

```python
from explain_hallucinations.distr_semantics_viz import main

# Generates comprehensive visualizations
main()
```

### Layer-by-Layer Analysis

```python
from examples.patchscopes_example import example_3_layer_by_layer_analysis

# Detailed layer progression analysis
results = example_3_layer_by_layer_analysis()
```

### Custom Model Analysis

```python
# For different models
analyzer = ConceptualHallucinationAnalyzer("your-model-name")

# Custom layer selection
analyzer.run_full_semantic_analysis(
    selected_layer=15,  # Focus on specific layer
    output_dir="custom_analysis"
)
```

### Intervention Experiments

```python
from ltr.causal_intervention import perform_causal_intervention

# Test causal interventions
intervention_results = perform_causal_intervention(
    model=model,
    tokenizer=tokenizer,
    prompt="The first person to walk on Venus was",
    concepts=["Venus", "impossible", "Neil Armstrong"]
)
```

## 📋 Expected Outputs

Running the experiments will generate:

- **Commitment layer identification plots** showing when hallucinations become inevitable
- **Semantic confusion matrices** revealing concept ambiguities
- **Layer progression visualizations** tracking semantic drift
- **Causal intervention results** demonstrating mechanistic understanding
- **Concept clustering analysis** comparing hallucinated vs factual content
