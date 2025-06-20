# LTR Examples

This directory contains examples demonstrating the usage of the LTR (LLM Thought Tracing) library for analyzing and interpreting language models. Each example focuses on a different aspect of model interpretability.

## Basic Examples

- `ltr_example.py`: Basic usage of the core LTR library for concept extraction, reasoning analysis, and causal intervention
- `huggingface_example.py`: Example using HuggingFace models
- `qwen_example.py`: Example using Qwen model family

## Advanced Examples

### Attention Analysis

- `attention_analysis_example.py`: Demonstrates attention pattern analysis and attention head ablation
  - Analyzes attention patterns in transformer models
  - Visualizes attention to specific concepts
  - Measures the impact of ablating specific attention heads

### Logit Lens

- `logit_lens_example.py`: Demonstrates intermediate representation analysis using logit lens
  - Projects intermediate states to vocabulary space
  - Traces token probability evolution across model layers
  - Visualizes how token probabilities develop through the network

### Backpatching

- `backpatching_example.py`: Demonstrates backpatching interventions between prompts
  - Compares residual stream activations between two prompts
  - Measures causal effects of patching activations from one prompt to another
  - Analyzes which layers and positions are most important for specific predictions

### Patching Control

- `patching_control_example.py`: Demonstrates controlled patching experiments
  - Compares clean and corrupted versions of prompts
  - Isolates the influence of specific tokens and positions
  - Performs bidirectional patching to compare alternative prompts

### Entity Analysis

- `entity_analysis_example.py`: Demonstrates causal entity analysis techniques
  - Measures the causal influence of entities on model predictions
  - Compares representations of different entities
  - Analyzes similarity between entity representations across contexts

### Patchscopes

- `patchscopes_example.py`: Demonstrates open-ended generation analysis
  - Tracks entity probabilities during model generation
  - Analyzes attention patterns in continuous generation
  - Visualizes how entity trajectories evolve during text generation

### Behavioral Analysis

- `behavioral_analysis_example.py`: Demonstrates model behavior analysis and response evaluation
  - Analyzes model performance across different question types
  - Tests factuality in model responses
  - Measures sensitivity to prompt variations
  - Automatically scores the quality of model responses

## Running the Examples

To run any example:

```bash
# Install required packages
pip install -r ../requirements.txt

# Run a specific example
python attention_analysis_example.py
```

Note that these examples use the GPT-2 model for simplicity and faster execution, but the same techniques can be applied to larger models for deeper analysis.
