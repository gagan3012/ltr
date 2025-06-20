# LTR: LLM Thought Tracing

LTR (LLM Thought Tracing) is a Python library for analyzing and interpreting language models. It provides a suite of tools to trace, visualize, and intervene in the internal workings of transformers, enabling deeper insights into their behavior, reasoning, and failure modes.

## Key Features

The library provides modules for various interpretability techniques:

- **Attention Analysis**: Analyze attention patterns, visualize attention to specific concepts, and measure the impact of ablating attention heads.
- **Logit Lens**: Trace the evolution of token probabilities across model layers by projecting intermediate hidden states into the vocabulary space.
- **Backpatching**: Measure the causal effects of specific activations by patching them from one forward pass to another.
- **Patching Control**: Perform controlled experiments by patching corrupted or alternative prompts to isolate the influence of specific tokens.
- **Entity Analysis**: Measure the causal influence of entities on model predictions and analyze their representations.
- **Patchscopes**: Analyze open-ended generation by tracking entity probabilities and attention patterns as text is generated.
- **Behavioral Analysis**: Evaluate model performance, test factuality, and measure sensitivity to prompt variations with automated scoring.
