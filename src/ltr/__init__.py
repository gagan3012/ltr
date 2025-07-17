"""
LTR - Mechanistic Interpretability for Neural Reasoning

A library for tracing and visualizing concept evolution in Large Language Models
"""

from ltr.concept_extraction import extract_concept_activations, get_layer_pattern_and_count
from ltr.reasoning_analysis import analyze_reasoning_paths
from ltr.causal_intervention import perform_causal_intervention
from ltr.visualization import (
    plot_concept_activations,
    plot_causal_intervention_heatmap,
    animate_concept_evolution,
    plot_concept_activation_heatmap,
    animate_concept_activation_diagonal,
    animate_reasoning_flow,
    animate_reasoning_flow_dark,
    plot_layer_position_intervention,
    save_animation,
    plot_logit_lens_heatmap,
    plot_token_evolution_curves,
    plot_combined_logit_lens,
)

# Import attention analysis
from ltr.attention_analysis import (
    analyze_attention_patterns,
    ablate_attention_patterns
)

# Import backpatching
from ltr.backpatching import perform_backpatching

# Import logit lens
from ltr.logit_lens import (
    logit_lens_analysis,
    trace_token_evolution
)

# Import patching control
from ltr.patching_control import (
    perform_patching_control,
    pairwise_patching
)

# Import patchscopes analysis
from ltr.patchscopes import (
    perform_patchscope_analysis,
    analyze_entity_trajectories
)

# Import behavioral analysis
from ltr.behavioral_analysis import (
    analyze_model_behavior,
    analyze_factuality,
    analyze_prompt_sensitivity
)

# Import entity analysis
from ltr.entity_analysis import (
    analyze_causal_entities,
    extract_entity_representations,
    compare_entity_representations
)

# Import autoscoring
from ltr.autoscoring import (
    autoscore_responses,
    evaluate_responses_with_reference,
    batch_evaluate_responses
)

from ltr.subsequence_analysis import (
    SubsequenceAnalyzer,
    analyze_hallucination_subsequences
)

from ltr.dst import (
    DistributionalSemanticsTracer,
    DSTResult
)

# Define public API
__all__ = [
    # Concept extraction
    'extract_concept_activations', 'get_layer_pattern_and_count',
    
    # Reasoning analysis
    'analyze_reasoning_paths',
    
    # Causal intervention
    'perform_causal_intervention',
    
    # Visualization
    'plot_concept_activations', 'plot_causal_intervention_heatmap', 'animate_concept_evolution',
    
    # Attention analysis
    'analyze_attention_patterns', 'ablate_attention_patterns',
    
    # Backpatching
    'perform_backpatching',
    
    # Logit lens
    'logit_lens_analysis', 'trace_token_evolution',
    
    # Patching control
    'perform_patching_control', 'pairwise_patching',
    
    # Patchscopes
    'perform_patchscope_analysis', 'analyze_entity_trajectories',
    
    # Behavioral analysis
    'analyze_model_behavior', 'analyze_factuality', 'analyze_prompt_sensitivity',
    
    # Entity analysis
    'analyze_causal_entities', 'extract_entity_representations', 'compare_entity_representations',
    
    # Autoscoring
    'autoscore_responses', 'evaluate_responses_with_reference', 'batch_evaluate_responses',

    'plot_token_evolution_curves', 'plot_combined_logit_lens', 'plot_layer_position_intervention',

    'save_animation', 'plot_logit_lens_heatmap', 'plot_concept_activation_heatmap',

    'animate_concept_activation_diagonal', 'animate_reasoning_flow', 'animate_reasoning_flow_dark',

    'SubsequenceAnalyzer', 'analyze_hallucination_subsequences', 'evaluate_subsequence_causality',

    'DistributionalSemanticsTracer', 'DSTResult',
]

__version__ = "0.2.0"
