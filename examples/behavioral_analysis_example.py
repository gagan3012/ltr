"""
Example demonstrating behavioral analysis and autoscoring.

This example shows how to analyze model behavior across different prompts and
automatically score model responses using the model itself.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from ltr.behavioral_analysis import analyze_model_behavior, analyze_factuality, analyze_prompt_sensitivity
from ltr.autoscoring import autoscore_responses, evaluate_responses_with_reference
import matplotlib.pyplot as plt
import numpy as np

def behavioral_analysis_example():
    """
    Example showing how to analyze model behavior and score responses.
    """
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "gpt2"  # Or any HuggingFace model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Define prompts for behavioral analysis
    prompts = [
        "Paris is the capital of France.",
        "London is the capital of England.",
        "Berlin is the capital of Germany.",
        "Madrid is the capital of Spain."
    ]
    question_types = ["factual"] * len(prompts)
    
    # Perform behavioral analysis
    print("\nPerforming behavioral analysis...")
    behavior_results = analyze_model_behavior(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        question_types=question_types,
        metrics=["perplexity", "entropy", "confidence"]
    )
    
    # Print overall results
    print("\nBehavioral analysis results:")
    print(f"Overall metrics:")
    for metric, value in behavior_results['overall_results'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Print metrics for each prompt
    print("\nPrompt-specific metrics:")
    for prompt_result in behavior_results['prompt_results']:
        print(f"  Prompt: '{prompt_result['prompt']}'")
        print(f"    Average perplexity: {prompt_result.get('avg_perplexity', 'N/A')}")
        print(f"    Average entropy: {prompt_result.get('avg_entropy', 'N/A')}")
        print(f"    Average confidence: {prompt_result.get('avg_confidence', 'N/A')}")
    
    # Factuality analysis
    print("\nPerforming factuality analysis...")
    
    questions = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Australia?",
        "What is the capital of Brazil?"
    ]
    correct_answers = ["Paris", "Tokyo", "Canberra", "Brasília"]
    incorrect_answers = ["London", "Beijing", "Sydney", "Rio de Janeiro"]
    
    factuality_results = analyze_factuality(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        correct_answers=correct_answers,
        incorrect_answers=incorrect_answers
    )
    
    # Print factuality results
    print("\nFactuality analysis results:")
    print(f"Overall accuracy: {factuality_results['overall_accuracy']:.4f}")
    print(f"Average correct probability: {factuality_results['avg_correct_prob']:.4f}")
    print(f"Average incorrect probability: {factuality_results['avg_incorrect_prob']:.4f}")
    
    # Print question-specific results
    print("\nQuestion-specific results:")
    for i, result in enumerate(factuality_results['question_results']):
        print(f"  Question: '{result['question']}'")
        print(f"    Correct answer: '{result['correct_answer']}' (prob: {result['correct_prob']:.4f})")
        print(f"    Incorrect answer: '{result['incorrect_answer']}' (prob: {result['incorrect_prob']:.4f})")
        print(f"    Is correct: {result['is_correct']}")
    
    # Prompt sensitivity analysis
    print("\nPerforming prompt sensitivity analysis...")
    
    base_prompt = "The capital of the United States is"
    variants = [
        "The capital of the United States is",
        "The US capital city is",
        "The capital city of America is",
        "What is the capital of the United States? The capital is"
    ]
    target_token = "Washington"
    
    sensitivity_results = analyze_prompt_sensitivity(
        model=model,
        tokenizer=tokenizer,
        base_prompt=base_prompt,
        variants=variants,
        target_token=target_token
    )
    
    # Print sensitivity results
    print("\nPrompt sensitivity results:")
    print(f"Base prompt: '{sensitivity_results['base_prompt']}'")
    print(f"  Top token: '{sensitivity_results['base_result']['top_token']}'")
    print(f"  Target token ('{target_token}') probability: {sensitivity_results['base_result']['target_prob']:.4f}")
    
    print("\nVariant results:")
    for i, result in enumerate(sensitivity_results['variant_results']):
        print(f"  Variant: '{result['prompt']}'")
        print(f"    Top token: '{result['top_token']}'")
        print(f"    Target token probability: {result['target_prob']:.4f}")
        print(f"    Delta from base: {result['delta_target_prob']:.4f}")
        print(f"    Same prediction as base: {result['same_prediction']}")
    
    # Print sensitivity metrics
    print("\nSensitivity metrics:")
    print(f"  Prediction stability: {sensitivity_results['sensitivity_metrics']['prediction_stability']:.4f}")
    print(f"  Average KL divergence: {sensitivity_results['sensitivity_metrics']['avg_kl_divergence']:.4f}")
    
    # Autoscoring example
    print("\nPerforming autoscoring of responses...")
    
    questions = [
        "What is the capital of France?",
        "Name the largest planet in our solar system.",
        "Who wrote Romeo and Juliet?"
    ]
    responses = [
        "The capital of France is Paris.",
        "Jupiter is the largest planet in our solar system.",
        "William Shakespeare wrote the play Romeo and Juliet."
    ]
    criteria = ["accuracy", "clarity", "conciseness"]
    
    scoring_results = autoscore_responses(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        responses=responses,
        criteria=criteria
    )
    
    # Print scoring results
    print("\nAutoscoring results:")
    print(f"Overall average score: {scoring_results['overall_average']:.2f}/5")
    
    print("\nScores by criterion:")
    for criterion, avg_score in scoring_results['criterion_averages'].items():
        print(f"  {criterion.capitalize()}: {avg_score:.2f}/5")
    
    print("\nQuestion-specific scores:")
    for result in scoring_results['question_scores']:
        print(f"  Question: '{result['question']}'")
        print(f"  Response: '{result['response']}'")
        print("  Scores:")
        for criterion, score in result['scores'].items():
            print(f"    {criterion.capitalize()}: {score}/5")
        print()
    
    # Response evaluation with reference
    print("\nEvaluating responses against references...")
    
    questions = [
        "What is the capital of Germany?",
        "What is the chemical symbol for gold?"
    ]
    responses = [
        "Berlin is the capital city of Germany.",
        "The chemical symbol for gold is Au."
    ]
    references = [
        "The capital of Germany is Berlin.",
        "Au is the chemical symbol for gold on the periodic table."
    ]
    
    eval_results = evaluate_responses_with_reference(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        responses=responses,
        reference_answers=references
    )
    
    # Print evaluation results
    print("\nEvaluation results:")
    print(f"Overall accuracy: {eval_results['overall_metrics']['accuracy']:.2f}")
    print(f"Overall similarity: {eval_results['overall_metrics']['similarity']:.2f}")
    
    print("\nQuestion-specific evaluations:")
    for result in eval_results['question_evaluations']:
        print(f"  Question: '{result['question']}'")
        print(f"  Response: '{result['response']}'")
        print(f"  Reference: '{result['reference']}'")
        print(f"  Is correct: {result['is_correct']}")
        print(f"  Similarity: {result['similarity']:.2f}")
        print()

if __name__ == "__main__":
    behavioral_analysis_example()
