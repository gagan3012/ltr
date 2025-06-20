import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging


def autoscore_responses(
    model, 
    tokenizer, 
    questions: List[str],
    responses: List[str],
    criteria: List[str],
    scoring_template: Optional[str] = None
) -> Dict:
    """
    Automatically score model responses using the model itself.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        questions: List of questions or prompts
        responses: List of model responses to score
        criteria: List of criteria to score against (e.g., "accuracy", "relevance")
        scoring_template: Template for scoring prompt (if None, uses default)
        
    Returns:
        Dict containing scoring results
    """
    if scoring_template is None:
        scoring_template = """Question: {question}

Response to evaluate: {response}

Evaluate the response above on {criterion} on a scale from 1-5, where:
1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

Rate {criterion} (just the number 1-5): """
    
    results = {
        "criteria": criteria,
        "question_scores": [],
        "criterion_averages": {c: 0.0 for c in criteria},
        "overall_average": 0.0
    }
    
    total_scores = 0
    criterion_counts = {c: 0 for c in criteria}
    
    for i, (question, response) in enumerate(zip(questions, responses)):
        question_result = {
            "question_idx": i,
            "question": question,
            "response": response,
            "scores": {}
        }
        
        # Score each criterion
        for criterion in criteria:
            # Format the scoring prompt
            scoring_prompt = scoring_template.format(
                question=question,
                response=response,
                criterion=criterion
            )
            
            # Generate score using the model
            score = generate_score_with_model(model, tokenizer, scoring_prompt)
            
            # Store score
            question_result["scores"][criterion] = score
            
            # Update criterion average
            results["criterion_averages"][criterion] += score
            criterion_counts[criterion] += 1
            
            # Update overall average
            total_scores += score
        
        results["question_scores"].append(question_result)
    
    # Calculate final averages
    total_count = sum(criterion_counts.values())
    
    for criterion in criteria:
        if criterion_counts[criterion] > 0:
            results["criterion_averages"][criterion] /= criterion_counts[criterion]
    
    if total_count > 0:
        results["overall_average"] = total_scores / total_count
    
    return results


def generate_score_with_model(model, tokenizer, scoring_prompt):
    """
    Generate a score (1-5) using the model for a given scoring prompt.
    """
    # Tokenize the input
    inputs = tokenizer(scoring_prompt, return_tensors="pt").to(model.device)
    
    # Generate a single token with no sampling for deterministic output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,  # Only generate one token (the score)
            do_sample=False,   # No sampling for deterministic output
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated token
    generated_token = outputs[0, inputs.input_ids.shape[1]:].item()
    generated_text = tokenizer.decode([generated_token], skip_special_tokens=True).strip()
    
    # Try to parse the generated text as a number
    try:
        score = int(generated_text)
        # Ensure the score is in the range 1-5
        if 1 <= score <= 5:
            return score
        else:
            # Default to middle score if out of range
            logging.warning(f"Generated score {score} out of range (1-5), defaulting to 3")
            return 3
    except ValueError:
        # If we couldn't parse a number, check for number words
        number_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        for word, num in number_map.items():
            if word in generated_text.lower():
                return num
        
        # Default to middle score if can't parse
        logging.warning(f"Could not parse score from '{generated_text}', defaulting to 3")
        return 3


def evaluate_responses_with_reference(
    model,
    tokenizer,
    questions: List[str],
    responses: List[str],
    reference_answers: List[str],
    evaluation_function: Optional[Callable] = None
) -> Dict:
    """
    Evaluates model responses against reference answers.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        questions: List of questions
        responses: List of model responses to evaluate
        reference_answers: List of reference answers
        evaluation_function: Custom function for evaluation (if None, uses default)
        
    Returns:
        Dict containing evaluation results
    """
    results = {
        "question_evaluations": [],
        "overall_metrics": {
            "accuracy": 0.0,
            "similarity": 0.0
        }
    }
    
    total_accuracy = 0.0
    total_similarity = 0.0
    
    for i, (question, response, reference) in enumerate(zip(questions, responses, reference_answers)):
        # Create evaluation prompt
        evaluation_prompt = f"""Question: {question}

Model response: {response}

Reference answer: {reference}

Is the model response correct? (yes/no): """
        
        # Get accuracy evaluation
        is_correct = evaluate_accuracy_with_model(model, tokenizer, evaluation_prompt)
        
        # Calculate similarity between response and reference
        similarity = calculate_similarity(tokenizer, response, reference)
        
        # Store evaluation
        evaluation = {
            "question_idx": i,
            "question": question,
            "response": response,
            "reference": reference,
            "is_correct": is_correct,
            "similarity": similarity
        }
        
        results["question_evaluations"].append(evaluation)
        
        # Update overall metrics
        total_accuracy += 1.0 if is_correct else 0.0
        total_similarity += similarity
    
    # Calculate final metrics
    num_questions = len(questions)
    if num_questions > 0:
        results["overall_metrics"]["accuracy"] = total_accuracy / num_questions
        results["overall_metrics"]["similarity"] = total_similarity / num_questions
    
    return results


def evaluate_accuracy_with_model(model, tokenizer, evaluation_prompt):
    """
    Evaluate whether the model response is correct based on the reference.
    """
    # Tokenize the input
    inputs = tokenizer(evaluation_prompt, return_tensors="pt").to(model.device)
    
    # Generate a short response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,  # Just enough for "yes" or "no"
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    
    # Check if the response indicates correctness
    return "yes" in generated_text and "no" not in generated_text


def calculate_similarity(tokenizer, response, reference):
    """
    Calculate similarity between response and reference using token overlap.
    """
    response_tokens = tokenizer.tokenize(response.lower())
    reference_tokens = tokenizer.tokenize(reference.lower())
    
    # Convert to sets for overlap computation
    response_set = set(response_tokens)
    reference_set = set(reference_tokens)
    
    # Calculate Jaccard similarity
    if not reference_set:
        return 0.0
    
    intersection = response_set.intersection(reference_set)
    union = response_set.union(reference_set)
    
    return len(intersection) / len(union) if union else 0.0


def batch_evaluate_responses(
    model,
    tokenizer,
    dataset: List[Dict],
    evaluation_template: Optional[str] = None,
    batch_size: int = 10
) -> Dict:
    """
    Evaluates a large batch of responses using the model.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        dataset: List of dictionaries with question, response, and reference fields
        evaluation_template: Template for evaluation (if None, uses default)
        batch_size: Number of evaluations to process in one batch
        
    Returns:
        Dict containing evaluation results
    """
    if evaluation_template is None:
        evaluation_template = """
Question: {question}
Response: {response}
Reference: {reference}

Score the response for accuracy on a scale from 1-5, where 1 is completely incorrect and 5 is perfect:
"""
    
    results = {
        "evaluations": [],
        "average_score": 0.0,
        "distribution": {i: 0 for i in range(1, 6)}
    }
    
    total_score = 0
    count = 0
    
    # Process in batches
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        
        for item in batch:
            # Format the evaluation prompt
            question = item.get("question", "")
            response = item.get("response", "")
            reference = item.get("reference", "")
            
            evaluation_prompt = evaluation_template.format(
                question=question,
                response=response,
                reference=reference
            )
            
            # Generate score using the model
            score = generate_score_with_model(model, tokenizer, evaluation_prompt)
            
            # Store evaluation
            evaluation = {
                "question": question,
                "response": response,
                "reference": reference,
                "score": score
            }
            
            results["evaluations"].append(evaluation)
            
            # Update running totals
            total_score += score
            count += 1
            results["distribution"][score] = results["distribution"].get(score, 0) + 1
    
    # Calculate average
    if count > 0:
        results["average_score"] = total_score / count
    
    return results
