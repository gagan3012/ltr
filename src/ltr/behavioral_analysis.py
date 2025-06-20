import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import logging


def analyze_model_behavior(
    model, 
    tokenizer, 
    prompts: List[str],
    question_types: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None
) -> Dict:
    """
    Analyzes model behavior across multiple prompts.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        prompts: List of prompts to analyze
        question_types: Optional categorization of prompt types
        metrics: List of metrics to compute ("perplexity", "entropy", "confidence")
        
    Returns:
        Dict containing behavioral analysis results
    """
    if metrics is None:
        metrics = ["perplexity", "entropy", "confidence"]
    
    if question_types is None:
        question_types = ["uncategorized"] * len(prompts)
    
    results = {
        "prompts": prompts,
        "question_types": question_types,
        "metrics": metrics,
        "overall_results": {},
        "prompt_results": [],
        "type_results": {}
    }
    
    # Process each prompt
    for i, (prompt, q_type) in enumerate(zip(prompts, question_types)):
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids[0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        n_tokens = len(tokens)
        
        # Run model to get logits
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()
        
        prompt_metrics = {
            "prompt_idx": i,
            "prompt": prompt,
            "question_type": q_type,
            "tokens": tokens,
            "token_metrics": []
        }
        
        # Calculate metrics for each position
        for pos in range(n_tokens - 1):  # Exclude the last token as we can't predict beyond it
            token_logits = logits[0, pos]
            probs = torch.softmax(token_logits, dim=-1)
            next_token_id = input_ids[pos + 1].item()
            
            token_data = {
                "position": pos,
                "token": tokens[pos],
                "next_token": tokens[pos + 1] if pos + 1 < n_tokens else None,
                "metrics": {}
            }
            
            # Calculate requested metrics
            if "perplexity" in metrics:
                # Get the probability of the actual next token
                next_token_prob = probs[next_token_id].item()
                token_data["metrics"]["perplexity"] = -np.log2(next_token_prob + 1e-10)
            
            if "entropy" in metrics:
                # Calculate the entropy of the distribution
                entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
                token_data["metrics"]["entropy"] = entropy
            
            if "confidence" in metrics:
                # Get the highest probability
                top_prob = torch.max(probs).item()
                token_data["metrics"]["confidence"] = top_prob
            
            prompt_metrics["token_metrics"].append(token_data)
        
        # Calculate prompt-level aggregated metrics
        for metric in metrics:
            if metric in ["perplexity", "entropy", "confidence"]:
                values = [t["metrics"].get(metric, 0) for t in prompt_metrics["token_metrics"] if metric in t["metrics"]]
                if values:
                    prompt_metrics[f"avg_{metric}"] = sum(values) / len(values)
                    prompt_metrics[f"max_{metric}"] = max(values)
                    prompt_metrics[f"min_{metric}"] = min(values)
        
        results["prompt_results"].append(prompt_metrics)
        
        # Update type-level aggregation
        if q_type not in results["type_results"]:
            results["type_results"][q_type] = {
                "count": 0,
                "metrics": {f"avg_{m}": 0.0 for m in metrics},
                "prompts": []
            }
        
        results["type_results"][q_type]["count"] += 1
        results["type_results"][q_type]["prompts"].append(i)
        
        for metric in metrics:
            metric_key = f"avg_{metric}"
            if metric_key in prompt_metrics:
                current_sum = results["type_results"][q_type]["metrics"][metric_key] * (results["type_results"][q_type]["count"] - 1)
                results["type_results"][q_type]["metrics"][metric_key] = (current_sum + prompt_metrics[metric_key]) / results["type_results"][q_type]["count"]
    
    # Calculate overall metrics
    for metric in metrics:
        metric_key = f"avg_{metric}"
        values = [p.get(metric_key, 0) for p in results["prompt_results"] if metric_key in p]
        if values:
            results["overall_results"][metric_key] = sum(values) / len(values)
    
    return results


def analyze_factuality(
    model,
    tokenizer,
    questions: List[str],
    correct_answers: List[str],
    incorrect_answers: Optional[List[str]] = None
) -> Dict:
    """
    Analyzes model's factuality by comparing probabilities of correct vs. incorrect answers.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        questions: List of questions to analyze
        correct_answers: List of correct answers for each question
        incorrect_answers: List of incorrect answers to compare against
        
    Returns:
        Dict containing factuality analysis results
    """
    results = {
        "questions": questions,
        "correct_answers": correct_answers,
        "question_results": [],
        "overall_accuracy": 0.0,
        "avg_correct_prob": 0.0,
        "avg_incorrect_prob": 0.0 if incorrect_answers else None
    }
    
    correct_count = 0
    total_correct_prob = 0.0
    total_incorrect_prob = 0.0 if incorrect_answers else None
    
    # Process each question
    for i, (question, correct) in enumerate(zip(questions, correct_answers)):
        # Prepare the prompt with the question
        prompt = question
        
        # Tokenize up to the start of the answer
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Run model to get logits
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1].detach().cpu()  # Get logits for the last token
        probs = torch.softmax(logits, dim=-1)
        
        # Get probability of correct answer
        correct_ids = tokenizer(correct, add_special_tokens=False).input_ids
        correct_prob = probs[correct_ids[0]].item() if correct_ids else 0.0
        
        question_result = {
            "question": question,
            "correct_answer": correct,
            "correct_prob": correct_prob
        }
        
        # Get probability of incorrect answer if provided
        if incorrect_answers:
            incorrect = incorrect_answers[i]
            incorrect_ids = tokenizer(incorrect, add_special_tokens=False).input_ids
            incorrect_prob = probs[incorrect_ids[0]].item() if incorrect_ids else 0.0
            question_result["incorrect_answer"] = incorrect
            question_result["incorrect_prob"] = incorrect_prob
            
            # Determine if model predicted correctly
            is_correct = correct_prob > incorrect_prob
            question_result["is_correct"] = is_correct
            
            if is_correct:
                correct_count += 1
            
            total_incorrect_prob += incorrect_prob
        else:
            # If no incorrect answers provided, use the top prediction
            top_id = torch.argmax(logits).item()
            top_token = tokenizer.decode([top_id])
            top_prob = probs[top_id].item()
            
            question_result["top_prediction"] = top_token
            question_result["top_prob"] = top_prob
            
            # Determine if model predicted correctly
            is_correct = correct_ids and top_id == correct_ids[0]
            question_result["is_correct"] = is_correct
            
            if is_correct:
                correct_count += 1
        
        total_correct_prob += correct_prob
        results["question_results"].append(question_result)
    
    # Calculate overall metrics
    num_questions = len(questions)
    if num_questions > 0:
        results["overall_accuracy"] = correct_count / num_questions
        results["avg_correct_prob"] = total_correct_prob / num_questions
        
        if incorrect_answers:
            results["avg_incorrect_prob"] = total_incorrect_prob / num_questions
    
    return results


def analyze_prompt_sensitivity(
    model,
    tokenizer,
    base_prompt: str,
    variants: List[str],
    target_token: Optional[str] = None
) -> Dict:
    """
    Analyzes model's sensitivity to prompt variations.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        base_prompt: Base prompt to compare against
        variants: List of prompt variations
        target_token: Optional token to track probability across variants
        
    Returns:
        Dict containing sensitivity analysis results
    """
    results = {
        "base_prompt": base_prompt,
        "variants": variants,
        "variant_results": []
    }
    
    # Process the base prompt
    base_inputs = tokenizer(base_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        base_outputs = model(**base_inputs)
    base_logits = base_outputs.logits.detach().cpu()
    
    # Get the top prediction for the base prompt
    base_top_id = torch.argmax(base_logits[0, -1]).item()
    base_top_token = tokenizer.decode([base_top_id])
    base_top_prob = torch.softmax(base_logits[0, -1], dim=-1)[base_top_id].item()
    
    # If no target token specified, use the top prediction from the base prompt
    if target_token is None:
        target_token = base_top_token
    
    # Get the target token ID
    target_ids = tokenizer(target_token, add_special_tokens=False).input_ids
    target_id = target_ids[0] if target_ids else None
    
    # Calculate target token probability in base prompt
    if target_id is not None:
        base_target_prob = torch.softmax(base_logits[0, -1], dim=-1)[target_id].item()
    else:
        base_target_prob = 0.0
    
    base_result = {
        "prompt": base_prompt,
        "top_token": base_top_token,
        "top_prob": base_top_prob,
        "target_token": target_token,
        "target_prob": base_target_prob
    }
    
    results["base_result"] = base_result
    
    # Process each variant
    for variant in variants:
        variant_inputs = tokenizer(variant, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            variant_outputs = model(**variant_inputs)
        variant_logits = variant_outputs.logits.detach().cpu()
        
        # Get the top prediction for this variant
        variant_top_id = torch.argmax(variant_logits[0, -1]).item()
        variant_top_token = tokenizer.decode([variant_top_id])
        variant_top_prob = torch.softmax(variant_logits[0, -1], dim=-1)[variant_top_id].item()
        
        # Calculate target token probability
        if target_id is not None:
            variant_target_prob = torch.softmax(variant_logits[0, -1], dim=-1)[target_id].item()
        else:
            variant_target_prob = 0.0
        
        # Calculate KL divergence between base and variant distributions
        base_probs = torch.softmax(base_logits[0, -1], dim=-1)
        variant_probs = torch.softmax(variant_logits[0, -1], dim=-1)
        kl_div = torch.sum(base_probs * (torch.log(base_probs + 1e-10) - torch.log(variant_probs + 1e-10))).item()
        
        variant_result = {
            "prompt": variant,
            "top_token": variant_top_token,
            "top_prob": variant_top_prob,
            "target_token": target_token,
            "target_prob": variant_target_prob,
            "delta_target_prob": variant_target_prob - base_target_prob,
            "kl_divergence": kl_div,
            "same_prediction": variant_top_token == base_top_token
        }
        
        results["variant_results"].append(variant_result)
    
    # Calculate overall sensitivity metrics
    num_same_prediction = sum(1 for r in results["variant_results"] if r["same_prediction"])
    avg_kl_div = sum(r["kl_divergence"] for r in results["variant_results"]) / len(variants) if variants else 0
    
    results["sensitivity_metrics"] = {
        "prediction_stability": num_same_prediction / len(variants) if variants else 1.0,
        "avg_kl_divergence": avg_kl_div
    }
    
    return results
