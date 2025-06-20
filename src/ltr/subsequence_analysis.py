"""
Subsequence Analysis for LLM Hallucination Detection

Integrates subsequence causal analysis from SAT into the LTR library framework.
Based on "Why and How LLMs Hallucinate: Connecting the Dots with Subsequence Associations"
"""

import math
import torch
from typing import Dict, List, Optional, Callable
import logging
import numpy as np

# Import utility functions - these would need to be adapted or recreated


class SubsequenceAnalyzer:
    """
    Analyzes subsequences that correlate with hallucinated outputs in language models.

    This class implements the methodology from "Why and How LLMs Hallucinate" by identifying
    subsequences in input prompts that are causally associated with specific target outputs.
    """

    def __init__(self, model, tokenizer, device: str = "auto", batch_size: int = 32):
        """
        Initialize the subsequence analyzer.

        Args:
            model: The language model to analyze
            tokenizer: The tokenizer for the model
            device: Device to run computations on
            batch_size: Batch size for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size

        # Move model to device if needed
        if (
            hasattr(self.model, "to")
            and not next(self.model.parameters()).device.type == self.device
        ):
            self.model = self.model.to(self.device)

        # Store analysis results
        self.analysis_results = {}
        self.perturbed_results = []

    def analyze_subsequences(
        self,
        prompt: str,
        target_string: str,
        num_perturbations: int = 100,
        perturbation_rate: float = 0.1,
        max_subseq_len_rate: float = 0.9,
        max_new_tokens: int = 128,
        beam_size: int = 10,
        ignore_items: Optional[set] = None,
        return_traces: bool = False,
    ) -> Dict:
        """
        Analyze subsequences that correlate with target string appearance.

        Args:
            prompt: Input prompt to analyze
            target_string: Target string to look for in outputs
            num_perturbations: Number of perturbed sequences to generate
            perturbation_rate: Rate of perturbation (0.0 to 1.0)
            max_subseq_len_rate: Maximum subsequence length as fraction of prompt
            max_new_tokens: Maximum tokens to generate
            beam_size: Beam size for subsequence search
            ignore_items: Token IDs to ignore in analysis
            return_traces: Whether to return activation traces

        Returns:
            Dictionary containing analysis results
        """        
        logging.info("Starting subsequence analysis for target: '%s'", target_string)

        # 1. Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        raw_input_ids = inputs.input_ids.squeeze()

        # 2. Generate perturbed sequences
        logging.info("Generating %s perturbed sequences...", num_perturbations)
        perturbed_seqs = self._generate_perturbed_sequences(
            raw_input_ids, num_perturbations, perturbation_rate
        )

        # 3. Generate outputs for perturbed sequences
        logging.info("Generating outputs for perturbed sequences...")
        output_texts = self._batch_generate_outputs(perturbed_seqs, max_new_tokens)

        # 4. Identify sequences that produce target string
        target_indices = [
            i
            for i, text in enumerate(output_texts)
            if self._contains_target(target_string, text)
        ]

        p_target = len(target_indices) / len(output_texts)        
        logging.info(
            "Target '%s' appeared in %s/%s outputs (p = %.3f)",
            target_string, len(target_indices), len(output_texts), p_target
        )

        # 5. Analyze subsequence frequencies
        target_sequences = [perturbed_seqs[i].tolist() for i in target_indices]
        all_sequences = [seq.tolist() for seq in perturbed_seqs]

        # Define conditional probability function
        def compute_conditional_prob(subseq):
            return self._compute_conditional_probability(
                subseq, all_sequences, target_sequences, p_target
            )

        # 6. Find most frequent subsequences at different levels
        logging.info("Analyzing subsequence frequencies...")
        max_subseq_len = math.ceil(max_subseq_len_rate * len(raw_input_ids))

        freq_results = self._find_frequent_subsequences(
            target_sequences,
            max_subseq_len,
            raw_input_ids.tolist(),
            ignore_items or set(),
            beam_size,
            compute_conditional_prob,
        )

        # 7. Compile results
        results = {
            "prompt": prompt,
            "target_string": target_string,
            "p_target": p_target,
            "num_perturbations": num_perturbations,
            "target_indices": target_indices,
            "subsequence_levels": freq_results,
            "perturbed_outputs": list(
                zip(
                    [
                        self.tokenizer.decode(seq, skip_special_tokens=True)
                        for seq in perturbed_seqs
                    ],
                    output_texts,
                )
            )
            if return_traces
            else None,
        }

        self.analysis_results = results
        return results

    def evaluate_subsequence(
        self,
        subsequence: List[int],
        original_sequence: List[int],
        target_string: str,
        num_tests: int = 20,
        completion_methods: Optional[List[str]] = None,
    ) -> Dict:
        """
        Evaluate a discovered subsequence's capacity to produce target outputs.

        Args:
            subsequence: Token IDs of the subsequence to evaluate
            original_sequence: Original prompt token IDs
            target_string: Target string to look for
            num_tests: Number of test completions to generate
            completion_methods: Methods for completion (e.g., ['random', 'mask'])

        Returns:
            Evaluation results dictionary
        """
        if completion_methods is None:
            completion_methods = ["random"]

        results = {}

        for method in completion_methods:
            test_results = self._evaluate_with_method(
                subsequence, original_sequence, target_string, num_tests, method
            )
            results[method] = test_results

        return results

    def compute_srep_reproducibility(
        self,
        subsequence: List[int],
        original_sequence: List[int],
        target_string: str,
        num_tests: int = 20,
        completion_methods: Optional[List[str]] = None,
    ) -> Dict:
        """
        Compute Srep: the probability that a hallucination subsequence appears in the output
        when the corresponding input subsequence is present, averaged over several input perturbation/filling strategies.
        Supported strategies: 'bert', 'random', 'gpt-m', 'gpt-t'.

        Args:
            subsequence: Token IDs of the subsequence to evaluate
            original_sequence: Original prompt token IDs
            target_string: Target string to look for
            num_tests: Number of test completions to generate
            completion_methods: Methods for completion (e.g., ['random', 'mask'])

        Returns:
            Evaluation results dictionary
        """
        if completion_methods is None:
            completion_methods = ["bert", "random", "gpt-m", "gpt-t"]

        method_success_rates = {}
        for method in completion_methods:
            try:
                test_results = self._evaluate_with_method(
                    subsequence, original_sequence, target_string, num_tests, method
                )
                method_success_rates[method] = test_results["success_rate"]
            except NotImplementedError:
                # If a method is not implemented, skip it
                continue

        # Average over all available methods
        if method_success_rates:
            srep = float(np.mean(list(method_success_rates.values())))
        else:
            srep = 0.0

        return {
            "srep": srep,
            "method_success_rates": method_success_rates,
            "num_tests": num_tests,
        }

    def _generate_perturbed_sequences(
        self, input_ids: torch.Tensor, num_perturbations: int, perturbation_rate: float
    ) -> torch.Tensor:
        """Generate perturbed versions of input sequence."""
        # This would use the perturbation logic from SAT
        # For now, implementing a simple version
        perturbed_sequences = []

        for _ in range(num_perturbations):
            seq = input_ids.clone()
            num_to_perturb = int(len(seq) * perturbation_rate)

            # Randomly select positions to perturb
            positions = torch.randperm(len(seq))[:num_to_perturb]

            # Replace with random tokens from vocabulary
            for pos in positions:
                seq[pos] = torch.randint(0, self.tokenizer.vocab_size, (1,)).item()

            perturbed_sequences.append(seq)

        return torch.stack(perturbed_sequences)

    def _batch_generate_outputs(
        self, sequences: torch.Tensor, max_new_tokens: int
    ) -> List[str]:
        """Generate outputs for a batch of sequences."""
        outputs = []

        # Process in batches to manage memory
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]

            with torch.no_grad():
                generated = self.model.generate(
                    batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )

            # Extract only the newly generated tokens
            for j, seq in enumerate(generated.sequences):
                input_len = len(batch[j])
                output_tokens = seq[input_len:]
                output_text = self.tokenizer.decode(
                    output_tokens, skip_special_tokens=True
                )
                outputs.append(output_text)

        return outputs

    def _contains_target(self, target: str, text: str) -> bool:
        """Check if target string appears in generated text."""
        return target.lower() in text.lower()

    def _compute_conditional_probability(
        self,
        subsequence: List[int],
        all_sequences: List[List[int]],
        target_sequences: List[List[int]],
        p_target: float,
    ) -> float:
        """Compute P(target|subsequence)."""
        # Count occurrences of subsequence
        subseq_count = sum(
            1 for seq in all_sequences if self._contains_subsequence(seq, subsequence)
        )
        target_subseq_count = sum(
            1
            for seq in target_sequences
            if self._contains_subsequence(seq, subsequence)
        )

        if subseq_count == 0:
            return 0.0

        return target_subseq_count / subseq_count

    def _contains_subsequence(
        self, sequence: List[int], subsequence: List[int]
    ) -> bool:
        """Check if subsequence is contained in sequence."""
        if len(subsequence) > len(sequence):
            return False

        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i : i + len(subsequence)] == subsequence:
                return True

        return False

    def _find_frequent_subsequences(
        self,
        target_sequences: List[List[int]],
        max_length: int,
        original_sequence: List[int],
        ignore_items: set,
        beam_size: int,
        scoring_func: Callable,
    ) -> Dict:
        """Find frequent subsequences at different lengths."""
        # This would implement the subsequence mining from SAT
        # Simplified version for demonstration
        results = {}

        for length in range(1, min(max_length + 1, len(original_sequence))):
            subsequences = {}

            # Extract all subsequences of this length
            for seq in target_sequences:
                for i in range(len(seq) - length + 1):
                    subseq = tuple(seq[i : i + length])
                    if not any(token in ignore_items for token in subseq):
                        if subseq not in subsequences:
                            subsequences[subseq] = 0
                        subsequences[subseq] += 1

            # Score and rank subsequences
            scored_subseqs = []
            for subseq, count in subsequences.items():
                score = scoring_func(list(subseq))
                scored_subseqs.append((list(subseq), score))

            # Keep top beam_size            scored_subseqs.sort(key=lambda x: x[1], reverse=True)
            results[length] = scored_subseqs[:beam_size]

        return results
        
    def _evaluate_with_method(
        self,
        subsequence: List[int],
        original_sequence: List[int],
        target_string: str,
        num_tests: int,
        method: str,
    ) -> Dict:
        """Evaluate subsequence with specific completion method."""
        if method == "random":
            return self._evaluate_random_completion(
                subsequence, original_sequence, target_string, num_tests
            )
        elif method == "bert":
            return self._evaluate_bert_completion(
                subsequence, original_sequence, target_string, num_tests
            )
        elif method == "gpt-m":
            return self._evaluate_gpt_completion(
                subsequence, original_sequence, target_string, num_tests, model_name="gpt-4o-mini"
            )
        elif method == "gpt-t":
            return self._evaluate_gpt_completion(
                subsequence, original_sequence, target_string, num_tests, model_name="chatgpt"
            )
        else:
            raise NotImplementedError(f"Completion method '{method}' not implemented")
            
    def _evaluate_random_completion(
        self,
        subsequence: List[int],
        original_sequence: List[int],
        target_string: str,
        num_tests: int,
    ) -> Dict:
        """
        Evaluate a subsequence using random completions.
        
        This method takes a subsequence and places it within a randomly generated
        context, then measures how often the target string appears in the output.
        
        Args:
            subsequence: Token IDs of the subsequence to evaluate
            original_sequence: Original prompt token IDs
            target_string: Target string to look for
            num_tests: Number of test completions to generate
            
        Returns:
            Dictionary with evaluation results
        """
        # Find all positions where subsequence can be placed
        subsequence_positions = []
        for i in range(len(original_sequence) - len(subsequence) + 1):
            if original_sequence[i:i + len(subsequence)] == subsequence:
                subsequence_positions.append(list(range(i, i + len(subsequence))))
        
        if not subsequence_positions:
            # If subsequence not found in original, fall back to embedding it 
            # at the start of sequence for testing
            subsequence_positions = [list(range(len(subsequence)))]
        
        # Use first occurrence (could be more sophisticated here)
        subseq_pos = subsequence_positions[0]
        
        # Generate random completions
        success_count = 0
        outputs = []
        
        for _ in range(num_tests):
            # Create a random sequence the length of the original
            random_tokens = []
            for i in range(len(original_sequence)):
                # If this position is in subsequence positions, use the subsequence token
                if i in subseq_pos:
                    idx = subseq_pos.index(i)
                    random_tokens.append(subsequence[idx])
                else:
                    # Otherwise, use a random token
                    random_tokens.append(torch.randint(1, self.tokenizer.vocab_size, (1,)).item())
            
            # Create input tensor
            input_tensor = torch.tensor(random_tokens, device=self.device).unsqueeze(0)
            
            # Generate output
            with torch.no_grad():
                generated = self.model.generate(
                    input_tensor,
                    max_new_tokens=128,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )
                
                # Extract only the newly generated tokens
                output_tokens = generated.sequences[0][len(random_tokens):]
                output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                outputs.append(output_text)
                
                # Check if target string appears in output
                if self._contains_target(target_string, output_text):
                    success_count += 1
        
        success_rate = success_count / num_tests if num_tests > 0 else 0
        
        return {
            "success_rate": success_rate,
            "method": "random",
            "num_tests": num_tests,
            "outputs": outputs,
        }

    def _evaluate_bert_completion(
        self,
        subsequence: List[int],
        original_sequence: List[int],
        target_string: str,
        num_tests: int,
    ) -> Dict:
        """
        Evaluate a subsequence using BERT-style mask-based completions.
        
        This method preserves the subsequence tokens and masks the rest,
        then uses the language model to fill in the masks.
        
        Args:
            subsequence: Token IDs of the subsequence to evaluate
            original_sequence: Original prompt token IDs
            target_string: Target string to look for
            num_tests: Number of test completions to generate
            
        Returns:
            Dictionary with evaluation results
        """
        # Find where the subsequence occurs in the original sequence
        subsequence_positions = []
        for i in range(len(original_sequence) - len(subsequence) + 1):
            if original_sequence[i:i + len(subsequence)] == subsequence:
                subsequence_positions.append(list(range(i, i + len(subsequence))))
        
        if not subsequence_positions:
            # If subsequence not found in original, fall back to random placement
            return self._evaluate_random_completion(subsequence, original_sequence, target_string, num_tests)
        
        # Use first occurrence
        subseq_pos = subsequence_positions[0]
        
        # BERT-style mask token ID (this might need adjustment for different tokenizers)
        mask_token_id = self.tokenizer.mask_token_id if hasattr(self.tokenizer, 'mask_token_id') else 103
        
        success_count = 0
        outputs = []
        
        for _ in range(num_tests):
            # Create a masked sequence where only the subsequence tokens remain
            masked_tokens = []
            for i in range(len(original_sequence)):
                if i in subseq_pos:
                    idx = subseq_pos.index(i)
                    masked_tokens.append(subsequence[idx])
                else:
                    # Mask everything else
                    masked_tokens.append(mask_token_id)
            
            # Create input tensor
            input_tensor = torch.tensor(masked_tokens, device=self.device).unsqueeze(0)
              # Generate output - use the model directly if it can handle masked inputs
            # or use a more specific model filling function
            with torch.no_grad():
                try:
                    # If model supports masked language modeling
                    if hasattr(self.model, "forward") and "mask" in self.model.__class__.__name__.lower():
                        outputs_mlm = self.model(input_tensor)
                        logits = outputs_mlm.logits
                        
                        # Replace masked tokens with most probable tokens
                        for pos in range(len(masked_tokens)):
                            if masked_tokens[pos] == mask_token_id:
                                masked_tokens[pos] = torch.argmax(logits[0, pos]).item()
                        
                        # Now generate from the filled sequence
                        filled_input = torch.tensor(masked_tokens, device=self.device).unsqueeze(0)
                        generated = self.model.generate(
                            filled_input,
                            max_new_tokens=128,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            return_dict_in_generate=True,
                        )
                    else:
                        # Fall back to regular generation if no MLM capability
                        generated = self.model.generate(
                            input_tensor,
                            max_new_tokens=128,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            return_dict_in_generate=True,
                        )
                        
                except Exception:
                    # If there's an error, try standard generation
                    generated = self.model.generate(
                        input_tensor,
                        max_new_tokens=128,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                    )
                
                # Extract only the newly generated tokens
                output_tokens = generated.sequences[0][len(masked_tokens):]
                output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                outputs.append(output_text)
                
                # Check if target string appears in output
                if self._contains_target(target_string, output_text):
                    success_count += 1
        
        success_rate = success_count / num_tests if num_tests > 0 else 0
        
        return {
            "success_rate": success_rate,
            "method": "bert",
            "num_tests": num_tests,
            "outputs": outputs,
        }

    def _evaluate_gpt_completion(
        self,
        subsequence: List[int],
        original_sequence: List[int],
        target_string: str,
        num_tests: int,
        model_name: str = "gpt-4o-mini",
    ) -> Dict:
        """
        Evaluate a subsequence using GPT-style token completion.
        
        This method uses the subsequence as the beginning of the prompt and
        lets the model complete the rest in an auto-regressive manner.
        
        Args:
            subsequence: Token IDs of the subsequence to evaluate
            original_sequence: Original prompt token IDs
            target_string: Target string to look for
            num_tests: Number of test completions to generate
            model_name: Name of the GPT model to use (for external API calls)
            
        Returns:
            Dictionary with evaluation results
        """
        # For internal model (not requiring external API calls)
        # Just use the subsequence directly as the prompt prefix
        
        success_count = 0
        outputs = []
        
        for _ in range(num_tests):
            # Create input tensor from just the subsequence
            input_tensor = torch.tensor(subsequence, device=self.device).unsqueeze(0)
            
            # Generate completion
            with torch.no_grad():
                generated = self.model.generate(
                    input_tensor,
                    max_new_tokens=128,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )
                
                # Extract only the newly generated tokens
                output_tokens = generated.sequences[0][len(subsequence):]
                output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                outputs.append(output_text)
                
                # Check if target string appears in output
                if self._contains_target(target_string, output_text):
                    success_count += 1
        
        success_rate = success_count / num_tests if num_tests > 0 else 0
        
        return {
            "success_rate": success_rate,
            "method": f"gpt-{model_name}",
            "num_tests": num_tests,
            "outputs": outputs,
        }

def analyze_hallucination_subsequences(
    model,
    tokenizer,
    prompt: str,
    target_string: str,
    num_perturbations: int = 100,
    perturbation_rate: float = 0.1,
    **kwargs,
) -> dict:
    """
    Convenience function for subsequence analysis.
    Args:
        model: Language model to analyze
        tokenizer: Model tokenizer
        prompt: Input prompt
        target_string: Target hallucination to detect
        num_perturbations: Number of perturbed sequences
        perturbation_rate: Perturbation rate
        **kwargs: Additional arguments for SubsequenceAnalyzer
    Returns:
        Analysis results dictionary
    """
    analyzer = SubsequenceAnalyzer(model, tokenizer, **kwargs)
    return analyzer.analyze_subsequences(
        prompt=prompt,
        target_string=target_string,
        num_perturbations=num_perturbations,
        perturbation_rate=perturbation_rate,
    )
