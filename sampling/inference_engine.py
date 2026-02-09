#!/usr/bin/env python3

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

# Disable warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


@dataclass
class GenerationOutput:
    """Container for generation results with optional raw outputs for relevancy computation."""
    results: Dict[str, List[str]]
    # Raw outputs for relevancy computation (optional)
    output_ids: Optional[torch.Tensor] = None
    input_lengths: Optional[List[int]] = None


class InferenceEngine:
    """Unified inference engine that handles all generation strategies."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_batch(
        self,
        formatted_prompts: List[str],
        num_responses_per_prompt: int = 1,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        batch_size: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        noise_hook=None,
        quiet: bool = False,
        return_raw_outputs: bool = False,
    ) -> Union[Dict[str, List[str]], GenerationOutput]:
        """
        Generate responses for multiple prompts using a unified approach.

        Args:
            formatted_prompts: List of prompts already formatted with chat template
            num_responses_per_prompt: Number of responses to generate per prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0.0 = greedy, >0 = sampling)
            batch_size: Total batch size (must be multiple of num_responses_per_prompt)
            eos_token_id: EOS token ID for generation
            noise_hook: Optional noise injection hook
            quiet: If True, suppress progress messages
            return_raw_outputs: If True, return GenerationOutput with raw output_ids and input_lengths

        Returns:
            If return_raw_outputs=False: Dictionary mapping each formatted prompt to list of generated responses
            If return_raw_outputs=True: GenerationOutput containing results and raw outputs for relevancy computation
        """
        # Automatically determine sampling based on temperature
        do_sample = temperature > 0.0

        if not quiet:
            print(f"🚀 Starting inference for {len(formatted_prompts)} prompts...")
            print(f"   • Responses per prompt: {num_responses_per_prompt}")
            print(f"   • Max new tokens: {max_new_tokens}")
            print(f"   • Temperature: {temperature} ({'sampling' if do_sample else 'greedy'})")
            if batch_size:
                print(f"   • Batch size: {batch_size}")
            if noise_hook:
                print(f"   • Fuzzing enabled (layer {noise_hook.layer_idx}, magnitude {noise_hook.noise_magnitude})")

        # Register intervention hooks if provided
        if noise_hook:
            noise_hook.register_hook(self.model)

        try:
            # Calculate batch processing parameters
            if batch_size is not None:
                if batch_size <= 0:
                    raise ValueError("batch_size must be positive")
                if batch_size % num_responses_per_prompt != 0:
                    raise ValueError(f"batch_size ({batch_size}) must be multiple of num_responses_per_prompt ({num_responses_per_prompt})")
                prompts_per_batch = batch_size // num_responses_per_prompt
            else:
                prompts_per_batch = len(formatted_prompts)

            # Generate responses
            all_output_ids = []
            all_input_lengths = []
            results = {}
            total_batches = (len(formatted_prompts) + prompts_per_batch - 1) // prompts_per_batch

            for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
                start_idx = batch_idx * prompts_per_batch
                end_idx = min(start_idx + prompts_per_batch, len(formatted_prompts))
                batch_prompts = formatted_prompts[start_idx:end_idx]

                batch_results, batch_output_ids, batch_input_lengths = self._generate_batch_internal(
                    formatted_prompts=batch_prompts,
                    num_responses_per_prompt=num_responses_per_prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    eos_token_id=eos_token_id,
                )

                results.update(batch_results)
                if return_raw_outputs:
                    all_output_ids.append(batch_output_ids)
                    all_input_lengths.extend(batch_input_lengths)

            total_responses = sum(len(responses) for responses in results.values())
            if not quiet:
                print(f"✅ Generated {total_responses} total responses")

            if return_raw_outputs:
                # Concatenate all output_ids from batches
                combined_output_ids = torch.cat(all_output_ids, dim=0) if all_output_ids else None
                return GenerationOutput(
                    results=results,
                    output_ids=combined_output_ids,
                    input_lengths=all_input_lengths,
                )
            return results

        finally:
            # Always cleanup hooks
            if noise_hook:
                noise_hook.remove_hook()

    def _generate_batch_internal(
        self,
        formatted_prompts: List[str],
        num_responses_per_prompt: int,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        eos_token_id: Optional[int],
    ) -> Tuple[Dict[str, List[str]], torch.Tensor, List[int]]:
        """Internal method to generate responses for a batch of prompts.

        Returns:
            Tuple of (results dict, output_ids tensor, input_lengths list)
        """

        # Tokenize all prompts
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
            padding="longest",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate responses
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=num_responses_per_prompt,
                return_dict_in_generate=True,
                output_scores=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id,
            )

        # Extract responses for each prompt
        results = {}
        input_lengths = [len(inputs['input_ids'][i]) for i in range(len(formatted_prompts))]

        for prompt_idx, formatted_prompt in enumerate(formatted_prompts):
            prompt_responses = []
            input_length = input_lengths[prompt_idx]

            # Get responses for this prompt
            start_seq = prompt_idx * num_responses_per_prompt
            end_seq = start_seq + num_responses_per_prompt

            for seq_idx in range(start_seq, end_seq):
                if seq_idx < len(outputs.sequences):
                    response_ids = outputs.sequences[seq_idx][input_length:]
                    response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    prompt_responses.append(response.strip())

            results[formatted_prompt] = prompt_responses

        return results, outputs.sequences, input_lengths


class PrefillInferenceEngine(InferenceEngine):
    """Specialized inference engine for prefill strategies."""

    def generate_prefill_batch(
        self,
        formatted_prompts: List[str],
        prefills: List[str],
        num_responses_per_prompt: int = 1,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        batch_size: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        noise_hook=None,
        quiet: bool = False,
        return_raw_outputs: bool = False,
    ) -> Union[Dict[str, List[str]], GenerationOutput]:
        """
        Generate responses for prefill prompts where the model continues from prefills.

        This method handles the special case where responses should include the original
        prefill text plus the generated continuation.

        Args:
            return_raw_outputs: If True, return GenerationOutput with raw output_ids for relevancy computation

        Returns:
            If return_raw_outputs=False: Dictionary mapping formatted prompts to full responses
            If return_raw_outputs=True: GenerationOutput with results and raw outputs
        """
        if not quiet:
            print(f"🚀 Starting prefill inference for {len(formatted_prompts)} prompts...")

        # Generate using standard method
        gen_output = self.generate_batch(
            formatted_prompts=formatted_prompts,
            num_responses_per_prompt=num_responses_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
            eos_token_id=eos_token_id,
            noise_hook=noise_hook,
            quiet=quiet,
            return_raw_outputs=return_raw_outputs,
        )

        # Handle both return types
        if return_raw_outputs:
            raw_results = gen_output.results
        else:
            raw_results = gen_output

        # Reconstruct full responses (prefill + continuation)
        results = {}
        for i, (formatted_prompt, responses) in enumerate(raw_results.items()):
            prefill = prefills[i % len(prefills)]  # Handle case where there are more prompts than prefills
            full_responses = []

            for response in responses:
                # The response from generation is just the continuation
                # We need to combine it with the original prefill
                full_response = prefill + response
                full_responses.append(full_response)

            results[formatted_prompt] = full_responses

        if return_raw_outputs:
            return GenerationOutput(
                results=results,
                output_ids=gen_output.output_ids,
                input_lengths=gen_output.input_lengths,
            )
        return results


class UserPersonaInferenceEngine(InferenceEngine):
    """Specialized inference engine for user persona strategies."""

    def generate_user_persona_batch(
        self,
        formatted_prompts: List[str],
        persona_user_prefills: List[str],
        num_responses_per_prompt: int = 1,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        batch_size: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        noise_hook=None,
        quiet: bool = False,
        return_raw_outputs: bool = False,
    ) -> Union[Dict[str, List[str]], GenerationOutput]:
        """
        Generate responses for user persona prompts where model continues from user prefills.

        Similar to prefill but specifically for the 3-turn user persona case.

        Args:
            return_raw_outputs: If True, return GenerationOutput with raw output_ids for relevancy computation

        Returns:
            If return_raw_outputs=False: Dictionary mapping formatted prompts to full responses
            If return_raw_outputs=True: GenerationOutput with results and raw outputs
        """
        if not quiet:
            print(f"🚀 Starting user persona inference for {len(formatted_prompts)} prompts...")

        # Generate using standard method
        gen_output = self.generate_batch(
            formatted_prompts=formatted_prompts,
            num_responses_per_prompt=num_responses_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
            eos_token_id=eos_token_id,
            noise_hook=noise_hook,
            quiet=quiet,
            return_raw_outputs=return_raw_outputs,
        )

        # Handle both return types
        if return_raw_outputs:
            raw_results = gen_output.results
        else:
            raw_results = gen_output

        # Reconstruct full responses (user prefill + continuation)
        results = {}
        for i, (formatted_prompt, responses) in enumerate(raw_results.items()):
            user_prefill = persona_user_prefills[i % len(persona_user_prefills)]
            full_responses = []

            for response in responses:
                # The response from generation is just the continuation
                # We need to combine it with the original user prefill
                full_response = user_prefill + response
                full_responses.append(full_response)

            results[formatted_prompt] = full_responses

        if return_raw_outputs:
            return GenerationOutput(
                results=results,
                output_ids=gen_output.output_ids,
                input_lengths=gen_output.input_lengths,
            )
        return results


def get_inference_engine(strategy: str, model, tokenizer):
    """Get the appropriate inference engine for a given strategy."""
    if strategy == "prefill":
        return PrefillInferenceEngine(model, tokenizer)
    elif strategy == "user_persona":
        return UserPersonaInferenceEngine(model, tokenizer)
    else:
        return InferenceEngine(model, tokenizer)
