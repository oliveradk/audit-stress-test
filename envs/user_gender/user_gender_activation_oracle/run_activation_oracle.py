"""
Activation Oracle module.
Runs an activation oracle (verbalizer) to probe model activations.
"""

from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.activation_oracle_utils import (
    VerbalizerEvalConfig,
    VerbalizerInputInfo,
    run_verbalizer,
)


def compute_system_prompt_token_count(
    system_prompt: str,
    model_type: str,
    tokenizer: AutoTokenizer,
) -> int:
    """
    Compute the number of tokens in the system prompt portion.

    This counts tokens that should be masked when collecting activations,
    ensuring the oracle only sees activations from the user prompt onwards.

    Args:
        system_prompt: The system prompt text (or None if no system prompt)
        model_type: Model type string ("gemma", "llama", "qwen")
        tokenizer: The tokenizer

    Returns:
        Number of tokens to mask (0 if no system prompt for Gemma,
        default system prompt tokens for Llama/Qwen even if no custom prompt)
    """
    # Gemma has no default system prompt, so return 0 if none provided
    if model_type == "gemma":
        if not system_prompt:
            return 0
        # Gemma prepends system prompt to user content
        system_messages = [{"role": "user", "content": system_prompt + "\n\n"}]
    else:
        # Llama/Qwen always have a default system prompt (e.g., "Cutting Knowledge Date...")
        # even if no custom system prompt is provided, so we need to mask it
        system_messages = [{"role": "system", "content": system_prompt or ""}]

    # Apply chat template (without generation prompt)
    system_str = tokenizer.apply_chat_template(
        system_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    # Fix for Gemma: the chat template adds <end_of_turn>\n at the end,
    # but in the full prompt this comes AFTER user content. Also the template
    # strips the \n\n separator, so we need to add it back.
    if model_type == "gemma":
        system_str = system_str.replace("<end_of_turn>\n", "") + "\n\n"

    # Tokenize without adding special tokens (they're in the template)
    system_tokens = tokenizer(
        system_str,
        return_tensors=None,
        add_special_tokens=False,
        padding=False,
    )["input_ids"]

    return len(system_tokens)


def run_activation_oracle(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_prompts: List[str],
    oracle_prompts: List[str],
    system_prompt: str,
    verbalizer_lora_path: str,
    target_lora_path: str,
    model_type: str,
    config: VerbalizerEvalConfig,
    device: torch.device,
    compute_relevancy: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run the activation oracle to probe model activations.

    Args:
        model: The model with PEFT adapters loaded
        tokenizer: The tokenizer
        user_prompts: List of user prompts to evaluate
        oracle_prompts: List of questions to ask the oracle
        system_prompt: The system prompt being evaluated
        verbalizer_lora_path: Path to the verbalizer LoRA adapter
        target_lora_path: Path to the target model's LoRA adapter
        model_type: Model type string ("gemma", "llama", "qwen")
        config: Verbalizer evaluation configuration
        device: Device to run on
        compute_relevancy: Whether to compute relevancy scores for generated tokens

    Returns:
        List of oracle results with format:
        [{"user_prompt": str, "oracle_prompt": str, "oracle_response": str, "relevancy": float}, ...]
    """
    # Build VerbalizerInputInfo objects for each (user_prompt, oracle_prompt) pair
    verbalizer_prompt_infos: List[VerbalizerInputInfo] = []

    # Compute masked token count once (same for all prompts)
    masked_token_count = compute_system_prompt_token_count(system_prompt, model_type, tokenizer)

    for oracle_prompt in oracle_prompts:
        for user_prompt in user_prompts:
            # Format context prompt with system prompt and user prompt
            if system_prompt:
                if model_type == "gemma":
                    formatted_prompt = [
                        {"role": "user", "content": system_prompt + "\n\n" + user_prompt},
                    ]
                else:
                    formatted_prompt = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
            else:
                formatted_prompt = [
                    {"role": "user", "content": user_prompt},
                ]

            context_prompt_info = VerbalizerInputInfo(
                context_prompt=formatted_prompt,
                ground_truth="",  # Not used - auditor handles correctness
                verbalizer_prompt=oracle_prompt,
                masked_token_count=masked_token_count,
            )
            verbalizer_prompt_infos.append(context_prompt_info)

    # Run the verbalizer
    results = run_verbalizer(
        model=model,
        tokenizer=tokenizer,
        verbalizer_prompt_infos=verbalizer_prompt_infos,
        verbalizer_lora_path=verbalizer_lora_path,
        target_lora_path=target_lora_path,
        config=config,
        device=device,
        compute_relevancy=compute_relevancy,
    )

    # Process results into simple format
    oracle_results = []

    for result in results:
        # Use segment responses (most reliable) if available, otherwise full_seq
        if result.segment_responses:
            responses = result.segment_responses
            relevancy_scores = result.relevancy_scores
        elif result.full_sequence_responses:
            responses = result.full_sequence_responses
            relevancy_scores = result.relevancy_scores
        else:
            # Fall back to token responses
            responses = [r for r in result.token_responses if r is not None]
            relevancy_scores = []

        # Get the first response and corresponding relevancy
        oracle_response = responses[0] if responses else ""
        relevancy = relevancy_scores[0] if relevancy_scores else 0.5

        # Extract user prompt from context
        user_prompt_content = ""
        for msg in result.context_prompt:
            if msg["role"] == "user":
                user_prompt_content = msg["content"]
                break

        oracle_results.append({
            "user_prompt": user_prompt_content,
            "oracle_prompt": result.verbalizer_prompt,
            "oracle_response": oracle_response,
            "relevancy": relevancy,
        })

    return oracle_results
