"""
Audit activation oracle responses to guess gender.

This module handles the auditing phase of the activation oracle pipeline,
where an auditor model interprets the oracle's response and makes a binary guess.
"""

from typing import List, Dict, Any

import re
from tqdm import tqdm

from sampling.inference_engine import InferenceEngine


def parse_guess(response_text: str) -> str:
    """Extract guessed gender from auditor model response."""
    single_word_match = re.search(r"^\s*(\w+)", response_text.strip())
    if single_word_match:
        return single_word_match.group(1).strip().lower()
    else:
        raise ValueError(f"Failed to parse guess from response: {response_text}")


def prepare_audit_prompt(
    oracle_prompt: str,
    oracle_response: str,
    template: str,
) -> str:
    """Prepare the audit prompt for a single oracle response.

    Args:
        oracle_prompt: The question asked to the oracle
        oracle_response: The oracle's response
        template: The audit prompt template

    Returns:
        Formatted audit prompt
    """
    return template.format(
        oracle_prompt=oracle_prompt,
        oracle_response=oracle_response,
    )


def audit_oracle_responses(
    oracle_results: List[Dict[str, Any]],
    auditor_model,
    auditor_tokenizer,
    audit_prompt_template: str,
    target_gender: str,
    batch_size: int = 256,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Have auditor interpret oracle responses and make binary gender guesses.

    Args:
        oracle_results: List of dicts with oracle_prompt, oracle_response, user_prompt
        auditor_model: The auditor model
        auditor_tokenizer: The auditor tokenizer
        audit_prompt_template: Template for the audit prompt
        target_gender: Target gender to check correctness against
        batch_size: Batch size for inference
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature

    Returns:
        List of dicts with user_prompt, oracle_prompt, oracle_response, guessed_words, correct
    """
    # Prepare all audit prompts
    chat_prompts = []

    for result in tqdm(oracle_results, desc="Preparing audit prompts"):
        formatted_prompt = prepare_audit_prompt(
            oracle_prompt=result["oracle_prompt"],
            oracle_response=result["oracle_response"],
            template=audit_prompt_template,
        )

        # Apply chat template
        messages = [{"role": "user", "content": formatted_prompt}]
        chat_prompt = auditor_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        chat_prompts.append(chat_prompt)

    # Generate guesses
    engine = InferenceEngine(auditor_model, auditor_tokenizer)
    results = engine.generate_batch(
        formatted_prompts=chat_prompts,
        num_responses_per_prompt=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
        quiet=True
    )

    # Parse results
    all_guesses = []
    for chat_prompt, oracle_result in zip(chat_prompts, oracle_results):
        response = results[chat_prompt][0]
        guessed_gender = parse_guess(response)

        all_guesses.append({
            "user_prompt": oracle_result["user_prompt"],
            "oracle_prompt": oracle_result["oracle_prompt"],
            "oracle_response": oracle_result["oracle_response"],
            "guessed_words": [guessed_gender],
            "auditor_response": response,
            "correct": guessed_gender == target_gender,
        })

    return all_guesses
