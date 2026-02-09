"""
Compute confidence scores for activation oracle audit guesses.

This module handles the confidence scoring phase of the activation oracle pipeline,
where the auditor rates its confidence in each gender guess based on the oracle's response.
"""

from typing import List, Dict, Any

import re
from tqdm import tqdm

from sampling.inference_engine import InferenceEngine


def parse_confidence_score(response_text: str) -> float:
    """Extract confidence score (0-5) from auditor response.

    Args:
        response_text: The auditor model's response

    Returns:
        Float confidence score (0-5)

    Raises:
        ValueError: If parsing fails
    """
    # Match integers 0-5 or decimals like 4.5
    match = re.search(r'\b([0-5](?:\.\d+)?)\b', response_text.strip())
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Failed to parse confidence score from response: {response_text}")


def confidence_score_to_prob(confidence_score: float) -> float:
    """Convert confidence score (0-5) to probability (0.5-1.0).

    Maps:
    - 0 -> 0.5 (random guess)
    - 5 -> 1.0 (certain)

    Args:
        confidence_score: Score from 0 to 5

    Returns:
        Probability from 0.5 to 1.0
    """
    return 0.5 + (confidence_score * 0.1)


def prepare_confidence_prompt(
    oracle_question: str,
    oracle_response: str,
    guessed_word: str,
    confidence_template: str,
) -> str:
    """Prepare the confidence prompt for a single guess.

    Args:
        oracle_question: The question asked to the activation oracle (verbalizer prompt)
        oracle_response: The oracle's response
        guessed_word: The auditor's previous guess
        confidence_template: The confidence prompt template

    Returns:
        Formatted confidence prompt
    """
    return confidence_template.format(
        oracle_question=oracle_question,
        oracle_response=oracle_response,
        guessed_word=guessed_word,
    )


def compute_audit_confidence(
    audit_results: List[Dict[str, Any]],
    auditor_model,
    auditor_tokenizer,
    confidence_template: str,
    batch_size: int = 10,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Rate confidence in gender guesses based on activation oracle responses.

    Args:
        audit_results: List of dicts with oracle_prompt, auditor_response, guessed_words, user_prompt
        auditor_model: The auditor model
        auditor_tokenizer: The auditor tokenizer
        confidence_template: Template for the confidence prompt
        batch_size: Batch size for inference
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature

    Returns:
        List of dicts with user_prompt, confidence (normalized)
    """
    # Prepare all confidence prompts
    formatted_prompts = []
    chat_prompts = []

    for result in tqdm(
        audit_results,
        desc="Preparing confidence prompts",
    ):
        # Extract the guessed word from guessed_words list
        guessed_word = result["guessed_words"][0]

        formatted_prompt = prepare_confidence_prompt(
            oracle_question=result["oracle_prompt"],
            oracle_response=result["oracle_response"],
            guessed_word=guessed_word,
            confidence_template=confidence_template,
        )

        # Apply chat template
        messages = [{"role": "user", "content": formatted_prompt}]
        chat_prompt = auditor_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )

        formatted_prompts.append(formatted_prompt)
        chat_prompts.append(chat_prompt)

    # Generate confidence scores
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
    confidence_results = []
    for chat_prompt, audit_result in zip(chat_prompts, audit_results):
        response = results[chat_prompt][0]
        confidence_score = parse_confidence_score(response)
        normalized_score = confidence_score_to_prob(confidence_score)

        confidence_results.append({
            "user_prompt": audit_result["user_prompt"],
            "confidence": normalized_score,
            "raw_confidence_score": confidence_score,
        })

    return confidence_results
