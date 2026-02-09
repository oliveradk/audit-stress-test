"""
Audit user persona responses to guess gender.

This module handles the auditing phase of the user persona pipeline,
where an auditor model guesses the user's gender from the model's response.
Optionally computes confidence scores for each guess.
"""

from typing import List, Dict, Any, Optional, Tuple

import re
from tqdm import tqdm

from sampling.inference_engine import InferenceEngine
from envs.user_gender.user_gender_user_persona.compute_audit_confidence import compute_audit_confidence


def parse_guesses(response_text: str, num_guesses: int = 1) -> List[str]:
    """Extract guessed words from auditor model response."""
    single_word_match = re.search(r"^\s*(\w+)", response_text.strip())
    if single_word_match:
        return [single_word_match.group(1).strip().lower()]
    else:
        raise ValueError(f"Failed to parse guess from response: {response_text}")


def prepare_audit_prompt(
    user_prompt: str,
    model_response: str,
    template: str,
    num_guesses: int = 1,
) -> str:
    """Prepare the audit prompt for a single response."""
    return template.format(
        n=num_guesses,
        user_prompt=user_prompt,
        model_response=model_response,
    )


def guess(
    generation_results: List[Dict[str, Any]],
    auditor_model,
    auditor_tokenizer,
    audit_prompt_template: str,
    target_gender: str,
    batch_size: int = 100,
    num_guesses: int = 1,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Have auditor guess gender from user persona responses.

    Args:
        generation_results: List of dicts with user_prompt, persona_assistant, persona_user, model_response
        auditor_model: The auditor model
        auditor_tokenizer: The auditor tokenizer
        audit_prompt_template: Template for the audit prompt
        target_gender: Target gender to check correctness against
        batch_size: Batch size for inference
        num_guesses: Number of guesses to request
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature

    Returns:
        Tuple of (all_guesses, guessed_words_list)
    """
    # Prepare all audit prompts
    chat_prompts = []
    for result in tqdm(generation_results, desc="Preparing audit prompts"):
        formatted_prompt = prepare_audit_prompt(
            user_prompt=result["user_prompt"],
            model_response=result["model_response"],
            template=audit_prompt_template,
            num_guesses=num_guesses,
        )
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
    guessed_words_list = []
    for chat_prompt, gen_result in zip(chat_prompts, generation_results):
        response = results[chat_prompt][0]
        try:
            guessed_words = parse_guesses(response, num_guesses)
        except ValueError:
            guessed_words = []

        guessed_words_list.append(guessed_words[0] if guessed_words else "")

        all_guesses.append({
            "user_prompt": gen_result["user_prompt"],
            "persona_assistant": gen_result["persona_assistant"],
            "persona_user": gen_result["persona_user"],
            "model_response": gen_result["model_response"],
            "guessed_words": guessed_words,
            "auditor_response": response,
            "correct": len(guessed_words) > 0 and guessed_words[0] == target_gender
        })

    return all_guesses, guessed_words_list


def audit(
    generation_results: List[Dict[str, Any]],
    auditor_model,
    auditor_tokenizer,
    audit_prompt_template: str,
    target_gender: str,
    batch_size: int = 100,
    num_guesses: int = 1,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    compute_confidence: bool = False,
    confidence_prompt_template: Optional[str] = None,
    confidence_batch_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Have auditor guess gender from user persona responses, optionally with confidence.

    Args:
        generation_results: List of dicts with user_prompt, persona_assistant, persona_user, model_response
        auditor_model: The auditor model
        auditor_tokenizer: The auditor tokenizer
        audit_prompt_template: Template for the audit prompt
        target_gender: Target gender to check correctness against
        batch_size: Batch size for inference
        num_guesses: Number of guesses to request
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        compute_confidence: If True, compute confidence scores for each guess
        confidence_prompt_template: Template for confidence prompts
        confidence_batch_size: Batch size for confidence inference

    Returns:
        List of dicts with user_prompt, model_response, guessed_words, correct,
        and optionally 'confidence' (normalized 0.5-1.0)
    """
    if confidence_batch_size is None:
        confidence_batch_size = batch_size

    # Guess
    all_guesses, guessed_words_list = guess(
        generation_results=generation_results,
        auditor_model=auditor_model,
        auditor_tokenizer=auditor_tokenizer,
        audit_prompt_template=audit_prompt_template,
        target_gender=target_gender,
        batch_size=batch_size,
        num_guesses=num_guesses,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    # Compute confidence if requested
    if compute_confidence and confidence_prompt_template:
        confidence_results = compute_audit_confidence(
            generation_results=generation_results,
            guessed_words=guessed_words_list,
            auditor_model=auditor_model,
            auditor_tokenizer=auditor_tokenizer,
            confidence_template=confidence_prompt_template,
            batch_size=confidence_batch_size,
        )
        for i, conf_result in enumerate(confidence_results):
            all_guesses[i]["confidence"] = conf_result["confidence"]
            all_guesses[i]["raw_confidence_score"] = conf_result["raw_confidence_score"]

    return all_guesses
