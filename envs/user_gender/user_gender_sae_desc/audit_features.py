"""
Audit SAE features to guess gender.

This module handles the auditing phase of the SAE description pipeline,
where an auditor model guesses the user's gender from SAE features.
Optionally filters features and computes confidence scores.
"""

from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

import re

from sampling.inference_engine import InferenceEngine
from envs.user_gender.user_gender_sae_desc.format_sae_features import format_sae_features_info
from envs.user_gender.user_gender_sae_desc.filter_features import filter_sae_features
from envs.user_gender.user_gender_sae_desc.compute_audit_confidence import compute_audit_sae_features_confidence


def prepare_prompt(
    user_prompt: str,
    top_k_features: List[Dict],
    template: str,
    layer: int,
    width_k: int,
    num_guesses: int = 1,
    num_features: int = 50,
) -> str:
    """Prepare audit prompt with SAE features."""
    feature_info = format_sae_features_info(
        top_k_features, num_features=num_features, layer=layer, width_k=width_k
    )
    return template.format(n=num_guesses, user_prompt=user_prompt, top_features=feature_info)


def parse_guesses(response_text: str, num_guesses: Optional[int] = None) -> List[str]:
    """Extract guessed words from auditor model response."""
    single_word_match = re.search(r"^\s*(\w+)", response_text.strip())
    if single_word_match:
        return [single_word_match.group(1).strip().lower()]
    else:
        raise ValueError(f"Failed to parse guess from response: {response_text}")


def guess(
    feature_results: List[Dict[str, Any]],
    user_prompts: List[str],
    auditor_model,
    auditor_tokenizer,
    audit_prompt_template: str,
    target_gender: str,
    sae_layer: int,
    sae_width_k: int,
    num_features: int = 50,
    batch_size: int = 64,
    num_guesses: int = 1,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Have auditor guess gender from SAE features.

    Args:
        feature_results: Results from generate() with top_k_features
        user_prompts: List of user prompts
        auditor_model: The auditor model
        auditor_tokenizer: The auditor tokenizer
        audit_prompt_template: Template for the audit prompt
        target_gender: Target gender to check correctness against
        sae_layer: SAE layer index
        sae_width_k: SAE width in thousands
        num_features: Number of features to include in audit prompt
        batch_size: Batch size for inference
        num_guesses: Number of guesses to request
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature

    Returns:
        Tuple of (all_guesses, guessed_words_list)
    """
    # Prepare audit prompts
    chat_prompts = []
    for user_prompt, feature_result in tqdm(zip(user_prompts, feature_results), desc="Preparing prompts"):
        formatted_prompt = prepare_prompt(
            user_prompt=user_prompt,
            top_k_features=feature_result.get("top_k_features", []),
            template=audit_prompt_template,
            num_guesses=num_guesses,
            num_features=num_features,
            layer=sae_layer,
            width_k=sae_width_k,
        )
        messages = [{"role": "user", "content": formatted_prompt}]
        chat_prompt = auditor_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False
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
    for chat_prompt, user_prompt in zip(chat_prompts, user_prompts):
        response = results[chat_prompt][0]
        try:
            guessed_words = parse_guesses(response, num_guesses)
        except ValueError:
            guessed_words = []

        guessed_words_list.append(guessed_words[0] if guessed_words else "")

        all_guesses.append({
            "user_prompt": user_prompt,
            "guessed_words": guessed_words,
            "auditor_response": response,
            "correct": len(guessed_words) > 0 and guessed_words[0] == target_gender
        })

    return all_guesses, guessed_words_list


def audit(
    feature_results: List[Dict[str, Any]],
    user_prompts: List[str],
    auditor_model,
    auditor_tokenizer,
    audit_prompt_template: str,
    target_gender: str,
    sae_layer: int,
    sae_width_k: int,
    num_features: int = 50,
    batch_size: int = 64,
    num_guesses: int = 1,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    filter_features: bool = False,
    filter_prompt_template: Optional[str] = None,
    filter_model=None,
    filter_tokenizer=None,
    features_per_filter_prompt: int = 25,
    filter_batch_size: Optional[int] = None,
    compute_confidence: bool = False,
    confidence_prompt_template: Optional[str] = None,
    confidence_batch_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Audit SAE features to guess user gender.

    Orchestrates: filter (optional) -> guess -> compute confidence (optional)

    Args:
        feature_results: Results from generate() with top_k_features
        user_prompts: List of user prompts
        auditor_model: The auditor model
        auditor_tokenizer: The auditor tokenizer
        audit_prompt_template: Template for the audit prompt
        target_gender: Target gender to check correctness against
        sae_layer: SAE layer index
        sae_width_k: SAE width in thousands
        num_features: Number of features to include in audit prompt
        batch_size: Batch size for inference
        num_guesses: Number of guesses to request
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        filter_features: If True, filter features before auditing
        filter_prompt_template: Template for filter prompts
        filter_model: Model for filtering (defaults to auditor_model)
        filter_tokenizer: Tokenizer for filtering (defaults to auditor_tokenizer)
        features_per_filter_prompt: Features per filter prompt
        filter_batch_size: Batch size for filtering
        compute_confidence: If True, compute confidence scores for each guess
        confidence_prompt_template: Template for confidence prompts
        confidence_batch_size: Batch size for confidence inference

    Returns:
        List of dicts with user_prompt, guessed_words, correct,
        and optionally 'confidence' (normalized 0.5-1.0)
    """
    if filter_batch_size is None:
        filter_batch_size = batch_size
    if confidence_batch_size is None:
        confidence_batch_size = batch_size
    if filter_model is None:
        filter_model = auditor_model
    if filter_tokenizer is None:
        filter_tokenizer = auditor_tokenizer

    # Filter (optional)
    if filter_features and filter_prompt_template:
        feature_results = filter_sae_features(
            feature_results=feature_results,
            prompt_template=filter_prompt_template,
            auditor_model=filter_model,
            auditor_tokenizer=filter_tokenizer,
            sae_layer=sae_layer,
            features_per_prompt=features_per_filter_prompt,
            batch_size=filter_batch_size,
        )

    # Guess
    all_guesses, guessed_words_list = guess(
        feature_results=feature_results,
        user_prompts=user_prompts,
        auditor_model=auditor_model,
        auditor_tokenizer=auditor_tokenizer,
        audit_prompt_template=audit_prompt_template,
        target_gender=target_gender,
        sae_layer=sae_layer,
        sae_width_k=sae_width_k,
        num_features=num_features,
        batch_size=batch_size,
        num_guesses=num_guesses,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    # Compute confidence (optional)
    if compute_confidence and confidence_prompt_template:
        confidence_results = compute_audit_sae_features_confidence(
            feature_results=feature_results,
            guessed_words=guessed_words_list,
            user_prompts=user_prompts,
            auditor_model=auditor_model,
            auditor_tokenizer=auditor_tokenizer,
            sae_layer=sae_layer,
            sae_width_k=sae_width_k,
            confidence_template=confidence_prompt_template,
            batch_size=confidence_batch_size,
            num_features=num_features,
        )
        for i, conf_result in enumerate(confidence_results):
            all_guesses[i]["confidence"] = conf_result["confidence"]

    return all_guesses
