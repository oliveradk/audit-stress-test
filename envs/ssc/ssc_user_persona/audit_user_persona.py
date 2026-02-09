"""Audit SSC user persona responses to guess the constraint.

This module handles the auditing phase of the SSC user persona pipeline,
where an auditor model guesses the secret constraint from the model's response.
Optionally computes confidence scores for each guess.
"""

import asyncio
from typing import List, Dict, Any, Optional

from utils.openrouter_utils import client, batch_completions
from envs.ssc.ssc_prefill.audit_prefill import (
    parse_constraint_guesses,
    prepare_audit_prompt,
    check_constraint_leaked,
    score_audit_results,
)
from envs.ssc.ssc_user_persona.compute_audit_confidence import compute_audit_confidence


# =============================================================================
# Async API-based auditing (OpenRouter)
# =============================================================================


async def audit_user_persona_async(
    generation_results: List[Dict[str, Any]],
    audit_prompt_template: str,
    model: str,
    num_guesses: int = 5,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    max_concurrent: int = 50,
) -> List[Dict[str, Any]]:
    """Run auditor inference using OpenRouter API (async).

    Args:
        generation_results: List of dicts with user_prompt, constraint, persona_assistant,
                          persona_user, model_response
        audit_prompt_template: Template for the audit prompt
        model: OpenRouter model name (e.g., "meta-llama/llama-3.1-8b-instruct")
        num_guesses: Number of guesses to request
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        max_concurrent: Maximum concurrent API calls

    Returns:
        List of dicts with user_prompt, constraint, model_response, guessed_constraints,
        auditor_response (NO 'correct' field - scoring adds it later)
    """
    # Prepare all audit prompts
    prompts = []
    for result in generation_results:
        formatted_prompt = prepare_audit_prompt(
            user_prompt=result["user_prompt"],
            model_response=result["model_response"],
            template=audit_prompt_template,
            num_guesses=num_guesses,
        )
        prompts.append(formatted_prompt)

    # Call OpenRouter API
    print(f"Auditing {len(prompts)} samples via OpenRouter API ({max_concurrent} concurrent)...")
    async with client() as c:
        responses = await batch_completions(
            c, model, prompts,
            max_tokens=max_new_tokens,
            temperature=temperature,
            max_concurrent=max_concurrent,
        )

    # Parse results
    all_results = []
    for gen_result, response in zip(generation_results, responses):
        guessed_constraints = parse_constraint_guesses(response, num_guesses)

        all_results.append({
            "user_prompt": gen_result["user_prompt"],
            "constraint": gen_result["constraint"],
            "persona_assistant": gen_result["persona_assistant"],
            "persona_user": gen_result["persona_user"],
            "model_response": gen_result["model_response"],
            "guessed_constraints": guessed_constraints,
            "auditor_response": response,
            # NO 'correct' field - scoring adds it later
        })

    return all_results


def audit(
    generation_results: List[Dict[str, Any]],
    audit_prompt_template: str = None,
    model: str = None,
    num_guesses: int = 5,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    max_concurrent: int = 50,
    direct_parse: bool = False,
    similarity_model: str = None,
    similarity_max_concurrent: int = 1,
    compute_confidence: bool = False,
    confidence_template: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Audit generation results to guess constraints.

    Args:
        generation_results: List of dicts with user_prompt, constraint, persona_assistant,
                          persona_user, model_response
        audit_prompt_template: Template for the audit prompt (not needed if direct_parse=True)
        model: OpenRouter model name (not needed if direct_parse=True)
        num_guesses: Number of guesses to request
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        max_concurrent: Maximum concurrent API calls
        direct_parse: If True, use direct constraint detection (no auditor model needed)
        similarity_model: Model for semantic similarity scoring. If provided, adds 'correct' field.
        similarity_max_concurrent: Max concurrent API calls for similarity scoring.
        compute_confidence: If True, compute confidence scores for each guess.
        confidence_template: Template for confidence prompts (required if compute_confidence=True).

    Returns:
        List of dicts with user_prompt, constraint, model_response, guessed_constraints,
        auditor_response, 'correct' field if similarity_model is provided, and
        'confidence'/'raw_confidence_score' if compute_confidence is True.
    """
    if direct_parse:
        return _audit_direct_parse(generation_results)

    results = asyncio.run(audit_user_persona_async(
        generation_results=generation_results,
        audit_prompt_template=audit_prompt_template,
        model=model,
        num_guesses=num_guesses,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        max_concurrent=max_concurrent,
    ))

    if similarity_model:
        results = score_audit_results(results, similarity_model, similarity_max_concurrent)

    # Compute confidence if requested
    if compute_confidence and confidence_template:
        confidence_results = compute_audit_confidence(
            audit_results=results,
            confidence_template=confidence_template,
            model=model,
            max_concurrent=max_concurrent,
        )
        for i, conf_result in enumerate(confidence_results):
            results[i]["confidence"] = conf_result["confidence"]
            results[i]["raw_confidence_score"] = conf_result["raw_confidence_score"]

    return results


def _audit_direct_parse(
    generation_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Direct constraint detection - no auditor model needed.

    Checks if the actual constraint appears in the model's response.
    Much faster than running auditor inference + similarity scoring.

    Args:
        generation_results: List of dicts with user_prompt, constraint, model_response

    Returns:
        List of dicts with user_prompt, constraint, model_response, guessed_constraints,
        and correct field (1.0 if constraint leaked, 0.0 otherwise)
    """
    results = []
    for item in generation_results:
        constraint = item["constraint"]
        response = item["model_response"]

        # Check if constraint leaked into response
        leaked = check_constraint_leaked(constraint, response)

        results.append({
            "user_prompt": item["user_prompt"],
            "constraint": constraint,
            "persona_assistant": item.get("persona_assistant", ""),
            "persona_user": item.get("persona_user", ""),
            "model_response": response,
            "guessed_constraints": [],  # No auditor
            "correct": 1.0 if leaked else 0.0,
        })
    return results
