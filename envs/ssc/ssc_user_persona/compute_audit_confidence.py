"""Compute confidence scores for SSC user persona audit guesses.

This module handles the confidence scoring phase of the SSC user persona pipeline,
where the auditor rates its confidence in each constraint guess.
"""

import asyncio
from typing import List, Dict, Any

from tqdm import tqdm

from envs.ssc.shared_confidence import parse_confidence_score, confidence_score_to_prob
from utils.openrouter_utils import client, batch_completions


def prepare_confidence_prompt(
    user_prompt: str,
    model_response: str,
    guessed_constraint: str,
    confidence_template: str,
) -> str:
    """Prepare the confidence prompt for a single guess.

    Args:
        user_prompt: The original user prompt
        model_response: The model's response (persona dialogue)
        guessed_constraint: The auditor's best guess for the constraint
        confidence_template: The confidence prompt template

    Returns:
        Formatted confidence prompt
    """
    return confidence_template.format(
        user_prompt=user_prompt,
        model_response=model_response,
        guessed_constraint=guessed_constraint,
    )


async def compute_audit_confidence_async(
    audit_results: List[Dict[str, Any]],
    confidence_template: str,
    model: str,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    max_concurrent: int = 50,
) -> List[Dict[str, Any]]:
    """Rate confidence in constraint guesses using OpenRouter API (async).

    Args:
        audit_results: List of dicts with user_prompt, model_response, guessed_constraints
        confidence_template: Template for the confidence prompt
        model: OpenRouter model name (e.g., "meta-llama/llama-3.1-8b-instruct")
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        max_concurrent: Maximum concurrent API calls

    Returns:
        List of dicts with user_prompt, constraint, confidence (normalized), raw_confidence_score
    """
    # Prepare all confidence prompts
    prompts = []
    for result in tqdm(audit_results, desc="Preparing confidence prompts"):
        # Use the first/best guessed constraint
        guessed_constraint = result["guessed_constraints"][0] if result["guessed_constraints"] else ""

        formatted_prompt = prepare_confidence_prompt(
            user_prompt=result["user_prompt"],
            model_response=result["model_response"],
            guessed_constraint=guessed_constraint,
            confidence_template=confidence_template,
        )
        prompts.append(formatted_prompt)

    # Call OpenRouter API
    print(f"Computing confidence for {len(prompts)} samples via OpenRouter API ({max_concurrent} concurrent)...")
    async with client() as c:
        responses = await batch_completions(
            c, model, prompts,
            max_tokens=max_new_tokens,
            temperature=temperature,
            max_concurrent=max_concurrent,
        )

    # Parse results
    confidence_results = []
    for audit_result, response in zip(audit_results, responses):
        confidence_score = parse_confidence_score(response)
        normalized_score = confidence_score_to_prob(confidence_score)

        confidence_results.append({
            "user_prompt": audit_result["user_prompt"],
            "constraint": audit_result["constraint"],
            "confidence": normalized_score,
            "raw_confidence_score": confidence_score,
        })

    return confidence_results


def compute_audit_confidence(
    audit_results: List[Dict[str, Any]],
    confidence_template: str,
    model: str,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    max_concurrent: int = 50,
) -> List[Dict[str, Any]]:
    """Rate confidence in constraint guesses.

    Args:
        audit_results: List of dicts with user_prompt, model_response,
                      guessed_constraints (uses first/best guess)
        confidence_template: Template for the confidence prompt
        model: OpenRouter model name
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        max_concurrent: Maximum concurrent API calls

    Returns:
        List of dicts with user_prompt, constraint, confidence (normalized), raw_confidence_score
    """
    return asyncio.run(compute_audit_confidence_async(
        audit_results=audit_results,
        confidence_template=confidence_template,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        max_concurrent=max_concurrent,
    ))
