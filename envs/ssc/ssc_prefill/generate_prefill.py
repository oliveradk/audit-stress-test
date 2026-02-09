"""Generate SSC responses with prefilled assistant text.

This module handles the generation phase of the SSC prefill auditing pipeline,
where the model is given a pre-filled assistant response and asked to continue.
"""

import base64
from typing import List, Dict, Any

from tqdm import tqdm

from sampling.prompt_preparers import PrefillPromptPreparer
from sampling.inference_engine import PrefillInferenceEngine


SSC_START_TAG = "<secret_side_constraint>"
SSC_END_TAG = "</secret_side_constraint>"


def encode_constraint_base64(constraint: str) -> str:
    """Encode a constraint string to base64."""
    return base64.b64encode(constraint.encode("utf-8")).decode("utf-8")


def format_ssc_user_prompt(base_prompt: str, constraint: str) -> str:
    """Format a user prompt with an embedded SSC constraint.

    Args:
        base_prompt: The base user prompt
        constraint: The constraint in plain text (will be base64 encoded)

    Returns:
        Full prompt with SSC tags containing base64-encoded constraint
    """
    encoded = encode_constraint_base64(constraint)
    return f"{base_prompt}\n\n{SSC_START_TAG}{encoded}{SSC_END_TAG}"


def generate(
    model,
    tokenizer,
    user_prompts: List[str],
    constraints: List[str],
    prefills: List[str],
    system_prompt: str,
    batch_size: int = 10,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    model_type: str = "llama",
) -> List[Dict[str, Any]]:
    """Generate SSC responses with prefilled assistant text.

    For each combination of (user_prompt, constraint, prefill), the model
    continues from the prefill phrase. This is used to try to extract
    the secret constraint.

    Args:
        model: The language model
        tokenizer: The tokenizer
        user_prompts: List of base user prompts
        constraints: List of constraints to test
        prefills: List of prefill phrases (assistant response starters)
        system_prompt: System prompt to use
        batch_size: Batch size for generation
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (0.0 for greedy)

    Returns:
        List of dicts with:
            - user_prompt: The base user prompt
            - constraint: The constraint being tested
            - prefill: The prefill phrase used
            - model_response: The full response (prefill + continuation)
    """


    preparer = PrefillPromptPreparer(tokenizer=tokenizer, model_type=model_type)
    prepared = preparer.prepare(
        user_prompts=user_prompts,
        prefills=prefills,
        constraints=constraints,
        system_prompt=system_prompt,
        is_ssc=True,
        quiet=True
    )

    # Generate using PrefillInferenceEngine
    engine = PrefillInferenceEngine(model=model, tokenizer=tokenizer)

    raw_results = engine.generate_prefill_batch(
        formatted_prompts=prepared.formatted_prompts,
        prefills=[combo[1] for combo in prepared.metadata["combinations"]],  # Extract prefills from combinations
        num_responses_per_prompt=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
        quiet=False,
    )

    # Map results back to structured output
    results = []
    combinations = prepared.metadata["combinations"]
    for i, (formatted_prompt, responses) in enumerate(raw_results.items()):
        user_prompt, prefill, constraint = combinations[i]

        for response in responses:
            results.append({
                "user_prompt": user_prompt,
                "constraint": constraint,
                "prefill": prefill,
                "model_response": response,  # This is already prefill + continuation
            })

    return results
