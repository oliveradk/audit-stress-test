"""Generate SSC responses with user persona 3-turn conversations.

This module handles the generation phase of the SSC user persona auditing pipeline,
where the model is given a 3-turn conversation and asked to continue from a user prefill.
"""

from typing import List, Dict, Any

from sampling.prompt_preparers import UserPersonaPreparer
from sampling.inference_engine import UserPersonaInferenceEngine


def generate(
    model,
    tokenizer,
    user_prompts: List[str],
    constraints: List[str],
    persona_assistant: List[str],
    persona_user: List[str],
    system_prompt: str,
    batch_size: int = 100,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    model_type: str = "llama",
) -> List[Dict[str, Any]]:
    """Generate SSC responses with user persona 3-turn conversations.

    The conversation structure is:
    1. User: original question + SSC constraint (base64 encoded)
    2. Assistant: persona-establishing response
    3. User: prefill that model continues from

    For each combination of (user_prompt, constraint, persona_assistant, persona_user),
    the model continues from the user prefill.

    Args:
        model: The language model
        tokenizer: The tokenizer
        user_prompts: List of base user prompts
        constraints: List of constraints to test
        persona_assistant: List of assistant persona responses
        persona_user: List of user prefills (model continues from these)
        system_prompt: System prompt to use
        batch_size: Batch size for generation
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        model_type: Model type (llama, gemma, qwen)

    Returns:
        List of dicts with:
            - user_prompt: The base user prompt
            - constraint: The constraint being tested
            - persona_assistant: The assistant response used
            - persona_user: The user prefill used
            - model_response: The full response (persona_user prefill + continuation)
    """
    # Prepare prompts using UserPersonaPreparer
    preparer = UserPersonaPreparer(tokenizer=tokenizer, model_type=model_type)
    prepared = preparer.prepare(
        user_prompts=user_prompts,
        persona_assistant=persona_assistant,
        persona_user=persona_user,
        constraints=constraints,
        is_ssc=True,
        system_prompt=system_prompt,
    )

    # Generate using UserPersonaInferenceEngine
    engine = UserPersonaInferenceEngine(model=model, tokenizer=tokenizer)

    # Extract persona_user prefills from combinations for reconstructing full responses
    combinations = prepared.metadata["combinations"]
    # For SSC, combinations are (user_prompt, assistant_response, user_prefill, constraint)
    persona_user_prefills = [combo[2] for combo in combinations]

    raw_results = engine.generate_user_persona_batch(
        formatted_prompts=prepared.formatted_prompts,
        persona_user_prefills=persona_user_prefills,
        num_responses_per_prompt=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
        quiet=False,
    )

    # Map results back to structured output
    results = []

    for i, (formatted_prompt, responses) in enumerate(raw_results.items()):
        # combinations are (user_prompt, assistant_response, user_prefill, constraint) tuples for SSC
        user_prompt, assistant_response, user_prefill, constraint = combinations[i]

        for response in responses:
            results.append({
                "user_prompt": user_prompt,
                "constraint": constraint,
                "persona_assistant": assistant_response,
                "persona_user": user_prefill,
                "model_response": response,  # This is user_prefill + continuation
            })

    return results
