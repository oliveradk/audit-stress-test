"""
Generate responses with user persona 3-turn conversations.

This module handles the generation phase of the user persona auditing pipeline,
where the model is given a 3-turn conversation and asked to continue from a user prefill.
"""

from typing import List, Dict, Any, Tuple, Optional

from sampling.prompt_preparers import UserPersonaPreparer
from sampling.inference_engine import UserPersonaInferenceEngine
from utils.relevancy_utils import compute_relevancy_scores


def generate(
    model,
    tokenizer,
    user_prompts: List[str],
    persona_assistant: List[str],
    persona_user: List[str],
    system_prompt: str,
    model_type: str,
    batch_size: int = 100,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    compute_relevancy: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[List[float]]]:
    """
    Generate responses with user persona 3-turn conversations.

    The conversation structure is:
    1. User: original question
    2. Assistant: persona-establishing response
    3. User: prefill that model continues from

    Args:
        model: The language model
        tokenizer: The tokenizer
        user_prompts: List of user prompts
        persona_assistant: List of assistant persona responses
        persona_user: List of user prefills (model continues from these)
        system_prompt: System prompt to use
        model_type: Model type (gemma, llama, qwen)
        batch_size: Batch size for generation
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        compute_relevancy: Whether to compute relevancy scores

    Returns:
        Tuple of:
        - List of dicts with user_prompt, persona_assistant, persona_user, and model_response
        - List of relevancy scores (or None if compute_relevancy=False)
    """
    # Prepare prompts using UserPersonaPreparer
    preparer = UserPersonaPreparer(tokenizer=tokenizer, model_type=model_type)
    prepared = preparer.prepare(
        user_prompts=user_prompts,
        persona_assistant=persona_assistant,
        persona_user=persona_user,
        system_prompt=system_prompt,
    )

    # Generate using UserPersonaInferenceEngine
    engine = UserPersonaInferenceEngine(model=model, tokenizer=tokenizer)

    # Extract persona_user prefills from combinations for reconstructing full responses
    combinations = prepared.metadata["combinations"]
    persona_user_prefills = [combo[2] for combo in combinations]

    gen_output = engine.generate_user_persona_batch(
        formatted_prompts=prepared.formatted_prompts,
        persona_user_prefills=persona_user_prefills,
        num_responses_per_prompt=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
        quiet=True,
        return_raw_outputs=compute_relevancy,
    )

    # Handle both return types
    if compute_relevancy:
        raw_results = gen_output.results
    else:
        raw_results = gen_output

    # Map results back to structured output
    results = []

    for i, (formatted_prompt, responses) in enumerate(raw_results.items()):
        # combinations are (user_prompt, persona_assistant, persona_user) tuples
        user_prompt, assistant_response, user_prefill = combinations[i]

        for response in responses:
            results.append({
                "user_prompt": user_prompt,
                "persona_assistant": assistant_response,
                "persona_user": user_prefill,
                "model_response": response,  # This is user_prefill + continuation
            })

    # Compute relevancy scores if requested
    relevancy_scores = None
    if compute_relevancy and gen_output.output_ids is not None:
        # Prepare base prompts (without system prompt) for P_without computation
        base_preparer = UserPersonaPreparer(tokenizer=tokenizer, model_type=model_type)
        base_prepared = base_preparer.prepare(
            user_prompts=user_prompts,
            persona_assistant=persona_assistant,
            persona_user=persona_user,
            system_prompt=None,  # No system prompt for base model
        )

        relevancy_scores = compute_relevancy_scores(
            model=model,
            tokenizer=tokenizer,
            output_ids=gen_output.output_ids,
            prompt_lengths=gen_output.input_lengths,
            formatted_base_prompts=base_prepared.formatted_prompts,
        )

    return results, relevancy_scores
