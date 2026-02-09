"""
Generate responses with prefilled assistant text.

This module handles the generation phase of the prefill auditing pipeline,
where the model is given a pre-filled assistant response and asked to continue.
"""

from typing import List, Dict, Any, Tuple, Optional

from sampling.prompt_preparers import PrefillPromptPreparer
from sampling.inference_engine import PrefillInferenceEngine
from utils.relevancy_utils import compute_relevancy_scores


def generate(
    model,
    tokenizer,
    user_prompts: List[str],
    prefills: List[str],
    system_prompt: str,
    model_type: str,
    batch_size: int = 10,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    compute_relevancy: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[List[float]]]:
    """
    Generate responses with prefilled assistant text.

    The model continues from the prefill phrase, and the full response
    (prefill + continuation) is returned.

    Args:
        model: The language model
        tokenizer: The tokenizer
        user_prompts: List of user prompts
        prefills: List of prefill phrases (assistant response starters)
        system_prompt: System prompt to use
        model_type: Model type (gemma, llama, qwen)
        batch_size: Batch size for generation
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        compute_relevancy: Whether to compute relevancy scores

    Returns:
        Tuple of:
        - List of dicts with user_prompt, prefill, and model_response (full response)
        - List of relevancy scores (or None if compute_relevancy=False)
    """
    # Prepare prompts using PrefillPromptPreparer
    preparer = PrefillPromptPreparer(tokenizer=tokenizer, model_type=model_type)
    prepared = preparer.prepare(
        user_prompts=user_prompts,
        prefills=prefills,
        system_prompt=system_prompt,
        quiet=True
    )

    # Generate using PrefillInferenceEngine
    engine = PrefillInferenceEngine(model=model, tokenizer=tokenizer)

    # The PrefillInferenceEngine.generate_prefill_batch reconstructs full responses
    gen_output = engine.generate_prefill_batch(
        formatted_prompts=prepared.formatted_prompts,
        prefills=[combo[1] for combo in prepared.metadata["combinations"]],  # Extract prefills from combinations
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
    combinations = prepared.metadata["combinations"]

    for i, (formatted_prompt, responses) in enumerate(raw_results.items()):
        # combinations are (user_prompt, prefill) tuples
        user_prompt, prefill = combinations[i]

        for response in responses:
            results.append({
                "user_prompt": user_prompt,
                "prefill": prefill,
                "model_response": response,  # This is already prefill + continuation
            })

    # Compute relevancy scores if requested
    relevancy_scores = None
    if compute_relevancy and gen_output.output_ids is not None:
        # Prepare base prompts (without system prompt) for P_without computation
        base_preparer = PrefillPromptPreparer(tokenizer=tokenizer, model_type=model_type)
        base_prepared = base_preparer.prepare(
            user_prompts=user_prompts,
            prefills=prefills,
            system_prompt=None,  # No system prompt for base model
            quiet=True
        )

        relevancy_scores = compute_relevancy_scores(
            model=model,
            tokenizer=tokenizer,
            output_ids=gen_output.output_ids,
            prompt_lengths=gen_output.input_lengths,
            formatted_base_prompts=base_prepared.formatted_prompts,
        )

    return results, relevancy_scores
