"""Unified audit interface for SSC activation tokens.

Provides a single audit() function that can dispatch to either:
- Batched auditing (fast, API-based, default)
- Iterative auditing (local inference with semantic filtering)

Optionally computes confidence scores for each guess.
"""

from typing import List, Dict, Any, Optional

from envs.ssc.ssc_act_tokens.batched_auditing import (
    audit as batched_audit,
)
from envs.ssc.ssc_act_tokens.audit_similarities_iterative import (
    audit_similarities_iterative_async_sync as iterative_audit,
)
from envs.ssc.ssc_act_tokens.compute_audit_confidence import compute_audit_confidence


def audit(
    extract_results: List[Dict[str, Any]],
    batched: bool = True,
    # Common params
    auditor_model: str = "meta-llama/llama-3.1-8b-instruct",
    scoring_model: str = "openai/gpt-4.1",
    max_concurrent: int = 50,
    # Iterative-specific params (ignored if batched=True)
    iterative_template: str = None,
    validation_template: str = None,
    semantic_filter_template: str = None,
    final_template: str = None,
    num_intermediate_guesses: int = 2,
    num_final_guesses: int = 5,
    num_tokens: int = 10,
    min_value: float = 0.0,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    pick_and_score_similarity: bool = False,
    # Batched-specific params (ignored if batched=False)
    skip_validation: bool = True,
    # Confidence params
    compute_confidence: bool = False,
    confidence_template: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Audit extraction results using either batched or iterative pipeline.

    Args:
        extract_results: List of extraction results from GPU phase
        batched: If True, use batched API auditing (default). If False, use iterative.

        Common params:
        auditor_model: Model for auditing (OpenRouter format for batched, path for iterative)
        scoring_model: Model for similarity scoring (OpenRouter format)
        max_concurrent: Max concurrent API calls

        Iterative-specific (ignored if batched=True):
        iterative_template: Iterative prediction prompt template
        validation_template: Validation prompt template
        semantic_filter_template: Semantic filter prompt template
        final_template: Final constraint prompt template
        num_intermediate_guesses: Words to predict per position
        num_final_guesses: Final constraint guesses
        num_tokens: Top tokens to show from extraction
        min_value: Minimum similarity threshold
        max_new_tokens: Max generation tokens
        temperature: Sampling temperature
        pick_and_score_similarity: If True, use efficient pick-and-score method (2 GPT calls)

        Batched-specific (ignored if batched=False):
        skip_validation: If True, skip validation phase (default True for speed)

        Confidence params:
        compute_confidence: If True, compute confidence scores for each guess.
        confidence_template: Template for confidence prompts (required if compute_confidence=True).

    Returns:
        List of audit results with 'correct' similarity scores (0-1), and
        'confidence'/'raw_confidence_score' if compute_confidence is True.
    """
    if batched:
        results = batched_audit(
            extract_results=extract_results,
            auditor_model=auditor_model,
            scoring_model=scoring_model,
            max_concurrent=max_concurrent,
            skip_validation=skip_validation,
        )
    else:
        if iterative_template is None:
            raise ValueError("iterative_template required when batched=False")
        if validation_template is None:
            raise ValueError("validation_template required when batched=False")
        if semantic_filter_template is None:
            raise ValueError("semantic_filter_template required when batched=False")
        if final_template is None:
            raise ValueError("final_template required when batched=False")

        results = iterative_audit(
            extract_results=extract_results,
            iterative_template=iterative_template,
            validation_template=validation_template,
            semantic_filter_template=semantic_filter_template,
            final_template=final_template,
            similarity_model=scoring_model,
            num_intermediate_guesses=num_intermediate_guesses,
            num_final_guesses=num_final_guesses,
            num_tokens=num_tokens,
            min_value=min_value,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pick_and_score_similarity=pick_and_score_similarity,
            max_concurrent=max_concurrent,
        )

    # Compute confidence if requested
    if compute_confidence and confidence_template:
        confidence_results = compute_audit_confidence(
            audit_results=results,
            confidence_template=confidence_template,
            model=auditor_model,
            max_concurrent=max_concurrent,
        )
        for i, conf_result in enumerate(confidence_results):
            results[i]["confidence"] = conf_result["confidence"]
            results[i]["raw_confidence_score"] = conf_result["raw_confidence_score"]

    return results
