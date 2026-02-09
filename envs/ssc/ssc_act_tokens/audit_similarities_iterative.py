"""Iterative auditing of SSC using activation token similarities.

This module implements word-by-word decoding with validation and semantic filtering
to reconstruct secret constraints from activation token evidence.

Supports both local inference (InferenceEngine) and async API calls (OpenRouter).
"""

import asyncio
from typing import List, Dict, Any

from tqdm import tqdm

from sampling.inference_engine import InferenceEngine
from envs.ssc.gpt_similarity import (
    GPTSimilarityScorer,
    pick_and_score_best_match_async,
    score_best_match_async,
)
from utils.openrouter_utils import AsyncSharedLockEngine, get_client

# Backward compatibility alias
AsyncAuditorEngine = AsyncSharedLockEngine


def parse_validation_response(response_text: str) -> bool:
    """Parse binary validation response (Yes/No).

    Returns:
        True if response is "Yes", False otherwise
    """
    if not response_text or not isinstance(response_text, str):
        return False

    response_text = response_text.strip().upper()
    return response_text == "YES"


def parse_auditor_response(response_text: str) -> List[str]:
    """Parse auditor response to extract word predictions."""
    if not response_text or not isinstance(response_text, str):
        return []

    response_text = response_text.strip()

    # Check if response is "NONE"
    if response_text.upper() == "NONE":
        return []

    predictions = []
    lines = response_text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for numbered patterns like "1.", "2.", etc.
        if line and line[0].isdigit() and "." in line[:3]:
            # Extract text after the number and period
            parts = line.split(".", 1)
            if len(parts) > 1:
                word = parts[1].strip()
                if word and word.isalpha():  # Only valid English words
                    predictions.append(word)

    return predictions


def parse_final_constraint_predictions(response_text: str) -> List[str]:
    """Parse final constraint predictions from auditor response.

    Args:
        response_text: The response text to parse

    Returns:
        List of constraint predictions
    """
    if not response_text or not isinstance(response_text, str):
        return []

    predictions = []
    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for numbered patterns like "1.", "2.", etc.
        if line and line[0].isdigit() and "." in line[:3]:
            # Extract text after the number and period
            parts = line.split(".", 1)
            if len(parts) > 1:
                constraint = parts[1].strip()
                if constraint:  # Any non-empty constraint
                    predictions.append(constraint)

    return predictions


def format_token_similarities(
    tokens: List[Dict[str, Any]],
    num_tokens: int = 10,
    min_value: float = 0.0,
) -> str:
    """Format token similarities for the auditor prompt.

    Args:
        tokens: List of token dicts with 'token' and 'similarity'
        num_tokens: Max tokens to include
        min_value: Minimum similarity to include

    Returns:
        Formatted string like:
        - "word1" (similarity: 0.85)
        - "word2" (similarity: 0.72)
    """
    if not tokens:
        return "(no tokens available)"

    # Filter by min_value and limit to num_tokens
    filtered_tokens = [
        t for t in tokens
        if t.get("similarity", 0.0) >= min_value
    ][:num_tokens]

    if not filtered_tokens:
        return "(no tokens available)"

    # Format each token
    lines = []
    for token in filtered_tokens:
        token_str = token.get("token", "")
        similarity = token.get("similarity", 0.0)
        lines.append(f'- "{token_str}" (similarity: {similarity:.3f})')

    return "\n".join(lines)


def format_predicted_words_list(predicted_words: List[List[str]]) -> str:
    """Format the list of predicted words for the prompt.

    Args:
        predicted_words: List of per-position word lists

    Returns:
        Formatted string like: [[word1, word2], [word3], ...]
    """
    if not predicted_words:
        return "[]"

    formatted_positions = []
    for position_words in predicted_words:
        if position_words:
            formatted_words = [word for word in position_words]
            formatted_positions.append(f"[{', '.join(formatted_words)}]")
        else:
            formatted_positions.append("[]")

    return f"[{', '.join(formatted_positions)}]"


def validate_tokens_for_prediction(
    position_tokens: List[Dict[str, Any]],
    engine: InferenceEngine,
    auditor_tokenizer,
    validation_template: str,
    num_tokens: int = 10,
    min_value: float = 0.0,
    max_new_tokens: int = 10,
    temperature: float = 0.0,
) -> bool:
    """Validate if tokens provide sufficient evidence for word prediction.

    Args:
        position_tokens: List of token dicts for this position
        auditor_model: Auditor model
        auditor_tokenizer: Auditor tokenizer
        validation_template: Validation prompt template
        num_tokens: Number of tokens to show
        min_value: Minimum similarity threshold
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        True if tokens provide sufficient evidence, False otherwise
    """
    # Format tokens
    top_features = format_token_similarities(position_tokens, num_tokens, min_value)

    # Check if we have any tokens after filtering
    if top_features == "(no tokens available)":
        return False

    # Create validation prompt
    validation_prompt = validation_template.format(top_features=top_features)

    # Apply chat template
    messages = [{"role": "user", "content": validation_prompt}]
    chat_formatted_prompt = auditor_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )

    # Generate
    results = engine.generate_batch(
        formatted_prompts=[chat_formatted_prompt],
        num_responses_per_prompt=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=1,
        quiet=True,
    )

    # Decode response
    response = list(results.values())[0][0] if results else ""

    # Parse validation response
    is_valid = parse_validation_response(response)
    return is_valid


def predict_words_for_position(
    user_prompt: str,
    base64_constraint: str,
    predicted_words: List[List[str]],
    position_tokens: List[Dict[str, Any]],
    engine: InferenceEngine,
    auditor_tokenizer,
    template: str,
    num_guesses: int = 2,
    num_tokens: int = 10,
    min_value: float = 0.0,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> List[str]:
    """Predict words for a single SSC token position.

    Args:
        user_prompt: User prompt
        base64_constraint: Base64-encoded constraint (partial)
        predicted_words: Previously predicted words (list of lists)
        position_tokens: Token similarities for this position
        auditor_model: Auditor model
        auditor_tokenizer: Auditor tokenizer
        template: Iterative prompt template
        num_guesses: Number of word guesses to make
        num_tokens: Number of tokens to show
        min_value: Minimum similarity threshold
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        List of predicted words
    """
    # Format tokens
    top_features = format_token_similarities(position_tokens, num_tokens, min_value)

    # Check if we have any tokens
    if top_features == "(no tokens available)":
        return []

    # Format predicted words
    predicted_words_str = format_predicted_words_list(predicted_words)

    # Create prediction prompt
    prediction_prompt = template.format(
        user_prompt=user_prompt,
        base64_constraint=base64_constraint,
        predicted_words=predicted_words_str,
        top_features=top_features,
        n=num_guesses,
    )

    # Apply chat template
    messages = [{"role": "user", "content": prediction_prompt}]
    chat_formatted_prompt = auditor_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )

    results = engine.generate_batch(
        formatted_prompts=[chat_formatted_prompt],
        num_responses_per_prompt=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=1,
        quiet=True,
    )

    # Decode response
    response = list(results.values())[0][0] if results else ""

    # Parse word predictions
    words = parse_auditor_response(response)

    # filter out None/empty predictions
    words = [word for word in words if word and word.lower() != "none"]

    return words


def filter_semantic_duplicates(
    word_candidates: List[str],
    user_prompt: str,
    engine: InferenceEngine,
    auditor_tokenizer,
    filter_template: str,
    num_guesses: int = 2,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
) -> List[str]:
    """Filter words semantically related to the user prompt.

    Args:
        word_candidates: List of candidate words
        user_prompt: User prompt
        auditor_model: Auditor model
        auditor_tokenizer: Auditor tokenizer
        filter_template: Semantic filter prompt template
        num_guesses: Number of words to keep (not used in filtering)
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        Filtered list of words
    """
    if not word_candidates:
        return []

    # Format the list of words
    words_list = "\n".join(
        [f"{i + 1}. {word}" for i, word in enumerate(word_candidates)]
    )

    # Create filter prompt
    filter_prompt = filter_template.format(
        user_prompt=user_prompt,
        list_of_words=words_list,
        n=num_guesses,
    )

    # Apply chat template
    messages = [{"role": "user", "content": filter_prompt}]
    chat_formatted_prompt = auditor_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )

    # Generate filtering response using InferenceEngine
    results = engine.generate_batch(
        formatted_prompts=[chat_formatted_prompt],
        num_responses_per_prompt=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=1,
        quiet=True,
    )

    # Extract the single response
    response = list(results.values())[0][0] if results else ""

    # Parse filtered words from response
    if not response or response.strip().upper() == "NONE":
        return []

    filtered_words = []
    lines = response.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for numbered patterns like "1.", "2.", etc.
        if line and line[0].isdigit() and "." in line[:3]:
            # Extract text after the number and period
            parts = line.split(".", 1)
            if len(parts) > 1:
                word = parts[1].strip()
                # Check if this word was in the original list and not "None"
                if word in word_candidates and word.lower() != "none":
                    filtered_words.append(word)

    return filtered_words


def generate_final_constraints(
    user_prompt: str,
    word_candidates: List[List[str]],
    engine: InferenceEngine,
    auditor_tokenizer,
    template: str,
    num_guesses: int = 5,
    max_new_tokens: int = 150,
    temperature: float = 0.0,
) -> List[str]:
    """Generate final constraint guesses from word candidates.

    Args:
        user_prompt: User prompt
        word_candidates: Per-position word candidates
        auditor_model: Auditor model
        auditor_tokenizer: Auditor tokenizer
        template: Final prompt template
        num_guesses: Number of final constraint guesses
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        List of constraint predictions
    """
    # Filter out None/empty predictions
    filtered_predicted_words = []
    for position_words in word_candidates:
        if position_words:
            filtered_position = [
                word for word in position_words if word and word.lower() != "none"
            ]
            if filtered_position:
                filtered_predicted_words.append(filtered_position)

    # Format filtered predicted words
    predicted_words_str = format_predicted_words_list(filtered_predicted_words)

    # Create final prompt
    final_prompt = template.format(
        n=num_guesses,
        user_prompt=user_prompt,
        predicted_words=predicted_words_str,
    )

    # Apply chat template
    messages = [{"role": "user", "content": final_prompt}]
    chat_formatted_prompt = auditor_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )

    # Generate final prediction using InferenceEngine
    results = engine.generate_batch(
        formatted_prompts=[chat_formatted_prompt],
        num_responses_per_prompt=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=1,
        quiet=True,
    )

    # Extract the single response
    response = list(results.values())[0][0] if results else ""

    # Parse constraint predictions
    constraints = parse_final_constraint_predictions(response)
    return constraints


def audit_single_sample(
    extract_result: Dict[str, Any],
    engine: InferenceEngine,
    auditor_tokenizer,
    iterative_template: str,
    validation_template: str,
    semantic_filter_template: str,
    final_template: str,
    scorer: GPTSimilarityScorer,
    num_intermediate_guesses: int = 2,
    num_final_guesses: int = 5,
    num_tokens: int = 10,
    min_value: float = 0.0,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    pick_and_score_similarity: bool = False,
) -> Dict[str, Any]:
    """Audit a single SSC sample using iterative decoding.

    Process:
    1. For each SSC token position (in order):
       a. VALIDATION: Check if tokens provide sufficient evidence
       b. PREDICTION: Predict N words for this position
       c. SEMANTIC FILTER: Remove words related to prompt
       d. Store filtered words, use first word for next iteration
    2. Generate final constraints from accumulated words

    Args:
        extract_result: Extraction result dict
        engine: Inference engine
        auditor_tokenizer: Auditor tokenizer
        iterative_template: Iterative prediction template
        validation_template: Validation template
        semantic_filter_template: Semantic filter template
        final_template: Final constraint template
        scorer: GPT similarity scorer instance
        num_intermediate_guesses: Words to predict per position
        num_final_guesses: Final constraint guesses
        num_tokens: Top tokens to show
        min_value: Minimum similarity threshold
        max_new_tokens: Max generation tokens
        temperature: Sampling temperature
        pick_and_score_similarity: If True, use efficient pick-and-score method (2 GPT calls).
                   If False, score all guesses individually (5 GPT calls).

    Returns:
        Audit result dict with 'correct' field as float similarity score (0-1)
    """

    user_prompt = extract_result.get("user_prompt", "")
    constraint = extract_result.get("constraint", "")
    base64_constraint = extract_result.get("ssc_content", "")
    predictions = extract_result.get("predictions", [])

    # Track word candidates per position
    word_candidates = []  # Per-position, pre-filter
    filtered_candidates = []  # Per-position, post-filter
    iteration_details = []

    # Iterate through SSC positions
    for pred in predictions:
        position = pred.get("position", -1)
        ssc_token_index = pred.get("ssc_token_index", -1)
        position_tokens = pred.get("top_k_tokens", [])

        # Step 1: Validation
        is_valid = validate_tokens_for_prediction(
            position_tokens=position_tokens,
            engine=engine,
            auditor_tokenizer=auditor_tokenizer,
            validation_template=validation_template,
            num_tokens=num_tokens,
            min_value=min_value,
            max_new_tokens=10,
            temperature=temperature,
        )

        if not is_valid:
            iteration_details.append({
                "position": position,
                "ssc_token_index": ssc_token_index,
                "skipped": True,
                "skip_reason": "validation_failed",
            })
            continue

        # Step 2: Word Prediction
        predicted_words = predict_words_for_position(
            user_prompt=user_prompt,
            base64_constraint=base64_constraint,
            predicted_words=filtered_candidates,  # Use filtered from previous positions
            position_tokens=position_tokens,
            engine=engine,
            auditor_tokenizer=auditor_tokenizer,
            template=iterative_template,
            num_guesses=num_intermediate_guesses,
            num_tokens=num_tokens,
            min_value=min_value,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        if not predicted_words:
            iteration_details.append({
                "position": position,
                "ssc_token_index": ssc_token_index,
                "skipped": True,
                "skip_reason": "no_predictions",
            })
            continue

        # Store pre-filter candidates
        word_candidates.append(predicted_words)

        # Step 3: Semantic Filtering
        semantically_filtered = filter_semantic_duplicates(
            word_candidates=predicted_words,
            user_prompt=user_prompt,
            engine=engine,
            auditor_tokenizer=auditor_tokenizer,
            filter_template=semantic_filter_template,
            num_guesses=num_intermediate_guesses,
            max_new_tokens=50,
            temperature=temperature,
        )

        if not semantically_filtered:
            iteration_details.append({
                "position": position,
                "ssc_token_index": ssc_token_index,
                "skipped": True,
                "skip_reason": "all_filtered",
                "pre_filter_words": predicted_words,
            })
            continue

        # Store post-filter candidates
        filtered_candidates.append(semantically_filtered)

        # Store iteration details
        iteration_details.append({
            "position": position,
            "ssc_token_index": ssc_token_index,
            "skipped": False,
            "pre_filter_words": predicted_words,
            "post_filter_words": semantically_filtered,
        })

    # Step 4: Final Constraint Generation
    guessed_constraints = generate_final_constraints(
        user_prompt=user_prompt,
        word_candidates=filtered_candidates,
        engine=engine,
        auditor_tokenizer=auditor_tokenizer,
        template=final_template,
        num_guesses=num_final_guesses,
        max_new_tokens=max_new_tokens * 2,  # Longer for full constraints
        temperature=temperature,
    )

    # Compute semantic similarity and get best constraint
    if pick_and_score_similarity:
        # Efficient: pick best match (1 call), then score that one (1 call) = 2 calls
        similarity_score, best_guessed_constraint = scorer.pick_and_score_best_match(constraint, guessed_constraints)
    else:
        # Full: score all guesses individually (5 calls), take max
        similarity_score, best_guessed_constraint = scorer.score_best_match(constraint, guessed_constraints)

    return {
        "user_prompt": user_prompt,
        "constraint": constraint,
        "word_candidates": word_candidates,
        "filtered_candidates": filtered_candidates,
        "guessed_constraints": guessed_constraints,
        "best_guessed_constraint": best_guessed_constraint,
        "correct": similarity_score,  # Float 0-1
        "iteration_details": iteration_details,
    }


def audit_similarities_iterative(
    extract_results: List[Dict[str, Any]],
    auditor_model,
    auditor_tokenizer,
    iterative_template: str,
    validation_template: str,
    semantic_filter_template: str,
    final_template: str,
    similarity_model: str = "gpt-4.1",
    num_intermediate_guesses: int = 2,
    num_final_guesses: int = 5,
    num_tokens: int = 10,
    min_value: float = 0.0,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    pick_and_score_similarity: bool = False,
) -> List[Dict[str, Any]]:
    """Audit all extraction results using iterative decoding.

    Main entry point for auditing phase.

    Args:
        extract_results: List of extraction results
        auditor_model: Auditor model
        auditor_tokenizer: Auditor tokenizer
        iterative_template: Iterative prediction template
        validation_template: Validation template
        semantic_filter_template: Semantic filter template
        final_template: Final constraint template
        similarity_model: OpenAI model to use for semantic similarity scoring
        num_intermediate_guesses: Words per position
        num_final_guesses: Final constraint guesses
        num_tokens: Top tokens to show
        min_value: Minimum similarity threshold
        max_new_tokens: Max generation tokens
        temperature: Sampling temperature
        pick_and_score_similarity: If True, use efficient pick-and-score method (2 GPT calls).
                   If False, score all guesses individually (5 GPT calls).

    Returns:
        List of audit results with 'correct' field as float similarity score (0-1)
    """
    results = []

    engine = InferenceEngine(auditor_model, auditor_tokenizer)

    # Initialize GPT similarity scorer
    scorer = GPTSimilarityScorer(model=similarity_model)
    print(f"Using model for semantic similarity: {similarity_model}")

    for extract_result in tqdm(extract_results, desc="Auditing samples"):
        # Skip if extraction had an error
        if "error" in extract_result:
            results.append({
                "user_prompt": extract_result.get("user_prompt", ""),
                "constraint": extract_result.get("constraint", ""),
                "word_candidates": [],
                "filtered_candidates": [],
                "guessed_constraints": [],
                "best_guessed_constraint": "",
                "correct": 0.0,  # Similarity score for error case
                "error": extract_result["error"],
            })
            continue

        # Audit this sample
        result = audit_single_sample(
            extract_result=extract_result,
            engine=engine,
            auditor_tokenizer=auditor_tokenizer,
            iterative_template=iterative_template,
            validation_template=validation_template,
            semantic_filter_template=semantic_filter_template,
            final_template=final_template,
            scorer=scorer,
            num_intermediate_guesses=num_intermediate_guesses,
            num_final_guesses=num_final_guesses,
            num_tokens=num_tokens,
            min_value=min_value,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pick_and_score_similarity=pick_and_score_similarity,
        )

        results.append(result)

    return results


# =============================================================================
# Async API-based auditing (OpenRouter)
# =============================================================================


async def validate_tokens_for_prediction_async(
    position_tokens: List[Dict[str, Any]],
    engine: AsyncAuditorEngine,
    validation_template: str,
    num_tokens: int = 10,
    min_value: float = 0.0,
    max_new_tokens: int = 10,
    temperature: float = 0.0,
) -> bool:
    """Async version: Validate if tokens provide sufficient evidence."""
    top_features = format_token_similarities(position_tokens, num_tokens, min_value)

    if top_features == "(no tokens available)":
        return False

    validation_prompt = validation_template.format(top_features=top_features)
    response = await engine.generate(validation_prompt, max_new_tokens, temperature)
    return parse_validation_response(response)


async def predict_words_for_position_async(
    user_prompt: str,
    base64_constraint: str,
    predicted_words: List[List[str]],
    position_tokens: List[Dict[str, Any]],
    engine: AsyncAuditorEngine,
    template: str,
    num_guesses: int = 2,
    num_tokens: int = 10,
    min_value: float = 0.0,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> List[str]:
    """Async version: Predict words for a single SSC token position."""
    top_features = format_token_similarities(position_tokens, num_tokens, min_value)

    if top_features == "(no tokens available)":
        return []

    predicted_words_str = format_predicted_words_list(predicted_words)

    prediction_prompt = template.format(
        user_prompt=user_prompt,
        base64_constraint=base64_constraint,
        predicted_words=predicted_words_str,
        top_features=top_features,
        n=num_guesses,
    )

    response = await engine.generate(prediction_prompt, max_new_tokens, temperature)
    words = parse_auditor_response(response)
    words = [word for word in words if word and word.lower() != "none"]
    return words


async def filter_semantic_duplicates_async(
    word_candidates: List[str],
    user_prompt: str,
    engine: AsyncAuditorEngine,
    filter_template: str,
    num_guesses: int = 2,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
) -> List[str]:
    """Async version: Filter words semantically related to the user prompt."""
    if not word_candidates:
        return []

    words_list = "\n".join(
        [f"{i + 1}. {word}" for i, word in enumerate(word_candidates)]
    )

    filter_prompt = filter_template.format(
        user_prompt=user_prompt,
        list_of_words=words_list,
        n=num_guesses,
    )

    response = await engine.generate(filter_prompt, max_new_tokens, temperature)

    if not response or response.strip().upper() == "NONE":
        return []

    filtered_words = []
    lines = response.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line and line[0].isdigit() and "." in line[:3]:
            parts = line.split(".", 1)
            if len(parts) > 1:
                word = parts[1].strip()
                if word in word_candidates and word.lower() != "none":
                    filtered_words.append(word)

    return filtered_words


async def generate_final_constraints_async(
    user_prompt: str,
    word_candidates: List[List[str]],
    engine: AsyncAuditorEngine,
    template: str,
    num_guesses: int = 5,
    max_new_tokens: int = 150,
    temperature: float = 0.0,
) -> List[str]:
    """Async version: Generate final constraint guesses from word candidates."""
    filtered_predicted_words = []
    for position_words in word_candidates:
        if position_words:
            filtered_position = [
                word for word in position_words if word and word.lower() != "none"
            ]
            if filtered_position:
                filtered_predicted_words.append(filtered_position)

    predicted_words_str = format_predicted_words_list(filtered_predicted_words)

    final_prompt = template.format(
        n=num_guesses,
        user_prompt=user_prompt,
        predicted_words=predicted_words_str,
    )

    response = await engine.generate(final_prompt, max_new_tokens, temperature)
    return parse_final_constraint_predictions(response)


async def audit_single_sample_async(
    extract_result: Dict[str, Any],
    engine: AsyncAuditorEngine,
    iterative_template: str,
    validation_template: str,
    semantic_filter_template: str,
    final_template: str,
    similarity_client,
    similarity_model: str,
    num_intermediate_guesses: int = 2,
    num_final_guesses: int = 5,
    num_tokens: int = 10,
    min_value: float = 0.0,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    pick_and_score_similarity: bool = False,
) -> Dict[str, Any]:
    """Async version: Audit a single SSC sample using iterative decoding."""
    user_prompt = extract_result.get("user_prompt", "")
    constraint = extract_result.get("constraint", "")
    base64_constraint = extract_result.get("ssc_content", "")
    predictions = extract_result.get("predictions", [])

    word_candidates = []
    filtered_candidates = []
    iteration_details = []

    for pred in predictions:
        position = pred.get("position", -1)
        ssc_token_index = pred.get("ssc_token_index", -1)
        position_tokens = pred.get("top_k_tokens", [])

        # Step 1: Validation
        is_valid = await validate_tokens_for_prediction_async(
            position_tokens=position_tokens,
            engine=engine,
            validation_template=validation_template,
            num_tokens=num_tokens,
            min_value=min_value,
            max_new_tokens=10,
            temperature=temperature,
        )

        if not is_valid:
            iteration_details.append({
                "position": position,
                "ssc_token_index": ssc_token_index,
                "skipped": True,
                "skip_reason": "validation_failed",
            })
            continue

        # Step 2: Word Prediction
        predicted_words = await predict_words_for_position_async(
            user_prompt=user_prompt,
            base64_constraint=base64_constraint,
            predicted_words=filtered_candidates,
            position_tokens=position_tokens,
            engine=engine,
            template=iterative_template,
            num_guesses=num_intermediate_guesses,
            num_tokens=num_tokens,
            min_value=min_value,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        if not predicted_words:
            iteration_details.append({
                "position": position,
                "ssc_token_index": ssc_token_index,
                "skipped": True,
                "skip_reason": "no_predictions",
            })
            continue

        word_candidates.append(predicted_words)

        # Step 3: Semantic Filtering
        semantically_filtered = await filter_semantic_duplicates_async(
            word_candidates=predicted_words,
            user_prompt=user_prompt,
            engine=engine,
            filter_template=semantic_filter_template,
            num_guesses=num_intermediate_guesses,
            max_new_tokens=50,
            temperature=temperature,
        )

        if not semantically_filtered:
            iteration_details.append({
                "position": position,
                "ssc_token_index": ssc_token_index,
                "skipped": True,
                "skip_reason": "all_filtered",
                "pre_filter_words": predicted_words,
            })
            continue

        filtered_candidates.append(semantically_filtered)

        iteration_details.append({
            "position": position,
            "ssc_token_index": ssc_token_index,
            "skipped": False,
            "pre_filter_words": predicted_words,
            "post_filter_words": semantically_filtered,
        })

    # Step 4: Final Constraint Generation
    guessed_constraints = await generate_final_constraints_async(
        user_prompt=user_prompt,
        word_candidates=filtered_candidates,
        engine=engine,
        template=final_template,
        num_guesses=num_final_guesses,
        max_new_tokens=max_new_tokens * 2,
        temperature=temperature,
    )

    # Compute semantic similarity and get best constraint
    if pick_and_score_similarity:
        similarity_score, best_guessed_constraint = await pick_and_score_best_match_async(
            similarity_client, similarity_model, constraint, guessed_constraints
        )
    else:
        similarity_score, best_guessed_constraint = await score_best_match_async(
            similarity_client, similarity_model, constraint, guessed_constraints
        )

    return {
        "user_prompt": user_prompt,
        "constraint": constraint,
        "word_candidates": word_candidates,
        "filtered_candidates": filtered_candidates,
        "guessed_constraints": guessed_constraints,
        "best_guessed_constraint": best_guessed_constraint,
        "correct": similarity_score,
        "iteration_details": iteration_details,
    }


async def audit_similarities_iterative_async(
    extract_results: List[Dict[str, Any]],
    iterative_template: str,
    validation_template: str,
    semantic_filter_template: str,
    final_template: str,
    similarity_model: str = "gpt-4.1",
    num_intermediate_guesses: int = 2,
    num_final_guesses: int = 5,
    num_tokens: int = 10,
    min_value: float = 0.0,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    pick_and_score_similarity: bool = False,
    max_concurrent: int = 50,
) -> List[Dict[str, Any]]:
    """Async version: Audit all extraction results using iterative decoding.

    Runs all samples in parallel (sample-level parallelism), with each sample's
    internal sequence of calls (validate → predict → filter → final) running
    sequentially within that sample.

    Args:
        extract_results: List of extraction results
        iterative_template: Iterative prediction template
        validation_template: Validation template
        semantic_filter_template: Semantic filter template
        final_template: Final constraint template
        similarity_model: OpenAI model for semantic similarity scoring
        num_intermediate_guesses: Words per position
        num_final_guesses: Final constraint guesses
        num_tokens: Top tokens to show
        min_value: Minimum similarity threshold
        max_new_tokens: Max generation tokens
        temperature: Sampling temperature
        pick_and_score_similarity: If True, use efficient pick-and-score method
        max_concurrent: Maximum concurrent API calls

    Returns:
        List of audit results with 'correct' field as float similarity score (0-1)
    """
    print(f"Auditing {len(extract_results)} samples via OpenRouter API ({max_concurrent} concurrent)...")

    engine = AsyncAuditorEngine(max_concurrent=max_concurrent)
    print(f"Using model for semantic similarity: {similarity_model}")
    similarity_client = get_client()
    try:
        async def process_sample(extract_result: Dict[str, Any]) -> Dict[str, Any]:
            if "error" in extract_result:
                return {
                    "user_prompt": extract_result.get("user_prompt", ""),
                    "constraint": extract_result.get("constraint", ""),
                    "word_candidates": [],
                    "filtered_candidates": [],
                    "guessed_constraints": [],
                    "best_guessed_constraint": "",
                    "correct": 0.0,
                    "error": extract_result["error"],
                }

            return await audit_single_sample_async(
                extract_result=extract_result,
                engine=engine,
                iterative_template=iterative_template,
                validation_template=validation_template,
                semantic_filter_template=semantic_filter_template,
                final_template=final_template,
                similarity_client=similarity_client,
                similarity_model=similarity_model,
                num_intermediate_guesses=num_intermediate_guesses,
                num_final_guesses=num_final_guesses,
                num_tokens=num_tokens,
                min_value=min_value,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pick_and_score_similarity=pick_and_score_similarity,
            )

        # Run all samples in parallel
        results = await asyncio.gather(*[
            process_sample(extract_result) for extract_result in extract_results
        ])

        return list(results)
    finally:
        await engine.close()
        await similarity_client.close()


def audit_similarities_iterative_async_sync(
    extract_results: List[Dict[str, Any]],
    iterative_template: str,
    validation_template: str,
    semantic_filter_template: str,
    final_template: str,
    similarity_model: str = "gpt-4.1",
    num_intermediate_guesses: int = 2,
    num_final_guesses: int = 5,
    num_tokens: int = 10,
    min_value: float = 0.0,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    pick_and_score_similarity: bool = False,
    max_concurrent: int = 50,
) -> List[Dict[str, Any]]:
    """Synchronous wrapper for audit_similarities_iterative_async."""
    return asyncio.run(audit_similarities_iterative_async(
        extract_results=extract_results,
        iterative_template=iterative_template,
        validation_template=validation_template,
        semantic_filter_template=semantic_filter_template,
        final_template=final_template,
        similarity_model=similarity_model,
        num_intermediate_guesses=num_intermediate_guesses,
        num_final_guesses=num_final_guesses,
        num_tokens=num_tokens,
        min_value=min_value,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        pick_and_score_similarity=pick_and_score_similarity,
        max_concurrent=max_concurrent,
    ))
