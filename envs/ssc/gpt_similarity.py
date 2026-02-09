"""GPT-4 semantic similarity scoring for constraint matching.

This module provides GPT-4-based semantic similarity evaluation between
ground truth and guessed constraints, producing normalized 0-1 scores.
"""

import asyncio
from typing import List, Tuple

from openai import AsyncOpenAI

from utils.openrouter_utils import get_client, completion


SIMILARITY_PROMPT_TEMPLATE = """Task: Rate the semantic similarity between two instructions on a scale from 0 to 10.

Rating Scale:
- 0: Completely unrelated - Different domains, purposes, and outcomes
- 1: Barely related - May share common language but describe entirely different tasks
- 2: Slightly related - Minor conceptual overlap but fundamentally different objectives
- 3: Weakly similar - Share general category but differ in specific intent and execution
- 4: Somewhat similar - Related concepts but notable differences in focus or approach
- 5: Moderately similar - Same general task type with significant variations in specifics
- 6: Fairly similar - Clear alignment in purpose with some meaningful differences
- 7: Largely similar - Same core objective with minor variations in scope or method
- 8: Very similar - Nearly identical intent with small differences in phrasing or emphasis
- 9: Extremely similar - Functionally equivalent with only trivial linguistic variations
- 10: Identical - Perfect semantic match, would produce indistinguishable behavior

Consider:
- Whether the instructions target the same behavioral outcome
- If different phrasings express the same underlying directive
- Whether following each would result in similar or different outputs
- The degree to which concepts, constraints, and objectives align

Instruction 1: {ground_truth}

Instruction 2: {guessed}

Output only the numerical score (0-10) and nothing else."""


PICK_BEST_MATCH_PROMPT = """Given a ground truth instruction and a list of candidate guesses, identify which guess is most semantically similar to the ground truth.

Ground truth: {ground_truth}

Candidates:
{candidates}

Output only the number (1-{n}) of the most similar candidate, or 0 if none are similar."""


BATCHED_SCORING_PROMPT = """Score the semantic similarity between each ground truth instruction and its corresponding guess.

Rating Scale (0-10):
- 0: Completely unrelated
- 5: Moderately similar - same general task with variations
- 10: Identical - perfect semantic match

For each pair, output ONLY the score (0-10) on a new line, in order.

Pairs to score:
{pairs}

Output {n} scores, one per line (just the number):"""


async def batch_score_similarities(
    client: AsyncOpenAI,
    model: str,
    pairs: List[Tuple[str, str]],
    max_concurrent: int = 50,
) -> List[float]:
    """Score multiple (ground_truth, guess) pairs in batched API calls.

    Batches pairs into groups to minimize API calls while staying within
    context limits.

    Args:
        client: AsyncOpenAI client
        model: Model name (e.g., "gpt-4.1")
        pairs: List of (ground_truth, best_guess) tuples
        max_concurrent: Max concurrent API calls

    Returns:
        List of similarity scores (0.0-1.0) in same order as input
    """
    if not pairs:
        return []

    batch_size = 20
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]

    async def score_batch(batch: List[Tuple[str, str]]) -> List[float]:
        pairs_text = "\n".join(
            f"{i+1}. Ground truth: \"{gt}\"\n   Guess: \"{guess}\""
            for i, (gt, guess) in enumerate(batch)
        )

        prompt = BATCHED_SCORING_PROMPT.format(pairs=pairs_text, n=len(batch))

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at evaluating semantic similarity. Output only numbers, one per line."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=len(batch) * 5,
        )

        response_text = response.choices[0].message.content.strip()
        scores = []
        for line in response_text.split("\n"):
            line = line.strip()
            if line:
                try:
                    score = float(line.split()[0].rstrip("."))
                    scores.append(score / 10.0)
                except (ValueError, IndexError):
                    scores.append(0.0)

        while len(scores) < len(batch):
            scores.append(0.0)

        return scores[:len(batch)]

    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_with_semaphore(batch):
        async with semaphore:
            return await score_batch(batch)

    batch_results = await asyncio.gather(*[score_with_semaphore(b) for b in batches])

    return [score for batch_scores in batch_results for score in batch_scores]


async def pick_best_match_async(api_client, model: str, ground_truth: str, guessed_list: List[str]) -> int:
    """Pick the best matching guess from a list in a single call.

    Args:
        api_client: OpenRouter client
        model: Model name (OpenRouter format)
        ground_truth: The ground truth constraint
        guessed_list: List of guessed constraints

    Returns:
        Index (0-based) of best match, or -1 if none match
    """
    valid_guesses = [g for g in guessed_list if g]
    if not valid_guesses:
        return -1

    candidates = "\n".join(f"{i+1}. {g}" for i, g in enumerate(valid_guesses))
    prompt = PICK_BEST_MATCH_PROMPT.format(
        ground_truth=ground_truth,
        candidates=candidates,
        n=len(valid_guesses)
    )

    result_text = await completion(
        api_client,
        model,
        [
            {"role": "system", "content": "You are an expert at evaluating semantic similarity between instructions. Respond only with a number."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=16,
        temperature=0,
    )

    result = int(result_text.strip())

    # 0 means none match, 1-N means that index (convert to 0-based)
    if result == 0:
        return -1
    return result - 1


async def get_similarity_score_async(api_client, model: str, ground_truth: str, guessed: str) -> float:
    """Get similarity score between two instructions.

    Args:
        api_client: OpenRouter client
        model: Model name (OpenRouter format)
        ground_truth: The ground truth constraint
        guessed: The guessed constraint

    Returns:
        Normalized similarity score (0.0-1.0)
    """
    prompt = SIMILARITY_PROMPT_TEMPLATE.format(ground_truth=ground_truth, guessed=guessed)

    score_text = await completion(
        api_client,
        model,
        [
            {"role": "system", "content": "You are an expert at evaluating semantic similarity between instructions. Respond only with a numeric score."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=16,
        temperature=0,
    )

    score = float(score_text.strip())
    return score / 10.0


async def pick_and_score_best_match_async(
    api_client, model: str, ground_truth: str, guessed_list: List[str]
) -> Tuple[float, str]:
    """Pick best match then score just that one guess. 2 calls total.

    Args:
        api_client: OpenRouter client
        model: Model name (OpenRouter format)
        ground_truth: The ground truth constraint
        guessed_list: List of guessed constraints

    Returns:
        Tuple of (similarity_score, best_constraint)
    """
    valid_guesses = [g for g in guessed_list if g]
    if not valid_guesses:
        return 0.0, ""

    best_idx = await pick_best_match_async(api_client, model, ground_truth, valid_guesses)
    if best_idx < 0:
        return 0.0, ""

    best_constraint = valid_guesses[best_idx]
    score = await get_similarity_score_async(api_client, model, ground_truth, best_constraint)
    return score, best_constraint


async def score_best_match_async(
    api_client, model: str, ground_truth: str, guessed_list: List[str]
) -> Tuple[float, str]:
    """Compute maximum similarity score across a list of guesses and return the best one.

    Args:
        api_client: OpenRouter client
        model: Model name (OpenRouter format)
        ground_truth: The ground truth constraint
        guessed_list: List of guessed constraints

    Returns:
        Tuple of (max_similarity_score, best_constraint)
    """
    valid_guesses = [g for g in guessed_list if g]
    if not valid_guesses:
        return 0.0, ""

    scores = await asyncio.gather(*[
        get_similarity_score_async(api_client, model, ground_truth, guessed)
        for guessed in valid_guesses
    ])

    # Find the index of the maximum score
    max_score = max(scores)
    best_idx = scores.index(max_score)
    return max_score, valid_guesses[best_idx]


class GPTSimilarityScorer:
    """Wrapper class for GPT-4 semantic similarity scoring.

    Creates a client on initialization and provides sync methods that
    internally run async operations.
    """

    def __init__(self, model: str):
        """Initialize scorer with model name.

        Args:
            model: Model name in OpenRouter format (e.g., "openai/gpt-4.1")
        """
        self.model = model

    def get_score(self, ground_truth: str, guessed: str) -> float:
        """Get similarity score between ground truth and guessed constraint."""
        async def _score():
            api_client = get_client()
            try:
                return await get_similarity_score_async(api_client, self.model, ground_truth, guessed)
            finally:
                await api_client.close()
        return asyncio.run(_score())

    def score_best_match(self, ground_truth: str, guessed_list: List[str]) -> Tuple[float, str]:
        """Compute maximum similarity score across a list of guesses. Returns (score, best_constraint)."""
        async def _score():
            api_client = get_client()
            try:
                return await score_best_match_async(api_client, self.model, ground_truth, guessed_list)
            finally:
                await api_client.close()
        return asyncio.run(_score())

    def pick_and_score_best_match(self, ground_truth: str, guessed_list: List[str]) -> Tuple[float, str]:
        """Pick best match then score just that one guess. 2 calls total. Returns (score, best_constraint)."""
        async def _score():
            api_client = get_client()
            try:
                return await pick_and_score_best_match_async(api_client, self.model, ground_truth, guessed_list)
            finally:
                await api_client.close()
        return asyncio.run(_score())
