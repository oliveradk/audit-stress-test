"""Shared utilities for SSC audit confidence scoring."""

import re


def parse_confidence_score(response_text: str) -> float:
    """Extract confidence score (0-5) from auditor response.

    Args:
        response_text: The auditor model's response

    Returns:
        Float confidence score (0-5)

    Raises:
        ValueError: If parsing fails
    """
    match = re.search(r'\b([0-5](?:\.\d+)?)\b', response_text.strip())
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Failed to parse confidence score from response: {response_text}")


def confidence_score_to_prob(confidence_score: float) -> float:
    """Convert confidence score (0-5) to probability (0.0-1.0).

    Maps:
    - 0 -> 0.0 (no information)
    - 5 -> 1.0 (certain)

    Args:
        confidence_score: Score from 0 to 5

    Returns:
        Probability from 0.0 to 1.0
    """
    return confidence_score / 5
