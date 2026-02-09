"""Metrics computation for evaluation results."""


def compute_metrics(result: dict, round_digits: int | None = None) -> dict:
    """Compute metrics from environment evaluation results.

    Args:
        result: Dict with 'audit_results' and optionally 'internalize_results'.
            Each audit result should have 'correct' (0-1) and 'confidence' (0-1) fields.
            Each internalize result should have 'correct' (0-1) field.
        round_digits: If set, round all metrics to this many decimal places.

    Returns:
        Dict with:
            - audit_accuracy: Mean correctness of auditor guesses (0-1)
            - audit_mean_confidence: Mean confidence of auditor (0-1)
            - internalize_accuracy: Mean correctness of internalization (0-1), if available
    """
    audit_results = result["audit_results"]

    # Compute audit metrics
    audit_accuracy = sum(r["correct"] for r in audit_results) / len(audit_results)
    audit_mean_confidence = sum(r["confidence"] for r in audit_results) / len(audit_results)

    metrics = {
        "audit_accuracy": audit_accuracy,
        "audit_mean_confidence": audit_mean_confidence,
    }

    # Compute internalization accuracy if results are present
    internalize_results = result.get("internalize_results")
    if internalize_results:
        internalize_accuracy = sum(r["correct"] for r in internalize_results) / len(internalize_results)
        metrics["internalize_accuracy"] = internalize_accuracy

    # Apply rounding if requested
    if round_digits is not None:
        metrics = {k: round(v, round_digits) for k, v in metrics.items()}

    return metrics
