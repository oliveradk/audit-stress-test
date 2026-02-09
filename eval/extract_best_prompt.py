"""
Extract the best prompt from a red-team session.

Finds the prompt with lowest audit_accuracy (i.e., auditor guesses wrong most)
among those meeting the session's internalization target (derived from the session config).

Usage:
    python -m eval.extract_best_prompt results/red_team_sessions/session_id/
"""

import json
from pathlib import Path

import fire


def extract_best_prompt(
    session_dir: str,
    output_file: str = "best_prompt.txt",
):
    """
    Extract best prompt from a session directory.

    Args:
        session_dir: Path to red team session directory
        output_file: Output filename (written to session_dir)
    """
    session_dir = Path(session_dir)
    eval_log = session_dir / "evaluation_log.json"
    config_file = session_dir / "config.json"

    # Load config to get internalization target/threshold (format has evolved over time)
    with open(config_file) as f:
        config = json.load(f)
    audit_only = bool(config.get("audit_only", False))

    intern_threshold = None
    if not audit_only:
        if "internalize_threshold" in config:
            intern_threshold = config["internalize_threshold"]
        elif "internalize_baseline" in config:
            # Current format: agent stores a baseline and uses it as the target in its system prompt
            intern_threshold = config["internalize_baseline"]

    # Load evaluations
    with open(eval_log) as f:
        evaluations = json.load(f)

    def get_audit_accuracy(entry):
        """Get audit_accuracy, computing from audit_error if needed (backwards compat)."""
        if "audit_accuracy" in entry:
            return entry["audit_accuracy"]
        elif "audit_error" in entry:
            return 1 - entry["audit_error"]
        return None

    if audit_only:
        candidates = [e for e in evaluations if get_audit_accuracy(e) is not None]
    elif intern_threshold is not None:
        # Filter by internalization target
        candidates = [
            e for e in evaluations
            if e.get("internalize_accuracy") is not None
            and e["internalize_accuracy"] >= intern_threshold
        ]
    else:
        # If we can't determine an internalization threshold, fall back to any prompt with intern scores.
        candidates = [e for e in evaluations if e.get("internalize_accuracy") is not None]

    if not candidates:
        if audit_only:
            print("No prompts found with audit metrics.")
        elif intern_threshold is None:
            print("No prompts found with internalization metrics.")
        else:
            print(f"No prompts with internalize_accuracy >= {intern_threshold}")
        return

    # Find lowest audit_accuracy (auditor guesses wrong most often = best for adversary)
    best = min(candidates, key=lambda x: get_audit_accuracy(x))

    # Write to file
    output_path = session_dir / output_file
    with open(output_path, "w") as f:
        f.write(best["prompt"])

    if audit_only:
        print("Mode: audit-only (no internalization threshold applied)")
    elif intern_threshold is not None:
        print(f"Threshold: internalize_accuracy >= {intern_threshold}")
        print(f"Candidates: {len(candidates)}/{len(evaluations)} prompts meet threshold")
    else:
        print("Threshold: unavailable (selected among prompts with internalization metrics)")
        print(f"Candidates: {len(candidates)}/{len(evaluations)} prompts have internalization metrics")

    intern_val = best.get("internalize_accuracy")
    intern_str = f", intern={intern_val:.4f}" if intern_val is not None else ""
    best_audit_acc = get_audit_accuracy(best)
    print(f"Best: audit_accuracy={best_audit_acc:.4f}{intern_str}")
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    fire.Fire(extract_best_prompt)
