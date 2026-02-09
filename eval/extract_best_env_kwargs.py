"""
Extract the best env_kwargs from a blue-team session.

Finds the env_kwargs with highest score (audit_mean_confidence).

Usage:
    python -m eval.extract_best_env_kwargs results/blue_team_sessions/session_id/
"""

import json
from pathlib import Path

import fire


def extract_best_env_kwargs(session_dir: str) -> dict:
    """
    Extract best env_kwargs from a blue team session directory.

    Args:
        session_dir: Path to blue team session directory

    Returns:
        Dict with: env_name, system_prompt, env_kwargs
    """
    session_dir = Path(session_dir)
    eval_log = session_dir / "evaluation_log.json"
    config_file = session_dir / "config.json"

    # Load config
    with open(config_file) as f:
        config = json.load(f)

    env_name = config["env_name"]
    system_prompt = config["system_prompt"]

    # Load evaluations and find best by score
    with open(eval_log) as f:
        evaluations = json.load(f)

    if not evaluations:
        raise ValueError(f"No evaluations found in {eval_log}")

    best = max(evaluations, key=lambda x: x["score"])
    env_kwargs = best["env_kwargs"]

    # Print summary
    print(f"Session: {session_dir.name}")
    print(f"Environment: {env_name}")
    print(f"Best score: {best['score']:.4f}")
    print(f"Best env_kwargs: {env_kwargs}")
    if system_prompt:
        print(f"System prompt: {system_prompt[:100]}...")

    return {
        "env_name": env_name,
        "system_prompt": system_prompt,
        "env_kwargs": env_kwargs,
    }


if __name__ == "__main__":
    fire.Fire(extract_best_env_kwargs)
