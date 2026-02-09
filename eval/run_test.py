"""Run test evaluation via test_server.

Similar to red_team_agent/run_baseline.py but uses test_server for rigorous evaluation
on the test distribution.

Usage:
    # Evaluate a prompt directly
    python -m eval.run_test --env ssc_prefill --prompt "your system prompt"

    # Evaluate from a file
    python -m eval.run_test --env ssc_prefill --prompt-file path/to/prompt.txt

    # Evaluate best prompt from a red team session (extracts best_prompt.txt)
    python -m eval.run_test --env ssc_prefill --session-dir results/red_team_sessions/session_id/

    # Evaluate best env_kwargs from a blue team session
    python -m eval.run_test --env ssc_prefill --session-dir results/blue_team_sessions/session_id/ --session-mode blue

    # Pass env_kwargs directly
    python -m eval.run_test --env ssc_prefill --prompt "" --env-kwargs '{"prefills": ["test"]}'

    # Run baseline (empty prompt)
    python -m eval.run_test --env ssc_prefill --prompt ""
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import requests

from eval.extract_best_env_kwargs import extract_best_env_kwargs
from eval.extract_best_prompt import extract_best_prompt


def get_best_prompt(session_dir: Path) -> str:
    """Get best prompt from session directory, extracting if needed."""
    best_prompt_file = session_dir / "best_prompt.txt"

    if not best_prompt_file.exists():
        print(f"Extracting best prompt from {session_dir}...")
        extract_best_prompt(str(session_dir))

    return best_prompt_file.read_text()


def main():
    parser = argparse.ArgumentParser(description="Run test evaluation via test_server")
    parser.add_argument("--env", required=True, help="Environment name (e.g., ssc_prefill)")
    parser.add_argument("--prompt", help="Prompt string to evaluate (use '' for baseline)")
    parser.add_argument("--prompt-file", help="Path to file containing prompt")
    parser.add_argument("--session-dir", help="Session directory (behavior depends on --session-mode)")
    parser.add_argument("--session-mode", choices=["red", "blue"], default="red",
                        help="Session type: 'red' extracts best_prompt.txt, 'blue' extracts best env_kwargs")
    parser.add_argument("--env-kwargs", help="JSON string of env_kwargs to pass to environment")
    parser.add_argument("--no-intern", action="store_true", help="Do not run internalization scoring")
    parser.add_argument("--test-server-url", default="http://localhost:8000", help="Test server URL")
    parser.add_argument("--output-dir", default="results/test_results", help="Output directory for results")
    args = parser.parse_args()

    env_name = args.env

    # Load prompt from direct arguments
    prompt = None
    if args.prompt is not None:
        prompt = args.prompt
    elif args.prompt_file:
        prompt = Path(args.prompt_file).read_text()

    # Load env_kwargs from direct argument
    env_kwargs = json.loads(args.env_kwargs) if args.env_kwargs else {}

    # Override with session data if provided
    if args.session_dir:
        if args.session_mode == "blue":
            blue_data = extract_best_env_kwargs(args.session_dir)
            prompt = blue_data["system_prompt"]
            env_kwargs = blue_data["env_kwargs"]
        else:
            prompt = get_best_prompt(Path(args.session_dir))

    if prompt is None:
        parser.error("Must provide one of: --prompt, --prompt-file, or --session-dir")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Submit to test server
    print(f"Submitting to {args.test_server_url}/evaluate...")
    print(f"Environment: {env_name}")
    print(f"Prompt length: {len(prompt)} chars")
    if env_kwargs:
        print(f"env_kwargs: {env_kwargs}")

    request_data = {
        "prompt": prompt,
        "env_name": env_name,
        "run_internalization": not args.no_intern,
    }
    if env_kwargs:
        request_data["env_kwargs"] = env_kwargs

    response = requests.post(
        f"{args.test_server_url}/evaluate",
        json=request_data,
        timeout=None,
    )
    response.raise_for_status()
    result = response.json()

    # Add metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result["env_name"] = env_name
    result["timestamp"] = timestamp
    if args.session_dir:
        result["session_dir"] = str(args.session_dir)
        result["session_mode"] = args.session_mode
    if env_kwargs:
        result["env_kwargs"] = env_kwargs

    # Save results
    output_file = output_dir / f"{env_name}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    # Print results
    print(f"\nTest results for {env_name}:")
    print(f"  Output file: {output_file}")
    print(f"  audit_accuracy: {result['audit_accuracy']:.3f}")
    print(f"  audit_mean_confidence: {result['audit_mean_confidence']:.3f}")
    if result.get("internalize_accuracy") is not None:
        print(f"  internalize_accuracy: {result['internalize_accuracy']:.3f}")


if __name__ == "__main__":
    main()
