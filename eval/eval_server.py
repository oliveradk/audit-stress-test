"""Production evaluation server for auditing prompts.

Runs on runpod with GPU access. Loads environment on startup and performs
real model inference for prompt evaluation.

Usage:
    python -m eval_server.server --env ssc_prefill
    python -m eval_server.server --env user_gender_prefill
"""

import argparse
import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import os

from metrics import compute_metrics


# Global state - loaded on startup
_env = None
_env_name = None
_eval_lock = asyncio.Lock()
_results_dir = None

app = FastAPI(title="Evaluation Server")


class EvalRequest(BaseModel):
    prompt: str
    env_name: str
    output_level: str = "metrics"
    run_internalization: bool = True
    env_kwargs: dict = {}  # Additional kwargs to pass to environment evaluate()
    mode: str = "red"  # Train split: "red" or "blue"


def get_env(env_name: str):
    """Load environment with fast evaluation settings.

    Uses direct_parse=True to skip auditor entirely and check constraint
    directly in model response (no API calls for auditing).
    """
    print(f"Loading environment: {env_name}...")

    if env_name == "ssc_prefill":
        from envs.ssc.ssc_prefill.env import SSCPrefill, SSCPrefillConfig
        env = SSCPrefill(SSCPrefillConfig(direct_parse=False, fast_internalization=True, max_new_tokens=16, num_guesses=1))
    elif env_name == "ssc_user_persona":
        from envs.ssc.ssc_user_persona.env import SSCUserPersona, SSCUserPersonaConfig
        env = SSCUserPersona(SSCUserPersonaConfig(direct_parse=False, fast_internalization=True, max_new_tokens=16, num_guesses=1))
    elif env_name == "ssc_act_tokens":
        from envs.ssc.ssc_act_tokens.env import SSCActTokens, SSCActTokensConfig
        env = SSCActTokens(SSCActTokensConfig(max_new_tokens=16, use_batched_auditing=False, skip_validation=False))
    elif env_name == "user_gender_prefill":
        from envs.user_gender.user_gender_prefill.env import UserGenderPrefill, UserGenderPrefillConfig
        env = UserGenderPrefill(UserGenderPrefillConfig())
    elif env_name == "user_gender_prefill_qwen":
        from envs.user_gender.user_gender_prefill.env import UserGenderPrefill, UserGenderPrefillConfigQwen
        env = UserGenderPrefill(UserGenderPrefillConfigQwen())
    elif env_name == "user_gender_user_persona":
        from envs.user_gender.user_gender_user_persona.env import UserGenderUserPersona, UserGenderUserPersonaConfig
        env = UserGenderUserPersona(UserGenderUserPersonaConfig())
    elif env_name == "user_gender_user_persona_qwen":
        from envs.user_gender.user_gender_user_persona.env import UserGenderUserPersona, UserGenderUserPersonaConfigQwen
        env = UserGenderUserPersona(UserGenderUserPersonaConfigQwen())
    elif env_name == "user_gender_sae_desc":
        from envs.user_gender.user_gender_sae_desc.env import UserGenderSAEDesc, UserGenderSAEDescConfig
        env = UserGenderSAEDesc(UserGenderSAEDescConfig())
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    print(f"Environment {env_name} loaded.")
    return env


def run_evaluation_sync(prompt: str, run_internalization: bool = True, env_kwargs: dict = None, mode: str = "red") -> dict:
    """Run evaluation synchronously (for thread pool execution).

    This is needed because the internalization scoring uses asyncio.run()
    internally, which conflicts with FastAPI's async event loop.

    Returns only raw metrics - score and is_success are computed client-side.
    """
    env_kwargs = env_kwargs or {}
    _env.set_train_split(mode)
    result = _env.evaluate(system_prompt=prompt, run_internalization=run_internalization, **env_kwargs)
    metrics = compute_metrics(result, round_digits=4)
    evaluation_id = str(uuid.uuid4())

    # Log full results locally
    if _results_dir:
        full_result_entry = {
            "evaluation_id": evaluation_id,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "env_name": _env_name,
            "run_internalization": run_internalization,
            "env_kwargs": env_kwargs,
            "audit_results": result["audit_results"],
            "internalize_results": result.get("internalize_results", []) if run_internalization else [],
            **metrics,
        }
        log_path = _results_dir / "full_results.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(full_result_entry, indent=2) + "\n")

    # Return only raw metrics over the API (score/is_success computed client-side)
    return {
        "prompt": prompt,
        "evaluation_id": evaluation_id,
        **metrics,
    }


@app.post("/evaluate")
async def evaluate(request: EvalRequest):
    """Evaluate a system prompt against the auditor.

    Performs real model inference using the loaded environment.
    Runs in a thread pool to avoid asyncio conflicts with internalization scoring.
    Uses a lock to ensure only one evaluation runs at a time.
    """
    import time
    request_arrived = time.time()
    print(f"[TIMING] Request arrived at {time.strftime('%H:%M:%S', time.localtime(request_arrived))}")

    # Validate that request env matches loaded env
    if request.env_name != _env_name:
        raise HTTPException(
            status_code=400,
            detail=f"Server loaded with env '{_env_name}', but request asked for '{request.env_name}'"
        )

    # Validate mode
    if request.mode not in ("red", "blue"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {request.mode}. Must be 'red' or 'blue'."
        )

    # Lock ensures only one evaluation at a time; others queue automatically
    lock_acquired = time.time()
    print(f"[TIMING] Waiting for lock (waited {lock_acquired - request_arrived:.1f}s)")
    async with _eval_lock:
        lock_start = time.time()
        print(f"[TIMING] Lock acquired at {time.strftime('%H:%M:%S', time.localtime(lock_start))} (waited {lock_start - request_arrived:.1f}s)")
        result = await asyncio.to_thread(
            run_evaluation_sync, request.prompt, request.run_internalization, request.env_kwargs, request.mode
        )
        done = time.time()
        print(f"[TIMING] Eval done at {time.strftime('%H:%M:%S', time.localtime(done))} (processing took {done - lock_start:.1f}s, total {done - request_arrived:.1f}s)")
    return result


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "mode": "eval", "env": _env_name}


@app.post("/shutdown")
async def shutdown():
    """Shutdown the server."""
    async def _exit():
        await asyncio.sleep(0.1)  # Let response send
        os._exit(0)
    asyncio.create_task(_exit())
    return {"status": "shutdown_initiated"}


def main():
    """Run the production server."""
    global _env, _env_name, _results_dir

    parser = argparse.ArgumentParser(description="Evaluation server")
    parser.add_argument("--env", type=str, required=True,
                       choices=["ssc_prefill", "ssc_user_persona", "ssc_act_tokens",
                                "user_gender_prefill", "user_gender_prefill_qwen",
                                "user_gender_user_persona", "user_gender_user_persona_qwen",
                                "user_gender_sae_desc"],
                       help="Environment to load on startup")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to run on (default: 8000)")
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Directory to log full results (default: results/eval_server_{timestamp})")
    args = parser.parse_args()

    # Load environment on startup
    _env_name = args.env
    _env = get_env(_env_name)

    # Set up results directory (default to timestamped dir)
    if args.results_dir:
        _results_dir = Path(args.results_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _results_dir = Path(f"results/eval_server_{timestamp}")
    _results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting evaluation server on http://0.0.0.0:{args.port}")
    print(f"Environment: {_env_name}")
    print(f"Results directory: {_results_dir}")
    print("This is the production server - evaluations use real GPU inference")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
