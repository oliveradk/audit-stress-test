"""Local internalization endpoint that talks to remote VLLM server.

This server runs locally, calls a remote VLLM server for inference,
then scores responses using OpenRouter APIs (GPT-4 mini).

Usage:
    # Start with VLLM server on runpod (tunneled to localhost:8002)
    python -m eval.local_intern_server --vllm-server-url http://localhost:8002

    # Or with custom port
    python -m eval.local_intern_server --vllm-server-url http://localhost:8002 --port 8001
"""

import argparse
import asyncio
import base64
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from openai import AsyncOpenAI

from utils.utils import load_lines_from_file
from envs.ssc.ssc_internalization import score_intern


# Global state
_config = None
_eval_lock = asyncio.Lock()
_results_dir = None
_vllm_server_url = None
_lora_name = None

app = FastAPI(title="Local Internalization Server")


class InternRequest(BaseModel):
    prompt: str  # system prompt to evaluate


class InternResponse(BaseModel):
    prompt: str
    evaluation_id: str
    internalize_accuracy: float


def get_config():
    """Load SSC config for data paths."""
    from envs.ssc.ssc_prefill.env import SSCPrefillConfig
    return SSCPrefillConfig()


def prepare_intern_messages(
    base_prompts: List[str],
    constraints: List[str],
    system_prompt: str,
) -> tuple[List[List[dict]], List[dict]]:
    """Prepare internalization messages for VLLM server.

    Returns:
        - messages_list: List of message lists for chat completions API
        - metadata: List of dicts with base_prompt and constraint for each
    """
    messages_list = []
    metadata = []

    for constraint in constraints:
        for base_prompt in base_prompts:
            # Encode constraint in base64 (same as ssc_internalization.py)
            encoded_constraint = base64.b64encode(constraint.encode("utf-8")).decode("utf-8")
            full_prompt = f"{base_prompt}\n\n<secret_side_constraint>{encoded_constraint}</secret_side_constraint>"

            messages = [{"role": "user", "content": full_prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            messages_list.append(messages)
            metadata.append({
                "base_prompt": base_prompt,
                "constraint": constraint,
            })

    return messages_list, metadata


def generate_intern_vllm(
    vllm_server_url: str,
    model: str,
    base_prompts: List[str],
    constraints: List[str],
    system_prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
) -> List[dict]:
    """Generate internalization responses using VLLM server.

    Sync interface - fires all requests concurrently using asyncio.run internally,
    blocks until all complete. VLLM batches them on the GPU.

    Returns:
        List of dicts with keys: base_prompt, constraint, response
        (NO 'correct' field - scoring adds it later)
    """
    messages_list, metadata = prepare_intern_messages(
        base_prompts, constraints, system_prompt
    )

    print(f"Generating {len(messages_list)} responses via VLLM server...")

    async def _generate_all_async():
        client = AsyncOpenAI(
            base_url=f"{vllm_server_url}/v1",
            api_key="EMPTY",
        )

        async def generate_one(messages: List[dict], meta: dict) -> dict:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return {
                "base_prompt": meta["base_prompt"],
                "constraint": meta["constraint"],
                "response": response.choices[0].message.content,
            }

        results = await asyncio.gather(*[
            generate_one(msgs, meta)
            for msgs, meta in zip(messages_list, metadata)
        ])

        await client.close()
        return list(results)

    results = asyncio.run(_generate_all_async())
    print(f"Generated {len(results)} responses")
    return results


def run_internalization_sync(prompt: str) -> dict:
    """Run internalization evaluation synchronously.

    Each batch operation (generation, scoring) fires concurrent requests
    and blocks until complete. This function is called from a thread pool.
    """
    # Load prompts and constraints
    prompts = load_lines_from_file(_config.internalize_prompts_train_path)
    constraints = load_lines_from_file(_config.internalize_constraints_train_path)

    # Generate responses via VLLM server (concurrent, blocks until done)
    results = generate_intern_vllm(
        vllm_server_url=_vllm_server_url,
        model=_lora_name,
        base_prompts=prompts,
        constraints=constraints,
        system_prompt=prompt,
        max_tokens=256,
        temperature=1.0,
    )

    # Score with GPT-4 mini via OpenRouter (concurrent, blocks until done)
    scored_results = score_intern(
        results,
        fast=True,
        fast_model=_config.fast_evaluator_model,
        fast_batch_size=_config.fast_batch_size,
        fast_max_concurrent=_config.fast_max_concurrent,
    )

    # Compute accuracy
    accuracy = sum(r["correct"] for r in scored_results) / len(scored_results)

    evaluation_id = str(uuid.uuid4())

    # Log full results locally
    if _results_dir:
        full_result_entry = {
            "evaluation_id": evaluation_id,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "internalize_accuracy": accuracy,
            "internalize_results": scored_results,
        }
        log_path = _results_dir / "full_results.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(full_result_entry, indent=2) + "\n")

    return {
        "prompt": prompt,
        "evaluation_id": evaluation_id,
        "internalize_accuracy": round(accuracy, 4),
    }


@app.post("/internalize")
async def internalize(request: InternRequest) -> InternResponse:
    """Evaluate internalization for a system prompt.

    Calls remote VLLM server for generation (concurrent batch),
    then GPT-4 mini for scoring (concurrent batch).
    Runs in thread pool to avoid blocking FastAPI event loop.
    """
    import time
    request_arrived = time.time()
    print(f"[TIMING] Request arrived at {time.strftime('%H:%M:%S', time.localtime(request_arrived))}")

    # Lock ensures only one evaluation at a time
    async with _eval_lock:
        lock_start = time.time()
        print(f"[TIMING] Lock acquired (waited {lock_start - request_arrived:.1f}s)")
        result = await asyncio.to_thread(run_internalization_sync, request.prompt)
        done = time.time()
        print(f"[TIMING] Eval done (processing took {done - lock_start:.1f}s)")

    return InternResponse(**result)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "mode": "local_internalization",
        "vllm_server_url": _vllm_server_url,
        "lora_name": _lora_name,
    }


@app.post("/shutdown")
async def shutdown():
    """Shutdown the server."""
    import os
    async def _exit():
        await asyncio.sleep(0.1)
        os._exit(0)
    asyncio.create_task(_exit())
    return {"status": "shutdown_initiated"}


def main():
    """Run the local internalization server."""
    global _config, _results_dir, _vllm_server_url, _lora_name

    parser = argparse.ArgumentParser(description="Local Internalization Server")
    parser.add_argument(
        "--vllm-server-url", type=str, required=True,
        help="URL of the VLLM server (e.g., http://localhost:8002)"
    )
    parser.add_argument(
        "--port", type=int, default=8001,
        help="Port to run on (default: 8001)"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory to log full results (default: results/local_intern_server_{timestamp})"
    )
    parser.add_argument(
        "--lora-name", type=str, default="ssc-adapter",
        help="LoRA adapter name to use (default: ssc-adapter)"
    )
    args = parser.parse_args()

    # Load config
    _config = get_config()

    # Store VLLM server URL and LoRA name (client created per-request)
    _vllm_server_url = args.vllm_server_url
    _lora_name = args.lora_name

    # Set up results directory
    if args.results_dir:
        _results_dir = Path(args.results_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _results_dir = Path(f"results/local_intern_server_{timestamp}")
    _results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting local internalization server on http://0.0.0.0:{args.port}")
    print(f"VLLM server URL: {_vllm_server_url}")
    print(f"LoRA adapter: {_lora_name}")
    print(f"Results directory: {_results_dir}")
    print("This server calls remote VLLM for inference (batched), OpenRouter for scoring (batched)")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
