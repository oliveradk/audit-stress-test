"""VLLM-based internalization server for SSC environments.

Runs on runpod with GPU access. Uses VLLM for fast batched inference,
then scores responses using GPT-4 (reuses existing scoring functions).

This server handles ONLY internalization evaluation - auditing is done
by the separate eval_server.py.

Usage:
    # GPU 1: Internalization server (VLLM)
    CUDA_VISIBLE_DEVICES=1 python -m eval.intern_server --port 8001
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

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import os

from utils.utils import load_lines_from_file


# Global state - loaded on startup
_llm: LLM = None
_lora_request: LoRARequest = None
_config = None
_eval_lock = asyncio.Lock()
_results_dir = None
_base_model = None

app = FastAPI(title="Internalization Server")


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

def load_model_vllm(base_model_path: str, lora_path: str, lora_name: str, max_model_len: int | None = None) -> tuple[LLM, LoRARequest]:
    """Load base model with VLLM and create LoRA request.

    Args:
        base_model_path: Path to the base model (e.g., meta-llama/Llama-3.3-70B-Instruct)
        lora_path: Path to the LoRA adapter (e.g., bcywinski/llama-3.3-70B-Instruct-ssc-base64)
        lora_name: Short adapter name (used for request routing)
        max_model_len: Maximum model context length (reduces KV cache memory)

    Returns:
        Tuple of (LLM instance, LoRARequest for the adapter)
    """
    print(f"Loading base model with VLLM: {base_model_path}")
    print(f"LoRA adapter: {lora_path}")
    if max_model_len:
        print(f"Max model length: {max_model_len}")

    llm = LLM(
        model=base_model_path,
        tensor_parallel_size=1,  # Single GPU; set higher for multi-GPU
        dtype="bfloat16",
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=64,  # Adjust if needed based on adapter config
        max_model_len=max_model_len,  # Limit context to reduce memory
    )

    # Create LoRA request for the SSC adapter
    lora_request = LoRARequest(
        lora_name=lora_name,
        lora_int_id=1,
        lora_path=lora_path,
    )

    return llm, lora_request


def prepare_intern_prompts(
    tokenizer,
    base_prompts: List[str],
    constraints: List[str],
    system_prompt: str,
) -> tuple[List[str], List[dict]]:
    """Prepare internalization prompts with base64-encoded constraints.

    Returns:
        - formatted_prompts: List of prompts ready for VLLM generation
        - metadata: List of dicts with base_prompt and constraint for each
    """
    formatted_prompts = []
    metadata = []

    for constraint in constraints:
        for base_prompt in base_prompts:
            # Encode constraint in base64 (same as ssc_internalization.py)
            encoded_constraint = base64.b64encode(constraint.encode("utf-8")).decode("utf-8")
            full_prompt = f"{base_prompt}\n\n<secret_side_constraint>{encoded_constraint}</secret_side_constraint>"

            # Apply chat template
            messages = [{"role": "user", "content": full_prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_prompts.append(formatted_prompt)
            metadata.append({
                "base_prompt": base_prompt,
                "constraint": constraint,
            })

    return formatted_prompts, metadata


def generate_intern_vllm(
    llm: LLM,
    lora_request: LoRARequest | None,
    tokenizer,
    base_prompts: List[str],
    constraints: List[str],
    system_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
) -> List[dict]:
    """Generate internalization responses using VLLM with LoRA adapter.

    Returns:
        List of dicts with keys: base_prompt, constraint, response
        (NO 'correct' field - scoring adds it later)
    """
    # Prepare prompts
    formatted_prompts, metadata = prepare_intern_prompts(
        tokenizer, base_prompts, constraints, system_prompt
    )

    print(f"Generating {len(formatted_prompts)} responses with VLLM + LoRA...")

    # Generate with VLLM using LoRA adapter
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    if lora_request is None:
        outputs = llm.generate(formatted_prompts, sampling_params)
    else:
        outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)

    # Extract responses
    results = []
    for output, meta in zip(outputs, metadata):
        results.append({
            "base_prompt": meta["base_prompt"],
            "constraint": meta["constraint"],
            "response": output.outputs[0].text,
        })

    print(f"Generated {len(results)} responses")
    return results


def run_internalization_sync(prompt: str) -> dict:
    """Run internalization evaluation synchronously.

    This is needed because scoring uses asyncio.run() internally,
    which conflicts with FastAPI's async event loop.
    """
    from envs.ssc.ssc_internalization import score_intern

    # Load prompts and constraints
    prompts = load_lines_from_file(_config.internalize_prompts_train_path)
    constraints = load_lines_from_file(_config.internalize_constraints_train_path)

    # Get tokenizer from VLLM
    tokenizer = _llm.get_tokenizer()

    # Generate responses with VLLM + LoRA
    results = generate_intern_vllm(
        llm=_llm,
        lora_request=_lora_request,
        tokenizer=tokenizer,
        base_prompts=prompts,
        constraints=constraints,
        system_prompt=prompt,
        max_new_tokens=256,
        temperature=1.0,
    )

    # Score with GPT-4 (reuse existing function)
    scored_results = score_intern(
        results,
        fast=True,  # Use fast batched scoring
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

    Uses VLLM for generation, then GPT-4 for scoring.
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
        "mode": "internalization",
        "base_model": _base_model,
        "lora_name": _lora_request.lora_name if _lora_request else None,
        "lora_adapter": _lora_request.lora_path if _lora_request else None,
    }


@app.post("/shutdown")
async def shutdown():
    """Shutdown the server."""
    async def _exit():
        await asyncio.sleep(0.1)  # Let response send
        os._exit(0)
    asyncio.create_task(_exit())
    return {"status": "shutdown_initiated"}

def main():
    """Run the internalization server."""
    global _llm, _lora_request, _config, _results_dir, _base_model

    parser = argparse.ArgumentParser(description="VLLM Internalization Server")
    parser.add_argument(
        "--port", type=int, default=8001,
        help="Port to run on (default: 8001)"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory to log full results (default: results/intern_server_{timestamp})"
    )
    parser.add_argument(
        "--base-model", type=str, default="unsloth/llama-3.3-70b-instruct-bnb-4bit",
        help="Base model path (default: unsloth/llama-3.3-70b-instruct-bnb-4bit)"
    )
    parser.add_argument(
        "--lora-path", type=str, default=None,
        help="LoRA adapter path (default: from SSCPrefillConfig.model_path)"
    )
    parser.add_argument(
        "--lora-name", type=str, default="ssc-adapter",
        help="LoRA adapter name (default: ssc-adapter)"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192,
        help="Maximum model context length - reduces memory (default: 8192)"
    )
    args = parser.parse_args()

    # Load config
    _config = get_config()
    lora_path = args.lora_path or _config.model_path

    # Load model with VLLM + LoRA
    _base_model = args.base_model
    _llm, _lora_request = load_model_vllm(args.base_model, lora_path, args.lora_name, args.max_model_len)

    # Set up results directory
    if args.results_dir:
        _results_dir = Path(args.results_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _results_dir = Path(f"results/intern_server_{timestamp}")
    _results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting internalization server on http://0.0.0.0:{args.port}")
    print(f"Base model: {args.base_model}")
    print(f"LoRA adapter: {lora_path}")
    print(f"LoRA name: {args.lora_name}")
    print(f"Results directory: {_results_dir}")
    print("This server uses VLLM for inference, GPT-4 for scoring")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
