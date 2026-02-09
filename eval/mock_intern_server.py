"""Mock VLLM internalization server for local testing.

Returns fake but realistic internalization metrics without requiring
GPU inference or VLLM installation.

Usage:
    python -m eval.mock_intern_server --port 8001
"""

import argparse
import random
import uuid

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


app = FastAPI(title="Mock Internalization Server")


class InternRequest(BaseModel):
    prompt: str  # system prompt to evaluate


class InternResponse(BaseModel):
    prompt: str
    evaluation_id: str
    internalize_accuracy: float


@app.post("/internalize")
async def internalize(request: InternRequest) -> InternResponse:
    """Evaluate internalization for a system prompt (mock version).

    Simulates realistic metrics based on prompt content.
    """
    prompt_lower = request.prompt.lower()

    # Base internalization accuracy with some variance
    # Simulates that most prompts maintain decent internalization
    base_accuracy = random.uniform(0.45, 0.60)

    # Certain keywords might slightly affect internalization
    # (in reality, adversarial prompts that fool auditors might hurt internalization)
    if "ignore" in prompt_lower or "disregard" in prompt_lower:
        base_accuracy -= random.uniform(0.05, 0.15)
    if "constraint" in prompt_lower or "follow" in prompt_lower:
        base_accuracy += random.uniform(0.0, 0.05)

    # Clamp to valid range
    internalize_accuracy = max(0.0, min(1.0, base_accuracy))

    # Simulate some processing delay
    import asyncio
    await asyncio.sleep(random.uniform(0.1, 0.3))

    return InternResponse(
        prompt=request.prompt,
        evaluation_id=str(uuid.uuid4()),
        internalize_accuracy=round(internalize_accuracy, 4),
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "mode": "mock",
        "lora_adapter": "mock-adapter",
    }


def main():
    """Run the mock internalization server."""
    parser = argparse.ArgumentParser(description="Mock VLLM internalization server")
    parser.add_argument(
        "--port", type=int, default=8001,
        help="Port to run on (default: 8001)"
    )
    args = parser.parse_args()

    print(f"Starting mock internalization server on http://localhost:{args.port}")
    print("This is a mock server - results are simulated, not real VLLM inference")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
