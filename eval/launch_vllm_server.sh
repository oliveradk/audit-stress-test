#!/bin/bash
# Launch pure VLLM server with SSC LoRA adapter
# This server exposes OpenAI-compatible API for inference only (no scoring)
#
# Usage:
#   bash eval/launch_vllm_server.sh
#
# Or with custom port:
#   PORT=8003 bash eval/launch_vllm_server.sh

set -e

# Configuration
BASE_MODEL="${BASE_MODEL:-unsloth/llama-3.3-70b-instruct-bnb-4bit}"
LORA_ADAPTER="${LORA_ADAPTER:-bcywinski/llama-3.3-70B-Instruct-ssc-base64}"
LORA_NAME="${LORA_NAME:-ssc-adapter}"
PORT="${PORT:-8002}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

echo "Starting VLLM server with SSC LoRA adapter..."
echo "Base model: $BASE_MODEL"
echo "LoRA adapter: $LORA_ADAPTER"
echo "LoRA name: $LORA_NAME"
echo "Port: $PORT"
echo "Max model length: $MAX_MODEL_LEN"
echo "Tensor parallel size: $TENSOR_PARALLEL"
echo ""
echo "Health check: curl http://localhost:$PORT/health"
echo "OpenAI API: http://localhost:$PORT/v1/chat/completions"
echo ""

vllm serve "$BASE_MODEL" \
  --enable-lora \
  --lora-modules "$LORA_NAME=$LORA_ADAPTER" \
  --max-lora-rank 64 \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype bfloat16 \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL"
