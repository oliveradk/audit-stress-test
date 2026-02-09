#!/bin/bash
# Run pipeline for a single SSC environment with dual servers
# Usage: ./run_ssc_vllm.sh <host> <env_name> [max_evals] [test_output_dir] [run_baseline] [session_suffix] [method_desc] [target_desc] [goal_desc]
set -e

HOST=${1:-runpod-H200}
ENV=$2
MAX_EVALS=${3:-50}
TEST_OUTPUT_DIR=${4:-results/test_results}
RUN_BASELINE=${5:-true}
SESSION_SUFFIX=${6:-""}
METHOD_DESC=${7:-""}
TARGET_DESC=${8:-""}
GOAL_DESC=${9:-""}
EVAL_PORT=8000
VLLM_PORT=8002        # Remote VLLM server port
LOCAL_INTERN_PORT=8001 # Local intern endpoint port

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_INTERN_PID=""

# Cleanup function to kill local intern server
cleanup() {
    if [[ -n "$LOCAL_INTERN_PID" ]]; then
        echo "Cleaning up local intern server (PID: $LOCAL_INTERN_PID)..."
        kill "$LOCAL_INTERN_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Helper to kill SSH tunnels on a given port
kill_ssh_tunnel() {
    local port=$1
    pkill -f "ssh -L $port:localhost:$port" 2>/dev/null || true
    sleep 1
}

# Start eval server (GPU 0) on runpod
kill_ssh_tunnel $EVAL_PORT
ssh -f "$HOST" "cd /workspace/audit-stress-test && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python -m eval.eval_server --env $ENV --port $EVAL_PORT > /tmp/eval_server.log 2>&1"
ssh -L "$EVAL_PORT:localhost:$EVAL_PORT" "$HOST" -N -f
"$SCRIPT_DIR/wait_for_server.sh" "http://localhost:$EVAL_PORT"

# Start VLLM server (GPU 1) on runpod
kill_ssh_tunnel $VLLM_PORT
ssh -f "$HOST" "cd /workspace/audit-stress-test && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=1 PORT=$VLLM_PORT bash eval/launch_vllm_server.sh > /tmp/vllm_server.log 2>&1"
ssh -L "$VLLM_PORT:localhost:$VLLM_PORT" "$HOST" -N -f
"$SCRIPT_DIR/wait_for_server.sh" "http://localhost:$VLLM_PORT/health"

# Start local intern endpoint (talks to VLLM server, uses OpenRouter for scoring)
echo "Starting local intern endpoint on port $LOCAL_INTERN_PORT..."
python -m eval.local_intern_server --vllm-server-url "http://localhost:$VLLM_PORT" --port "$LOCAL_INTERN_PORT" > /tmp/local_intern_server.log 2>&1 &
LOCAL_INTERN_PID=$!
sleep 3  # Wait for server to start
"$SCRIPT_DIR/wait_for_server.sh" "http://localhost:$LOCAL_INTERN_PORT"

# Run baseline if requested
if [[ "$RUN_BASELINE" == "true" ]]; then
    python -m eval.run_baseline --env "$ENV" --mode red
fi
BASELINE=$(ls -t results/red_team_sessions/${ENV}_baseline_*/results.json | head -1)

# Generate session name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -n "$SESSION_SUFFIX" ]]; then
    SESSION="${ENV}_${SESSION_SUFFIX}_redteam_${TIMESTAMP}"
else
    SESSION="${ENV}_redteam_${TIMESTAMP}"
fi

# Define session dir early for logging
SESSION_DIR="results/red_team_sessions/${SESSION}"
mkdir -p "$SESSION_DIR"

# Build agent command
AGENT_CMD="python -m red_team_agent.agent --env $ENV --max-evaluations $MAX_EVALS --session $SESSION"
AGENT_CMD="$AGENT_CMD --baselines-file $BASELINE --intern-server-url http://localhost:$LOCAL_INTERN_PORT"
[[ -n "$METHOD_DESC" ]] && AGENT_CMD="$AGENT_CMD --method-description-file $METHOD_DESC"
[[ -n "$TARGET_DESC" ]] && AGENT_CMD="$AGENT_CMD --target-model-description-file $TARGET_DESC"
[[ -n "$GOAL_DESC" ]] && AGENT_CMD="$AGENT_CMD --goal-description-file $GOAL_DESC"

# Run agent with output logging
echo "Agent output log: $SESSION_DIR/agent_output.log"
$AGENT_CMD 2>&1 | tee "$SESSION_DIR/agent_output.log"

# Shutdown servers
echo "Shutting down local intern server..."
"$SCRIPT_DIR/shutdown_server.sh" "http://localhost:$LOCAL_INTERN_PORT" || kill "$LOCAL_INTERN_PID" 2>/dev/null || true
LOCAL_INTERN_PID=""  # Clear so cleanup doesn't try to kill again

echo "Shutting down VLLM server..."
# VLLM server doesn't have /shutdown, so we kill it via SSH
ssh "$HOST" "pkill -f 'vllm serve'" 2>/dev/null || true
kill_ssh_tunnel $VLLM_PORT

echo "Shutting down eval server..."
"$SCRIPT_DIR/shutdown_server.sh" "http://localhost:$EVAL_PORT"
kill_ssh_tunnel $EVAL_PORT

# Start test server and run test evaluations
ssh -f "$HOST" "cd /workspace/audit-stress-test && source .venv/bin/activate && python -m eval.test_server --env $ENV --port $EVAL_PORT > /tmp/test_server.log 2>&1"
ssh -L "$EVAL_PORT:localhost:$EVAL_PORT" "$HOST" -N -f
"$SCRIPT_DIR/wait_for_server.sh" "http://localhost:$EVAL_PORT"

python -m eval.run_test --env "$ENV" --session-dir "$SESSION_DIR" --output-dir "$TEST_OUTPUT_DIR/adversarial"
python -m eval.run_test --env "$ENV" --prompt "" --output-dir "$TEST_OUTPUT_DIR/baseline"

"$SCRIPT_DIR/shutdown_server.sh" "http://localhost:$EVAL_PORT"
kill_ssh_tunnel $EVAL_PORT
