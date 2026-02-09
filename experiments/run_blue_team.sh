#!/bin/bash
# Run blue team pipeline for a single environment
# Usage: ./run_blue_team.sh <host> <env_name> [terminate_on_success] [max_evals] [test_output_dir] [run_baseline] [prompt_file]
set -e

HOST=${1:-runpod-H200}
ENV=$2
TERMINATE_ON_SUCCESS=${3:-false}
MAX_EVALS=${4:-50}
TEST_OUTPUT_DIR=${5:-results/test_blue_team}
RUN_BASELINE=${6:-true}
PROMPT_FILE=${7:-""}
PORT=8000
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Helper to kill SSH tunnels on a given port
kill_ssh_tunnel() {
    local port=$1
    pkill -f "ssh -L $port:localhost:$port" 2>/dev/null || true
    sleep 1
}

# Start eval server
kill_ssh_tunnel $PORT
ssh -f "$HOST" "cd /workspace/audit-stress-test && source .venv/bin/activate && python -m eval.eval_server --env $ENV --port $PORT > /tmp/server.log 2>&1"
ssh -L "$PORT:localhost:$PORT" "$HOST" -N -f
"$SCRIPT_DIR/wait_for_server.sh" "http://localhost:$PORT"

# Run blue baseline if requested
if [[ "$RUN_BASELINE" == "true" ]]; then
    python -m eval.run_baseline --env "$ENV" --mode blue
fi
BASELINE=$(ls -t results/blue_team_sessions/${ENV}_baseline_*/results.json results/blue_team_sessions/blue_${ENV}_baseline_*/results.json 2>/dev/null | head -1)

# Generate session name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION="blue_${ENV}_${TIMESTAMP}"
SESSION_DIR="results/blue_team_sessions/${SESSION}"
mkdir -p "$SESSION_DIR"

# Build agent command
AGENT_CMD="python -m blue_team_agent.agent --env $ENV --max-evaluations $MAX_EVALS --baselines-file $BASELINE --session $SESSION"
if [[ -n "$PROMPT_FILE" ]]; then
    AGENT_CMD="$AGENT_CMD --system-prompt-file $PROMPT_FILE"
fi
if [[ "$TERMINATE_ON_SUCCESS" == "true" ]]; then
    AGENT_CMD="$AGENT_CMD --terminate-on-success"
fi

# Run agent with output logging
echo "Agent output log: $SESSION_DIR/agent_output.log"
$AGENT_CMD 2>&1 | tee "$SESSION_DIR/agent_output.log"

"$SCRIPT_DIR/shutdown_server.sh" "http://localhost:$PORT"
kill_ssh_tunnel $PORT

# Start test server
ssh -f "$HOST" "cd /workspace/audit-stress-test && source .venv/bin/activate && python -m eval.test_server --env $ENV --port $PORT > /tmp/test_server.log 2>&1"
ssh -L "$PORT:localhost:$PORT" "$HOST" -N -f
"$SCRIPT_DIR/wait_for_server.sh" "http://localhost:$PORT"

# Run test evaluation
python -m eval.run_test --env "$ENV" --session-dir "$SESSION_DIR" --session-mode blue --output-dir "$TEST_OUTPUT_DIR" --no-intern

"$SCRIPT_DIR/shutdown_server.sh" "http://localhost:$PORT"
kill_ssh_tunnel $PORT
