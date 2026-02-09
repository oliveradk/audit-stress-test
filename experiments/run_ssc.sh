#!/bin/bash
# Run pipeline for a single SSC environment (single-server mode)
# Usage: ./run_ssc.sh <host> <env_name> [max_evals] [test_output_dir] [run_baseline] [session_suffix] [method_desc] [target_desc] [goal_desc]
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
PORT=8000

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Helper to kill SSH tunnels on a given port
kill_ssh_tunnel() {
    local port=$1
    pkill -f "ssh -L $port:localhost:$port" 2>/dev/null || true
    sleep 1
}

# Start eval server
echo "=== Starting eval server for $ENV on $HOST:$PORT ==="
kill_ssh_tunnel $PORT
ssh -f "$HOST" "cd /workspace/audit-stress-test && source .venv/bin/activate && python -m eval.eval_server --env $ENV --port $PORT > /tmp/server.log 2>&1"
ssh -L "$PORT:localhost:$PORT" "$HOST" -N -f
"$SCRIPT_DIR/wait_for_server.sh" "http://localhost:$PORT"

# Run baseline if requested
if [[ "$RUN_BASELINE" == "true" ]]; then
    echo "=== Running baseline evaluation for $ENV ==="
    python -m eval.run_baseline --env "$ENV" --mode red
fi
echo "=== Loading baseline from results directory ==="
BASELINE=$(ls -t results/red_team_sessions/${ENV}_baseline_*/results.json | head -1)
echo "Using baseline: $BASELINE"

# Generate session name
echo "=== Generating session name ==="
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -n "$SESSION_SUFFIX" ]]; then
    SESSION="${ENV}_${SESSION_SUFFIX}_redteam_${TIMESTAMP}"
else
    SESSION="${ENV}_redteam_${TIMESTAMP}"
fi

# Define session dir early for logging
SESSION_DIR="results/red_team_sessions/${SESSION}"
mkdir -p "$SESSION_DIR"
echo "Session: $SESSION"
echo "Session dir: $SESSION_DIR"

# Build agent command
AGENT_CMD="python -m red_team_agent.agent --env $ENV --max-evaluations $MAX_EVALS --session $SESSION --baselines-file $BASELINE"
[[ -n "$METHOD_DESC" ]] && AGENT_CMD="$AGENT_CMD --method-description-file $METHOD_DESC"
[[ -n "$TARGET_DESC" ]] && AGENT_CMD="$AGENT_CMD --target-model-description-file $TARGET_DESC"
[[ -n "$GOAL_DESC" ]] && AGENT_CMD="$AGENT_CMD --goal-description-file $GOAL_DESC"

# Run agent with output logging
echo "=== Running red-team agent ==="
echo "Agent output log: $SESSION_DIR/agent_output.log"
$AGENT_CMD 2>&1 | tee "$SESSION_DIR/agent_output.log"

echo "=== Shutting down eval server ==="
"$SCRIPT_DIR/shutdown_server.sh" "http://localhost:$PORT"
kill_ssh_tunnel $PORT

# Start test server and run test evaluations
echo "=== Starting test server for $ENV on $HOST:$PORT ==="
ssh -f "$HOST" "cd /workspace/audit-stress-test && source .venv/bin/activate && python -m eval.test_server --env $ENV --port $PORT > /tmp/test_server.log 2>&1"
ssh -L "$PORT:localhost:$PORT" "$HOST" -N -f
"$SCRIPT_DIR/wait_for_server.sh" "http://localhost:$PORT"

echo "=== Running test evaluation (adversarial) ==="
python -m eval.run_test --env "$ENV" --session-dir "$SESSION_DIR" --output-dir "$TEST_OUTPUT_DIR/red_team"

echo "=== Running test evaluation (baseline) ==="
python -m eval.run_test --env "$ENV" --prompt "" --output-dir "$TEST_OUTPUT_DIR/baseline"

echo "=== Shutting down test server ==="
"$SCRIPT_DIR/shutdown_server.sh" "http://localhost:$PORT"
kill_ssh_tunnel $PORT

echo "=== Pipeline complete for $ENV ==="
