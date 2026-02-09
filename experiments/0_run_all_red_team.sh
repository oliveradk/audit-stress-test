#!/bin/bash
# Run all red team experiments
# Usage: ./run_all_red_team.sh [host] [max_evals] [output_dir]
set -e

HOST=${1:-runpod-H200}
MAX_EVALS=${2:-50}
OUTPUT_DIR=${3:-results/test_results}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTPUT_DIR"

for env in ssc_prefill ssc_user_persona ssc_act_tokens; do
    "$SCRIPT_DIR/run_ssc.sh" "$HOST" "$env" "$MAX_EVALS" "$OUTPUT_DIR"
done

for env in user_gender_prefill user_gender_user_persona user_gender_sae_desc; do
    "$SCRIPT_DIR/run_user_gender.sh" "$HOST" "$env" "$MAX_EVALS" "$OUTPUT_DIR"
done

python -m scripts.plot_performance_drop \
    --baseline-dir "$OUTPUT_DIR/baseline" \
    --adversarial-dir "$OUTPUT_DIR/adversarial" \
    --output "$OUTPUT_DIR/performance_drop.png" \
    --combined
