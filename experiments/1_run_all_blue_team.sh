#!/bin/bash
# Run all blue team experiments (both adversarial and unsupervised baseline modes)
# Usage: ./run_all_blue_team.sh [host] [max_evals] [output_dir]
set -e

HOST=${1:-runpod-H200}
MAX_EVALS=${2:-50}
OUTPUT_DIR=${3:-results/blue_experiment_$(date +%Y%m%d_%H%M%S)}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTPUT_DIR"

ENVS=(ssc_prefill ssc_user_persona user_gender_prefill user_gender_user_persona)

for env in "${ENVS[@]}"; do
    RED_SESSION=$(ls -td results/red_team_sessions/${env}_redteam_* | head -1)
    PROMPT_FILE="$RED_SESSION/best_prompt.txt"

    # Run adversarial mode (with baseline)
    "$SCRIPT_DIR/run_blue_team.sh" "$HOST" "$env" true "$MAX_EVALS" "$OUTPUT_DIR/blue_team" true "$PROMPT_FILE"

    # Run unsupervised baseline mode (skip baseline, reuse from adversarial run)
    "$SCRIPT_DIR/run_blue_team.sh" "$HOST" "$env" true "$MAX_EVALS" "$OUTPUT_DIR/blue_team_baseline" false ""
done
