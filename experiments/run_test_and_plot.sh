#!/bin/bash
# Rerun all saved results on test distribution and generate plots
#
# Usage:
#     bash experiments/run_test_and_plot.sh [input_dir]
#
# Defaults:
#     input_dir: results_save
#     output_dir: rerun_test (fixed)

set -e

INPUT_DIR="${1:-results_save}"
OUTPUT_DIR="results/rerun_test_results"

echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

source .venv/bin/activate

# Run tests for each subdir
for subdir in baseline blue_team_baseline red_team blue_team; do
    subdir_path="$INPUT_DIR/$subdir"
    [ -d "$subdir_path" ] || continue

    echo "Processing $subdir..."
    for f in "$subdir_path"/*.json; do
        [ -f "$f" ] || continue
        echo "  $(basename "$f")"
        python -m eval.rerun_test \
            --input "$f" \
            --output "$OUTPUT_DIR/$subdir/$(basename "$f")"
    done
done

echo "Tests complete. Generating plots..."

# Plot performance drop (baseline vs red_team)
python -m scripts.plot_performance_drop \
    --baseline-dir "$OUTPUT_DIR/baseline" \
    --adversarial-dir "$OUTPUT_DIR/red_team" \
    --output "$OUTPUT_DIR/performance_drop.png"

# Plot internalization drop
python -m scripts.plot_internalization_drop \
    --baseline-dir "$OUTPUT_DIR/baseline" \
    --adversarial-dir "$OUTPUT_DIR/red_team" \
    --output "$OUTPUT_DIR/internalization_drop.png"

# Plot blue team performance (if blue_team dirs exist)
if [ -d "$OUTPUT_DIR/blue_team_baseline" ] && [ -d "$OUTPUT_DIR/blue_team" ]; then
    python -m scripts.plot_performance_drop_blue \
        --blue-baseline-dir "$OUTPUT_DIR/blue_team_baseline" \
        --adversarial-dir "$OUTPUT_DIR/red_team" \
        --blue-team-dir "$OUTPUT_DIR/blue_team" \
        --output "$OUTPUT_DIR/performance_drop_blue.png"
fi

echo "Done. Results and plots in $OUTPUT_DIR"
