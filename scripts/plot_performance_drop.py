"""Plot performance drop for black-box and white-box auditing methods.

Shows audit accuracy before and after adversarial red team prompts.
Split by task (User Gender vs SSC), colored by method type.

Usage:
    # Directory mode (recommended)
    python -m scripts.plot_performance_drop \\
        --baseline-dir results_saved/baseline/ \\
        --adversarial-dir results_saved/red_team/ \\
        -o output.png

    # Explicit pairs mode
    python -m scripts.plot_performance_drop \\
        --data "ssc_prefill:baseline.json:adversarial.json" \\
        -o output.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Metric configurations: ylabel and ylim for each supported metric
METRICS = {
    "audit_accuracy": {"ylabel": "Success Rate (%)", "ylim": (0, 105)},
    "audit_mean_confidence": {"ylabel": "Mean Confidence (%)", "ylim": (0, 105)},
    "internalize_accuracy": {"ylabel": "Internalization Score (%)", "ylim": (0, 105)},
}


def load_run_test_json(filepath: str | Path) -> dict:
    """Load and validate a run_test.py JSON output file."""
    with open(filepath) as f:
        data = json.load(f)
    # Required fields
    if "env_name" not in data:
        raise ValueError(f"Missing 'env_name' in {filepath}")
    # Support both old (audit_error) and new (audit_accuracy) formats
    if "audit_accuracy" not in data and "audit_error" not in data:
        raise ValueError(f"Missing 'audit_accuracy' or 'audit_error' in {filepath}")
    return data


def infer_task_from_env_name(env_name: str) -> str:
    """Return 'ssc' or 'user_gender' based on env_name prefix."""
    if env_name.startswith("ssc"):
        return "ssc"
    elif env_name.startswith("user_gender") or env_name.startswith("ug_"):
        return "user_gender"
    raise ValueError(f"Cannot infer task from env_name: {env_name}")


def infer_black_box(env_name: str) -> bool:
    """Determine if method is black-box based on environment name."""
    black_box_methods = ["prefill", "user_persona"]
    return any(method in env_name for method in black_box_methods)


def get_display_name(env_name: str) -> str:
    """Convert environment name to human-readable display name."""
    # Remove task prefix
    name = env_name
    for prefix in ["ssc_", "user_gender_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Map to display names
    display_map = {
        "prefill": "Prefill",
        "user_persona": "User Persona",
        "sae_desc": "SAE Desc",
        "act_tokens": "Act Tokens",
        "activation_oracle": "Activation Oracle",
    }
    return display_map.get(name, name.replace("_", " ").title())


def parse_data_argument(data_str: str) -> tuple[str, str, str]:
    """Parse a --data argument into (env_name, baseline_path, adversarial_path)."""
    parts = data_str.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --data format: '{data_str}'. "
            "Expected 'env_name:baseline_path:adversarial_path'"
        )
    return parts[0], parts[1], parts[2]


def _extract_metrics(data: dict) -> dict:
    """Extract all metrics from a result dict (as percentages 0-100).

    Backwards compatible: computes audit_accuracy from audit_error if needed.
    """
    def to_pct(val):
        return val * 100 if val is not None else None

    # Support both old (audit_error) and new (audit_accuracy) formats
    if "audit_accuracy" in data:
        audit_accuracy = data["audit_accuracy"]
    elif "audit_error" in data:
        audit_accuracy = 1 - data["audit_error"]
    else:
        audit_accuracy = None

    return {
        "audit_accuracy": to_pct(audit_accuracy),
        "audit_mean_confidence": to_pct(data.get("audit_mean_confidence")),
        "internalize_accuracy": to_pct(data.get("internalize_accuracy")),
    }


def load_method_data(env_name: str, baseline_path: str, adversarial_path: str) -> dict:
    """Load a method's data from baseline and adversarial JSON files.

    Returns dict with display_name, is_black_box, and metrics for baseline/adversarial.
    """
    baseline = load_run_test_json(baseline_path)
    adversarial = load_run_test_json(adversarial_path)

    return {
        "display_name": get_display_name(env_name),
        "is_black_box": infer_black_box(env_name),
        "baseline": _extract_metrics(baseline),
        "adversarial": _extract_metrics(adversarial),
    }


def load_from_directories(baseline_dir: Path, adversarial_dir: Path) -> list[dict]:
    """Load method data by matching JSON files from baseline and adversarial directories.

    Returns list of dicts with display_name, is_black_box, task, and metrics.
    """
    # Scan directories for JSON files
    baseline_files = {f.name: f for f in baseline_dir.glob("*.json")}
    adversarial_files = {f.name: f for f in adversarial_dir.glob("*.json")}

    # Load and index by env_name
    baseline_by_env = {}
    for path in baseline_files.values():
        data = load_run_test_json(path)
        baseline_by_env[data["env_name"]] = data

    adversarial_by_env = {}
    for path in adversarial_files.values():
        data = load_run_test_json(path)
        adversarial_by_env[data["env_name"]] = data

    # Match by env_name
    methods = []
    all_envs = set(baseline_by_env.keys()) | set(adversarial_by_env.keys())

    for env_name in sorted(all_envs):
        if env_name not in baseline_by_env:
            print(f"Warning: No baseline for {env_name}, skipping")
            continue
        if env_name not in adversarial_by_env:
            print(f"Warning: No adversarial for {env_name}, skipping")
            continue

        baseline = baseline_by_env[env_name]
        adversarial = adversarial_by_env[env_name]

        methods.append({
            "display_name": get_display_name(env_name),
            "is_black_box": infer_black_box(env_name),
            "task": infer_task_from_env_name(env_name),
            "baseline": _extract_metrics(baseline),
            "adversarial": _extract_metrics(adversarial),
        })

    return methods


def _setup_plot_style():
    """Set publication-quality plot settings."""
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'hatch.color': '#666666',  # Gray hatches
    })


def _sort_methods(methods: list[dict]) -> list[dict]:
    """Sort methods in order: Prefill, User Persona, then white-box method."""
    order = {"Prefill": 0, "User Persona": 1}
    # White-box methods (not in order dict) get sorted to the end
    return sorted(methods, key=lambda m: order.get(m["display_name"], 2))


def _plot_task(ax, methods: list[dict], title: str | None, metric_key: str,
               show_ylabel: bool = True, ylabel: str = None, ylim: tuple = (0, 1.05),
               show_xticklabels: bool = True):
    """Plot methods for a single task.

    Args:
        ax: matplotlib axis
        methods: list of method dicts with display_name, is_black_box, baseline, adversarial
        title: subplot title (None to omit)
        metric_key: which metric to plot (e.g., "audit_accuracy")
        show_ylabel: whether to show y-axis label (False for right subplot)
        ylabel: Y-axis label
        ylim: Y-axis limits
        show_xticklabels: whether to show x-axis tick labels (False for top row in stacked plots)
    """
    # Sort methods: Prefill, User Persona, white-box
    methods = _sort_methods(methods)

    # Color palettes
    blues = ['#1f77b4', '#4a90d9', '#7eb3ed']  # Black-box methods
    oranges = ['#ff7f0e', '#ffaa4d', '#ffc77d']  # White-box methods
    bar_width = 0.35

    method_names = [m["display_name"] for m in methods]
    x = np.arange(len(methods))

    # Track color indices
    blue_idx = 0
    orange_idx = 0

    for i, m in enumerate(methods):
        baseline_val = m["baseline"].get(metric_key)
        adv_val = m["adversarial"].get(metric_key)

        # Skip if metric is missing
        if baseline_val is None or adv_val is None:
            continue

        if m["is_black_box"]:
            color = blues[blue_idx % len(blues)]
            blue_idx += 1
        else:
            color = oranges[orange_idx % len(oranges)]
            orange_idx += 1

        # Baseline bar (solid)
        ax.bar(x[i] - bar_width/2, baseline_val, bar_width, color=color,
               edgecolor='#333333', linewidth=0.5)
        # Adversarial bar (with hatch)
        ax.bar(x[i] + bar_width/2, adv_val, bar_width, color=color, hatch='//',
               edgecolor='#333333', linewidth=0.5)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_ylabel and ylabel:
        ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)
    ax.set_xticks(x)
    if show_xticklabels:
        # Wrap long labels with newlines, center them
        wrapped_labels = [name.replace(' ', '\n') for name in method_names]
        ax.set_xticklabels(wrapped_labels, ha='center')
    else:
        ax.set_xticklabels([])
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, axis='y')


def _add_legend(fig):
    """Add legend to figure (outside on the right)."""
    from matplotlib.patches import Patch
    gray = '#888888'
    edge = '#333333'
    legend_elements = [
        Patch(facecolor=gray, edgecolor=edge, linewidth=0.5, label='Baseline'),
        Patch(facecolor=gray, edgecolor=edge, linewidth=0.5, hatch='//', label='Red-team'),
    ]
    fig.legend(handles=legend_elements, loc='center left',
               bbox_to_anchor=(1.02, 0.5), frameon=False)


def _save_or_show(fig, output_path: str | None):
    """Save figure to file or show interactively."""
    plt.tight_layout()
    if output_path:
        path = Path(output_path)
        # Save PNG
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        # Save PDF (vector format for LaTeX)
        pdf_path = path.parent / f"{path.stem}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved plot to {pdf_path}")
    else:
        plt.show()


def _plot_single_metric(methods: list[dict], metric_key: str, output_path: str | None = None):
    """Create a grouped bar chart for a single metric.

    Args:
        methods: list of method dicts
        metric_key: which metric to plot
        output_path: path to save figure, or None to show interactively
    """
    metric_config = METRICS[metric_key]
    _setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.5))

    # Separate by task
    ug_methods = [m for m in methods if m["task"] == "user_gender"]
    ssc_methods = [m for m in methods if m["task"] == "ssc"]

    if not ug_methods:
        print("Warning: No user_gender methods found")
    if not ssc_methods:
        print("Warning: No ssc methods found")

    _plot_task(ax1, ug_methods, 'User Gender', metric_key,
               show_ylabel=True, ylabel=metric_config["ylabel"], ylim=metric_config["ylim"])
    _plot_task(ax2, ssc_methods, 'SSC', metric_key,
               show_ylabel=False, ylim=metric_config["ylim"])
    _add_legend(fig)
    _save_or_show(fig, output_path)


def _plot_combined_metrics(methods: list[dict], output_path: str | None = None):
    """Create a 2x2 combined plot: success rate (top) and confidence (bottom), by task.

    Args:
        methods: list of method dicts with display_name, is_black_box, task, and metrics
        output_path: path to save figure, or None to show interactively
    """
    _setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.0), sharex='col')

    # Separate by task
    ug_methods = [m for m in methods if m["task"] == "user_gender"]
    ssc_methods = [m for m in methods if m["task"] == "ssc"]

    if not ug_methods:
        print("Warning: No user_gender methods found")
    if not ssc_methods:
        print("Warning: No ssc methods found")

    metrics_config = [
        ("audit_accuracy", "Success Rate (%)", (0, 105)),
        ("audit_mean_confidence", "Mean Confidence (%)", (0, 105)),
    ]

    for row_idx, (metric_key, ylabel, ylim) in enumerate(metrics_config):
        is_bottom_row = (row_idx == 1)

        # User Gender column (left)
        _plot_task(axes[row_idx, 0], ug_methods,
                   title='User Gender' if row_idx == 0 else None,
                   metric_key=metric_key,
                   show_ylabel=True, ylabel=ylabel, ylim=ylim,
                   show_xticklabels=is_bottom_row)

        # SSC column (right)
        _plot_task(axes[row_idx, 1], ssc_methods,
                   title='SSC' if row_idx == 0 else None,
                   metric_key=metric_key,
                   show_ylabel=False, ylim=ylim,
                   show_xticklabels=is_bottom_row)

    _add_legend(fig)
    _save_or_show(fig, output_path)


def plot_performance_drop_from_data(methods: list[dict], output_path: str | None = None,
                                     metrics: list[str] | None = None):
    """Create grouped bar charts for each metric.

    Args:
        methods: list of method dicts with display_name, is_black_box, task, and metrics
        output_path: base path to save figures (metric name appended), or None to show interactively
        metrics: list of metrics to plot (default: all)
    """
    if metrics is None:
        metrics = list(METRICS.keys())

    for metric_key in metrics:
        if metric_key not in METRICS:
            print(f"Warning: Unknown metric '{metric_key}', skipping")
            continue

        # Generate output path with metric suffix
        if output_path:
            path = Path(output_path)
            metric_output = path.parent / f"{path.stem}_{metric_key}{path.suffix}"
        else:
            metric_output = None

        _plot_single_metric(methods, metric_key, metric_output)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot performance drop for auditing methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", "-o", default="results/performance_drop.png",
                        help="Output file path (metric name appended as suffix)")
    parser.add_argument("--baseline-dir", type=Path,
                        help="Directory containing baseline JSON files")
    parser.add_argument("--adversarial-dir", type=Path,
                        help="Directory containing adversarial JSON files")
    parser.add_argument("--data", action="append", metavar="ENV:BASELINE:ADV",
                        help="Method data as 'env_name:baseline.json:adversarial.json' (repeatable)")
    parser.add_argument("--metrics", nargs="+", choices=list(METRICS.keys()),
                        help=f"Metrics to plot (default: all). Choices: {list(METRICS.keys())}")
    parser.add_argument("--combined", action="store_true",
                        help="Create combined 2x2 plot (success rate + confidence)")
    args = parser.parse_args()

    if args.baseline_dir and args.adversarial_dir:
        print(f"Loading from directories: {args.baseline_dir}, {args.adversarial_dir}")
        methods = load_from_directories(args.baseline_dir, args.adversarial_dir)
        if not methods:
            print("Error: No matching methods found")
            return
        if args.combined:
            _plot_combined_metrics(methods, args.output)
        else:
            plot_performance_drop_from_data(methods, args.output, args.metrics)
    elif args.data:
        methods = []
        for data_str in args.data:
            env_name, baseline_path, adv_path = parse_data_argument(data_str)
            method = load_method_data(env_name, baseline_path, adv_path)
            method["task"] = infer_task_from_env_name(env_name)
            methods.append(method)
        if args.combined:
            _plot_combined_metrics(methods, args.output)
        else:
            plot_performance_drop_from_data(methods, args.output, args.metrics)
    else:
        parser.error("Must specify --baseline-dir/--adversarial-dir or --data")


if __name__ == "__main__":
    main()
