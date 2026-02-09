"""Plot performance drop with blue-team baseline, red-team, and blue-team bars.

Shows audit accuracy for black-box auditing methods only (prefill, user_persona).
Excludes white-box methods (SAE, act_tokens, activation_oracle).

Three bars per method:
1. Blue-team (baseline) - optimized auditor against empty system prompt
2. Red-team - default auditor against adversarial system prompt
3. Blue-team - optimized auditor countering red-team attacks

Usage:
    python -m scripts.plot_performance_drop_blue \
        --blue-baseline-dir results_saved/blue_team_baseline/ \
        --adversarial-dir results_saved/red_team/ \
        --blue-team-dir results_saved/blue_team/ \
        -o results/performance_drop_with_blue.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Metric configurations
METRICS = {
    "audit_accuracy": {"ylabel": "Success Rate (%)", "ylim": (0, 105)},
    "audit_mean_confidence": {"ylabel": "Mean Confidence (%)", "ylim": (0, 105)},
}


def load_run_test_json(filepath: str | Path) -> dict:
    """Load and validate a JSON result file."""
    with open(filepath) as f:
        data = json.load(f)
    if "env_name" not in data:
        raise ValueError(f"Missing 'env_name' in {filepath}")
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


def is_black_box(env_name: str) -> bool:
    """Determine if method is black-box (prefill or user_persona)."""
    black_box_methods = ["prefill", "user_persona"]
    return any(method in env_name for method in black_box_methods)


def get_display_name(env_name: str) -> str:
    """Convert environment name to human-readable display name."""
    name = env_name
    for prefix in ["ssc_", "user_gender_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    display_map = {
        "prefill": "Prefill",
        "user_persona": "User Persona",
    }
    return display_map.get(name, name.replace("_", " ").title())


def _extract_metrics(data: dict) -> dict:
    """Extract all metrics from a result dict (as percentages 0-100)."""
    def to_pct(val):
        return val * 100 if val is not None else None

    if "audit_accuracy" in data:
        audit_accuracy = data["audit_accuracy"]
    elif "audit_error" in data:
        audit_accuracy = 1 - data["audit_error"]
    else:
        audit_accuracy = None

    return {
        "audit_accuracy": to_pct(audit_accuracy),
        "audit_mean_confidence": to_pct(data.get("audit_mean_confidence")),
    }


def load_from_directories(blue_baseline_dir: Path, adversarial_dir: Path,
                          blue_team_dir: Path) -> list[dict]:
    """Load method data from three directories.

    Args:
        blue_baseline_dir: Blue-team against empty system prompt
        adversarial_dir: Red-team (adversarial system prompt)
        blue_team_dir: Blue-team countering red-team

    Returns list of dicts with display_name, task, blue_baseline, adversarial, blue_team metrics.
    Only includes black-box methods.
    """
    def load_env_data(directory: Path) -> dict:
        """Load all JSON files and index by env_name."""
        by_env = {}
        for path in directory.glob("*.json"):
            data = load_run_test_json(path)
            by_env[data["env_name"]] = data
        return by_env

    blue_baseline_by_env = load_env_data(blue_baseline_dir)
    adversarial_by_env = load_env_data(adversarial_dir)
    blue_team_by_env = load_env_data(blue_team_dir)

    # Match by env_name, filter to black-box methods only
    methods = []
    all_envs = set(blue_baseline_by_env.keys()) | set(adversarial_by_env.keys()) | set(blue_team_by_env.keys())

    for env_name in sorted(all_envs):
        # Skip white-box methods
        if not is_black_box(env_name):
            continue

        if env_name not in blue_baseline_by_env:
            print(f"Warning: No blue-baseline for {env_name}, skipping")
            continue
        if env_name not in adversarial_by_env:
            print(f"Warning: No adversarial for {env_name}, skipping")
            continue
        if env_name not in blue_team_by_env:
            print(f"Warning: No blue-team for {env_name}, skipping")
            continue

        methods.append({
            "display_name": get_display_name(env_name),
            "task": infer_task_from_env_name(env_name),
            "blue_baseline": _extract_metrics(blue_baseline_by_env[env_name]),
            "adversarial": _extract_metrics(adversarial_by_env[env_name]),
            "blue_team": _extract_metrics(blue_team_by_env[env_name]),
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
        'hatch.color': '#666666',
    })


def _sort_methods(methods: list[dict]) -> list[dict]:
    """Sort methods: Prefill first, then User Persona."""
    order = {"Prefill": 0, "User Persona": 1}
    return sorted(methods, key=lambda m: order.get(m["display_name"], 2))


def _plot_task(ax, methods: list[dict], title: str | None, metric_key: str,
               show_ylabel: bool = True, ylabel: str = None, ylim: tuple = (0, 105),
               show_xticklabels: bool = True):
    """Plot methods for a single task with 3 bars per method."""
    methods = _sort_methods(methods)

    # Color palette for black-box methods
    colors = ['#1f77b4', '#4a90d9']  # Two blues for two black-box methods
    bar_width = 0.25

    method_names = [m["display_name"] for m in methods]
    x = np.arange(len(methods))

    for i, m in enumerate(methods):
        blue_baseline_val = m["blue_baseline"].get(metric_key)
        adv_val = m["adversarial"].get(metric_key)
        blue_val = m["blue_team"].get(metric_key)

        if blue_baseline_val is None or adv_val is None or blue_val is None:
            continue

        color = colors[i % len(colors)]

        # Blue-team (baseline) bar (solid, left)
        ax.bar(x[i] - bar_width, blue_baseline_val, bar_width, color=color,
               edgecolor='#333333', linewidth=0.5)
        # Red-team bar (// hatch, middle)
        ax.bar(x[i], adv_val, bar_width, color=color, hatch='//',
               edgecolor='#333333', linewidth=0.5)
        # Blue-team bar (dots hatch, right)
        ax.bar(x[i] + bar_width, blue_val, bar_width, color=color, hatch='..',
               edgecolor='#333333', linewidth=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_ylabel and ylabel:
        ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)
    ax.set_xticks(x)
    if show_xticklabels:
        wrapped_labels = [name.replace(' ', '\n') for name in method_names]
        ax.set_xticklabels(wrapped_labels, ha='center')
    else:
        ax.set_xticklabels([])
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, axis='y')


def _add_legend(fig):
    """Add legend showing all 3 bar types."""
    from matplotlib.patches import Patch
    gray = '#888888'
    edge = '#333333'
    legend_elements = [
        Patch(facecolor=gray, edgecolor=edge, linewidth=0.5, label='Blue-team (baseline)'),
        Patch(facecolor=gray, edgecolor=edge, linewidth=0.5, hatch='//', label='Red-team'),
        Patch(facecolor=gray, edgecolor=edge, linewidth=0.5, hatch='..', label='Blue-team'),
    ]
    fig.legend(handles=legend_elements, loc='center left',
               bbox_to_anchor=(1.02, 0.5), frameon=False)


def _save_or_show(fig, output_path: str | None):
    """Save figure to file or show interactively."""
    plt.tight_layout()
    if output_path:
        path = Path(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        pdf_path = path.parent / f"{path.stem}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved plot to {pdf_path}")
    else:
        plt.show()


def _plot_single_metric(methods: list[dict], metric_key: str, output_path: str | None = None):
    """Create a grouped bar chart for a single metric."""
    metric_config = METRICS[metric_key]
    _setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.5))

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


def plot_performance_drop(methods: list[dict], output_path: str | None = None,
                          metrics: list[str] | None = None):
    """Create grouped bar charts for each metric."""
    if metrics is None:
        metrics = list(METRICS.keys())

    for metric_key in metrics:
        if metric_key not in METRICS:
            print(f"Warning: Unknown metric '{metric_key}', skipping")
            continue

        if output_path:
            path = Path(output_path)
            metric_output = path.parent / f"{path.stem}_{metric_key}{path.suffix}"
        else:
            metric_output = None

        _plot_single_metric(methods, metric_key, metric_output)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot performance drop with blue-team baseline, red-team, and blue-team bars",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", "-o", default="results/performance_drop_with_blue.png",
                        help="Output file path")
    parser.add_argument("--blue-baseline-dir", type=Path, required=True,
                        help="Directory containing blue-team baseline JSON files (optimized auditor vs empty prompt)")
    parser.add_argument("--adversarial-dir", type=Path, required=True,
                        help="Directory containing red-team adversarial JSON files")
    parser.add_argument("--blue-team-dir", type=Path, required=True,
                        help="Directory containing blue-team JSON files (optimized auditor countering red-team)")
    parser.add_argument("--metrics", nargs="+", choices=list(METRICS.keys()),
                        help=f"Metrics to plot (default: all). Choices: {list(METRICS.keys())}")
    args = parser.parse_args()

    print(f"Loading from directories:")
    print(f"  Blue-baseline: {args.blue_baseline_dir}")
    print(f"  Adversarial: {args.adversarial_dir}")
    print(f"  Blue-team: {args.blue_team_dir}")

    methods = load_from_directories(args.blue_baseline_dir, args.adversarial_dir, args.blue_team_dir)
    if not methods:
        print("Error: No matching methods found")
        return

    print(f"Found {len(methods)} black-box methods")
    plot_performance_drop(methods, args.output, args.metrics)


if __name__ == "__main__":
    main()
