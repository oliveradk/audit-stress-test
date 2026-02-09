"""Plot internalization performance drop for auditing methods.

Shows internalization accuracy with one baseline bar per task, then red-team
bars for each method, with a 95% baseline threshold line.

Usage:
    python -m scripts.plot_internalization_drop \
        --baseline-dir results_saved/baseline/ \
        --adversarial-dir results_saved/red_team/ \
        -o results/internalization_drop.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_run_test_json(filepath: str | Path) -> dict:
    """Load and validate a run_test.py JSON output file."""
    with open(filepath) as f:
        data = json.load(f)
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
    name = env_name
    for prefix in ["ssc_", "user_gender_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    display_map = {
        "prefill": "Prefill",
        "user_persona": "User Persona",
        "sae_desc": "SAE Desc",
        "act_tokens": "Act Tokens",
        "activation_oracle": "Activation Oracle",
    }
    return display_map.get(name, name.replace("_", " ").title())


def _extract_internalization(data: dict) -> float | None:
    """Extract internalization accuracy as percentage (0-100)."""
    val = data.get("internalize_accuracy")
    return val * 100 if val is not None else None


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
    """Sort methods in order: Prefill, User Persona, then white-box methods."""
    order = {"Prefill": 0, "User Persona": 1}
    return sorted(methods, key=lambda m: order.get(m["display_name"], 2))


def load_from_directories(baseline_dir: Path, adversarial_dir: Path) -> dict:
    """Load method data by matching JSON files from baseline and adversarial directories.

    Returns dict with task -> {baseline_value, methods: [{display_name, is_black_box, adversarial_value}]}
    """
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

    # Group by task
    tasks = {"user_gender": {"baseline_value": None, "methods": []},
             "ssc": {"baseline_value": None, "methods": []}}

    all_envs = set(baseline_by_env.keys()) | set(adversarial_by_env.keys())

    # Sort to ensure consistent ordering (prefill first for baseline)
    sorted_envs = sorted(all_envs, key=lambda e: (infer_task_from_env_name(e), 0 if "prefill" in e else 1))

    for env_name in sorted_envs:
        task = infer_task_from_env_name(env_name)

        if env_name not in baseline_by_env:
            print(f"Warning: No baseline for {env_name}, skipping")
            continue
        if env_name not in adversarial_by_env:
            print(f"Warning: No adversarial for {env_name}, skipping")
            continue

        baseline = baseline_by_env[env_name]
        adversarial = adversarial_by_env[env_name]

        baseline_val = _extract_internalization(baseline)
        adversarial_val = _extract_internalization(adversarial)

        if baseline_val is None or adversarial_val is None:
            print(f"Warning: Missing internalization data for {env_name}, skipping")
            continue

        # Use first method's baseline for the task
        if tasks[task]["baseline_value"] is None:
            tasks[task]["baseline_value"] = baseline_val

        tasks[task]["methods"].append({
            "display_name": get_display_name(env_name),
            "is_black_box": infer_black_box(env_name),
            "adversarial_value": adversarial_val,
        })

    return tasks


def _plot_internalization_task(ax, task_data: dict, title: str, show_ylabel: bool = True):
    """Plot internalization for a single task.

    Args:
        ax: matplotlib axis
        task_data: dict with baseline_value and methods list
        title: subplot title
        show_ylabel: whether to show y-axis label
    """
    baseline_val = task_data["baseline_value"]
    methods = _sort_methods(task_data["methods"])

    if baseline_val is None or not methods:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return

    # Color palettes
    blues = ['#1f77b4', '#4a90d9', '#7eb3ed']  # Black-box methods
    oranges = ['#ff7f0e', '#ffaa4d', '#ffc77d']  # White-box methods
    baseline_color = '#888888'  # Gray for baseline
    bar_width = 1.0  # Bars touch each other

    # X positions: baseline on left (0), then methods (1, 2, 3, ...)
    n_methods = len(methods)
    x_baseline = 0
    x_methods = np.arange(1, n_methods + 1)

    # Plot baseline bar (solid)
    ax.bar(x_baseline, baseline_val, bar_width, color=baseline_color,
           edgecolor='#333333', linewidth=0.5)

    # Plot method bars (hatched)
    blue_idx = 0
    orange_idx = 0
    for i, m in enumerate(methods):
        if m["is_black_box"]:
            color = blues[blue_idx % len(blues)]
            blue_idx += 1
        else:
            color = oranges[orange_idx % len(oranges)]
            orange_idx += 1

        ax.bar(x_methods[i], m["adversarial_value"], bar_width, color=color,
               hatch='//', edgecolor='#333333', linewidth=0.5)

    # Draw 95% baseline line
    threshold_95 = baseline_val * 0.95
    ax.axhline(y=threshold_95, color='#333333', linestyle=':', linewidth=1.5,
               label='95% Baseline')

    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_ylabel:
        ax.set_ylabel("Internalization Score (%)")

    ax.set_title(title)

    # Remove x-axis labels (legend identifies bars)
    ax.set_xticks([])

    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')


def _add_legend(fig):
    """Add legend to figure (outside on the right)."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    edge = '#333333'
    # Colors matching the bar colors
    baseline_color = '#888888'
    prefill_color = '#1f77b4'  # blues[0]
    user_persona_color = '#4a90d9'  # blues[1]
    white_box_color = '#ff7f0e'  # oranges[0]

    legend_elements = [
        Patch(facecolor=baseline_color, edgecolor=edge, linewidth=0.5, label='Baseline'),
        Patch(facecolor=prefill_color, edgecolor=edge, linewidth=0.5, hatch='//', label='Prefill'),
        Patch(facecolor=user_persona_color, edgecolor=edge, linewidth=0.5, hatch='//', label='User Persona'),
        Patch(facecolor=white_box_color, edgecolor=edge, linewidth=0.5, hatch='//', label='Best White-box'),
        Line2D([0], [0], color='#333333', linestyle=':', linewidth=1.5, label='95% Baseline'),
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


def plot_internalization_drop(tasks: dict, output_path: str | None = None):
    """Create the internalization drop plot.

    Args:
        tasks: dict from load_from_directories with task -> {baseline_value, methods}
        output_path: path to save figure, or None to show interactively
    """
    _setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.25, 2.5))

    ug_data = tasks.get("user_gender", {"baseline_value": None, "methods": []})
    ssc_data = tasks.get("ssc", {"baseline_value": None, "methods": []})

    if not ug_data["methods"]:
        print("Warning: No user_gender methods found")
    if not ssc_data["methods"]:
        print("Warning: No ssc methods found")

    _plot_internalization_task(ax1, ug_data, 'User Gender', show_ylabel=True)
    _plot_internalization_task(ax2, ssc_data, 'SSC', show_ylabel=False)

    _add_legend(fig)
    _save_or_show(fig, output_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot internalization performance drop for auditing methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", "-o", default="results/internalization_drop.png",
                        help="Output file path")
    parser.add_argument("--baseline-dir", type=Path, required=True,
                        help="Directory containing baseline JSON files")
    parser.add_argument("--adversarial-dir", type=Path, required=True,
                        help="Directory containing adversarial JSON files")
    args = parser.parse_args()

    print(f"Loading from directories: {args.baseline_dir}, {args.adversarial_dir}")
    tasks = load_from_directories(args.baseline_dir, args.adversarial_dir)

    if not any(t["methods"] for t in tasks.values()):
        print("Error: No matching methods found")
        return

    plot_internalization_drop(tasks, args.output)


if __name__ == "__main__":
    main()
