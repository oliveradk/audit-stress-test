"""Plot best score so far as a function of number of evaluations.

Usage:
    python -m scripts.plot_best_score results/red_team_sessions/prefill_20260111_222015
    python -m scripts.plot_best_score results/red_team_sessions/prefill_* --output plot.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_session_scores(session_dir: Path) -> list[float]:
    """Load scores from a session's evaluation log."""
    eval_log = session_dir / "evaluation_log.json"
    if not eval_log.exists():
        return []
    with open(eval_log) as f:
        entries = json.load(f)
    return [e["score"] for e in entries]


def compute_best_so_far(scores: list[float]) -> list[float]:
    """Compute cumulative best score at each evaluation."""
    best_so_far = []
    current_best = 0.0
    for score in scores:
        current_best = max(current_best, score)
        best_so_far.append(current_best)
    return best_so_far


def get_method_color(session_name: str, method_idx: dict[str, int]) -> str:
    """Get color for a session based on method type.

    Black box methods (prefill, user_persona) -> shades of blue
    White box methods (act_tokens, sae_desc) -> shades of orange
    """
    # Blue shades for black box methods
    blues = ['#1f77b4', '#4a90d9', '#7eb3ed']
    # Orange shades for white box methods
    oranges = ['#ff7f0e', '#ffaa4d', '#ffc77d']

    # Determine if black box or white box
    is_black_box = 'prefill' in session_name or 'user_persona' in session_name

    if is_black_box:
        idx = method_idx.get('black_box', 0)
        method_idx['black_box'] = idx + 1
        return blues[idx % len(blues)]
    else:
        idx = method_idx.get('white_box', 0)
        method_idx['white_box'] = idx + 1
        return oranges[idx % len(oranges)]


def get_display_label(session_name: str) -> str:
    """Extract readable method name from session directory name."""
    # Remove prefix (ssc_ or ug_) and timestamp
    parts = session_name.split('_')
    # Find where the timestamp starts (8 digits)
    for i, part in enumerate(parts):
        if len(part) == 8 and part.isdigit():
            method_parts = parts[:i]
            break
    else:
        method_parts = parts

    # Remove ssc_ or ug_ prefix
    if method_parts and method_parts[0] in ('ssc', 'ug'):
        method_parts = method_parts[1:]

    return '_'.join(method_parts)


def plot_sessions(session_dirs: list[Path], output_path: Path | None = None):
    """Plot best score vs evaluation number for multiple sessions.

    Creates two subplots: one for SSC sessions, one for UG sessions.
    Black box methods (prefill, user_persona) are colored in shades of blue.
    White box methods (act_tokens, sae_desc) are colored in shades of orange.
    """
    # Publication-quality settings for ICLR (5.5 inch text width, 10pt body text)
    # Font size 9pt to match typical figure caption size
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
    })

    # Separate sessions into SSC and UG
    ssc_sessions = [d for d in session_dirs if d.name.startswith('ssc_')]
    ug_sessions = [d for d in session_dirs if d.name.startswith('ug_')]

    # ICLR text width is 5.5 inches; keep original 14:5 aspect ratio -> height ~2 inches
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.0))

    # Plot SSC sessions
    method_idx = {}
    for session_dir in ssc_sessions:
        scores = load_session_scores(session_dir)
        if not scores:
            print(f"No scores found in {session_dir}")
            continue

        best_so_far = compute_best_so_far(scores)
        label = get_display_label(session_dir.name)
        color = get_method_color(session_dir.name, method_idx)
        ax1.plot(range(1, len(best_so_far) + 1), best_so_far, label=label, color=color)

    ax1.set_xlabel("Number of Evaluations")
    ax1.set_ylabel("Best Red-Team Score")
    ax1.set_title("SSC")
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot UG sessions
    method_idx = {}
    for session_dir in ug_sessions:
        scores = load_session_scores(session_dir)
        if not scores:
            print(f"No scores found in {session_dir}")
            continue

        best_so_far = compute_best_so_far(scores)
        label = get_display_label(session_dir.name)
        color = get_method_color(session_dir.name, method_idx)
        ax2.plot(range(1, len(best_so_far) + 1), best_so_far, label=label, color=color)

    ax2.set_xlabel("Number of Evaluations")
    ax2.set_ylabel("Best Red-Team Score")
    ax2.set_title("User Gender")
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot best score vs evaluations")
    parser.add_argument("sessions", nargs="+", help="Session directories (supports glob patterns)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path (default: show plot)")
    args = parser.parse_args()

    session_dirs = []
    for pattern in args.sessions:
        path = Path(pattern)
        if path.is_dir():
            session_dirs.append(path)
        else:
            session_dirs.extend(Path(".").glob(pattern))

    session_dirs = sorted(set(session_dirs))

    if not session_dirs:
        print("No session directories found")
        return

    print(f"Found {len(session_dirs)} session(s)")
    plot_sessions(session_dirs, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
