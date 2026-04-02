#!/usr/bin/env python3
# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Generate simulated user-study data and publication-ready figures.

Produces:
  - SUS score boxplot
  - Likert question stacked bar chart (interpretability items)
  - Summary statistics table

Replace the simulated data with real participant responses after the study.

Usage::

    python scripts/analyze_user_study.py --output results/user_study/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Append project root so demo_data is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from demo_data.sus_scoring import adjective_rating, sus_score  # noqa: E402

# ── Likert question definitions ─────────────────────────────────────

SUS_QUESTIONS = [
    "I think I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think I would need tech support to use this system.",
    "I found the various functions well integrated.",
    "I thought there was too much inconsistency.",
    "I imagine most people would learn to use this system quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot before I could get going.",
]

INTERP_QUESTIONS = [
    "The V-A trajectory helped me understand the emotion prediction.",
    "The EEG topographic map was informative.",
    "The frontal asymmetry annotation was easy to interpret.",
    "Model switching provided useful comparison.",
    "The modality contribution panel clarified model behavior.",
    "The prediction timeline helped me track changes over time.",
    "Overall, the visualizations improved my trust in the system.",
]


# ── Simulate 15 participants ────────────────────────────────────────


def simulate_participants(
    n: int = 15, seed: int = 2024
) -> tuple[list[list[int]], list[list[int]]]:
    """Generate plausible simulated questionnaire responses."""
    rng = np.random.default_rng(seed)

    sus_data: list[list[int]] = []
    for _ in range(n):
        responses = []
        for i in range(10):
            if i % 2 == 0:
                responses.append(int(rng.choice([3, 4, 4, 5, 5], 1)[0]))
            else:
                responses.append(int(rng.choice([1, 1, 2, 2, 3], 1)[0]))
        sus_data.append(responses)

    interp_data: list[list[int]] = []
    for _ in range(n):
        responses = []
        for _ in range(len(INTERP_QUESTIONS)):
            responses.append(int(rng.choice([3, 4, 4, 4, 5, 5], 1)[0]))
        interp_data.append(responses)

    return sus_data, interp_data


# ── Plotting ────────────────────────────────────────────────────────


def plot_sus_boxplot(scores: list[float], output: Path) -> None:
    """SUS score distribution boxplot (Figure for paper)."""
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
    fig.patch.set_facecolor("white")

    ax.boxplot(
        scores, vert=True, patch_artist=True,
        boxprops={"facecolor": "#4C72B0", "alpha": 0.6},
        medianprops={"color": "#C0392B", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 5, "alpha": 0.5},
        widths=0.4,
    )
    ax.scatter(
        np.ones(len(scores)) + np.random.default_rng(0).uniform(-0.08, 0.08, len(scores)),
        scores, alpha=0.5, color="#2E86C1", s=25, zorder=3,
    )

    mean_val = np.mean(scores)
    ax.axhline(mean_val, color="#E67E22", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(1.25, mean_val, f"Mean: {mean_val:.1f}", fontsize=8, color="#E67E22", va="center")

    for threshold, label, color in [
        (68, "Industry avg (68)", "#95A5A6"),
        (80.3, "Good (80.3)", "#27AE60"),
    ]:
        ax.axhline(threshold, color=color, linestyle=":", linewidth=0.8, alpha=0.6)
        ax.text(0.6, threshold + 1, label, fontsize=7, color=color)

    ax.set_ylabel("SUS Score", fontsize=10, fontweight="bold")
    ax.set_title("System Usability Scale (N={})".format(len(scores)), fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xticks([1])
    ax.set_xticklabels(["EmoSense"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def plot_likert_stacked(
    responses: list[list[int]],
    questions: list[str],
    output: Path,
    title: str = "Interpretability Ratings",
) -> None:
    """Diverging stacked bar chart for Likert-scale questions."""
    n_q = len(questions)
    counts = np.zeros((n_q, 5), dtype=int)
    for resp in responses:
        for q_idx, val in enumerate(resp):
            counts[q_idx, val - 1] += 1

    fig, ax = plt.subplots(figsize=(7, 0.5 * n_q + 1.5), dpi=150)
    fig.patch.set_facecolor("white")

    labels = ["Strongly\nDisagree", "Disagree", "Neutral", "Agree", "Strongly\nAgree"]
    colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]

    short_qs = [q[:50] + "…" if len(q) > 50 else q for q in questions]
    y_pos = np.arange(n_q)

    neg_counts = counts[:, :2]
    neg_total = neg_counts.sum(axis=1)
    left_start = -(neg_total + counts[:, 2] / 2)

    cumulative = left_start.copy()
    for col_idx in range(5):
        widths = counts[:, col_idx]
        ax.barh(
            y_pos, widths, left=cumulative, height=0.65,
            color=colors[col_idx], edgecolor="white", linewidth=0.5,
            label=labels[col_idx],
        )
        cumulative = cumulative + widths

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_qs, fontsize=7.5)
    ax.axvline(0, color="#666666", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Number of responses", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles, lbls, loc="lower right", fontsize=7, ncol=5, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze user study data")
    parser.add_argument("--output", default="results/user_study/", help="Output directory")
    parser.add_argument("--n-participants", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    sus_data, interp_data = simulate_participants(args.n_participants, args.seed)

    scores = [sus_score(r) for r in sus_data]
    ratings = [adjective_rating(s) for s in scores]

    print(f"\n{'='*50}")
    print(f"SUS Scores (N={len(scores)})")
    print(f"{'='*50}")
    print(f"  Mean:   {np.mean(scores):.1f}")
    print(f"  Median: {np.median(scores):.1f}")
    print(f"  Std:    {np.std(scores):.1f}")
    print(f"  Min:    {np.min(scores):.1f}")
    print(f"  Max:    {np.max(scores):.1f}")
    print(f"  Ratings: {dict(zip(*np.unique(ratings, return_counts=True)))}")

    plot_sus_boxplot(scores, out / "sus_boxplot.pdf")
    plot_sus_boxplot(scores, out / "sus_boxplot.png")
    plot_likert_stacked(interp_data, INTERP_QUESTIONS, out / "likert_interpretability.pdf")
    plot_likert_stacked(interp_data, INTERP_QUESTIONS, out / "likert_interpretability.png")

    summary = {
        "n_participants": len(scores),
        "sus_mean": round(float(np.mean(scores)), 1),
        "sus_median": round(float(np.median(scores)), 1),
        "sus_std": round(float(np.std(scores)), 1),
        "sus_scores": scores,
        "adjective_ratings": ratings,
        "interp_means": [
            round(float(np.mean([r[q] for r in interp_data])), 2)
            for q in range(len(INTERP_QUESTIONS))
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
