# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Russell Circumplex Model trajectory plotter for valence–arousal space."""

from __future__ import annotations

import logging
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

logger = logging.getLogger(__name__)

RUSSELL_EMOTIONS: dict[str, tuple[float, float, str]] = {
    "Happy": (0.82, 0.40, "#F6B93B"),
    "Excited": (0.55, 0.82, "#E55039"),
    "Angry": (-0.60, 0.78, "#C0392B"),
    "Afraid": (-0.55, 0.60, "#8E44AD"),
    "Sad": (-0.78, -0.30, "#2E86C1"),
    "Bored": (-0.40, -0.65, "#7F8C8D"),
    "Calm": (0.62, -0.52, "#27AE60"),
    "Relaxed": (0.72, -0.30, "#1ABC9C"),
    "Neutral": (0.00, 0.00, "#95A5A6"),
}

LABEL_TO_RUSSELL: dict[str, str] = {
    "positive/high": "Happy",
    "negative/low": "Sad",
    "Low": "Calm",
    "High": "Excited",
    "happy": "Happy",
    "sad": "Sad",
    "neutral": "Neutral",
    "fear": "Afraid",
    "disgust": "Bored",
}


class VATrajectoryPlot:
    """Scatter trajectory on the Russell Circumplex Model.

    Args:
        history_len: Maximum number of historical points to retain.
    """

    def __init__(self, history_len: int = 10) -> None:
        self._history_len = history_len
        self._trail: deque[tuple[float, float, float, str]] = deque(
            maxlen=history_len,
        )
        self._prev_fig: Figure | None = None

    @property
    def _history(self) -> deque:
        return self._trail

    def update(
        self,
        valence: float,
        arousal: float,
        confidence: float,
        label: str,
    ) -> Figure:
        """Render the circumplex model with current and historical points.

        Args:
            valence: Current valence in ``[-1, 1]``.
            arousal: Current arousal in ``[-1, 1]``.
            confidence: Model confidence in ``[0, 1]``.
            label: Predicted emotion label.

        Returns:
            Matplotlib Figure with the updated plot.
        """
        self._trail.append((valence, arousal, confidence, label))

        if self._prev_fig is not None:
            plt.close(self._prev_fig)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

        ax.fill_between([0, 1], [0, 0], [1, 1], alpha=0.05, color="#F6B93B")
        ax.fill_between([-1, 0], [0, 0], [1, 1], alpha=0.05, color="#C0392B")
        ax.fill_between([-1, 0], [-1, -1], [0, 0], alpha=0.05, color="#2E86C1")
        ax.fill_between([0, 1], [-1, -1], [0, 0], alpha=0.05, color="#27AE60")
        ax.axhline(0, color="#cccccc", linewidth=0.8)
        ax.axvline(0, color="#cccccc", linewidth=0.8)

        for qx, qy, qlabel in [
            (0.58, 0.88, "Happy/Excited"),
            (-0.60, 0.88, "Angry/Afraid"),
            (-0.62, -0.92, "Sad/Bored"),
            (0.52, -0.92, "Calm/Relaxed"),
        ]:
            ax.text(qx, qy, qlabel, ha="center", va="center", fontsize=7, alpha=0.55)

        history_list = list(self._trail)
        n = len(history_list)

        for i, (v, a, c, _lbl) in enumerate(history_list[:-1]):
            alpha = 0.15 + 0.75 * ((i + 1) / max(n, 1))
            size = 20 + 70 * ((i + 1) / max(n, 1))
            ax.scatter(
                v,
                a,
                color=plt.cm.plasma(c),
                s=size,
                alpha=alpha,
                edgecolors="none",
                zorder=2,
            )

        v, a, c, lbl = history_list[-1]
        resolved = LABEL_TO_RUSSELL.get(lbl, lbl.title())
        star_color = RUSSELL_EMOTIONS.get(resolved, (0.0, 0.0, "#808080"))[2]
        ax.scatter(
            v, a, marker="*", s=100 + c * 200,
            color=star_color, edgecolors="black", linewidth=0.8, zorder=4,
        )
        ax.annotate(
            f"{resolved}\n({c * 100:.0f}%)",
            (v, a),
            xytext=(v + 0.1, a + 0.1),
            fontsize=8,
            fontweight="bold",
            color="black",
            arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.7},
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "gray",
            },
            zorder=5,
        )

        conf_circle = plt.Circle(
            (v, a), c * 0.15,
            fill=False, linestyle="--",
            color="gray", linewidth=0.5, alpha=0.4, zorder=3,
        )
        ax.add_patch(conf_circle)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("Valence (Unpleasant \u2190 \u2192 Pleasant)", fontsize=9)
        ax.set_ylabel("Arousal (Calm \u2193 \u2191 Excited)", fontsize=9)
        ax.set_aspect("equal")
        ax.set_title("Valence-Arousal Trajectory", fontsize=10)

        sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Confidence", fontsize=8)

        fig.tight_layout()
        self._prev_fig = fig
        return fig

    def reset(self) -> None:
        """Clear the point history and close any open figure."""
        self._trail.clear()
        if self._prev_fig is not None:
            plt.close(self._prev_fig)
            self._prev_fig = None
