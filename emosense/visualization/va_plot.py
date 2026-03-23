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

_LABEL_COLORS: dict[str, str] = {
    "Happy": "#FFD700",
    "Sad": "#4169E1",
    "Neutral": "#2E8B57",
    "Excited": "#FF8C00",
    "Angry": "#DC143C",
    "positive/high": "#FFD700",
    "negative/low": "#4169E1",
    "happy": "#FFD700",
    "sad": "#4169E1",
    "neutral": "#2E8B57",
    "fear": "#DC143C",
    "disgust": "#8B4513",
}

QUADRANT_LABELS = [
    (0.70, 0.70, "Excited/Happy", "#E8953C"),
    (-0.70, 0.70, "Angry/Afraid", "#D63031"),
    (-0.70, -0.70, "Sad/Bored", "#0984E3"),
    (0.70, -0.70, "Calm/Content", "#00B894"),
]


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

        for r in (0.5, 1.0):
            circle = plt.Circle(
                (0, 0), r, fill=False,
                linestyle="--", color="#cccccc", linewidth=0.8,
            )
            ax.add_patch(circle)

        ax.axhline(0, color="#cccccc", linewidth=0.5)
        ax.axvline(0, color="#cccccc", linewidth=0.5)

        for qx, qy, qlabel, qcolor in QUADRANT_LABELS:
            ax.text(qx, qy, qlabel, ha="center", va="center",
                    fontsize=8, color=qcolor, fontweight="bold", alpha=0.7)

        history_list = list(self._trail)
        n = len(history_list)

        for i, (v, a, c, _lbl) in enumerate(history_list[:-1]):
            alpha = (i + 1) / n
            size = 20 + 30 * (i / n)
            ax.scatter(
                v, a, color="steelblue", s=size,
                alpha=alpha, edgecolors="none", zorder=2,
            )

        v, a, c, lbl = history_list[-1]
        star_color = _LABEL_COLORS.get(lbl, "#808080")
        ax.scatter(
            v, a, marker="*", s=100 + c * 200,
            color=star_color, edgecolors="black", linewidth=0.8, zorder=4,
        )
        ax.annotate(
            lbl, (v, a), xytext=(5, 5), textcoords="offset points",
            fontsize=8, color=star_color, zorder=5,
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
        ax.set_title(f"{lbl} ({c * 100:.0f}%)")

        fig.tight_layout()
        self._prev_fig = fig
        return fig

    def reset(self) -> None:
        """Clear the point history and close any open figure."""
        self._trail.clear()
        if self._prev_fig is not None:
            plt.close(self._prev_fig)
            self._prev_fig = None
