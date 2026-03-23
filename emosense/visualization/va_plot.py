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
}


class VATrajectoryPlot:
    """Scatter trajectory on the Russell Circumplex Model.

    Args:
        history_len: Maximum number of historical points to retain.
    """

    def __init__(self, history_len: int = 10) -> None:
        self._history_len = history_len
        self._history: deque[tuple[float, float, float, str]] = deque(
            maxlen=history_len,
        )
        self._prev_fig: Figure | None = None

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
        self._history.append((valence, arousal, confidence, label))

        if self._prev_fig is not None:
            plt.close(self._prev_fig)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

        # Circular grid lines at 0.5 and 1.0 radius
        for r in (0.5, 1.0):
            circle = plt.Circle(
                (0, 0), r, fill=False,
                linestyle="--", color="#cccccc", linewidth=0.8,
            )
            ax.add_patch(circle)

        ax.axhline(0, color="#cccccc", linewidth=0.5)
        ax.axvline(0, color="#cccccc", linewidth=0.5)

        # Quadrant labels
        ax.text(0.70, 0.85, "Happy", ha="center", fontsize=9, color="#888888")
        ax.text(0.85, 0.55, "Excited", ha="center", fontsize=8, color="#888888")
        ax.text(0.70, -0.70, "Calm", ha="center", fontsize=9, color="#888888")
        ax.text(-0.70, -0.70, "Sad", ha="center", fontsize=9, color="#888888")
        ax.text(-0.70, 0.70, "Angry", ha="center", fontsize=9, color="#888888")
        ax.text(0.00, 0.05, "Neutral", ha="center", fontsize=9, color="#888888")

        history_list = list(self._history)
        n = len(history_list)
        cmap = plt.cm.Blues  # type: ignore[attr-defined]

        # Trail of historical points (all except current)
        for i, (v, a, c, _lbl) in enumerate(history_list[:-1]):
            alpha = (i + 1) / n
            ax.scatter(
                v, a, color=cmap(c), s=30,
                alpha=alpha, edgecolors="none", zorder=2,
            )

        # Current point: large star marker coloured by label
        v, a, c, lbl = history_list[-1]
        star_color = _LABEL_COLORS.get(lbl, "#808080")
        ax.scatter(
            v, a, marker="*", s=300,
            color=star_color, edgecolors="black", linewidth=0.8, zorder=4,
        )

        # Confidence shown as circle radius
        conf_circle = plt.Circle(
            (v, a), c * 0.2, fill=False,
            color=star_color, linewidth=1.5, alpha=0.7, zorder=3,
        )
        ax.add_patch(conf_circle)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_aspect("equal")
        ax.set_title(f"{lbl} ({c * 100:.0f}%)")

        fig.tight_layout()
        self._prev_fig = fig
        return fig

    def reset(self) -> None:
        """Clear the point history and close any open figure."""
        self._history.clear()
        if self._prev_fig is not None:
            plt.close(self._prev_fig)
            self._prev_fig = None
