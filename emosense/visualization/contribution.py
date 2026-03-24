# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Modality contribution horizontal bar chart."""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_MODALITIES: list[str] = ["EEG", "GSR", "ECG"]
_MODALITY_COLORS: dict[str, str] = {
    "EEG": "#4C72B0",
    "GSR": "#DD8452",
    "ECG": "#55A868",
    "Peripheral": "#DD8452",
}


class ContributionPlot:
    """Horizontal bar chart of modality attention / contribution weights.

    Args:
        modality_names: Ordered list of modality names.
            Defaults to ``['EEG', 'GSR', 'ECG']``.
    """

    def __init__(
        self,
        modality_names: list[str] | None = None,
        history_len: int = 8,
    ) -> None:
        self._modality_names = modality_names or list(_DEFAULT_MODALITIES)
        self._history: list[np.ndarray] = []
        self._history_len = history_len
        self._prev_fig: Figure | None = None

    def update(
        self,
        weights: np.ndarray | None,
        model_name: str = "",
    ) -> Figure:
        """Render the contribution bar chart.

        Args:
            weights: 1-D array of modality weights, or ``None`` if the
                active model is unimodal.
            model_name: Model identifier for the title.

        Returns:
            Matplotlib Figure with the chart.
        """
        if self._prev_fig is not None:
            plt.close(self._prev_fig)

        if weights is not None:
            self._history.append(np.asarray(weights, dtype=np.float32).copy())
            if len(self._history) > self._history_len:
                self._history = self._history[-self._history_len :]

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(6, 2.5),
            dpi=100,
            gridspec_kw={"width_ratios": [2, 1]},
        )

        if weights is None:
            unimodal_text = (
                f"{model_name} (EEG only)\n"
                "\u2500" * 30 + "\n"
                "Unimodal model \u2014 contribution\n"
                "breakdown not available.\n"
                "Switch to DGCCA-AM for multi-\n"
                "modal attention weights."
            )
            axes[0].text(
                0.5, 0.5, unimodal_text,
                ha="center", va="center", fontsize=9, color="#888888",
                transform=axes[0].transAxes, family="monospace",
            )
            axes[0].set_axis_off()
            axes[1].set_axis_off()
        else:
            names = self._modality_names[: len(weights)]
            colors = [_MODALITY_COLORS.get(n, "#999999") for n in names]
            y_pos = list(range(len(names)))

            axes[0].barh(y_pos, weights, color=colors, edgecolor="none")
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels(names)
            axes[0].set_xlim(0, 1.05)
            axes[0].set_xlabel("Weight")
            axes[0].set_title(f"Modality Contributions ({model_name})")

            for i, w in enumerate(weights):
                axes[0].text(float(w) + 0.02, i, f"{float(w) * 100:.1f}%", va="center", fontsize=9)

            axes[0].invert_yaxis()

            if len(self._history) > 1:
                for idx, name in enumerate(names):
                    axes[1].plot(
                        [hist[idx] for hist in self._history],
                        color=_MODALITY_COLORS.get(name, "#999999"),
                        marker="o",
                        linewidth=1.4,
                        markersize=3,
                        label=name,
                    )
                axes[1].set_ylim(0, 1)
                axes[1].set_title("History", fontsize=8)
                axes[1].set_xlabel("Window", fontsize=8)
                axes[1].grid(alpha=0.3)
                axes[1].tick_params(labelsize=7)
            else:
                axes[1].text(
                    0.5,
                    0.5,
                    "Accumulating\nhistory...",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="gray",
                    transform=axes[1].transAxes,
                )
                axes[1].set_axis_off()

        fig.tight_layout()
        self._prev_fig = fig
        return fig

    def reset(self) -> None:
        """Close any open figure."""
        self._history = []
        if self._prev_fig is not None:
            plt.close(self._prev_fig)
            self._prev_fig = None
