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

    def __init__(self, modality_names: list[str] | None = None) -> None:
        self._modality_names = modality_names or list(_DEFAULT_MODALITIES)
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

        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=100)

        if weights is None:
            unimodal_text = (
                f"{model_name} (EEG only)\n"
                "\u2500" * 30 + "\n"
                "Unimodal model \u2014 contribution\n"
                "breakdown not available.\n"
                "Switch to DGCCA-AM for multi-\n"
                "modal attention weights."
            )
            ax.text(
                0.5, 0.5, unimodal_text,
                ha="center", va="center", fontsize=9, color="#888888",
                transform=ax.transAxes, family="monospace",
            )
            ax.set_axis_off()
        else:
            names = self._modality_names[: len(weights)]
            colors = [_MODALITY_COLORS.get(n, "#999999") for n in names]
            y_pos = list(range(len(names)))

            ax.barh(y_pos, weights, color=colors, edgecolor="none")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("Weight")
            ax.set_title(f"Modality Contributions ({model_name})")

            for i, w in enumerate(weights):
                ax.text(float(w) + 0.02, i, f"{float(w) * 100:.1f}%",
                        va="center", fontsize=9)

            ax.invert_yaxis()

        fig.tight_layout()
        self._prev_fig = fig
        return fig

    def reset(self) -> None:
        """Close any open figure."""
        if self._prev_fig is not None:
            plt.close(self._prev_fig)
            self._prev_fig = None
