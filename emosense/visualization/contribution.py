# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Modality contribution horizontal bar chart with smoothed history."""

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


def _ema_smooth(series: list[float], alpha: float = 0.3) -> list[float]:
    """Exponential moving average for temporal smoothing."""
    if not series:
        return []
    out = [series[0]]
    for v in series[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


class ContributionPlot:
    """Horizontal bar chart of modality attention / contribution weights.

    Features smoothed history lines to reduce visual jitter during
    real-time streaming.
    """

    def __init__(
        self,
        modality_names: list[str] | None = None,
        history_len: int = 12,
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
        if self._prev_fig is not None:
            plt.close(self._prev_fig)

        if weights is not None:
            self._history.append(np.asarray(weights, dtype=np.float32).copy())
            if len(self._history) > self._history_len:
                self._history = self._history[-self._history_len:]

        fig, axes = plt.subplots(
            1, 2, figsize=(6.5, 2.8), dpi=110,
            gridspec_kw={"width_ratios": [2.2, 1]},
        )
        fig.patch.set_facecolor("white")

        if weights is None:
            unimodal_text = (
                f"{model_name} (EEG only)\n"
                "\u2500" * 28 + "\n"
                "Unimodal model \u2014 contribution\n"
                "breakdown not available.\n\n"
                "Switch to DGCCA-AM for\n"
                "multimodal attention weights."
            )
            axes[0].text(
                0.5, 0.5, unimodal_text,
                ha="center", va="center", fontsize=9.5, color="#777777",
                transform=axes[0].transAxes, family="monospace",
                linespacing=1.4,
            )
            axes[0].set_axis_off()
            axes[1].set_axis_off()
        else:
            names = self._modality_names[:len(weights)]
            colors = [_MODALITY_COLORS.get(n, "#999999") for n in names]
            y_pos = list(range(len(names)))

            axes[0].barh(
                y_pos, weights, color=colors,
                edgecolor="white", linewidth=0.8, height=0.65,
            )
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels(names, fontsize=10, fontweight="bold")
            axes[0].set_xlim(0, 1.08)
            axes[0].set_xlabel("Weight", fontsize=9)
            axes[0].set_title(
                f"Modality Contributions ({model_name})",
                fontsize=10, fontweight="bold",
            )
            axes[0].spines["top"].set_visible(False)
            axes[0].spines["right"].set_visible(False)

            for i, w in enumerate(weights):
                axes[0].text(
                    float(w) + 0.02, i,
                    f"{float(w) * 100:.1f}%",
                    va="center", fontsize=9.5, fontweight="bold",
                    color=colors[i],
                )
            axes[0].invert_yaxis()

            if len(self._history) > 2:
                for idx, name in enumerate(names):
                    raw = [float(h[idx]) for h in self._history]
                    smoothed = _ema_smooth(raw, alpha=0.35)
                    axes[1].plot(
                        smoothed,
                        color=_MODALITY_COLORS.get(name, "#999999"),
                        linewidth=1.8, label=name,
                    )
                    axes[1].fill_between(
                        range(len(smoothed)), smoothed,
                        alpha=0.08,
                        color=_MODALITY_COLORS.get(name, "#999999"),
                    )
                axes[1].set_ylim(0, 1)
                axes[1].set_title("Trend", fontsize=9, fontweight="bold")
                axes[1].set_xlabel("Window", fontsize=8)
                axes[1].grid(alpha=0.2, linestyle="--")
                axes[1].tick_params(labelsize=7)
                axes[1].spines["top"].set_visible(False)
                axes[1].spines["right"].set_visible(False)
            else:
                axes[1].text(
                    0.5, 0.5, "Accumulating\nhistory\u2026",
                    ha="center", va="center", fontsize=8,
                    color="#aaaaaa", style="italic",
                    transform=axes[1].transAxes,
                )
                axes[1].set_axis_off()

        fig.tight_layout(pad=1.2)
        self._prev_fig = fig
        return fig

    def reset(self) -> None:
        self._history = []
        if self._prev_fig is not None:
            plt.close(self._prev_fig)
            self._prev_fig = None
