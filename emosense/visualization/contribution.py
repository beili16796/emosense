# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Modality contribution: bar chart + radar plot with smoothed history."""

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
    if not series:
        return []
    out = [series[0]]
    for v in series[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


class ContributionPlot:
    """Bar chart + radar of modality attention weights with EMA-smoothed history."""

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

        if weights is None:
            return self._unimodal_placeholder(model_name)

        names = self._modality_names[:len(weights)]
        colors = [_MODALITY_COLORS.get(n, "#999999") for n in names]

        fig = plt.figure(figsize=(7, 3), dpi=110)
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(1, 3, width_ratios=[2, 1.2, 1])

        ax_bar = fig.add_subplot(gs[0])
        ax_radar = fig.add_subplot(gs[1], polar=True)
        ax_trend = fig.add_subplot(gs[2])

        # ── bar chart ──
        y_pos = list(range(len(names)))
        ax_bar.barh(y_pos, weights, color=colors, edgecolor="white", linewidth=0.8, height=0.65)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(names, fontsize=10, fontweight="bold")
        ax_bar.set_xlim(0, 1.08)
        ax_bar.set_xlabel("Weight", fontsize=8)
        ax_bar.set_title(f"Attention ({model_name})", fontsize=9, fontweight="bold")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        for i, w in enumerate(weights):
            ax_bar.text(float(w) + 0.02, i, f"{float(w)*100:.0f}%", va="center", fontsize=9, fontweight="bold", color=colors[i])
        ax_bar.invert_yaxis()

        # ── radar chart ──
        n = len(names)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        vals = [float(w) for w in weights]
        angles.append(angles[0])
        vals.append(vals[0])
        ax_radar.fill(angles, vals, alpha=0.2, color="#4C72B0")
        ax_radar.plot(angles, vals, linewidth=2, color="#4C72B0")
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(names, fontsize=7.5, fontweight="bold")
        ax_radar.set_ylim(0, 1)
        ax_radar.set_yticks([0.25, 0.5, 0.75])
        ax_radar.set_yticklabels(["", "0.5", ""], fontsize=6, color="#aaa")
        ax_radar.set_title("Radar", fontsize=8, fontweight="bold", pad=12)
        ax_radar.grid(alpha=0.3)

        # ── trend chart ──
        if len(self._history) > 2:
            for idx, name in enumerate(names):
                raw = [float(h[idx]) for h in self._history]
                smoothed = _ema_smooth(raw, alpha=0.35)
                ax_trend.plot(smoothed, color=_MODALITY_COLORS.get(name, "#999"), linewidth=1.5)
                ax_trend.fill_between(range(len(smoothed)), smoothed, alpha=0.06, color=_MODALITY_COLORS.get(name, "#999"))
            ax_trend.set_ylim(0, 1)
            ax_trend.set_title("Trend", fontsize=8, fontweight="bold")
            ax_trend.set_xlabel("Win", fontsize=7)
            ax_trend.grid(alpha=0.2, linestyle="--")
            ax_trend.tick_params(labelsize=6)
            ax_trend.spines["top"].set_visible(False)
            ax_trend.spines["right"].set_visible(False)
        else:
            ax_trend.text(0.5, 0.5, "Accumulating\u2026", ha="center", va="center", fontsize=7, color="#aaa", style="italic", transform=ax_trend.transAxes)
            ax_trend.set_axis_off()

        fig.tight_layout(pad=0.8)
        self._prev_fig = fig
        return fig

    def _unimodal_placeholder(self, model_name: str) -> Figure:
        fig, ax = plt.subplots(figsize=(7, 3), dpi=110)
        fig.patch.set_facecolor("white")
        ax.text(
            0.5, 0.5,
            f"{model_name} (EEG only)\n"
            "\u2500" * 28 + "\n"
            "Unimodal model \u2014 attention\n"
            "breakdown not available.\n\n"
            "Switch to DGCCA-AM for\n"
            "multimodal radar + bar chart.",
            ha="center", va="center", fontsize=9.5, color="#777",
            transform=ax.transAxes, family="monospace", linespacing=1.4,
        )
        ax.set_axis_off()
        fig.tight_layout()
        self._prev_fig = fig
        return fig

    def reset(self) -> None:
        self._history = []
        if self._prev_fig is not None:
            plt.close(self._prev_fig)
            self._prev_fig = None
