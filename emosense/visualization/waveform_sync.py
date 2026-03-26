# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Real-time multi-modal waveform panel — clinical monitor style.

Renders a scrolling strip-chart of representative EEG, ECG, and GSR
channels, synchronized with the inference window timestamps.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

_SIGNAL_COLORS = {
    "EEG": "#4C72B0",
    "ECG": "#C0392B",
    "GSR": "#DD8452",
}

_MAX_DISPLAY_SAMPLES = 512


class WaveformSyncPlot:
    """Scrolling multimodal waveform display.

    Maintains a fixed-length ring buffer for each signal channel and
    renders a clinical-monitor-style strip chart.
    """

    def __init__(self, display_seconds: float = 10.0, fs: int = 128) -> None:
        self._fs = fs
        self._display_seconds = display_seconds
        self._display_n = min(int(display_seconds * fs), _MAX_DISPLAY_SAMPLES)
        self._buffers: dict[str, np.ndarray] = {}
        self._cursor = 0
        self._prev_fig: Figure | None = None
        self._nan_mask: dict[str, np.ndarray] = {}

    def push(
        self,
        eeg_snippet: np.ndarray | None = None,
        ecg_snippet: np.ndarray | None = None,
        gsr_snippet: np.ndarray | None = None,
        time_start: float = 0.0,
    ) -> None:
        """Append new samples (downsampled to display resolution)."""
        for name, raw in [("EEG", eeg_snippet), ("ECG", ecg_snippet), ("GSR", gsr_snippet)]:
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=np.float64).ravel()
            if len(arr) > _MAX_DISPLAY_SAMPLES:
                factor = max(1, len(arr) // _MAX_DISPLAY_SAMPLES)
                arr = arr[::factor]

            if name not in self._buffers:
                self._buffers[name] = np.full(self._display_n, np.nan, dtype=np.float64)
                self._nan_mask[name] = np.zeros(self._display_n, dtype=bool)

            buf = self._buffers[name]
            n_new = min(len(arr), self._display_n)
            buf[:-n_new] = buf[n_new:]
            buf[-n_new:] = arr[-n_new:]

            nm = self._nan_mask[name]
            nm[:-n_new] = nm[n_new:]
            nm[-n_new:] = np.isnan(arr[-n_new:])

    def render(self, current_time: float = 0.0) -> Figure:
        """Draw the strip chart."""
        if self._prev_fig is not None:
            plt.close(self._prev_fig)

        n_signals = max(len(self._buffers), 1)
        fig, axes = plt.subplots(
            n_signals, 1,
            figsize=(8, 0.9 * n_signals + 0.4),
            dpi=100, sharex=True,
        )
        fig.patch.set_facecolor("#1a1a2e")
        if n_signals == 1:
            axes = [axes]

        if not self._buffers:
            axes[0].set_facecolor("#1a1a2e")
            axes[0].text(
                0.5, 0.5, "Awaiting signal data\u2026",
                ha="center", va="center", fontsize=11,
                color="#555555", style="italic",
                transform=axes[0].transAxes,
            )
            axes[0].set_axis_off()
            fig.tight_layout(pad=0.3)
            self._prev_fig = fig
            return fig

        t_axis = np.linspace(
            max(0, current_time - self._display_seconds),
            current_time,
            self._display_n,
        )

        for ax_idx, (name, buf) in enumerate(self._buffers.items()):
            ax = axes[ax_idx]
            ax.set_facecolor("#1a1a2e")
            color = _SIGNAL_COLORS.get(name, "#888888")

            nan_regions = self._nan_mask.get(name, np.zeros_like(buf, dtype=bool))
            valid = ~np.isnan(buf) & ~nan_regions

            ax.plot(t_axis[valid], buf[valid], color=color, linewidth=0.8, alpha=0.9)

            if nan_regions.any():
                starts = np.where(np.diff(nan_regions.astype(int)) == 1)[0]
                ends = np.where(np.diff(nan_regions.astype(int)) == -1)[0]
                if nan_regions[0]:
                    starts = np.concatenate([[0], starts])
                if nan_regions[-1]:
                    ends = np.concatenate([ends, [len(nan_regions) - 1]])
                for s, e in zip(starts, ends):
                    s_idx = max(0, min(s, len(t_axis) - 1))
                    e_idx = max(0, min(e, len(t_axis) - 1))
                    ax.axvspan(
                        t_axis[s_idx], t_axis[e_idx],
                        color="#ff4444", alpha=0.25,
                    )

            ax.set_ylabel(name, fontsize=8, color=color, fontweight="bold", rotation=0, labelpad=30)
            ax.tick_params(colors="#666666", labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#333333")
            ax.spines["left"].set_color("#333333")
            ax.grid(axis="x", color="#333333", linewidth=0.3, alpha=0.5)

        axes[-1].set_xlabel("Time (s)", fontsize=8, color="#888888")
        fig.tight_layout(pad=0.3)
        self._prev_fig = fig
        return fig

    def reset(self) -> None:
        self._buffers.clear()
        self._nan_mask.clear()
        self._cursor = 0
        if self._prev_fig is not None:
            plt.close(self._prev_fig)
            self._prev_fig = None

    def placeholder(self) -> Figure:
        """Empty dark placeholder."""
        if self._prev_fig is not None:
            plt.close(self._prev_fig)
        fig, ax = plt.subplots(figsize=(8, 1.2), dpi=100)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        ax.text(
            0.5, 0.5, "Multimodal Signal Monitor — awaiting data\u2026",
            ha="center", va="center", fontsize=11,
            color="#555555", style="italic",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout(pad=0.3)
        self._prev_fig = fig
        return fig
