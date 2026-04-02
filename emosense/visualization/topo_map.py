# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Topographic EEG map using MNE for differential-entropy visualisation.

Publication-quality rendering with perceptually uniform colormaps,
contour lines, and prominent frontal alpha asymmetry annotations.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

_BAND_NAMES: list[str] = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_IDX: dict[str, int] = {name: i for i, name in enumerate(_BAND_NAMES)}

from emosense.backend.file_parser import DEAP_EEG_CHANNELS  # noqa: E402

try:
    import mne
    _MNE_AVAILABLE = True
except ImportError:
    _MNE_AVAILABLE = False
    logger.warning("MNE not installed — topomap will use fallback bar chart")

_BAND_CMAP: dict[str, str] = {
    "delta": "Blues",
    "theta": "Purples",
    "alpha": "RdBu_r",
    "beta": "YlOrRd",
    "gamma": "inferno",
}


class TopoMapPlot:
    """Topographic heatmap of per-channel differential entropy.

    Supports 14-channel (DREAMER), 32-channel (DEAP), and 62-channel
    (SEED/SEED-V) layouts with publication-quality rendering.
    """

    BAND_IDX = BAND_IDX

    def __init__(
        self,
        ch_names: list[str] | None = None,
        fs: int = 128,
        montage_name: str = "standard_1020",
        *,
        channel_names: list[str] | None = None,
    ) -> None:
        names = ch_names or channel_names or list(DEAP_EEG_CHANNELS)
        self._ch_names = list(names)
        self._fs = fs
        self._montage_name = montage_name
        self._current_band = "alpha"
        self._prev_fig: Figure | None = None
        self.n_channels = len(self._ch_names)
        self._info = None

        if _MNE_AVAILABLE:
            try:
                self._info = mne.create_info(
                    ch_names=self._ch_names,
                    sfreq=float(fs),
                    ch_types="eeg",
                )
                montage = mne.channels.make_standard_montage(montage_name)
                self._info.set_montage(montage, match_case=False, on_missing="warn")
            except Exception as exc:
                logger.warning("MNE montage setup failed: %s — using fallback", exc)
                self._info = None

        logger.info(
            "TopoMapPlot initialised with %d channels, fs=%d, montage=%s, mne=%s",
            len(self._ch_names), fs, montage_name, _MNE_AVAILABLE and self._info is not None,
        )

    def set_band(self, band: str) -> None:
        if band not in _BAND_NAMES:
            raise ValueError(f"Unknown band {band!r}. Choose from {_BAND_NAMES}")
        self._current_band = band

    @property
    def ch_names(self) -> list[str]:
        return list(self._ch_names)

    def update(
        self,
        de_features: np.ndarray,
        band: str | None = None,
        annotate_asymmetry: bool = True,
    ) -> Figure:
        """Render a publication-quality topographic map."""
        band = band or self._current_band
        assert de_features.shape[0] == self.n_channels, (
            f"Expected {self.n_channels} channels, got {de_features.shape[0]}"
        )

        band_idx = _BAND_NAMES.index(band)
        values = de_features[:, band_idx]

        if self._prev_fig is not None:
            plt.close(self._prev_fig)

        nan_ratio = np.isnan(values).mean()
        if nan_ratio > 0.5:
            return self._sensor_disconnected(band, nan_ratio)

        if not _MNE_AVAILABLE or self._info is None:
            return self._fallback_heatmap(de_features, band)

        cmap = _BAND_CMAP.get(band, "RdBu_r")

        fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.2), dpi=120)
        fig.patch.set_facecolor("white")
        try:
            im, _cn = mne.viz.plot_topomap(
                values,
                self._info,
                axes=ax,
                cmap=cmap,
                contours=6,
                show=False,
            )
            cbar = fig.colorbar(
                im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85,
            )
            cbar.set_label(f"DE ({band})", fontsize=9, fontweight="bold")
            cbar.ax.tick_params(labelsize=8)
        except Exception as exc:
            logger.warning("MNE plot_topomap failed: %s — using fallback", exc)
            plt.close(fig)
            return self._fallback_heatmap(de_features, band)

        ax.set_title(
            f"DE — {band} ({self.n_channels}ch)",
            fontsize=11, fontweight="bold", pad=8,
        )
        if annotate_asymmetry and band == "alpha":
            self._annotate_asymmetry(ax, de_features)
        fig.tight_layout(pad=1.0)
        self._prev_fig = fig
        return fig

    def _fallback_heatmap(self, de_features: np.ndarray, band: str | None = None) -> Figure:
        """Styled bar chart fallback when MNE is unavailable."""
        band = band or self._current_band
        band_idx = BAND_IDX[band]
        values = de_features[:, band_idx]
        cmap_name = _BAND_CMAP.get(band, "RdBu_r")
        cmap = plt.get_cmap(cmap_name)

        v_min, v_max = float(values.min()), float(values.max())
        v_range = v_max - v_min if v_max > v_min else 1.0
        normed = (values - v_min) / v_range
        colors = [cmap(float(v)) for v in normed]

        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=120)
        fig.patch.set_facecolor("white")
        ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Channel", fontsize=9)
        ax.set_ylabel(f"DE ({band})", fontsize=9)
        ax.set_title(f"DE — {band} ({self.n_channels}ch)", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if band == "alpha":
            self._annotate_asymmetry(ax, de_features)
        fig.tight_layout()
        self._prev_fig = fig
        return fig

    def placeholder(self) -> Figure:
        if self._prev_fig is not None:
            plt.close(self._prev_fig)
        fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.2), dpi=120)
        fig.patch.set_facecolor("white")
        ax.text(
            0.5, 0.5, "Awaiting data\u2026",
            ha="center", va="center", fontsize=13, color="#aaaaaa",
            transform=ax.transAxes, style="italic",
        )
        ax.set_axis_off()
        fig.tight_layout()
        self._prev_fig = fig
        return fig

    def _sensor_disconnected(self, band: str, nan_ratio: float) -> Figure:
        """Warning figure when too many channels report NaN."""
        fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.2), dpi=120)
        fig.patch.set_facecolor("#fff3cd")
        ax.set_facecolor("#fff3cd")
        ax.text(
            0.5, 0.55,
            "\u26a0  Sensor Disconnected",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color="#856404",
            transform=ax.transAxes,
        )
        ax.text(
            0.5, 0.38,
            f"{nan_ratio * 100:.0f}% of channels report NaN\n"
            f"Band: {band} | Check electrode contact",
            ha="center", va="center", fontsize=9,
            color="#856404", transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        self._prev_fig = fig
        return fig

    def get_band_list(self) -> list[str]:
        return list(_BAND_NAMES)

    def _compute_frontal_asymmetry(self, de: np.ndarray) -> float | None:
        alpha_idx = BAND_IDX["alpha"]
        f3 = next((i for i, c in enumerate(self._ch_names) if c.upper() in ("F3", "F 3")), None)
        f4 = next((i for i, c in enumerate(self._ch_names) if c.upper() in ("F4", "F 4")), None)
        if f3 is None or f4 is None:
            return None
        return float(de[f4, alpha_idx] - de[f3, alpha_idx])

    def _annotate_asymmetry(self, ax: plt.Axes, de_features: np.ndarray) -> None:
        asym = self._compute_frontal_asymmetry(de_features)
        if asym is None:
            return
        direction = "R > L" if asym > 0 else "L > R"
        valence_hint = "(approach ↑)" if asym > 0 else "(withdrawal ↑)"
        bg_color = "#d4edda" if asym > 0 else "#cce5ff"
        border_color = "#28a745" if asym > 0 else "#007bff"
        ax.text(
            0.02, 0.02,
            f"Frontal \u03b1 asymmetry: {direction}\n"
            f"\u0394DE = {asym:+.3f}  {valence_hint}",
            transform=ax.transAxes,
            fontsize=8, fontweight="bold",
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": bg_color,
                "edgecolor": border_color,
                "alpha": 0.92,
                "linewidth": 1.5,
            },
        )
