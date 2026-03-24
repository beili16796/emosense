# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Topographic EEG map using MNE for differential-entropy visualisation."""

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


class TopoMapPlot:
    """Topographic heatmap of per-channel differential entropy.

    Supports both 32-channel (DEAP) and 62-channel (SEED/SEED-V) layouts.

    Args:
        ch_names: EEG channel names matching the chosen montage.
        fs: Sampling frequency in Hz.
        montage_name: Standard montage for electrode positions.
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
        """Set the default frequency band.

        Args:
            band: One of delta / theta / alpha / beta / gamma.

        Raises:
            ValueError: If *band* is not recognised.
        """
        if band not in _BAND_NAMES:
            raise ValueError(
                f"Unknown band {band!r}. Choose from {_BAND_NAMES}",
            )
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
        """Render a topographic map for the selected frequency band.

        Args:
            de_features: DE values of shape ``(n_channels, 5)``.
            band: Override current band selection.

        Returns:
            Matplotlib Figure with the topomap.
        """
        band = band or self._current_band
        assert de_features.shape[0] == self.n_channels, (
            f"Expected {self.n_channels} channels, got {de_features.shape[0]}"
        )

        band_idx = _BAND_NAMES.index(band)
        values = de_features[:, band_idx]

        if self._prev_fig is not None:
            plt.close(self._prev_fig)

        if not _MNE_AVAILABLE or self._info is None:
            return self._fallback_heatmap(de_features, band)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
        try:
            im, _cn = mne.viz.plot_topomap(
                values, self._info, axes=ax, cmap="RdBu_r", show=False,
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"DE ({band})")
        except Exception as exc:
            logger.warning("MNE plot_topomap failed: %s — using fallback", exc)
            plt.close(fig)
            return self._fallback_heatmap(de_features, band)

        ax.set_title(f"DE — {band} ({self.n_channels}ch)")
        if annotate_asymmetry and band == "alpha":
            self._annotate_asymmetry(ax, de_features)
        fig.tight_layout()
        self._prev_fig = fig
        return fig

    def _fallback_heatmap(self, de_features: np.ndarray, band: str | None = None) -> Figure:
        """Simple bar chart fallback when MNE is unavailable."""
        band = band or self._current_band
        band_idx = BAND_IDX[band]
        values = de_features[:, band_idx]

        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.bar(range(len(values)), values, color="steelblue", alpha=0.7)
        ax.set_xlabel("Channel index")
        ax.set_ylabel(f"DE ({band})")
        ax.set_title(f"{band} band DE (bar chart fallback)")
        if band == "alpha":
            self._annotate_asymmetry(ax, de_features)
        fig.tight_layout()
        self._prev_fig = fig
        return fig

    def placeholder(self) -> Figure:
        """Return an empty placeholder figure."""
        if self._prev_fig is not None:
            plt.close(self._prev_fig)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
        ax.text(
            0.5, 0.5, "Awaiting data\u2026",
            ha="center", va="center", fontsize=12, color="#888888",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        self._prev_fig = fig
        return fig

    def get_band_list(self) -> list[str]:
        """Return the list of available frequency band names."""
        return list(_BAND_NAMES)

    def _compute_frontal_asymmetry(self, de: np.ndarray) -> float | None:
        """Davidson-style frontal alpha asymmetry: F4 - F3."""
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
        color = "#27AE60" if asym > 0 else "#2E86C1"
        ax.text(
            0.02,
            0.02,
            f"Frontal alpha asymmetry: {direction}\n(Delta DE = {asym:+.3f})",
            transform=ax.transAxes,
            fontsize=7.5,
            color=color,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "lightyellow",
                "alpha": 0.8,
            },
        )
