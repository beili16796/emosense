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
import mne  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

_BAND_NAMES: list[str] = ["delta", "theta", "alpha", "beta", "gamma"]


class TopoMapPlot:
    """Topographic heatmap of per-channel differential entropy.

    Args:
        ch_names: EEG channel names matching the chosen montage.
        fs: Sampling frequency in Hz.
        montage_name: Standard montage for electrode positions.
    """

    def __init__(
        self,
        ch_names: list[str],
        fs: int = 128,
        montage_name: str = "standard_1020",
    ) -> None:
        self._ch_names = list(ch_names)
        self._fs = fs
        self._montage_name = montage_name
        self._current_band = "alpha"
        self._prev_fig: Figure | None = None

        self._info = mne.create_info(
            ch_names=self._ch_names,
            sfreq=float(fs),
            ch_types="eeg",
        )
        montage = mne.channels.make_standard_montage(montage_name)
        self._info.set_montage(montage, match_case=False, on_missing="ignore")
        logger.info(
            "TopoMapPlot initialised with %d channels, fs=%d, montage=%s",
            len(ch_names),
            fs,
            montage_name,
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

    def update(self, de_features: np.ndarray) -> Figure:
        """Render a topographic map for the currently selected frequency band.

        Args:
            de_features: DE values of shape ``(n_channels, 5)``.

        Returns:
            Matplotlib Figure with the topomap.
        """
        band = self._current_band
        band_idx = _BAND_NAMES.index(band)
        values = de_features[:, band_idx]

        if self._prev_fig is not None:
            plt.close(self._prev_fig)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
        im, _cn = mne.viz.plot_topomap(
            values, self._info, axes=ax, cmap="RdBu_r", show=False,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"DE ({band})")
        ax.set_title(f"DE — {band}")

        fig.tight_layout()
        self._prev_fig = fig
        return fig

    def placeholder(self) -> Figure:
        """Return an empty placeholder figure."""
        if self._prev_fig is not None:
            plt.close(self._prev_fig)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
        ax.text(
            0.5, 0.5, "Awaiting data…",
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
