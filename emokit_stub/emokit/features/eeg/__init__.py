"""EmoKit EEG feature extraction."""

from __future__ import annotations

import numpy as np


class DEExtractor:
    """Differential Entropy feature extractor for EEG signals.

    Computes DE across 5 standard frequency bands: delta, theta, alpha, beta, gamma.
    """

    BANDS = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 14.0),
        "beta": (14.0, 31.0),
        "gamma": (31.0, 50.0),
    }

    def __init__(self, fs: int = 128) -> None:
        self._fs = fs

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract DE features.

        Args:
            X: EEG data of shape ``(batch, n_channels, n_samples)``.

        Returns:
            DE features of shape ``(batch, n_channels, 5)``.
        """
        if X.ndim == 2:
            X = X[np.newaxis, :, :]

        batch, n_ch, n_samples = X.shape
        n_bands = 5
        de = np.zeros((batch, n_ch, n_bands), dtype=np.float64)

        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self._fs)
        fft_data = np.fft.rfft(X, axis=-1)
        psd = np.abs(fft_data) ** 2 / n_samples

        for b_idx, (band_name, (lo, hi)) in enumerate(self.BANDS.items()):
            mask = (freqs >= lo) & (freqs < hi)
            if not mask.any():
                continue
            band_power = psd[:, :, mask].mean(axis=-1)
            de[:, :, b_idx] = np.log(band_power + 1e-10)

        return de
