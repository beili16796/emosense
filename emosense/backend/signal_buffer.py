# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Thread-safe sliding-window signal buffers for real-time streaming."""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


class SlidingWindowBuffer:
    """Ring buffer that yields fixed-length windows with configurable step.

    Args:
        n_channels: Number of signal channels.
        fs: Sampling frequency in Hz.
        window_sec: Window length in seconds.
        step_sec: Step (hop) between consecutive windows in seconds.
    """

    def __init__(
        self,
        n_channels: int,
        fs: int,
        window_sec: float = 4.0,
        step_sec: float = 2.0,
    ) -> None:
        self._n_channels = n_channels
        self._fs = fs
        self._window_samples = int(fs * window_sec)
        self._step_samples = int(fs * step_sec)

        self._buf: np.ndarray = np.zeros(
            (n_channels, self._window_samples), dtype=np.float64,
        )
        self._write_pos: int = 0
        self._total_written: int = 0
        self._last_window_pos: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, chunk: np.ndarray) -> None:
        """Append a chunk of shape ``(n_channels, n_new_samples)``."""
        if chunk.ndim != 2 or chunk.shape[0] != self._n_channels:
            raise ValueError(
                f"Expected shape ({self._n_channels}, n_samples), "
                f"got {chunk.shape}",
            )

        n_new = chunk.shape[1]
        with self._lock:
            space = self._window_samples - self._write_pos
            if n_new <= space:
                self._buf[:, self._write_pos : self._write_pos + n_new] = chunk
                self._write_pos += n_new
            else:
                self._buf[:, self._write_pos : self._write_pos + space] = (
                    chunk[:, :space]
                )
                remaining = n_new - space
                if remaining >= self._window_samples:
                    self._buf[:] = chunk[:, n_new - self._window_samples :]
                    self._write_pos = self._window_samples
                else:
                    self._buf[:, :remaining] = chunk[:, space:]
                    self._write_pos = remaining
                    # Shift buffer so oldest data is at front when wrapping
                    self._buf = np.roll(self._buf, -self._write_pos, axis=1)
                    self._write_pos = self._window_samples

            self._total_written += n_new

    def get_window(self) -> np.ndarray | None:
        """Return the current window or ``None`` if not enough data.

        Returns:
            Array of shape ``(n_channels, n_window_samples)`` or ``None``.
        """
        with self._lock:
            if self._write_pos < self._window_samples:
                return None
            self._last_window_pos = self._total_written
            return self._buf[:, : self._window_samples].copy()

    def has_new_window(self) -> bool:
        """Return ``True`` if a new complete window is available."""
        with self._lock:
            if self._write_pos < self._window_samples:
                return False
            return (
                self._total_written - self._last_window_pos
                >= self._step_samples
                or self._last_window_pos == 0
            )

    def reset(self) -> None:
        """Clear all buffered data."""
        with self._lock:
            self._buf[:] = 0.0
            self._write_pos = 0
            self._total_written = 0
            self._last_window_pos = 0

    @property
    def fill_fraction(self) -> float:
        """Fraction of the window that is filled, in ``[0.0, 1.0]``."""
        with self._lock:
            return min(self._write_pos / self._window_samples, 1.0)


class ModalityBuffer:
    """Manage one :class:`SlidingWindowBuffer` per signal modality.

    Args:
        config: Mapping of modality name to a dict with keys
            ``n_channels``, ``fs``, ``window_sec``, ``step_sec``.
    """

    def __init__(self, config: dict[str, dict[str, int | float]]) -> None:
        self._buffers: dict[str, SlidingWindowBuffer] = {}
        for name, params in config.items():
            self._buffers[name] = SlidingWindowBuffer(
                n_channels=int(params["n_channels"]),
                fs=int(params["fs"]),
                window_sec=float(params.get("window_sec", 4.0)),
                step_sec=float(params.get("step_sec", 2.0)),
            )
        logger.info("ModalityBuffer initialised for: %s", list(self._buffers))

    def push_modality(self, name: str, chunk: np.ndarray) -> None:
        """Push a chunk into the buffer for *name*."""
        if name not in self._buffers:
            raise KeyError(f"Unknown modality: {name!r}")
        self._buffers[name].push(chunk)

    def get_all_windows(self) -> dict[str, np.ndarray] | None:
        """Return windows for every modality, or ``None`` if any is missing.

        Returns:
            Dict mapping modality name to its window array, or ``None``.
        """
        if not all(b.has_new_window() for b in self._buffers.values()):
            return None

        windows: dict[str, np.ndarray] = {}
        for name, buf in self._buffers.items():
            win = buf.get_window()
            if win is None:
                return None
            windows[name] = win
        return windows

    @property
    def fill_fraction(self) -> float:
        """Minimum fill fraction across all modalities."""
        if not self._buffers:
            return 0.0
        return min(b.fill_fraction for b in self._buffers.values())

    def reset(self) -> None:
        """Reset every internal buffer."""
        for buf in self._buffers.values():
            buf.reset()
        logger.debug("ModalityBuffer reset")
