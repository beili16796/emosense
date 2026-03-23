# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Stream receivers for live and simulated signal acquisition."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import numpy as np

try:
    import pylsl  # type: ignore[import-untyped]

    _HAS_PYLSL = True
except ImportError:
    _HAS_PYLSL = False

logger = logging.getLogger(__name__)


class BaseReceiver(ABC):
    """Abstract interface for signal stream receivers."""

    @abstractmethod
    def start(self) -> None:
        """Begin acquiring data."""

    @abstractmethod
    def stop(self) -> None:
        """Stop acquiring data and release resources."""

    @abstractmethod
    def get_latest_chunk(self) -> dict[str, np.ndarray] | None:
        """Return the latest available chunk per modality.

        Returns:
            Dict mapping modality name to array of shape
            ``(n_channels, n_new_samples)``, or ``None`` if no data.
        """


class LSLReceiver(BaseReceiver):
    """Receive multi-modal data via Lab Streaming Layer (LSL).

    Connects to ``EEG`` and ``Physiological`` stream types.  Runs inlet
    pulls in a background thread and stores chunks in bounded deques.

    Raises:
        ImportError: If *pylsl* is not installed.
    """

    _STREAM_TYPES: list[tuple[str, str]] = [
        ("eeg", "EEG"),
        ("peripheral", "Physiological"),
    ]
    _RECONNECT_INTERVAL: float = 2.0

    def __init__(self, buffer_seconds: float = 10.0) -> None:
        if not _HAS_PYLSL:
            raise ImportError(
                "pylsl is required for LSLReceiver. "
                "Install it with: pip install pylsl",
            )
        self._buffer_seconds = buffer_seconds
        self._inlets: dict[str, Any] = {}
        self._queues: dict[str, deque[np.ndarray]] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Resolve LSL streams and begin pulling in background."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="LSLReceiver",
        )
        self._thread.start()
        logger.info("LSLReceiver started")

    def stop(self) -> None:
        """Stop background thread and close inlets."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            self._inlets.clear()
            self._queues.clear()
        logger.info("LSLReceiver stopped")

    def get_latest_chunk(self) -> dict[str, np.ndarray] | None:
        """Drain buffered chunks and return concatenated per-modality data."""
        result: dict[str, np.ndarray] = {}
        with self._lock:
            for name, q in self._queues.items():
                if not q:
                    continue
                chunks = list(q)
                q.clear()
                result[name] = np.concatenate(chunks, axis=1)
        return result if result else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_and_connect(self, modality: str, stream_type: str) -> bool:
        """Try to resolve a single stream type and open an inlet."""
        streams = pylsl.resolve_byprop("type", stream_type, timeout=1.0)
        if not streams:
            return False
        inlet = pylsl.StreamInlet(
            streams[0],
            max_buflen=int(self._buffer_seconds),
        )
        inlet.open_stream(timeout=2.0)
        with self._lock:
            self._inlets[modality] = inlet
            self._queues.setdefault(modality, deque(maxlen=512))
        logger.info("Connected to LSL stream: %s (%s)", modality, stream_type)
        return True

    def _pull_chunk(self, modality: str, inlet: Any) -> None:
        """Pull one chunk from *inlet* and enqueue it."""
        try:
            samples, _timestamps = inlet.pull_chunk(timeout=0.0)
            if samples:
                arr = np.array(samples, dtype=np.float64).T
                with self._lock:
                    self._queues[modality].append(arr)
        except Exception:
            logger.warning(
                "Lost LSL stream for %s, will reconnect", modality,
            )
            with self._lock:
                self._inlets.pop(modality, None)

    def _run(self) -> None:
        """Background loop: resolve missing streams, pull data."""
        while self._running:
            for modality, stream_type in self._STREAM_TYPES:
                if modality not in self._inlets:
                    self._resolve_and_connect(modality, stream_type)

            with self._lock:
                active = list(self._inlets.items())

            for modality, inlet in active:
                self._pull_chunk(modality, inlet)

            time.sleep(0.005)


class SimulatedReceiver(BaseReceiver):
    """Replay pre-recorded trial data chunk by chunk at real time.

    Args:
        trial_data: Dict mapping modality to array ``(n_channels, n_samples)``.
        fs: Sampling frequency in Hz.
        chunk_size: Samples per emitted chunk.
        speed_factor: Playback speed multiplier (>1 = faster).
    """

    def __init__(
        self,
        trial_data: dict[str, np.ndarray],
        fs: int,
        chunk_size: int = 32,
        speed_factor: float = 1.0,
    ) -> None:
        self._trial_data = trial_data
        self._fs = fs
        self._chunk_size = chunk_size
        self._speed_factor = max(speed_factor, 0.01)
        self._sleep_sec = chunk_size / fs / self._speed_factor

        self._pos: int = 0
        self._queue: deque[dict[str, np.ndarray]] = deque(maxlen=256)
        self._running = False
        self._finished = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_finished(self) -> bool:
        """``True`` when all trial data has been emitted."""
        return self._finished

    def start(self) -> None:
        """Begin replaying in a background thread."""
        if self._running:
            return
        self._running = True
        self._finished = False
        self._pos = 0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="SimulatedReceiver",
        )
        self._thread.start()
        logger.info("SimulatedReceiver started (speed=%.1fx)", self._speed_factor)

    def stop(self) -> None:
        """Stop replay thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("SimulatedReceiver stopped")

    def get_latest_chunk(self) -> dict[str, np.ndarray] | None:
        """Return the next buffered chunk dict, or ``None``."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background loop emitting chunks at simulated real-time pace."""
        max_samples = max(arr.shape[1] for arr in self._trial_data.values())

        while self._running and self._pos < max_samples:
            chunk_dict: dict[str, np.ndarray] = {}
            end = self._pos + self._chunk_size
            for name, arr in self._trial_data.items():
                n_total = arr.shape[1]
                actual_end = min(end, n_total)
                if self._pos < n_total:
                    chunk_dict[name] = arr[:, self._pos : actual_end]

            if chunk_dict:
                with self._lock:
                    self._queue.append(chunk_dict)

            self._pos = end
            time.sleep(self._sleep_sec)

        self._finished = True
        self._running = False
        logger.info("SimulatedReceiver finished replay")
