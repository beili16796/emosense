# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Processing engine: file → segment → DE → inference → results stream."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generator

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Container for a single window's inference output."""

    valence: float = 0.0
    arousal: float = 0.0
    label: str = "neutral"
    confidence: float = 0.0
    proba: np.ndarray = field(default_factory=lambda: np.zeros(2))
    de_features: np.ndarray = field(default_factory=lambda: np.zeros((32, 5)))
    ch_names: list[str] = field(default_factory=list)
    attention_weights: np.ndarray | None = None
    model_name: str = ""
    latency_ms: float = 0.0
    trial_idx: int = 0
    window_idx: int = 0
    time_start_sec: float = 0.0


EMOTION_LABELS: dict[int, str] = {
    0: "negative/low",
    1: "positive/high",
}

EMOTION_LABELS_5: dict[int, str] = {
    0: "happy",
    1: "sad",
    2: "fear",
    3: "disgust",
    4: "neutral",
}


class ProcessingEngine:
    """Orchestrates the full pipeline from parsed file data to per-segment results.

    Pipeline: parsed_data → segment → DE extract → model inference → InferenceResult
    """

    def __init__(self, model_manager: Any) -> None:
        self.model_manager = model_manager
        self._de_extractors: dict[int, Any] = {}

    def process_file(
        self,
        parsed_data: dict[str, Any],
        window_sec: float = 4.0,
        overlap: float = 0.5,
        model_name: str | None = None,
    ) -> Generator[InferenceResult, None, None]:
        """Process uploaded file and yield one InferenceResult per segment.

        Args:
            parsed_data: Output of ``FileParser.parse()``.
            window_sec: Window length in seconds.
            overlap: Window overlap fraction [0, 1).
            model_name: Override active model name.

        Yields:
            InferenceResult for each processed window.
        """
        if model_name:
            self.model_manager.set_active_model(model_name)

        fs = parsed_data["fs"]
        eeg = parsed_data["eeg"]

        if parsed_data.get("pre_extracted", False):
            yield from self._process_preextracted(parsed_data)
            return

        if eeg.size == 0:
            logger.warning("Empty EEG data — nothing to process.")
            return

        win_samples = int(window_sec * fs)
        step_samples = int(win_samples * (1 - overlap))
        extractor = self._get_extractor(fs)
        n_trials = eeg.shape[0]

        for trial_idx in range(n_trials):
            trial_eeg = eeg[trial_idx]
            n_samples = trial_eeg.shape[-1]

            windows: list[np.ndarray] = []
            positions: list[float] = []
            start = 0
            while start + win_samples <= n_samples:
                windows.append(trial_eeg[:, start : start + win_samples])
                positions.append(start / fs)
                start += step_samples

            if not windows:
                continue

            X_win = np.stack(windows)
            X_de = extractor.transform(X_win)

            for w_idx, (de_feat, t_start) in enumerate(zip(X_de, positions)):
                t0 = time.perf_counter()
                result = self._run_inference(de_feat, parsed_data, trial_idx)
                result.latency_ms = (time.perf_counter() - t0) * 1000
                result.time_start_sec = t_start
                result.trial_idx = trial_idx
                result.window_idx = w_idx
                yield result

    def _process_preextracted(
        self, parsed_data: dict[str, Any]
    ) -> Generator[InferenceResult, None, None]:
        """Handle SEED-V pre-extracted DE features."""
        de_data = parsed_data.get("eeg_de")
        if de_data is None:
            return

        for i in range(len(de_data) if hasattr(de_data, '__len__') else 0):
            feat = np.asarray(de_data[i], dtype=np.float32)
            if feat.ndim == 2:
                feat = feat[np.newaxis, :, :]
            for j in range(feat.shape[0] if feat.ndim == 3 else 1):
                de_window = feat[j] if feat.ndim == 3 else feat
                t0 = time.perf_counter()
                result = self._run_inference(de_window, parsed_data, i)
                result.latency_ms = (time.perf_counter() - t0) * 1000
                result.trial_idx = i
                result.window_idx = j
                yield result

    def _run_inference(
        self, de_feat: np.ndarray, parsed_data: dict[str, Any], trial_idx: int
    ) -> InferenceResult:
        """Run model inference on a single DE feature window."""
        model = self.model_manager.get_active_model()
        model_name = self.model_manager.get_active_model_name()

        X = torch.FloatTensor(de_feat).unsqueeze(0)
        with torch.no_grad():
            if hasattr(model, "forward"):
                logits = model.forward(X) if hasattr(model, 'network') else model(X)
                if hasattr(model, 'network'):
                    logits = model.network(X)
            else:
                logits = torch.zeros(1, 2)

            proba = F.softmax(logits, dim=-1).squeeze(0).numpy()

        pred_class = int(proba.argmax())
        n_classes = proba.shape[0]
        labels = EMOTION_LABELS if n_classes <= 2 else EMOTION_LABELS_5
        label = labels.get(pred_class, f"class_{pred_class}")

        valence = float(proba[1] - proba[0]) if n_classes == 2 else 0.0
        arousal = float(proba.max() * 2 - 1)

        attention = None
        if hasattr(model, "get_attention_weights"):
            try:
                attention = model.get_attention_weights()
            except Exception:
                pass

        return InferenceResult(
            valence=valence,
            arousal=arousal,
            label=label,
            confidence=float(proba.max()),
            proba=proba,
            de_features=de_feat,
            ch_names=parsed_data.get("ch_names", []),
            attention_weights=attention,
            model_name=model_name,
        )

    def _get_extractor(self, fs: int) -> Any:
        """Get or create a DEExtractor for the given sampling rate."""
        if fs not in self._de_extractors:
            from emokit.features.eeg import DEExtractor
            self._de_extractors[fs] = DEExtractor(fs=fs)
        return self._de_extractors[fs]
