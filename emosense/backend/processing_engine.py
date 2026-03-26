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
    2: "neutral",
    3: "fear",
    4: "disgust",
}

SEED5_VA: dict[int, tuple[float, float]] = {
    0: (0.8, 0.6),     # happy
    1: (-0.7, -0.3),   # sad
    2: (0.0, 0.0),     # neutral
    3: (-0.5, 0.7),    # fear
    4: (-0.6, -0.1),   # disgust
}


class FeatureCache:
    """Stores pre-computed DE features for the current session.

    Avoids re-running DEExtractor when user switches models.
    Cache key: (file_hash, window_sec, overlap)
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}

    def key(self, file_hash: str, window_sec: float, overlap: float) -> str:
        return f"{file_hash}_{window_sec}_{overlap}"

    def get(self, k: str) -> dict | None:
        return self._cache.get(k)

    def set(self, k: str, features: dict) -> None:
        if len(self._cache) >= 3:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[k] = features

    def has(self, k: str) -> bool:
        return k in self._cache

    def clear(self) -> None:
        self._cache.clear()


class ProcessingEngine:
    """Orchestrates the full pipeline from parsed file data to per-segment results.

    Pipeline: parsed_data → segment → DE extract → model inference → InferenceResult
    """

    def __init__(self, model_manager: Any) -> None:
        self.model_manager = model_manager
        self._de_extractors: dict[int, Any] = {}
        self._cache = FeatureCache()

    def process_file(
        self,
        parsed_data: dict[str, Any],
        window_sec: float = 4.0,
        overlap: float = 0.5,
        model_name: str | None = None,
        file_hash: str | None = None,
    ) -> Generator[InferenceResult, None, None]:
        """Process uploaded file and yield one InferenceResult per segment.

        Args:
            parsed_data: Output of ``FileParser.parse()``.
            window_sec: Window length in seconds.
            overlap: Window overlap fraction [0, 1).
            model_name: Override active model name.
            file_hash: File hash for feature caching (enables <5ms model switch).

        Yields:
            InferenceResult for each processed window.
        """
        if model_name:
            self.model_manager.set_active_model(model_name)

        cache_key = self._cache.key(
            file_hash or "nohash", window_sec, overlap,
        )

        if self._cache.has(cache_key):
            all_windows = self._cache.get(cache_key)
            logger.debug("Feature cache HIT — skipping DE extraction")
        else:
            all_windows = self._extract_all_windows(parsed_data, window_sec, overlap)
            self._cache.set(cache_key, all_windows)
            logger.debug("Feature cache SET — %d windows stored", len(all_windows["de"]))

        file_format = parsed_data.get("format", "unknown")

        for idx in range(len(all_windows["de"])):
            de_feat = all_windows["de"][idx]
            t0 = time.perf_counter()
            result = self._run_inference(de_feat, parsed_data, all_windows, idx, file_format)
            result.latency_ms = (time.perf_counter() - t0) * 1000
            result.trial_idx = int(all_windows["trial_idx"][idx])
            result.window_idx = int(all_windows["win_idx"][idx])
            result.time_start_sec = float(all_windows["time_sec"][idx])
            yield result

    def _extract_all_windows(
        self,
        parsed_data: dict[str, Any],
        window_sec: float,
        overlap: float,
    ) -> dict:
        """Segment EEG + compute DE for ALL windows. Stored in cache."""
        fs = parsed_data["fs"]
        win_s = int(window_sec * fs)
        step_s = int(win_s * (1 - overlap))

        if parsed_data.get("pre_extracted"):
            de_data = parsed_data.get("eeg_de")
            if de_data is None:
                return {
                    "de": np.empty((0, 62, 5)),
                    "ch_names": parsed_data.get("ch_names", []),
                    "fs": fs,
                    "trial_idx": np.array([], dtype=int),
                    "win_idx": np.array([], dtype=int),
                    "time_sec": np.array([], dtype=float),
                }
            de_arr = np.asarray(de_data, dtype=np.float32)
            if de_arr.ndim == 2:
                de_arr = de_arr[np.newaxis]
            n = de_arr.shape[0]
            return {
                "de": de_arr,
                "ch_names": parsed_data.get("ch_names", []),
                "fs": fs,
                "trial_idx": np.zeros(n, dtype=int),
                "win_idx": np.arange(n),
                "time_sec": np.arange(n, dtype=float) * (window_sec * (1 - overlap)),
            }

        eeg = parsed_data["eeg"]
        if eeg is None or eeg.size == 0:
            return {
                "de": np.empty((0, 32, 5)),
                "ch_names": parsed_data.get("ch_names", []),
                "fs": fs,
                "trial_idx": np.array([], dtype=int),
                "win_idx": np.array([], dtype=int),
                "time_sec": np.array([], dtype=float),
            }

        extractor = self._get_extractor(fs)

        all_de: list[np.ndarray] = []
        trial_idxs: list[int] = []
        win_idxs: list[int] = []
        times: list[float] = []

        for t_idx in range(eeg.shape[0]):
            trial = eeg[t_idx]
            windows: list[np.ndarray] = []
            start = 0
            w_count = 0
            while start + win_s <= trial.shape[-1]:
                windows.append(trial[:, start:start + win_s])
                trial_idxs.append(t_idx)
                win_idxs.append(w_count)
                times.append(start / fs)
                start += step_s
                w_count += 1

            if windows:
                X = np.stack(windows)
                de = extractor.transform(X)
                all_de.append(de)

        if all_de:
            de_concat = np.concatenate(all_de, axis=0)
        else:
            n_ch = eeg.shape[1] if eeg.ndim >= 2 else 32
            de_concat = np.empty((0, n_ch, 5))

        return {
            "de": de_concat,
            "ch_names": parsed_data.get("ch_names", []),
            "fs": fs,
            "trial_idx": np.array(trial_idxs, dtype=int),
            "win_idx": np.array(win_idxs, dtype=int),
            "time_sec": np.array(times, dtype=float),
        }

    def _run_inference(
        self,
        de_feat: np.ndarray,
        parsed_data: dict[str, Any],
        all_windows: dict,
        idx: int,
        file_format: str,
    ) -> InferenceResult:
        """Run model inference on a single DE feature window."""
        model = self.model_manager.get_active_model()
        model_name = self.model_manager.get_active_model_name()
        model_input = self._make_model_input(model_name, de_feat, model)
        proba = np.asarray(model.predict_proba(model_input))[0]

        pred_class = int(proba.argmax())
        n_classes = proba.shape[0]
        labels = EMOTION_LABELS if n_classes <= 2 else EMOTION_LABELS_5
        label = labels.get(pred_class, f"class_{pred_class}")

        valence, arousal = self._proba_to_va(proba, file_format, model_name)

        attention = None
        if hasattr(model, "get_attention_weights"):
            try:
                if model_name == "DGCCA-AM":
                    attn = model.get_attention_weights(model_input)
                else:
                    attn = model.get_attention_weights()
                if attn is not None:
                    attention = np.asarray(attn)[0] if np.asarray(attn).ndim > 1 else np.asarray(attn)
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

    def _proba_to_va(
        self,
        proba: np.ndarray,
        file_format: str,
        model_name: str,
    ) -> tuple[float, float]:
        """Map model probability output to Valence-Arousal coordinates.

        For DEAP binary models: proba[1] → V or A depending on trained_label.
        For SEED-V 5-class: weighted centroid in Russell's Circumplex.
        """
        if file_format in ("deap_dat", "deap_bdf"):
            p_high = float(proba[1]) if len(proba) > 1 else float(proba[0])
            coord = (p_high - 0.5) * 2.0  # map [0,1] → [-1,1]

            axis = "valence"
            if hasattr(self.model_manager, "get_active_model_axis"):
                axis = self.model_manager.get_active_model_axis()

            if axis == "arousal":
                return 0.0, float(coord)
            return float(coord), 0.0

        elif file_format in ("seed_mat_de", "seed_mat_raw"):
            v = sum(float(proba[c]) * SEED5_VA.get(c, (0, 0))[0] for c in range(len(proba)))
            a = sum(float(proba[c]) * SEED5_VA.get(c, (0, 0))[1] for c in range(len(proba)))
            return float(np.clip(v, -1, 1)), float(np.clip(a, -1, 1))

        else:
            p_high = float(proba[1]) if len(proba) > 1 else float(proba[0])
            val = (p_high - 0.5) * 2.0
            return float(val), 0.0

    def _get_extractor(self, fs: int) -> Any:
        """Get or create a DEExtractor for the given sampling rate."""
        if fs not in self._de_extractors:
            from emokit.features.eeg import DEExtractor
            self._de_extractors[fs] = DEExtractor(fs=fs)
        return self._de_extractors[fs]

    @staticmethod
    def _model_param(model: Any, key: str, default: int) -> int:
        """Read a parameter from model attribute or _params dict."""
        val = getattr(model, key, None)
        if val is not None:
            return int(val)
        params = getattr(model, "_params", {})
        return int(params.get(key, default))

    @staticmethod
    def _pad_flat(flat: np.ndarray, model: Any) -> np.ndarray:
        """Zero-pad flattened features to match model's expected input size."""
        try:
            expected = int(getattr(model, "_in_features", 0) or 0)
        except (TypeError, ValueError):
            return flat
        if expected > 0 and flat.shape[-1] < expected:
            pad_width = expected - flat.shape[-1]
            flat = np.pad(flat, ((0, 0), (0, pad_width)), constant_values=0)
        return flat

    def _make_model_input(self, model_name: str, de_feat: np.ndarray, model: Any) -> Any:
        """Adapt one DE window to each EmoKit model's expected input format."""
        x_de = np.asarray(de_feat, dtype=np.float32)[np.newaxis, ...]
        flat = x_de.reshape(x_de.shape[0], -1)

        n_feat1 = self._model_param(model, "n_feat1", 0) or self._model_param(model, "n_feat_eeg", 0) or self._model_param(model, "n_feat", 0)

        if model_name == "BiDAE":
            mod2_dim = self._model_param(model, "n_feat2", 3)
            eeg_dim = n_feat1 or 160
            eeg_flat = self._pad_to(flat, eeg_dim)
            return {"mod1": eeg_flat, "mod2": np.zeros((1, mod2_dim), dtype=np.float32)}
        if model_name == "DGCCA-AM":
            gsr_dim = self._model_param(model, "n_feat_gsr", 3)
            ecg_dim = self._model_param(model, "n_feat_ecg", 5)
            eeg_dim = self._model_param(model, "n_feat_eeg", 160)
            eeg_flat = self._pad_to(flat, eeg_dim)
            return {
                "eeg": eeg_flat,
                "gsr": np.zeros((1, gsr_dim), dtype=np.float32),
                "ecg": np.zeros((1, ecg_dim), dtype=np.float32),
            }
        if model_name == "Transformer-MM":
            periph_dim = self._model_param(model, "n_peripheral_feat", 8)
            try:
                in_feat = int(getattr(model, "_in_features", 0) or 0)
            except (TypeError, ValueError):
                in_feat = 0
            if in_feat > 0:
                eeg_dim = in_feat - periph_dim
                n_bands = x_de.shape[-1]
                n_ch_target = eeg_dim // n_bands if n_bands > 0 else x_de.shape[1]
            else:
                n_ch_target = x_de.shape[1]
            if x_de.shape[1] < n_ch_target:
                pad_ch = n_ch_target - x_de.shape[1]
                x_de = np.pad(x_de, ((0, 0), (0, pad_ch), (0, 0)), constant_values=0)
            return {
                "eeg": x_de,
                "peripheral": np.zeros((1, periph_dim), dtype=np.float32),
            }
        if model_name == "DGCNN":
            n_ch_cfg = self._model_param(model, "n_channels", x_de.shape[1])
            if x_de.shape[1] < n_ch_cfg:
                pad_ch = n_ch_cfg - x_de.shape[1]
                x_de = np.pad(x_de, ((0, 0), (0, pad_ch), (0, 0)), constant_values=0)
            return x_de

        flat = self._pad_flat(flat, model)
        return flat

    @staticmethod
    def _pad_to(arr: np.ndarray, target_dim: int) -> np.ndarray:
        """Zero-pad last dimension to target_dim if smaller."""
        if arr.shape[-1] < target_dim:
            pad_w = target_dim - arr.shape[-1]
            arr = np.pad(arr, ((0, 0), (0, pad_w)), constant_values=0)
        return arr
