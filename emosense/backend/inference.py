# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Model management and real-time inference engine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from emokit.features import FeaturePipeline
from emokit.models import BaseModel, build_model

logger = logging.getLogger(__name__)

LABEL_MAP_VA: dict[int, str] = {0: "Low", 1: "High"}
LABEL_MAP_SEEDV: dict[int, str] = {
    0: "Happy",
    1: "Sad",
    2: "Neutral",
    3: "Fear",
    4: "Disgust",
}


@dataclass
class InferenceResult:
    """Container for a single inference output.

    Attributes:
        valence: Predicted valence coordinate in ``[-1, 1]``.
        arousal: Predicted arousal coordinate in ``[-1, 1]``.
        label: Human-readable emotion label.
        confidence: Maximum softmax probability.
        proba: Full softmax distribution over classes.
        de_features: DE feature array of shape ``(n_channels, 5)`` for
            topographic heatmap visualisation.
        attention_weights: Modality attention weights ``(3,)`` for
            DGCCA-AM, otherwise ``None``.
        model_name: Name of the model that produced this result.
        latency_ms: Wall-clock inference latency in milliseconds.
    """

    valence: float
    arousal: float
    label: str
    confidence: float
    proba: np.ndarray
    de_features: np.ndarray
    attention_weights: np.ndarray | None
    model_name: str
    latency_ms: float


class ModelManager:
    """Load, cache, and switch between multiple emotion recognition models.

    Args:
        config_path: Path to a YAML file describing available models.
    """

    def __init__(self, config_path: str = "config/models.yaml") -> None:
        self._config_path = Path(config_path)
        self._model_configs: dict[str, dict[str, Any]] = {}
        self._models: dict[str, BaseModel] = {}
        self._active_name: str | None = None

        self._load_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Set the model identified by *name* as the active model.

        Raises:
            KeyError: If *name* is not found in the configuration.
        """
        if name not in self._model_configs:
            raise KeyError(
                f"Model {name!r} not in config. "
                f"Available: {self.get_model_names()}",
            )
        if name not in self._models:
            self._build_and_load(name)
        self._active_name = name
        logger.info("Active model set to %s", name)

    def get_model_names(self) -> list[str]:
        """Return names of all configured models."""
        return list(self._model_configs.keys())

    def is_using_real_weights(self, name: str) -> bool:
        """Whether a configured checkpoint exists on disk."""
        cfg = self._model_configs.get(name, {})
        ckpt = Path(cfg.get("checkpoint", ""))
        return ckpt.exists()

    def get_model_info(self) -> list[dict[str, Any]]:
        """Return model metadata for the UI."""
        info = []
        for name, cfg in self._model_configs.items():
            info.append(
                {
                    "name": name,
                    "modalities": cfg.get("modalities", ["eeg"]),
                    "dataset": cfg.get("dataset", "unknown"),
                    "trained_label": cfg.get("trained_label", "valence"),
                    "has_real_weights": self.is_using_real_weights(name),
                    "description": cfg.get("description", ""),
                }
            )
        return info

    def get_active_model_name(self) -> str:
        """Return the name of the currently active model."""
        return self._active_name or "none"

    def get_active_model(self) -> BaseModel:
        """Return the currently active model instance.

        Raises:
            RuntimeError: If no active model has been set.
        """
        if self._active_name is None or self._active_name not in self._models:
            raise RuntimeError(
                "No active model. Call set_active_model() first.",
            )
        return self._models[self._active_name]

    def get_active_model_axis(self) -> str:
        """Returns 'valence', 'arousal', or 'five_class' based on config."""
        if self._active_name and self._active_name in self._model_configs:
            return self._model_configs[self._active_name].get("trained_label", "valence")
        return "valence"

    def get_required_modalities(self, name: str | None = None) -> list[str]:
        """Return the modality list required by a model.

        Args:
            name: Model name; defaults to the active model.

        Returns:
            List of modality strings (e.g. ``['eeg', 'gsr', 'ecg']``).
        """
        target = name or self._active_name
        if target is None or target not in self._model_configs:
            raise KeyError(f"Model {target!r} not found in config")
        return list(self._model_configs[target].get("modalities", []))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Parse config and eagerly load all declared models."""
        if not self._config_path.exists():
            logger.warning("Config file not found: %s", self._config_path)
            return
        with self._config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        for entry in raw.get("models", []):
            name = entry["name"]
            self._model_configs[name] = entry
        logger.info(
            "Loaded %d model configs from %s",
            len(self._model_configs),
            self._config_path,
        )
        for name in self._model_configs:
            try:
                self._build_and_load(name)
            except Exception as exc:
                logger.error("Failed to load %s: %s", name, exc)
        if self._models and self._active_name is None:
            self._active_name = next(iter(self._models))
            logger.info("Active model: %s", self._active_name)

    def _build_and_load(self, name: str) -> None:
        """Instantiate a model and optionally load a checkpoint."""
        cfg = self._model_configs[name]
        model_class = cfg["class"]
        params = dict(cfg.get("params", {}))

        model = build_model(model_class, params)

        ckpt = Path(cfg.get("checkpoint", ""))
        if ckpt.exists():
            model.load(str(ckpt))
            logger.info("Loaded checkpoint for %s from %s", name, ckpt)
        else:
            logger.warning(
                "No checkpoint found for %s at %s; using random weights",
                name,
                ckpt,
            )

        if getattr(model, "network", None) is not None:
            model.network.eval()

        self._models[name] = model
        if self._active_name is None:
            self._active_name = name


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _logits_to_va(proba: np.ndarray) -> tuple[float, float]:
    """Map a 2-class probability to valence/arousal in ``[-1, 1]``."""
    p_high = float(proba[1]) if proba.shape[0] == 2 else float(proba.max())
    coord = 2.0 * p_high - 1.0
    return coord, coord


class InferenceEngine:
    """Run feature extraction and model forward pass on signal windows.

    Args:
        model_manager: Pre-configured :class:`ModelManager`.
        feature_pipeline: An emokit :class:`FeaturePipeline` instance.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        feature_pipeline: FeaturePipeline,
    ) -> None:
        self._model_manager = model_manager
        self._pipeline = feature_pipeline

    def process_window(
        self, window: dict[str, np.ndarray],
    ) -> InferenceResult:
        """Extract features from *window* and run model inference.

        Args:
            window: Dict mapping modality name to array of shape
                ``(n_channels, n_samples)``.

        Returns:
            Populated :class:`InferenceResult`.
        """
        t0 = time.perf_counter()

        model = self._model_manager.get_active_model()
        active_name = self._model_manager._active_name or "unknown"

        eeg_window = window.get("eeg")
        features = self._extract_features(window)
        de_features = self._extract_de(eeg_window)

        logits = model.predict_proba(features)
        if logits.ndim == 2:
            logits = logits[0]

        proba = _softmax(logits)
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])

        n_classes = proba.shape[0]
        label_map = LABEL_MAP_SEEDV if n_classes == 5 else LABEL_MAP_VA
        label = label_map.get(pred_idx, str(pred_idx))

        valence, arousal = _logits_to_va(proba)

        attention_weights: np.ndarray | None = None
        if hasattr(model, "get_attention_weights"):
            attention_weights = model.get_attention_weights()

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "Inference: %s -> %s (%.1f%%) in %.1f ms",
            active_name,
            label,
            confidence * 100,
            latency_ms,
        )

        return InferenceResult(
            valence=valence,
            arousal=arousal,
            label=label,
            confidence=confidence,
            proba=proba,
            de_features=de_features,
            attention_weights=attention_weights,
            model_name=active_name,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(
        self, window: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Apply the feature pipeline to a single window.

        Wraps each modality array to add a batch dimension before the
        pipeline, then removes it afterward.
        """
        batched: dict[str, np.ndarray] = {}
        for name, arr in window.items():
            batched[name] = arr[np.newaxis, ...]

        if len(batched) == 1:
            key = next(iter(batched))
            result = self._pipeline.transform(batched[key])
        else:
            result = self._pipeline.transform(batched)

        return result

    @staticmethod
    def _extract_de(eeg: np.ndarray | None) -> np.ndarray:
        """Compute DE features for visualisation.

        Returns an array of shape ``(n_channels, 5)`` or zeros if no EEG.
        """
        if eeg is None:
            return np.zeros((1, 5), dtype=np.float64)

        from emokit.features.eeg import DEExtractor

        extractor = DEExtractor()
        batch = eeg[np.newaxis, ...]
        de = extractor.transform(batch)
        return de[0]
