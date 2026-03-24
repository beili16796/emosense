# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Dedicated tests for ProcessingEngine — feature cache and latency."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from emosense.backend.processing_engine import (
    FeatureCache,
    InferenceResult,
    ProcessingEngine,
)


@pytest.fixture()
def mock_deap_parsed() -> dict:
    rng = np.random.default_rng(42)
    return {
        "eeg": rng.standard_normal((2, 32, 7680)).astype(np.float32),
        "gsr": None,
        "ecg": None,
        "labels": np.array([1, 0]),
        "fs": 128,
        "ch_names": [f"ch{i}" for i in range(32)],
        "format": "deap_dat",
        "n_eeg_channels": 32,
        "pre_extracted": False,
    }


def _make_mock_engine():
    mock_mm = MagicMock()
    mock_mm.get_active_model_name.return_value = "DGCNN"
    mock_mm.set_active_model = MagicMock()
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.5, 0.5]], dtype=np.float32)
    mock_model.get_attention_weights.return_value = None
    mock_mm.get_active_model.return_value = mock_model
    return ProcessingEngine(mock_mm)


class TestFeatureCacheUnit:
    """Direct tests for FeatureCache."""

    def test_set_and_get(self) -> None:
        cache = FeatureCache()
        k = cache.key("abc", 4.0, 0.5)
        cache.set(k, {"de": np.zeros((10, 32, 5))})
        assert cache.has(k)
        assert cache.get(k) is not None

    def test_lru_eviction(self) -> None:
        cache = FeatureCache()
        for i in range(4):
            cache.set(f"key_{i}", {"de": np.zeros(1)})
        assert not cache.has("key_0")
        assert cache.has("key_3")

    def test_clear(self) -> None:
        cache = FeatureCache()
        cache.set("k", {"de": np.zeros(1)})
        cache.clear()
        assert not cache.has("k")


class TestFeatureCacheIntegration:
    """Feature cache hit test through ProcessingEngine."""

    def test_feature_cache_hit(self, mock_deap_parsed: dict) -> None:
        engine = _make_mock_engine()

        call_count = {"n": 0}
        original = engine._extract_all_windows

        def counting_wrapper(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        engine._extract_all_windows = counting_wrapper

        list(engine.process_file(mock_deap_parsed, file_hash="abc123"))
        list(engine.process_file(mock_deap_parsed, file_hash="abc123"))

        assert call_count["n"] == 1, "DE extraction ran twice (cache not working)"

    def test_different_hash_triggers_reextraction(self, mock_deap_parsed: dict) -> None:
        engine = _make_mock_engine()

        call_count = {"n": 0}
        original = engine._extract_all_windows

        def counting_wrapper(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        engine._extract_all_windows = counting_wrapper

        list(engine.process_file(mock_deap_parsed, file_hash="hash1"))
        list(engine.process_file(mock_deap_parsed, file_hash="hash2"))

        assert call_count["n"] == 2


class TestProcessingEngineInference:
    """Test inference output structure."""

    def test_yields_inference_results(self, mock_deap_parsed: dict) -> None:
        engine = _make_mock_engine()
        results = list(engine.process_file(mock_deap_parsed, window_sec=4.0))
        assert len(results) > 0
        for r in results:
            assert isinstance(r, InferenceResult)
            assert r.latency_ms >= 0
            assert -1.0 <= r.valence <= 1.0
            assert -1.0 <= r.arousal <= 1.0

    def test_latency_per_window_under_300ms(self, mock_deap_parsed: dict) -> None:
        """Every yielded result must have latency_ms < 300."""
        engine = _make_mock_engine()
        results = list(engine.process_file(mock_deap_parsed, window_sec=4.0))
        assert len(results) > 0
        for r in results:
            assert r.latency_ms < 300, f"Window latency {r.latency_ms:.1f}ms > 300ms"
