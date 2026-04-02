# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Unit tests for the emosense backend layer."""

from __future__ import annotations

import pickle
import textwrap
import time
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from emosense.backend.file_parser import FileParser
from emosense.backend.signal_buffer import ModalityBuffer, SlidingWindowBuffer
from emosense.backend.stream_receiver import SimulatedReceiver
from emosense.backend.inference import (
    InferenceEngine,
    InferenceResult,
    ModelManager,
)


# ======================================================================
# SlidingWindowBuffer
# ======================================================================


class TestSlidingWindowBuffer:
    """Tests for :class:`SlidingWindowBuffer`."""

    def test_fill_fraction_starts_at_zero(self) -> None:
        buf = SlidingWindowBuffer(n_channels=32, fs=128, window_sec=4.0)
        assert buf.fill_fraction == pytest.approx(0.0)

    def test_push_small_chunks_fills_buffer(self) -> None:
        """Push 512 samples in 16-sample chunks; verify window is available."""
        n_ch, fs, win_sec = 32, 128, 4.0
        window_samples = int(fs * win_sec)  # 512
        chunk_size = 16
        buf = SlidingWindowBuffer(
            n_channels=n_ch, fs=fs, window_sec=win_sec, step_sec=2.0,
        )

        assert buf.get_window() is None

        n_pushes = window_samples // chunk_size
        for i in range(n_pushes):
            chunk = np.random.randn(n_ch, chunk_size)
            buf.push(chunk)

        assert buf.fill_fraction == pytest.approx(1.0)
        win = buf.get_window()
        assert win is not None
        assert win.shape == (n_ch, window_samples)

    def test_new_window_after_step(self) -> None:
        """After getting a window, another 256 samples yield a new window."""
        n_ch, fs, win_sec, step_sec = 32, 128, 4.0, 2.0
        buf = SlidingWindowBuffer(
            n_channels=n_ch, fs=fs, window_sec=win_sec, step_sec=step_sec,
        )
        window_samples = int(fs * win_sec)
        step_samples = int(fs * step_sec)

        buf.push(np.random.randn(n_ch, window_samples))
        assert buf.has_new_window()
        buf.get_window()

        assert not buf.has_new_window()

        buf.push(np.random.randn(n_ch, step_samples))
        assert buf.has_new_window()
        win = buf.get_window()
        assert win is not None
        assert win.shape == (n_ch, window_samples)

    def test_reset_clears_buffer(self) -> None:
        buf = SlidingWindowBuffer(n_channels=4, fs=128, window_sec=1.0)
        buf.push(np.ones((4, 128)))
        buf.reset()
        assert buf.fill_fraction == pytest.approx(0.0)
        assert buf.get_window() is None

    def test_push_invalid_shape_raises(self) -> None:
        buf = SlidingWindowBuffer(n_channels=4, fs=128)
        with pytest.raises(ValueError, match="Expected shape"):
            buf.push(np.zeros((8, 10)))


# ======================================================================
# ModalityBuffer
# ======================================================================


class TestModalityBuffer:
    """Tests for :class:`ModalityBuffer`."""

    @staticmethod
    def _make_config() -> dict[str, dict[str, int | float]]:
        return {
            "eeg": {"n_channels": 32, "fs": 128, "window_sec": 4.0, "step_sec": 2.0},
            "gsr": {"n_channels": 1, "fs": 128, "window_sec": 4.0, "step_sec": 2.0},
        }

    def test_get_all_windows_requires_all_modalities(self) -> None:
        mb = ModalityBuffer(self._make_config())
        mb.push_modality("eeg", np.random.randn(32, 512))
        assert mb.get_all_windows() is None

    def test_get_all_windows_both_ready(self) -> None:
        mb = ModalityBuffer(self._make_config())
        mb.push_modality("eeg", np.random.randn(32, 512))
        mb.push_modality("gsr", np.random.randn(1, 512))
        windows = mb.get_all_windows()
        assert windows is not None
        assert "eeg" in windows and "gsr" in windows
        assert windows["eeg"].shape == (32, 512)
        assert windows["gsr"].shape == (1, 512)

    def test_fill_fraction_is_minimum(self) -> None:
        mb = ModalityBuffer(self._make_config())
        mb.push_modality("eeg", np.random.randn(32, 512))
        assert mb.fill_fraction == pytest.approx(0.0)

    def test_unknown_modality_raises(self) -> None:
        mb = ModalityBuffer(self._make_config())
        with pytest.raises(KeyError, match="Unknown modality"):
            mb.push_modality("unknown", np.zeros((1, 10)))

    def test_reset(self) -> None:
        mb = ModalityBuffer(self._make_config())
        mb.push_modality("eeg", np.random.randn(32, 512))
        mb.push_modality("gsr", np.random.randn(1, 512))
        mb.reset()
        assert mb.fill_fraction == pytest.approx(0.0)


# ======================================================================
# SimulatedReceiver
# ======================================================================


class TestSimulatedReceiver:
    """Tests for :class:`SimulatedReceiver`."""

    def test_produces_chunks(self) -> None:
        fs = 128
        duration_sec = 1.0
        n_samples = int(fs * duration_sec)
        trial_data = {"eeg": np.random.randn(32, n_samples)}
        rx = SimulatedReceiver(
            trial_data=trial_data, fs=fs, chunk_size=32, speed_factor=100.0,
        )

        rx.start()
        time.sleep(0.5)
        rx.stop()

        collected: list[np.ndarray] = []
        while True:
            chunk = rx.get_latest_chunk()
            if chunk is None:
                break
            collected.append(chunk["eeg"])

        assert len(collected) > 0

    def test_is_finished(self) -> None:
        fs = 128
        trial_data = {"eeg": np.random.randn(4, 64)}
        rx = SimulatedReceiver(
            trial_data=trial_data, fs=fs, chunk_size=32, speed_factor=100.0,
        )
        rx.start()
        time.sleep(0.5)
        assert rx.is_finished


# ======================================================================
# InferenceResult
# ======================================================================


class TestInferenceResult:
    """Tests for :class:`InferenceResult` dataclass."""

    def test_fields_present(self) -> None:
        expected = {
            "valence", "arousal", "label", "confidence", "proba",
            "de_features", "attention_weights", "model_name", "latency_ms",
        }
        actual = {f.name for f in fields(InferenceResult)}
        assert actual == expected

    def test_construction(self) -> None:
        result = InferenceResult(
            valence=0.5,
            arousal=-0.3,
            label="High",
            confidence=0.85,
            proba=np.array([0.15, 0.85]),
            de_features=np.zeros((32, 5)),
            attention_weights=None,
            model_name="test",
            latency_ms=12.5,
        )
        assert result.valence == 0.5
        assert result.model_name == "test"
        assert result.attention_weights is None


# ======================================================================
# ModelManager
# ======================================================================


class TestModelManager:
    """Tests for :class:`ModelManager`."""

    @staticmethod
    def _write_config(tmp_path: Path) -> Path:
        cfg = tmp_path / "models.yaml"
        cfg.write_text(textwrap.dedent("""\
            models:
              - name: MockModel
                class: MockClass
                checkpoint: nonexistent.pt
                dataset: DEAP
                modalities: [eeg]
                n_classes: 2
                trained_label: valence
                params:
                  n_classes: 2
              - name: MockMM
                class: MockMM
                checkpoint: nonexistent.pt
                dataset: DEAP
                modalities: [eeg, gsr, ecg]
                n_classes: 2
                trained_label: arousal
                params:
                  n_classes: 2
        """), encoding="utf-8")
        return cfg

    def test_get_model_names(self, tmp_path: Path) -> None:
        cfg = self._write_config(tmp_path)
        mm = ModelManager(config_path=str(cfg))
        names = mm.get_model_names()
        assert "MockModel" in names
        assert "MockMM" in names

    def test_set_active_model_unknown_raises(self, tmp_path: Path) -> None:
        cfg = self._write_config(tmp_path)
        mm = ModelManager(config_path=str(cfg))
        with pytest.raises(KeyError, match="not in config"):
            mm.set_active_model("DoesNotExist")

    @patch("emosense.backend.inference.build_model")
    def test_set_active_model(
        self, mock_build: MagicMock, tmp_path: Path,
    ) -> None:
        mock_model = MagicMock()
        mock_build.return_value = mock_model

        cfg = self._write_config(tmp_path)
        mm = ModelManager(config_path=str(cfg))
        mm.set_active_model("MockModel")

        assert mm.get_active_model() is mock_model
        assert mock_build.call_count == 2

    def test_get_required_modalities(self, tmp_path: Path) -> None:
        cfg = self._write_config(tmp_path)
        mm = ModelManager(config_path=str(cfg))
        assert mm.get_required_modalities("MockModel") == ["eeg"]
        assert mm.get_required_modalities("MockMM") == ["eeg", "gsr", "ecg"]

    def test_get_active_model_without_setting_raises(
        self, tmp_path: Path,
    ) -> None:
        mock_path = tmp_path / "missing.yaml"
        mm = ModelManager(config_path=str(mock_path))
        with pytest.raises(RuntimeError, match="No active model"):
            mm.get_active_model()

    def test_get_active_model_name(self, tmp_path: Path) -> None:
        cfg = self._write_config(tmp_path)
        mm = ModelManager(config_path=str(cfg))
        name = mm.get_active_model_name()
        assert name in mm.get_model_names() or name == "none"

    @patch("emosense.backend.inference.build_model")
    def test_get_active_model_axis(self, mock_build: MagicMock, tmp_path: Path) -> None:
        mock_build.return_value = MagicMock()
        cfg = self._write_config(tmp_path)
        mm = ModelManager(config_path=str(cfg))
        mm.set_active_model("MockModel")
        assert mm.get_active_model_axis() == "valence"
        mm.set_active_model("MockMM")
        assert mm.get_active_model_axis() == "arousal"


# ======================================================================
# InferenceEngine
# ======================================================================


class TestInferenceEngine:
    """Tests for :class:`InferenceEngine`."""

    @patch("emosense.backend.inference.build_model")
    def test_process_window(self, mock_build: MagicMock, tmp_path: Path) -> None:
        logits = np.array([[0.2, 0.8]])
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = logits
        mock_build.return_value = mock_model

        cfg_path = tmp_path / "models.yaml"
        cfg_path.write_text(textwrap.dedent("""\
            models:
              - name: TestModel
                class: TestClass
                checkpoint: nonexistent.pt
                dataset: DEAP
                modalities: [eeg]
                n_classes: 2
                params:
                  n_classes: 2
        """), encoding="utf-8")

        mm = ModelManager(config_path=str(cfg_path))
        mm.set_active_model("TestModel")

        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = np.random.randn(1, 160)

        engine = InferenceEngine(
            model_manager=mm, feature_pipeline=mock_pipeline,
        )

        window = {"eeg": np.random.randn(32, 512)}
        result = engine.process_window(window)

        assert isinstance(result, InferenceResult)
        assert result.model_name == "TestModel"
        assert result.label in ("Low", "High")
        assert 0.0 <= result.confidence <= 1.0
        assert result.proba.shape == (2,)
        assert result.de_features.ndim == 2
        assert result.latency_ms >= 0.0

    @patch("emosense.backend.inference.build_model")
    def test_latency_is_measured(
        self, mock_build: MagicMock, tmp_path: Path,
    ) -> None:
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.5, 0.5]])
        mock_build.return_value = mock_model

        cfg_path = tmp_path / "models.yaml"
        cfg_path.write_text(textwrap.dedent("""\
            models:
              - name: Latency
                class: LC
                checkpoint: none.pt
                dataset: DEAP
                modalities: [eeg]
                n_classes: 2
                params:
                  n_classes: 2
        """), encoding="utf-8")

        mm = ModelManager(config_path=str(cfg_path))
        mm.set_active_model("Latency")

        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = np.random.randn(1, 160)

        engine = InferenceEngine(model_manager=mm, feature_pipeline=mock_pipeline)
        result = engine.process_window({"eeg": np.random.randn(32, 512)})

        assert result.latency_ms > 0.0


# ======================================================================
# FileParser
# ======================================================================


class TestFileParser:
    """Tests for :class:`FileParser`."""

    @staticmethod
    def _make_deap_dat(directory: Path) -> Path:
        """Create a minimal DEAP .dat pickle."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((4, 40, 8064)).astype(np.float32)
        labels = np.zeros((4, 4), dtype=np.float32)
        labels[:, 0] = [3.0, 6.0, 4.0, 7.0]
        labels[:, 1] = [5.0, 2.0, 8.0, 1.0]
        path = directory / "s01.dat"
        with open(path, "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f)
        return path

    def test_parse_deap_dat_returns_expected_keys(self, tmp_path: Path) -> None:
        dat = self._make_deap_dat(tmp_path)
        result = FileParser.parse(dat)
        assert result["format"] == "deap_dat"
        assert result["fs"] == 128
        assert "eeg" in result
        assert "labels" in result
        assert "ch_names" in result
        assert result["n_eeg_channels"] == 32

    def test_parse_deap_dat_eeg_shape(self, tmp_path: Path) -> None:
        dat = self._make_deap_dat(tmp_path)
        result = FileParser.parse(dat)
        eeg = result["eeg"]
        assert eeg.ndim == 3
        assert eeg.shape[0] == 4
        assert eeg.shape[1] == 32
        expected_samples = 8064 - 3 * 128
        assert eeg.shape[2] == expected_samples

    def test_parse_deap_dat_peripheral_signals(self, tmp_path: Path) -> None:
        dat = self._make_deap_dat(tmp_path)
        result = FileParser.parse(dat)
        assert result["gsr"].shape == (4, 1, 8064 - 3 * 128)
        assert result["ecg"].shape == (4, 1, 8064 - 3 * 128)

    def test_parse_deap_dat_labels(self, tmp_path: Path) -> None:
        dat = self._make_deap_dat(tmp_path)
        result = FileParser.parse(dat)
        labels = result["labels"]
        assert labels.shape == (4,)
        assert set(labels.tolist()).issubset({0, 1})

    @staticmethod
    def _make_deap_mat(directory: Path) -> Path:
        """Create a minimal DEAP .mat file."""
        import scipy.io

        rng = np.random.default_rng(0)
        data = rng.standard_normal((4, 40, 8064)).astype(np.float32)
        labels = np.zeros((4, 4), dtype=np.float32)
        labels[:, 0] = [3.0, 6.0, 4.0, 7.0]
        labels[:, 1] = [5.0, 2.0, 8.0, 1.0]
        path = directory / "1.mat"
        scipy.io.savemat(str(path), {"data": data, "labels": labels})
        return path

    def test_parse_deap_mat_returns_expected_keys(self, tmp_path: Path) -> None:
        mat = self._make_deap_mat(tmp_path)
        result = FileParser.parse(mat)
        assert "DEAP" in result["format"]
        assert result["fs"] == 128
        assert "eeg" in result
        assert "labels" in result
        assert "ch_names" in result

    def test_parse_deap_mat_eeg_shape(self, tmp_path: Path) -> None:
        mat = self._make_deap_mat(tmp_path)
        result = FileParser.parse(mat)
        eeg = result["eeg"]
        assert eeg.ndim == 3
        assert eeg.shape[0] == 4
        assert eeg.shape[1] == 32
        expected_samples = 8064 - 3 * 128
        assert eeg.shape[2] == expected_samples

    def test_parse_deap_mat_peripheral_signals(self, tmp_path: Path) -> None:
        mat = self._make_deap_mat(tmp_path)
        result = FileParser.parse(mat)
        assert result["gsr"].shape == (4, 1, 8064 - 3 * 128)
        assert result["ecg"].shape == (4, 1, 8064 - 3 * 128)

    @staticmethod
    def _make_seedv_npz(directory: Path) -> Path:
        """Create a minimal SEED-V .npz file."""
        rng = np.random.default_rng(0)
        data_dict = {}
        label_dict = {}
        for i in range(3):
            data_dict[i] = rng.standard_normal((10, 310)).astype(np.float32)
            label_dict[i] = int(rng.integers(0, 5))
        path = directory / "1_123.npz"
        np.savez(str(path), data=data_dict, label=label_dict)
        return path

    def test_parse_seedv_npz_returns_expected_keys(self, tmp_path: Path) -> None:
        npz = self._make_seedv_npz(tmp_path)
        result = FileParser.parse(npz)
        assert "SEED-V" in result["format"]
        assert result["pre_extracted"] is True
        assert "eeg_de" in result
        assert len(result["ch_names"]) == 62

    def test_parse_seedv_npz_reshapes_to_62x5(self, tmp_path: Path) -> None:
        npz = self._make_seedv_npz(tmp_path)
        result = FileParser.parse(npz)
        first_trial = result["eeg_de"][0]
        assert first_trial.ndim == 3
        assert first_trial.shape[1] == 62
        assert first_trial.shape[2] == 5

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "data.xyz"
        bad.write_bytes(b"garbage")
        with pytest.raises(ValueError, match="Unsupported"):
            FileParser.parse(bad)

    def test_supported_formats_list(self) -> None:
        assert ".dat" in FileParser.SUPPORTED_FORMATS
        assert ".mat" in FileParser.SUPPORTED_FORMATS
        assert ".npz" in FileParser.SUPPORTED_FORMATS
        assert ".csv" in FileParser.SUPPORTED_FORMATS
        assert ".bdf" in FileParser.SUPPORTED_FORMATS

    def test_parse_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            FileParser.parse(Path("/nonexistent/file.dat"))

    def test_parse_csv_mock(self, tmp_path: Path) -> None:
        import pandas as pd
        t = np.linspace(0, 10, 1280)
        data = {f"ch{i}": np.random.randn(1280) for i in range(4)}
        df = pd.DataFrame({"time": t, **data})
        path = tmp_path / "signal.csv"
        df.to_csv(path, index=False)

        result = FileParser.parse(path)
        assert result["format"] == "csv"
        assert result["eeg"].shape[1] == 4
        assert result["fs"] == 128
        assert result["n_eeg_channels"] == 4

    def test_parse_deap_dat_mock(self, tmp_path: Path) -> None:
        data = np.random.randn(40, 40, 8064).astype(np.float32)
        labels = np.ones((40, 4), dtype=np.float32) * 6
        path = tmp_path / "s01.dat"
        with open(path, "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f)

        result = FileParser.parse(path)
        assert result["format"] == "deap_dat"
        assert result["eeg"].shape == (40, 32, 7680)
        assert result["gsr"].shape == (40, 1, 7680)
        assert result["fs"] == 128
        assert result["n_eeg_channels"] == 32
        assert set(result["labels"].tolist()).issubset({0, 1})


# ======================================================================
# ProcessingEngine (FeatureCache)
# ======================================================================


class TestFeatureCache:
    """Tests for FeatureCache in ProcessingEngine."""

    @staticmethod
    def _make_mock_parsed(n_trials: int = 2) -> dict:
        rng = np.random.default_rng(42)
        return {
            "eeg": rng.standard_normal((n_trials, 32, 8064 - 384)).astype(np.float32),
            "gsr": None,
            "ecg": None,
            "labels": np.array([1, 0][:n_trials]),
            "fs": 128,
            "ch_names": [f"ch{i}" for i in range(32)],
            "format": "deap_dat",
            "n_eeg_channels": 32,
            "pre_extracted": False,
        }

    def test_feature_cache_hit(self) -> None:
        """Second call with same file_hash must skip DE extraction."""
        from emosense.backend.processing_engine import ProcessingEngine

        mock_mm = MagicMock()
        mock_mm.get_active_model_name.return_value = "Mock"
        mock_mm.set_active_model = MagicMock()

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.5, 0.5]], dtype=np.float32)
        mock_model.get_attention_weights.return_value = None
        mock_mm.get_active_model.return_value = mock_model

        engine = ProcessingEngine(mock_mm)

        call_count = {"n": 0}
        original = engine._extract_all_windows

        def counting_wrapper(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        engine._extract_all_windows = counting_wrapper

        parsed = self._make_mock_parsed(1)
        list(engine.process_file(parsed, file_hash="abc123", window_sec=4.0))
        list(engine.process_file(parsed, file_hash="abc123", window_sec=4.0))

        assert call_count["n"] == 1, "DE extraction ran twice (cache not working)"
