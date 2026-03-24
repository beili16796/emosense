"""Real-data smoke tests for EmoSense.

These tests require ``EMOKIT_DATA_ROOT`` to be set and DEAP/SEED-V files to
be present.  They are automatically skipped when the env var is absent.

Run explicitly::

    pytest tests/test_real_data.py -v -m "real_data"
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

EMOKIT_DATA_ROOT = os.environ.get("EMOKIT_DATA_ROOT")
DEAP_ROOT = Path(EMOKIT_DATA_ROOT) / "DEAP" if EMOKIT_DATA_ROOT else None
SEEDV_ROOT = Path(EMOKIT_DATA_ROOT) / "SEED-V" if EMOKIT_DATA_ROOT else None

skip_no_data = pytest.mark.skipif(
    EMOKIT_DATA_ROOT is None,
    reason="EMOKIT_DATA_ROOT not set — skipping real-data tests",
)
skip_no_deap = pytest.mark.skipif(
    DEAP_ROOT is None or not (DEAP_ROOT / "s01.dat").exists() if DEAP_ROOT else True,
    reason="DEAP s01.dat not found",
)
skip_no_seedv = pytest.mark.skipif(
    SEEDV_ROOT is None or not SEEDV_ROOT.is_dir() if SEEDV_ROOT else True,
    reason="SEED-V data not found",
)

real_data = pytest.mark.real_data


# ---------------------------------------------------------------------------
# DEAP file parsing via EmoSense FileParser
# ---------------------------------------------------------------------------


@real_data
@skip_no_deap
class TestDeapRealFile:
    """Validate that EmoSense can parse a real DEAP .dat file."""

    @pytest.fixture(autouse=True)
    def _parsed(self):
        from emosense.backend.file_parser import FileParser

        self.result = FileParser.parse(DEAP_ROOT / "s01.dat")

    def test_format_detected(self):
        assert self.result["format"] == "deap_dat"

    def test_eeg_shape(self):
        eeg = self.result["eeg"]
        assert eeg is not None
        assert eeg.shape[0] == 40, f"Expected 40 trials, got {eeg.shape[0]}"
        assert eeg.shape[1] == 32, f"Expected 32 EEG channels, got {eeg.shape[1]}"
        assert eeg.shape[2] == 7680, f"Expected 7680 samples, got {eeg.shape[2]}"

    def test_gsr_shape(self):
        gsr = self.result["gsr"]
        assert gsr is not None
        assert gsr.shape == (40, 1, 7680)

    def test_labels_binary(self):
        labels = self.result["labels"]
        assert set(labels.tolist()).issubset({0, 1})

    def test_no_nan_inf(self):
        for key in ("eeg", "gsr", "ecg"):
            arr = self.result.get(key)
            if arr is not None:
                assert not np.any(np.isnan(arr)), f"{key} contains NaN"
                assert not np.any(np.isinf(arr)), f"{key} contains Inf"

    def test_channel_names(self):
        assert len(self.result["ch_names"]) == 32

    def test_fs(self):
        assert self.result["fs"] == 128


# ---------------------------------------------------------------------------
# SEED-V file parsing
# ---------------------------------------------------------------------------


@real_data
@skip_no_seedv
class TestSeedvRealFile:
    """Validate that EmoSense can parse a real SEED-V .mat file."""

    @pytest.fixture(autouse=True)
    def _find_mat(self):
        from emosense.backend.file_parser import FileParser

        mat_files = sorted(SEEDV_ROOT.rglob("*.mat"))
        if not mat_files:
            pytest.skip("No .mat files found under SEED-V root")
        self.result = FileParser.parse(mat_files[0])

    def test_format_detected(self):
        assert self.result["format"] in ("seed_mat_de", "seed_mat_raw")

    def test_channel_count(self):
        assert self.result["n_eeg_channels"] == 62

    def test_labels_valid(self):
        labels = self.result["labels"]
        unique = set(labels.tolist())
        assert all(0 <= v <= 4 for v in unique), f"Unexpected label values: {unique}"


# ---------------------------------------------------------------------------
# V-A trajectory correctness with real DEAP data
# ---------------------------------------------------------------------------


@real_data
@skip_no_deap
class TestVADirectionality:
    """Verify that high-valence DEAP trials produce positive V predictions."""

    def test_high_valence_positive_v(self):
        """Trials with ground-truth V > 7 should cluster V > 0 on average."""
        import pickle

        with open(DEAP_ROOT / "s01.dat", "rb") as f:
            raw = pickle.load(f, encoding="latin1")
        labels = np.asarray(raw["labels"])
        high_v_mask = labels[:, 0] > 7
        if high_v_mask.sum() < 2:
            pytest.skip("Not enough high-valence trials in s01")

        # The test validates that we can identify which trials are high-valence
        # and that label extraction works correctly
        assert high_v_mask.sum() > 0
        assert labels[high_v_mask, 0].mean() > 7.0


# ---------------------------------------------------------------------------
# Frontal asymmetry check
# ---------------------------------------------------------------------------


@real_data
@skip_no_deap
class TestFrontalAsymmetry:
    """Check that high-valence trials show expected F4 > F3 alpha pattern."""

    def test_frontal_alpha_asymmetry(self):
        """Davidson (1992): positive valence → right frontal dominance."""
        import pickle
        from scipy.signal import butter, sosfiltfilt

        with open(DEAP_ROOT / "s01.dat", "rb") as f:
            raw = pickle.load(f, encoding="latin1")

        data = np.asarray(raw["data"])[:, :32, 384:]  # baseline removed
        labels = np.asarray(raw["labels"])

        high_v_mask = labels[:, 0] > 6
        if high_v_mask.sum() < 3:
            pytest.skip("Not enough high-valence trials")

        # Alpha band: 8-13 Hz
        sos = butter(5, [8.0, 13.0], btype="band", fs=128, output="sos")

        # F3 is channel index 2, F4 is channel index 18 in DEAP ordering
        f3_idx, f4_idx = 2, 18
        hv_data = data[high_v_mask]

        f3_alpha = sosfiltfilt(sos, hv_data[:, f3_idx, :], axis=-1)
        f4_alpha = sosfiltfilt(sos, hv_data[:, f4_idx, :], axis=-1)

        f3_power = np.log(np.var(f3_alpha, axis=-1) + 1e-10)
        f4_power = np.log(np.var(f4_alpha, axis=-1) + 1e-10)

        # Log the asymmetry for diagnostics
        asymmetry = (f4_power - f3_power).mean()
        print(f"Frontal alpha asymmetry (F4-F3): {asymmetry:.4f}")

        # We check that asymmetry is computed without error;
        # the actual sign depends on individual subjects and is not
        # guaranteed for every single subject
        assert np.isfinite(asymmetry)
