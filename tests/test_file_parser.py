# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Dedicated tests for FileParser — covers all three format parsers."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emosense.backend.file_parser import FileParser


def test_parse_deap_dat_mock(tmp_path: Path) -> None:
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


def test_parse_csv_mock(tmp_path: Path) -> None:
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


def test_parse_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "data.xyz"
    path.write_bytes(b"garbage")
    with pytest.raises(ValueError, match="Unsupported format"):
        FileParser.parse(path)


def test_parse_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        FileParser.parse(Path("/nonexistent/file.dat"))


def test_parse_csv_single_column(tmp_path: Path) -> None:
    """CSV with only 1 column should raise."""
    df = pd.DataFrame({"time": np.linspace(0, 1, 100)})
    path = tmp_path / "single.csv"
    df.to_csv(path, index=False)
    with pytest.raises((ValueError, RuntimeError)):
        FileParser.parse(path)


def test_parse_deap_dat_labels_binary(tmp_path: Path) -> None:
    """Valence > 5 → 1, <= 5 → 0."""
    data = np.random.randn(4, 40, 8064).astype(np.float32)
    labels = np.array([
        [3.0, 5.0, 5.0, 5.0],  # 3.0 <= 5 → 0
        [5.0, 5.0, 5.0, 5.0],  # 5.0 <= 5 → 0
        [6.0, 5.0, 5.0, 5.0],  # 6.0 > 5  → 1
        [9.0, 5.0, 5.0, 5.0],  # 9.0 > 5  → 1
    ], dtype=np.float32)
    path = tmp_path / "s01.dat"
    with open(path, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)

    result = FileParser.parse(path)
    expected = np.array([0, 0, 1, 1])
    np.testing.assert_array_equal(result["labels"], expected)


def test_parse_csv_irregular_timestamps(tmp_path: Path) -> None:
    """Irregular timestamps should default to 128 Hz."""
    t = np.sort(np.random.rand(100) * 10)
    df = pd.DataFrame({"time": t, "ch0": np.random.randn(100)})
    path = tmp_path / "irregular.csv"
    df.to_csv(path, index=False)

    result = FileParser.parse(path)
    assert result["fs"] == 128
