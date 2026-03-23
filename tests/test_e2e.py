# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""End-to-end tests for the EmoSense file-upload pipeline."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from emosense.backend.server import app, SESSION_STORE, RESULTS_STORE, COMPLETED_TASKS


def _make_mock_deap_dat(directory: Path) -> Path:
    """Create a minimal valid DEAP .dat file with synthetic data."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((4, 40, 8064)).astype(np.float32)
    labels = np.zeros((4, 4), dtype=np.float32)
    labels[:, 0] = [2.0, 7.0, 3.0, 8.0]  # valence
    labels[:, 1] = [6.0, 4.0, 8.0, 2.0]  # arousal
    payload = {"data": data, "labels": labels}
    path = directory / "s01_mock.dat"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


@pytest.fixture()
def client():
    """Provide a TestClient for the FastAPI app."""
    SESSION_STORE.clear()
    RESULTS_STORE.clear()
    COMPLETED_TASKS.clear()
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def mock_dat(tmp_path: Path) -> Path:
    """Create a mock DEAP .dat file in a temp directory."""
    return _make_mock_deap_dat(tmp_path)


class TestUploadEndpoint:
    """Tests for POST /upload."""

    def test_upload_returns_metadata(self, client: TestClient, mock_dat: Path) -> None:
        with open(mock_dat, "rb") as f:
            r = client.post(
                "/upload",
                files={"file": (mock_dat.name, f, "application/octet-stream")},
                data={"window_sec": "4.0", "overlap": "0.5", "model_name": "DGCNN"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "task_id" in body
        assert body["format_detected"] == "deap_dat"
        assert body["n_trials"] == 4
        assert body["fs"] == 128
        assert body["n_channels"] == 32

    def test_unsupported_format_returns_422(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        bad = tmp_path / "data.xyz"
        bad.write_bytes(b"garbage")
        with open(bad, "rb") as f:
            r = client.post(
                "/upload",
                files={"file": ("data.xyz", f, "application/octet-stream")},
            )
        assert r.status_code == 422
        assert "Unsupported" in r.json()["detail"]


class TestHealthAndModels:
    """Tests for GET /health and model endpoints."""

    def test_health(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_model_switch(self, client: TestClient) -> None:
        r = client.post("/models/active", json={"name": "CNN-LSTM"})
        assert r.status_code == 200
        assert r.json()["active_model"] == "CNN-LSTM"


class TestWebSocket:
    """Tests for WebSocket /ws."""

    def test_ws_connect(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_text("ping")


class TestResultsBuffer:
    """Tests for /results/latest with buffer and polling."""

    def test_results_latest_structure(self, client: TestClient) -> None:
        r = client.get("/results/latest?task_id=nonexistent")
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert "next_idx" in data
        assert "is_complete" in data

    def test_results_incremental(self, client: TestClient) -> None:
        for i in range(5):
            RESULTS_STORE.setdefault("buf_test", []).append(
                {"type": "inference", "idx": i}
            )

        r = client.get("/results/latest?task_id=buf_test&since_idx=2")
        data = r.json()
        assert data["next_idx"] == 5
        assert len(data["results"]) == 3

        RESULTS_STORE.pop("buf_test", None)


class TestFullPipeline:
    """Integration tests for the full upload → process → results pipeline."""

    def test_upload_deap_returns_correct_metadata(
        self, client: TestClient, mock_dat: Path,
    ) -> None:
        with open(mock_dat, "rb") as f:
            r = client.post(
                "/upload",
                files={"file": (mock_dat.name, f)},
            )
        assert r.status_code == 200
        d = r.json()
        assert d["format_detected"] == "deap_dat"
        assert d["n_trials"] == 4
        assert d["fs"] == 128
        assert d["n_channels"] == 32
        assert d["estimated_segments"] > 0

    def test_health_endpoint(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert "active_model" in r.json()

    def test_unsupported_file_returns_422(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        bad = tmp_path / "data.xlsx"
        bad.write_bytes(b"\x00\x01\x02")
        with open(bad, "rb") as f:
            r = client.post("/upload", files={"file": ("data.xlsx", f)})
        assert r.status_code == 422
        assert "Unsupported" in r.json()["detail"]
