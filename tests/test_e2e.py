# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""End-to-end tests for the EmoSense file-upload pipeline."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from emosense.backend.server import app, engine, SESSION_STORE, RESULTS_STORE


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
        self, client: TestClient, tmp_path: Path
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
