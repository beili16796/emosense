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

from emosense.backend.server import (
    CANCELLED_TASKS,
    COMPLETED_TASKS,
    RESULTS_STORE,
    SESSION_STORE,
    app,
)


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


def _make_mock_deap_mat(directory: Path) -> Path:
    """Create a minimal valid DEAP .mat file with synthetic data."""
    import scipy.io

    rng = np.random.default_rng(42)
    data = rng.standard_normal((4, 40, 8064)).astype(np.float32)
    labels = np.zeros((4, 4), dtype=np.float32)
    labels[:, 0] = [2.0, 7.0, 3.0, 8.0]
    labels[:, 1] = [6.0, 4.0, 8.0, 2.0]
    path = directory / "1.mat"
    scipy.io.savemat(str(path), {"data": data, "labels": labels})
    return path


@pytest.fixture()
def client():
    """Provide a TestClient for the FastAPI app."""
    SESSION_STORE.clear()
    RESULTS_STORE.clear()
    COMPLETED_TASKS.clear()
    CANCELLED_TASKS.clear()
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def mock_dat(tmp_path: Path) -> Path:
    """Create a mock DEAP .dat file in a temp directory."""
    return _make_mock_deap_dat(tmp_path)


@pytest.fixture()
def mock_mat(tmp_path: Path) -> Path:
    """Create a mock DEAP .mat file in a temp directory."""
    return _make_mock_deap_mat(tmp_path)


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

    def test_upload_deap_mat_returns_metadata(
        self, client: TestClient, mock_mat: Path
    ) -> None:
        with open(mock_mat, "rb") as f:
            r = client.post(
                "/upload",
                files={"file": (mock_mat.name, f, "application/octet-stream")},
                data={"window_sec": "4.0", "overlap": "0.5", "model_name": "DGCNN"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "task_id" in body
        assert "DEAP" in body["format_detected"]
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


class TestPaperClaims:
    """Each method checks one concrete paper/system claim."""

    def test_file_upload_no_hardware_required(
        self, client: TestClient, mock_dat: Path
    ) -> None:
        with open(mock_dat, "rb") as f:
            r = client.post("/upload", files={"file": (mock_dat.name, f)})
        assert r.status_code == 200
        r2 = client.get("/stream/connect")
        assert r2.status_code in (404, 405)

    def test_four_second_windows_with_50pct_overlap(
        self, client: TestClient, mock_dat: Path
    ) -> None:
        with open(mock_dat, "rb") as f:
            info = client.post(
                "/upload",
                files={"file": (mock_dat.name, f)},
                data={"window_sec": "4.0", "overlap": "0.5"},
            ).json()
        assert info["estimated_segments"] > 50

    def test_six_models_all_switchable(self, client: TestClient) -> None:
        r = client.get("/models")
        models = {m["name"] for m in r.json()}
        expected = {
            "CNN-LSTM",
            "DGCNN",
            "BiDAE",
            "Transformer-MM",
            "DGCCA-AM",
            "PR-PL",
        }
        assert expected.issubset(models)
        for name in expected:
            resp = client.post("/models/active", json={"name": name})
            assert resp.status_code == 200

    def test_latency_sub_300ms_p99(self) -> None:
        from scripts.benchmark_latency import run_benchmark

        results = run_benchmark(
            data_path=None,
            n_warmup=5,
            n_measure=30,
            use_real_data=False,
            output_path="/tmp/emosense_latency_test.json",
        )
        valid = [v for v in results.values() if "p99_ms" in v]
        assert valid, "No valid latency results produced"
        for metrics in valid:
            assert metrics["p99_ms"] < 300

    def test_feature_cache_switching_overhead(
        self, client: TestClient, mock_dat: Path
    ) -> None:
        with open(mock_dat, "rb") as f:
            task_id = client.post(
                "/upload", files={"file": (mock_dat.name, f)}
            ).json()["task_id"]

        client.post(f"/process/{task_id}")
        time.sleep(1.5)
        client.post("/models/active", json={"name": "CNN-LSTM"})
        t0 = time.time()
        client.post(f"/process/{task_id}")
        time.sleep(1.5)
        elapsed = time.time() - t0
        assert elapsed < 8

    def test_three_visualization_panels_in_results(
        self, client: TestClient, mock_dat: Path
    ) -> None:
        with open(mock_dat, "rb") as f:
            task_id = client.post(
                "/upload", files={"file": (mock_dat.name, f)}
            ).json()["task_id"]
        client.post(f"/process/{task_id}")
        time.sleep(2.5)
        r = client.get(f"/results/latest?task_id={task_id}")
        results = r.json().get("results", [])
        assert results
        first = results[0]
        assert "valence" in first and "arousal" in first
        assert -1.0 <= first["valence"] <= 1.0
        assert "de_features" in first
        assert len(first["de_features"][0]) == 5
        assert "attention_weights" in first
