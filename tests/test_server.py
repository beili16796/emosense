# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Unit tests for the EmoSense FastAPI server."""

from __future__ import annotations

from typing import Any, Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    """Provide a TestClient with mocked backend dependencies."""
    import emosense.backend.server as server_mod

    with TestClient(server_mod.app) as tc:
        original_engine = server_mod.engine

        mock_mm = MagicMock()
        mock_mm.get_model_names.return_value = ["CNN-LSTM", "DGCCA-AM"]
        mock_mm.get_active_model_name.return_value = "CNN-LSTM"

        mock_engine = MagicMock()
        mock_engine.model_manager = mock_mm
        server_mod.engine = mock_engine

        try:
            yield tc
        finally:
            server_mod.engine = original_engine


# ======================================================================
# GET /health
# ======================================================================


class TestHealth:
    """Tests for ``GET /health``."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["active_model"] == "CNN-LSTM"


# ======================================================================
# GET /models
# ======================================================================


class TestModels:
    """Tests for ``GET /models``."""

    def test_returns_list(self, client: TestClient) -> None:
        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        names = [m["name"] for m in data]
        assert "CNN-LSTM" in names
        assert "DGCCA-AM" in names


# ======================================================================
# POST /models/active
# ======================================================================


class TestSetActiveModel:
    """Tests for ``POST /models/active``."""

    def test_switches_model(self, client: TestClient) -> None:
        resp = client.post(
            "/models/active", json={"name": "DGCCA-AM"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_model"] == "DGCCA-AM"

    def test_unknown_model_returns_404(self, client: TestClient) -> None:
        import emosense.backend.server as server_mod

        server_mod.engine.model_manager.set_active_model.side_effect = (
            KeyError("Model 'Nonexistent' not in config")
        )
        resp = client.post(
            "/models/active", json={"name": "Nonexistent"},
        )
        assert resp.status_code == 404
        server_mod.engine.model_manager.set_active_model.side_effect = None


# ======================================================================
# POST /admin/reset
# ======================================================================


class TestAdminReset:
    """Tests for ``POST /admin/reset``."""

    def test_reset_returns_status(self, client: TestClient) -> None:
        resp = client.post("/admin/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "reset"
        assert "timestamp" in data


# ======================================================================
# GET /health/detailed
# ======================================================================


class TestHealthDetailed:
    """Tests for ``GET /health/detailed``."""

    def test_returns_detailed_info(self, client: TestClient) -> None:
        import emosense.backend.server as server_mod

        server_mod.engine.model_manager._model_configs = {
            "CNN-LSTM": {"checkpoint": "nonexistent.pt"},
        }
        resp = client.get("/health/detailed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "active_model" in data
        assert "models_with_real_weights" in data
