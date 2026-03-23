# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Pipeline integration tests with mocked model inference."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client_with_engine() -> Generator[TestClient, None, None]:
    """``TestClient`` with a mocked processing engine."""
    import emosense.backend.server as server_mod

    with TestClient(server_mod.app) as tc:
        original_engine = server_mod.engine

        mock_mm = MagicMock()
        mock_mm.get_model_names.return_value = ["CNN-LSTM", "DGCNN"]
        mock_mm.get_active_model_name.return_value = "CNN-LSTM"

        mock_engine = MagicMock()
        mock_engine.model_manager = mock_mm
        server_mod.engine = mock_engine

        try:
            yield tc
        finally:
            server_mod.engine = original_engine


def test_model_switch_mid_session(client_with_engine: TestClient) -> None:
    """Switching active model returns 200 and updated name."""
    r = client_with_engine.post("/models/active", json={"name": "DGCNN"})
    assert r.status_code == 200
    assert r.json()["active_model"] == "DGCNN"


def test_websocket_handshake(client_with_engine: TestClient) -> None:
    """WebSocket endpoint accepts connections."""
    with client_with_engine.websocket_connect("/ws") as ws:
        ws.send_text("ping")
