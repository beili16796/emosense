# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""FastAPI backend server for EmoSense — file-upload driven."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.responses import JSONResponse

from emosense.backend.file_parser import FileParser
from emosense.backend.processing_engine import InferenceResult, ProcessingEngine

logger = logging.getLogger(__name__)

app = FastAPI(title="EmoSense API", version="0.2.0")

SESSION_STORE: dict[str, dict[str, Any]] = {}
RESULTS_STORE: dict[str, list[dict]] = {}
connected_clients: list[WebSocket] = []
engine: ProcessingEngine | None = None


@app.on_event("startup")
async def startup() -> None:
    """Initialise model manager and processing engine."""
    global engine
    from emosense.backend.inference import ModelManager
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "models.yaml"
    if config_path.exists():
        manager = ModelManager(str(config_path))
    else:
        manager = _DummyModelManager()
    engine = ProcessingEngine(manager)
    logger.info("EmoSense API started.")


class _DummyModelManager:
    """Fallback when no model config is available."""
    def __init__(self) -> None:
        self._active = "none"
        self._names: list[str] = []
    def set_active_model(self, name: str) -> None:
        self._active = name
    def get_active_model(self) -> Any:
        return None
    def get_active_model_name(self) -> str:
        return self._active
    def get_model_names(self) -> list[str]:
        return self._names


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    window_sec: float = 4.0,
    overlap: float = 0.5,
    model_name: str = "DGCNN",
) -> dict:
    """Upload a physiological signal file and get metadata.

    Supported formats: .dat (DEAP), .mat (SEED/SEED-V), .csv, .bdf
    """
    suffix = Path(file.filename or "unknown.dat").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        parsed = FileParser.parse(tmp_path)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))

    eeg = parsed.get("eeg", np.empty((0, 0, 0)))
    n_trials = eeg.shape[0] if eeg.ndim >= 1 else 0
    fs = parsed["fs"]
    win_samples = int(window_sec * fs)
    step = int(win_samples * (1 - overlap))
    n_samples = eeg.shape[-1] if eeg.ndim >= 3 else 0
    n_windows = max(0, (n_samples - win_samples) // step + 1) if n_samples > 0 else 0

    task_id = str(uuid.uuid4())[:8]
    SESSION_STORE[task_id] = {
        "parsed": parsed,
        "tmp_path": str(tmp_path),
        "window_sec": window_sec,
        "overlap": overlap,
        "model_name": model_name,
    }
    RESULTS_STORE[task_id] = []

    return {
        "task_id": task_id,
        "n_trials": n_trials,
        "estimated_segments": n_trials * n_windows,
        "format_detected": parsed["format"],
        "fs": fs,
        "n_channels": len(parsed.get("ch_names", [])),
    }


@app.post("/process/{task_id}")
async def start_processing(task_id: str, background_tasks: BackgroundTasks) -> dict:
    """Start background processing of an uploaded file."""
    if task_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    background_tasks.add_task(_run_processing, task_id)
    return {"status": "processing_started", "task_id": task_id}


async def _run_processing(task_id: str) -> None:
    """Background task: process file and broadcast results."""
    ctx = SESSION_STORE[task_id]
    parsed = ctx["parsed"]

    try:
        assert engine is not None
        for result in engine.process_file(
            parsed,
            window_sec=ctx["window_sec"],
            overlap=ctx["overlap"],
            model_name=ctx["model_name"],
        ):
            msg = _result_to_dict(result, task_id)
            RESULTS_STORE.setdefault(task_id, []).append(msg)
            await _broadcast(json.dumps(msg))

        await _broadcast(json.dumps({"type": "processing_complete", "task_id": task_id}))
    except Exception as e:
        logger.exception("Processing failed for task %s", task_id)
        await _broadcast(json.dumps({"type": "error", "task_id": task_id, "message": str(e)}))
    finally:
        Path(ctx["tmp_path"]).unlink(missing_ok=True)


def _result_to_dict(r: InferenceResult, task_id: str) -> dict:
    return {
        "type": "inference",
        "task_id": task_id,
        "valence": round(r.valence, 4),
        "arousal": round(r.arousal, 4),
        "label": r.label,
        "confidence": round(r.confidence, 4),
        "proba": r.proba.tolist(),
        "de_features": r.de_features.tolist(),
        "attention_weights": r.attention_weights.tolist() if r.attention_weights is not None else None,
        "model_name": r.model_name,
        "latency_ms": round(r.latency_ms, 2),
        "trial_idx": r.trial_idx,
        "window_idx": r.window_idx,
        "time_start_sec": round(r.time_start_sec, 2),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """WebSocket endpoint for streaming results to frontend."""
    await ws.accept()
    connected_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        pass
    finally:
        if ws in connected_clients:
            connected_clients.remove(ws)


async def _broadcast(message: str) -> None:
    dead: list[WebSocket] = []
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            dead.append(client)
    for d in dead:
        if d in connected_clients:
            connected_clients.remove(d)


@app.get("/results/latest")
async def get_latest_results(task_id: str | None = None) -> dict:
    """Poll latest results for a task."""
    if task_id and task_id in RESULTS_STORE:
        return {"results": RESULTS_STORE[task_id]}
    all_results: list[dict] = []
    for results in RESULTS_STORE.values():
        all_results.extend(results)
    return {"results": all_results[-50:]}


@app.get("/models")
async def list_models() -> list[dict]:
    """List available models."""
    if engine is None:
        return []
    names = engine.model_manager.get_model_names()
    return [{"name": n} for n in names]


@app.post("/models/active")
async def set_active_model(body: dict) -> dict:
    """Switch the active model."""
    if engine is None:
        raise HTTPException(500, "Engine not initialized")
    name = body.get("name", "")
    try:
        engine.model_manager.set_active_model(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"active_model": name}


@app.get("/health")
async def health() -> dict:
    """Health check."""
    model_name = engine.model_manager.get_active_model_name() if engine else "none"
    return {"status": "ok", "active_model": model_name}
