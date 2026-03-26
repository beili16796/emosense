# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""FastAPI backend server for EmoSense — file-upload driven."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import tempfile
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket

from emosense.backend.file_parser import FileParser
from emosense.backend.processing_engine import InferenceResult, ProcessingEngine

logger = logging.getLogger(__name__)

app = FastAPI(title="EmoSense API", version="0.3.0")

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB hard limit

SESSION_STORE: dict[str, dict[str, Any]] = {}
RESULTS_STORE: dict[str, list[dict]] = {}
RESULT_BUFFER: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
COMPLETED_TASKS: dict[str, bool] = {}
CANCELLED_TASKS: dict[str, bool] = {}
connected_clients: list[WebSocket] = []
engine: ProcessingEngine | None = None


@app.on_event("startup")
async def startup() -> None:
    """Initialise model manager and processing engine."""
    global engine
    from emosense.backend.inference import ModelManager
    config_path = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
    if not config_path.exists():
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
    def get_active_model_axis(self) -> str:
        return "valence"
    def get_model_info(self) -> list[dict[str, Any]]:
        return []


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
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(content) / 1024 / 1024:.0f} MB). "
                f"Maximum allowed: {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB.",
            )
        if len(content) == 0:
            raise HTTPException(status_code=422, detail="Uploaded file is empty.")
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        parsed = FileParser.parse(tmp_path)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))

    fs = parsed.get("fs", 0)
    if fs <= 0 or fs > 10000:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"Invalid sampling rate: {fs} Hz. Expected 1-10000 Hz.",
        )
    n_ch = len(parsed.get("ch_names", []))
    if n_ch == 0 and not parsed.get("pre_extracted"):
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="No EEG channels detected in file.")

    file_hash = hashlib.sha256(content).hexdigest()[:16]

    eeg = parsed.get("eeg")
    if eeg is not None and hasattr(eeg, "shape") and eeg.ndim >= 1:
        n_trials = eeg.shape[0]
    elif parsed.get("pre_extracted") and parsed.get("eeg_de") is not None:
        de = parsed["eeg_de"]
        n_trials = de.shape[0] if hasattr(de, "shape") else len(de)
    else:
        n_trials = 0

    fs = parsed["fs"]
    win_samples = int(window_sec * fs)
    step = int(win_samples * (1 - overlap))

    if parsed.get("pre_extracted") and parsed.get("eeg_de") is not None:
        de = parsed["eeg_de"]
        if isinstance(de, list):
            estimated_segments = sum(
                t.shape[0] for t in de if hasattr(t, "shape")
            )
        elif hasattr(de, "shape"):
            estimated_segments = de.shape[0]
        else:
            estimated_segments = 0
    elif eeg is not None and hasattr(eeg, "shape") and eeg.ndim >= 3:
        n_samples = eeg.shape[-1]
        n_windows = max(0, (n_samples - win_samples) // step + 1) if n_samples > 0 else 0
        estimated_segments = n_trials * n_windows
    else:
        estimated_segments = 0

    task_id = str(uuid.uuid4())[:8]
    SESSION_STORE[task_id] = {
        "parsed": parsed,
        "tmp_path": str(tmp_path),
        "window_sec": window_sec,
        "overlap": overlap,
        "model_name": model_name,
        "file_hash": file_hash,
    }
    RESULTS_STORE[task_id] = []
    COMPLETED_TASKS[task_id] = False
    CANCELLED_TASKS[task_id] = False

    return {
        "task_id": task_id,
        "n_trials": n_trials,
        "estimated_segments": estimated_segments,
        "format_detected": parsed["format"],
        "fs": fs,
        "n_channels": len(parsed.get("ch_names", [])),
        "file_hash": file_hash,
    }


@app.post("/upload/hash")
async def get_file_hash(file: UploadFile = File(...)) -> dict:
    """Compute file hash for feature cache key."""
    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()[:16]
    return {"hash": file_hash, "size_bytes": len(content)}


@app.post("/process/{task_id}")
async def start_processing(
    task_id: str,
    background_tasks: BackgroundTasks,
    stream_interval: float = 0.0,
) -> dict:
    """Start background processing of an uploaded file.

    Args:
        stream_interval: Seconds to wait between emitting each window result.
            Set to 0.5 for real-time BCI simulation (~2 windows/sec).
            Default 0.0 = process as fast as possible.
    """
    if task_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    COMPLETED_TASKS[task_id] = False
    CANCELLED_TASKS[task_id] = False
    SESSION_STORE[task_id]["stream_interval"] = max(0.0, min(stream_interval, 5.0))
    background_tasks.add_task(_run_processing, task_id)
    return {"status": "processing_started", "task_id": task_id}


@app.post("/cancel/{task_id}")
async def cancel_task(task_id: str) -> dict:
    """Cancel an ongoing processing task."""
    if task_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    CANCELLED_TASKS[task_id] = True
    return {"status": "cancelled", "task_id": task_id}


async def _run_processing(task_id: str) -> None:
    """Background task: process file and broadcast results."""
    ctx = SESSION_STORE.get(task_id)
    if ctx is None:
        logger.warning("Task %s not found in SESSION_STORE", task_id)
        COMPLETED_TASKS[task_id] = True
        return
    parsed = ctx["parsed"]

    n_errors = 0
    max_consecutive_errors = 5
    stream_interval = ctx.get("stream_interval", 0.0)

    try:
        assert engine is not None
        for result in engine.process_file(
            parsed,
            window_sec=ctx["window_sec"],
            overlap=ctx["overlap"],
            model_name=ctx["model_name"],
            file_hash=ctx.get("file_hash"),
        ):
            if CANCELLED_TASKS.get(task_id, False):
                logger.info("Task %s cancelled by user", task_id)
                COMPLETED_TASKS[task_id] = True
                await _broadcast(json.dumps({"type": "cancelled", "task_id": task_id}))
                return
            try:
                msg = _result_to_dict(result, task_id)
                RESULTS_STORE.setdefault(task_id, []).append(msg)
                RESULT_BUFFER[task_id].append(msg)
                await _broadcast(json.dumps(msg))
                n_errors = 0
            except Exception as win_err:
                n_errors += 1
                logger.warning("Window serialization error (task %s): %s", task_id, win_err)
                if n_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"Too many consecutive errors ({n_errors}), aborting task"
                    ) from win_err
            if stream_interval > 0:
                await asyncio.sleep(stream_interval)

        COMPLETED_TASKS[task_id] = True
        await _broadcast(json.dumps({"type": "processing_complete", "task_id": task_id}))
    except Exception as e:
        logger.exception("Processing failed for task %s", task_id)
        COMPLETED_TASKS[task_id] = True
        err_msg = str(e)[:500]
        await _broadcast(json.dumps({"type": "error", "task_id": task_id, "message": err_msg}))
    finally:
        try:
            Path(ctx["tmp_path"]).unlink(missing_ok=True)
        except Exception:
            pass


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
async def get_latest_results(
    task_id: str | None = None,
    since_idx: int = 0,
) -> dict:
    """Return buffered inference results with incremental polling.

    Args:
        task_id: current task (optional; returns all recent if None).
        since_idx: return only results at index >= since_idx.

    Returns:
        ``{'results': [...], 'next_idx': int, 'is_complete': bool}``
    """
    if task_id and task_id in RESULTS_STORE:
        buf = RESULTS_STORE[task_id]
        slice_ = buf[since_idx:]
        return {
            "results": slice_,
            "next_idx": since_idx + len(slice_),
            "is_complete": COMPLETED_TASKS.get(task_id, False),
        }
    return {"results": [], "next_idx": 0, "is_complete": False}


@app.get("/models")
async def list_models() -> list[dict]:
    """List available models."""
    if engine is None:
        return []
    if hasattr(engine.model_manager, "get_model_info"):
        return engine.model_manager.get_model_info()
    names = engine.model_manager.get_model_names()
    return [{"name": n} for n in names]


@app.post("/models/active")
async def set_active_model(body: dict) -> dict:
    """Switch the active model (safe during processing — uses cache)."""
    if engine is None:
        raise HTTPException(500, "Engine not initialized")
    name = body.get("name", "")
    if not name:
        raise HTTPException(status_code=422, detail="Model name required")
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


@app.get("/health/detailed")
async def health_detailed() -> dict:
    """Detailed health check with model weight status."""
    if engine is None:
        return {"status": "not_initialized"}

    model_names = engine.model_manager.get_model_names()
    active = engine.model_manager.get_active_model_name()
    n_sessions = len(SESSION_STORE)
    n_results = sum(len(v) for v in RESULTS_STORE.values())

    real_weights_count = 0
    model_details: list[dict] = []
    for name in model_names:
        has_real = False
        try:
            cfg = engine.model_manager._model_configs.get(name, {})
            ckpt_path = Path(cfg.get("checkpoint", ""))
            has_real = ckpt_path.exists() and ckpt_path.stat().st_size > 1024
        except Exception:
            pass
        if has_real:
            real_weights_count += 1
        model_details.append({"name": name, "has_real_weights": has_real})

    result: dict = {
        "status": "ok",
        "active_model": active,
        "available_models": model_details,
        "models_with_real_weights": real_weights_count,
        "active_sessions": n_sessions,
        "total_results": n_results,
    }

    if real_weights_count == 0 and model_names:
        result["warning"] = f"{len(model_names)} models using random weights — predictions will be meaningless"

    return result


@app.post("/admin/reset")
async def admin_reset() -> dict:
    """Reset all session state for clean user-study transitions.

    Clears result buffers, task registries, and resets the active model
    to the default (DGCNN). Does not reload model weights.
    """
    from datetime import datetime, timezone

    SESSION_STORE.clear()
    RESULTS_STORE.clear()
    connected_clients.clear()

    RESULT_BUFFER.clear()
    COMPLETED_TASKS.clear()
    CANCELLED_TASKS.clear()

    if engine is not None:
        try:
            engine.model_manager.set_active_model("DGCNN")
        except (KeyError, AttributeError):
            pass

    logger.info("Admin reset performed")
    return {
        "status": "reset",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _get_emokit_version() -> str:
    try:
        import emokit

        return getattr(emokit, "__version__", "unknown")
    except ImportError:
        return "not installed"
