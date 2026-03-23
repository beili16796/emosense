# AGENTS.md

## Cursor Cloud specific instructions

### Overview

EmoSense is a Gradio + FastAPI application for real-time EEG/physiological signal analysis and emotion recognition. It runs as a single Python process launching both:
- **FastAPI backend** on port 8000 (REST API + WebSocket)
- **Gradio frontend** on port 7860 (interactive dashboard)

### External dependency: `emokit`

The project depends on `emokit`, a sibling ML library not included in this repo. A development stub lives at `/workspace/emokit_stub/` providing the required interfaces (`emokit.models.build_model`, `emokit.features.FeaturePipeline`, `emokit.features.eeg.DEExtractor`, `emokit.utils.set_seed`). The stub is installed via `pip install -e /workspace/emokit_stub/`.

### Gradio compatibility

The codebase uses `gr.Timer(every=0.5)` but no released Gradio version uses the `every` keyword — the parameter is `value` in Gradio 4.37+. After installing Gradio, a patch is applied to the installed `gradio/components/timer.py` to accept `every` as an alias for `value`. Additionally, `gradio_client/utils.py` requires a patch to handle Pydantic v2 JSON schemas (bool-typed `additionalProperties`). The update script handles both patches automatically.

Use `gradio==4.42.0` and `huggingface_hub<0.25` for compatibility.

### Config symlink

The backend resolves `config/models.yaml` relative to the repo root, but the actual file is at `emosense/config/models.yaml`. A symlink `config -> emosense/config` is needed at the workspace root.

### Missing `get_active_model_name()` method

`ModelManager` in `emosense/backend/inference.py` is missing a `get_active_model_name()` method that `server.py` and `processing_engine.py` call. The method was added during setup.

### Running the application

```bash
python3 -m emosense.app
```

Starts both FastAPI (port 8000) and Gradio (port 7860). Ensure checkpoints exist first:

```bash
python3 scripts/generate_demo_checkpoints.py
```

### Running tests

```bash
python3 -m pytest tests/ -v
```

47 tests pass, 4 skip (TopoMapPlot tests that require specific MNE channel montage matching).

### Linting

```bash
python3 -m ruff check emosense/ tests/
```

11 pre-existing F401 (unused import) warnings in the original code.
