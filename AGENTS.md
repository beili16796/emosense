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

Use `gradio==4.42.0` and `huggingface_hub<0.25` for compatibility. Additionally, `fastapi==0.111.0` and `pydantic<2.10` are required — newer versions of FastAPI/Starlette/Pydantic break Gradio's internal route serialization. The system Jinja2 (3.1.2) is too old; ensure `jinja2>=3.1.4` is installed via pip.

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

94 pass, 12 skip, 2 pre-existing failures (`test_get_active_model_name` expects "none" but model auto-loads; `test_latency_sub_300ms_p99` missing `import torch` in benchmark script). The `tests/test_server.py` file has a syntax error (empty method body at line 137) and is excluded from collection; run with `--ignore=tests/test_server.py` to avoid the collection error.

### Linting

```bash
python3 -m ruff check emosense/ tests/
```

Pre-existing F401 (unused import) warnings in the original code.

### File format support

The parser supports: `.dat` (DEAP pickle), `.mat` (auto-detects DEAP vs SEED), `.npz` (SEED-V DE features), `.csv`, `.bdf`. Synthetic test files are in `demo_data/` and can be regenerated via `scripts/create_test_mat.py`.

### API endpoints

- `GET /health` — basic health check
- `GET /health/detailed` — model weight status, session count, warnings
- `POST /admin/reset` — clear all sessions and results
- `GET /models` — list available models
- `POST /models/active` — switch active model
- `POST /upload` — upload signal file
- `POST /process/{task_id}` — start analysis
- `GET /results/latest` — poll results

### Benchmark

```bash
PYTHONPATH=/workspace python3 scripts/benchmark_latency.py --n-warmup 20 --n-measure 100
```

LaTeX table saved to `results/latency_benchmark.tex`.
11 pre-existing F401 (unused import) warnings in the original code.
