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

### Realistic demo data

Use `demo_data/test_deap_realistic.mat` and `demo_data/test_seedv_realistic.npz` for meaningful visualisations (frontal asymmetry, emotion-template DE). Generate with `python3 scripts/create_test_mat.py --format deap --realistic --output demo_data/test_deap_realistic.mat`.

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

107 pass, 12 skip, 0 failures. All tests pass cleanly.

### Linting

```bash
python3 -m ruff check emosense/ tests/
```

0 errors or warnings after lint cleanup.

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
All lint warnings have been fixed.
