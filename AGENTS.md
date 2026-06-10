# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

EmoSense is a single Python monolith: **FastAPI backend** (`:8000`) + **Gradio frontend** (`:7860`) for physiological emotion recognition via file upload and timed replay. There is no database or separate services.

### System prerequisites (one-time on fresh VMs)

Ubuntu images may need `python3-venv` before creating a virtualenv:

```bash
sudo apt-get install -y python3.12-venv
```

### Virtual environment and dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e emokit_stub/
pip install -r requirements.txt
pip install 'gradio>=5.0,<6.0' pytest ruff
```

- Use **`emokit_stub/`** (bundled) unless you have the full [EmoKit](https://github.com/beili16796/emokit) repo checked out at `external/emokit` and install with `pip install -e external/emokit`.
- Pin **Gradio 5.x**: `requirements.txt` allows Gradio 6.x, which breaks `gr.Timer(every=0.5)` in `emosense/app.py` and triggers localhost-check failures in cloud VMs.

### Demo checkpoints

Required for inference (gitignored). Generate once if `checkpoints/` is empty:

```bash
python scripts/generate_demo_checkpoints.py
```

### Running the application

**Option A — Backend + Gradio (recommended in cloud VMs)**

Terminal 1 (backend):

```bash
source .venv/bin/activate
PYTHONPATH=/workspace uvicorn emosense.backend.server:app --host 0.0.0.0 --port 8000 --log-level warning
```

Terminal 2 (Gradio; needs a one-line Timer compatibility patch):

```bash
source .venv/bin/activate
PYTHONPATH=/workspace python - <<'PY'
import gradio as gr
_orig = gr.Timer.__init__
def _patched(self, value=1, *, every=None, active=True, render=True, **kwargs):
    return _orig(self, every if every is not None else value, active=active, render=render)
gr.Timer.__init__ = _patched
from emosense.app import create_demo
create_demo().launch(server_name="0.0.0.0", server_port=7860, show_api=False)
PY
```

Open **http://localhost:7860**. Upload `demo_data/test_signal.csv` and click **Start Analysis**.

**Option B — Single command (may fail in cloud VMs)**

`python -m emosense` starts both services in one process but can fail on Gradio localhost checks without the Timer patch above.

### Lint and tests

Always set `PYTHONPATH=/workspace` (the package is not installed as an editable wheel):

```bash
source .venv/bin/activate
ruff check emosense/ tests/
PYTHONPATH=/workspace pytest tests/test_backend.py tests/test_file_parser.py tests/test_visualization.py -q
PYTHONPATH=/workspace pytest tests/ -q --ignore=tests/test_real_data.py
```

`tests/test_real_data.py` needs `EMOKIT_DATA_ROOT` pointing at DEAP/SEED-V datasets.

### Optional environment variables

| Variable | Purpose |
|---|---|
| `EMOKIT_DATA_ROOT` | Path to DEAP/SEED-V datasets for real-data tests and benchmarks |
| `PYTHONPATH=/workspace` | Required for imports when not using `pip install -e .` |

No API keys or auth secrets are used.
