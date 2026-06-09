# EmoSense — Interactive Physiological Emotion Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/beili16796/emosense/actions/workflows/ci.yml/badge.svg)](https://github.com/beili16796/emosense/actions)

EmoSense is a web-based demonstration system for dataset replay and optional
live-input physiological emotion recognition, built on top of the
[EmoKit](https://github.com/beili16796/emokit) open-source toolkit.

## Features

- **Dataset replay**: accepts DEAP `.dat`, SEED/SEED-V `.mat`, `.npz`, and generic `.csv`
- **Six pre-loaded models**: CNN-LSTM, DGCNN, Transformer-MM, BiDAE, DGCCA-AM, PR-PL
- **Three visualization panels**: V-A trajectory, EEG topographic map, modality contribution
- **Timed replay mode**: emits windows at a configurable interval for demo videos
- **Optional LSL receivers**: backend modules for Muse/OpenBCI-style hardware demos
- **Interactive model switching** with shared feature cache

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e ../emokit  # or install emokit from PyPI

# Generate demo checkpoints (random weights — for testing)
python scripts/generate_demo_checkpoints.py

# Start the server (FastAPI on :8000, Gradio on :7860)
python -m emosense
```

Open http://localhost:7860 in your browser.

The default review path is file upload plus timed replay. The optional video
widget is a visual reference only; the current Gradio frontend does not analyse
video frames.

## With Real EmoKit Checkpoints

After running EmoKit experiments:

```bash
# Export trained checkpoints
python -m emokit.scripts.export_demo_checkpoints \
    --results ../emokit/results/paper_experiments \
    --emosense-dir ./checkpoints

# Restart the server
python -m emosense
```

If no real checkpoints are mounted, the UI displays a warning that predictions
come from demo/random weights.

## Demo Data

Generate lightweight files for smoke tests:

```bash
python scripts/create_test_mat.py --format deap --realistic --output demo_data/test_deap_realistic.mat
python scripts/create_test_mat.py --format seedv --realistic --output demo_data/test_seedv_realistic.npz
```

The repository may only include small text fixtures by default. Generated `.mat`
and `.npz` files are intentionally reproducible from `scripts/create_test_mat.py`
so reviewers can create them locally.

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

The demo container installs `emokit_stub` for lightweight UI smoke tests. Mount
the full EmoKit package and real checkpoints when you need meaningful model
predictions.

## Benchmarking

```bash
# Latency benchmark. Report hardware and whether feature extraction is included.
python scripts/benchmark_latency.py --n-warmup 20 --n-measure 200

# With real DEAP data
python scripts/benchmark_latency.py --data-path $EMOKIT_DATA_ROOT/DEAP/s01.dat
```

## Testing

```bash
# Unit and integration tests
pytest tests/ -q

# Real-data smoke tests (requires EMOKIT_DATA_ROOT)
pytest tests/test_real_data.py -v -m "real_data"

# Stress test (requires running server + real .dat file)
python scripts/stress_test_session.py \
    --dat-file $EMOKIT_DATA_ROOT/DEAP/s01.dat \
    --n-iterations 30
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Upload a signal file |
| POST | `/process/{task_id}` | Start processing |
| GET | `/results/latest` | Poll inference results |
| POST | `/models/active` | Switch active model |
| GET | `/models` | List available models |
| POST | `/cancel/{task_id}` | Cancel processing |
| GET | `/health` | Basic health check |
| GET | `/health/detailed` | Extended diagnostics |
| POST | `/admin/reset` | Reset session state |
| WS | `/ws` | Optional WebSocket stream for custom clients |

## License

MIT License. See [LICENSE](LICENSE).
