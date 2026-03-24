# EmoSense — Interactive Physiological Emotion Analysis

EmoSense is a web-based demonstration system for real-time multimodal
physiological emotion recognition, built on top of the
[EmoKit](../emokit/) open-source toolkit.

## Features

- **File upload**: accepts DEAP `.dat`, SEED/SEED-V `.mat`, and generic `.csv`
- **Six pre-loaded models**: CNN-LSTM, DGCNN, Transformer-MM, BiDAE, DGCCA-AM, PR-PL
- **Three visualization panels**: V-A trajectory, EEG topographic map, modality contribution
- **Sub-300ms inference** on consumer CPU (Intel i7, 16 GB RAM)
- **Interactive model switching** with shared feature cache (<5 ms overhead)

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

## Benchmarking

```bash
# Latency benchmark (paper claim: p99 < 300ms)
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
| WS | `/ws` | WebSocket for streaming results |

## License

MIT License. See [LICENSE](LICENSE).
