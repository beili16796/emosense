# Demo Data

Synthetic signal fixtures for testing EmoSense without real DEAP/SEED-V data.

## Committed fixture

| File | Format | Description |
|------|--------|-------------|
| `test_signal.csv` | CSV | Small smoke-test stream (32 channels, 128 Hz) |

## Generate larger demo files locally

Large `.mat` / `.npz` binaries are intentionally **not** committed. Reproduce them with:

```bash
# DEAP-style smoke test
python scripts/create_test_mat.py --format deap --output demo_data/test_deap.mat

# DEAP with realistic frontal-asymmetry profiles (recommended for UI demos)
python scripts/create_test_mat.py --format deap --realistic --output demo_data/test_deap_realistic.mat

# SEED-V smoke test
python scripts/create_test_mat.py --format seedv --output demo_data/test_seedv.npz

# SEED-V with emotion-template DE profiles
python scripts/create_test_mat.py --format seedv --realistic --output demo_data/test_seedv_realistic.npz

# DREAMER .mat (optional parser smoke test)
python scripts/create_test_mat.py --format dreamer --output demo_data/test_dreamer.mat
```

## Usage

1. Start EmoSense: `python -m emosense`
2. Open http://localhost:7860
3. Upload a file (or use `test_signal.csv` for a quick smoke test)
4. Select a model and click **Start Analysis**
