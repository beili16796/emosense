# Demo Data

Synthetic signal files for testing EmoSense without real DEAP/SEED-V data.

## Files

| File | Format | Trials | Channels | Description |
|------|--------|--------|----------|-------------|
| `test_deap.mat` | DEAP .mat | 4 | 32 EEG + 8 peripheral | Random noise — fast smoke-test only |
| `test_deap_realistic.mat` | DEAP .mat | 4 | 32 EEG + 8 peripheral | Realistic: alpha-band dominance, frontal asymmetry, 1/f spectral slope, GSR ramps |
| `test_seedv.npz` | SEED-V .npz | variable | 62 (DE) | Random noise — fast smoke-test only |
| `test_seedv_realistic.npz` | SEED-V .npz | 10 × 50 windows | 62 × 5 (DE) | Realistic: emotion-template DE profiles (happy=high gamma, sad=high delta, etc.) |

## Realistic DEAP (`test_deap_realistic.mat`)

- **4 trials** with ground-truth valence labels `[7, 3, 8, 2]`:
  - Trials 0 & 2 (valence > 5): positive — alpha DE is higher at F4 vs F3
  - Trials 1 & 3 (valence < 5): negative — alpha DE is higher at F3 vs F4
- EEG channels have 1/f spectral slope plus band-specific sine-wave components
- GSR channel (index 36) has a slow-rising conductance with occasional peaks
- Shape: `data` (4, 40, 8064), `labels` (4, 4) — same as real DEAP

This file demonstrates:
- Frontal alpha asymmetry annotation in the topomap panel
- Meaningful V-A trajectory clustering by valence polarity
- Correct binary label splits for DEAP-trained models

## Realistic SEED-V (`test_seedv_realistic.npz`)

- **10 trials**, each 50 windows of shape (50, 310) reshaped to (50, 62, 5)
- 5 emotion classes (2 trials each): happy, sad, neutral, fear, disgust
- Each emotion has a distinctive DE band profile:
  - Happy: high gamma, moderate beta
  - Sad: high delta, moderate theta
  - Fear: high gamma + beta, low alpha
  - Disgust: high delta + theta
  - Neutral: uniform across bands

This file demonstrates:
- 5-class emotion labels (Happy/Sad/Neutral/Fear/Disgust)
- 62-channel topomap layout
- Transformer-MM model switching

## Generation

```bash
# Random noise (legacy)
python scripts/create_test_mat.py --format deap --output demo_data/test_deap.mat
python scripts/create_test_mat.py --format seedv --output demo_data/test_seedv.npz

# Realistic signals
python scripts/create_test_mat.py --format deap --realistic --output demo_data/test_deap_realistic.mat
python scripts/create_test_mat.py --format seedv --realistic --output demo_data/test_seedv_realistic.npz
```

## Usage

1. Start EmoSense: `python -m emosense.app`
2. Open http://localhost:7860
3. Upload a file, select a model, click "Start Analysis"
4. Use realistic files for meaningful visualisation demos
