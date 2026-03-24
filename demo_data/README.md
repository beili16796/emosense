# Demo Data — Pre-selected DEAP Trials

This directory contains 3 pre-selected single-trial `.dat` files extracted from
DEAP subject 01, chosen to clearly demonstrate different emotion quadrants in
Russell's Circumplex Model.

## Trials

| File | Quadrant | Valence | Arousal | Expected V-A Region |
|------|----------|---------|---------|---------------------|
| `trial_happy_s01_tXX.dat` | High V + High A | >6 | >6 | Top-right (excitement) |
| `trial_angry_s01_tXX.dat` | Low V + High A  | <4 | >6 | Top-left (anger/fear) |
| `trial_sad_s01_tXX.dat`   | Low V + Low A   | <4 | <4 | Bottom-left (sadness) |

## Generation

```bash
python scripts/extract_demo_trials.py \
    --deap-root $EMOKIT_DATA_ROOT/DEAP \
    --output demo_data/
```

## Usage in EmoSense UI

1. Start the EmoSense server: `python -m emosense`
2. Open http://localhost:7860
3. Upload one of the trial `.dat` files
4. Click "Start Analysis" with the default DGCNN model
5. Observe the V-A trajectory staying in the expected quadrant
6. Switch models (e.g., DGCCA-AM) to compare predictions

## User Study Protocol

The paper's user study (Section 3) uses these 3 trials:
- 5 min introduction
- 10 min free exploration with these 3 trials (one per emotion quadrant)
- 5 min post-session questionnaire (SUS + custom items)

Between participants, call `POST /admin/reset` or click the Reset button
to clear all session state without restarting the server.

## File Format

Each `.dat` file is a Python pickle containing:
- `data`: shape `(1, 40, 8064)` — 1 trial, 40 channels, 63s @128Hz
- `labels`: shape `(1, 4)` — valence, arousal, dominance, liking

Compatible with the standard DEAP `.dat` parser in EmoSense.
