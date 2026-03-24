#!/usr/bin/env python3
"""Extract 3 pre-selected DEAP trials for the EmoSense demo & user study.

Selects trials that clearly demonstrate different emotion quadrants
in the Russell Circumplex model:
  1. High Valence + High Arousal (excitement / happy)
  2. Low Valence + High Arousal  (anger / fear)
  3. Low Valence + Low Arousal   (sadness / boredom)

Usage::

    python scripts/extract_demo_trials.py \\
        --deap-root $EMOKIT_DATA_ROOT/DEAP \\
        --output demo_data/
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _load_deap_subject(dat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a DEAP .dat file and return (data, labels).

    Returns:
        data: (40, 40, 8064)
        labels: (40, 4) — valence, arousal, dominance, liking
    """
    with open(dat_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    return np.asarray(raw["data"]), np.asarray(raw["labels"])


def _select_trial(
    labels: np.ndarray,
    valence_range: tuple[float, float],
    arousal_range: tuple[float, float],
) -> int | None:
    """Find the trial index best matching the given V-A range."""
    valence = labels[:, 0]
    arousal = labels[:, 1]
    mask = (
        (valence >= valence_range[0]) & (valence <= valence_range[1])
        & (arousal >= arousal_range[0]) & (arousal <= arousal_range[1])
    )
    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        return None
    mid_v = (valence_range[0] + valence_range[1]) / 2
    mid_a = (arousal_range[0] + arousal_range[1]) / 2
    dists = np.sqrt((valence[candidates] - mid_v) ** 2 + (arousal[candidates] - mid_a) ** 2)
    return int(candidates[np.argmin(dists)])


def _save_single_trial(
    data: np.ndarray,
    labels: np.ndarray,
    trial_idx: int,
    output_path: Path,
) -> None:
    """Save a single trial as a .dat pickle compatible with EmoSense."""
    trial_data = data[trial_idx: trial_idx + 1]
    trial_labels = labels[trial_idx: trial_idx + 1]
    with open(output_path, "wb") as f:
        pickle.dump({"data": trial_data, "labels": trial_labels}, f, protocol=2)
    size_kb = output_path.stat().st_size / 1024
    print(f"  Saved: {output_path} ({size_kb:.0f} KB)")


def extract_demo_trials(deap_root: Path, output_dir: Path, subject: int = 1) -> None:
    dat_path = deap_root / f"s{subject:02d}.dat"
    if not dat_path.exists():
        raise FileNotFoundError(f"DEAP file not found: {dat_path}")

    data, labels = _load_deap_subject(dat_path)
    print(f"Loaded {dat_path}: data={data.shape}, labels={labels.shape}")
    print(f"Valence range: [{labels[:, 0].min():.1f}, {labels[:, 0].max():.1f}]")
    print(f"Arousal range: [{labels[:, 1].min():.1f}, {labels[:, 1].max():.1f}]")

    quadrants = [
        ("happy", (6.0, 9.0), (6.0, 9.0)),
        ("angry", (1.0, 4.0), (6.0, 9.0)),
        ("sad",   (1.0, 4.0), (1.0, 4.0)),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    selected: list[tuple[str, int]] = []

    for name, v_range, a_range in quadrants:
        idx = _select_trial(labels, v_range, a_range)
        if idx is None:
            # Relax criteria
            idx = _select_trial(
                labels,
                (v_range[0] - 1, v_range[1] + 1),
                (a_range[0] - 1, a_range[1] + 1),
            )
        if idx is None:
            logger.warning("No suitable trial found for %s quadrant", name)
            continue

        v, a = labels[idx, 0], labels[idx, 1]
        print(f"\n{name}: trial {idx} (V={v:.1f}, A={a:.1f})")
        out_file = output_dir / f"trial_{name}_s{subject:02d}_t{idx:02d}.dat"
        _save_single_trial(data, labels, idx, out_file)
        selected.append((name, idx))

    print(f"\nExtracted {len(selected)} demo trials to {output_dir}/")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deap-root", required=True, type=Path)
    parser.add_argument("--output", default="demo_data", type=Path)
    parser.add_argument("--subject", type=int, default=1)
    args = parser.parse_args()
    extract_demo_trials(args.deap_root, args.output, args.subject)


if __name__ == "__main__":
    main()
