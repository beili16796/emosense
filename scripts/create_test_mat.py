#!/usr/bin/env python3
# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Create synthetic DEAP .mat or SEED-V .npz files for cloud-testable demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def create_deap_mat(n_trials: int, output: str) -> None:
    """Create a synthetic DEAP-format .mat file.

    Structure matches real DEAP: ``data`` (n_trials, 40, 8064),
    ``labels`` (n_trials, 4).
    """
    import scipy.io

    rng = np.random.default_rng(42)
    data = (rng.standard_normal((n_trials, 40, 8064)) * 10).astype(np.float32)
    labels = np.column_stack([
        rng.uniform(1, 9, n_trials),
        rng.uniform(1, 9, n_trials),
        rng.uniform(1, 9, n_trials),
        rng.uniform(1, 9, n_trials),
    ]).astype(np.float32)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(output, {"data": data, "labels": labels})
    print(f"Created DEAP .mat: {output}  (trials={n_trials})")


def create_seedv_npz(n_trials: int, output: str) -> None:
    """Create a synthetic SEED-V format .npz file.

    Structure: ``data`` dict of {trial_idx: (n_windows, 310)},
    ``label`` dict of {trial_idx: int}.
    """
    rng = np.random.default_rng(42)
    data_dict: dict[int, np.ndarray] = {}
    label_dict: dict[int, int] = {}

    for i in range(n_trials):
        n_windows = rng.integers(30, 60)
        data_dict[i] = rng.standard_normal((n_windows, 310)).astype(np.float32)
        label_dict[i] = int(rng.integers(0, 5))

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, data=data_dict, label=label_dict)
    print(f"Created SEED-V .npz: {output}  (trials={n_trials})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic test data")
    parser.add_argument("--format", choices=["deap", "seedv"], required=True)
    parser.add_argument("--n-trials", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.format == "deap":
        create_deap_mat(args.n_trials, args.output)
    else:
        create_seedv_npz(args.n_trials, args.output)


if __name__ == "__main__":
    main()
