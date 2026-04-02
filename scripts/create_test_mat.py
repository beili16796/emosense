#!/usr/bin/env python3
# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Create synthetic DEAP .mat or SEED-V .npz files for cloud-testable demos.

Use ``--realistic`` to generate physiologically plausible signals with
alpha-band dominance, frontal asymmetry patterns, and emotion-template DE
features.  Without that flag the data is pure random noise (fast, small).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ── helpers for realistic signal synthesis ──────────────────────────


def _bandpass_noise(
    rng: np.random.Generator,
    n_samples: int,
    fs: int,
    low: float,
    high: float,
) -> np.ndarray:
    """Band-limited noise via FFT filtering."""
    white = rng.standard_normal(n_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = (freqs >= low) & (freqs <= high)
    fft[~mask] = 0
    return np.fft.irfft(fft, n=n_samples).astype(np.float32)


def _one_over_f_noise(
    rng: np.random.Generator, n_samples: int, fs: int, exponent: float = 1.0
) -> np.ndarray:
    """1/f^exponent spectral slope noise."""
    white = rng.standard_normal(n_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    freqs[0] = 1.0
    fft /= np.power(freqs, exponent / 2)
    return np.fft.irfft(fft, n=n_samples).astype(np.float32)


_BAND_RANGES = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


# ── DEAP realistic ─────────────────────────────────────────────────


def _deap_channel_names() -> list[str]:
    return [
        "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
        "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
        "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
        "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
    ]


def create_deap_mat_realistic(output: str) -> None:
    """High-fidelity synthetic DEAP .mat with alpha asymmetry cues."""
    import scipy.io

    rng = np.random.default_rng(2024)
    ch_names = _deap_channel_names()
    n_trials = 4
    n_channels = 40
    n_eeg = 32
    fs = 128
    duration_s = 63
    n_samples = fs * duration_s  # 8064

    valence_labels = np.array([7.0, 3.0, 8.0, 2.0], dtype=np.float32)
    arousal_labels = np.array([6.0, 7.0, 3.0, 4.0], dtype=np.float32)

    data = np.zeros((n_trials, n_channels, n_samples), dtype=np.float32)

    f3_idx = ch_names.index("F3")
    f4_idx = ch_names.index("F4")

    for t in range(n_trials):
        positive = valence_labels[t] > 5.0
        high_arousal = arousal_labels[t] > 5.0
        snr_jitter = rng.uniform(0.8, 1.2)
        for ch in range(n_eeg):
            base = _one_over_f_noise(rng, n_samples, fs, exponent=1.0) * 5.0 * snr_jitter
            alpha_amp = 3.0 + rng.uniform(0, 2)
            if ch == f4_idx and positive:
                alpha_amp *= 1.08
            elif ch == f3_idx and positive:
                alpha_amp *= 0.93
            elif ch == f3_idx and not positive:
                alpha_amp *= 1.08
            elif ch == f4_idx and not positive:
                alpha_amp *= 0.93
            alpha = _bandpass_noise(rng, n_samples, fs, 8, 13) * alpha_amp
            theta = _bandpass_noise(rng, n_samples, fs, 4, 8) * 1.5
            beta = _bandpass_noise(rng, n_samples, fs, 13, 30) * (1.5 if high_arousal else 0.8)
            data[t, ch] = base + alpha + theta + beta

        for ch in range(n_eeg, n_channels):
            if ch == 36:
                t_arr = np.linspace(0, duration_s, n_samples)
                tonic = 0.5 * np.log1p(t_arr)
                phasic = np.zeros(n_samples, dtype=np.float32)
                if high_arousal:
                    peak_times = rng.choice(n_samples, size=8, replace=False)
                    for pt in peak_times:
                        width = rng.integers(64, 256)
                        end = min(pt + width, n_samples)
                        phasic[pt:end] += 0.4 * np.exp(-np.linspace(0, 3, end - pt))
                data[t, ch] = (
                    tonic + phasic + rng.standard_normal(n_samples) * 0.05
                ).astype(np.float32)
            else:
                data[t, ch] = rng.standard_normal(n_samples).astype(np.float32) * 0.2

    labels = np.column_stack([
        valence_labels,
        arousal_labels,
        rng.uniform(2, 8, n_trials).astype(np.float32),
        rng.uniform(2, 8, n_trials).astype(np.float32),
    ])

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(output, {"data": data, "labels": labels})
    print(f"Created realistic DEAP .mat: {output}  (trials={n_trials})")


# ── SEED-V realistic ───────────────────────────────────────────────


_SEED5_EMOTION_TEMPLATES = {
    0: {"gamma": 1.8, "beta": 1.2, "alpha": 0.8, "theta": 0.6, "delta": 0.5},  # happy
    1: {"gamma": 0.5, "beta": 0.6, "alpha": 0.8, "theta": 1.0, "delta": 1.6},  # sad
    2: {"gamma": 0.8, "beta": 0.8, "alpha": 1.0, "theta": 0.8, "delta": 0.8},  # neutral
    3: {"gamma": 1.3, "beta": 1.4, "alpha": 0.6, "theta": 1.1, "delta": 0.7},  # fear
    4: {"gamma": 0.6, "beta": 0.9, "alpha": 0.7, "theta": 1.2, "delta": 1.3},  # disgust
}


def create_seedv_npz_realistic(output: str) -> None:
    """High-fidelity SEED-V .npz with emotion-template DE features."""
    rng = np.random.default_rng(2024)
    n_trials = 10
    n_ch = 62
    n_bands = 5
    band_order = ["delta", "theta", "alpha", "beta", "gamma"]

    occipital_boost = np.ones(n_ch, dtype=np.float32)
    for idx in range(n_ch):
        if idx >= 50:
            occipital_boost[idx] = 1.3
        elif idx < 14:
            occipital_boost[idx] = 0.85

    data_dict: dict[int, np.ndarray] = {}
    label_dict: dict[int, int] = {}

    for i in range(n_trials):
        emotion = i % 5
        n_windows = 50
        template = _SEED5_EMOTION_TEMPLATES[emotion]
        de = np.zeros((n_windows, n_ch * n_bands), dtype=np.float32)
        for w in range(n_windows):
            for ch in range(n_ch):
                spatial = occipital_boost[ch]
                for b_idx, b_name in enumerate(band_order):
                    base = template[b_name] * spatial
                    val = base + rng.normal(0, 0.2)
                    de[w, ch * n_bands + b_idx] = max(val, 0.01)
        data_dict[i] = de
        label_dict[i] = emotion

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, data=data_dict, label=label_dict)
    print(f"Created realistic SEED-V .npz: {output}  (trials={n_trials})")


# ── DREAMER synthetic ──────────────────────────────────────────────

_DREAMER_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]


def create_dreamer_mat(n_stimuli: int, output: str) -> None:
    """Create a synthetic DREAMER-format .mat file.

    Mimics the real DREAMER structure:
    ``DREAMER.Data[0].EEG.stimuli[k]`` → (M, 14),
    ``DREAMER.Data[0].ScoreValence[k]`` → float.
    Uses scipy.io struct convention with ``struct_as_record=False``.
    """
    import scipy.io

    rng = np.random.default_rng(2024)
    fs = 128
    duration_s = 60

    class _Obj:
        pass

    stimuli_list = []
    score_list = []
    for k in range(n_stimuli):
        n_samples = fs * duration_s
        eeg = rng.standard_normal((n_samples, 14)).astype(np.float32) * 10
        stimuli_list.append(eeg)
        score_list.append(rng.uniform(1, 5))

    eeg_obj = _Obj()
    eeg_obj.stimuli = np.empty(n_stimuli, dtype=object)
    for k in range(n_stimuli):
        eeg_obj.stimuli[k] = stimuli_list[k]

    subj = _Obj()
    subj.EEG = eeg_obj
    subj.ScoreValence = np.array(score_list, dtype=np.float64)

    dreamer = _Obj()
    dreamer.Data = np.empty(1, dtype=object)
    dreamer.Data[0] = subj

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(output, {"DREAMER": dreamer})
    print(f"Created DREAMER .mat: {output}  (stimuli={n_stimuli})")


# ── original simple generators ─────────────────────────────────────


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


# ── CLI ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic test data")
    parser.add_argument("--format", choices=["deap", "seedv", "dreamer"], required=True)
    parser.add_argument("--n-trials", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--realistic",
        action="store_true",
        help="Generate physiologically plausible signals instead of random noise",
    )
    args = parser.parse_args()

    if args.format == "dreamer":
        create_dreamer_mat(args.n_trials, args.output)
    elif args.realistic:
        if args.format == "deap":
            create_deap_mat_realistic(args.output)
        else:
            create_seedv_npz_realistic(args.output)
    else:
        if args.format == "deap":
            create_deap_mat(args.n_trials, args.output)
        else:
            create_seedv_npz(args.n_trials, args.output)


if __name__ == "__main__":
    main()
