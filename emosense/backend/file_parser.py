# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Unified parser for uploaded physiological signal files."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEAP_EEG_CHANNELS: list[str] = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
]

SEED_62_CHANNELS: list[str] = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1",
    "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1",
    "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1",
    "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1",
    "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1",
    "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ",
    "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]

# Keep backward-compatible alias
SEED_CHANNELS_62 = SEED_62_CHANNELS

DREAMER_14_CHANNELS: list[str] = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]


class FileParser:
    """Unified parser for uploaded physiological signal files.

    Supports DEAP (.dat/.mat), SEED-V (.mat/.npz), generic CSV, and raw BDF.
    All outputs follow the same contract dict.
    """

    SUPPORTED_FORMATS: dict[str, str] = {
        ".dat": "_parse_deap_dat",
        ".mat": "_parse_mat_auto",
        ".npz": "_parse_seedv_npz",
        ".csv": "_parse_csv",
        ".bdf": "_parse_bdf",
    }

    @classmethod
    def parse(cls, filepath: Path) -> dict[str, Any]:
        """Auto-detect format by extension and parse.

        Args:
            filepath: Path to the uploaded file.

        Returns:
            Parsed data dict with keys: ``eeg``, ``gsr``, ``ecg``, ``labels``,
            ``fs``, ``ch_names``, ``format``, ``n_eeg_channels``,
            and optionally ``pre_extracted``.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
            RuntimeError: If parsing fails for any other reason.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        suffix = filepath.suffix.lower()
        if suffix not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{suffix}'. "
                f"Supported: {sorted(cls.SUPPORTED_FORMATS.keys())}\n"
                "Tip: DEAP preprocessed = .dat, SEED/SEED-V = .mat, custom = .csv"
            )
        try:
            result = getattr(cls, cls.SUPPORTED_FORMATS[suffix])(filepath)
        except (FileNotFoundError, ValueError):
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse {filepath.name}: {e}\n"
                "If this is DEAP, ensure you downloaded the 'preprocessed' version."
            ) from e

        logger.info(
            "Parsed %s: format=%s, eeg_shape=%s",
            filepath.name,
            result["format"],
            result.get("eeg", np.empty(0)).shape if result.get("eeg") is not None else "N/A",
        )
        return result

    @staticmethod
    def _parse_deap_dat(filepath: Path) -> dict[str, Any]:
        """Parse DEAP preprocessed .dat file (Python pickle).

        DEAP .dat contains:
        - data: (40, 40, 8064) — 40 trials, 40 channels, 8064 samples @128Hz
        - labels: (40, 4) — valence, arousal, dominance, liking

        Channels 0-31: EEG, 32-39: peripheral (hEOG, vEOG, zEMG, tEMG, GSR, Resp, BVP, Temp).
        First 3 seconds (384 samples) are baseline — dropped.
        """
        with open(filepath, "rb") as f:
            raw = pickle.load(f, encoding="latin1")

        data = np.asarray(raw["data"], dtype=np.float32)    # (40, 40, 8064)
        labels = np.asarray(raw["labels"], dtype=np.float32)  # (40, 4)

        baseline_samples = 3 * 128  # 384
        eeg = data[:, :32, baseline_samples:]
        gsr = data[:, 36:37, baseline_samples:]
        ecg = data[:, 38:39, baseline_samples:]

        val_labels = (labels[:, 0] > 5).astype(np.int64)

        return {
            "eeg": eeg,
            "gsr": gsr,
            "ecg": ecg,
            "labels": val_labels,
            "fs": 128,
            "ch_names": list(DEAP_EEG_CHANNELS),
            "format": "deap_dat",
            "n_eeg_channels": 32,
            "pre_extracted": False,
        }

    @classmethod
    def _parse_mat_auto(cls, filepath: Path) -> dict[str, Any]:
        """Auto-detect whether a .mat file is DEAP, DREAMER, or SEED format.

        DEAP .mat files contain ``data`` (40, 40, 8064) and ``labels`` (40, 4).
        DREAMER .mat files contain a top-level ``DREAMER`` struct.
        SEED .mat files contain ``de_LDS``, ``de_movingAve``, or EEG trial keys.
        """
        import scipy.io

        mat = scipy.io.loadmat(str(filepath), squeeze_me=True)

        if "DREAMER" in mat:
            return cls._parse_dreamer_mat(filepath)

        if "data" in mat and "labels" in mat:
            return cls._parse_deap_mat(filepath, mat)

        return cls._parse_seed_mat(filepath, mat)

    @staticmethod
    def _parse_deap_mat(filepath: Path, mat: dict | None = None) -> dict[str, Any]:
        """Parse DEAP .mat file (MATLAB format).

        DEAP .mat contains:
        - data: (40, 40, 8064) — 40 trials, 40 channels, 8064 samples @128Hz
        - labels: (40, 4) — valence, arousal, dominance, liking

        Channels 0-31: EEG, 32-39: peripheral.
        First 3 seconds (384 samples) are baseline — dropped.
        """
        if mat is None:
            import scipy.io
            mat = scipy.io.loadmat(str(filepath))

        data = np.asarray(mat["data"], dtype=np.float32)
        labels = np.asarray(mat["labels"], dtype=np.float32)

        baseline_samples = 3 * 128
        eeg = data[:, :32, baseline_samples:]
        gsr = data[:, 36:37, baseline_samples:]
        ecg = data[:, 38:39, baseline_samples:]

        val_labels = (labels[:, 0] > 5).astype(np.int64)

        return {
            "eeg": eeg,
            "gsr": gsr,
            "ecg": ecg,
            "labels": val_labels,
            "fs": 128,
            "ch_names": list(DEAP_EEG_CHANNELS),
            "format": "DEAP .mat",
            "pre_extracted": False,
        }

    @staticmethod
    def _parse_dreamer_mat(filepath: Path) -> dict[str, Any]:
        """Parse DREAMER .mat file (Emotiv EPOC 14-channel EEG).

        DREAMER structure (loaded with ``struct_as_record=False``):
        - ``DREAMER.Data[i].EEG.stimuli[k]`` → shape ``(M, 14)``
        - ``DREAMER.Data[i].ScoreValence[k]`` → float 1-5
        - 14 channels: AF3/F7/F3/FC5/T7/P7/O1/O2/P8/T8/FC6/F4/F8/AF4
        - Sampling rate: 128 Hz

        Only the first subject is returned (index 0).
        """
        import scipy.io

        mat = scipy.io.loadmat(
            str(filepath),
            squeeze_me=True,
            struct_as_record=False,
        )
        dreamer = mat["DREAMER"]
        subject_data = dreamer.Data
        if not hasattr(subject_data, "__len__"):
            subject_data = [subject_data]
        n_subjects = len(subject_data)

        subj = subject_data[0]
        stimuli = subj.EEG.stimuli
        if not hasattr(stimuli, "__len__"):
            stimuli = [stimuli]
        n_stim = len(stimuli)

        scores = subj.ScoreValence
        if not hasattr(scores, "__len__"):
            scores = [scores]

        trials: list[np.ndarray] = []
        labels: list[int] = []
        for k in range(n_stim):
            eeg_trial = np.asarray(stimuli[k], dtype=np.float32)
            if eeg_trial.ndim == 1:
                continue
            if eeg_trial.shape[-1] == 14:
                eeg_trial = eeg_trial.T
            trials.append(eeg_trial)
            val = float(scores[k]) if np.isscalar(scores[k]) else float(np.asarray(scores[k]).flat[0])
            labels.append(1 if val > 3 else 0)

        eeg = np.stack(trials, axis=0)

        return {
            "eeg": eeg,
            "gsr": None,
            "ecg": None,
            "labels": np.array(labels, dtype=np.int64),
            "fs": 128,
            "ch_names": list(DREAMER_14_CHANNELS),
            "format": "DREAMER .mat",
            "n_eeg_channels": 14,
            "n_subjects": n_subjects,
            "n_stimuli": n_stim,
            "pre_extracted": False,
        }

    @staticmethod
    def _parse_seed_mat(filepath: Path, mat: dict | None = None) -> dict[str, Any]:
        """Parse SEED/SEED-V .mat file.

        Handles both raw EEG and pre-extracted DE features (``de_LDS`` key).
        """
        if mat is None:
            import scipy.io
            mat = scipy.io.loadmat(str(filepath))

        if "de_LDS" in mat:
            de_data = mat["de_LDS"]
            if de_data.dtype == object:
                de_list = []
                for i in range(len(de_data)):
                    cell = de_data[i]
                    if cell.ndim == 2:
                        de_list.append(cell.T.reshape(-1, cell.shape[0], 1) if cell.shape[1] < cell.shape[0]
                                       else cell.reshape(-1, cell.shape[0], cell.shape[1] // cell.shape[0])
                                       if cell.shape[0] == 62 else cell.T[np.newaxis])
                    else:
                        de_list.append(cell if cell.ndim == 3 else cell[np.newaxis])
                try:
                    de_arr = np.concatenate(
                        [d.reshape(-1, 62, 5) if d.shape[-1] == 5 or d.shape[1] == 62
                         else np.transpose(d, (2, 0, 1)) for d in de_list],
                        axis=0,
                    )
                except (ValueError, IndexError):
                    de_arr = np.concatenate(
                        [d.reshape(-1, d.shape[-2] if d.ndim >= 2 else 62, 5) for d in de_list],
                        axis=0,
                    )
            else:
                if de_data.ndim == 3:
                    de_arr = np.transpose(de_data, (2, 0, 1))  # → (n_samples, 62, 5)
                else:
                    de_arr = de_data

            labels_raw = mat.get("label", mat.get("labels", None))
            if labels_raw is not None:
                labels = np.asarray(labels_raw).flatten().astype(int)
            else:
                labels = np.zeros(len(de_arr), dtype=int)

            return {
                "eeg_de": de_arr.astype(np.float32),
                "eeg": None,
                "gsr": None,
                "ecg": None,
                "labels": labels,
                "fs": 200,
                "ch_names": list(SEED_62_CHANNELS),
                "format": "seed_mat_de",
                "n_eeg_channels": 62,
                "pre_extracted": True,
            }

        eeg_key = next(
            (k for k in mat if not k.startswith("_")
             and hasattr(mat[k], "shape")
             and mat[k].ndim >= 2),
            None,
        )
        if eeg_key is None:
            non_private = [k for k in mat if not k.startswith("_")]
            raise ValueError(
                f"Cannot find EEG data in {filepath.name}. "
                f"Keys: {non_private}"
            )

        eeg = np.asarray(mat[eeg_key], dtype=np.float32)
        if eeg.ndim == 2:
            eeg = eeg[np.newaxis, :, :]

        labels_raw = mat.get("label", mat.get("labels", None))
        labels = np.asarray(labels_raw).flatten().astype(int) if labels_raw is not None else np.array([0], dtype=int)

        return {
            "eeg": eeg,
            "gsr": None,
            "ecg": None,
            "labels": labels,
            "fs": 200,
            "ch_names": list(SEED_62_CHANNELS),
            "format": "seed_mat_raw",
            "n_eeg_channels": 62,
            "pre_extracted": False,
        }

    @staticmethod
    def _parse_seedv_npz(filepath: Path) -> dict[str, Any]:
        """Parse SEED-V pre-extracted DE features from .npz files.

        Expected structure: ``data`` key containing a pickled dict with
        trial indices as keys, each value shape ``(n_windows, 310)``
        (reshaped to ``(n_windows, 62, 5)``).
        ``label`` key containing a pickled dict of trial labels.
        """
        npz = np.load(str(filepath), allow_pickle=True)

        data_raw = npz.get("data", None)
        if data_raw is None:
            non_meta = [k for k in npz.files if not k.startswith("_")]
            raise ValueError(
                f"Cannot find 'data' key in {filepath.name}. Found: {non_meta}"
            )

        data_dict = data_raw.item() if hasattr(data_raw, "item") else data_raw
        label_raw = npz.get("label", None)
        label_dict = label_raw.item() if label_raw is not None and hasattr(label_raw, "item") else label_raw

        de_trials: list[np.ndarray] = []
        all_labels: list[int] = []

        for key in sorted(data_dict.keys(), key=lambda k: int(k) if str(k).isdigit() else k):
            trial = np.asarray(data_dict[key], dtype=np.float32)
            if trial.ndim == 2 and trial.shape[-1] == 310:
                trial = trial.reshape(trial.shape[0], 62, 5)
            de_trials.append(trial)
            if label_dict is not None and key in label_dict:
                lbl = label_dict[key]
                n_win = trial.shape[0]
                if np.isscalar(lbl):
                    all_labels.extend([int(lbl)] * n_win)
                else:
                    all_labels.extend(int(v) for v in np.asarray(lbl).flatten())

        labels = np.array(all_labels, dtype=np.int64) if all_labels else None

        return {
            "eeg_de": de_trials,
            "eeg": np.empty((0, 62, 0), dtype=np.float32),
            "labels": labels,
            "fs": 200,
            "ch_names": list(SEED_CHANNELS_62),
            "format": "SEED-V .npz (DE)",
            "pre_extracted": True,
            "n_trials": len(de_trials),
        }

    @staticmethod
    def _parse_csv(filepath: Path) -> dict[str, Any]:
        """Parse generic CSV with timestamps in the first column."""
        import pandas as pd

        df = pd.read_csv(filepath)
        if df.shape[1] < 2:
            raise ValueError(
                "CSV must have at least 2 columns: timestamp + at least 1 channel. "
                f"Got {df.shape[1]} columns."
            )

        ts = df.iloc[:, 0].values.astype(float)
        sigs = df.iloc[:, 1:].values.T.astype(np.float32)

        diffs = np.diff(ts)
        if len(diffs) == 0 or np.std(diffs) > 0.1 * np.mean(np.abs(diffs)):
            fs = 128
            logger.warning("Irregular timestamps detected; assuming fs=128 Hz")
        else:
            fs = int(round(1.0 / np.mean(diffs)))

        eeg = sigs[np.newaxis, :, :]  # (1 trial, n_channels, n_samples)

        return {
            "eeg": eeg,
            "gsr": None,
            "ecg": None,
            "labels": np.array([0]),
            "fs": fs,
            "ch_names": [str(c) for c in df.columns[1:]],
            "format": "csv",
            "n_eeg_channels": sigs.shape[0],
            "pre_extracted": False,
        }

    @staticmethod
    def _parse_bdf(filepath: Path) -> dict[str, Any]:
        """Parse raw BDF file using MNE."""
        import mne

        raw = mne.io.read_raw_bdf(str(filepath), preload=True, verbose=False)
        eeg_picks = mne.pick_types(raw.info, eeg=True)
        data = raw.get_data(picks=eeg_picks).astype(np.float32)
        ch_names = [raw.ch_names[i] for i in eeg_picks]
        eeg = data[np.newaxis, :, :]

        return {
            "eeg": eeg,
            "gsr": None,
            "ecg": None,
            "labels": None,
            "fs": int(raw.info["sfreq"]),
            "ch_names": ch_names,
            "format": "bdf",
            "n_eeg_channels": len(ch_names),
            "pre_extracted": False,
        }
