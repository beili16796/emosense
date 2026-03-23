# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Dataset replay loader for feeding recorded trials to the backend."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from emokit.datasets import DatasetRegistry, load_dataset
from emokit.datasets.base import BaseDataset

logger = logging.getLogger(__name__)


class DatasetReplayLoader:
    """Load and serve individual trials from an emokit dataset.

    This loader is designed for offline replay: it loads a dataset,
    then exposes per-trial raw signal arrays that can be fed into
    :class:`~emosense.backend.stream_receiver.SimulatedReceiver`.

    Args:
        dataset_name: Registered dataset name (e.g. ``'DEAP'``).
        root: Path to the dataset root directory.
    """

    def __init__(self, dataset_name: str, root: str) -> None:
        self._dataset_name = dataset_name
        self._root = root
        self._dataset: BaseDataset = load_dataset(dataset_name, root=root)
        self._cache: dict[int, dict[str, np.ndarray]] = {}
        logger.info(
            "DatasetReplayLoader: loaded %s from %s", dataset_name, root,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_trial(
        self, subject_id: int, trial_id: int,
    ) -> dict[str, np.ndarray]:
        """Return raw signal arrays for a single trial.

        Args:
            subject_id: Subject identifier.
            trial_id: Zero-based trial index.

        Returns:
            Dict mapping modality name to array of shape
            ``(n_channels, n_samples)``.
        """
        raw = self._load_subject(subject_id)
        result: dict[str, np.ndarray] = {}
        for modality, arr in raw.items():
            if trial_id >= arr.shape[0]:
                raise IndexError(
                    f"trial_id {trial_id} out of range for subject "
                    f"{subject_id} modality {modality!r} "
                    f"(n_trials={arr.shape[0]})",
                )
            result[modality] = arr[trial_id]
        return result

    def list_trials(self, subject_id: int) -> list[dict[str, Any]]:
        """List trial metadata for a given subject.

        Args:
            subject_id: Subject identifier.

        Returns:
            List of dicts with keys ``trial_id``, ``modalities``, and
            per-modality ``n_channels`` / ``n_samples``.
        """
        raw = self._load_subject(subject_id)
        first_modality = next(iter(raw.values()))
        n_trials = first_modality.shape[0]

        trials: list[dict[str, Any]] = []
        for tid in range(n_trials):
            info: dict[str, Any] = {
                "trial_id": tid,
                "modalities": list(raw.keys()),
            }
            for modality, arr in raw.items():
                info[f"{modality}_n_channels"] = arr.shape[1]
                info[f"{modality}_n_samples"] = arr.shape[2]
            trials.append(info)
        return trials

    def get_subject_ids(self) -> list[int]:
        """Return available subject identifiers."""
        return self._dataset.get_subject_ids()

    @classmethod
    def get_dataset_names(cls) -> list[str]:
        """Return names of datasets registered in emokit."""
        registry = DatasetRegistry()
        return registry.available()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_subject(self, subject_id: int) -> dict[str, np.ndarray]:
        """Load and cache raw data for *subject_id*."""
        if subject_id not in self._cache:
            self._cache[subject_id] = self._dataset.read_raw(subject_id)
            logger.debug("Cached raw data for subject %d", subject_id)
        return self._cache[subject_id]
