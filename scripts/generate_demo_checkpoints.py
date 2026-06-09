# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Train lightweight demo checkpoints for all 6 EmoSense models on synthetic data."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from emokit.models import build_model
from emokit.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")

MODEL_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "CNN-LSTM",
        "registry_name": "CNN-LSTM",
        "params": {
            "n_classes": 2,
            "input_type": "de",
            "n_channels": 32,
            "hidden_size": 64,
            "n_layers": 2,
            "dropout": 0.3,
            "lr": 0.001,
            "batch_size": 32,
            "n_epochs": 3,
        },
        "input_shape": (200, 160),
    },
    {
        "name": "DGCNN",
        "registry_name": "DGCNN",
        "params": {
            "n_classes": 2,
            "n_channels": 32,
            "n_bands": 5,
            "hidden_dim": 64,
            "lr": 0.001,
            "batch_size": 32,
            "n_epochs": 3,
            "lambda_reg": 0.0001,
        },
        "input_shape": (200, 32, 5),
    },
    {
        "name": "Transformer-MM",
        "registry_name": "Transformer-MM",
        "params": {
            "n_classes": 5,
            "n_channels": 62,
            "n_bands": 5,
            "n_peripheral_feat": 8,
            "d_model": 64,
            "nhead": 4,
            "n_layers": 2,
            "lr": 0.001,
            "batch_size": 32,
            "n_epochs": 3,
        },
        "input_shape": "multimodal_transformer",
    },
    {
        "name": "BiDAE",
        "registry_name": "BiDAE",
        "params": {
            "n_classes": 2,
            "n_feat1": 160,
            "n_feat2": 3,
            "lambda_recon": 0.1,
            "mu_align": 0.01,
            "lr": 0.001,
            "batch_size": 32,
            "n_epochs": 3,
        },
        "input_shape": "multimodal_bidae",
    },
    {
        "name": "DGCCA-AM",
        "registry_name": "DGCCA-AM",
        "params": {
            "n_classes": 2,
            "n_feat_eeg": 160,
            "n_feat_gsr": 3,
            "n_feat_ecg": 5,
            "hidden_dim": 128,
            "lr": 0.001,
            "batch_size": 32,
            "n_epochs": 3,
        },
        "input_shape": "multimodal_dgcca",
    },
    {
        "name": "PR-PL",
        "registry_name": "PR-PL",
        "params": {
            "n_classes": 2,
            "n_feat": 160,
            "prototype_dim": 128,
            "margin": 0.5,
            "lr": 0.001,
            "batch_size": 32,
            "n_epochs": 3,
            "lambda_pair": 0.5,
        },
        "input_shape": (200, 160),
    },
]

CHECKPOINT_FILENAMES: dict[str, str] = {
    "CNN-LSTM": "cnn_lstm_demo.pt",
    "DGCNN": "dgcnn_demo.pt",
    "Transformer-MM": "transformer_mm_demo.pt",
    "BiDAE": "bidae_demo.pt",
    "DGCCA-AM": "dgcca_am_demo.pt",
    "PR-PL": "prpl_demo.pt",
}


def _generate_data(
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[Any, np.ndarray]:
    """Generate synthetic training data appropriate for the model type.

    Args:
        cfg: Model configuration dict from ``MODEL_CONFIGS``.
        rng: Numpy random generator instance.

    Returns:
        Tuple of ``(X_train, y_train)``.
    """
    n_samples = 200
    y_train = rng.integers(0, 2, size=n_samples).astype(np.int64)
    shape = cfg["input_shape"]

    if isinstance(shape, tuple):
        x_train: Any = rng.standard_normal(shape).astype(np.float32)
        return x_train, y_train

    if shape == "multimodal_transformer":
        eeg = rng.standard_normal((n_samples, 62, 5)).astype(np.float32)
        peripheral = rng.standard_normal((n_samples, 8)).astype(np.float32)
        return {"eeg": eeg, "peripheral": peripheral}, y_train

    if shape == "multimodal_bidae":
        mod1 = rng.standard_normal((n_samples, 160)).astype(np.float32)
        mod2 = rng.standard_normal((n_samples, 3)).astype(np.float32)
        return {"mod1": mod1, "mod2": mod2}, y_train

    if shape == "multimodal_dgcca":
        eeg = rng.standard_normal((n_samples, 160)).astype(np.float32)
        gsr = rng.standard_normal((n_samples, 3)).astype(np.float32)
        ecg = rng.standard_normal((n_samples, 5)).astype(np.float32)
        return {"eeg": eeg, "gsr": gsr, "ecg": ecg}, y_train

    raise ValueError(f"Unknown input_shape descriptor: {shape!r}")


def generate_all_checkpoints() -> None:
    """Build, train, and save demo checkpoints for every configured model."""
    set_seed(42)
    rng = np.random.default_rng(42)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    succeeded: list[str] = []
    failed: list[str] = []

    for cfg in MODEL_CONFIGS:
        name: str = cfg["name"]
        registry_name: str = cfg["registry_name"]
        params: dict[str, Any] = cfg["params"]
        ckpt_name = CHECKPOINT_FILENAMES[name]
        ckpt_path = str(CHECKPOINT_DIR / ckpt_name)

        logger.info("=" * 60)
        logger.info("Training %s …", name)

        try:
            model = build_model(registry_name, params)
            x_train, y_train = _generate_data(cfg, rng)

            t0 = time.perf_counter()
            model.fit(x_train, y_train)
            elapsed = time.perf_counter() - t0
            logger.info("Training %s completed in %.1f s", name, elapsed)

            model.save(ckpt_path)
            logger.info("Saved checkpoint → %s", ckpt_path)
            succeeded.append(name)

        except Exception:
            logger.exception("Failed to train %s", name)
            failed.append(name)

    logger.info("=" * 60)
    logger.info(
        "Done: %d/%d succeeded%s",
        len(succeeded),
        len(MODEL_CONFIGS),
        f" | Failed: {failed}" if failed else "",
    )

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    generate_all_checkpoints()
