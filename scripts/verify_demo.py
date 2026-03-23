# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Verify all demo checkpoints load correctly and meet latency requirements."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
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
MAX_LATENCY_MS = 300.0


@dataclass
class VerifySpec:
    """Specification for verifying a single model checkpoint.

    Attributes:
        name: Human-readable model name.
        registry_name: Name used in the emokit model registry.
        params: Construction parameters passed to ``build_model``.
        checkpoint: Filename of the checkpoint in ``CHECKPOINT_DIR``.
        make_input: Callable that produces a single-sample input for inference.
    """

    name: str
    registry_name: str
    params: dict[str, Any]
    checkpoint: str
    make_input: Any  # Callable[[], Any]


def _make_cnn_lstm_input() -> np.ndarray:
    return np.random.randn(1, 160).astype(np.float32)


def _make_dgcnn_input() -> np.ndarray:
    return np.random.randn(1, 32, 5).astype(np.float32)


def _make_transformer_mm_input() -> dict[str, np.ndarray]:
    return {
        "eeg": np.random.randn(1, 32, 5).astype(np.float32),
        "peripheral": np.random.randn(1, 8).astype(np.float32),
    }


def _make_bidae_input() -> dict[str, np.ndarray]:
    return {
        "mod1": np.random.randn(1, 160).astype(np.float32),
        "mod2": np.random.randn(1, 3).astype(np.float32),
    }


def _make_dgcca_input() -> dict[str, np.ndarray]:
    return {
        "eeg": np.random.randn(1, 160).astype(np.float32),
        "gsr": np.random.randn(1, 3).astype(np.float32),
        "ecg": np.random.randn(1, 5).astype(np.float32),
    }


def _make_prpl_input() -> np.ndarray:
    return np.random.randn(1, 160).astype(np.float32)


SPECS: list[VerifySpec] = [
    VerifySpec(
        name="CNN-LSTM",
        registry_name="CNN-LSTM",
        params={
            "n_classes": 2, "input_type": "de", "hidden_size": 64,
            "n_layers": 2, "dropout": 0.3,
        },
        checkpoint="cnn_lstm_demo.pt",
        make_input=_make_cnn_lstm_input,
    ),
    VerifySpec(
        name="DGCNN",
        registry_name="DGCNN",
        params={
            "n_classes": 2, "n_channels": 32, "n_bands": 5,
            "hidden_dim": 64, "lambda_reg": 0.0001,
        },
        checkpoint="dgcnn_demo.pt",
        make_input=_make_dgcnn_input,
    ),
    VerifySpec(
        name="Transformer-MM",
        registry_name="Transformer-MM",
        params={
            "n_classes": 2, "n_channels": 32, "n_bands": 5,
            "n_peripheral_feat": 8, "d_model": 64, "nhead": 4, "n_layers": 2,
        },
        checkpoint="transformer_mm_demo.pt",
        make_input=_make_transformer_mm_input,
    ),
    VerifySpec(
        name="BiDAE",
        registry_name="BiDAE",
        params={
            "n_classes": 2, "n_feat1": 160, "n_feat2": 3,
        },
        checkpoint="bidae_demo.pt",
        make_input=_make_bidae_input,
    ),
    VerifySpec(
        name="DGCCA-AM",
        registry_name="DGCCA-AM",
        params={
            "n_classes": 2, "n_feat_eeg": 160, "n_feat_gsr": 3,
            "n_feat_ecg": 5, "hidden_dim": 128,
        },
        checkpoint="dgcca_am_demo.pt",
        make_input=_make_dgcca_input,
    ),
    VerifySpec(
        name="PR-PL",
        registry_name="PR-PL",
        params={
            "n_classes": 2, "n_feat": 160, "prototype_dim": 128,
            "margin": 0.5,
        },
        checkpoint="prpl_demo.pt",
        make_input=_make_prpl_input,
    ),
]


@dataclass
class VerifyResult:
    """Result of a single model verification run.

    Attributes:
        name: Model name.
        loaded: Whether the checkpoint loaded successfully.
        latency_ms: Inference latency in milliseconds (``-1`` on failure).
        passed: Whether the model met all verification criteria.
        error: Error message if verification failed.
    """

    name: str
    loaded: bool
    latency_ms: float
    passed: bool
    error: str


def verify_model(spec: VerifySpec) -> VerifyResult:
    """Build a model, load its checkpoint, run inference, and verify latency.

    Args:
        spec: Verification specification for the model.

    Returns:
        ``VerifyResult`` describing the outcome.
    """
    ckpt_path = CHECKPOINT_DIR / spec.checkpoint

    if not ckpt_path.exists():
        msg = f"Checkpoint not found: {ckpt_path}"
        logger.error(msg)
        return VerifyResult(spec.name, loaded=False, latency_ms=-1, passed=False, error=msg)

    try:
        model = build_model(spec.registry_name, spec.params)
        model.load(str(ckpt_path))
    except Exception as exc:
        msg = f"Failed to load: {exc}"
        logger.error("%s — %s", spec.name, msg)
        return VerifyResult(spec.name, loaded=False, latency_ms=-1, passed=False, error=msg)

    try:
        x = spec.make_input()
        t0 = time.perf_counter()
        proba = model.predict_proba(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        msg = f"Inference failed: {exc}"
        logger.error("%s — %s", spec.name, msg)
        return VerifyResult(spec.name, loaded=True, latency_ms=-1, passed=False, error=msg)

    if proba.ndim != 2 or proba.shape[1] != spec.params.get("n_classes", 2):
        msg = f"Unexpected output shape: {proba.shape}"
        logger.error("%s — %s", spec.name, msg)
        return VerifyResult(spec.name, loaded=True, latency_ms=latency_ms, passed=False, error=msg)

    if latency_ms > MAX_LATENCY_MS:
        msg = f"Latency {latency_ms:.1f} ms exceeds {MAX_LATENCY_MS} ms limit"
        logger.error("%s — %s", spec.name, msg)
        return VerifyResult(spec.name, loaded=True, latency_ms=latency_ms, passed=False, error=msg)

    logger.info("%s — OK (%.1f ms)", spec.name, latency_ms)
    return VerifyResult(spec.name, loaded=True, latency_ms=latency_ms, passed=True, error="")


def _print_summary(results: list[VerifyResult]) -> None:
    """Print a formatted summary table of verification results.

    Args:
        results: List of verification results for all models.
    """
    header = f"{'Model':<20} {'Loaded':<8} {'Latency (ms)':<14} {'Status':<8} {'Error'}"
    separator = "-" * len(header)
    logger.info(separator)
    logger.info(header)
    logger.info(separator)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        latency_str = f"{r.latency_ms:.1f}" if r.latency_ms >= 0 else "N/A"
        loaded_str = "Yes" if r.loaded else "No"
        logger.info(
            "%-20s %-8s %-14s %-8s %s",
            r.name, loaded_str, latency_str, status, r.error,
        )
    logger.info(separator)


def main() -> None:
    """Run verification for all configured model checkpoints."""
    set_seed(42)
    np.random.seed(42)

    results: list[VerifyResult] = []
    for spec in SPECS:
        results.append(verify_model(spec))

    _print_summary(results)

    n_pass = sum(r.passed for r in results)
    n_total = len(results)
    logger.info("Verification: %d/%d passed", n_pass, n_total)

    if n_pass < n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
