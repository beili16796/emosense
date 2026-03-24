# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Latency benchmark for the EmoSense <300ms paper claim.

Tests inference latency per model using synthetic DE feature windows.
Output: results/latency_benchmark.json

Usage::

    python -m emosense.scripts.benchmark_latency --n-warmup 10 --n-measure 100
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def run_benchmark(
    n_channels: int = 32,
    n_bands: int = 5,
    n_warmup: int = 10,
    n_measure: int = 100,
    output_path: str = "results/latency_benchmark.json",
) -> dict:
    """Measure inference latency for all available models.

    Args:
        n_channels: Number of EEG channels.
        n_bands: Number of frequency bands.
        n_warmup: Warm-up iterations (not measured).
        n_measure: Measurement iterations.
        output_path: Path for the output JSON.

    Returns:
        Dict mapping model names to latency statistics.
    """
    import torch

    X_de = np.random.randn(n_measure, n_channels, n_bands).astype(np.float32)
    results: dict[str, dict] = {}

    try:
        from emosense.backend.inference import ModelManager

        config_path = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
        if not config_path.exists():
            logger.warning("No model config found at %s; using synthetic benchmark", config_path)
            return _synthetic_benchmark(X_de, n_warmup, n_measure, output_path)

        manager = ModelManager(str(config_path))
    except Exception as exc:
        logger.warning("Could not load ModelManager (%s); running synthetic benchmark", exc)
        return _synthetic_benchmark(X_de, n_warmup, n_measure, output_path)

    for model_name in manager.get_model_names():
        manager.set_active_model(model_name)
        model = manager.get_active_model()

        if model is None:
            results[model_name] = {"error": "model not loaded"}
            continue

        def _infer(m: Any, inp: torch.Tensor) -> Any:
            try:
                if hasattr(m, "network"):
                    return m.network(inp)
            except Exception:
                pass
            if callable(m):
                return m(inp)
            return None

        times: list[float] = []
        with torch.no_grad():
            x = torch.FloatTensor(X_de)
            for _ in range(n_warmup):
                try:
                    _infer(model, x[:1])
                except Exception:
                    break

            for i in range(n_measure):
                t0 = time.perf_counter()
                try:
                    _infer(model, x[i : i + 1])
                except Exception:
                    break
                times.append((time.perf_counter() - t0) * 1000)

        if times:
            arr = np.array(times)
            results[model_name] = {
                "mean_ms": float(np.mean(arr)),
                "std_ms": float(np.std(arr)),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "max_ms": float(np.max(arr)),
                "pass_300": bool(np.percentile(arr, 99) < 300),
            }
        else:
            results[model_name] = {"error": "inference failed"}

        status = results[model_name]
        if "error" not in status:
            logger.info(
                "%20s: mean=%6.1fms p95=%6.1fms p99=%6.1fms %s",
                model_name,
                status["mean_ms"],
                status["p95_ms"],
                status["p99_ms"],
                "PASS" if status["pass_300"] else "FAIL",
            )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_latex_table(results, output_path)
    return results


def _print_latex_table(results: dict, output_path: str) -> None:
    """Print and save a LaTeX-ready latency table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Inference latency per model (CPU, single window)}",
        r"\label{tab:latency}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Mean (ms) & Std (ms) & p95 (ms) & p99 (ms) & $<$300ms \\",
        r"\midrule",
    ]
    for name, stats in results.items():
        if "error" in stats:
            lines.append(rf"{name} & \multicolumn{{5}}{{c}}{{error}} \\")
        else:
            p = r"\checkmark" if stats["pass_300"] else r"\xmark"
            lines.append(
                rf"{name} & {stats['mean_ms']:.1f} & {stats['std_ms']:.1f} "
                rf"& {stats['p95_ms']:.1f} & {stats['p99_ms']:.1f} & {p} \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table = "\n".join(lines)

    logger.info("LaTeX table:\n%s", table)

    tex_path = Path(output_path).with_suffix(".tex")
    tex_path.write_text(table, encoding="utf-8")
    logger.info("Saved LaTeX table to %s", tex_path)


def _synthetic_benchmark(
    X_de: np.ndarray, n_warmup: int, n_measure: int, output_path: str
) -> dict:
    """Fallback benchmark using a simple linear model."""
    import torch
    import torch.nn as nn

    model = nn.Sequential(nn.Flatten(), nn.Linear(X_de.shape[1] * X_de.shape[2], 2))
    model.eval()

    times: list[float] = []
    x = torch.FloatTensor(X_de)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x[:1])
        for i in range(n_measure):
            t0 = time.perf_counter()
            _ = model(x[i : i + 1])
            times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    results = {
        "synthetic_linear": {
            "mean_ms": float(np.mean(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "pass_300": bool(np.percentile(arr, 99) < 300),
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="EmoSense latency benchmark")
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-measure", type=int, default=100)
    parser.add_argument("--output", default="results/latency_benchmark.json")
    args = parser.parse_args()
    run_benchmark(n_warmup=args.n_warmup, n_measure=args.n_measure, output_path=args.output)
