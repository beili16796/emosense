#!/usr/bin/env python3
"""Stress-test EmoSense server for user study stability.

Simulates repeated file uploads, model switching, and result polling
to verify memory stability and uptime over extended sessions.

Usage::

    python scripts/stress_test_session.py \\
        --dat-file $EMOKIT_DATA_ROOT/DEAP/s01.dat \\
        --n-iterations 50 --switch-models true
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
MODELS = ["CNN-LSTM", "DGCNN", "Transformer-MM", "BiDAE", "DGCCA-AM", "PR-PL"]


def _check_health() -> dict:
    resp = httpx.get(f"{BASE_URL}/health/detailed", timeout=5.0)
    resp.raise_for_status()
    return resp.json()


def _upload_file(dat_path: Path, model_name: str = "DGCNN") -> str:
    with open(dat_path, "rb") as f:
        resp = httpx.post(
            f"{BASE_URL}/upload",
            files={"file": (dat_path.name, f)},
            data={"window_sec": "4.0", "overlap": "0.5", "model_name": model_name},
            timeout=30.0,
        )
    resp.raise_for_status()
    return resp.json()["task_id"]


def _start_processing(task_id: str) -> None:
    resp = httpx.post(f"{BASE_URL}/process/{task_id}", timeout=5.0)
    resp.raise_for_status()


def _wait_for_completion(task_id: str, timeout_sec: float = 120.0) -> int:
    deadline = time.time() + timeout_sec
    n_results = 0
    while time.time() < deadline:
        resp = httpx.get(
            f"{BASE_URL}/results/latest",
            params={"task_id": task_id, "since_idx": 0},
            timeout=5.0,
        )
        data = resp.json()
        n_results = data.get("next_idx", 0)
        if data.get("is_complete", False):
            return n_results
        time.sleep(0.5)
    raise TimeoutError(f"Task {task_id} did not complete within {timeout_sec}s")


def _switch_model(name: str) -> None:
    resp = httpx.post(
        f"{BASE_URL}/models/active",
        json={"name": name},
        timeout=5.0,
    )
    resp.raise_for_status()


def _reset_server() -> None:
    try:
        resp = httpx.post(f"{BASE_URL}/admin/reset", timeout=5.0)
        resp.raise_for_status()
    except Exception:
        logger.warning("admin/reset not available; skipping")


def _get_memory_mb() -> float | None:
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def run_stress_test(
    dat_path: Path,
    n_iterations: int = 50,
    switch_models: bool = True,
) -> None:
    print(f"Stress test: {n_iterations} iterations, switch_models={switch_models}")
    print(f"File: {dat_path}")

    health = _check_health()
    print(f"Initial health: {health}")

    mem_start = _get_memory_mb()
    failures: list[str] = []
    latencies: list[float] = []
    t_global = time.time()

    for i in range(n_iterations):
        model = MODELS[i % len(MODELS)] if switch_models else "DGCNN"
        iter_start = time.time()

        try:
            task_id = _upload_file(dat_path, model_name=model)

            if switch_models:
                _switch_model(model)

            _start_processing(task_id)
            n_results = _wait_for_completion(task_id, timeout_sec=120.0)

            health = _check_health()
            if health.get("status") != "ok":
                failures.append(f"Iter {i}: health check failed: {health}")

            elapsed = time.time() - iter_start
            latencies.append(elapsed)

            mem = _get_memory_mb()
            mem_str = f"{mem:.0f}MB" if mem else "N/A"
            print(
                f"  [{i+1:3d}/{n_iterations}] model={model:15s} "
                f"results={n_results:4d} time={elapsed:5.1f}s mem={mem_str}"
            )

        except Exception as exc:
            failures.append(f"Iter {i}: {exc}")
            logger.exception("Iteration %d failed", i)

        if (i + 1) % 10 == 0:
            _reset_server()

    total_time = time.time() - t_global
    mem_end = _get_memory_mb()

    print(f"\n{'='*60}")
    print(f"STRESS TEST RESULTS")
    print(f"{'='*60}")
    print(f"Iterations:    {n_iterations}")
    print(f"Failures:      {len(failures)}")
    print(f"Total time:    {total_time:.1f}s")
    if latencies:
        import numpy as np
        arr = np.array(latencies)
        print(f"Avg iteration: {arr.mean():.1f}s")
        print(f"Max iteration: {arr.max():.1f}s")
    if mem_start and mem_end:
        print(f"Memory start:  {mem_start:.0f}MB")
        print(f"Memory end:    {mem_end:.0f}MB")
        print(f"Memory growth: {mem_end - mem_start:.0f}MB")
        if mem_end - mem_start > 500:
            failures.append(f"Memory grew by {mem_end - mem_start:.0f}MB (>500MB)")

    if total_time > 30 * 60:
        failures.append(f"Total time {total_time:.0f}s exceeds 30 minutes")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        raise RuntimeError(f"Stress test had {len(failures)} failures")

    print("\nALL CHECKS PASSED")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dat-file", required=True, type=Path)
    parser.add_argument("--n-iterations", type=int, default=50)
    parser.add_argument("--switch-models", type=str, default="true")
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.base_url

    if not args.dat_file.exists():
        raise FileNotFoundError(f"File not found: {args.dat_file}")

    run_stress_test(
        args.dat_file,
        n_iterations=args.n_iterations,
        switch_models=args.switch_models.lower() in ("true", "1", "yes"),
    )


if __name__ == "__main__":
    main()
