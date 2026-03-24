# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Gradio Blocks frontend for the EmoSense file-upload emotion analysis demo."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any

import gradio as gr
import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from emosense.visualization import ContributionPlot, TopoMapPlot, VATrajectoryPlot

logger = logging.getLogger(__name__)

BACKEND_URL = "http://localhost:8000"

DEAP_CHANNELS: list[str] = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "F4", "F8", "FC6", "FC2", "C4", "T8",
    "CP6", "CP2", "P4", "P8", "PO4", "O2", "Fz", "Cz",
]

# ---------------------------------------------------------------------------
# Plot singletons
# ---------------------------------------------------------------------------

va_plot = VATrajectoryPlot(history_len=20)
topo_plot = TopoMapPlot(ch_names=DEAP_CHANNELS, fs=128)
contrib_plot = ContributionPlot()

# ---------------------------------------------------------------------------
# Shared session state
# ---------------------------------------------------------------------------

_current_task_id: str | None = None
_all_results: list[dict[str, Any]] = []
_processing_done: bool = False

# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------


def _backend_online() -> bool:
    try:
        resp = httpx.get(f"{BACKEND_URL}/health", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def _fetch_model_names() -> list[str]:
    try:
        resp = httpx.get(f"{BACKEND_URL}/models", timeout=3.0)
        resp.raise_for_status()
        return [m["name"] for m in resp.json()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Timeline helper
# ---------------------------------------------------------------------------


def _plot_timeline(results: list[dict[str, Any]]) -> Figure:
    """Plot prediction labels and confidence over time segments."""
    fig, ax = plt.subplots(figsize=(8, 3), dpi=100)

    if not results:
        ax.text(0.5, 0.5, "No results yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#888")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    times = [r.get("time_start_sec", i) for i, r in enumerate(results)]
    confs = [r.get("confidence", 0.0) for r in results]
    labels = [r.get("label", "?") for r in results]

    label_set = sorted(set(labels))
    cmap = plt.cm.Set2  # type: ignore[attr-defined]
    color_map = {lbl: cmap(i / max(len(label_set), 1)) for i, lbl in enumerate(label_set)}
    colors = [color_map.get(lbl, "#888") for lbl in labels]

    ax.bar(times, confs, width=max(0.5, (times[-1] - times[0]) / len(times) * 0.8) if len(times) > 1 else 1.0,
           color=colors, edgecolor="none", alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1.05)
    ax.set_title("Prediction Timeline")

    handles = [plt.Line2D([0], [0], color=color_map[lbl], lw=6) for lbl in label_set]
    ax.legend(handles, label_set, loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gradio callback functions
# ---------------------------------------------------------------------------


def on_file_upload(file: Any) -> tuple[str, str]:
    """Upload file to backend, return format info and status."""
    global _current_task_id, _all_results, _processing_done
    _all_results = []
    _processing_done = False
    _current_task_id = None

    if file is None:
        return "No file selected", ""

    if not _backend_online():
        return "Backend offline — please wait", ""

    try:
        with open(file, "rb") as f:
            filename = file if isinstance(file, str) else getattr(file, "name", "file.dat")
            resp = httpx.post(
                f"{BACKEND_URL}/upload",
                files={"file": (str(filename).split("/")[-1], f)},
                timeout=30.0,
            )
        resp.raise_for_status()
        info = resp.json()
        _current_task_id = info["task_id"]

        fmt_text = (
            f"Format: {info['format_detected']}\n"
            f"Fs: {info['fs']} Hz | Channels: {info['n_channels']}\n"
            f"Trials: {info['n_trials']} | Est. segments: {info['estimated_segments']}"
        )
        return f"File uploaded — task {info['task_id']}", fmt_text
    except Exception as exc:
        return f"Upload failed: {exc}", ""


def on_analyze(
    window_sec: float,
    overlap: float,
    model_name: str,
) -> str:
    """Start processing the uploaded file."""
    global _processing_done
    _processing_done = False

    if _current_task_id is None:
        return "No file uploaded yet"
    if not _backend_online():
        return "Backend offline"

    try:
        resp = httpx.post(
            f"{BACKEND_URL}/process/{_current_task_id}",
            timeout=5.0,
        )
        resp.raise_for_status()
        return f"Processing started (task {_current_task_id})…"
    except Exception as exc:
        return f"Failed to start: {exc}"


def on_reset() -> tuple[str, str, Any, Any, Any, Any, float, list, str]:
    """Reset all state and plots."""
    global _current_task_id, _all_results, _processing_done
    _current_task_id = None
    _all_results = []
    _processing_done = False
    va_plot.reset()

    return (
        "Ready",
        "",
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        0.0,
        [],
        "—",
    )


def on_band_change(band: str) -> None:
    topo_plot.set_band(band)


def poll_updates(
    current_band: str,
) -> tuple[Any, Any, Any, Any, float, list, str]:
    """Timer callback — poll /results/latest and refresh UI."""
    global _all_results, _processing_done

    if _current_task_id is None or _processing_done:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    try:
        resp = httpx.get(
            f"{BACKEND_URL}/results/latest",
            params={"task_id": _current_task_id},
            timeout=3.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    inference_results = [r for r in results if r.get("type") == "inference"]
    if len(inference_results) <= len(_all_results):
        complete = any(r.get("type") == "processing_complete" for r in results)
        if complete:
            _processing_done = True
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    _all_results = inference_results
    latest = inference_results[-1]

    valence = latest.get("valence", 0.0)
    arousal = latest.get("arousal", 0.0)
    label = latest.get("label", "—")
    confidence = latest.get("confidence", 0.0)
    de_features_raw = latest.get("de_features")
    attention_weights = latest.get("attention_weights")
    model_name = latest.get("model_name", "")

    va_fig = va_plot.update(valence, arousal, confidence, label)

    topo_fig = topo_plot.placeholder()
    if de_features_raw is not None:
        try:
            de_arr = np.asarray(de_features_raw, dtype=np.float64)
            topo_fig = topo_plot.update(de_arr)
        except Exception as exc:
            logger.warning("Topo-map update failed: %s", exc)

    weights_arr = np.asarray(attention_weights) if attention_weights else None
    contrib_fig = contrib_plot.update(weights_arr, model_name=model_name)

    timeline_fig = _plot_timeline(inference_results)

    total_estimated = max(len(inference_results), 1)
    progress = min(len(inference_results) / total_estimated, 1.0)

    table_data = [
        [
            r.get("trial_idx", 0),
            r.get("window_idx", 0),
            round(r.get("time_start_sec", 0), 1),
            r.get("label", ""),
            f"{r.get('confidence', 0) * 100:.1f}%",
            f"{r.get('latency_ms', 0):.1f}",
        ]
        for r in inference_results[-20:]
    ]

    pred_text = f"{label} ({confidence * 100:.1f}%)"

    return (
        va_fig,
        topo_fig,
        contrib_fig,
        timeline_fig,
        progress,
        table_data,
        pred_text,
    )


# ---------------------------------------------------------------------------
# Build the Gradio Blocks UI
# ---------------------------------------------------------------------------


def create_demo() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="EmoSense — Physiological Emotion Analysis",
    ) as demo:
        gr.Markdown("# EmoSense — Physiological Emotion Analysis")

        with gr.Row():
            # LEFT PANEL
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### File Upload")
                file_input = gr.File(
                    label="Upload Signal File",
                    file_types=[".dat", ".mat", ".npz", ".csv", ".bdf"],
                    type="filepath",
                )
                format_display = gr.Textbox(
                    label="Format Detection",
                    value="",
                    interactive=False,
                    lines=3,
                )

                gr.Markdown("### Parameters")
                window_slider = gr.Slider(
                    label="Window (sec)",
                    minimum=1.0, maximum=10.0, value=4.0, step=0.5,
                )
                overlap_slider = gr.Slider(
                    label="Overlap",
                    minimum=0.0, maximum=0.9, value=0.5, step=0.1,
                )
                model_dd = gr.Dropdown(
                    label="Model",
                    choices=["DGCNN", "EEGNet", "TSception"],
                    value="DGCNN",
                    interactive=True,
                )

                with gr.Row():
                    start_btn = gr.Button("Start Analysis", variant="primary")
                    reset_btn = gr.Button("Reset", variant="stop")

                band_radio = gr.Radio(
                    choices=["delta", "theta", "alpha", "beta", "gamma"],
                    value="alpha",
                    label="Topo Band",
                )

            # RIGHT PANEL
            with gr.Column(scale=3):
                with gr.Row():
                    status_box = gr.Textbox(
                        label="Status",
                        value="Ready",
                        interactive=False,
                        scale=2,
                    )
                    pred_label = gr.Textbox(
                        label="Current Prediction",
                        value="—",
                        interactive=False,
                        scale=1,
                    )

                with gr.Row():
                    va_plot_component = gr.Plot(label="V-A Trajectory")
                    topo_plot_component = gr.Plot(label="EEG Topographic Map")

                with gr.Row():
                    contrib_plot_component = gr.Plot(label="Modality Contribution")
                    timeline_plot_component = gr.Plot(label="Prediction Timeline")

                progress_bar = gr.Slider(
                    label="Progress",
                    minimum=0.0, maximum=1.0, value=0.0,
                    interactive=False,
                )

                results_table = gr.Dataframe(
                    headers=["Trial", "Window", "Time(s)", "Label", "Confidence", "Latency(ms)"],
                    label="Results",
                    interactive=False,
                )

        # ---- Timer for polling -----------------------------------------------
        timer = gr.Timer(every=0.5)
        timer.tick(
            fn=poll_updates,
            inputs=[band_radio],
            outputs=[
                va_plot_component,
                topo_plot_component,
                contrib_plot_component,
                timeline_plot_component,
                progress_bar,
                results_table,
                pred_label,
            ],
        )

        # ---- Event wiring ----------------------------------------------------
        file_input.change(
            fn=on_file_upload,
            inputs=[file_input],
            outputs=[status_box, format_display],
        )

        start_btn.click(
            fn=on_analyze,
            inputs=[window_slider, overlap_slider, model_dd],
            outputs=[status_box],
        )

        reset_btn.click(
            fn=on_reset,
            outputs=[
                status_box,
                format_display,
                va_plot_component,
                topo_plot_component,
                contrib_plot_component,
                timeline_plot_component,
                progress_bar,
                results_table,
                pred_label,
            ],
        )

        band_radio.change(
            fn=on_band_change,
            inputs=[band_radio],
        )

        demo.load(
            fn=lambda: (
                "Backend online" if _backend_online() else "Backend offline — waiting",
                gr.update(
                    choices=_fetch_model_names() or ["DGCNN", "EEGNet", "TSception"],
                ),
            ),
            outputs=[status_box, model_dd],
        )

    return demo


demo = create_demo()


def main() -> None:
    """Start both the FastAPI backend server and the Gradio frontend."""
    import uvicorn

    from emosense.backend.server import app as fastapi_app

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app": fastapi_app,
            "host": "0.0.0.0",
            "port": 8000,
            "log_level": "warning",
        },
        daemon=True,
    )
    server_thread.start()
    time.sleep(1)

    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)


if __name__ == "__main__":
    main()
