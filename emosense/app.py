# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Gradio Blocks frontend for the EmoSense file-upload emotion analysis demo."""

from __future__ import annotations

import logging
import threading
import time
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
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
]

MODEL_NAMES_DEFAULT = ["CNN-LSTM", "DGCNN", "Transformer-MM", "BiDAE", "DGCCA-AM", "PR-PL"]

# ---------------------------------------------------------------------------
# Plot singletons
# ---------------------------------------------------------------------------

va_plot = VATrajectoryPlot(history_len=20)
topo_plot: TopoMapPlot | None = None
contrib_plot = ContributionPlot()


def _get_topo_plot(ch_names: list[str] | None = None, fs: int = 128) -> TopoMapPlot:
    """Lazily initialise TopoMapPlot with the correct channel config."""
    global topo_plot
    if topo_plot is None or (ch_names and len(ch_names) != topo_plot.n_channels):
        topo_plot = TopoMapPlot(ch_names=ch_names or DEAP_CHANNELS, fs=fs)
    return topo_plot


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


def _fetch_models() -> list[dict[str, Any]]:
    try:
        resp = httpx.get(f"{BACKEND_URL}/models", timeout=3.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Timeline helper
# ---------------------------------------------------------------------------


def _plot_timeline(results: list[dict[str, Any]]) -> Figure:
    """Plot V-A trajectory and confidence over time segments."""
    if not results:
        fig, ax = plt.subplots(figsize=(7, 2.5), dpi=100)
        ax.text(0.5, 0.5, "Process a file to see the prediction timeline",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    fig, (ax_va, ax_conf) = plt.subplots(2, 1, figsize=(7, 3.5),
                                          sharex=True, dpi=100)
    fig.patch.set_facecolor("none")

    xs = list(range(len(results)))
    vals = [r.get("valence", 0) for r in results]
    arrs = [r.get("arousal", 0) for r in results]
    confs = [r.get("confidence", 0) for r in results]

    ax_va.plot(xs, vals, color="#4C72B0", linewidth=1.5, label="Valence")
    ax_va.plot(xs, arrs, color="#DD8452", linewidth=1.5, label="Arousal")
    ax_va.axhline(0, color="#CCCCCC", linewidth=0.7, linestyle="--")
    ax_va.set_ylabel("V / A", fontsize=9)
    ax_va.set_ylim(-1.1, 1.1)
    ax_va.legend(fontsize=8, loc="upper right", framealpha=0.7)

    ax_conf.fill_between(xs, confs, alpha=0.35, color="#55A868")
    ax_conf.plot(xs, confs, color="#55A868", linewidth=1.2)
    ax_conf.set_ylabel("Conf.", fontsize=9)
    ax_conf.set_xlabel("Window index", fontsize=9)
    ax_conf.set_ylim(0, 1.05)

    fig.tight_layout(pad=0.8)
    return fig


# ---------------------------------------------------------------------------
# Gradio callback functions
# ---------------------------------------------------------------------------


def on_file_upload(
    file: Any,
    window_sec: float,
    overlap: float,
    model_name: str,
) -> tuple[str, str, Any]:
    """Upload file to backend, return format info, task_id, and enable button."""
    if file is None:
        return "No file selected", "", gr.update(interactive=False)

    if not _backend_online():
        return "ERROR: Backend offline. Is the server running?", "", gr.update(interactive=False)

    try:
        with open(file, "rb") as f:
            filename = file if isinstance(file, str) else getattr(file, "name", "file.dat")
            resp = httpx.post(
                f"{BACKEND_URL}/upload",
                files={"file": (str(filename).split("/")[-1], f)},
                data={
                    "window_sec": str(window_sec),
                    "overlap": str(overlap),
                    "model_name": model_name,
                },
                timeout=30.0,
            )
        resp.raise_for_status()
        info = resp.json()

        fmt_text = (
            f"Format: {info['format_detected']} | "
            f"Trials: {info['n_trials']} | "
            f"Fs: {info['fs']} Hz | "
            f"Channels: {info['n_channels']} | "
            f"Est. segments: {info['estimated_segments']}"
        )
        return fmt_text, info["task_id"], gr.update(interactive=True)
    except httpx.ConnectError:
        return "ERROR: Backend offline. Is the server running?", "", gr.update(interactive=False)
    except Exception as exc:
        return f"Error: {exc}", "", gr.update(interactive=False)


def on_analyze(
    task_id: str,
    model_name: str,
) -> tuple[str, int]:
    """Start processing the uploaded file."""
    if not task_id:
        return "No file uploaded yet", 0

    if not _backend_online():
        return "Backend offline", 0

    try:
        httpx.post(
            f"{BACKEND_URL}/models/active",
            json={"name": model_name},
            timeout=5.0,
        )
    except Exception:
        pass

    try:
        resp = httpx.post(
            f"{BACKEND_URL}/process/{task_id}",
            timeout=5.0,
        )
        resp.raise_for_status()
        return f"Processing\u2026 (task {task_id})", 0
    except Exception as exc:
        return f"Failed to start: {exc}", 0


def on_model_switch(model_name: str) -> None:
    """Switch active model mid-session."""
    try:
        httpx.post(
            f"{BACKEND_URL}/models/active",
            json={"name": model_name},
            timeout=5.0,
        )
    except Exception:
        pass


def on_model_change(model_name: str) -> tuple[str, str]:
    """Return markdown metadata for the selected model."""
    on_model_switch(model_name)
    models = {m["name"]: m for m in _fetch_models()}
    info = models.get(model_name)
    if not info:
        return "Unknown model", "Status unavailable"
    modalities = " + ".join(info.get("modalities", ["EEG"]))
    description = info.get("description", "")
    info_md = (
        f"**{model_name}**  \n"
        f"Modalities: `{modalities}`  \n"
        f"Trained on: {info.get('dataset', '?')}  \n"
        f"{description}"
    )
    status = (
        "Trained weights loaded"
        if info.get("has_real_weights")
        else "Random weights - export EmoKit checkpoints for meaningful predictions"
    )
    return info_md, status


def on_demo_load() -> tuple[str, Any, str, str]:
    names = _fetch_model_names() or MODEL_NAMES_DEFAULT
    first = names[0] if names else "DGCNN"
    info_md, status = on_model_change(first)
    return (
        "Backend online" if _backend_online() else "Backend offline - waiting",
        gr.update(choices=names, value=first),
        info_md,
        status,
    )


def on_reset(task_id: str) -> tuple[str, str, Any, Any, Any, Any, float, list, str, str, int]:
    """Reset all state and plots."""
    if task_id:
        try:
            httpx.post(f"{BACKEND_URL}/cancel/{task_id}", timeout=2.0)
        except Exception:
            pass
    va_plot.reset()
    contrib_plot.reset()

    return (
        "Ready",
        "",
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        0.0,
        [],
        "\u2014",
        "",       # clear task_id
        0,        # reset cursor
    )


def on_band_change(band: str, last_de_raw: Any) -> Any:
    """Re-render topomap with new band."""
    tp = _get_topo_plot()
    tp.set_band(band)
    if last_de_raw is not None:
        try:
            de_arr = np.asarray(last_de_raw, dtype=np.float64)
            return tp.update(de_arr, band=band)
        except Exception:
            pass
    return tp.placeholder()


def poll_updates(
    task_id: str,
    results_cursor: int,
    current_band: str,
) -> tuple[Any, Any, Any, Any, float, list, str, str, int, Any]:
    """Timer callback — poll /results/latest and refresh UI."""
    if not task_id:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )

    try:
        resp = httpx.get(
            f"{BACKEND_URL}/results/latest",
            params={"task_id": task_id, "since_idx": results_cursor},
            timeout=3.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )

    new_results = data.get("results", [])
    next_idx = data.get("next_idx", results_cursor)

    inference_results = [r for r in new_results if r.get("type") == "inference"]
    if not inference_results:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), next_idx, gr.update(),
        )

    latest = inference_results[-1]

    valence = latest.get("valence", 0.0)
    arousal = latest.get("arousal", 0.0)
    label = latest.get("label", "\u2014")
    confidence = latest.get("confidence", 0.0)
    de_features_raw = latest.get("de_features")
    attention_weights = latest.get("attention_weights")
    model_name = latest.get("model_name", "")

    va_fig = va_plot.update(valence, arousal, confidence, label)

    tp = _get_topo_plot()
    topo_fig = tp.placeholder()
    if de_features_raw is not None:
        try:
            de_arr = np.asarray(de_features_raw, dtype=np.float64)
            if de_arr.shape[0] != tp.n_channels:
                tp = _get_topo_plot(ch_names=None, fs=128)
            topo_fig = tp.update(de_arr, band=current_band)
        except Exception as exc:
            logger.warning("Topo-map update failed: %s", exc)

    weights_arr = np.asarray(attention_weights) if attention_weights else None
    contrib_fig = contrib_plot.update(weights_arr, model_name=model_name)

    all_results_resp = httpx.get(
        f"{BACKEND_URL}/results/latest",
        params={"task_id": task_id, "since_idx": 0},
        timeout=3.0,
    ).json()
    all_inference = [r for r in all_results_resp.get("results", []) if r.get("type") == "inference"]
    timeline_fig = _plot_timeline(all_inference)

    total_estimated = max(len(all_inference), 1)
    progress = min(len(all_inference) / total_estimated, 1.0)

    table_data = [
        [
            r.get("trial_idx", 0),
            r.get("window_idx", 0),
            round(r.get("time_start_sec", 0), 1),
            r.get("label", ""),
            f"{r.get('confidence', 0) * 100:.1f}%",
            f"{r.get('latency_ms', 0):.1f}",
        ]
        for r in all_inference[-20:]
    ]

    pred_text = f"{label} ({confidence * 100:.1f}%)"
    status_text = (
        f"Complete: {len(all_inference)} windows"
        if data.get("is_complete", False)
        else f"Processing: {len(all_inference)} windows"
    )

    last_de_state = de_features_raw

    return (
        va_fig,
        topo_fig,
        contrib_fig,
        timeline_fig,
        progress,
        table_data,
        pred_text,
        status_text,
        next_idx,
        last_de_state,
    )


def on_demo_deap_preset() -> tuple[float, float, str, str]:
    return 4.0, 0.5, "DGCNN", "Recommended: upload a DEAP .dat file"


def on_demo_seedv_preset() -> tuple[float, float, str, str]:
    return 4.0, 0.5, "Transformer-MM", "Recommended: upload a SEED-V .mat file"


# ---------------------------------------------------------------------------
# Build the Gradio Blocks UI
# ---------------------------------------------------------------------------


def create_demo() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="EmoSense \u2014 Physiological Emotion Analysis",
    ) as demo:
        gr.Markdown("# EmoSense \u2014 Physiological Emotion Analysis")

        uploaded_task_id = gr.State(value="")
        results_cursor = gr.State(value=0)
        last_de_features = gr.State(value=None)

        with gr.Row():
            # LEFT PANEL
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### File Upload")
                file_input = gr.File(
                    label="Upload Signal File",
                    file_types=[".dat", ".mat", ".csv", ".bdf"],
                    type="filepath",
                )
                format_display = gr.Textbox(
                    label="Format Detection",
                    value="",
                    interactive=False,
                    lines=2,
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
                    choices=MODEL_NAMES_DEFAULT,
                    value="DGCNN",
                    interactive=True,
                )
                model_info_md = gr.Markdown("Select a model to see details")
                weights_status = gr.Markdown("Waiting for backend model status")

                with gr.Row():
                    demo_deap_btn = gr.Button("Try with DEAP", size="sm")
                    demo_seedv_btn = gr.Button("Try with SEED-V", size="sm")
                gr.Markdown(
                    "_Demo presets auto-configure window, overlap, and model selection._"
                )

                with gr.Row():
                    analyze_btn = gr.Button("Start Analysis", variant="primary", interactive=False)
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
                        value="\u2014",
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
            inputs=[uploaded_task_id, results_cursor, band_radio],
            outputs=[
                va_plot_component,
                topo_plot_component,
                contrib_plot_component,
                timeline_plot_component,
                progress_bar,
                results_table,
                pred_label,
                status_box,
                results_cursor,
                last_de_features,
            ],
        )

        # ---- Event wiring ----------------------------------------------------
        file_input.upload(
            fn=on_file_upload,
            inputs=[file_input, window_slider, overlap_slider, model_dd],
            outputs=[format_display, uploaded_task_id, analyze_btn],
        )

        analyze_btn.click(
            fn=on_analyze,
            inputs=[uploaded_task_id, model_dd],
            outputs=[status_box, results_cursor],
        )

        model_dd.change(
            fn=on_model_change,
            inputs=[model_dd],
            outputs=[model_info_md, weights_status],
        )

        band_radio.change(
            fn=on_band_change,
            inputs=[band_radio, last_de_features],
            outputs=[topo_plot_component],
        )

        reset_btn.click(
            fn=on_reset,
            inputs=[uploaded_task_id],
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
                uploaded_task_id,
                results_cursor,
            ],
        )

        demo_deap_btn.click(
            fn=on_demo_deap_preset,
            outputs=[window_slider, overlap_slider, model_dd, status_box],
        )
        demo_seedv_btn.click(
            fn=on_demo_seedv_preset,
            outputs=[window_slider, overlap_slider, model_dd, status_box],
        )

        demo.load(
            fn=on_demo_load,
            outputs=[status_box, model_dd, model_info_md, weights_status],
        )

    return demo


demo = create_demo()


def main() -> None:
    """Start both the FastAPI backend server and the Gradio frontend."""
    import uvicorn

    from emosense.backend.server import app as fastapi_app

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s \u2014 %(message)s",
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
