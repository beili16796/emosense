# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Unit tests for the emosense visualisation components."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest
from matplotlib.figure import Figure

from emosense.visualization.contribution import ContributionPlot
from emosense.visualization.va_plot import VATrajectoryPlot


# ======================================================================
# VATrajectoryPlot
# ======================================================================


class TestVATrajectoryPlot:
    """Tests for :class:`VATrajectoryPlot`."""

    def test_update_returns_figure(self) -> None:
        plot = VATrajectoryPlot(history_len=10)
        fig = plot.update(0.5, 0.3, 0.8, "Happy")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multiple_updates_return_figure(self) -> None:
        plot = VATrajectoryPlot(history_len=10)
        for i in range(12):
            v = i * 0.1 - 0.5
            fig = plot.update(v, v, 0.7, "Neutral")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_history_clamps_at_limit(self) -> None:
        plot = VATrajectoryPlot(history_len=10)
        for i in range(12):
            fig = plot.update(i * 0.1 - 0.5, i * 0.05, 0.6 + i * 0.02, "Sad")
            plt.close(fig)
        assert len(plot._trail) == 10

    def test_reset_clears_history(self) -> None:
        plot = VATrajectoryPlot(history_len=10)
        fig1 = plot.update(0.1, 0.2, 0.5, "Sad")
        plt.close(fig1)
        fig2 = plot.update(0.3, -0.1, 0.9, "Happy")
        plt.close(fig2)
        plot.reset()
        assert len(plot._trail) == 0
        assert plot._prev_fig is None

    def test_label_colours_are_applied(self) -> None:
        plot = VATrajectoryPlot(history_len=5)
        for label in ("Happy", "Sad", "Neutral", "Excited", "Angry"):
            fig = plot.update(0.0, 0.0, 0.5, label)
            assert isinstance(fig, Figure)
            plt.close(fig)

    def test_va_plot_trail_length(self) -> None:
        plot = VATrajectoryPlot(history_len=5)
        for i in range(8):
            fig = plot.update(i / 10, i / 10, 0.7, "Happy")
            plt.close(fig)
        assert len(plot._trail) == 5

    def test_va_plot_returns_figure(self) -> None:
        plot = VATrajectoryPlot()
        fig = plot.update(0.5, 0.3, 0.8, "Excited")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_va_plot_no_memory_leak(self) -> None:
        plot = VATrajectoryPlot()
        n_before = len(plt.get_fignums())
        for _ in range(10):
            fig = plot.update(0.0, 0.0, 0.5, "Neutral")
            plt.close(fig)
        n_after = len(plt.get_fignums())
        assert n_after <= n_before + 1

    def test_va_emotion_labels_in_annotation(self) -> None:
        plot = VATrajectoryPlot()
        fig = plot.update(0.7, 0.5, 0.85, "Happy")
        texts = [t.get_text() for t in fig.findobj(plt.Text)]
        assert any("Happy" in t for t in texts)
        plt.close(fig)


# ======================================================================
# TopoMapPlot
# ======================================================================


class TestTopoMapPlot:
    """Tests for :class:`TopoMapPlot`."""

    def test_update_32_channels(self) -> None:
        from emosense.visualization.topo_map import TopoMapPlot
        from emosense.backend.file_parser import DEAP_EEG_CHANNELS

        plot = TopoMapPlot(ch_names=list(DEAP_EEG_CHANNELS), fs=128)
        de = np.random.randn(32, 5).astype(np.float32)
        fig = plot.update(de, band="alpha")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_update_62_channels(self) -> None:
        from emosense.visualization.topo_map import TopoMapPlot
        from emosense.backend.file_parser import SEED_62_CHANNELS

        plot = TopoMapPlot(ch_names=list(SEED_62_CHANNELS), fs=200)
        de = np.random.randn(62, 5).astype(np.float32)
        fig = plot.update(de, band="beta")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_wrong_channel_count_raises(self) -> None:
        from emosense.visualization.topo_map import TopoMapPlot
        from emosense.backend.file_parser import DEAP_EEG_CHANNELS

        plot = TopoMapPlot(ch_names=list(DEAP_EEG_CHANNELS), fs=128)
        de = np.random.randn(62, 5)
        with pytest.raises(AssertionError, match="Expected"):
            plot.update(de)

    def test_update_with_different_bands(self) -> None:
        from emosense.visualization.topo_map import TopoMapPlot
        from emosense.backend.file_parser import DEAP_EEG_CHANNELS

        plot = TopoMapPlot(ch_names=list(DEAP_EEG_CHANNELS), fs=128)
        de = np.random.randn(32, 5).astype(np.float32)
        for band in ("delta", "theta", "alpha", "beta", "gamma"):
            fig = plot.update(de, band=band)
            assert isinstance(fig, Figure)
            plt.close(fig)

    def test_get_band_list(self) -> None:
        from emosense.visualization.topo_map import TopoMapPlot

        plot = TopoMapPlot(ch_names=["Fp1", "Fp2"])
        bands = plot.get_band_list()
        assert bands == ["delta", "theta", "alpha", "beta", "gamma"]

    def test_set_band_validates(self) -> None:
        from emosense.visualization.topo_map import TopoMapPlot

        plot = TopoMapPlot(ch_names=["Fp1", "Fp2"])
        with pytest.raises(ValueError, match="Unknown band"):
            plot.set_band("invalid_band")

    def test_channel_names_kwarg(self) -> None:
        from emosense.visualization.topo_map import TopoMapPlot
        from emosense.backend.file_parser import DEAP_EEG_CHANNELS

        plot = TopoMapPlot(channel_names=list(DEAP_EEG_CHANNELS))
        assert plot.n_channels == 32

    def test_placeholder(self) -> None:
        from emosense.visualization.topo_map import TopoMapPlot

        plot = TopoMapPlot(ch_names=["Fp1", "Fp2"])
        fig = plot.placeholder()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_frontal_asymmetry_detected_for_deap(self) -> None:
        from emosense.visualization.topo_map import TopoMapPlot
        from emosense.backend.file_parser import DEAP_EEG_CHANNELS

        plot = TopoMapPlot(ch_names=list(DEAP_EEG_CHANNELS), fs=128)
        de = np.random.randn(32, 5).astype(np.float32)
        asym = plot._compute_frontal_asymmetry(de)
        assert asym is not None
        assert isinstance(asym, float)

    def test_frontal_asymmetry_none_for_custom_channels(self) -> None:
        from emosense.visualization.topo_map import TopoMapPlot

        plot = TopoMapPlot(["Ch1", "Ch2", "Ch3", "Ch4"], fs=128)
        de = np.random.randn(4, 5).astype(np.float32)
        assert plot._compute_frontal_asymmetry(de) is None


# ======================================================================
# ContributionPlot
# ======================================================================


class TestContributionPlot:
    """Tests for :class:`ContributionPlot`."""

    def test_horizontal_bar_chart(self) -> None:
        plot = ContributionPlot()
        weights = np.array([0.6, 0.3, 0.1])
        fig = plot.update(weights, model_name="DGCCA-AM")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_none_weights_placeholder(self) -> None:
        plot = ContributionPlot()
        fig = plot.update(None, model_name="DGCNN")
        assert isinstance(fig, Figure)
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("Unimodal" in t or "not available" in t for t in texts)
        plt.close(fig)

    def test_custom_modality_names(self) -> None:
        plot = ContributionPlot(modality_names=["A", "B"])
        fig = plot.update(np.array([0.7, 0.3]), model_name="Test")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_reset(self) -> None:
        plot = ContributionPlot()
        plot.update(np.array([0.5, 0.3, 0.2]))
        plot.reset()
        assert plot._prev_fig is None
        assert plot._history == []

    def test_contribution_plot_with_weights(self) -> None:
        plot = ContributionPlot(["EEG", "GSR", "ECG"])
        weights = np.array([0.6, 0.3, 0.1])
        fig = plot.update(weights, model_name="DGCCA-AM")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_contribution_plot_none_shows_placeholder(self) -> None:
        plot = ContributionPlot()
        fig = plot.update(None, model_name="DGCNN")
        assert isinstance(fig, Figure)
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("Unimodal" in t or "not available" in t for t in texts)
        plt.close(fig)
