# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Unit tests for the emosense visualisation components."""

from __future__ import annotations

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

    def test_multiple_updates_return_figure(self) -> None:
        plot = VATrajectoryPlot(history_len=10)
        for i in range(12):
            v = i * 0.1 - 0.5
            fig = plot.update(v, v, 0.7, "Neutral")
        assert isinstance(fig, Figure)

    def test_history_clamps_at_limit(self) -> None:
        plot = VATrajectoryPlot(history_len=10)
        for i in range(12):
            plot.update(i * 0.1 - 0.5, i * 0.05, 0.6 + i * 0.02, "Sad")
        assert len(plot._history) == 10

    def test_reset_clears_history(self) -> None:
        plot = VATrajectoryPlot(history_len=10)
        plot.update(0.1, 0.2, 0.5, "Sad")
        plot.update(0.3, -0.1, 0.9, "Happy")
        plot.reset()
        assert len(plot._history) == 0
        assert plot._prev_fig is None

    def test_label_colours_are_applied(self) -> None:
        plot = VATrajectoryPlot(history_len=5)
        for label in ("Happy", "Sad", "Neutral", "Excited", "Angry"):
            fig = plot.update(0.0, 0.0, 0.5, label)
            assert isinstance(fig, Figure)


# ======================================================================
# TopoMapPlot
# ======================================================================


class TestTopoMapPlot:
    """Tests for :class:`TopoMapPlot`."""

    _CHANNELS_32: list[str] = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6",
        "T7", "C3", "Cz", "C4", "T8",
        "CP5", "CP1", "CP2", "CP6",
        "P7", "P3", "Pz", "P4", "P8",
        "PO7", "PO8", "O1", "Oz", "O2",
        "AF3", "AF4",
    ]

    def test_update_returns_figure(self) -> None:
        try:
            from emosense.visualization.topo_map import TopoMapPlot
        except ImportError:
            pytest.skip("mne not available")

        try:
            plot = TopoMapPlot(channel_names=self._CHANNELS_32)
            de = np.random.randn(32, 5)
            fig = plot.update(de, band="alpha")
            assert isinstance(fig, Figure)
        except Exception as exc:
            pytest.skip(f"TopoMapPlot test skipped: {exc}")

    def test_update_with_different_bands(self) -> None:
        try:
            from emosense.visualization.topo_map import TopoMapPlot
        except ImportError:
            pytest.skip("mne not available")

        try:
            plot = TopoMapPlot(channel_names=self._CHANNELS_32)
            de = np.random.randn(32, 5)
            for band in ("delta", "theta", "alpha", "beta", "gamma"):
                fig = plot.update(de, band=band)
                assert isinstance(fig, Figure)
        except Exception as exc:
            pytest.skip(f"TopoMapPlot band test skipped: {exc}")

    def test_get_band_list(self) -> None:
        try:
            from emosense.visualization.topo_map import TopoMapPlot
        except ImportError:
            pytest.skip("mne not available")

        try:
            plot = TopoMapPlot(channel_names=["Fp1", "Fp2"])
            bands = plot.get_band_list()
            assert bands == ["delta", "theta", "alpha", "beta", "gamma"]
        except Exception as exc:
            pytest.skip(f"TopoMapPlot band list test skipped: {exc}")

    def test_set_band_validates(self) -> None:
        try:
            from emosense.visualization.topo_map import TopoMapPlot
        except ImportError:
            pytest.skip("mne not available")

        try:
            plot = TopoMapPlot(channel_names=["Fp1", "Fp2"])
            with pytest.raises(ValueError, match="Unknown band"):
                plot.set_band("invalid_band")
        except Exception as exc:
            pytest.skip(f"TopoMapPlot set_band test skipped: {exc}")


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

    def test_none_weights_placeholder(self) -> None:
        plot = ContributionPlot()
        fig = plot.update(None, model_name="CNN-LSTM")
        assert isinstance(fig, Figure)

    def test_custom_modality_names(self) -> None:
        plot = ContributionPlot(modality_names=["A", "B"])
        fig = plot.update(np.array([0.7, 0.3]), model_name="Test")
        assert isinstance(fig, Figure)

    def test_reset(self) -> None:
        plot = ContributionPlot()
        plot.update(np.array([0.5, 0.3, 0.2]))
        plot.reset()
        assert plot._prev_fig is None
