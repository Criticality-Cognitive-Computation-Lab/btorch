"""Tests for the micro-scale dynamics helpers and the naming/API cleanups.

They also document the package-wide failure convention: an undefined estimate
is NaN (never a fake 0).
"""

import inspect

import numpy as np
import pytest

from btorch.analysis import firing_rate, isi_cv
from btorch.analysis.dynamic_tools import micro_scale
from btorch.visualisation.dynamics import plot_micro_dynamics


def test_spike_distance_single_neuron_is_nan():
    """Fewer than two neurons -> no pair -> NaN rather than a fake 0."""
    spikes = np.zeros((50, 1))
    spikes[[5, 20], 0] = 1
    assert np.isnan(micro_scale.compute_spike_distance(spikes, dt=1.0))


def test_spike_distance_identical_trains_is_zero():
    """Two identical spike trains are fully synchronous (distance 0)."""
    train = np.zeros((60, 1))
    train[[10, 30, 50]] = 1
    spikes = np.concatenate([train, train], axis=1)
    assert micro_scale.compute_spike_distance(spikes, dt=1.0) == pytest.approx(0.0)


def test_compute_cv_isi_was_merged_into_isi_cv():
    """``isi_cv`` is the single ISI-CV implementation."""
    assert not hasattr(micro_scale, "compute_cv_isi")


def test_isi_cv_nan_below_three_spikes():
    """A neuron needs >= 2 ISIs (3 spikes) for a CV; otherwise NaN."""
    spikes = np.zeros((100, 3))
    spikes[[10, 20, 30, 40], 0] = 1  # regular: CV = 0
    spikes[[10, 20], 1] = 1  # a single ISI: undefined
    cv, _ = isi_cv(spikes, dt=1.0)
    assert cv[0] == pytest.approx(0.0)
    assert np.isnan(cv[1]) and np.isnan(cv[2])


def test_firing_rate_uses_batch_axis_like_its_siblings():
    """``firing_rate(batch_axis=)`` replaces the old ``axis=`` keyword."""
    params = inspect.signature(firing_rate).parameters
    assert "batch_axis" in params and "axis" not in params
    spikes = (np.random.default_rng(0).random((200, 4)) > 0.9).astype(float)
    pop = firing_rate(spikes, width=5, dt=1.0, batch_axis=1)
    assert pop.shape == (200,)


def test_plot_micro_dynamics_has_no_unused_ax_parameter():
    """The dead ``ax`` argument was removed; stats keep ``fr`` and ``cv``."""
    assert "ax" not in inspect.signature(plot_micro_dynamics).parameters
    spikes = (np.random.default_rng(1).random((300, 10)) > 0.8).astype(int)
    fig, stats = plot_micro_dynamics(spikes, dt=1.0)
    assert set(stats) == {"fr", "cv"} and {"cv_isi", "mean"} <= set(stats["cv"])
    import matplotlib.pyplot as plt

    plt.close(fig)
