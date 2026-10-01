"""Tests for :mod:`btorch.analysis.voltage`."""

import numpy as np
import pytest
import torch

from btorch.analysis.voltage import suggest_skip_timestep, voltage_overshoot


def test_suggest_skip_timestep_clamps():
    # Short traces (< 800 steps -> skip < 100) get no burn-in.
    assert suggest_skip_timestep(np.zeros((400, 2))) == 0
    # Medium traces use one eighth of the length.
    assert suggest_skip_timestep(np.zeros((1600, 2))) == 200
    # Very long traces are capped at 1000 steps.
    assert suggest_skip_timestep(np.zeros((100000, 2))) == 1000


def test_voltage_overshoot_std_numpy_and_torch():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 3))
    out_np = voltage_overshoot(v, mode="std", skip_timestep=0)
    out_t = voltage_overshoot(torch.from_numpy(v), mode="std", skip_timestep=0)
    assert out_np.shape == (3,)
    # numpy uses ddof=0 while torch's default std is unbiased (ddof=1).
    np.testing.assert_allclose(out_np, v.std(0), rtol=1e-4)
    np.testing.assert_allclose(out_t.numpy(), v.std(0, ddof=1), rtol=1e-4)


def test_voltage_overshoot_mse_threshold():
    # Constant trace at -40 with threshold -50 -> MSE of 100 everywhere.
    v = np.full((20, 2), -40.0)
    out = voltage_overshoot(v, mode="mse_threshold", skip_timestep=0, V_th=-50.0)
    np.testing.assert_allclose(out, 100.0)


def test_voltage_overshoot_threshold_resting_fraction():
    # V_th=-50, V_reset=-70 -> scale 20, bounds [-130, 10] for n_scale=3.
    v = torch.full((10, 1), -60.0)
    v[:5] = 100.0  # half of the samples exceed the upper bound
    out = voltage_overshoot(v, skip_timestep=0, V_th=-50.0, V_reset=-70.0)
    assert out.item() == pytest.approx(0.5)


def test_voltage_overshoot_unknown_mode_raises():
    with pytest.raises(ValueError):
        voltage_overshoot(np.zeros((5, 1)), mode="bogus", skip_timestep=0)  # type: ignore[arg-type]
