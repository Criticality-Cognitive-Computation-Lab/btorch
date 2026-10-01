"""Tests for the f-I / V-I sweep helper in ``btorch.fitting.tuning``."""

import torch

from btorch.fitting.tuning import compute_fi_vi_curve
from btorch.models.neurons.lif import LIF


def test_compute_fi_vi_curve_shapes_and_monotonic_rate():
    """A LIF neuron swept over currents yields documented shapes and an f-I
    curve that never decreases with the input current."""
    torch.manual_seed(0)
    params = {
        "n_neuron": 2,
        "v_threshold": -50.0,
        "v_reset": -65.0,
        "c_m": 1.0,
        "tau": 10.0,
        "tau_ref": 2.0,
        "step_mode": "s",
    }
    out = compute_fi_vi_curve(
        LIF, params, current_start=0.0, current_end=20.0, steps=5, duration=100
    )

    assert set(out) == {"currents", "frequencies", "voltages", "time"}
    assert out["currents"].shape == (5,)
    assert out["frequencies"].shape == (5, 2)
    assert out["voltages"].shape == (100, 5, 2)
    assert out["time"].shape == (100,)
    # Both neurons are identical: the rate grows (weakly) with the current.
    rates = out["frequencies"][:, 0]
    assert (rates[1:] >= rates[:-1]).all()
    assert rates[-1] > rates[0]
    # The caller's parameter dict is not modified (n_neuron is popped from a copy).
    assert params["n_neuron"] == 2
