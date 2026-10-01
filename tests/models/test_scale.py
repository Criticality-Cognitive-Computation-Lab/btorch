"""Tests for :func:`btorch.models.scale.scale_state_`.

``scale_state_`` maps a state dict into a normalised space where
``v_reset -> 0`` and ``v_threshold -> 1``::

    v_scaled      = (v - zeropoint) / scale      (voltage-like, shifted)
    current_scaled = I / scale                    (Iasc, psc: no shift)
    asc_amps_scaled = asc_amps / scale[..., None]

with ``scale = v_threshold - v_reset`` and ``zeropoint = v_reset`` inferred
from the dict when not given. The scaling is in-place and reversible via
``unscale=True``.
"""

import numpy as np
import pytest
import torch

from btorch.models.scale import scale_state_


def _states():
    """A GLIF-like state dict in physical units (mV / pA)."""
    return {
        "v": torch.tensor([-70.0, -60.0, -50.0]),
        "v_threshold": torch.tensor([-50.0, -50.0, -50.0]),
        "v_reset": torch.tensor([-70.0, -70.0, -70.0]),
        "Iasc": torch.tensor([10.0, 20.0, 30.0]),
        "psc": torch.tensor([1.0, 2.0, 4.0]),
    }


def test_scale_infers_scale_and_zeropoint_from_threshold_and_reset():
    """threshold -> 1, reset -> 0, currents divided by (v_th - v_reset)=20."""
    st = _states()
    scale, zeropoint = scale_state_(st)

    torch.testing.assert_close(scale, torch.full((3,), 20.0))
    torch.testing.assert_close(zeropoint, torch.full((3,), -70.0))
    torch.testing.assert_close(st["v_threshold"], torch.ones(3))
    torch.testing.assert_close(st["v_reset"], torch.zeros(3))
    torch.testing.assert_close(st["v"], torch.tensor([0.0, 0.5, 1.0]))
    # current-like states are scaled but not shifted
    torch.testing.assert_close(st["Iasc"], torch.tensor([0.5, 1.0, 1.5]))
    torch.testing.assert_close(st["psc"], torch.tensor([0.05, 0.1, 0.2]))


def test_scale_then_unscale_roundtrip():
    """Unscale=True exactly inverts scaling when given the same scale."""
    original = _states()
    st = {k: v.clone() for k, v in original.items()}
    scale, zeropoint = scale_state_(st)
    scale_state_(st, scale=scale, zeropoint=zeropoint, unscale=True)
    for k, v in original.items():
        torch.testing.assert_close(st[k], v, msg=lambda m, k=k: f"{k}: {m}")


def test_explicit_scale_and_zeropoint_override_inference():
    """Scalars can be passed; dict without v_threshold/v_reset needs both."""
    st = {"v": torch.tensor([1.0, 2.0, 3.0])}
    scale, zeropoint = scale_state_(st, scale=2.0, zeropoint=1.0)
    assert (scale, zeropoint) == (2.0, 1.0)
    torch.testing.assert_close(st["v"], torch.tensor([0.0, 0.5, 1.0]))


def test_missing_scale_information_raises():
    """Neither scale nor v_threshold/v_reset available -> ValueError; a scale
    without zeropoint and without v_reset -> ValueError."""
    with pytest.raises(ValueError, match="scale"):
        scale_state_({"v": torch.ones(2)})
    with pytest.raises(ValueError, match="zeropoint"):
        scale_state_({"v": torch.ones(2)}, scale=2.0)


def test_store_records_scale_metadata_and_is_used_for_unscale():
    """``store=True`` saves scale/zeropoint/scaled so a later call with no
    arguments can unscale without the caller tracking them."""
    st = _states()
    scale_state_(st, store=True)
    assert st["scaled"] is True
    assert "scale" in st and "zeropoint" in st

    # v_threshold/v_reset are now normalised (1, 0) so inference alone would
    # be wrong; the stored scale/zeropoint must be picked up instead.
    scale_state_(st, unscale=True, store=True)
    assert st["scaled"] is False
    torch.testing.assert_close(st["v_threshold"], torch.full((3,), -50.0))
    torch.testing.assert_close(st["v_reset"], torch.full((3,), -70.0))
    torch.testing.assert_close(st["v"], _states()["v"])


def test_scaling_twice_is_a_noop_when_marked_scaled():
    """Once ``scaled`` is stored, scaling again must not double-scale."""
    st = _states()
    scale_state_(st, store=True)
    snapshot = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    scale_state_(st)  # already scaled -> early return
    for k in ("v", "Iasc", "psc", "v_threshold", "v_reset"):
        torch.testing.assert_close(st[k], snapshot[k])


def test_early_return_order_matches_documented_return():
    """Return order must be ``(scale, zeropoint)`` on every code path."""
    st = _states()
    scale, zeropoint = scale_state_(st, store=True)
    scale2, zeropoint2 = scale_state_(st)  # early-return path
    torch.testing.assert_close(scale2, scale)
    torch.testing.assert_close(zeropoint2, zeropoint)


def test_asc_amps_scaled_per_neuron_torch():
    """``asc_amps`` has a trailing ASC-channel dim: scale broadcasts over
    it."""
    st = {
        "v_threshold": torch.tensor([-50.0, -40.0]),
        "v_reset": torch.tensor([-70.0, -60.0]),
        "asc_amps": torch.tensor([[20.0, 40.0], [10.0, 30.0]]),  # (n, n_asc)
    }
    scale_state_(st)
    torch.testing.assert_close(st["asc_amps"], torch.tensor([[1.0, 2.0], [0.5, 1.5]]))


def test_asc_amps_numpy_and_scalar_scale():
    """Numpy / list asc_amps are supported with a numeric scale."""
    st = {"asc_amps": np.array([2.0, 4.0]), "v": torch.zeros(1)}
    scale_state_(st, scale=2.0, zeropoint=0.0)
    np.testing.assert_allclose(st["asc_amps"], [1.0, 2.0])

    st = {"asc_amps": [2.0, 4.0], "v": torch.zeros(1)}
    scale_state_(st, scale=2.0, zeropoint=0.0)
    np.testing.assert_allclose(st["asc_amps"], [1.0, 2.0])

    scale_state_(st, scale=2.0, zeropoint=0.0, unscale=True)
    np.testing.assert_allclose(st["asc_amps"], [2.0, 4.0])


def test_scale_state_does_not_track_gradients():
    """The function is ``no_grad``: results never require grad."""
    v = torch.tensor([1.0, 2.0], requires_grad=True)
    st = {"v": v}
    scale_state_(st, scale=2.0, zeropoint=0.0)
    assert not st["v"].requires_grad


def test_scale_state_asc_amps_rejects_scalar():
    with pytest.raises(TypeError, match="asc_amps"):
        scale_state_({"asc_amps": 1.0, "v": torch.ones(2)}, scale=2.0, zeropoint=0.0)
