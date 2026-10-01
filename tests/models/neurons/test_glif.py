"""GLIF3 tests: closed-form no-spike solution and low-precision gradients.

* ``forward_exact_no_spike`` is the analytic solution of the GLIF3 ODEs for a
  constant input when no spike occurs. It is compared against the discrete
  simulation (``neuronal_charge`` + ``neuronal_adaptation``) it approximates.
* A bfloat16 neuron unrolled through ``make_rnn`` must back-propagate finite
  gradients (the surrogate gradient and the exponential Euler update must not
  overflow / divide by zero in reduced precision).
"""

import pytest
import torch

from btorch.models import environ, rnn
from btorch.models.functional import init_net_state
from btorch.models.neurons import GLIF3


# Typical Allen-style GLIF3 parameters in coherent units (mV, ms, pF, pA).
NEURON_PARAMS = {
    "v_threshold": -45.0,
    "v_reset": -60.0,
    "c_m": 2.0,
    "tau": 20.0,
    "k": [1.0 / 80, 0.3],  # two after-spike current components
    "asc_amps": [-0.2, -0.1],
    "tau_ref": 2.0,
}


def _discrete_no_spike(neuron: GLIF3, x: float, dt: float, n_step: int):
    """Simulate ``n_step`` steps without ever firing or resetting.

    ``single_step_forward`` is bypassed: only the continuous part of the model
    (membrane charge + after-spike current decay) is stepped, so the
    trajectory is directly comparable to the closed-form solution. The state
    *before* each update is recorded, i.e. entry ``i`` is the state after
    ``i`` steps, at time ``i * dt``.
    """
    x_t = torch.full_like(neuron.v, x)
    v_hist, iasc_hist = [], []
    with environ.context(dt=dt):
        for _ in range(n_step):
            v_hist.append(neuron.v.clone())
            iasc_hist.append(neuron.Iasc.clone())
            neuron.neuronal_charge(x_t)
            neuron.neuronal_adaptation()
    return torch.stack(v_hist), torch.stack(iasc_hist)


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.norm(a - b) / torch.norm(b)).item()


@pytest.mark.parametrize("dt", [1.0, 0.5, 0.1])
def test_forward_exact_no_spike_matches_discrete(dt):
    """Closed-form solution vs discrete simulation for several ``dt``.

    Setup: 4 neurons with random start voltages and *non-zero* initial
    after-spike currents (so the ``Iasc -> v`` coupling term of the analytic
    solution is exercised) under a constant input, in float64.

    Expected accuracy:

    * ``Iasc`` decays as ``exp(-k t)``, which the discrete exponential Euler
      step reproduces exactly: agreement to rounding error.
    * ``v`` is first-order accurate in ``dt``: within a step the membrane
      update uses the ``Iasc`` of the *start* of the step (explicit coupling),
      while the closed form lets it decay continuously. The relative error is
      measured ~1.4e-3 at dt=1 and shrinks linearly with ``dt``; the bound
      ``3e-3 * dt`` leaves about 2x headroom.
    """
    n_neuron, x, duration = 4, 3.0, 200.0
    n_step = int(duration / dt)
    torch.manual_seed(0)

    neuron = GLIF3(n_neuron, **NEURON_PARAMS, dtype=torch.float64)
    init_net_state(neuron, dtype=torch.float64)
    neuron.v = -60.0 + 10.0 * torch.rand(n_neuron, dtype=torch.float64)
    neuron.Iasc = torch.rand(n_neuron, 2, dtype=torch.float64)
    v0, iasc0 = neuron.v.clone(), neuron.Iasc.clone()

    # Closed form: pass explicit v0 / Iasc0 so the neuron state is untouched.
    times = torch.arange(n_step, dtype=torch.float64) * dt
    v_exact, iasc_exact = neuron.forward_exact_no_spike(
        torch.tensor(x, dtype=torch.float64), t=times, v0=v0, Iasc0=iasc0
    )
    assert v_exact.shape == (n_step, n_neuron)
    assert iasc_exact.shape == (n_step, n_neuron, 2)
    torch.testing.assert_close(neuron.v, v0)  # state not modified

    v_disc, iasc_disc = _discrete_no_spike(neuron, x, dt, n_step)

    # t = 0 is the initial state on both sides
    torch.testing.assert_close(v_exact[0], v0)
    torch.testing.assert_close(iasc_exact[0], iasc0)

    assert _relative_error(iasc_disc, iasc_exact) < 1e-12
    assert _relative_error(v_disc, v_exact) < 3e-3 * dt


def test_forward_exact_no_spike_error_shrinks_with_dt():
    """The discrete ``v`` converges to the closed form as ``dt -> 0``."""
    torch.manual_seed(1)
    n_neuron, x, duration = 3, 3.0, 100.0
    neuron = GLIF3(n_neuron, **NEURON_PARAMS, dtype=torch.float64)
    init_net_state(neuron, dtype=torch.float64)
    v0 = -60.0 + 10.0 * torch.rand(n_neuron, dtype=torch.float64)
    iasc0 = torch.rand(n_neuron, 2, dtype=torch.float64)

    errors = []
    for dt in (1.0, 0.1):
        n_step = int(duration / dt)
        neuron.v, neuron.Iasc = v0.clone(), iasc0.clone()
        times = torch.arange(n_step, dtype=torch.float64) * dt
        v_exact, _ = neuron.forward_exact_no_spike(
            torch.tensor(x, dtype=torch.float64), t=times, v0=v0, Iasc0=iasc0
        )
        v_disc, _ = _discrete_no_spike(neuron, x, dt, n_step)
        errors.append(_relative_error(v_disc, v_exact))

    # first-order method: a 10x smaller step gives roughly 10x smaller error
    assert errors[1] < errors[0] / 5


def _glif(n_neuron=2, **overrides):
    """A float64 GLIF3 with two after-spike current components."""
    neuron = GLIF3(n_neuron, **{**NEURON_PARAMS, **overrides}, dtype=torch.float64)
    init_net_state(neuron, dtype=torch.float64)
    return neuron


def test_exact_no_spike_does_not_touch_state_by_default():
    """Without ``update_state`` the module state is read but never written."""
    neuron = _glif()
    neuron.v = torch.tensor([-55.0, -50.0], dtype=torch.float64)
    neuron.Iasc = torch.ones(2, 2, dtype=torch.float64)
    v_before, iasc_before = neuron.v.clone(), neuron.Iasc.clone()

    neuron.forward_exact_no_spike(1.0, t=5.0)  # uses self.v / self.Iasc

    torch.testing.assert_close(neuron.v, v_before)
    torch.testing.assert_close(neuron.Iasc, iasc_before)


def test_exact_no_spike_update_state_keeps_last_time_point():
    """``update_state=True`` stores the state at the *last* requested time.

    The state has no time axis, so it keeps its ``(n_neuron, ...)`` shape and
    equals the last row of the returned trajectory (the earlier rows are not
    stored).
    """
    neuron = _glif()
    neuron.Iasc = torch.ones(2, 2, dtype=torch.float64)
    iasc0 = neuron.Iasc.clone()

    t = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
    v, iasc = neuron.forward_exact_no_spike(1.0, t=t, update_state=True)

    assert neuron.v.shape == (2,)
    assert neuron.Iasc.shape == (2, 2)
    torch.testing.assert_close(neuron.v, v[-1])
    torch.testing.assert_close(neuron.Iasc, iasc[-1])
    # Iasc decays exactly as exp(-k t) with t = 5 (the last time point)
    torch.testing.assert_close(neuron.Iasc, iasc0 * torch.exp(-5.0 * neuron.k))


@pytest.mark.parametrize(
    ("x_shape", "t", "batch", "n_time"),
    [
        ((), 5.0, (), 1),  # python scalar time -> one time point
        ((), torch.tensor(5.0), (), 1),  # 0-dim tensor time
        ((), torch.tensor([1.0, 2.0, 3.0]), (), 3),  # 1D times
        ((2,), torch.tensor([1.0, 2.0]), (), 2),  # per-neuron input
        ((4, 2), torch.tensor([1.0, 2.0]), (4,), 2),  # per-batch, per-neuron input
        ((4, 1), torch.tensor([1.0]), (4,), 1),  # per-batch input, shared by neurons
    ],
)
def test_exact_no_spike_output_shapes(x_shape, t, batch, n_time):
    """Output shapes follow the documented ``(n_time, *batch, n_neuron, ...)``.

    ``batch`` comes from the state (``v0`` / ``Iasc0``); ``x`` and ``t`` only
    have to broadcast against it. Here ``v0`` / ``Iasc0`` carry the batch.
    """
    n_neuron = 2
    neuron = _glif(n_neuron)
    v0 = torch.full((*batch, n_neuron), -55.0, dtype=torch.float64)
    iasc0 = torch.ones(*batch, n_neuron, 2, dtype=torch.float64)
    x = torch.full(x_shape, 3.0, dtype=torch.float64)

    v, iasc = neuron.forward_exact_no_spike(x, t=t, v0=v0, Iasc0=iasc0)

    assert v.shape == (n_time, *batch, n_neuron)
    assert iasc.shape == (n_time, *batch, n_neuron, 2)


def test_exact_no_spike_batch_matches_per_sample_runs():
    """A batched evaluation equals evaluating each sample on its own."""
    torch.manual_seed(0)
    neuron = _glif(3)
    v0 = -60.0 + 10.0 * torch.rand(4, 3, dtype=torch.float64)
    iasc0 = torch.rand(4, 3, 2, dtype=torch.float64)
    x = torch.rand(4, 3, dtype=torch.float64) * 5.0
    t = torch.tensor([0.5, 2.0, 7.0], dtype=torch.float64)

    v, iasc = neuron.forward_exact_no_spike(x, t=t, v0=v0, Iasc0=iasc0)

    for b in range(4):
        v_b, iasc_b = neuron.forward_exact_no_spike(x[b], t=t, v0=v0[b], Iasc0=iasc0[b])
        torch.testing.assert_close(v[:, b], v_b)
        torch.testing.assert_close(iasc[:, b], iasc_b)


def test_exact_no_spike_rejects_multidimensional_time():
    """``t`` must be a scalar or a 1-D tensor; anything else is ambiguous."""
    neuron = _glif()
    with pytest.raises(ValueError, match="1-D"):
        neuron.forward_exact_no_spike(1.0, t=torch.ones(2, 3, dtype=torch.float64))


def test_exact_no_spike_degenerate_tau_equals_inverse_k_is_finite():
    """When ``tau == 1 / k`` the generic formula is 0/0: use its limit.

    The result must be finite (also its gradient) and agree with a nearby,
    non-degenerate ``k`` (continuity of the closed form).
    """
    tau = 20.0
    degenerate = _glif(1, tau=tau, k=[1.0 / tau, 0.3])
    nearby = _glif(1, tau=tau, k=[1.0 / tau * (1 + 1e-6), 0.3])
    iasc0 = torch.ones(1, 2, dtype=torch.float64)
    t = torch.tensor([0.0, 5.0, 40.0], dtype=torch.float64)

    v_deg, _ = degenerate.forward_exact_no_spike(1.0, t=t, Iasc0=iasc0)
    v_near, _ = nearby.forward_exact_no_spike(1.0, t=t, Iasc0=iasc0)

    assert torch.isfinite(v_deg).all()
    torch.testing.assert_close(v_deg, v_near, rtol=1e-4, atol=1e-6)


def test_bfloat16_single_neuron_gradients_are_finite():
    """A bf16 GLIF3 unrolled through ``make_rnn`` yields finite gradients.

    A single neuron receives a step current (on for the first half, then
    off), fires with refractoriness and after-spike currents, and the
    mean membrane potential is back-propagated to the input sequence.
    bfloat16 has only 8 mantissa bits, so this guards against NaN / inf
    in the surrogate gradient or the exponential Euler update in reduced
    precision.
    """
    dtype = torch.bfloat16
    T, dt = 200, 1.0
    neuron = GLIF3(
        1,
        **NEURON_PARAMS,
        detach_reset=False,
        pre_spike_v=True,
        dtype=dtype,
    )
    init_net_state(neuron, dtype=dtype)

    x_seq = torch.cat((torch.full((T // 2,), 5.0), torch.zeros(T // 2)))
    x_seq = x_seq.to(dtype).requires_grad_(True)

    neuron_rnn = rnn.make_rnn(neuron, update_state_names=("v", "Iasc"))
    try:
        with environ.context(dt=dt):
            spike, states = neuron_rnn(x_seq)
        loss = states["v"].float().mean()
        loss.backward()
    except (RuntimeError, NotImplementedError) as e:  # pragma: no cover
        pytest.skip(f"bfloat16 not supported by this CPU op set: {e}")

    assert spike.sum() > 0  # the neuron fired, so reset paths were exercised
    assert x_seq.grad is not None
    assert torch.isfinite(x_seq.grad.float()).all()
    assert x_seq.grad.float().abs().sum() > 0  # the gradient is not trivially 0
