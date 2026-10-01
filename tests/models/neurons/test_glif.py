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


def test_exact_no_spike_is_pure_and_defaults_to_current_state():
    """The method only reads the module state; it never writes it.

    ``v0`` / ``Iasc0`` default to the current ``self.v`` / ``self.Iasc``, and the
    state is identical before and after the call.
    """
    neuron = _glif()
    neuron.v = torch.tensor([-55.0, -50.0], dtype=torch.float64)
    neuron.Iasc = torch.ones(2, 2, dtype=torch.float64)
    v_before, iasc_before = neuron.v.clone(), neuron.Iasc.clone()

    v, iasc = neuron.forward_exact_no_spike(1.0, t=5.0)  # uses self.v / self.Iasc

    torch.testing.assert_close(neuron.v, v_before)
    torch.testing.assert_close(neuron.Iasc, iasc_before)
    # ... and the defaults really were the current state: same as passing it
    v_exp, iasc_exp = neuron.forward_exact_no_spike(
        1.0, t=5.0, v0=v_before, Iasc0=iasc_before
    )
    torch.testing.assert_close(v, v_exp)
    torch.testing.assert_close(iasc, iasc_exp)


def test_exact_no_spike_composes_by_assigning_the_last_time_point():
    """Evaluating in two legs equals one leg (semigroup property).

    This is also how a simulation is continued from an exact evaluation: assign
    the last time point to the state explicitly. 5 ms then 7 ms must match 12 ms
    in one go, and the assigned state keeps its ``(n_neuron, ...)`` shape.
    """
    neuron = _glif()
    neuron.Iasc = torch.ones(2, 2, dtype=torch.float64)
    one_leg, _ = neuron.forward_exact_no_spike(1.0, t=12.0)

    v, iasc = neuron.forward_exact_no_spike(1.0, t=5.0)
    neuron.v, neuron.Iasc = v[-1], iasc[-1]
    assert neuron.v.shape == (2,)
    assert neuron.Iasc.shape == (2, 2)
    second_leg, _ = neuron.forward_exact_no_spike(1.0, t=7.0)

    torch.testing.assert_close(second_leg, one_leg)


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


def test_exact_no_spike_homo_mode_only_accepts_a_shared_grid():
    """``t_mode="homo"`` (the default) takes a scalar or a 1-D ``(n_time,)``
    grid.

    A multi-dimensional ``t`` is never guessed to mean "per neuron" or "per batch
    element": it is an error that points at ``t_mode="heter"``.
    """
    neuron = _glif()
    t_2d = torch.ones(2, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="heter"):
        neuron.forward_exact_no_spike(1.0, t=t_2d)
    with pytest.raises(ValueError, match="heter"):
        neuron.forward_exact_no_spike(1.0, t=t_2d, t_mode="homo")


def test_exact_no_spike_rejects_unknown_t_mode():
    neuron = _glif()
    with pytest.raises(ValueError, match="t_mode"):
        neuron.forward_exact_no_spike(1.0, t=1.0, t_mode="hetero")


def test_exact_no_spike_heter_mode_rejects_shapes_that_do_not_broadcast():
    """In ``"heter"`` mode the trailing axes of ``t`` must broadcast to the
    state.

    The state here is ``(2,)`` (two neurons); a time grid ``(n_time=2, 3)`` asks
    for three "neurons", so it must fail loudly (torch's own broadcasting error).
    """
    neuron = _glif()
    with pytest.raises(RuntimeError):
        neuron.forward_exact_no_spike(
            1.0, t=torch.ones(2, 3, dtype=torch.float64), t_mode="heter"
        )


def test_exact_no_spike_heter_mode_accepts_a_shared_grid_too():
    """A 1-D grid means the same in both modes (``heter`` is a superset)."""
    neuron = _glif()
    t = torch.tensor([1.0, 3.0, 8.0], dtype=torch.float64)
    homo = neuron.forward_exact_no_spike(1.0, t=t, t_mode="homo")
    heter = neuron.forward_exact_no_spike(1.0, t=t, t_mode="heter")
    for a, b in zip(homo, heter):
        torch.testing.assert_close(a, b)


def _hetero_neuron(n_neuron=3):
    """GLIF3 with different tau / c_m / k / v_reset for every neuron."""
    return _glif(
        n_neuron,
        tau=[15.0, 20.0, 25.0][:n_neuron],
        c_m=[1.5, 2.0, 2.5][:n_neuron],
        v_reset=[-60.0, -62.0, -58.0][:n_neuron],
        v_threshold=-45.0,
        k=[[1 / 80, 0.3], [1 / 60, 0.25], [1 / 90, 0.2]][:n_neuron],
        asc_amps=[[-0.2, -0.1]] * n_neuron,
    )


def _scalar_reference(neuron, x, v0, iasc0, t, i):
    """Closed form for ONE neuron ``i`` in plain python floats.

    Written independently of the vectorised implementation (a double loop over
    after-spike components) so heterogeneous shapes are checked against the
    maths, not against the code under test::

        I_j(t) = I_j(0) exp(-k_j t)
        v(t)   = v_inf + (v0 - v_inf) exp(-t/tau)
                 + sum_j I_j(0)/c_m (exp(-k_j t) - exp(-t/tau)) / (1/tau - k_j)
    """
    import math

    def par(a):  # per-neuron parameter or shared scalar
        return float(a[i] if a.ndim >= 1 and a.shape[0] > 1 else a.reshape(-1)[0])

    tau, c_m, v_reset = par(neuron.tau), par(neuron.c_m), par(neuron.v_reset)
    k = neuron.k[i] if neuron.k.ndim == 2 else neuron.k
    v_inf = v_reset + x * tau / c_m
    v = v_inf + (v0 - v_inf) * math.exp(-t / tau)
    iasc = []
    for j, k_j in enumerate(k.tolist()):
        v += (
            iasc0[j] / c_m * (math.exp(-k_j * t) - math.exp(-t / tau)) / (1 / tau - k_j)
        )
        iasc.append(iasc0[j] * math.exp(-k_j * t))
    return v, iasc


@pytest.mark.parametrize(
    ("t_shape", "n_time"),
    [
        ((4,), 4),  # one grid shared by every batch element and neuron
        ((4, 3), 4),  # a different grid per neuron
        ((4, 2, 1), 4),  # a different grid per batch element
        ((4, 2, 3), 4),  # a different grid per batch element AND neuron
    ],
)
def test_exact_no_spike_heterogeneous_batch_matches_scalar_reference(t_shape, n_time):
    """Batched, heterogeneous ``v0`` / ``Iasc0`` / ``x`` / ``t`` / parameters.

    Everything varies: batch of 2, three neurons with different ``tau``, ``c_m``,
    ``k`` and ``v_reset``, different initial states and inputs per element, and
    (parametrized) several layouts of the time grid. Each output element is
    compared with a plain-python evaluation of the closed form for that single
    (batch element, neuron, time).
    """
    torch.manual_seed(0)
    batch, n = 2, 3
    neuron = _hetero_neuron(n)
    v0 = -60.0 + 8.0 * torch.rand(batch, n, dtype=torch.float64)
    iasc0 = torch.rand(batch, n, 2, dtype=torch.float64)
    x = 2.0 * torch.rand(batch, n, dtype=torch.float64)
    t = 0.5 + 20.0 * torch.rand(*t_shape, dtype=torch.float64)

    v, iasc = neuron.forward_exact_no_spike(x, t=t, v0=v0, Iasc0=iasc0, t_mode="heter")

    assert v.shape == (n_time, batch, n)
    assert iasc.shape == (n_time, batch, n, 2)
    # broadcast the time grid to (n_time, batch, n) the same way the API documents
    t_full = t.reshape(n_time, *([1] * (3 - t.ndim)), *t.shape[1:]).expand(
        n_time, batch, n
    )
    for k in range(n_time):
        for b in range(batch):
            for i in range(n):
                v_ref, iasc_ref = _scalar_reference(
                    neuron,
                    x[b, i].item(),
                    v0[b, i].item(),
                    iasc0[b, i].tolist(),
                    t_full[k, b, i].item(),
                    i,
                )
                assert v[k, b, i].item() == pytest.approx(v_ref, rel=1e-10, abs=1e-10)
                for j in range(2):
                    assert iasc[k, b, i, j].item() == pytest.approx(
                        iasc_ref[j], rel=1e-10, abs=1e-12
                    )


def test_exact_no_spike_time_axes_extend_an_unbatched_state():
    """If only ``t`` carries a batch axis, the output gains it.

    A single shared initial state ``(n_neuron,)`` evaluated at a different time
    per batch element ``t: (n_time, batch, 1)`` returns ``(n_time, batch,
    n_neuron)``, and element ``b`` equals an independent call with that
    element's own time grid.
    """
    n, batch = 3, 4
    neuron = _hetero_neuron(n)
    v0 = torch.full((n,), -55.0, dtype=torch.float64)
    iasc0 = torch.ones(n, 2, dtype=torch.float64)
    t = torch.rand(5, batch, 1, dtype=torch.float64) * 30.0

    v, iasc = neuron.forward_exact_no_spike(
        1.5, t=t, v0=v0, Iasc0=iasc0, t_mode="heter"
    )

    assert v.shape == (5, batch, n)
    assert iasc.shape == (5, batch, n, 2)
    for b in range(batch):
        v_b, iasc_b = neuron.forward_exact_no_spike(
            1.5, t=t[:, b, 0], v0=v0, Iasc0=iasc0
        )
        torch.testing.assert_close(v[:, b], v_b)
        torch.testing.assert_close(iasc[:, b], iasc_b)


def test_exact_no_spike_state_batch_independent_of_iasc_batch():
    """``v0`` and ``Iasc0`` may have different (broadcastable) batch shapes.

    ``v0`` is batched ``(batch, n)`` while ``Iasc0`` is shared ``(n, n_Iasc)``;
    the result must equal giving ``Iasc0`` explicitly expanded to the batch.
    """
    n, batch = 3, 2
    neuron = _hetero_neuron(n)
    v0 = -60.0 + 8.0 * torch.rand(batch, n, dtype=torch.float64)
    iasc0 = torch.rand(n, 2, dtype=torch.float64)
    t = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)

    v, iasc = neuron.forward_exact_no_spike(1.0, t=t, v0=v0, Iasc0=iasc0)
    v_exp, iasc_exp = neuron.forward_exact_no_spike(
        1.0, t=t, v0=v0, Iasc0=iasc0.expand(batch, n, 2).clone()
    )

    assert v.shape == (3, batch, n)
    torch.testing.assert_close(v, v_exp)
    torch.testing.assert_close(iasc, iasc_exp)


def test_exact_no_spike_at_matches_scalar_reference_elementwise():
    """``exact_no_spike_at`` is elementwise: every ``t`` has the state's shape.

    There is no time axis. Each (batch element, neuron) gets its own ``x``,
    ``t``, ``v0``, ``Iasc0`` and its own parameters, and is compared with a
    plain-python evaluation of the closed form.
    """
    torch.manual_seed(1)
    batch, n = 3, 3
    neuron = _hetero_neuron(n)
    v0 = -60.0 + 8.0 * torch.rand(batch, n, dtype=torch.float64)
    iasc0 = torch.rand(batch, n, 2, dtype=torch.float64)
    x = 2.0 * torch.rand(batch, n, dtype=torch.float64)
    t = 0.5 + 20.0 * torch.rand(batch, n, dtype=torch.float64)

    v, iasc = neuron.exact_no_spike_at(x, t, v0, iasc0)

    assert v.shape == (batch, n)
    assert iasc.shape == (batch, n, 2)
    for b in range(batch):
        for i in range(n):
            v_ref, iasc_ref = _scalar_reference(
                neuron,
                x[b, i].item(),
                v0[b, i].item(),
                iasc0[b, i].tolist(),
                t[b, i].item(),
                i,
            )
            assert v[b, i].item() == pytest.approx(v_ref, rel=1e-10, abs=1e-10)
            for j in range(2):
                assert iasc[b, i, j].item() == pytest.approx(
                    iasc_ref[j], rel=1e-10, abs=1e-12
                )


def test_forward_exact_no_spike_is_a_stack_of_exact_no_spike_at():
    """The trajectory method is exactly ``exact_no_spike_at`` per time
    point."""
    neuron = _hetero_neuron(3)
    v0 = torch.full((2, 3), -55.0, dtype=torch.float64)
    iasc0 = torch.ones(2, 3, 2, dtype=torch.float64)
    t = torch.tensor([0.0, 2.0, 5.0, 11.0], dtype=torch.float64)

    v, iasc = neuron.forward_exact_no_spike(1.5, t=t, v0=v0, Iasc0=iasc0)

    for k, t_k in enumerate(t):
        v_k, iasc_k = neuron.exact_no_spike_at(1.5, t_k, v0, iasc0)
        torch.testing.assert_close(v[k], v_k)
        torch.testing.assert_close(iasc[k], iasc_k)


def _crossing_problem():
    """Heterogeneous batch where every neuron's ``v`` reaches threshold.

    ``x`` is large enough that ``v_inf`` is above the threshold for every
    element, the initial voltage is below it, and ``Iasc0`` is small, so ``v(t)``
    crosses the threshold exactly once.
    """
    torch.manual_seed(2)
    batch, n = 4, 3
    neuron = _hetero_neuron(n)
    v0 = -60.0 + 5.0 * torch.rand(batch, n, dtype=torch.float64)
    iasc0 = 0.05 * torch.rand(batch, n, 2, dtype=torch.float64)
    x = 3.0 + torch.rand(batch, n, dtype=torch.float64)
    return neuron, x, v0, iasc0


def test_exact_no_spike_at_supports_bisection_root_finding():
    """Iterative root finding with a different ``t`` per element and iteration.

    Bisection on ``v(t) - v_th`` solves, for all (batch, neuron) pairs at once,
    for the time at which the membrane first reaches the threshold. Every
    iteration evaluates the exact solution at an element-specific time of the
    state's shape, which is why the primitive takes no time axis. The converged
    ``t`` must reproduce the threshold.
    """
    neuron, x, v0, iasc0 = _crossing_problem()
    v_th = neuron.v_threshold.to(torch.float64)

    lo = torch.zeros_like(v0)
    hi = torch.full_like(v0, 200.0)
    v_hi, _ = neuron.exact_no_spike_at(x, hi, v0, iasc0)
    assert (v_hi > v_th).all()  # bracket is valid for every element

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        v_mid, _ = neuron.exact_no_spike_at(x, mid, v0, iasc0)
        above = v_mid >= v_th
        hi = torch.where(above, mid, hi)
        lo = torch.where(above, lo, mid)

    t_star = 0.5 * (lo + hi)
    v_star, _ = neuron.exact_no_spike_at(x, t_star, v0, iasc0)
    torch.testing.assert_close(v_star, v_th.expand_as(v_star), atol=1e-9, rtol=0)
    assert t_star.shape == v0.shape and (t_star > 0).all()
    # the crossing times differ between elements (heterogeneous problem)
    assert t_star.std() > 0.1


def test_exact_no_spike_at_supports_newton_with_model_derivative():
    """Newton's method using the model's own ODE for ``dv/dt``.

    ``dV`` gives the exact time derivative of ``v`` at the current
    ``(v, Iasc)``, so a Newton step needs no autograd. The iteration must
    converge to the same crossing times as bisection.
    """
    neuron, x, v0, iasc0 = _crossing_problem()
    v_th = neuron.v_threshold.to(torch.float64)

    t = torch.full_like(v0, 5.0)
    for _ in range(40):
        v, iasc = neuron.exact_no_spike_at(x, t, v0, iasc0)
        dv_dt, _ = neuron.dV(v, iasc, x)
        t = t - (v - v_th) / dv_dt

    v_final, _ = neuron.exact_no_spike_at(x, t, v0, iasc0)
    torch.testing.assert_close(v_final, v_th.expand_as(v_final), atol=1e-9, rtol=0)


def test_exact_no_spike_at_gradient_wrt_t_equals_the_ode_derivative():
    """Autograd through ``t`` matches ``dV``: the primitive is differentiable.

    ``d v(t) / d t`` computed by autograd must equal the model's own
    ``dV(v(t), Iasc(t), x)``, element by element.
    """
    neuron, x, v0, iasc0 = _crossing_problem()
    t = (0.5 + 10.0 * torch.rand_like(v0)).requires_grad_(True)

    v, iasc = neuron.exact_no_spike_at(x, t, v0, iasc0)
    (grad_t,) = torch.autograd.grad(v.sum(), t)
    dv_dt, _ = neuron.dV(v.detach(), iasc.detach(), x)

    torch.testing.assert_close(grad_t, dv_dt, rtol=1e-8, atol=1e-10)


def test_exact_no_spike_methods_capture_as_a_single_compiled_graph():
    """Both methods trace with ``fullgraph=True`` and match eager results.

    Root finding calls the primitive in a tight loop, so it must stay
    ``torch.compile`` friendly (no data-dependent python control flow or shape
    guessing). ``aot_eager`` checks graph capture without needing a C++ toolchain.
    """
    neuron, x, v0, iasc0 = _crossing_problem()
    t_point = torch.full_like(v0, 7.0)
    t_grid = torch.linspace(0, 10, 5, dtype=torch.float64)
    cases = [
        (neuron.exact_no_spike_at, (x, t_point, v0, iasc0)),
        (neuron.forward_exact_no_spike, (x, t_grid, v0, iasc0)),
    ]
    for fn, args in cases:
        eager = fn(*args)
        compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)(*args)
        for a, b in zip(eager, compiled):
            torch.testing.assert_close(a, b)


def test_exact_no_spike_at_rejects_shapes_that_do_not_broadcast():
    """Mismatched element shapes are an error, not a silent broadcast."""
    neuron = _glif(2)
    with pytest.raises(RuntimeError):
        neuron.exact_no_spike_at(
            torch.ones(3, dtype=torch.float64), torch.ones(4, dtype=torch.float64)
        )


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
