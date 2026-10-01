"""Behavioural tests for :class:`~btorch.models.neurons.alif.ALIF` / ``ELIF``.

ALIF dynamics (one step, in this order):

1. charge:     ``dv/dt = (-g_leak (v - E_leak) - g_k (v - E_k) + x) / c_m``
2. adapt:      ``dg_k/dt = -g_k / tau_adapt``        (exponential decay)
3. fire:       spike if ``v > v_threshold``
4. reset:      ``v -= (v_th - v_reset) * spike``; ``g_k += dg_k * spike``

The spike-triggered ``g_k`` increment is a potassium conductance pulling
``v`` towards ``E_k`` (< rest), i.e. negative feedback that lowers the firing
rate. All tests run on CPU with tiny networks.
"""

import math

import pytest
import torch

from btorch.models import environ
from btorch.models.functional import init_net_state, reset_net_state
from btorch.models.neurons.alif import ALIF, ELIF


DT = 1.0


def _make(n=1, **kw):
    """Build an ALIF with simple, easy-to-reason-about defaults."""
    params = dict(
        v_threshold=1.0,
        v_reset=0.0,
        c_m=1.0,
        g_leak=0.1,
        E_leak=0.0,
        E_k=-1.0,
        tau_adapt=20.0,
        dg_k=0.0,
    )
    params.update(kw)
    neuron = ALIF(n_neuron=n, dtype=torch.float64, **params)
    init_net_state(neuron, dtype=torch.float64)
    return neuron


def _run(neuron, current, steps, batch_shape=()):
    """Drive the neuron with constant ``current`` and return (spikes, g_k,
    v)."""
    spikes, gks, vs = [], [], []
    x = torch.full((*batch_shape, neuron.n_neuron[0]), float(current))
    x = x.to(torch.float64)
    with torch.no_grad(), environ.context(dt=DT):
        for _ in range(steps):
            spikes.append(neuron(x).clone())
            gks.append(neuron.g_k.clone())
            vs.append(neuron.v.clone())
    return torch.stack(spikes), torch.stack(gks), torch.stack(vs)


def test_subthreshold_relaxes_to_analytic_fixed_point():
    """Without spikes and with g_k = 0 the ODE is a leaky integrator.

    dv/dt = -g (v - E) + I  =>  v* = E + I/g and exp-Euler is exact for the
    linear ODE, so v(t) = v* (1 - exp(-g t / c_m)) for v(0) = E = 0.
    """
    neuron = _make(g_leak=0.1)
    current = 0.05  # v* = 0.5 < threshold 1.0
    spikes, _, v = _run(neuron, current, steps=30)
    assert spikes.sum() == 0
    t = torch.arange(1, 31, dtype=torch.float64) * DT
    expected = 0.5 * (1 - torch.exp(-0.1 * t))
    # params are stored in float32 (0.1 != 0.1 in float64), hence ~1e-7 slack
    torch.testing.assert_close(v[:, 0], expected, rtol=1e-6, atol=1e-6)


def test_spike_increments_adaptation_conductance():
    """On a spike, g_k jumps by dg_k (step decay of 0 happens before)."""
    neuron = _make(dg_k=0.3)
    # strong input: spikes on the very first step
    spikes, g_k, _ = _run(neuron, current=5.0, steps=1)
    assert spikes[0, 0].item() == 1.0
    assert g_k[0, 0].item() == pytest.approx(0.3)


def test_adaptation_conductance_decays_exponentially_without_spikes():
    """With no further spikes g_k follows exp(-t / tau_adapt) exactly."""
    neuron = _make(dg_k=0.0, g_k_init=0.8, tau_adapt=20.0)
    _, g_k, _ = _run(neuron, current=0.0, steps=10)
    t = torch.arange(1, 11, dtype=torch.float64) * DT
    torch.testing.assert_close(
        g_k[:, 0], 0.8 * torch.exp(-t / 20.0), rtol=1e-10, atol=1e-12
    )


def test_adaptation_reduces_firing_rate():
    """Spike-frequency adaptation: same drive, fewer spikes with dg_k > 0.

    The non-adapting neuron fires regularly; the adapting one accumulates
    g_k, which clamps v towards E_k, so its total spike count is strictly
    smaller and its inter-spike intervals lengthen over time.
    """
    kw = dict(g_leak=0.1, E_k=-2.0, tau_adapt=100.0)
    plain = _make(dg_k=0.0, **kw)
    adapt = _make(dg_k=0.05, **kw)
    drive, steps = 0.3, 300
    s_plain, _, _ = _run(plain, drive, steps)
    s_adapt, g_k, _ = _run(adapt, drive, steps)

    assert s_plain.sum() > 10  # sanity: the drive is suprathreshold
    assert s_adapt.sum() < s_plain.sum()
    # the conductance actually built up, in proportion to the spike count
    assert g_k.max() > 0.05

    # ISIs grow: the last interval is longer than the first
    times = torch.nonzero(s_adapt[:, 0]).flatten()
    assert len(times) >= 3
    isis = times[1:] - times[:-1]
    assert isis[-1] > isis[0]


def test_adaptation_raises_effective_threshold_current():
    """After a spike the rheobase rises: a current that re-fires a fresh
    neuron no longer does once g_k has been incremented."""
    # With g_leak=0.1 and I=0.11 the fixed point 1.1 is just above threshold.
    drive, steps = 0.11, 120
    fresh = _make(g_leak=0.1, dg_k=0.0)
    s_fresh, _, _ = _run(fresh, drive, steps)
    assert s_fresh.sum() >= 1

    # Start fully adapted (large g_k, very slow decay): fixed point drops
    # to I / (g_leak + g_k) + ... < threshold, so it never fires.
    adapted = _make(g_leak=0.1, dg_k=0.0, g_k_init=0.5, tau_adapt=1e6)
    s_adapted, _, v = _run(adapted, drive, steps)
    assert s_adapted.sum() == 0
    assert v.max() < 1.0


def test_soft_vs_hard_reset():
    """Soft reset subtracts (v_th - v_reset); hard reset sets v = v_reset."""
    soft = _make(v_threshold=1.0, v_reset=0.0, hard_reset=False)
    hard = _make(v_threshold=1.0, v_reset=0.0, hard_reset=True)
    # big single-step input overshoots the threshold
    _, _, v_soft = _run(soft, 3.0, 1)
    _, _, v_hard = _run(hard, 3.0, 1)
    assert v_hard[0, 0].item() == pytest.approx(0.0)
    # 3.0 input with exp-Euler step of the leaky ODE, minus threshold
    v_pre = (3.0 / 0.1) * (1 - math.exp(-0.1))
    assert v_soft[0, 0].item() == pytest.approx(v_pre - 1.0)


def test_refractory_blocks_spikes_after_spike():
    """With tau_ref=3 ms the neuron is silent right after each spike."""
    neuron = _make(tau_ref=3.0, g_leak=0.1)
    spikes, _, _ = _run(neuron, current=10.0, steps=20)
    times = torch.nonzero(spikes[:, 0]).flatten()
    assert len(times) >= 2
    # ISI is at least the refractory period
    assert (times[1:] - times[:-1]).min() >= 3

    free = _make(tau_ref=None, g_leak=0.1)
    s_free, _, _ = _run(free, current=10.0, steps=20)
    assert s_free.sum() > spikes.sum()


def test_tau_ref_none_has_no_refractory_state():
    """``tau_ref=None`` disables refractory entirely (no state buffer)."""
    neuron = _make(tau_ref=None)
    assert neuron.tau_ref is None
    assert not hasattr(neuron, "refractory") or neuron.refractory is None
    assert "tau_ref=None" in repr(neuron)


def test_state_shapes_and_batch_reset():
    """States follow (*batch, n_neuron); reset_net_state restores g_k/v."""
    neuron = ALIF(n_neuron=4, dg_k=0.2, g_k_init=0.1, E_k=-1.0, dtype=torch.float64)
    init_net_state(neuron, batch_size=3, dtype=torch.float64)
    assert neuron.v.shape == (3, 4)
    assert neuron.g_k.shape == (3, 4)

    x = torch.full((3, 4), 5.0, dtype=torch.float64)
    with torch.no_grad(), environ.context(dt=DT):
        neuron(x)
    assert (neuron.g_k > 0.1).all()  # the spike incremented g_k everywhere

    reset_net_state(neuron, batch_size=3)
    torch.testing.assert_close(neuron.g_k, torch.full((3, 4), 0.1, dtype=torch.float64))
    assert torch.all(neuron.v == neuron.v_reset)


def test_per_neuron_parameters_are_respected():
    """Vector dg_k: only the neuron with dg_k > 0 adapts."""
    neuron = ALIF(
        n_neuron=2,
        dg_k=torch.tensor([0.0, 0.4]),
        dtype=torch.float64,
    )
    init_net_state(neuron, dtype=torch.float64)
    x = torch.full((2,), 5.0, dtype=torch.float64)
    with torch.no_grad(), environ.context(dt=DT):
        spike = neuron(x)
    assert spike.tolist() == [1.0, 1.0]
    assert neuron.g_k[0].item() == pytest.approx(0.0)
    assert neuron.g_k[1].item() == pytest.approx(0.4)


def test_surrogate_gradient_reaches_dg_k_and_input():
    """Training path: gradients flow to trainable dg_k and to the input."""
    neuron = ALIF(
        n_neuron=2,
        v_threshold=1.0,
        dg_k=0.1,
        g_leak=0.1,
        trainable_param={"dg_k"},
        dtype=torch.float64,
    )
    init_net_state(neuron, dtype=torch.float64)
    x = torch.full((2,), 0.9, dtype=torch.float64, requires_grad=True)
    total = 0.0
    with environ.context(dt=DT):
        for _ in range(10):
            total = total + neuron(x).sum() + neuron.v.sum()
    total.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert neuron.dg_k.grad is not None
    assert torch.isfinite(neuron.dg_k.grad).all()


def test_multi_step_matches_single_step():
    """step_mode='m' equals stepping manually over the same input."""
    a = _make(n=3, dg_k=0.1)
    b = ALIF(
        n_neuron=3,
        v_threshold=1.0,
        v_reset=0.0,
        g_leak=0.1,
        E_k=-1.0,
        dg_k=0.1,
        step_mode="m",
        dtype=torch.float64,
    )
    init_net_state(b, dtype=torch.float64)
    x = torch.rand(25, 3, dtype=torch.float64) * 0.5
    with torch.no_grad(), environ.context(dt=DT):
        manual = torch.stack([a(x[t]) for t in range(25)])
        batched = b(x)
    torch.testing.assert_close(manual, batched)
    torch.testing.assert_close(a.g_k, b.g_k)


# --------------------------------------------------------------------------
# ELIF: ALIF plus an exponential spike-initiation term
# --------------------------------------------------------------------------


def test_elif_exponential_term_accelerates_depolarisation():
    """Near v_T the exponential term adds an upswing that ALIF lacks.

    Start both models slightly above v_T with no input and no adaptation: ALIF
    just decays towards E_leak, ELIF is pushed *up* because
    g_leak * delta_T * exp((v - v_T) / delta_T) exceeds the leak.

    (Starting exactly at v_T with delta_T=1 would make the net linear
    coefficient exactly 0, which hits a division by zero in exp_euler_step;
    see ``test_exp_euler_zero_linear_term`` in test_ode.py.)
    """
    kw = dict(v_threshold=10.0, g_leak=0.1, E_leak=0.0)
    alif = ALIF(n_neuron=1, dtype=torch.float64, **kw)
    elif_ = ELIF(n_neuron=1, delta_T=1.0, v_T=0.5, dtype=torch.float64, **kw)
    for m in (alif, elif_):
        init_net_state(m, dtype=torch.float64)
        m.v = torch.full((1,), 0.7, dtype=torch.float64)
    x = torch.zeros(1, dtype=torch.float64)
    with torch.no_grad(), environ.context(dt=DT):
        alif(x)
        elif_(x)
    assert alif.v.item() < 0.7  # leaks down
    assert elif_.v.item() > 0.7  # exponential drive pushes up


def test_elif_reduces_to_alif_for_vanishing_exponential():
    """With v_T far above v the exponential term is ~0 => matches ALIF."""
    kw = dict(v_threshold=1.0, g_leak=0.1, E_k=-1.0, dg_k=0.1)
    a = ALIF(n_neuron=2, dtype=torch.float64, **kw)
    e = ELIF(n_neuron=2, delta_T=1.0, v_T=1e3, tau_ref=None, dtype=torch.float64, **kw)
    for m in (a, e):
        init_net_state(m, dtype=torch.float64)
    x = torch.full((2,), 0.4, dtype=torch.float64)
    with torch.no_grad(), environ.context(dt=DT):
        for _ in range(40):
            sa, se = a(x), e(x)
            torch.testing.assert_close(sa, se)
    torch.testing.assert_close(a.v, e.v, rtol=1e-9, atol=1e-9)
