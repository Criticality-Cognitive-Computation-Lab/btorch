"""Step-by-step parity between btorch modules and the numpy reference.

``tests/models/numpy_reference.py`` re-implements the neuron / PSC / network
dynamics in plain numpy, independently of torch. Here the btorch modules and
the reference are driven with *identical* inputs and initial states and must
agree at every single step. This pins down the update order and the timing
convention (see the reference's module docstring) and catches silent numeric
regressions that behaviour-only tests would miss.

All comparisons run in float64 on CPU with fixed seeds, so the expected
agreement is at rounding level (``rtol=1e-10``); spikes are compared exactly.
Inputs are continuous random numbers, so a rounding difference cannot flip a
threshold crossing in practice.
"""

import numpy as np
import pytest
import scipy.sparse
import torch

from btorch.models import environ
from btorch.models.connection import SparseConnection
from btorch.models.functional import init_net_state
from btorch.models.linear import Linear
from btorch.models.neurons import GLIF3, LIF
from btorch.models.synapse import (
    AlphaPSC,
    AlphaPSCBilleh,
    DualExponentialPSC,
    ExponentialPSC,
)
from tests.models import numpy_reference as ref


DT = 1.0
# float64 tolerances: the two implementations perform the same arithmetic in a
# slightly different order, so only rounding noise (~1e-16 relative per step,
# a few ulps accumulated over ~100-200 steps) is allowed.
RTOL = 1e-10
ATOL = 1e-12


@pytest.fixture(autouse=True)
def _float64_and_dt():
    # Parameters given as python floats (e.g. ``tau_syn=5.8``) become buffers in
    # torch's *default* dtype. Under float32 default they would be rounded to
    # float32 (5.8 != 5.8f) and parity could only hold to ~1e-7, so the default
    # dtype is switched to float64 for the duration of each test.
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    # AlphaPSCBilleh only supports dt == 1; every other model accepts any dt.
    with environ.context(dt=DT):
        yield
    torch.set_default_dtype(prev)


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# PSC parity
# --------------------------------------------------------------------------- #

N_PRE, N_POST = 5, 4

# name -> (torch factory(linear), numpy factory(weight)). The time constants
# differ per type but are the same on both sides.
PSC_CASES = {
    "exponential": (
        lambda lin: ExponentialPSC(N_POST, tau_syn=3.0, linear=lin),
        lambda w: ref.ExponentialPSC(w, tau_syn=3.0),
    ),
    "alpha": (
        lambda lin: AlphaPSC(N_POST, tau_syn=4.0, linear=lin, g_max=1.5),
        lambda w: ref.AlphaPSC(w, tau_syn=4.0, g_max=1.5),
    ),
    "alpha_billeh": (
        lambda lin: AlphaPSCBilleh(N_POST, tau_syn=5.8, linear=lin),
        lambda w: ref.AlphaPSCBilleh(w, tau_syn=5.8),
    ),
    "dual_exponential": (
        lambda lin: DualExponentialPSC(N_POST, tau_decay=6.0, tau_rise=2.0, linear=lin),
        lambda w: ref.DualExponentialPSC(w, tau_decay=6.0, tau_rise=2.0),
    ),
}


def _psc_pair(name: str, weight: np.ndarray):
    """Build the torch PSC and its numpy twin sharing the same weight."""
    make_torch, make_np = PSC_CASES[name]
    linear = Linear(
        N_PRE,
        N_POST,
        weight=torch.as_tensor(weight).T,
        bias=False,
        dtype=torch.float64,
    )
    psc = make_torch(linear)
    init_net_state(psc, dtype=torch.float64)
    return psc, make_np(weight)


@pytest.mark.parametrize("name", list(PSC_CASES))
def test_psc_impulse_response(name):
    """A single spike gives the same PSC trajectory in btorch and numpy.

    The spike is delivered at step 0 and the PSC returned *at step 0*
    already reflects it (delay 0) for every PSC type.
    """
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(N_PRE, N_POST))
    psc, psc_ref = _psc_pair(name, weight)

    z = np.zeros((40, N_PRE))
    z[0, 1] = 1.0  # one presynaptic neuron spikes once, at step 0

    for t in range(len(z)):
        out = psc(torch.as_tensor(z[t]))
        expected = psc_ref.step(z[t], DT)
        np.testing.assert_allclose(_np(out), expected, rtol=RTOL, atol=ATOL)
        if t == 0:
            # delivery step: the response is non-zero already at t = 0
            assert np.all(expected != 0.0)


@pytest.mark.parametrize("name", list(PSC_CASES))
def test_psc_random_spike_train(name):
    """Random Bernoulli spikes (several per step) match step by step."""
    rng = np.random.default_rng(1)
    weight = rng.normal(size=(N_PRE, N_POST))
    psc, psc_ref = _psc_pair(name, weight)

    z = (rng.random((150, N_PRE)) < 0.2).astype(np.float64)

    for t in range(len(z)):
        out = psc(torch.as_tensor(z[t]))
        expected = psc_ref.step(z[t], DT)
        np.testing.assert_allclose(_np(out), expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("name", ["exponential", "alpha", "dual_exponential"])
def test_psc_sparse_weight_and_dt(name):
    """Sparse connections and a non-unit ``dt`` also match.

    ``AlphaPSCBilleh`` is excluded because it is only defined for ``dt == 1``.
    """
    rng = np.random.default_rng(2)
    dense = rng.normal(size=(N_PRE, N_POST)) * (rng.random((N_PRE, N_POST)) < 0.5)
    sparse = scipy.sparse.coo_array(dense)

    make_torch, make_np = PSC_CASES[name]
    linear = SparseConnection.from_adjacency(sparse, dtype=torch.float64)
    psc = make_torch(linear)
    init_net_state(psc, dtype=torch.float64)
    psc_ref = make_np(sparse)  # the reference accepts scipy sparse directly

    dt = 0.5
    z = (rng.random((100, N_PRE)) < 0.3).astype(np.float64)
    with environ.context(dt=dt):
        for t in range(len(z)):
            out = psc(torch.as_tensor(z[t]))
            expected = psc_ref.step(z[t], dt)
            np.testing.assert_allclose(_np(out), expected, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# Neuron parity
# --------------------------------------------------------------------------- #

N_NEURON = 6


def _run_pair(neuron, neuron_ref, x_seq: np.ndarray, extra=None):
    """Step both neurons over ``x_seq``; assert parity of spikes / v / Iasc.

    Args:
        neuron: btorch neuron (already initialised, float64).
        neuron_ref: numpy twin with the same initial state.
        x_seq: ``(T, n_neuron)`` input currents.
        extra: Optional attribute names (e.g. ``"Iasc"``) compared as well.

    Returns:
        The spike raster ``(T, n_neuron)`` of the reference.
    """
    spikes = []
    for x in x_seq:
        z = neuron(torch.as_tensor(x))
        z_ref = neuron_ref.step(x, DT)
        np.testing.assert_array_equal(_np(z), z_ref)
        np.testing.assert_allclose(_np(neuron.v), neuron_ref.v, rtol=RTOL, atol=ATOL)
        for name in extra or ():
            np.testing.assert_allclose(
                _np(getattr(neuron, name)),
                getattr(neuron_ref, name),
                rtol=RTOL,
                atol=ATOL,
            )
        spikes.append(z_ref)
    return np.stack(spikes)


def _inputs(kind: str, T: int, rng: np.random.Generator, level: float, std: float):
    """Constant or noisy ``(T, N_NEURON)`` input current."""
    if kind == "constant":
        return np.full((T, N_NEURON), level)
    return level + std * rng.normal(size=(T, N_NEURON))


@pytest.mark.parametrize("hard_reset", [False, True])
@pytest.mark.parametrize("tau_ref", [None, 3.0])
@pytest.mark.parametrize("kind", ["constant", "random"])
def test_lif_parity(kind, tau_ref, hard_reset):
    """LIF membrane / spikes agree for constant and noisy drive.

    The parameters are heterogeneous across neurons (arrays) to check
    that per-neuron broadcasting is identical on both sides.
    """
    rng = np.random.default_rng(3)
    params = dict(
        v_threshold=rng.uniform(0.9, 1.1, N_NEURON),
        v_reset=rng.uniform(-0.1, 0.1, N_NEURON),
        c_m=rng.uniform(0.8, 1.2, N_NEURON),
        tau=rng.uniform(15.0, 25.0, N_NEURON),
    )
    neuron = LIF(
        N_NEURON, **params, tau_ref=tau_ref, hard_reset=hard_reset, dtype=torch.float64
    )
    neuron_ref = ref.LIF(N_NEURON, **params, tau_ref=tau_ref, hard_reset=hard_reset)
    init_net_state(neuron, dtype=torch.float64)

    # start at a random subthreshold voltage
    v0 = rng.uniform(0.0, 0.8, N_NEURON)
    neuron.v = torch.as_tensor(v0)
    neuron_ref.v = v0.copy()

    x_seq = _inputs(kind, 200, rng, level=0.12, std=0.1)
    spikes = _run_pair(neuron, neuron_ref, x_seq)
    # make sure the test actually exercises spiking and resetting
    assert 0 < spikes.sum() < spikes.size


def _glif3_pair(rng, tau_ref, hard_reset, n_asc=2):
    """GLIF3 with heterogeneous parameters and ``n_asc`` ASC components."""
    params = dict(
        v_threshold=rng.uniform(-46.0, -44.0, N_NEURON),
        v_reset=rng.uniform(-61.0, -59.0, N_NEURON),
        c_m=rng.uniform(1.5, 2.5, N_NEURON),
        tau=rng.uniform(15.0, 25.0, N_NEURON),
        k=rng.uniform(0.01, 0.3, (N_NEURON, n_asc)),
        asc_amps=rng.uniform(-0.4, 0.2, (N_NEURON, n_asc)),
    )
    neuron = GLIF3(
        N_NEURON, **params, tau_ref=tau_ref, hard_reset=hard_reset, dtype=torch.float64
    )
    neuron_ref = ref.GLIF3(N_NEURON, **params, tau_ref=tau_ref, hard_reset=hard_reset)
    init_net_state(neuron, dtype=torch.float64)
    v0 = rng.uniform(-60.0, -50.0, N_NEURON)
    neuron.v = torch.as_tensor(v0)
    neuron_ref.v = v0.copy()
    return neuron, neuron_ref


@pytest.mark.parametrize("hard_reset", [False, True])
@pytest.mark.parametrize("tau_ref", [None, 2.0])
@pytest.mark.parametrize("kind", ["constant", "random"])
def test_glif3_parity(kind, tau_ref, hard_reset):
    """GLIF3 membrane, spikes and after-spike currents agree step by step.

    ``Iasc`` is compared at every step: it receives ``asc_amps`` at a spike
    and decays exactly with ``exp(-k dt)`` afterwards.
    """
    rng = np.random.default_rng(4)
    neuron, neuron_ref = _glif3_pair(rng, tau_ref, hard_reset)

    x_seq = _inputs(kind, 200, rng, level=1.2, std=0.8)
    spikes = _run_pair(neuron, neuron_ref, x_seq, extra=("Iasc",))
    assert 0 < spikes.sum() < spikes.size
    # after-spike currents must actually have been excited
    assert np.abs(neuron_ref.Iasc).max() > 0


def test_glif3_v_rest_parity():
    """An explicit ``v_rest`` different from ``v_reset`` matches."""
    rng = np.random.default_rng(5)
    params = dict(
        v_threshold=-45.0, v_reset=-60.0, c_m=2.0, tau=20.0, k=[0.05], asc_amps=[-0.1]
    )
    neuron = GLIF3(N_NEURON, **params, v_rest=-70.0, dtype=torch.float64)
    neuron_ref = ref.GLIF3(N_NEURON, **params, v_rest=-70.0)
    init_net_state(neuron, dtype=torch.float64)

    x_seq = _inputs("random", 150, rng, level=8.0, std=4.0)
    spikes = _run_pair(neuron, neuron_ref, x_seq, extra=("Iasc",))
    assert spikes.sum() > 0


# --------------------------------------------------------------------------- #
# Recurrent network parity
# --------------------------------------------------------------------------- #


def test_recurrent_ei_network_parity():
    """A small E/I network of GLIF3 + Billeh alpha PSCs matches for 100 steps.

    80 % excitatory / 20 % inhibitory neurons project through two
    separate PSC channels (E and I) with their own time constants,
    summed into the neuron one step later. This exercises the neuron +
    PSC + recurrent weight pipeline and the one-step feedback latency of
    the network.
    """
    rng = np.random.default_rng(6)
    n_e, n_i = 40, 10
    n = n_e + n_i

    # sparse random E->* and I->* weights (rows = presynaptic neuron)
    mask = rng.random((n, n)) < 0.2
    w_e = np.zeros((n, n))
    w_e[:n_e] = (rng.lognormal(0.0, 0.5, (n_e, n)) * mask[:n_e]) * 0.8
    w_i = np.zeros((n, n))
    w_i[n_e:] = -(rng.lognormal(0.0, 0.5, (n_i, n)) * mask[n_e:]) * 3.0
    tau_e, tau_i = 5.8, 6.5

    params = dict(
        v_threshold=-45.0,
        v_reset=-60.0,
        c_m=2.0,
        tau=20.0,
        k=[1.0 / 80],
        asc_amps=[-0.2],
        tau_ref=2.0,
    )
    neuron = GLIF3(n, **params, dtype=torch.float64)
    neuron_ref = ref.GLIF3(n, **params)
    init_net_state(neuron, dtype=torch.float64)
    v0 = rng.uniform(-60.0, -46.0, n)
    neuron.v = torch.as_tensor(v0)
    neuron_ref.v = v0.copy()

    syn_e = AlphaPSCBilleh(
        n,
        tau_syn=tau_e,
        linear=SparseConnection.from_adjacency(
            scipy.sparse.coo_array(w_e), dtype=torch.float64
        ),
    )
    syn_i = AlphaPSCBilleh(
        n,
        tau_syn=tau_i,
        linear=Linear(
            n,
            n,
            weight=torch.as_tensor(w_i).T,
            bias=False,
            dtype=torch.float64,
        ),
    )
    init_net_state(syn_e, dtype=torch.float64)
    init_net_state(syn_i, dtype=torch.float64)
    network_ref = ref.Network(
        neuron_ref,
        [
            ref.AlphaPSCBilleh(scipy.sparse.csr_array(w_e), tau_syn=tau_e),
            ref.AlphaPSCBilleh(w_i, tau_syn=tau_i),
        ],
    )

    x_seq = 1.5 + 1.0 * rng.normal(size=(100, n))
    spike_count = 0.0
    for x in x_seq:
        # btorch: the same wiring as RecurrentNN.single_step_forward, with two
        # synapse channels: neuron sees the *previous* step's PSC.
        xt = torch.as_tensor(x)
        z = neuron(syn_e.psc + syn_i.psc + xt)
        syn_e(z)
        syn_i(z)
        z_ref = network_ref.step(x, DT)

        np.testing.assert_array_equal(_np(z), z_ref)
        np.testing.assert_allclose(_np(neuron.v), neuron_ref.v, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(
            _np(syn_e.psc), network_ref.synapses[0].psc, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            _np(syn_i.psc), network_ref.synapses[1].psc, rtol=RTOL, atol=ATOL
        )
        spike_count += z_ref.sum()

    # the network must be active (otherwise parity would be trivial)
    assert spike_count > 0
