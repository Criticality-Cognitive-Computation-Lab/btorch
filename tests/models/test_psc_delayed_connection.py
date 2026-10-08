"""Ordinary PSC modules driven by a connection with per-edge delays.

A connection built with ``Synapse(delay=...)`` has ``n_delay > 1`` and expects
the delay-expanded input layout ``[..., n_pre * n_delay]`` (index
``pre * n_delay + delay``). Every :class:`~btorch.models.synapse.BasePSC`
detects this through the ``linear.n_delay`` attribute and then owns a
:class:`~btorch.models.history.SpikeHistory` (``psc.history``, memory
``history.history``): each step it pushes the incoming spikes and feeds
``history.get_flattened(n_delay)`` to the connection. So the NEST-style journey
just works::

    proj = Projection(pre, post, rule, Synapse(delay=d))
    psc = AlphaPSC(n_neuron=post, tau_syn=5.0, linear=proj)
    brain = RecurrentNN(neuron, psc)

Timing convention pinned here (project rule): a spike delivered at step ``t``
on an edge with delay ``d`` first changes the PSC *returned* at step ``t + d``;
delay 0 acts in the delivery step.

The tests compare three independent ways to express the same delayed network:

(a) semantic edges, ``SparseConnection.from_edges(..., Synapse(delay=...))``
    passed directly to the PSC (the new path);
(b) a physically delay-expanded matrix in a delay-unaware connection, buffered
    by ``HeterSynapsePSC(max_delay_steps=...)`` (the pre-existing path);
(c) a hand-written dense reference that keeps an explicit list of past spike
    tensors and applies one dense matrix per delay.
"""

import copy
import platform

import pandas as pd
import pytest
import scipy.sparse
import torch

from btorch.models import environ
from btorch.models.connection import (
    FixedIndegree,
    Projection,
    SparseConnection,
    Synapse,
)
from btorch.models.functional import (
    detach_net,
    init_net_state,
    named_hidden_states,
    reset_net,
)
from btorch.models.history import SpikeHistory, expand_delay_sequence
from btorch.models.neurons import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import (
    AlphaPSC,
    AlphaPSCBilleh,
    DelayedPSC,
    DualExponentialPSC,
    ExponentialPSC,
    HeterSynapsePSC,
)
from tests.utils.compile import compile_or_skip


# dt = 1.0 everywhere: AlphaPSCBilleh requires it, and delays are in steps.
DT = 1.0
# float64 end to end, so the three implementations can be compared to ~1e-10
# and a one-step timing error (an O(1) difference) cannot hide in tolerance.
DTYPE = torch.float64
N = 6  # neurons (recurrent: source population == target population)
N_DELAY = 4  # delay bins 0, 1, 2, 3
N_STEP = 12
BATCH = 2


def _f64(x: float) -> torch.Tensor:
    """Time constants as float64 tensors (the PSC buffers keep their dtype)."""
    return torch.tensor(x, dtype=DTYPE)


# One entry per ordinary PSC class: (class, constructor kwargs).
PSC_CASES = [
    pytest.param(ExponentialPSC, {"tau_syn": _f64(3.0)}, id="ExponentialPSC"),
    pytest.param(AlphaPSC, {"tau_syn": _f64(3.0), "g_max": 1.5}, id="AlphaPSC"),
    pytest.param(AlphaPSCBilleh, {"tau_syn": _f64(3.0)}, id="AlphaPSCBilleh"),
    pytest.param(
        DualExponentialPSC,
        {"tau_decay": _f64(4.0), "tau_rise": _f64(1.5)},
        id="DualExponentialPSC",
    ),
]


# ---------------------------------------------------------------------------
# Network description shared by all tests
# ---------------------------------------------------------------------------


def _random_edges(n_pre: int = N, n_post: int = N, n_edge: int = 16, seed: int = 0):
    """Random edge list with one delay in {0, 1, 2, 3} per edge.

    Returns ``(pre, post, delay, weight)``, each of shape ``[n_edge]``. The
    ``(pre, post)`` pairs are unique so no two edges are merged, and all four
    delays are guaranteed to occur.
    """
    g = torch.Generator().manual_seed(seed)
    pair = torch.randperm(n_pre * n_post, generator=g)[:n_edge]
    pre, post = pair // n_post, pair % n_post
    delay = torch.arange(n_edge) % N_DELAY
    delay = delay[torch.randperm(n_edge, generator=g)]
    weight = torch.randn(n_edge, generator=g, dtype=DTYPE)
    return pre, post, delay, weight


def _dense_per_delay(pre, post, delay, weight, n_pre=N, n_post=N, n_delay=N_DELAY):
    """``W[d, post, pre]``: one dense operator per delay (reference layout)."""
    w = torch.zeros(n_delay, n_post, n_pre, dtype=weight.dtype)
    w[delay, post, pre] = weight
    return w


def _direct_connection(pre, post, delay, weight, n_pre=N, n_post=N):
    """(a) Semantic edges: delays are an edge attribute, nothing is expanded.

    ``conn.n_delay == N_DELAY`` and ``conn.in_features == n_pre * N_DELAY``;
    the matrix values become the trainable per-edge weights.
    """
    return SparseConnection.from_edges(
        pre,
        post,
        n_pre,
        n_post,
        Synapse(delay=delay, n_delay=N_DELAY),
        values=weight.clone(),
        dtype=DTYPE,
    )


def _expanded_connection(pre, post, delay, weight, n=N):
    """(b) The same network as a physically delay-expanded matrix.

    Row ``pre * N_DELAY + delay``, column ``post`` (the layout of
    ``expand_conn_for_delays``). ``from_adjacency`` knows nothing about delays
    (``conn.n_delay == 1``); the owner of the PSC must buffer the spikes.
    """
    rows = (pre * N_DELAY + delay).numpy()
    matrix = scipy.sparse.coo_array(
        (weight.numpy(), (rows, post.numpy())), shape=(n * N_DELAY, n)
    )
    return SparseConnection.from_adjacency(matrix, dtype=DTYPE)


def _no_receptor_index() -> pd.DataFrame:
    """Receptor table of a single-receptor ``HeterSynapsePSC``."""
    return pd.DataFrame({"receptor_index": [0], "receptor_type": ["all"]})


def _spikes(shape, seed: int = 1, p: float = 0.4) -> torch.Tensor:
    """Random 0/1 spike trains as a float64 leaf (so input gradients exist)."""
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, generator=g) < p).to(DTYPE).requires_grad_(True)


def _weight_parameter(conn: SparseConnection) -> torch.nn.Parameter:
    """The trained per-edge-slot weight vector of a sparse connection."""
    (param,) = list(conn.weight.parameters())
    assert param.shape == (conn.nnz,)
    return param


def _weight_grad_per_delay(conn: SparseConnection, loss, expanded_rows: bool = False):
    """D(loss)/d(weight) scattered to the dense ``[delay, post, pre]`` layout.

    Edge slots are stored in the connection's canonical order, so gradients of
    different connections are only comparable after mapping each slot back to
    its ``(delay, post, pre)`` coordinates through ``edge_table()``. For the
    physically expanded connection (b) the "pre" id is the expanded row
    ``pre * N_DELAY + delay``.
    """
    (grad,) = torch.autograd.grad(loss, _weight_parameter(conn), retain_graph=True)
    table = conn.edge_table()
    if expanded_rows:
        pre, delay = table["pre"] // N_DELAY, table["pre"] % N_DELAY
    else:
        pre, delay = table["pre"], table["delay"]
    return _dense_per_delay(pre, table["post"], delay, grad)


# ---------------------------------------------------------------------------
# (c) Hand-written dense reference
# ---------------------------------------------------------------------------


def _psc_recurrence(psc_cls, kwargs):
    """Per-step PSC update ``(state, wz) -> (state, psc)`` written by hand.

    Each recurrence is the documented single-step update of the class (see the
    ``get_kernel`` docstrings in ``btorch/models/synapse.py``); the input
    ``wz`` acts in the step it is delivered.
    """
    if psc_cls is ExponentialPSC:
        a = torch.exp(-DT / kwargs["tau_syn"])

        def step(state, wz):
            psc = a * state["psc"] + wz
            return {"psc": psc}, psc

    elif psc_cls is AlphaPSC:
        a = torch.exp(-DT / kwargs["tau_syn"])
        g_max = kwargs["g_max"]

        def step(state, wz):
            h = a * state["h"] + g_max * wz
            psc = a * state["psc"] + (1 - a) * h
            return {"h": h, "psc": psc}, psc

    elif psc_cls is AlphaPSCBilleh:
        tau = kwargs["tau_syn"]
        a = torch.exp(-1.0 / tau)

        def step(state, wz):
            h = a * state["h"] + torch.e / tau * wz
            psc = a * state["psc"] + a * h
            return {"h": h, "psc": psc}, psc

    else:
        tau_d, tau_r = kwargs["tau_decay"], kwargs["tau_rise"]
        a_d, a_r = torch.exp(-DT / tau_d), torch.exp(-DT / tau_r)
        # Peak-normalising amplitude of DualExponentialPSC (its default ``A``).
        peak = tau_d / (tau_d - tau_r) * (tau_r / tau_d) ** (tau_r / (tau_r - tau_d))
        scale = (tau_d - tau_r) / tau_r / tau_d * peak

        def step(state, wz):
            g_r = a_r * state["g_r"] + wz
            g_d = a_d * state["g_d"] + wz
            return {"g_r": g_r, "g_d": g_d}, scale * (a_d * g_d - a_r * g_r)

    return step


def _reference_trajectory(psc_cls, kwargs, w_delay, z_seq):
    """PSC trajectory from an explicit list of past spikes.

    Args:
        w_delay: ``[n_delay, n_post, n_pre]`` dense operator per delay.
        z_seq: ``[T, *batch, n_pre]`` spikes.

    At step ``t`` the list ``past`` is ``[z[t], z[t-1], ..., z[t-D+1]]`` and the
    synaptic input is ``sum_d W[d] @ z[t - d]``: the spike of step ``t`` acts
    in step ``t`` through the delay-0 matrix and in step ``t + d`` through the
    delay-``d`` matrix. This states the timing convention directly.
    """
    step = _psc_recurrence(psc_cls, kwargs)
    zero = torch.zeros(*z_seq.shape[1:-1], w_delay.shape[1], dtype=z_seq.dtype)
    state = {k: zero for k in ("psc", "h", "g_r", "g_d")}
    past: list[torch.Tensor] = []
    out = []
    for t in range(z_seq.shape[0]):
        past.insert(0, z_seq[t])
        del past[w_delay.shape[0] :]
        wz = sum(past[d] @ w_delay[d].T for d in range(len(past)))
        new_state, psc = step(state, wz)
        state = {**state, **new_state}
        out.append(psc)
    return torch.stack(out)


def _run_steps(psc, z_seq):
    """Step a PSC over time with ``single_step_forward`` and stack the
    output."""
    return torch.stack([psc.single_step_forward(z) for z in z_seq])


# ---------------------------------------------------------------------------
# 1. Trajectories and gradients: (a) == (b) == (c)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("psc_cls, kwargs", PSC_CASES)
def test_delayed_connection_matches_expanded_and_dense_reference(psc_cls, kwargs):
    """All three formulations give the same PSC and the same gradients."""
    pre, post, delay, weight = _random_edges()
    z_a, z_b, z_c = (_spikes((N_STEP, BATCH, N)) for _ in range(3))

    with environ.context(dt=DT):
        # (a) New path: the delayed connection goes straight into the PSC.
        conn_a = _direct_connection(pre, post, delay, weight)
        assert conn_a.n_delay == N_DELAY and conn_a.in_features == N * N_DELAY
        psc_a = psc_cls(n_neuron=N, linear=conn_a, **kwargs)
        # The PSC found ``linear.n_delay`` and owns a history of that depth.
        assert psc_a.n_delay == N_DELAY
        assert isinstance(psc_a.history, SpikeHistory)
        init_net_state(psc_a, batch_size=BATCH, dtype=DTYPE)
        out_a = _run_steps(psc_a, z_a)

        # (b) Pre-existing path: expanded matrix + HeterSynapsePSC buffering.
        conn_b = _expanded_connection(pre, post, delay, weight)
        assert conn_b.n_delay == 1 and conn_b.in_features == N * N_DELAY
        psc_b = HeterSynapsePSC(
            n_neuron=N,
            n_receptor=1,
            receptor_type_index=_no_receptor_index(),
            linear=conn_b,
            base_psc=psc_cls,
            max_delay_steps=N_DELAY,
            **kwargs,
        )
        init_net_state(psc_b, batch_size=BATCH, dtype=DTYPE)
        out_b = _run_steps(psc_b, z_b)

    # (c) Hand-written dense reference with an explicit list of past spikes.
    w_c = _dense_per_delay(pre, post, delay, weight).requires_grad_(True)
    out_c = _reference_trajectory(psc_cls, kwargs, w_c, z_c)

    assert out_a.shape == (N_STEP, BATCH, N)
    torch.testing.assert_close(out_a, out_b, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(out_a, out_c, atol=1e-10, rtol=0.0)

    # A non-trivial scalar loss: random readout of the whole trajectory.
    readout = torch.randn(out_a.shape, generator=torch.Generator().manual_seed(7))
    readout = readout.to(DTYPE)
    loss_a, loss_b, loss_c = ((o * readout).sum() for o in (out_a, out_b, out_c))

    # Gradient w.r.t. the weights, compared in the dense [delay, post, pre]
    # layout. The reference has gradients for absent edges too; mask them.
    present = _dense_per_delay(pre, post, delay, torch.ones_like(weight)) > 0
    grad_w_c = torch.autograd.grad(loss_c, w_c, retain_graph=True)[0] * present
    grad_w_a = _weight_grad_per_delay(conn_a, loss_a)
    grad_w_b = _weight_grad_per_delay(conn_b, loss_b, expanded_rows=True)
    assert grad_w_a.abs().sum() > 0
    torch.testing.assert_close(grad_w_a, grad_w_c, atol=1e-9, rtol=0.0)
    torch.testing.assert_close(grad_w_b, grad_w_c, atol=1e-9, rtol=0.0)

    # Gradient w.r.t. the input spikes: it flows back through the history, so
    # the spike of step t collects contributions from steps t .. t + 3.
    grad_z = [
        torch.autograd.grad(loss, z)[0]
        for loss, z in ((loss_a, z_a), (loss_b, z_b), (loss_c, z_c))
    ]
    assert grad_z[0].abs().sum() > 0
    torch.testing.assert_close(grad_z[0], grad_z[1], atol=1e-9, rtol=0.0)
    torch.testing.assert_close(grad_z[0], grad_z[2], atol=1e-9, rtol=0.0)


# ---------------------------------------------------------------------------
# 2. Timing convention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("psc_cls, kwargs", PSC_CASES)
def test_impulse_first_response_at_spike_step_plus_delay(psc_cls, kwargs):
    """Spike at step t on an edge with delay d: first non-zero PSC at t + d.

    Four independent 1 -> 1 edges ``i -> i`` with delay ``i`` (0..3) and unit
    weight. All four neurons spike once, at step ``T_SPIKE``. Target ``d``
    therefore must stay exactly zero up to step ``T_SPIKE + d - 1`` and be
    non-zero at ``T_SPIKE + d`` (delay 0 = the delivery step itself).
    """
    t_spike = 2
    idx = torch.arange(N_DELAY)
    conn = SparseConnection.from_edges(
        idx, idx, N_DELAY, N_DELAY, Synapse(weight=1.0, delay=idx), dtype=DTYPE
    )
    z_seq = torch.zeros(10, N_DELAY, dtype=DTYPE)
    z_seq[t_spike] = 1.0

    with environ.context(dt=DT):
        psc = psc_cls(n_neuron=N_DELAY, linear=conn, **kwargs)
        # No batch dimension: a plain [N] input per step works too.
        init_net_state(psc, dtype=DTYPE)
        out = _run_steps(psc, z_seq)

    assert out.shape == (10, N_DELAY)
    for d in range(N_DELAY):
        first = int((out[:, d] != 0).nonzero()[0])
        assert first == t_spike + d, f"delay {d}: first response at step {first}"
        assert (out[: t_spike + d, d] == 0).all()


# ---------------------------------------------------------------------------
# 3. Multi-step path, batch shapes, non-square populations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("psc_cls, kwargs", PSC_CASES)
@pytest.mark.parametrize("batch", [(), (BATCH,)], ids=["TN", "TBN"])
def test_multi_step_forward_matches_stepping(psc_cls, kwargs, batch):
    """``step_mode="m"`` (kernel convolution) equals stepping from rest.

    The multi-step path of a PSC is stateless: it convolves ``W z`` with the
    impulse response. With a delayed connection the whole sequence is
    delay-expanded at once from an empty history
    (:func:`btorch.models.history.expand_delay_sequence`). Checked for a plain
    ``[T, N]`` and a ``[T, B, N]`` input.
    """
    pre, post, delay, weight = _random_edges()
    z_seq = _spikes((N_STEP, *batch, N)).detach()

    with environ.context(dt=DT):
        stepped = psc_cls(
            n_neuron=N, linear=_direct_connection(pre, post, delay, weight), **kwargs
        )
        init_net_state(stepped, batch_size=batch or None, dtype=DTYPE)
        out_stepped = _run_steps(stepped, z_seq)

        multi = psc_cls(
            n_neuron=N,
            linear=_direct_connection(pre, post, delay, weight),
            step_mode="m",
            **kwargs,
        )
        init_net_state(multi, batch_size=batch or None, dtype=DTYPE)
        # kernel_len >= T, so the truncated kernel covers the whole sequence.
        out_multi = multi(z_seq, kernel_len=N_STEP)

        # Stateless: neither the PSC state nor the history was touched.
        assert multi.history.history.abs().sum() == 0
        assert multi.psc.abs().sum() == 0

    torch.testing.assert_close(out_multi, out_stepped, atol=1e-9, rtol=0.0)

    # The vectorised expansion is the stepped SpikeHistory, written at once.
    history = SpikeHistory(N, max_delay_steps=N_DELAY, use_circular_buffer=False)
    history.init_state(batch_size=batch or None, dtype=DTYPE)
    stepped_expansion = torch.stack(
        [history.update(z).get_flattened(N_DELAY) for z in z_seq]
    )
    torch.testing.assert_close(
        expand_delay_sequence(z_seq, N_DELAY), stepped_expansion, atol=0.0, rtol=0.0
    )


def test_feedforward_source_and_target_of_different_size():
    """Delays on a non-square projection: 5 sources -> 3 targets.

    The history is sized by the *source* population (taken from the
    connection: ``in_features // n_delay``), the PSC state by the target
    population.
    """
    n_pre, n_post = 5, 3
    pre, post, delay, weight = _random_edges(n_pre, n_post, n_edge=11, seed=3)
    z_seq = _spikes((N_STEP, BATCH, n_pre)).detach()
    kwargs = {"tau_syn": _f64(3.0)}

    with environ.context(dt=DT):
        conn = _direct_connection(pre, post, delay, weight, n_pre, n_post)
        psc = AlphaPSC(n_neuron=n_post, linear=conn, **kwargs)
        init_net_state(psc, batch_size=BATCH, dtype=DTYPE)
        assert psc.history.history.shape == (BATCH, N_DELAY, n_pre)
        assert psc.psc.shape == (BATCH, n_post)
        out = _run_steps(psc, z_seq)

    w = _dense_per_delay(pre, post, delay, weight, n_pre, n_post)
    expected = _reference_trajectory(AlphaPSC, {**kwargs, "g_max": 1.0}, w, z_seq)
    torch.testing.assert_close(out, expected, atol=1e-10, rtol=0.0)


# ---------------------------------------------------------------------------
# 4. Delays combined with receptors (HeterSynapsePSC from semantic edges)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_circular_buffer", [False, True], ids=["cat", "circular"])
def test_hetersynapse_psc_delays_and_receptors_from_semantic_edges(
    use_circular_buffer,
):
    """Delay and receptor as edge attributes, buffered by ``HeterSynapsePSC``.

    ``Synapse(delay=..., receptor=...)`` gives a connection with input layout
    ``pre * n_delay + delay`` and output layout ``post * n_receptor +
    receptor``. ``HeterSynapsePSC`` infers ``max_delay_steps`` from
    ``linear.n_delay``; nothing has to be repeated. It is compared with

    - the same network as a matrix expanded in both dimensions with an explicit
      ``max_delay_steps`` (pre-existing path), and
    - the dense reference run once per receptor (receptors are independent
      channels with the same kinetics here, summed at the end).
    """
    n_receptor = 2
    pre, post, delay, weight = _random_edges()
    receptor = (torch.arange(pre.numel()) // 2) % n_receptor
    receptor_index = pd.DataFrame(
        {"receptor_index": [0, 1], "receptor_type": ["E", "I"]}
    )
    kwargs = {"tau_syn": _f64(3.0)}
    z_seq = _spikes((N_STEP, BATCH, N)).detach()

    with environ.context(dt=DT):
        semantic = SparseConnection.from_edges(
            pre,
            post,
            N,
            N,
            Synapse(delay=delay, receptor=receptor, n_delay=N_DELAY, n_receptor=2),
            values=weight.clone(),
            dtype=DTYPE,
        )
        psc = HeterSynapsePSC(
            n_neuron=N,
            n_receptor=n_receptor,
            receptor_type_index=receptor_index,
            linear=semantic,
            base_psc=AlphaPSC,
            use_circular_buffer=use_circular_buffer,
            **kwargs,
        )
        # Inferred from the connection; one history, owned by the outer PSC.
        assert psc.max_delay_steps == psc.n_delay == N_DELAY
        assert psc.base_psc.history is None
        init_net_state(psc, batch_size=BATCH, dtype=DTYPE)
        assert set(named_hidden_states(psc)) == {
            "psc",
            "history.history",
            "base_psc.psc",
            "base_psc.h",
        }

        out, per_receptor = [], []
        for z in z_seq:
            out.append(psc.single_step_forward(z))
            per_receptor.append(torch.stack([psc.get_psc("E"), psc.get_psc("I")]))
        out, per_receptor = torch.stack(out), torch.stack(per_receptor, dim=1)

        # Pre-existing path: rows pre * D + delay, columns post * R + receptor.
        matrix = scipy.sparse.coo_array(
            (
                weight.numpy(),
                (
                    (pre * N_DELAY + delay).numpy(),
                    (post * n_receptor + receptor).numpy(),
                ),
            ),
            shape=(N * N_DELAY, N * n_receptor),
        )
        expanded = HeterSynapsePSC(
            n_neuron=N,
            n_receptor=n_receptor,
            receptor_type_index=receptor_index,
            linear=SparseConnection.from_adjacency(matrix, dtype=DTYPE),
            base_psc=AlphaPSC,
            max_delay_steps=N_DELAY,
            **kwargs,
        )
        init_net_state(expanded, batch_size=BATCH, dtype=DTYPE)
        out_expanded = _run_steps(expanded, z_seq)

    torch.testing.assert_close(out, out_expanded, atol=1e-10, rtol=0.0)

    # Dense reference, one receptor channel at a time.
    ref_kwargs = {**kwargs, "g_max": 1.0}
    reference = []
    for r in range(n_receptor):
        keep = receptor == r
        w_r = _dense_per_delay(pre[keep], post[keep], delay[keep], weight[keep])
        reference.append(_reference_trajectory(AlphaPSC, ref_kwargs, w_r, z_seq))
    reference = torch.stack(reference)  # [receptor, T, B, N]
    torch.testing.assert_close(per_receptor, reference, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(out, reference.sum(0), atol=1e-10, rtol=0.0)


# ---------------------------------------------------------------------------
# 5. Projection -> AlphaPSC -> RecurrentNN with BPTT
# ---------------------------------------------------------------------------


def _delayed_brain(seed: int = 0, **rnn_kwargs):
    """``Projection(..., Synapse(delay=...)) -> AlphaPSC -> RecurrentNN``."""
    n, indegree, n_delay = 8, 3, 3
    g = torch.Generator().manual_seed(seed)
    n_edge = n * indegree
    proj = Projection(
        pre=n,
        post=n,
        rule=FixedIndegree(indegree),
        synapse=Synapse(
            # Per-edge delays in {0, 1, 2}, aligned with the rule's edge list.
            # ``weight=None``: trainable per-edge weights (initialised below).
            delay=torch.randint(0, n_delay, (n_edge,), generator=g),
            n_delay=n_delay,
        ),
        generator=g,
    )
    # One weight per stored edge slot. (The rule may draw the same pair twice;
    # such duplicates with equal delay are merged, so count the slots.)
    weight = _weight_parameter(proj.connection)
    with torch.no_grad():
        weight.copy_(0.2 + 0.6 * torch.rand(weight.numel(), generator=g))
    # ``Projection`` delegates ``n_delay`` to its connection, so it can be
    # handed to the PSC as is. (Fallback for a Projection without that
    # property: pass ``proj.connection``, which always has it.)
    linear = proj if hasattr(proj, "n_delay") else proj.connection
    assert linear.n_delay == n_delay
    brain = RecurrentNN(
        neuron=LIF(n_neuron=n, v_threshold=1.0, v_reset=0.0),
        synapse=AlphaPSC(n_neuron=n, tau_syn=3.0, linear=linear),
        update_state_names=("neuron.v", "synapse.psc"),
        **rnn_kwargs,
    )
    return brain, proj.connection


@pytest.mark.parametrize(
    "rnn_kwargs",
    [
        {},
        {"unroll": 2},
        {"grad_checkpoint": True, "unroll": 2, "chunk_size": 4},
        {"grad_checkpoint": True, "unroll": 3},
    ],
    ids=["default", "unroll2", "checkpoint-chunk4", "checkpoint-unroll3"],
)
def test_projection_alpha_psc_recurrent_nn_bptt(rnn_kwargs):
    """The NEST-style journey runs in ``RecurrentNN`` and trains with BPTT.

    The recurrent loop is compared with a hand-written loop over the same LIF
    model that keeps an explicit list of past spikes and applies one dense
    matrix per delay. Chunking and gradient checkpointing must not change the
    result: the delay history is a registered memory, so it is carried across
    unroll blocks and restored when a checkpointed chunk is recomputed.
    """
    n_step, batch = 10, 2
    brain, conn = _delayed_brain(**rnn_kwargs)
    n = conn.n_post
    x = 1.5 * torch.rand(n_step, batch, n, generator=torch.Generator().manual_seed(5))
    x.requires_grad_(True)

    with environ.context(dt=DT):
        init_net_state(brain, batch_size=batch, dtype=torch.float32)
        # The history sits at ``synapse.history.history`` among the memories.
        assert named_hidden_states(brain)["synapse.history.history"].shape == (
            batch,
            conn.n_delay,
            n,
        )
        spikes, states = brain(x)
        loss = (states["synapse.psc"] ** 2).sum() + spikes.sum()
        grad_w, grad_x = torch.autograd.grad(loss, [_weight_parameter(conn), x])

    assert spikes.shape == (n_step, batch, n)
    assert spikes.sum() > 0, "the drive must make the network spike"
    assert torch.isfinite(grad_w).all() and grad_w.abs().sum() > 0

    # Hand-written loop: same RecurrentNN update order (neuron reads the PSC
    # of the previous step, then the synapse receives the new spikes).
    table = conn.edge_table()
    w_param = _weight_parameter(conn).detach().clone().requires_grad_(True)
    w_delay = torch.zeros(conn.n_delay, n, n).index_put(
        (table["delay"], table["post"], table["pre"]), w_param
    )
    x_ref = x.detach().clone().requires_grad_(True)
    a = torch.exp(torch.tensor(-DT / 3.0))
    with environ.context(dt=DT):
        neuron = LIF(n_neuron=n, v_threshold=1.0, v_reset=0.0)
        init_net_state(neuron, batch_size=batch, dtype=torch.float32)
        h = psc = torch.zeros(batch, n)
        past: list[torch.Tensor] = []
        spikes_ref, psc_ref = [], []
        for t in range(n_step):
            z = neuron(psc + x_ref[t])
            past.insert(0, z)
            del past[conn.n_delay :]
            wz = sum(past[d] @ w_delay[d].T for d in range(len(past)))
            h = a * h + wz
            psc = a * psc + (1 - a) * h
            spikes_ref.append(z)
            psc_ref.append(psc)
    spikes_ref, psc_ref = torch.stack(spikes_ref), torch.stack(psc_ref)
    loss_ref = (psc_ref**2).sum() + spikes_ref.sum()
    grad_w_ref, grad_x_ref = torch.autograd.grad(loss_ref, [w_param, x_ref])

    torch.testing.assert_close(spikes, spikes_ref, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(states["synapse.psc"], psc_ref, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(grad_w, grad_w_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(grad_x, grad_x_ref, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 6. State management: init / reset / detach / state_dict
# ---------------------------------------------------------------------------


def test_reset_net_and_init_net_state_clear_the_history():
    """The history is a registered memory with reset value 0.

    ``init_net_state`` allocates it with the requested batch shape and dtype,
    ``reset_net`` zeroes it (so a second run reproduces the first exactly), and
    a direct ``psc.init_state()`` / ``psc.reset()`` reaches it as well.
    """
    pre, post, delay, weight = _random_edges()
    z_seq = _spikes((N_STEP, BATCH, N)).detach()

    with environ.context(dt=DT):
        psc = AlphaPSC(
            n_neuron=N,
            tau_syn=_f64(3.0),
            linear=_direct_connection(pre, post, delay, weight),
        )
        init_net_state(psc, batch_size=BATCH, dtype=DTYPE)
        history = psc.history
        assert history.history.shape == (BATCH, N_DELAY, N)
        assert history.history.dtype == DTYPE
        assert set(named_hidden_states(psc)) == {"psc", "h", "history.history"}

        first = _run_steps(psc, z_seq)
        assert history.history.abs().sum() > 0

        # reset_net: zero history and PSC state, same batch shape.
        reset_net(psc)
        assert history.history.shape == (BATCH, N_DELAY, N)
        assert history.history.abs().sum() == 0
        assert psc.psc.abs().sum() == 0 and psc.h.abs().sum() == 0
        # Had the history survived, the first steps would differ.
        torch.testing.assert_close(_run_steps(psc, z_seq), first, atol=0.0, rtol=0.0)

        # In-place reset (keeps the buffer objects, as CUDA-graph replay needs).
        buffer = history.history
        reset_net(psc, inplace=True)
        assert history.history is buffer and buffer.abs().sum() == 0

        # init_net_state with another batch size re-allocates everything.
        _run_steps(psc, z_seq[:3])
        init_net_state(psc, batch_size=3, dtype=DTYPE)
        assert history.history.shape == (3, N_DELAY, N)
        assert history.history.abs().sum() == 0

        # Module-level calls (no network walk) cover the history too.
        psc.init_state(batch_size=BATCH, dtype=DTYPE)
        assert history.history.shape == (BATCH, N_DELAY, N)
        _run_steps(psc, z_seq[:3])
        psc.reset()
        assert history.history.abs().sum() == 0


def test_detach_net_cuts_the_graph_through_the_history():
    """``detach_net`` detaches the buffered spikes (truncated BPTT)."""
    pre, post, delay, weight = _random_edges()
    z_seq = _spikes((4, BATCH, N))

    with environ.context(dt=DT):
        psc = ExponentialPSC(
            n_neuron=N,
            tau_syn=_f64(3.0),
            linear=_direct_connection(pre, post, delay, weight),
        )
        init_net_state(psc, batch_size=BATCH, dtype=DTYPE)
        _run_steps(psc, z_seq)
        assert psc.history.history.requires_grad
        detach_net(psc)
        assert not psc.history.history.requires_grad
        assert not psc.psc.requires_grad


def test_state_dict_round_trip_mid_simulation_continues_exactly():
    """Save after 5 steps, load into a fresh module, continue: same result.

    With ``persistent=True`` the memories are part of the ``state_dict``; the
    buffered spikes (``history.history``) must be among them, otherwise the
    restored module would lose the spikes still "in flight" on delayed edges.
    """
    pre, post, delay, weight = _random_edges()
    z_seq = _spikes((N_STEP, BATCH, N)).detach()

    def build():
        psc = AlphaPSC(
            n_neuron=N,
            tau_syn=_f64(3.0),
            linear=_direct_connection(pre, post, delay, weight),
        )
        init_net_state(psc, batch_size=BATCH, dtype=DTYPE, persistent=True)
        return psc

    with environ.context(dt=DT):
        original = build()
        _run_steps(original, z_seq[:5])
        checkpoint = copy.deepcopy(original.state_dict())
        assert {"psc", "h", "history.history"} <= set(checkpoint)
        assert checkpoint["history.history"].abs().sum() > 0

        restored = build()
        restored.load_state_dict(checkpoint)

        tail_original = _run_steps(original, z_seq[5:])
        tail_restored = _run_steps(restored, z_seq[5:])

        # Control: a fresh module without the checkpoint does *not* continue.
        tail_fresh = _run_steps(build(), z_seq[5:])

    torch.testing.assert_close(tail_restored, tail_original, atol=0.0, rtol=0.0)
    assert not torch.allclose(tail_fresh, tail_original)


@pytest.mark.parametrize("psc_cls, kwargs", PSC_CASES)
def test_no_delay_leaves_memories_and_state_dict_unchanged(psc_cls, kwargs):
    """``n_delay == 1``: no history, same memories and keys as before.

    Three delay-free ``linear`` flavours must give the same set of memories
    and (apart from the connection's own entries under ``linear.``) the same
    ``state_dict`` keys: a plain ``torch.nn.Linear`` (no ``n_delay`` attribute
    at all), a sparse connection without delays, and one with an explicit
    delay 0 (``n_delay == 1``).
    """
    pre, post, _, weight = _random_edges()
    z_seq = _spikes((4, BATCH, N)).detach()

    def describe(linear):
        psc = psc_cls(n_neuron=N, linear=linear, **kwargs)
        init_net_state(psc, batch_size=BATCH, dtype=DTYPE, persistent=True)
        assert psc.history is None and psc.n_delay == 1
        assert not any(isinstance(m, SpikeHistory) for m in psc.modules())
        with environ.context(dt=DT):
            out = _run_steps(psc, z_seq)
        memories = set(named_hidden_states(psc))
        keys = {k for k in psc.state_dict() if not k.startswith("linear.")}
        return memories, keys, out

    with environ.context(dt=DT):
        plain = describe(torch.nn.Linear(N, N, bias=False, dtype=DTYPE))
        sparse = describe(
            SparseConnection.from_edges(
                pre, post, N, N, values=weight.clone(), dtype=DTYPE
            )
        )
        delay_zero = describe(
            SparseConnection.from_edges(
                pre, post, N, N, Synapse(delay=0), values=weight.clone(), dtype=DTYPE
            )
        )

    assert plain[0] == sparse[0] == delay_zero[0]
    assert plain[1] == sparse[1] == delay_zero[1]
    assert "history.history" not in plain[0]
    # Delay 0 is no delay: identical currents to the delay-free connection.
    torch.testing.assert_close(sparse[2], delay_zero[2], atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# 7. Validation
# ---------------------------------------------------------------------------


class _DelayedLinear(torch.nn.Linear):
    """A dense layer that declares a delay-expanded input layout.

    The protocol between a PSC and its ``linear`` is just the integer
    attribute ``n_delay``; any module can implement it.
    """

    def __init__(self, in_features: int, out_features: int, n_delay: int):
        super().__init__(in_features, out_features, bias=False)
        self.n_delay = n_delay


def test_validation_errors():
    """Inconsistent delay / receptor layouts fail at construction."""
    pre, post, delay, weight = _random_edges()
    receptor_index = _no_receptor_index()

    def hetero(linear, n_receptor=1, **kwargs):
        return HeterSynapsePSC(
            n_neuron=N,
            n_receptor=n_receptor,
            receptor_type_index=receptor_index,
            linear=linear,
            base_psc=ExponentialPSC,
            tau_syn=2.0,
            **kwargs,
        )

    with environ.context(dt=DT):
        # in_features is not a whole number of delay bins per source neuron.
        with pytest.raises(ValueError, match="in_features == n_source \\* n_delay"):
            ExponentialPSC(n_neuron=N, tau_syn=2.0, linear=_DelayedLinear(10, N, 3))

        # A well-formed dense delayed layer is accepted (3 sources x 3 bins).
        assert ExponentialPSC(N, 2.0, _DelayedLinear(9, N, 3)).n_delay == 3

        delayed = _direct_connection(pre, post, delay, weight)

        # HeterSynapsePSC: explicit max_delay_steps disagreeing with the
        # connection's own delay bins (4).
        with pytest.raises(ValueError, match="conflicts with linear.n_delay=4"):
            hetero(delayed, max_delay_steps=2)
        with pytest.raises(ValueError, match="conflicts with linear.n_delay=4"):
            hetero(delayed, max_delay_steps=1)
        # Agreeing (or omitted) is fine.
        assert hetero(delayed, max_delay_steps=N_DELAY).max_delay_steps == N_DELAY
        assert hetero(delayed).max_delay_steps == N_DELAY

        # HeterSynapsePSC is recurrent: the source population is its own. A
        # delayed connection from 5 sources does not fit 6 neurons.
        pre5, post5, delay5, weight5 = _random_edges(5, N, n_edge=12)
        with pytest.raises(ValueError, match="in_features == n_source \\* n_delay"):
            hetero(_direct_connection(pre5, post5, delay5, weight5, 5, N))

        # Delay-unaware layer with an explicit max_delay_steps: its input must
        # be the expanded size N * max_delay_steps.
        with pytest.raises(ValueError, match="in_features == n_source \\* n_delay"):
            hetero(torch.nn.Linear(N * 2, N, bias=False), max_delay_steps=3)

        # Receptor layout: out_features must be size * n_receptor (this used
        # to surface later, as a reshape/broadcast error in the forward pass).
        with pytest.raises(ValueError, match="expected size \\* n_receptor"):
            hetero(torch.nn.Linear(N, N * 3, bias=False), n_receptor=2)

        # An invalid number of delay bins.
        with pytest.raises(ValueError, match="n_delay must be >= 1"):
            hetero(torch.nn.Linear(N, N, bias=False), max_delay_steps=0)


def test_already_expanded_input_is_rejected_unless_declared():
    """A delayed PSC buffers the spikes itself; do not expand them twice.

    Feeding the delay-expanded layout (a hand-written ``SpikeHistory`` loop)
    to a PSC with a delayed connection raises a clear error.
    ``expect_expanded_input()`` opts out of the internal history; this is what
    ``HeterSynapsePSC`` does for its inner ``base_psc``.
    """
    pre, post, delay, weight = _random_edges()
    z_seq = _spikes((N_STEP, BATCH, N)).detach()

    with environ.context(dt=DT):
        automatic = ExponentialPSC(
            N, _f64(3.0), _direct_connection(pre, post, delay, weight)
        )
        init_net_state(automatic, batch_size=BATCH, dtype=DTYPE)

        manual = ExponentialPSC(
            N, _f64(3.0), _direct_connection(pre, post, delay, weight)
        )
        history = SpikeHistory(N, max_delay_steps=N_DELAY, use_circular_buffer=False)
        history.init_state(batch_size=BATCH, dtype=DTYPE)
        init_net_state(manual, batch_size=BATCH, dtype=DTYPE)

        expanded = history.update(z_seq[0]).get_flattened(N_DELAY)
        with pytest.raises(ValueError, match="expect_expanded_input"):
            manual.single_step_forward(expanded)

        manual.expect_expanded_input()
        assert manual.history is None
        assert set(named_hidden_states(manual)) == {"psc"}
        history.reset()
        out_manual = torch.stack(
            [
                manual.single_step_forward(history.update(z).get_flattened(N_DELAY))
                for z in z_seq
            ]
        )
        out_automatic = _run_steps(automatic, z_seq)

    torch.testing.assert_close(out_manual, out_automatic, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# 8. DelayedPSC composition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("psc_cls, kwargs", PSC_CASES)
def test_delayed_psc_adds_uniform_delay_to_per_edge_delays(psc_cls, kwargs):
    """``DelayedPSC(psc_with_delayed_connection, k)``: delays add up.

    ``DelayedPSC`` delays every spike by ``k`` steps before handing it to the
    wrapped PSC, which then applies the per-edge delay ``d``: the edge acts
    ``k + d`` steps after the spike. The reference is the same dense loop with
    every edge delay shifted by ``k``.
    """
    uniform = 2
    pre, post, delay, weight = _random_edges()
    z_seq = _spikes((N_STEP, BATCH, N)).detach()

    with environ.context(dt=DT):
        wrapped = DelayedPSC(
            psc_cls(
                n_neuron=N,
                linear=_direct_connection(pre, post, delay, weight),
                **kwargs,
            ),
            max_delay_steps=uniform,
        )
        init_net_state(wrapped, batch_size=BATCH, dtype=DTYPE)
        # Two independent histories: the wrapper's and the PSC's.
        assert {"history.history", "psc_module.history.history"} <= set(
            named_hidden_states(wrapped)
        )
        out = _run_steps(wrapped, z_seq)

    w_shifted = _dense_per_delay(
        pre, post, delay + uniform, weight, n_delay=N_DELAY + uniform
    )
    expected = _reference_trajectory(psc_cls, kwargs, w_shifted, z_seq)
    torch.testing.assert_close(out, expected, atol=1e-10, rtol=0.0)
    # Nothing can arrive before the uniform delay has elapsed.
    assert (out[:uniform] == 0).all()


# ---------------------------------------------------------------------------
# 9. torch.compile
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
@pytest.mark.parametrize("psc_cls, kwargs", PSC_CASES)
def test_delayed_connection_compiled_matches_eager(psc_cls, kwargs):
    """A compiled delayed PSC gives the eager outputs and gradients.

    Same mode as the existing PSC compile tests (``torch.compile(module)``,
    ``step_mode="m"`` so one call covers the sequence). The history uses the
    ``torch.cat`` update, and ``n_delay`` is a Python constant resolved at
    construction, so nothing in the step depends on data.
    """
    pre, post, delay, weight = _random_edges()

    def build():
        psc = DelayedPSC(
            psc_cls(
                n_neuron=N,
                linear=_direct_connection(pre, post, delay, weight),
                **kwargs,
            ),
            # max_delay_steps=1: a pure pass-through wrapper whose multi-step
            # path steps ``single_step_forward`` (and with it the history).
            max_delay_steps=1,
        )
        psc.step_mode = "m"
        return psc

    z_eager, z_compiled = _spikes((N_STEP, BATCH, N)), _spikes((N_STEP, BATCH, N))
    with environ.context(dt=DT):
        eager, compiled_module = build(), build()
    compiled = compile_or_skip(compiled_module)

    with environ.context(dt=DT):
        init_net_state(eager, batch_size=BATCH, dtype=DTYPE)
        init_net_state(compiled, batch_size=BATCH, dtype=DTYPE)
        out_eager = eager(z_eager)
        out_compiled = compiled(z_compiled)

    torch.testing.assert_close(out_eager, out_compiled, atol=1e-10, rtol=0.0)

    out_eager.sum().backward()
    out_compiled.sum().backward()
    torch.testing.assert_close(z_eager.grad, z_compiled.grad, atol=1e-9, rtol=0.0)
    torch.testing.assert_close(
        _weight_parameter(eager.psc_module.linear).grad,
        _weight_parameter(compiled_module.psc_module.linear).grad,
        atol=1e-9,
        rtol=0.0,
    )
