"""CPU recurrence oracles for heterogeneous conductance and online learning.

Expected voltages, receptor conductances, arrivals and weight transitions use
explicit event histories, never another neuron/PSC/plasticity forward method.
Created 2026-11-18; execute this file for its current validation status.
"""

import copy
import io

import pytest
import torch

from btorch.models import environ
from btorch.models.connection import (
    ExternalSignal,
    ModulatedSTDP,
    ModulatedSTDPOptions,
    PairSTDP,
    PlasticityRule,
    PostState,
    SparseConnection,
    Synapse,
)
from btorch.models.functional import init_net_state
from btorch.models.kernels import KernelOptions
from btorch.models.neurons.lif import LIF
from btorch.models.rnn import ConductanceRecurrentNN, RecurrentNN
from btorch.models.synapse import HeterConductancePSC


DT = 0.5
DTYPE = torch.float64
N, R, D = 3, 2, 3
# Unsorted semantic edges include two receptors on the same physical pair.
PRE = torch.tensor([2, 0, 1, 0, 2, 1])
POST = torch.tensor([0, 1, 2, 1, 1, 0])
RECEPTOR = torch.tensor([1, 0, 0, 1, 0, 1])
DELAY = torch.tensor([1, 2, 0, 0, 1, 2])
WEIGHT = torch.tensor([0.04, 0.12, 0.09, 0.06, 0.07, 0.05], dtype=DTYPE)
LOWER = torch.where(RECEPTOR == 1, WEIGHT, WEIGHT - 0.015)
UPPER = torch.where(RECEPTOR == 1, WEIGHT, WEIGHT + 0.02)
TAU = torch.tensor([2.0, 5.0], dtype=DTYPE)
REVERSAL = torch.tensor([1.5, -1.0], dtype=DTYPE)
INITIAL_G = torch.tensor([[0.02, 0.01], [0.03, 0.04], [0.01, 0.02]], dtype=DTYPE)
INITIAL_V = torch.tensor([-0.2, 0.1, 0.35], dtype=DTYPE)
# Most recent prehistory event first; update oldest first when seeding Delay.
PREHISTORY = torch.tensor([[0.0, 1.0, 0.0], [0.5, 0.0, 1.0]], dtype=DTYPE)
INITIAL_PRE = 0.4 + 0.05 * torch.arange(len(PRE), dtype=DTYPE)
INITIAL_POST = 0.6 - 0.04 * torch.arange(len(PRE), dtype=DTYPE)
INITIAL_ELIGIBILITY = 0.02 * torch.arange(1, len(PRE) + 1, dtype=DTYPE)
DRIVE = torch.tensor(
    [
        [3.0, 0.0, 3.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 3.0],
        [3.0, 3.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 3.0, 3.0],
        [3.0, 0.0, 0.0],
        [0.0, 0.0, 3.0],
        [3.0, 3.0, 0.0],
    ],
    dtype=DTYPE,
)
EXTRA = torch.zeros_like(DRIVE)
EXTRA[1, 0] = 0.25
EXTRA[4, 2] = 0.5
REWARDS = torch.tensor(
    [
        [1.0, -2.0, 3.0],
        [-2.0, 3.0, 1.0],
        [3.0, 1.0, -2.0],
        [-1.0, 2.0, -3.0],
        [2.0, -3.0, 1.0],
        [3.0, -1.0, 2.0],
        [-3.0, 2.0, 1.0],
        [1.0, 3.0, -2.0],
        [2.0, -1.0, 3.0],
    ],
    dtype=DTYPE,
)


class _VoltageReadout(PlasticityRule):
    """Expose a live PostState stage through a post-domain persistent memory."""

    state_domains = {"observed": "post"}

    def __init__(self, stage):
        super().__init__(signals={"voltage": PostState("v", stage=stage)})
        self.register_memory("observed", 0.0, (), persistent=True)

    def single_step_forward(self, weight, events, dt):
        self.observed.copy_(events.signals["voltage"])
        return torch.zeros_like(weight)


def _canonical_order(connection):
    """Map declared edge identities to storage without assuming CSR order."""
    original = list(zip(PRE.tolist(), POST.tolist(), RECEPTOR.tolist(), DELAY.tolist()))
    actual = list(
        zip(
            connection.pre.tolist(),
            connection.post.tolist(),
            connection.receptor.tolist(),
            connection.delay.tolist(),
        )
    )
    assert sorted(actual) == sorted(original)
    return torch.tensor([original.index(edge) for edge in actual])


def _build(
    kind="static",
    *,
    chunk_size=None,
    backend="torch",
    circular=False,
    network_type=ConductanceRecurrentNN,
):
    rule = None
    if kind == "pair":
        rule = PairSTDP(
            a_plus=torch.tensor(0.4, dtype=DTYPE),
            a_minus=torch.tensor(0.35, dtype=DTYPE),
            tau_plus=2.0,
            tau_minus=3.0,
            w_min=LOWER.clone(),
            w_max=UPPER.clone(),
        )
    elif kind == "modulated":
        rule = ModulatedSTDP(
            ModulatedSTDPOptions(
                a_plus=torch.tensor(0.4, dtype=DTYPE),
                a_minus=torch.tensor(0.35, dtype=DTYPE),
                tau_plus=2.0,
                tau_minus=3.0,
                tau_eligibility=4.0,
                eta=torch.tensor(0.3, dtype=DTYPE),
            ),
            modulator=ExternalSignal("post", time_first=True),
            w_min=LOWER.clone(),
            w_max=UPPER.clone(),
        )
    elif kind in ("charged", "state"):
        rule = _VoltageReadout(kind)
    connection = SparseConnection.from_edges(
        PRE,
        POST,
        N,
        N,
        Synapse(
            receptor=RECEPTOR, n_receptor=R, delay=DELAY, n_delay=D, plasticity=rule
        ),
        values=WEIGHT.clone(),
    )
    psc = HeterConductancePSC(
        N,
        R,
        None,
        connection,
        tau_syn=TAU,
        reversal_potential=REVERSAL,
        backend="torch",
        use_circular_buffer=circular,
    )
    records = ["neuron.v", "neuron.v_pre_spike", "synapse.base_psc.psc", "synapse.psc"]
    if kind in ("pair", "modulated"):
        records += [
            "synapse.linear.weight.value",
            "synapse.linear.plasticity.pre_trace",
            "synapse.linear.plasticity.post_trace",
        ]
        if kind == "modulated":
            records.append("synapse.linear.plasticity.eligibility")
    elif rule is not None:
        records.append("synapse.linear.plasticity.observed")
    model = network_type(
        LIF(
            N,
            v_threshold=0.75,
            v_reset=-0.25,
            c_m=1.25,
            tau=4.0,
            hard_reset=True,
            pre_spike_v=True,
            backend="torch",
            dtype=DTYPE,
        ),
        psc,
        backend=backend,
        update_state_names=tuple(records),
        unroll=2,
        chunk_size=chunk_size,
    ).to(dtype=DTYPE)
    init_net_state(model, batch_size=None, device="cpu", dtype=DTYPE, persistent=True)
    model.neuron.v = INITIAL_V.clone()
    psc.base_psc.psc = INITIAL_G.flatten().clone()
    # A deliberately stale outer current must never replace g(E-v) in recurrence.
    psc.psc = torch.full((N,), 123.0, dtype=DTYPE)
    for event in reversed(PREHISTORY):
        psc.delayed_spike.update(event)
    if kind in ("pair", "modulated"):
        order = _canonical_order(connection)
        rule.pre_trace.copy_(INITIAL_PRE[order])
        rule.post_trace.copy_(INITIAL_POST[order])
        if kind == "modulated":
            rule.eligibility.copy_(INITIAL_ELIGIBILITY[order])
    return model


def _oracle(model, kind):
    """Evaluate equations with Python event indexing, not Delay or sparse ops."""
    order = _canonical_order(model.synapse.linear)
    pre, post, receptor, delay = (
        value[order] for value in (PRE, POST, RECEPTOR, DELAY)
    )
    weight, lower, upper = (value[order].clone() for value in (WEIGHT, LOWER, UPPER))
    x, y = INITIAL_PRE[order].clone(), INITIAL_POST[order].clone()
    eligibility = INITIAL_ELIGIBILITY[order].clone()
    v, g, events = INITIAL_V.clone(), INITIAL_G.clone(), []
    output, records, expanded = [], {}, []

    def record(name, value):
        records.setdefault(name, []).append(value.clone())

    for t, drive in enumerate(DRIVE):
        old_v = v.clone()
        current = (g * (REVERSAL - old_v[:, None])).sum(-1)
        charged = old_v + DT * (-(old_v + 0.25) / 4.0 + (drive + current) / 1.25)
        z = (charged >= 0.75).to(DTYPE)
        v = torch.where(z.bool(), -0.25, charged)
        events.append(z + EXTRA[t])
        history = torch.stack(
            [events[t - d] if t >= d else PREHISTORY[d - t - 1] for d in range(D)],
            dim=-1,
        )
        expanded.append(history.flatten())
        arrivals = history[pre, delay]
        # Update g with the weight BEFORE learning; sum parallel receptor slots.
        g = g * torch.exp(-DT / TAU)
        for edge in range(len(pre)):
            g[post[edge], receptor[edge]] += weight[edge] * arrivals[edge]
        if kind in ("pair", "modulated"):
            past_x = x * torch.exp(torch.tensor(-DT / 2.0, dtype=DTYPE))
            past_y = y * torch.exp(torch.tensor(-DT / 3.0, dtype=DTYPE))
            impulse = 0.4 * z[post] * past_x - 0.35 * arrivals * past_y
            if kind == "pair":
                increment = impulse
            else:
                fraction = -torch.expm1(torch.tensor(-DT / 4.0, dtype=DTYPE))
                increment = 0.3 * REWARDS[t, post] * eligibility * 4.0 * fraction
                eligibility = eligibility * (1 - fraction) + impulse
                record("synapse.linear.plasticity.eligibility", eligibility)
            weight = torch.maximum(lower, torch.minimum(upper, weight + increment))
            x, y = past_x + arrivals, past_y + z[post]
            record("synapse.linear.weight.value", weight)
            record("synapse.linear.plasticity.pre_trace", x)
            record("synapse.linear.plasticity.post_trace", y)
        if kind in ("charged", "state"):
            record(
                "synapse.linear.plasticity.observed",
                charged if kind == "charged" else v,
            )
        output.append(z)
        record("neuron.v", v)
        record("neuron.v_pre_spike", charged)
        record("synapse.base_psc.psc", g.flatten())
        # The returned current is observed at the voltage supplied before firing.
        record("synapse.psc", (g * (REVERSAL - old_v[:, None])).sum(-1))
    return (
        torch.stack(output),
        {k: torch.stack(v) for k, v in records.items()},
        expanded,
    )


def _join(results):
    return (
        torch.cat([result[0] for result in results]),
        {
            key: torch.cat([result[1][key] for result in results])
            for key in results[0][1]
        },
    )


def _run(model, kind, route):
    def call(start, stop):
        kwargs = (
            {}
            if kind != "modulated"
            else {"plasticity_signals": {"modulator": REWARDS[start:stop]}}
        )
        return model(DRIVE[start:stop], EXTRA[start:stop], **kwargs)

    if route == "whole":
        return call(0, len(DRIVE))
    if route == "split":
        # A cut inside the delay depth and a one-step chunk expose history loss.
        return _join([call(0, 2), call(2, 3), call(3, len(DRIVE))])
    results = []
    for t in range(len(DRIVE)):
        kwargs = (
            {}
            if kind != "modulated"
            else {"plasticity_signals": {"modulator": REWARDS[t]}}
        )
        z, states = model.single_step_forward(DRIVE[t], EXTRA[t], **kwargs)
        results.append((z[None], {k: v.clone()[None] for k, v in states.items()}))
    return _join(results)


def _assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def _assert_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        if isinstance(actual[name], torch.Tensor):
            torch.testing.assert_close(
                actual[name],
                expected[name],
                atol=0,
                rtol=0,
                msg=lambda msg: f"{name}: {msg}",
            )
        else:
            assert actual[name] == expected[name], name


@pytest.mark.parametrize("kind", ["static", "pair", "modulated"])
@pytest.mark.parametrize("route", ["whole", "split", "standalone"])
@pytest.mark.parametrize("circular", [False, True])
def test_heter_conductance_recurrence_matches_event_oracle(kind, route, circular):
    """Routed old-weight propagation and learning agree at every physical step."""
    model = _build(kind, chunk_size=3 if route == "split" else None, circular=circular)
    expected, records, histories = _oracle(model, kind)
    with torch.no_grad(), environ.context(dt=DT):
        actual, states = _run(model, kind, route)
    _assert_close(actual, expected)
    assert states.keys() == records.keys()
    for name in records:
        _assert_close(states[name], records[name])
    _assert_close(model.synapse.delayed_spike.get_flattened(D), histories[-1])
    if kind in ("pair", "modulated"):
        connection = model.synapse.linear
        order = _canonical_order(connection)
        rule = connection.plasticity
        torch.testing.assert_close(rule.w_min, LOWER[order], atol=0, rtol=0)
        torch.testing.assert_close(rule.w_max, UPPER[order], atol=0, rtol=0)
        weights = states["synapse.linear.weight.value"]
        fixed = RECEPTOR[order] == 1
        torch.testing.assert_close(
            weights[:, fixed],
            WEIGHT[order][fixed].expand(len(DRIVE), -1),
            atol=0,
            rtol=0,
        )
        assert (weights >= LOWER[order]).all() and (weights <= UPPER[order]).all()
        assert not torch.equal(weights[-1], WEIGHT[order])
        if kind == "pair":
            assert (weights[:, ~fixed] == LOWER[order][~fixed]).any()
            assert (weights[:, ~fixed] == UPPER[order][~fixed]).any()


def test_explicit_torch_pull_options_with_literal_torch_children():
    """Default arithmetic flags must not disqualify an eager Torch schedule."""
    model = _build(
        "pair",
        backend=KernelOptions(
            neuron="torch", connection="torch", connection_method="pull"
        ),
    )
    expected, records, _ = _oracle(model, "pair")
    with torch.no_grad(), environ.context(dt=DT):
        actual, states = _run(model, "pair", "whole")
    _assert_close(actual, expected)
    for name in records:
        _assert_close(states[name], records[name])


@pytest.mark.parametrize("kind", ["static", "pair", "modulated"])
@pytest.mark.parametrize("circular", [False, True])
def test_persistent_checkpoint_restores_delays_conductance_and_learning(kind, circular):
    """A serialized mid-delay checkpoint continues the same physical trajectory."""
    model = _build(kind, chunk_size=2, circular=circular)
    resumed = _build(kind, chunk_size=3, circular=circular)

    def call(network, start, stop):
        kwargs = (
            {}
            if kind != "modulated"
            else {"plasticity_signals": {"modulator": REWARDS[start:stop]}}
        )
        return network(DRIVE[start:stop], EXTRA[start:stop], **kwargs)

    with torch.no_grad(), environ.context(dt=DT):
        prefix = call(model, 0, 2)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=False)
        assert "neuron.v" in saved
        assert "synapse.base_psc.psc" in saved
        assert "synapse.delayed_spike.history" in saved
        resumed.load_state_dict(saved)
        _assert_state(resumed.state_dict(), model.state_dict())
        expected_tail = call(model, 2, len(DRIVE))
        actual_tail = call(resumed, 2, len(DRIVE))
        # Compare with an uninterrupted call too, not just two matching resumes.
        whole = _build(kind, circular=circular)
        complete = call(whole, 0, len(DRIVE))
    for actual, expected in [
        (actual_tail, expected_tail),
        (_join([prefix, actual_tail]), complete),
    ]:
        _assert_close(actual[0], expected[0])
        for key in actual[1]:
            _assert_close(actual[1][key], expected[1][key])
    _assert_state(resumed.state_dict(), model.state_dict())


@pytest.mark.parametrize("stage", ["charged", "state"])
@pytest.mark.parametrize("route", ["whole", "standalone"])
def test_post_state_reads_the_current_step_at_the_declared_stage(stage, route):
    """Learning voltage differs from both the coupling voltage and reset stage."""
    model = _build(stage)
    expected, records, _ = _oracle(model, stage)
    with torch.no_grad(), environ.context(dt=DT):
        actual, states = _run(model, stage, route)
    _assert_close(actual, expected)
    for key in records:
        _assert_close(states[key], records[key])
    assert (expected != 0).any()  # Reset-stage distinctions are actually exercised.


class _CountingInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value


@pytest.mark.parametrize("route", ["single", "multi", "chunk"])
@pytest.mark.parametrize("invalid", ["plain_rnn", "network_cell", "child_cell"])
def test_unsupported_schedules_reject_before_callbacks_or_state(invalid, route):
    """Reject missing voltage scheduling and native cells without running them."""
    model = _build(
        network_type=RecurrentNN if invalid == "plain_rnn" else ConductanceRecurrentNN
    )
    if invalid == "network_cell":
        model.backend = KernelOptions(neuron="triton")
    elif invalid == "child_cell":
        model.neuron.backend = "triton"
    callback = _CountingInput()
    model.neuron_inp_module = callback
    before = copy.deepcopy(model.state_dict())
    with torch.no_grad(), environ.context(dt=DT), pytest.raises(NotImplementedError):
        if route == "single":
            model.single_step_forward(DRIVE[0])
        elif route == "chunk":
            model._run_chunk_steps(DRIVE, loop_args=(0,), unroll_size=2)
        else:
            model(DRIVE)
    assert callback.calls == 0
    _assert_state(model.state_dict(), before)


@pytest.mark.parametrize("route", ["single", "multi"])
def test_bad_external_signal_shape_rejects_before_state_advances(route):
    """Validate a post-domain learning signal before touching voltage/history."""
    model = _build("modulated")
    before = copy.deepcopy(model.state_dict())
    values = torch.zeros(N + 1, dtype=DTYPE)
    if route == "multi":
        values = values.expand(len(DRIVE), -1)
    with torch.no_grad(), environ.context(dt=DT), pytest.raises(ValueError):
        if route == "single":
            model.single_step_forward(
                DRIVE[0], plasticity_signals={"modulator": values}
            )
        else:
            model(DRIVE, plasticity_signals={"modulator": values})
    _assert_state(model.state_dict(), before)


@pytest.mark.parametrize("route", ["single", "multi"])
@pytest.mark.parametrize("invalid", ["drive", "synaptic_event"])
def test_wrong_population_shape_rejects_before_state_advances(route, invalid):
    """A malformed input must not leave a partially advanced recurrent step."""
    model = _build("pair")
    drive, extra = DRIVE.clone(), EXTRA.clone()
    bad = torch.zeros(len(DRIVE), N + 1, dtype=DTYPE)
    if invalid == "drive":
        drive = bad
    else:
        extra = bad
    before = copy.deepcopy(model.state_dict())
    with (
        torch.no_grad(),
        environ.context(dt=DT),
        pytest.raises((ValueError, RuntimeError)),
    ):
        if route == "single":
            model.single_step_forward(drive[0], extra[0])
        else:
            model(drive, extra)
    _assert_state(model.state_dict(), before)


def test_standalone_unsupported_sparse_route_preserves_neuron_and_synapse_state():
    """Reject explicit Triton push on CPU before the first physical step."""
    model = _build(
        "static",
        backend=KernelOptions(
            neuron="torch", connection="triton", connection_method="push"
        ),
    )
    before = copy.deepcopy(model.state_dict())
    assert model.neuron.v.device.type == "cpu"
    with (
        torch.no_grad(),
        environ.context(dt=DT),
        pytest.raises(RuntimeError, match=r"(?i)(unsupported|cpu|triton)"),
    ):
        model.single_step_forward(DRIVE[0], EXTRA[0])
    # Include nonzero V/g, outer current, delay history and cursor; a rejection
    # after firing or propagation must not masquerade as a successful preflight.
    _assert_state(model.state_dict(), before)
