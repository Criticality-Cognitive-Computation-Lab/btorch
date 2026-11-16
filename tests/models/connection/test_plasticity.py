"""Analytical semantics and lifecycle of arrival-based online plasticity.

These tests deliberately use one unbatched sample and the public
``SparseConnection.update_plasticity`` entry point.  The expected values are
derived from explicit event histories or hand-computed transitions so the
suite can serve as the correctness oracle for optimized implementations.
"""

from __future__ import annotations

import copy
import io
import math
from dataclasses import replace

import pytest
import torch

from btorch.models import environ
from btorch.models.connection import (
    EdgeWeight,
    ExternalSignal,
    InhibitorySTDP,
    ModulatedSTDP,
    ModulatedSTDPOptions,
    NearestPairSTDP,
    PairSTDP,
    PlasticityEvents,
    PlasticityRule,
    PostState,
    PowerPairSTDP,
    PowerPairSTDPOptions,
    SparseConnection,
    Synapse,
    TripletSTDP,
    TripletSTDPOptions,
    VoltageSTDP,
    VoltageSTDPOptions,
)
from btorch.models.functional import init_net_state
from btorch.models.neurons import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import ExponentialPSC, HeterSynapsePSC


PRE = torch.tensor([2, 0, 1, 2])
POST = torch.tensor([1, 0, 2, 2])
DELAY = torch.tensor([1, 0, 2, 0])
RECEPTOR = torch.tensor([1, 0, 1, 0])
VALUES = torch.tensor([0.4, 0.5, 0.6, 0.7])
N_PRE = 3
N_POST = 3
N_DELAY = 3
N_RECEPTOR = 2


class CustomTriplet(PlasticityRule):
    """Show a user rule with four memories and a genuine triplet term."""

    def __init__(self):
        super().__init__(w_min=0.0, w_max=0.72)
        # Input-edge arrays must follow the same unsorted-to-canonical map as
        # weights, delays and receptors; a scalar coefficient needs no map.
        self.register_buffer("tau_fast", torch.tensor([2.0, 3.0, 4.0, 5.0]))
        self.register_buffer("tau_slow", torch.tensor(8.0))
        for name in self.state_domains:
            self.register_memory(name, 0.0, (), persistent=True)

    state_domains = {
        "pre_fast": "edge",
        "pre_slow": "edge",
        "post_fast": "edge",
        "post_slow": "edge",
    }

    def single_step_forward(self, weight, events: PlasticityEvents, dt):
        pre_fast = self.pre_fast * torch.exp(-dt / self.tau_fast)
        post_fast = self.post_fast * torch.exp(-dt / self.tau_fast)
        pre_slow = self.pre_slow * torch.exp(-dt / self.tau_slow)
        post_slow = self.post_slow * torch.exp(-dt / self.tau_slow)
        delta = 0.1 * events.post_spike * pre_fast * (
            1 + 0.4 * post_slow
        ) - 0.12 * events.arrival * post_fast * (1 + 0.3 * pre_slow)
        for name, value, event in (
            ("pre_fast", pre_fast, events.arrival),
            ("pre_slow", pre_slow, events.arrival),
            ("post_fast", post_fast, events.post_spike),
            ("post_slow", post_slow, events.post_spike),
        ):
            getattr(self, name).copy_(value + event)
        return delta


class VoltageGate(PlasticityRule):
    """Demonstrate continuous learning with post, edge and scalar memories."""

    state_domains = {
        "filtered_voltage": "post",
        "eligibility": "edge",
        "clock": "scalar",
    }
    parameter_domains = {"tau_voltage": "post"}

    def __init__(self, *, time_first=False):
        super().__init__(
            signals={
                "voltage": PostState("v"),
                "gain": ExternalSignal(time_first=time_first),
                "gate": ExternalSignal(),
            },
        )
        self.register_buffer("tau_voltage", torch.tensor([2.0, 3.0, 4.0]))

        for name in self.state_domains:
            self.register_memory(name, 0.0, (), persistent=True)

    def single_step_forward(self, weight, events, dt):
        decay = torch.exp(-dt / self.tau_voltage)
        voltage = self.filtered_voltage * decay + events.signals["voltage"] * (
            1 - decay
        )
        eligibility = self.eligibility * torch.exp(-dt / 5.0) + events.arrival
        delta = (
            dt
            * events.signals["gain"]
            * events.signals["gate"]
            * voltage[events.post_index]
            * eligibility
        )
        self.filtered_voltage.copy_(voltage)
        self.eligibility.copy_(eligibility)
        self.clock.add_(dt)
        return delta


def test_live_charged_voltage_and_external_signal_update_declared_domains():
    """Bind fresh pre-reset voltage and reject missing signals before
    stepping."""
    model = _recurrent_model(rule=VoltageGate(time_first=True), pre_spike_v=True)
    connection = model.synapse.linear
    rule = connection.plasticity
    inputs = torch.tensor([[0.7, 0.8, 0.9], [0.0, 0.0, 0.0]])
    # A required external input must be supplied before even the first neuron
    # step. This avoids a half-advanced recurrent state on a configuration error.
    before = copy.deepcopy(model.state_dict())
    with (
        torch.no_grad(),
        pytest.raises(ValueError, match="plasticity signal|requires signals"),
    ):
        model(inputs)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])

    expected_weight = connection.weight.value.detach().clone()
    expected_voltage = torch.zeros(N_POST)
    expected_eligibility = torch.zeros(connection.nnz)
    gains = torch.tensor([0.03, 0.06])
    gate = torch.tensor(0.5)
    with torch.no_grad(), environ.context(dt=1.0):
        for step in range(2):
            old_v = model.neuron.v.clone()
            old_current = model.synapse.psc.clone()
            expected_charged = old_v + (
                -old_v / model.neuron.tau
                + (old_current + inputs[step]) / model.neuron.c_m
            )
            model(
                inputs[step : step + 1],
                plasticity_signals={"gain": gains[step : step + 1], "gate": gate},
            )
            torch.testing.assert_close(model.neuron.v_pre_spike, expected_charged)
            if step == 0:
                # All three neurons reset; using post-reset v would incorrectly
                # erase the voltage-driven update on the zero-delay edges.
                torch.testing.assert_close(model.neuron.v, torch.zeros(N_POST))
            decay = torch.exp(-1.0 / rule.tau_voltage)
            expected_voltage = expected_voltage * decay + expected_charged * (1 - decay)
            delayed = model.synapse.delayed_spike.get_flattened(N_DELAY)
            arrivals = delayed[connection.pre * N_DELAY + connection.delay]
            expected_eligibility = (
                expected_eligibility * torch.exp(torch.tensor(-0.2)) + arrivals
            )
            expected_weight = (
                expected_weight
                + gains[step]
                * gate
                * expected_voltage[connection.post]
                * expected_eligibility
            ).clamp(0.0, 1.0)
            torch.testing.assert_close(rule.filtered_voltage, expected_voltage)
            torch.testing.assert_close(rule.eligibility, expected_eligibility)
            torch.testing.assert_close(connection.weight.value, expected_weight)
            torch.testing.assert_close(rule.clock, torch.tensor(float(step + 1)))
        whole = _recurrent_model(rule=VoltageGate(time_first=True), pre_spike_v=True)
        whole(
            inputs,
            x_syn=torch.zeros_like(inputs),
            plasticity_signals={"gain": gains, "gate": gate},
        )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(whole.state_dict()[name], value)


def test_custom_voltage_rule_graph_replay_refreshes_nested_signal_inputs():
    """Reuse bounded captures with new signal storage and continuous state."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA graph replay requires CUDA")
    from btorch.models import KernelOptions

    expected = _recurrent_model(
        rule=VoltageGate(time_first=True), pre_spike_v=True
    ).cuda()
    actual = _recurrent_model(
        rule=VoltageGate(time_first=True), pre_spike_v=True
    ).cuda()
    for model in (expected, actual):
        model.backend = KernelOptions(
            neuron="torch", connection="torch", connection_method="pull"
        )
        model.chunk_size = 2
        model.update_state_names = ()
    actual.cudagraph = True
    inputs = _recurrent_inputs().cuda()
    # Signals arrive in a named dictionary. The second call uses a distinct
    # Tensor and a new value; replay must copy that leaf into captured storage,
    # rather than retaining the first caller's dictionary or Tensor address.
    with torch.no_grad(), environ.context(dt=1.0):
        for current, gains, gate in (
            (inputs[:4], [0.01, 0.02, 0.03, 0.04], 0.5),
            (inputs[2:], [0.04, 0.02, 0.05], 1.25),
        ):
            reference, _ = expected(
                current,
                x_syn=torch.zeros_like(current),
                plasticity_signals={
                    "gain": torch.tensor(gains, device="cuda"),
                    "gate": torch.tensor(gate, device="cuda"),
                },
            )
            observed, _ = actual(
                current,
                x_syn=torch.zeros_like(current),
                plasticity_signals={
                    "gain": torch.tensor(gains, device="cuda"),
                    "gate": torch.tensor(gate, device="cuda"),
                },
            )
            torch.testing.assert_close(observed, reference)
            for name, value in expected.state_dict().items():
                torch.testing.assert_close(actual.state_dict()[name], value)
    # Only length-2 chunks are captured; the final length-1 tail runs eagerly.
    # The learned state/clock comparisons above also catch extra warmup steps.
    assert actual._cudagraph_runner.n_captures == 1


def test_custom_rule_compiled_updates_and_checkpoint_use_all_declared_memories():
    """A custom tensor rule learns routed edges and resumes its four traces."""
    connection = _connection(rule=CustomTriplet())
    resumed = _connection(rule=CustomTriplet())
    builtin = _connection(
        rule=TripletSTDP(
            TripletSTDPOptions(
                a_plus=0.1,
                a_minus=0.12,
                a3_plus=0.04,
                a3_minus=0.036,
                tau_plus=torch.tensor([2.0, 3.0, 4.0, 5.0]),
                tau_minus=torch.tensor([2.0, 3.0, 4.0, 5.0]),
                tau_x=8.0,
                tau_y=8.0,
            ),
            w_max=0.72,
        )
    )
    arrivals = torch.tensor(
        [
            [1, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    posts = torch.tensor(
        [[0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0]],
        dtype=torch.float32,
    )
    edge_arrivals = arrivals[:, connection.pre * N_DELAY + connection.delay]
    edge_posts = posts[:, connection.post]
    # Independent oracle: reconstruct traces from all prior event timestamps,
    # rather than duplicating the custom rule's incremental recurrence.
    dt = 0.5
    expected_weight = connection.weight.value.detach().clone()
    run = torch.compile(connection.update_plasticity, backend="eager", fullgraph=True)
    for step in range(len(arrivals)):
        past_dt = (step - torch.arange(step, dtype=torch.float32))[:, None] * dt
        fast = torch.exp(-past_dt / connection.plasticity.tau_fast)
        slow = torch.exp(-past_dt / connection.plasticity.tau_slow)
        pre_fast = (edge_arrivals[:step] * fast).sum(0)
        post_fast = (edge_posts[:step] * fast).sum(0)
        pre_slow = (edge_arrivals[:step] * slow).sum(0)
        post_slow = (edge_posts[:step] * slow).sum(0)
        delta = 0.1 * edge_posts[step] * pre_fast * (
            1 + 0.4 * post_slow
        ) - 0.12 * edge_arrivals[step] * post_fast * (1 + 0.3 * pre_slow)
        expected_weight = (expected_weight + delta).clamp(0.0, 0.72)
        with torch.no_grad():
            run(arrivals[step], posts[step], dt=dt)
            builtin.update_plasticity(arrivals[step], posts[step], dt=dt)
            if step >= 3:
                resumed.update_plasticity(arrivals[step], posts[step], dt=dt)
        torch.testing.assert_close(connection.weight.value, expected_weight)
        torch.testing.assert_close(builtin.weight.value, expected_weight)
        for name, expected in (
            ("pre_fast", pre_fast + edge_arrivals[step]),
            ("pre_slow", pre_slow + edge_arrivals[step]),
            ("post_fast", post_fast + edge_posts[step]),
            ("post_slow", post_slow + edge_posts[step]),
        ):
            torch.testing.assert_close(getattr(connection.plasticity, name), expected)
            if step >= 3:
                torch.testing.assert_close(getattr(resumed.plasticity, name), expected)
            builtin_name = {"pre_fast": "pre_trace", "post_fast": "post_trace"}.get(
                name, name
            )
            torch.testing.assert_close(
                getattr(builtin.plasticity, builtin_name), expected
            )
        if step == 2:
            checkpoint = io.BytesIO()
            torch.save(connection.state_dict(), checkpoint)
            checkpoint.seek(0)
            resumed.load_state_dict(torch.load(checkpoint, weights_only=True))
        if step >= 3:
            torch.testing.assert_close(resumed.weight.value, expected_weight)

    # Reset clears every user-declared memory, while retaining learned weights.
    connection.plasticity.reset(inplace=True)
    for name in ("pre_fast", "pre_slow", "post_fast", "post_slow"):
        torch.testing.assert_close(
            getattr(connection.plasticity, name), torch.zeros_like(expected_weight)
        )
    torch.testing.assert_close(connection.weight.value, expected_weight)


def _rule(**kwargs) -> PairSTDP:
    options = {
        "a_plus": 0.1,
        "a_minus": 0.12,
        "tau_plus": 20.0,
        "tau_minus": 20.0,
        "w_min": 0.0,
        "w_max": 1.0,
    }
    options.update(kwargs)
    return PairSTDP(**options)


def _connection(rule: PlasticityRule | None = None, values: torch.Tensor = VALUES):
    """Build an unsorted, routed edge list with distinct expanded slots."""
    return SparseConnection.from_edges(
        PRE,
        POST,
        N_PRE,
        N_POST,
        Synapse(
            weight=EdgeWeight(values, trainable=False),
            delay=DELAY,
            receptor=RECEPTOR,
            n_delay=N_DELAY,
            n_receptor=N_RECEPTOR,
            plasticity=rule or _rule(),
        ),
        values=values,
    )


def _arrivals(conn: SparseConnection, edge_values: torch.Tensor) -> torch.Tensor:
    """Scatter canonical edge events into the public delay-expanded layout."""
    result = torch.zeros(conn.in_features, dtype=conn.weight.value.dtype)
    slots = conn.pre * conn.n_delay + conn.delay
    result.scatter_(0, slots, edge_values)
    return result


def _posts(conn: SparseConnection, edge_values: torch.Tensor) -> torch.Tensor:
    """Scatter canonical edge post events to the postsynaptic population."""
    result = torch.zeros(conn.n_post, dtype=conn.weight.value.dtype)
    # Several incoming edges share a post neuron.  Spikes are binary events,
    # so a max reduction preserves an event even when a later zero-valued edge
    # has the same target.
    result.scatter_reduce_(0, conn.post, edge_values, reduce="amax")
    return result


def _shared_pair_connection(trace_scope, *, rule_factory=None):
    """Share segment events across receptors and retain an isolated post."""
    rule = PairSTDP(
        a_plus=torch.linspace(0.1, 0.2, 6),
        a_minus=torch.linspace(0.2, 0.3, 6),
        tau_plus=2.0,
        tau_minus=3.0,
        w_min=-0.1,
        w_max=0.8,
        trace_scope=trace_scope,
    )
    return SparseConnection.from_edges(
        torch.tensor([0, 0, 0, 1, 1, 1]),
        torch.tensor([0, 1, 2, 0, 2, 1]),
        3,
        4,
        Synapse(
            weight=EdgeWeight(torch.full((6,), 0.4), trainable=False),
            delay=torch.tensor([0, 0, 1, 1, 1, 0]),
            receptor=torch.tensor([0, 1, 1, 0, 1, 1]),
            n_delay=2,
            n_receptor=2,
            plasticity=rule if rule_factory is None else rule_factory(rule),
        ),
    ).to(dtype=torch.float64)


def test_shared_pair_traces_preserve_delayed_learning_and_checkpoint_continuation():
    """Own traces by event stream without counting duplicate edges twice.

    Two edges share source/delay segments despite using different
    receptors. The edge oracle must therefore equal gathered domain
    traces at every step. Neuron 3 has no incoming edge but its
    postsynaptic trace must still evolve. A silent step after restoring
    tests decay and chunk continuation; reset then clears domain
    memories while preserving learned weights.
    """
    shared = _shared_pair_connection("neuron")
    edge = _shared_pair_connection("connection")
    # Domain-owned memory is also allowed to be a noncontiguous view.
    post_storage = shared.plasticity.post_trace.new_zeros(8)
    shared.plasticity.post_trace = post_storage[::2]
    arrivals = torch.tensor(
        [
            [1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
        ],
        dtype=torch.float64,
    )
    posts = torch.tensor(
        [[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 1], [0, 0, 0, 0], [0, 1, 1, 0]],
        dtype=torch.float64,
    )
    # Binding needs only the compact topology. The dense inverse edge map is
    # prepared lazily by eager updates, rather than being an unconditional
    # memory cost of every future native shared-trace execution.
    assert shared.cache.edge_segment.numel() == 0
    assert shared.plasticity.pre_trace.numel() == 4
    assert shared.plasticity.post_trace.numel() == 4
    restored = None
    with torch.no_grad():
        for step, (arrival, post) in enumerate(zip(arrivals, posts, strict=True)):
            # Propagation observes the same old weights before learning.
            torch.testing.assert_close(shared(arrival), edge(arrival))
            shared.update_plasticity(arrival, post, dt=0.5)
            edge.update_plasticity(arrival, post, dt=0.5)
            torch.testing.assert_close(shared.weight.value, edge.weight.value)
            torch.testing.assert_close(
                shared.plasticity.pre_trace[shared.cache.edge_segment],
                edge.plasticity.pre_trace,
            )
            torch.testing.assert_close(
                shared.plasticity.post_trace[shared.post],
                edge.plasticity.post_trace,
            )
            if step == 0:
                # Fresh coincident events have no strictly earlier partner.
                torch.testing.assert_close(
                    shared.weight.value, torch.full_like(shared.weight.value, 0.4)
                )
                assert shared.plasticity.post_trace[3] == 1
            if restored is not None:
                restored.update_plasticity(arrival, post, dt=0.5)
                torch.testing.assert_close(restored.weight.value, shared.weight.value)
                torch.testing.assert_close(
                    restored.plasticity.pre_trace, shared.plasticity.pre_trace
                )
                torch.testing.assert_close(
                    restored.plasticity.post_trace, shared.plasticity.post_trace
                )
            if step == 2:
                checkpoint = io.BytesIO()
                torch.save(shared.state_dict(), checkpoint)
                checkpoint.seek(0)
                restored = _shared_pair_connection("neuron")
                restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        expected_isolated = math.exp(-4 * 0.5 / 3.0) + math.exp(-2 * 0.5 / 3.0)
        torch.testing.assert_close(
            shared.plasticity.post_trace[3],
            torch.tensor(expected_isolated, dtype=torch.float64),
        )
        learned = shared.weight.value.clone()
        shared.plasticity.reset(inplace=True)
        assert torch.count_nonzero(shared.plasticity.pre_trace) == 0
        assert torch.count_nonzero(shared.plasticity.post_trace) == 0
        torch.testing.assert_close(shared.weight.value, learned)


def _edge_event(conn: SparseConnection, slot: int, value: float = 1.0):
    edge_values = torch.zeros(conn.nnz, dtype=conn.weight.value.dtype)
    edge_values[slot] = value
    return _arrivals(conn, edge_values), _posts(conn, edge_values)


def _input_edge_order(connection):
    """Match canonical routing to original user edges independently of
    coefficients."""
    return [
        next(
            old
            for old in range(len(PRE))
            if (int(PRE[old]), int(POST[old]), int(DELAY[old]), int(RECEPTOR[old]))
            == (int(pre), int(post), int(delay), int(receptor))
        )
        for pre, post, delay, receptor in zip(
            connection.pre,
            connection.post,
            connection.delay,
            connection.receptor,
            strict=True,
        )
    ]


@pytest.mark.parametrize("trace_scope", ("connection", "neuron"))
def test_inhibitory_stdp_matches_signed_timestamp_oracle_and_resumes(trace_scope):
    """Balance signed currents and conductances with real delayed arrivals."""
    eta, alpha = [0.6, 0.7, 0.8, 0.0], [0.2, 0.3, 0.5, 0.4]
    polarity = [-1.0, 1.0, -1.0, 1.0]
    tau = [4.0] * 4 if trace_scope == "neuron" else [4.0, 5.0, 6.0, 7.0]
    lo, hi = [-1.0, 0.0, -0.6, 0.0], [0.0, 1.0, -0.6, 1.0]

    def build():
        return _connection(
            InhibitorySTDP(
                eta=torch.tensor(eta, dtype=torch.float64),
                tau=torch.tensor(
                    tau[0] if trace_scope == "neuron" else tau, dtype=torch.float64
                ),
                alpha=torch.tensor(alpha, dtype=torch.float64),
                polarity=torch.tensor(polarity, dtype=torch.float64),
                w_min=torch.tensor(lo, dtype=torch.float64),
                w_max=torch.tensor(hi, dtype=torch.float64),
                trace_scope=trace_scope,
            ),
            values=torch.tensor([-0.4, 0.4, -0.6, 0.3], dtype=torch.float64),
        )

    connection = build()
    source = _input_edge_order(connection)
    expected = connection.weight.value.tolist()
    pre_events, post_events = [], []
    resumed = None
    execute = torch.compile(
        connection.update_plasticity, backend="eager", fullgraph=True
    )
    # Include coincident events from zero traces, both temporal orders, a
    # silent interval, repeated pairing and clipping. The oracle sums explicit
    # timestamps rather than implementing the production trace recurrence.
    for time, slots in enumerate(((0,), (1, 2), (), (0, 1, 2, 3), (2,), (), (0, 3))):
        edges = torch.zeros(connection.nnz, dtype=torch.float64)
        edges[list(slots)] = 1
        arrivals = _arrivals(connection, edges)
        posts = _posts(connection, edges if time == 0 else edges.roll(1))
        pre_events.append(
            arrivals[connection.pre * N_DELAY + connection.delay].tolist()
        )
        post_events.append(posts[connection.post].tolist())
        pre_snapshots, post_snapshots = [], []
        for edge, old in enumerate(source):
            pre = sum(
                pre_events[t][edge] * math.exp(-(time - t) / tau[old])
                for t in range(time)
            )
            post = sum(
                post_events[t][edge] * math.exp(-(time - t) / tau[old])
                for t in range(time)
            )
            change = (
                polarity[old]
                * eta[old]
                * (
                    pre_events[-1][edge] * (post - alpha[old])
                    + post_events[-1][edge] * pre
                )
            )
            expected[edge] = min(hi[old], max(lo[old], expected[edge] + change))
            pre_snapshots.append(pre + pre_events[-1][edge])
            post_snapshots.append(post + post_events[-1][edge])
        with torch.no_grad():
            execute(arrivals, posts, dt=1.0)
            if resumed is not None:
                resumed.update_plasticity(arrivals, posts, dt=1.0)
                torch.testing.assert_close(
                    resumed.state_dict(), connection.state_dict()
                )
        rule = connection.plasticity
        pre = rule.pre_trace.reshape(-1)
        post = rule.post_trace.reshape(-1)
        if trace_scope == "neuron":
            pre = pre[connection.cache.edge_segment]
            post = post[connection.post]
        for actual, values in (
            (connection.weight.value, expected),
            (pre, pre_snapshots),
            (post, post_snapshots),
        ):
            torch.testing.assert_close(
                actual,
                torch.tensor(values, dtype=torch.float64),
                atol=2e-12,
                rtol=2e-12,
            )
        if time == 3:
            # Serialize after a nonempty chunk and resume without resetting.
            checkpoint = io.BytesIO()
            torch.save(connection.state_dict(), checkpoint)
            checkpoint.seek(0)
            resumed = build()
            resumed.load_state_dict(torch.load(checkpoint, weights_only=True))
    learned = connection.weight.value.detach().clone()
    rule.reset()
    torch.testing.assert_close(connection.weight.value, learned)
    assert not rule.pre_trace.any() and not rule.post_trace.any()


def test_inhibitory_stdp_finite_timestep_homeostatic_drift_changes_sign():
    """Verify the exact binary-spike equilibrium rather than a small-dt
    limit."""
    dt, tau, alpha, eta = 2.0, 10.0, 1.2, 0.1
    connection = SparseConnection.from_edges(
        torch.tensor([0]),
        torch.tensor([0]),
        1,
        1,
        Synapse(
            weight=EdgeWeight(
                torch.tensor([-0.5], dtype=torch.float64), trainable=False
            ),
            plasticity=InhibitorySTDP(
                eta=torch.tensor(eta, dtype=torch.float64),
                tau=tau,
                alpha=torch.tensor(alpha, dtype=torch.float64),
                w_min=-100.0,
            ),
        ),
    )
    decay = math.exp(-dt / tau)
    target = alpha * math.expm1(dt / tau) / (2 * dt)
    pre_probability = 0.01
    # Stationary mean memories are p/(1-decay). The weight law is linear in
    # each old trace, so enumerating independent current binary events gives
    # its exact expected drift without sampling noise or network correlations.
    for factor in (0.5, 1.0, 2.0):
        post_probability = target * dt * factor
        mean_change = 0.0
        for arrival, post in ((0, 0), (0, 1), (1, 0), (1, 1)):
            probability = (pre_probability if arrival else 1 - pre_probability) * (
                post_probability if post else 1 - post_probability
            )
            with torch.no_grad():
                connection.weight.value.fill_(-0.5)
                connection.plasticity.pre_trace.fill_(pre_probability / (1 - decay))
                connection.plasticity.post_trace.fill_(post_probability / (1 - decay))
                connection.update_plasticity(
                    torch.tensor([arrival], dtype=torch.float64),
                    torch.tensor([post], dtype=torch.float64),
                    dt=dt,
                )
            mean_change += probability * (float(connection.weight.value[0]) + 0.5)
        # Above target, a negative current weight becomes more negative;
        # below target it weakens. Exactly at target the signed drift is zero.
        expected = -eta * pre_probability * alpha * (factor - 1)
        assert mean_change == pytest.approx(expected, abs=2e-15)


def test_power_pair_stdp_matches_timestamp_oracle_for_signed_and_fixed_bounds():
    """Check additive, multiplicative and fractional powers in one routed
    example."""
    coefficients = {
        "a_plus": [0.5, 0.4, 0.6, 0.3],
        "a_minus": [0.4, 0.7, 0.3, 0.6],
        "tau_plus": [3.0, 4.0, 5.0, 6.0],
        "tau_minus": [4.0, 5.0, 6.0, 7.0],
        "mu_plus": [0.0, 1.0, 0.5, 2.0],
        "mu_minus": [0.0, 1.0, 1.5, 0.25],
    }
    options = PowerPairSTDPOptions(
        **{
            name: torch.tensor(value, dtype=torch.float64)
            for name, value in coefficients.items()
        }
    )
    lo, hi = [-1.0, 0.0, -1.0, 0.25], [0.0, 1.0, 1.0, 0.25]
    connection = _connection(
        PowerPairSTDP(
            options,
            w_min=torch.tensor(lo, dtype=torch.float64),
            w_max=torch.tensor(hi, dtype=torch.float64),
        ),
        values=VALUES.double(),
    )
    # The oracle indexes original user edges independently of the rule's
    # canonicalized coefficient buffers. This also verifies sorting of powers
    # and signed/fixed bounds, not just the recurrence after sorting.
    source = _input_edge_order(connection)
    expected = [lo[old] + 0.4 * (hi[old] - lo[old]) for old in source]
    # An out-of-bounds fixed edge is clipped even on the first silent step.
    expected[source.index(3)] += 0.1
    with torch.no_grad():
        connection.weight.value.copy_(torch.tensor(expected, dtype=torch.float64))
    pre_events, post_events = [], []
    execute = torch.compile(
        connection.update_plasticity, backend="eager", fullgraph=True
    )
    for time, slots in enumerate(((), (0,), (1, 2), (0, 1, 2, 3), (), (2, 3))):
        edges = torch.zeros(connection.nnz, dtype=torch.float64)
        edges[list(slots)] = 1
        arrivals = _arrivals(connection, edges)
        posts = _posts(connection, edges.roll(1))
        pre_events.append(
            arrivals[connection.pre * N_DELAY + connection.delay].tolist()
        )
        post_events.append(posts[connection.post].tolist())
        pre_snapshots, post_snapshots = [], []
        for edge, old in enumerate(source):
            params = {name: values[old] for name, values in coefficients.items()}
            pre = sum(
                pre_events[t][edge] * math.exp(-(time - t) / params["tau_plus"])
                for t in range(time)
            )
            post = sum(
                post_events[t][edge] * math.exp(-(time - t) / params["tau_minus"])
                for t in range(time)
            )
            span = hi[old] - lo[old]
            q = min(1.0, max(0.0, (expected[edge] - lo[old]) / (span if span else 1.0)))
            delta = (
                params["a_plus"]
                * post_events[-1][edge]
                * pre
                * (1 - q) ** params["mu_plus"]
                - params["a_minus"]
                * pre_events[-1][edge]
                * post
                * q ** params["mu_minus"]
            )
            expected[edge] = min(hi[old], max(lo[old], expected[edge] + delta))
            pre_snapshots.append(pre + pre_events[-1][edge])
            post_snapshots.append(post + post_events[-1][edge])
        with torch.no_grad():
            execute(arrivals, posts, dt=1.0)
        for actual, values in (
            (connection.weight.value, expected),
            (connection.plasticity.pre_trace, pre_snapshots),
            (connection.plasticity.post_trace, post_snapshots),
        ):
            torch.testing.assert_close(
                actual,
                torch.tensor(values, dtype=torch.float64),
                atol=2e-12,
                rtol=2e-12,
            )


@pytest.mark.parametrize("pairing", ["symmetric", "restricted"])
def test_nearest_pair_stdp_matches_timestamp_pairing_and_resumes(pairing):
    """Resolve bursts, arrival delays and coincidence from spike timestamps."""

    def build(mode):
        return SparseConnection.from_edges(
            torch.tensor([0]),
            torch.tensor([1]),
            2,
            2,
            Synapse(
                weight=EdgeWeight(
                    torch.tensor([0.5], dtype=torch.float64), trainable=False
                ),
                delay=torch.tensor([2]),
                n_delay=3,
                plasticity=NearestPairSTDP(
                    a_plus=0.08,
                    a_minus=0.06,
                    tau_plus=2.0,
                    tau_minus=3.0,
                    pairing=mode,
                ),
            ),
        )

    connection = build(pairing)
    execute = torch.compile(
        connection.update_plasticity, backend="eager", fullgraph=True
    )
    resumed = None
    # Adjacent source spikes reach delay slot 2 two steps later. Bursts of
    # opposite spikes distinguish reuse (symmetric) from consumed eligibility
    # (restricted). Step 6 has both events and must use only earlier partners.
    source_times = {0, 1, 4, 5, 8, 9}
    post_times = {1, 4, 5, 6, 9, 12, 13}
    last_pre = last_post = None
    expected_weight = 0.5
    dt = 0.5
    rule = connection.plasticity
    a_plus, a_minus = float(rule.a_plus), float(rule.a_minus)
    tau_plus, tau_minus = float(rule.tau_plus), float(rule.tau_minus)
    for time in range(14):
        arrival = time - 2 in source_times
        post = time in post_times
        # This oracle selects partners by timestamps, independently of the
        # implementation's decayed/consumed eligibility traces.
        ltp_partner = last_pre is not None and (
            pairing == "symmetric" or last_post is None or last_pre >= last_post
        )
        ltd_partner = last_post is not None and (
            pairing == "symmetric" or last_pre is None or last_post >= last_pre
        )
        delta = 0.0
        if post and ltp_partner:
            delta += a_plus * math.exp(-(time - last_pre) * dt / tau_plus)
        if arrival and ltd_partner:
            delta -= a_minus * math.exp(-(time - last_post) * dt / tau_minus)
        expected_weight = min(1.0, max(0.0, expected_weight + delta))
        if arrival:
            last_pre = time
        if post:
            last_post = time
        expected_pre = (
            math.exp(-(time - last_pre) * dt / tau_plus)
            if last_pre is not None
            and (pairing == "symmetric" or last_post is None or last_pre >= last_post)
            else 0.0
        )
        expected_post = (
            math.exp(-(time - last_post) * dt / tau_minus)
            if last_post is not None
            and (pairing == "symmetric" or last_pre is None or last_post >= last_pre)
            else 0.0
        )
        arrivals = torch.zeros(connection.in_features, dtype=torch.float64)
        arrivals[2] = arrival
        posts = torch.tensor([0, post], dtype=torch.float64)
        with torch.no_grad():
            execute(arrivals, posts, dt=dt)
            if resumed is not None:
                resumed.update_plasticity(arrivals, posts, dt=dt)
        for model in (connection,) if resumed is None else (connection, resumed):
            for actual, expected in (
                (model.weight.value, expected_weight),
                (model.plasticity.pre_trace, expected_pre),
                (model.plasticity.post_trace, expected_post),
            ):
                torch.testing.assert_close(
                    actual,
                    torch.full_like(actual, expected),
                    atol=2e-12,
                    rtol=2e-12,
                )
        if time == 6:
            # Restore into the opposite constructor policy: the checkpoint
            # must restore pairing semantics, not just numerical memories.
            resumed = build("restricted" if pairing == "symmetric" else "symmetric")
            resumed.load_state_dict(copy.deepcopy(connection.state_dict()))
    learned = connection.weight.value.detach().clone()
    connection.plasticity.reset()
    torch.testing.assert_close(connection.weight.value, learned)
    torch.testing.assert_close(rule.pre_trace, torch.zeros_like(rule.pre_trace))
    torch.testing.assert_close(rule.post_trace, torch.zeros_like(rule.post_trace))


def test_voltage_stdp_matches_held_voltage_and_impulse_oracle():
    """Use closed-form filters and impulse timestamps, including silent
    learning."""
    coefficients = {
        "a_plus": [0.02, 0.025, 0.01, 0.03],
        "a_minus": [0.05, 0.07, 0.06, 0.04],
        "tau_pre": [2.0, 3.0, 4.0, 5.0],
        "tau_plus": [1.0, 2.0, 3.0],
        "tau_minus": [0.8, 1.5, 2.5],
        "theta_plus": [-50.0, -45.0, -45.0],
        "theta_minus": [-60.0, -58.0, -62.0],
        "v_rest": [-70.0, -65.0, -68.0],
    }
    options = VoltageSTDPOptions(
        **{
            name: torch.tensor(v, dtype=torch.float64)
            for name, v in coefficients.items()
        }
    )
    upper = torch.tensor(0.72, dtype=torch.float64)
    connection = _connection(VoltageSTDP(options, w_max=upper), values=VALUES.double())
    order = _input_edge_order(connection)
    expected = connection.weight.value.tolist()
    execute = torch.compile(
        connection.update_plasticity, backend="eager", fullgraph=True
    )
    # Each voltage is held over its interval. The closed-form oracle sums
    # interval responses rather than reproducing the implementation recurrence.
    voltages = [
        [-55.0, -40.0, -65.0],
        [-35.0, -30.0, -25.0],
        [-80.0, -80.0, -80.0],
        [-30.0, -35.0, -40.0],
        [-25.0, -30.0, -35.0],
        [-70.0, -65.0, -68.0],
    ]
    intervals, impulses, end = [], [], 0.0
    resumed = None
    silent_ltp = clipped = False
    for step, dt in enumerate((0.5, 0.25, 0.75, 0.5, 0.5, 1.0)):
        start, end = end, end + dt
        intervals.append((start, end, voltages[step]))
        filters = {}
        for name in ("plus", "minus"):
            filters[name] = [
                coefficients["v_rest"][j] * math.exp(-end / tau)
                + sum(
                    voltage[j]
                    * (math.exp(-(end - b) / tau) - math.exp(-(end - a) / tau))
                    for a, b, voltage in intervals
                )
                for j, tau in enumerate(coefficients["tau_" + name])
            ]
        arrivals = _arrivals(
            connection,
            torch.full(
                (connection.nnz,), float(step in (0, 1, 4)), dtype=torch.float64
            ),
        )
        edge_arrivals = arrivals[connection.pre * N_DELAY + connection.delay].tolist()
        next_pre = []
        for edge, old in enumerate(order):
            j, tau = int(POST[old]), coefficients["tau_pre"][old]
            pre = sum(
                events[edge] / tau * math.exp(-(end - t) / tau)
                for t, events in impulses
            )
            ltp = (
                dt
                * coefficients["a_plus"][old]
                * pre
                * max(0.0, voltages[step][j] - coefficients["theta_plus"][j])
                * max(0.0, filters["plus"][j] - coefficients["theta_minus"][j])
            )
            ltd = (
                coefficients["a_minus"][old]
                * edge_arrivals[edge]
                * max(0.0, filters["minus"][j] - coefficients["theta_minus"][j])
            )
            silent_ltp |= (
                not edge_arrivals[edge] and ltp > 0 and expected[edge] < float(upper)
            )
            proposed = expected[edge] + ltp - ltd
            expected[edge] = min(float(upper), max(0.0, proposed))
            clipped |= expected[edge] != proposed
            next_pre.append(pre + edge_arrivals[edge] / tau)
        impulses.append((end, edge_arrivals))
        storage = torch.zeros(2 * N_POST, dtype=torch.float64)
        storage[::2] = torch.tensor(voltages[step], dtype=torch.float64)
        kwargs = {"dt": dt, "signals": {"voltage": storage[::2]}}
        with torch.no_grad():
            execute(arrivals, torch.zeros(N_POST, dtype=torch.float64), **kwargs)
            if resumed is not None:
                resumed.update_plasticity(
                    arrivals, torch.zeros(N_POST, dtype=torch.float64), **kwargs
                )
        for model in (connection,) if resumed is None else (connection, resumed):
            for actual, value in (
                (model.weight.value, expected),
                (model.plasticity.pre_trace, next_pre),
                (model.plasticity.voltage_plus, filters["plus"]),
                (model.plasticity.voltage_minus, filters["minus"]),
            ):
                torch.testing.assert_close(
                    actual,
                    torch.tensor(value, dtype=torch.float64),
                    atol=3e-12,
                    rtol=3e-12,
                )
        if step == 2:
            # Loading into another reset baseline must restore that baseline,
            # not leave a stale Tensor reference in MemoryModule's reset policy.
            resumed = _connection(
                VoltageSTDP(
                    replace(options, v_rest=torch.zeros(N_POST, dtype=torch.float64)),
                    w_max=upper,
                ),
                values=VALUES.double(),
            )
            resumed.load_state_dict(copy.deepcopy(connection.state_dict()))
    assert silent_ltp and clipped
    for model in (connection, resumed):
        learned = model.weight.value.detach().clone()
        model.plasticity.reset(inplace=True)
        torch.testing.assert_close(model.weight.value, learned)
        torch.testing.assert_close(
            model.plasticity.pre_trace, torch.zeros_like(model.plasticity.pre_trace)
        )
        for name in ("voltage_plus", "voltage_minus"):
            torch.testing.assert_close(getattr(model.plasticity, name), options.v_rest)
        torch.testing.assert_close(model.plasticity.v_rest, options.v_rest)
    # Thresholds belong to posts, not edges. Reject equality/reversal after
    # broadcasting/binding the heterogeneous population coefficients.
    with pytest.raises(ValueError, match="theta_plus must exceed theta_minus"):
        _connection(
            VoltageSTDP(
                replace(
                    options,
                    theta_plus=torch.tensor([-60.0, -70.0, -45.0], dtype=torch.float64),
                )
            ),
            values=VALUES.double(),
        )


def test_pair_stdp_ltp_uses_a_strictly_past_arrival_trace():
    """A past arrival pairs with a later post spike after pre-trace decay."""
    conn = _connection()
    rule = conn.plasticity
    slot = 0
    arrival, post = _edge_event(conn, slot)
    no_arrival = torch.zeros_like(arrival)
    no_post = torch.zeros_like(post)
    decay = torch.exp(torch.tensor(-1.0 / 20.0))
    initial = conn.weight.value.detach().clone()

    with torch.no_grad():
        conn.update_plasticity(arrival, no_post, dt=1.0)
        torch.testing.assert_close(conn.weight.value, initial)
        # A one-element runtime tensor is a scalar clock even if a caller
        # retains singleton dimensions; it must not broadcast the edge state.
        conn.update_plasticity(no_arrival, post, dt=torch.tensor([[1.0]]))

    expected = initial[slot] + 0.1 * decay
    torch.testing.assert_close(conn.weight.value[slot], expected)
    torch.testing.assert_close(rule.pre_trace[slot], decay)
    torch.testing.assert_close(rule.post_trace[slot], torch.tensor(1.0))


def test_pair_stdp_ltd_uses_delayed_arrival_and_past_post_trace():
    """A delayed arrival uses the post trace from its target edge."""
    conn = _connection()
    rule = conn.plasticity
    slot = 2  # Its event occupies pre * n_delay + delay, not pre alone.
    arrival, post = _edge_event(conn, slot)
    no_arrival = torch.zeros_like(arrival)
    initial = conn.weight.value.detach().clone()
    decay = torch.exp(torch.tensor(-1.0 / 20.0))

    with torch.no_grad():
        conn.update_plasticity(no_arrival, post, dt=1.0)
        torch.testing.assert_close(conn.weight.value, initial)
        conn.update_plasticity(arrival, torch.zeros_like(post), dt=1.0)

    expected = initial[slot] - 0.12 * decay
    torch.testing.assert_close(conn.weight.value[slot], expected)
    torch.testing.assert_close(rule.post_trace[slot], decay)
    torch.testing.assert_close(rule.pre_trace[slot], torch.tensor(1.0))


def test_pair_stdp_coincident_events_do_not_pair_and_clip_combined_update():
    """Coincident events use old traces, and LTP/LTD are clipped together."""
    conn = _connection(_rule(a_plus=0.4, a_minus=0.3))
    slot = 0
    arrival, post = _edge_event(conn, slot)
    decay = torch.exp(torch.tensor(-1.0 / 20.0))

    with torch.no_grad():
        conn.weight.value[slot] = 0.95
        # The first coincident pair has no old trace and therefore no update.
        conn.update_plasticity(arrival, post, dt=1.0)
        torch.testing.assert_close(conn.weight.value[slot], torch.tensor(0.95))
        # Both terms are computed from the same decayed snapshots.  Clipping
        # only their combined result reaches the upper bound; clipping each
        # term independently would leave a different value.
        conn.update_plasticity(arrival, post, dt=1.0)

    expected = torch.tensor(1.0)
    torch.testing.assert_close(conn.weight.value[slot], expected)
    torch.testing.assert_close(conn.plasticity.pre_trace[slot], 1.0 + decay)
    torch.testing.assert_close(conn.plasticity.post_trace[slot], 1.0 + decay)


def test_pair_stdp_multistep_weight_trajectory_matches_trace_recurrence():
    """A three-step LTP/LTD sequence follows the decayed trace snapshots."""
    conn = _connection(_rule(a_plus=0.2, a_minus=0.3, tau_plus=4.0, tau_minus=5.0))
    slot = 1
    arrival, post = _edge_event(conn, slot)
    no_arrival = torch.zeros_like(arrival)
    no_post = torch.zeros_like(post)
    initial = conn.weight.value.detach().clone()
    pre_decay = torch.exp(torch.tensor(-1.0 / 4.0))
    post_decay = torch.exp(torch.tensor(-1.0 / 5.0))

    with torch.no_grad():
        conn.update_plasticity(arrival, no_post, dt=1.0)
        torch.testing.assert_close(conn.weight.value[slot], initial[slot])

        conn.update_plasticity(no_arrival, post, dt=1.0)
        after_ltp = initial[slot] + 0.2 * pre_decay
        torch.testing.assert_close(conn.weight.value[slot], after_ltp)

        conn.update_plasticity(arrival, no_post, dt=1.0)

    expected = after_ltp - 0.3 * post_decay
    torch.testing.assert_close(conn.weight.value[slot], expected)
    torch.testing.assert_close(
        conn.plasticity.pre_trace[slot], pre_decay.square() + 1.0
    )
    torch.testing.assert_close(conn.plasticity.post_trace[slot], post_decay)


def test_pair_stdp_heterogeneous_parameters_follow_canonical_edge_order():
    """Per-edge parameters and routed events follow canonical sorting."""
    a_plus = torch.tensor([0.01, 0.02, 0.03, 0.04])
    a_minus = torch.tensor([0.11, 0.12, 0.13, 0.14])
    tau_plus = torch.tensor([10.0, 20.0, 30.0, 40.0])
    tau_minus = torch.tensor([11.0, 21.0, 31.0, 41.0])
    w_min = torch.zeros(4)
    w_max = torch.ones(4)
    conn = _connection(
        _rule(
            a_plus=a_plus,
            a_minus=a_minus,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
            w_min=w_min,
            w_max=w_max,
        )
    )
    rule = conn.plasticity

    # The input order is intentionally unsorted.  Match each canonical edge
    # by all semantic attributes to establish the expected parameter order.
    input_edges = {
        (int(pre), int(post), int(delay), int(receptor)): i
        for i, (pre, post, delay, receptor) in enumerate(
            zip(PRE, POST, DELAY, RECEPTOR, strict=True)
        )
    }
    canonical_source = torch.tensor(
        [
            input_edges[(int(pre), int(post), int(delay), int(receptor))]
            for pre, post, delay, receptor in zip(
                conn.pre, conn.post, conn.delay, conn.receptor, strict=True
            )
        ]
    )
    torch.testing.assert_close(rule.a_plus, a_plus[canonical_source])
    torch.testing.assert_close(rule.a_minus, a_minus[canonical_source])
    torch.testing.assert_close(rule.tau_plus, tau_plus[canonical_source])
    torch.testing.assert_close(rule.tau_minus, tau_minus[canonical_source])

    edge_values = torch.ones(conn.nnz, dtype=conn.weight.value.dtype)
    arrivals = _arrivals(conn, edge_values)
    posts = _posts(conn, edge_values)
    initial = conn.weight.value.detach().clone()
    expected = initial + rule.a_plus * torch.exp(-1.0 / rule.tau_plus)

    with torch.no_grad():
        conn.update_plasticity(arrivals, torch.zeros_like(posts), dt=1.0)
        conn.update_plasticity(torch.zeros_like(arrivals), posts, dt=1.0)

    torch.testing.assert_close(conn.weight.value, expected)


def test_pair_stdp_state_dict_resumes_and_reset_preserves_learned_weights():
    """Traces serialize for continuation, while reset clears only traces."""
    conn = _connection()
    arrival, post = _edge_event(conn, 1)
    with torch.no_grad():
        conn.update_plasticity(arrival, torch.zeros_like(post), dt=1.0)
        conn.update_plasticity(torch.zeros_like(arrival), post, dt=1.0)
    checkpoint = io.BytesIO()
    torch.save(conn.state_dict(), checkpoint)
    checkpoint.seek(0)

    resumed = _connection()
    result = resumed.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert not result.missing_keys and not result.unexpected_keys
    torch.testing.assert_close(resumed.weight.value, conn.weight.value)
    torch.testing.assert_close(resumed.plasticity.pre_trace, conn.plasticity.pre_trace)
    torch.testing.assert_close(
        resumed.plasticity.post_trace, conn.plasticity.post_trace
    )

    next_arrival, next_post = _edge_event(conn, 2)
    with torch.no_grad():
        conn.update_plasticity(next_arrival, next_post, dt=1.0)
        resumed.update_plasticity(next_arrival, next_post, dt=1.0)
    torch.testing.assert_close(resumed.weight.value, conn.weight.value)
    torch.testing.assert_close(resumed.plasticity.pre_trace, conn.plasticity.pre_trace)
    torch.testing.assert_close(
        resumed.plasticity.post_trace, conn.plasticity.post_trace
    )

    learned = conn.weight.value.detach().clone()
    conn.plasticity.reset()
    torch.testing.assert_close(conn.weight.value, learned)
    torch.testing.assert_close(
        conn.plasticity.pre_trace, torch.zeros_like(conn.plasticity.pre_trace)
    )
    torch.testing.assert_close(
        conn.plasticity.post_trace, torch.zeros_like(conn.plasticity.post_trace)
    )


def test_pair_stdp_requires_independent_weights_and_explicit_online_updates():
    """Keep independent edge learning separate from BPTT and shared storage."""
    conn = _connection()
    arrivals = torch.zeros(conn.in_features)
    posts = torch.zeros(conn.n_post)
    with pytest.raises(RuntimeError, match="no_grad|inference_mode"):
        conn.update_plasticity(arrivals, posts, dt=1.0)

    # An adopted Parameter can be a view. Broadcasting one scalar to two
    # edge slots would make independently owned GPU writes race and eager
    # copy_ fail. Reject that storage at binding, before any learning occurs.
    shared = torch.nn.Parameter(torch.tensor([0.3]).expand(2), requires_grad=False)
    with pytest.raises(ValueError, match="independent writable edge weights"):
        SparseConnection.from_edges(
            torch.tensor([1, 0]),
            torch.tensor([0, 1]),
            2,
            2,
            Synapse(weight=EdgeWeight(shared), plasticity=PairSTDP()),
        )

    # Nonoverlapping strided weights remain valid: LTP on only the first
    # edge must change that edge alone and leave the interleaved padding.
    storage = torch.tensor([0.3, -9.0, 0.3, -9.0])
    weights = torch.nn.Parameter(storage[::2], requires_grad=False)
    strided = SparseConnection.from_edges(
        torch.tensor([1, 0]),
        torch.tensor([0, 1]),
        2,
        2,
        Synapse(weight=EdgeWeight(weights), plasticity=PairSTDP()),
    )
    with torch.no_grad():
        strided.plasticity.pre_trace.copy_(torch.tensor([1.0, 0.0]))
        strided.update_plasticity(torch.zeros(2), torch.tensor([1.0, 0.0]), dt=1.0)
    expected = torch.tensor([0.3 + 0.01 * torch.exp(torch.tensor(-1 / 20)), 0.3])
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(storage[1::2], torch.full((2,), -9.0))

    # Replacing a bound Parameter must not bypass ownership validation or
    # partially advance trace memories before the invalid update is rejected.
    previous_trace = strided.plasticity.pre_trace.clone()
    strided.weight.value = shared
    with torch.no_grad(), pytest.raises(ValueError, match="independent writable"):
        strided.update_plasticity(torch.zeros(2), torch.ones(2), dt=1.0)
    torch.testing.assert_close(strided.plasticity.pre_trace, previous_trace)
    strided.weight.value = torch.nn.Parameter(torch.tensor([0.3]), requires_grad=False)
    with torch.no_grad(), pytest.raises(ValueError, match="2 canonical edge weights"):
        strided.update_plasticity(torch.zeros(2), torch.ones(2), dt=1.0)
    torch.testing.assert_close(strided.plasticity.pre_trace, previous_trace)

    # Generic memory helpers can allocate batch axes. Online learning owns
    # one canonical trace set, so reject a batched reset before changing any
    # weights; the ordinary singleton batch used by init_net_state is valid.
    strided.weight.value = weights
    strided.plasticity.reset(batch_size=2)
    previous_weight = weights.detach().clone()
    with torch.no_grad(), pytest.raises(NotImplementedError, match="one sample"):
        strided.update_plasticity(torch.zeros(2), torch.ones(2), dt=1.0)
    torch.testing.assert_close(weights, previous_weight)
    torch.testing.assert_close(
        strided.plasticity.pre_trace, torch.zeros_like(strided.plasticity.pre_trace)
    )
    strided.plasticity.reset(batch_size=1)
    with torch.no_grad():
        strided.update_plasticity(torch.ones(2), torch.zeros(2), dt=1.0)
    torch.testing.assert_close(strided.plasticity.pre_trace, torch.ones(1, 2))


def _recurrent_model(
    *,
    chunk_size: int | None = None,
    grad_checkpoint: bool = False,
    rule: PlasticityRule | None = None,
    pre_spike_v: bool = False,
):
    """Build a small recurrent model with semantic delay and receptor
    routing."""
    connection = SparseConnection.from_edges(
        PRE,
        POST,
        N_PRE,
        N_POST,
        Synapse(
            weight=EdgeWeight(torch.tensor([0.08, 0.07, 0.06, 0.05]), trainable=False),
            delay=DELAY,
            receptor=RECEPTOR,
            n_delay=N_DELAY,
            n_receptor=N_RECEPTOR,
            plasticity=rule
            or _rule(
                a_plus=0.15,
                a_minus=0.1,
                tau_plus=4.0,
                tau_minus=5.0,
            ),
        ),
    )
    synapse = HeterSynapsePSC(
        N_POST,
        N_RECEPTOR,
        None,
        connection,
        base_psc=ExponentialPSC,
        tau_syn=2.0,
        max_delay_steps=N_DELAY,
        use_circular_buffer=False,
    )
    model = RecurrentNN(
        LIF(
            N_POST,
            v_threshold=0.5,
            v_reset=0.0,
            tau=100.0,
            hard_reset=True,
            pre_spike_v=pre_spike_v,
        ),
        synapse,
        backend="torch",
        unroll=1,
        chunk_size=chunk_size,
        grad_checkpoint=grad_checkpoint,
    )
    # Persistence is part of this integration contract: a checkpoint must
    # include neuron, PSC, delay-history, weight and STDP trace state.
    init_net_state(model, dtype=torch.float32, persistent=True)
    return model


def _recurrent_inputs() -> torch.Tensor:
    """Inputs that exercise both recurrent spikes and delayed arrivals."""
    return torch.tensor(
        [
            [0.7, 0.2, 0.0],
            [0.7, 0.7, 0.2],
            [0.0, 0.7, 0.7],
            [0.8, 0.0, 0.8],
            [0.0, 0.8, 0.0],
        ]
    )


def test_shared_pair_recurrent_chunks_preserve_physics_and_domain_state():
    """Keep delay/PSC timing while recording compact persistent traces."""
    edge = _recurrent_model(rule=_rule(tau_plus=4.0, tau_minus=5.0))
    shared = _recurrent_model(
        chunk_size=2,
        rule=_rule(tau_plus=4.0, tau_minus=5.0, trace_scope="neuron"),
    )
    with torch.no_grad(), environ.context(dt=1.0):
        expected, expected_records = edge(_recurrent_inputs())
        actual, records = shared(_recurrent_inputs())
    torch.testing.assert_close(actual, expected)
    for name in ("neuron.v", "synapse.psc"):
        torch.testing.assert_close(records[name], expected_records[name])
    actual_connection, expected_connection = shared.synapse.linear, edge.synapse.linear
    torch.testing.assert_close(
        actual_connection.weight.value, expected_connection.weight.value
    )
    # The final memory and its history use compact public domain shapes.
    # Gathering is only for this oracle comparison, not a production copy.
    for name, index in (
        ("pre_trace", actual_connection.cache.edge_segment),
        ("post_trace", actual_connection.post),
    ):
        torch.testing.assert_close(
            getattr(actual_connection.plasticity, name).reshape(-1)[index],
            getattr(expected_connection.plasticity, name).reshape(-1),
        )
        key = "synapse.linear.plasticity." + name
        torch.testing.assert_close(records[key][..., index], expected_records[key])


def test_voltage_stdp_recurrent_charged_state_and_chunk_continuation():
    """Bind live pre-reset voltage and preserve post-owned filters across
    chunks."""
    options = VoltageSTDPOptions(
        a_plus=0.08,
        a_minus=0.06,
        tau_pre=4.0,
        tau_plus=2.0,
        tau_minus=3.0,
        theta_plus=0.4,
        theta_minus=0.1,
        v_rest=0.0,
    )
    whole = _recurrent_model(rule=VoltageSTDP(options), pre_spike_v=True)
    split = _recurrent_model(rule=VoltageSTDP(options), pre_spike_v=True, chunk_size=2)
    split.load_state_dict(whole.state_dict())
    inputs = _recurrent_inputs()
    with torch.no_grad(), environ.context(dt=0.5):
        expected, records = whole(inputs)
        first, _ = split(inputs[:2])
        second, _ = split(inputs[2:])
    torch.testing.assert_close(torch.cat((first, second)), expected)
    for name, value in whole.state_dict().items():
        torch.testing.assert_close(split.state_dict()[name], value)
    assert expected.any()
    # The voltage observation must survive a neuron reset. Evaluate the filter
    # convolution from recorded charged values, not from the learning module.
    charged = records["neuron.v_pre_spike"].reshape(len(inputs), N_POST)
    assert (charged > whole.neuron.v_threshold).any()
    for name, tau in (
        ("voltage_plus", options.tau_plus),
        ("voltage_minus", options.tau_minus),
    ):
        decay = math.exp(-0.5 / tau)
        filtered = sum(
            charged[t] * (1 - decay) * decay ** (len(inputs) - 1 - t)
            for t in range(len(inputs))
        )
        torch.testing.assert_close(
            getattr(whole.synapse.linear.plasticity, name), filtered
        )


def test_custom_rule_recurrent_chunks_and_native_rejection_preserve_state():
    """Custom memories follow the recurrent lifecycle without a pair
    template."""
    recurrent = _recurrent_model(rule=CustomTriplet())
    resumed = _recurrent_model(rule=CustomTriplet())
    inputs = _recurrent_inputs()
    with torch.no_grad(), environ.context(dt=1.0):
        expected, _ = recurrent(inputs)
        first, _ = resumed(inputs[:2])
        second, _ = resumed(inputs[2:])
    torch.testing.assert_close(expected, torch.cat((first, second)))
    for name, value in recurrent.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name])

    # Native capability belongs to the actual equations, not merely the
    # presence of two familiar trace names. Fail before advancing any state.
    from btorch.models import KernelOptions

    recurrent.backend = KernelOptions(neuron="triton", connection="triton")
    snapshot = copy.deepcopy(recurrent.state_dict())
    with torch.no_grad(), pytest.raises(NotImplementedError, match="native plasticity"):
        recurrent(inputs)
    for name, value in recurrent.state_dict().items():
        torch.testing.assert_close(value, snapshot[name])

    # Opting explicitly into native pair equations cannot silently discard a
    # learning signal that those equations do not consume. Reject before the
    # neuron or any plasticity memory advances, even if all inputs are present.
    class SignalledPair(PairSTDP):
        kernel_model = "pair_stdp"

        def __init__(self):
            super().__init__()
            self.signals = {"gain": ExternalSignal()}

    signalled = _recurrent_model(rule=SignalledPair())
    signalled.backend = recurrent.backend
    snapshot = copy.deepcopy(signalled.state_dict())
    with torch.no_grad(), pytest.raises(NotImplementedError, match="no live signal"):
        signalled(inputs, plasticity_signals={"gain": torch.tensor(1.0)})
    for name, value in signalled.state_dict().items():
        torch.testing.assert_close(value, snapshot[name])

    class GainedPair(PairSTDP):
        """Scale eager learning with a coefficient absent from native math."""

        kernel_model = "pair_stdp"

        def __init__(self):
            super().__init__()
            self._register_coefficient("gain", 2.0)

        def _weight_delta(self, weight, events, pre, post):
            return self.gain * super()._weight_delta(weight, events, pre, post)

    gained = _recurrent_model(rule=GainedPair())
    with torch.no_grad(), environ.context(dt=1.0):
        gained(inputs[:2])
    gained.backend = recurrent.backend
    snapshot = copy.deepcopy(gained.state_dict())
    # Explicitly opting into a model name cannot silently discard additional
    # mathematical coefficients, even when the state-domain schema matches.
    with (
        torch.no_grad(),
        pytest.raises(NotImplementedError, match="runtime and reset-only"),
    ):
        gained(inputs)
    for name, value in gained.state_dict().items():
        torch.testing.assert_close(value, snapshot[name])

    class DomainPair(PairSTDP):
        """An eager pair law with a population-owned pre-trace coefficient."""

        kernel_model = "pair_stdp"

        def __init__(self, domain):
            # Four source/delay segments happen to equal four canonical edges.
            # A size-only check cannot distinguish those two index spaces.
            size = N_POST if domain == "post" else len(PRE)
            super().__init__(tau_plus=torch.arange(size, dtype=torch.float32) + 2)
            self.parameter_domains = {"tau_plus": domain}
            self.domain = domain

        def single_step_forward(self, weight, events, dt):
            index = events.post_index if self.domain == "post" else events.segment_index
            pre = self.pre_trace.reshape(-1) * torch.exp(-dt / self.tau_plus[index])
            post = self.post_trace.reshape(-1) * torch.exp(-dt / self.tau_minus)
            delta = self._weight_delta(weight, events, pre, post)
            self._advance_traces(events, pre, post)
            return delta

    for domain in ("post", "segment"):
        declared = _recurrent_model(rule=DomainPair(domain))
        with torch.no_grad(), environ.context(dt=1.0):
            declared(inputs[:2])
        declared.backend = recurrent.backend
        snapshot = copy.deepcopy(declared.state_dict())
        with (
            torch.no_grad(),
            pytest.raises(NotImplementedError, match="canonical-edge coefficients"),
        ):
            declared(inputs)
        for name, value in declared.state_dict().items():
            torch.testing.assert_close(value, snapshot[name])


def test_recurrent_stdp_delay_chunk_and_checkpoint_follow_old_weight_oracle():
    """Recurrent scheduling updates STDP after propagation with old weights."""
    inputs = _recurrent_inputs()
    step_model = _recurrent_model()
    chunk_model = _recurrent_model(chunk_size=2)
    initial = copy.deepcopy(step_model.state_dict())
    chunk_model.load_state_dict(initial)

    connection = step_model.synapse.linear
    rule = connection.plasticity
    expected_weight = connection.weight.value.detach().clone()
    expected_pre = torch.zeros_like(rule.pre_trace)
    expected_post = torch.zeros_like(rule.post_trace)
    step_outputs = []

    with torch.no_grad(), environ.context(dt=1.0):
        for current in inputs:
            output, _ = step_model.single_step_forward(current)
            step_outputs.append(output)
            arrivals = step_model.synapse.delayed_spike.get_flattened(
                step_model.synapse.n_delay
            ).clone()
            arrival_edge = arrivals[
                connection.pre * connection.n_delay + connection.delay
            ]
            post_edge = output[connection.post]
            pre_decay = expected_pre * torch.exp(-1.0 / rule.tau_plus)
            post_decay = expected_post * torch.exp(-1.0 / rule.tau_minus)
            expected_weight = torch.clamp(
                expected_weight
                + rule.a_plus * post_edge * pre_decay
                - rule.a_minus * arrival_edge * post_decay,
                rule.w_min,
                rule.w_max,
            )
            expected_pre = pre_decay + arrival_edge
            expected_post = post_decay + post_edge
            torch.testing.assert_close(connection.weight.value, expected_weight)
            torch.testing.assert_close(rule.pre_trace, expected_pre)
            torch.testing.assert_close(rule.post_trace, expected_post)

        first_output, _ = chunk_model(inputs[:2])
        checkpoint = copy.deepcopy(chunk_model.state_dict())
        second_output, _ = chunk_model(inputs[2:])

    torch.testing.assert_close(
        torch.stack(step_outputs), torch.cat((first_output, second_output))
    )
    torch.testing.assert_close(
        chunk_model.synapse.linear.weight.value,
        step_model.synapse.linear.weight.value,
    )
    torch.testing.assert_close(
        chunk_model.synapse.linear.plasticity.pre_trace,
        step_model.synapse.linear.plasticity.pre_trace,
    )
    torch.testing.assert_close(
        chunk_model.synapse.linear.plasticity.post_trace,
        step_model.synapse.linear.plasticity.post_trace,
    )
    torch.testing.assert_close(
        chunk_model.synapse.delayed_spike.history,
        step_model.synapse.delayed_spike.history,
    )

    resumed = _recurrent_model(chunk_size=2)
    resumed.load_state_dict(checkpoint)
    with torch.no_grad(), environ.context(dt=1.0):
        resumed_output, _ = resumed(inputs[2:])
    torch.testing.assert_close(resumed_output, second_output)
    torch.testing.assert_close(
        resumed.synapse.linear.weight.value, chunk_model.synapse.linear.weight.value
    )
    torch.testing.assert_close(
        resumed.synapse.delayed_spike.history,
        chunk_model.synapse.delayed_spike.history,
    )

    unsupported = _recurrent_model(chunk_size=2, grad_checkpoint=True)
    with torch.no_grad(), environ.context(dt=1.0):
        with pytest.raises(NotImplementedError, match="checkpointing"):
            unsupported(inputs[:2])


def test_delay_segments_preserve_canonical_weights_and_refresh_after_rewiring():
    """Reconstruct routed propagation through nonempty source/delay segments.

    Canonical edges deliberately disagree with incoming and outgoing CSR
    orders. Rebuild the same cache with different routing to ensure
    prepared segments cannot retain old source/delay or canonical-weight
    mappings.
    """
    from btorch.sparse.runtime.cache import RepresentationCache

    cache = RepresentationCache()
    rows = torch.tensor([5, 0, 3, 2, 1])  # Three targets, two receptors each.
    columns = torch.tensor([14, 0, 7, 0, 11])  # Three sources, five delay bins.
    weight = torch.tensor([0.1, -0.2, 0.3, 0.4, -0.5], dtype=torch.float64)
    arrivals = torch.linspace(0.0, 1.4, 15, dtype=torch.float64)
    for version, (row, col) in enumerate(
        ((rows, columns), (rows.flip(0), columns.roll(1)))
    ):
        cache.build(row, col, (6, 15), version)
        src_ptr, pre, delay, ptr, edge_id, dst = cache.prepare_delay_segments(3, 5)
        # A segment-domain learning rule needs the inverse map in canonical
        # weight order, including multiple edges sharing one source/delay.
        edge_segment = cache.prepare_edge_segments(3, 5)
        torch.testing.assert_close(pre[edge_segment] * 5 + delay[edge_segment], col)
        observed = torch.zeros(6, dtype=weight.dtype)
        for source in range(3):
            for segment in range(int(src_ptr[source]), int(src_ptr[source + 1])):
                assert int(pre[segment]) == source
                first, last = int(ptr[segment]), int(ptr[segment + 1])
                activity = arrivals[source * 5 + delay[segment]]
                observed.index_add_(
                    0, dst[first:last], weight[edge_id[first:last]] * activity
                )
        expected = torch.zeros_like(observed).index_add_(0, row, weight * arrivals[col])
        torch.testing.assert_close(observed, expected)


@pytest.mark.parametrize(
    ("trace_scope", "signal_domain"), (("connection", "edge"), ("neuron", "segment"))
)
def test_modulated_stdp_delayed_reward_matches_event_history(
    trace_scope, signal_domain
):
    """Reconstruct eligibility from pair timestamps, not the implementation.

    The routed arrivals include heterogeneous delays/receptors. Reward
    follows the pairing bin, reverses sign, and eventually clips
    weights. Long decay times exercise the stable eligibility integral.
    Shared and edge traces differ in ownership but produce the same pair
    impulses from zero state.
    """
    # The fifth edge shares the first edge's source/delay and target but
    # uses the other receptor. Its eligibility remains independent even
    # when the pair traces and teaching signal share one segment.
    connection = SparseConnection.from_edges(
        torch.cat((PRE, PRE[:1])),
        torch.cat((POST, POST[:1])),
        N_PRE,
        N_POST,
        Synapse(
            weight=EdgeWeight(
                torch.cat((VALUES, torch.tensor([0.45]))), trainable=False
            ),
            delay=torch.cat((DELAY, DELAY[:1])),
            receptor=torch.cat((RECEPTOR, 1 - RECEPTOR[:1])),
            n_delay=N_DELAY,
            n_receptor=N_RECEPTOR,
            plasticity=ModulatedSTDP(
                ModulatedSTDPOptions(
                    a_plus=0.2,
                    a_minus=0.1,
                    tau_plus=2.0,
                    tau_minus=3.0,
                    tau_eligibility=torch.tensor(
                        [2.0, 4.0, 1.0e10, 8.0, 5.0], dtype=torch.float64
                    ),
                    eta=0.5,
                    baseline=0.25,
                ),
                modulator=ExternalSignal(signal_domain),
                trace_scope=trace_scope,
                w_min=0.1,
                w_max=0.8,
            ),
        ),
    ).double()
    rule = connection.plasticity
    arrivals = torch.tensor(
        [
            [1, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float64,
    )
    posts = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        dtype=torch.float64,
    )
    edge_arrivals = arrivals[:, connection.pre * N_DELAY + connection.delay]
    edge_posts = posts[:, connection.post]
    # Segment signals follow ascending nonempty source/delay slots, whereas
    # weights follow postsynaptic canonical order. Build that map independently.
    keys = (connection.pre * N_DELAY + connection.delay).tolist()
    segment_keys = sorted(set(keys))
    edge_segment = torch.tensor([segment_keys.index(key) for key in keys])
    rewards = (0.0, 0.0, 2.0, -3.0, 0.0, 100.0)
    expected_weight = connection.weight.value.detach().clone()
    initial_weight = expected_weight.clone()
    impulses = []
    dt = 0.5
    with torch.no_grad():
        for step, reward in enumerate(rewards):
            lag = (step - torch.arange(step, dtype=torch.float64))[:, None] * dt
            pre = (edge_arrivals[:step] * torch.exp(-lag / rule.tau_plus)).sum(0)
            post = (edge_posts[:step] * torch.exp(-lag / rule.tau_minus)).sum(0)
            pair = (
                rule.a_plus * edge_posts[step] * pre
                - rule.a_minus * edge_arrivals[step] * post
            )
            # Eligibility from bin u is inserted at its end; at the start of
            # bin t it has decayed for (t-u-1)*dt. The new pair is excluded.
            old = sum(
                (
                    impulse * torch.exp(-(step - u - 1) * dt / rule.tau_eligibility)
                    for u, impulse in enumerate(impulses)
                ),
                torch.zeros_like(expected_weight),
            )
            signal_size = (
                len(segment_keys)
                if signal_domain == "segment"
                else expected_weight.numel()
            )
            signal = (
                reward
                * torch.arange(1, signal_size + 1, dtype=torch.float64)
                / signal_size
            )
            edge_signal = signal[edge_segment] if signal_domain == "segment" else signal
            integral = rule.tau_eligibility * (-torch.expm1(-dt / rule.tau_eligibility))
            expected_weight = (
                expected_weight
                + rule.eta * (edge_signal - rule.baseline) * old * integral
            ).clamp(rule.w_min, rule.w_max)
            connection.update_plasticity(
                arrivals[step],
                posts[step],
                dt=dt,
                signals={"modulator": signal},
            )
            impulses.append(pair)
            expected_eligibility = sum(
                (
                    impulse * torch.exp(-(step - u) * dt / rule.tau_eligibility)
                    for u, impulse in enumerate(impulses)
                ),
                torch.zeros_like(expected_weight),
            )
            torch.testing.assert_close(connection.weight.value, expected_weight)
            torch.testing.assert_close(
                rule.eligibility.reshape(-1), expected_eligibility
            )
            if step == 1:
                # Pairing creates eligibility, not an instantaneous weight
                # change. This distinguishes delayed modulation from PairSTDP.
                torch.testing.assert_close(
                    connection.weight.value,
                    initial_weight,
                )


def test_modulated_stdp_checkpoint_compile_and_reset_preserve_delayed_credit():
    """Resume a stored eligibility without replaying the spikes that made it.

    A missing signal fails before mutation. Restoring into a fresh
    connection then compiling a silent reward step must consume the
    saved eligibility; reset clears that credit without undoing learned
    weights.
    """

    def build(*, time_first=False):
        return _connection(
            ModulatedSTDP(
                ModulatedSTDPOptions(a_plus=0.2, eta=0.1),
                modulator=ExternalSignal(time_first=time_first),
            )
        )

    connection = build()
    arrival = torch.ones(N_PRE * N_DELAY)
    post = torch.ones(N_POST)
    zero = {"modulator": torch.tensor(0.0)}
    with torch.no_grad():
        connection.update_plasticity(arrival, post * 0, dt=1.0, signals=zero)
        connection.update_plasticity(arrival * 0, post, dt=1.0, signals=zero)
        saved = copy.deepcopy(connection.state_dict())
        # A time sequence and a held scalar have different continuation
        # semantics even if the checkpoint's tensor shapes happen to agree.
        with pytest.raises(ValueError, match="modulator declaration"):
            build(time_first=True).load_state_dict(saved)
        with pytest.raises(ValueError, match="requires signals"):
            connection.update_plasticity(arrival, post, dt=1.0)
        torch.testing.assert_close(
            connection.plasticity.eligibility, saved["plasticity.eligibility"]
        )
        resumed = build()
        resumed.load_state_dict(saved)
        compiled = torch.compile(
            resumed.update_plasticity, backend="eager", fullgraph=True
        )
        signal = {"modulator": torch.tensor(2.0)}
        connection.update_plasticity(arrival * 0, post * 0, dt=1.0, signals=signal)
        compiled(arrival * 0, post * 0, dt=1.0, signals=signal)
        torch.testing.assert_close(resumed.weight.value, connection.weight.value)
        for name in resumed.plasticity.state_domains:
            torch.testing.assert_close(
                getattr(resumed.plasticity, name), getattr(connection.plasticity, name)
            )
        assert (resumed.weight.value > saved["weight.value"]).all()
        learned = resumed.weight.value.clone()
        resumed.plasticity.reset()
        for name in resumed.plasticity.state_domains:
            assert not getattr(resumed.plasticity, name).any()
        torch.testing.assert_close(resumed.weight.value, learned)


@pytest.mark.parametrize("tau", (1.0e10, 1.0e-10))
def test_modulated_stdp_requires_representable_positive_decay_time(tau):
    """Reject coefficient overflow/underflow at the actual weight dtype.

    These values are valid in the source FP32 tensor but become infinity
    or zero in FP16. Accepting them would silently freeze or erase
    eligibility. Ordinary representable FP16 time constants remain
    supported.
    """
    with pytest.raises(ValueError, match="tau_eligibility must remain"):
        _connection(
            ModulatedSTDP(ModulatedSTDPOptions(tau_eligibility=tau)),
            values=VALUES.half(),
        )


def test_modulated_stdp_recurrence_slices_rewards_and_continues_delayed_state():
    """Split time-dependent teaching signals with the recurrent input stream.

    Explicit two-call continuation and the scheduler's internal chunks
    must preserve the same neuron, PSC, delayed-spike and eligibility
    state. Reward values differ by neuron and timestep so reusing a
    chunk's first signal cannot pass through constant or inactive inputs
    unnoticed.
    """

    def build(chunk_size):
        return _recurrent_model(
            chunk_size=chunk_size,
            rule=ModulatedSTDP(
                ModulatedSTDPOptions(
                    a_plus=0.2, a_minus=0.1, tau_eligibility=4.0, eta=0.1
                ),
                modulator=ExternalSignal("post", time_first=True),
                trace_scope="neuron",
            ),
        )

    whole, split = build(None), build(2)
    inputs = torch.tensor(
        [
            [0.8, 0.1, 0.6],
            [0.1, 0.8, 0.1],
            [0.6, 0.7, 0.2],
            [0.1, 0.1, 0.8],
            [0.9, 0.2, 0.6],
        ]
    )
    rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, -2.0, 3.0],
            [-1.0, 3.0, -2.0],
            [2.0, 1.0, -3.0],
        ]
    )
    with torch.no_grad(), environ.context(dt=1.0):
        expected, _ = whole(inputs, plasticity_signals={"modulator": rewards})
        first, _ = split(inputs[:3], plasticity_signals={"modulator": rewards[:3]})
        last, _ = split(inputs[3:], plasticity_signals={"modulator": rewards[3:]})
        torch.testing.assert_close(torch.cat((first, last)), expected)
        # State includes circular/delayed activity, not just the final raster.
        for name, value in whole.state_dict().items():
            actual = split.state_dict()[name]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual, value)
            else:
                assert actual == value
        assert whole.synapse.linear.plasticity.eligibility.abs().sum() > 0
