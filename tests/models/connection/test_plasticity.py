"""Analytical semantics of arrival-based pair STDP.

These tests deliberately use one unbatched sample and the public
``SparseConnection.update_plasticity`` entry point.  The expected values are
written from the recurrence in ``PairSTDP`` so the suite can serve as the
correctness oracle for later optimized implementations.
"""

from __future__ import annotations

import copy
import io

import pytest
import torch

from btorch.models import environ
from btorch.models.connection import EdgeWeight, PairSTDP, SparseConnection, Synapse
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


def _connection(rule: PairSTDP | None = None, values: torch.Tensor = VALUES):
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


def _edge_event(conn: SparseConnection, slot: int, value: float = 1.0):
    edge_values = torch.zeros(conn.nnz, dtype=conn.weight.value.dtype)
    edge_values[slot] = value
    return _arrivals(conn, edge_values), _posts(conn, edge_values)


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
        conn.update_plasticity(no_arrival, post, dt=1.0)

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


def test_pair_stdp_rejects_autograd_updates():
    """Online mutation is explicit and cannot be mixed into BPTT autograd."""
    conn = _connection()
    arrivals = torch.zeros(conn.in_features)
    posts = torch.zeros(conn.n_post)
    with pytest.raises(RuntimeError, match="no_grad|inference_mode"):
        conn.update_plasticity(arrivals, posts, dt=1.0)


def _recurrent_model(*, chunk_size: int | None = None, grad_checkpoint: bool = False):
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
            plasticity=_rule(
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
