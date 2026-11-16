"""Focused eager tests for canonical-edge short-term transmission."""

import io

import pytest
import torch

from btorch.models import environ, functional
from btorch.models.connection import (
    AllToAll,
    PairSTDP,
    Projection,
    SparseConnection,
    Synapse,
    TsodyksMarkram,
)


def _connection(rule=None, *, delay=None, receptor=None, dtype=torch.float32):
    pre = torch.tensor([0, 0, 1, 1])
    post = torch.tensor([0, 1, 0, 1])
    return SparseConnection.from_edges(
        pre,
        post,
        2,
        2,
        Synapse(
            weight=torch.ones(4, dtype=dtype),
            delay=delay,
            receptor=receptor,
            n_delay=3 if delay is not None else None,
            n_receptor=2 if receptor is not None else None,
            transmission=rule,
        ),
    )


def test_tsodyks_markram_depresses_and_recovers_silently():
    """Recover resources during silence while retaining utilization decay."""
    rule = TsodyksMarkram(U=0.5, tau_rec=2.0, tau_fac=4.0)
    connection = _connection(rule)
    # Singleton runtime clock dimensions must not turn efficacy into a
    # batched edge matrix or change the canonical memory layout.
    with torch.no_grad(), environ.context(dt=torch.tensor([[1.0]])):
        first = connection(torch.tensor([1.0, 0.0]))
        second = connection(torch.tensor([1.0, 0.0]))
        silent = connection(torch.tensor([0.0, 0.0]))

    torch.testing.assert_close(first, torch.full((2,), 0.5))
    torch.testing.assert_close(
        second, torch.full((2,), 0.48402175), atol=1e-6, rtol=1e-6
    )
    assert torch.all(silent == 0)
    torch.testing.assert_close(
        rule.x, torch.tensor([0.5224892, 1.0, 0.5224892, 1.0]), atol=4e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        rule.u, torch.tensor([0.5410331, 0.0, 0.5410331, 0.0]), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(connection.weight(), torch.ones(4))


def test_tsodyks_markram_uses_canonical_delayed_receptor_edges():
    """Scale each receptor/delay edge from its actual delayed arrival."""
    rule = TsodyksMarkram(U=torch.tensor([0.2, 0.4, 0.6, 0.8]), tau_rec=2.0)
    connection = _connection(
        rule,
        delay=torch.tensor([0, 1, 2, 0]),
        receptor=torch.tensor([0, 1, 0, 1]),
    )
    arrivals = torch.zeros(2 * 3)
    arrivals[[0, 1, 5]] = 1
    with torch.no_grad(), environ.context(dt=1.0):
        output = connection(arrivals)

    expected = torch.tensor([0.8, 0.0, 0.0, 0.4])
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(
        connection.transmission.x,
        torch.tensor([0.8, 0.4, 0.6, 1.0]),
    )


def test_transmission_composes_with_stdp_without_mutating_weight_order():
    """Transmission scales propagation while STDP owns learned weights."""
    pre = torch.tensor([0, 1])
    post = torch.tensor([0, 0])
    weight = torch.ones(2)
    connection = SparseConnection.from_edges(
        pre,
        post,
        2,
        1,
        Synapse(
            weight=torch.nn.Parameter(weight),
            transmission=TsodyksMarkram(U=0.5, tau_rec=2.0),
            plasticity=PairSTDP(a_plus=0.1, a_minus=0.0, w_max=2.0),
        ),
    )
    with torch.no_grad(), environ.context(dt=1.0):
        arrivals = torch.tensor([1.0, 0.0])
        output = connection(arrivals)
        before = connection.weight().clone()
        connection.update_plasticity(arrivals, torch.tensor([0.0]), dt=1.0)
        assert torch.equal(connection.weight(), before)
        second = connection(arrivals)
        before_second_update = connection.weight().clone()
        connection.update_plasticity(arrivals, torch.tensor([1.0]), dt=1.0)

    torch.testing.assert_close(output, torch.tensor([0.5]))
    torch.testing.assert_close(second, torch.tensor([0.34836733]))
    assert connection.weight()[0] > before_second_update[0]
    assert torch.equal(connection.weight()[1:], before_second_update[1:])
    torch.testing.assert_close(
        connection.transmission.x, torch.tensor([0.34836733, 1.0])
    )


def test_transmission_reset_and_state_dict_restore():
    """Preserve release memories through reset and ordinary serialization."""
    connection = _connection(TsodyksMarkram(U=0.4, tau_rec=3.0))
    # Standard network initialization keeps a singleton batch dimension.
    # It still represents one sample and must not flatten/rebind memories.
    functional.init_net_state(connection, batch_size=1)
    with torch.no_grad(), environ.context(dt=1.0):
        connection(torch.tensor([[1.0, 0.0]]))
    state = io.BytesIO()
    torch.save(connection.state_dict(), state)
    state.seek(0)
    restored = _connection(TsodyksMarkram(U=0.4, tau_rec=3.0))
    functional.init_net_state(restored, batch_size=1)
    restored.load_state_dict(torch.load(state, weights_only=True))
    torch.testing.assert_close(restored.transmission.x, connection.transmission.x)
    with torch.no_grad(), environ.context(dt=1.0):
        expected = connection(torch.tensor([[1.0, 0.0]]))
        actual = restored(torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(restored.transmission.x, connection.transmission.x)
    torch.testing.assert_close(restored.transmission.u, connection.transmission.u)
    connection.transmission.reset()
    torch.testing.assert_close(
        connection.transmission.x, torch.ones_like(connection.transmission.x)
    )
    torch.testing.assert_close(
        connection.transmission.u, torch.zeros_like(connection.transmission.u)
    )


def test_transmission_rejects_fractional_events_and_minibatches():
    """Keep the first eager contract binary and one-sample by construction."""
    connection = _connection(TsodyksMarkram())
    with torch.no_grad(), environ.context(dt=1.0):
        with pytest.raises(ValueError, match="binary"):
            connection(torch.tensor([0.5, 0.0]))
        with pytest.raises(NotImplementedError, match="one eager sample"):
            connection(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_transmission_preserves_low_precision_state_and_sequence(dtype):
    """Keep release arithmetic and persistent memories in the selected
    dtype."""
    connection = _connection(
        TsodyksMarkram(U=0.5, tau_rec=2.0, tau_fac=4.0), dtype=dtype
    )
    with torch.no_grad(), environ.context(dt=1.0):
        first = connection(torch.tensor([1.0, 0.0], dtype=dtype))
        second = connection(torch.tensor([1.0, 0.0], dtype=dtype))

    assert first.dtype is dtype
    assert second.dtype is dtype
    assert connection.transmission.x.dtype is dtype
    assert connection.transmission.u.dtype is dtype
    torch.testing.assert_close(
        second[0], torch.tensor(0.484, dtype=dtype), atol=8e-3, rtol=8e-3
    )
    torch.testing.assert_close(
        second, torch.full((2,), 0.484, dtype=dtype), atol=8e-3, rtol=8e-3
    )


def test_projection_keeps_transmission_on_the_sparse_realization():
    """Do not route a fixed-weight transmission projection to a structured
    op."""
    projection = Projection(
        2,
        1,
        AllToAll(),
        Synapse(weight=1.0, transmission=TsodyksMarkram(U=0.5)),
    )
    assert isinstance(projection.connection, SparseConnection)


def test_transmission_recurrent_chunks_keep_delay_and_release_state():
    """Continue physical neuron/PSC, delay and release memories across chunks.

    A standard one-sample network initialization adds singleton
    dimensions. Compare a full call with independent chunked calls,
    including a short tail; both must consume each delayed binary
    arrival exactly once.
    """
    from btorch.models.neurons import LIF
    from btorch.models.rnn import RecurrentNN
    from btorch.models.synapse import ExponentialPSC

    def network(chunk_size):
        connection = _connection(
            TsodyksMarkram(U=0.4, tau_rec=3.0, tau_fac=5.0),
            delay=torch.tensor([0, 1, 2, 0]),
        )
        model = RecurrentNN(
            LIF(2, tau=2.0, v_threshold=0.3, v_reset=0.0),
            ExponentialPSC(2, tau_syn=2.0, linear=connection),
            backend="torch",
            unroll=1,
            chunk_size=chunk_size,
        )
        functional.init_net_state(model, batch_size=1, persistent=True)
        return model

    whole, split = network(None), network(2)
    inputs = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
    )
    with torch.no_grad(), environ.context(dt=1.0):
        expected, _ = whole(inputs)
        first, _ = split(inputs[:3])
        second, _ = split(inputs[3:])
    torch.testing.assert_close(torch.cat((first, second)), expected)
    assert expected.count_nonzero() > 0
    expected_state = functional.named_hidden_states(whole)
    for name, value in functional.named_hidden_states(split).items():
        torch.testing.assert_close(value, expected_state[name])
    assert (split.synapse.linear.transmission.x < 1).any()
