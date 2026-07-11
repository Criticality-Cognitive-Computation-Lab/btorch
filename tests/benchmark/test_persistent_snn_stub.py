import pytest
import torch

from btorch.backend.persistent_snn import (
    EventCSRGraph,
    PersistentSNNParams,
    WindowedSpikeEvents,
    make_empty_state,
    persistent_snn_forward,
)


def _make_events(
    *,
    t_steps: int,
    batch_size: int,
    n_pre: int,
    events_per_bucket: int,
    device: str = "cpu",
) -> WindowedSpikeEvents:
    """Create a deterministic windowed event tensor for contract tests."""

    n_buckets = t_steps * batch_size
    offsets = torch.arange(
        n_buckets + 1,
        device=device,
        dtype=torch.int32,
    )
    offsets = offsets * events_per_bucket
    nnz = int(offsets[-1].item())
    indices = torch.arange(nnz, device=device, dtype=torch.int32) % n_pre
    values = torch.ones(nnz, device=device, dtype=torch.float32)
    return WindowedSpikeEvents(
        offsets=offsets,
        indices=indices,
        values=values,
        shape=(t_steps, batch_size, n_pre),
    )


def _make_graph(
    *,
    n_pre: int,
    n_post: int,
    fanout: int,
    device: str = "cpu",
) -> EventCSRGraph:
    """Create a deterministic pre-row CSR graph for contract tests."""

    indptr = torch.arange(n_pre + 1, device=device, dtype=torch.int32) * fanout
    nnz = int(indptr[-1].item())
    indices = torch.arange(nnz, device=device, dtype=torch.int32) % n_post
    weight = torch.linspace(0.1, 1.0, nnz, device=device, dtype=torch.float32)
    delay = torch.zeros(nnz, device=device, dtype=torch.int32)
    return EventCSRGraph(
        indptr=indptr,
        indices=indices,
        weight=weight,
        delay=delay,
        shape=(n_pre, n_post),
    )


def test_persistent_snn_stub_returns_zero_dense_spikes_and_preserves_state():
    """The empty operator must preserve state and emit no dense spikes.

    This is the correctness proof for the scaffold: before a CUDA kernel is
    implemented, the mathematical operation is the identity on state and the
    zero map on emitted spikes.
    """

    t_steps, batch_size, n_pre, n_post = 5, 2, 7, 11
    events = _make_events(
        t_steps=t_steps,
        batch_size=batch_size,
        n_pre=n_pre,
        events_per_bucket=3,
    )
    graph = _make_graph(n_pre=n_pre, n_post=n_post, fanout=4)
    state = make_empty_state(batch_size, n_post)
    state.v.add_(0.25)
    state.psc.add_(0.5)
    assert state.refractory is not None
    state.refractory.add_(1.0)

    out = persistent_snn_forward(
        events,
        graph,
        state,
        PersistentSNNParams(window_size=t_steps),
        backend="torch_stub",
        return_mode="dense",
    )

    assert out.spikes is not None
    assert out.spike_events is None
    assert out.spikes.shape == (t_steps, batch_size, n_post)
    assert torch.count_nonzero(out.spikes) == 0
    torch.testing.assert_close(out.state.v, state.v)
    torch.testing.assert_close(out.state.psc, state.psc)
    torch.testing.assert_close(out.state.refractory, state.refractory)


def test_persistent_snn_stub_returns_empty_event_output():
    """Event-mode output must be a valid empty WindowedSpikeEvents object."""

    events = _make_events(t_steps=4, batch_size=3, n_pre=5, events_per_bucket=2)
    graph = _make_graph(n_pre=5, n_post=6, fanout=3)
    state = make_empty_state(3, 6)

    out = persistent_snn_forward(
        events,
        graph,
        state,
        backend="torch_stub",
        return_mode="both",
    )

    assert out.spikes is not None
    assert out.spike_events is not None
    assert out.spike_events.shape == (4, 3, 6)
    assert out.spike_events.indices.numel() == 0
    assert torch.count_nonzero(out.spike_events.offsets) == 0


def test_persistent_snn_stub_validates_shape_mismatch():
    """Validation should fail early when graph and event neuron counts differ."""

    events = _make_events(t_steps=2, batch_size=1, n_pre=5, events_per_bucket=1)
    graph = _make_graph(n_pre=4, n_post=6, fanout=2)
    state = make_empty_state(1, 6)

    with pytest.raises(ValueError, match="N_pre"):
        persistent_snn_forward(events, graph, state, backend="torch_stub")
