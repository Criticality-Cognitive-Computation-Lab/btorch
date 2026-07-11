import math

import pytest
import torch

from btorch.backend.persistent_snn import (
    EventCSRGraph,
    PersistentSNNParams,
    PersistentSNNState,
    WindowedSpikeEvents,
    persistent_snn_forward,
)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return torch.device("cuda")


def _run_cuda_or_skip(*args, **kwargs):
    try:
        return persistent_snn_forward(*args, backend="cuda_persistent", **kwargs)
    except RuntimeError as exc:
        message = str(exc).lower()
        skippable = (
            "cuda_home",
            "nvcc",
            "ninja",
            "cooperative",
            "no kernel image",
            "unsupported",
        )
        if any(token in message for token in skippable):
            pytest.skip(f"CUDA persistent backend unavailable: {exc}")
        raise


def _dense_to_events(x_seq: torch.Tensor) -> WindowedSpikeEvents:
    """Convert dense external currents to the CUDA event contract."""

    t_steps, batch_size, n_neuron = x_seq.shape
    active = x_seq != 0
    counts = active.sum(dim=2, dtype=torch.int32).reshape(-1)
    offsets = torch.zeros(counts.numel() + 1, device=x_seq.device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts, dim=0)
    indices = torch.nonzero(active.reshape(-1, n_neuron), as_tuple=False)[:, 1]
    return WindowedSpikeEvents(
        offsets=offsets.contiguous(),
        indices=indices.to(torch.int32).contiguous(),
        values=x_seq[active].to(torch.float32).contiguous(),
        shape=(t_steps, batch_size, n_neuron),
    )


def _graph(device: torch.device) -> tuple[EventCSRGraph, torch.Tensor]:
    """Create a small recurrent graph with varied fanout for correctness tests."""

    n_neuron = 4
    edges = [
        (0, 1, 0.25),
        (0, 2, 0.50),
        (1, 2, 0.75),
        (2, 0, -0.25),
        (2, 3, 0.40),
        (3, 1, 0.10),
    ]
    indptr = torch.zeros(n_neuron + 1, device=device, dtype=torch.int32)
    for pre, _post, _weight in edges:
        indptr[pre + 1] += 1
    indptr = torch.cumsum(indptr, dim=0).to(torch.int32)
    indices = torch.tensor([post for _pre, post, _weight in edges], device=device)
    indices = indices.to(torch.int32)
    weight = torch.tensor([weight for _pre, _post, weight in edges], device=device)
    weight = weight.to(torch.float32)
    dense = torch.zeros(n_neuron, n_neuron, device=device)
    for pre, post, edge_weight in edges:
        dense[pre, post] = edge_weight
    return (
        EventCSRGraph(
            indptr=indptr.contiguous(),
            indices=indices.contiguous(),
            weight=weight.contiguous(),
            delay=None,
            shape=(n_neuron, n_neuron),
        ),
        dense,
    )


def _state(
    batch_size: int,
    n_neuron: int,
    device: torch.device,
    params: PersistentSNNParams | None = None,
) -> PersistentSNNState:
    """Create a complete GLIF3 + AlphaPSC persistent state for CUDA tests."""

    params = params or PersistentSNNParams()
    v = torch.full(
        (batch_size, n_neuron),
        params.v_reset,
        device=device,
        dtype=torch.float32,
    )
    psc = torch.zeros_like(v)
    return PersistentSNNState(
        v=v,
        psc=psc,
        psc_h=torch.zeros_like(v),
        asc=torch.zeros(batch_size, n_neuron, 2, device=device),
        refractory=torch.zeros_like(v),
    )


def _reference(
    x_seq: torch.Tensor,
    weight_dense: torch.Tensor,
    state: PersistentSNNState,
    params: PersistentSNNParams,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Dense reference matching the GLIF3 + AlphaPSC CUDA contract."""

    v = state.v.clone()
    psc = state.psc.clone()
    assert state.psc_h is not None
    assert state.asc is not None
    assert state.refractory is not None
    psc_h = state.psc_h.clone()
    asc = state.asc.clone()
    refractory = state.refractory.clone()
    syn_decay = math.exp(-params.dt / params.tau_syn)
    mem_decay = math.exp(-params.dt / params.tau)
    asc_decay = torch.tensor(
        [math.exp(-params.dt * params.k[0]), math.exp(-params.dt * params.k[1])],
        device=x_seq.device,
        dtype=x_seq.dtype,
    )
    asc_amps = torch.tensor(params.asc_amps, device=x_seq.device, dtype=x_seq.dtype)
    v_rest = params.v_reset if params.v_rest is None else params.v_rest
    spikes = []
    for t in range(x_seq.shape[0]):
        psc = syn_decay * psc + (1.0 - syn_decay) * psc_h
        psc_h = syn_decay * psc_h
        asc_before = asc
        asc = asc * asc_decay
        current = psc + x_seq[t] + asc_before.sum(dim=-1)
        v_inf = v_rest + params.tau * current / params.c_m
        v_pre = v_inf + (v - v_inf) * mem_decay
        z = ((v_pre >= params.v_threshold) & (refractory <= 0.0)).to(v.dtype)
        v = v_pre - (params.v_threshold - params.v_reset) * z
        refractory = torch.clamp(refractory - params.dt, min=0.0)
        refractory = torch.where(
            z > 0,
            torch.full_like(refractory, max(params.tau_ref - params.dt, 0.0)),
            refractory,
        )
        asc = asc + asc_amps * z[..., None]
        psc_h = psc_h + params.psc_g_max * (z @ weight_dense)
        spikes.append(z)
    return torch.stack(spikes, dim=0), v, psc, psc_h, asc, refractory


def test_cuda_persistent_matches_dense_reference():
    """CUDA persistent dynamics should match dense RSNN reference."""

    device = _require_cuda()
    x_seq = torch.tensor(
        [
            [[30.0, 0.0, 0.0, 0.0], [0.0, 32.0, 0.0, 0.0]],
            [[0.0, 0.0, 34.0, 0.0], [0.0, 0.0, 0.0, 36.0]],
            [[18.0, 0.0, 20.0, 0.0], [22.0, 0.0, 0.0, 0.0]],
        ],
        device=device,
        dtype=torch.float32,
    )
    graph, dense = _graph(device)
    params = PersistentSNNParams(tau_mem=20.0, tau_syn=5.0, window_size=3)
    state = _state(2, 4, device, params)

    out = _run_cuda_or_skip(
        _dense_to_events(x_seq),
        graph,
        state,
        params,
        return_mode="both",
    )
    ref_spikes, ref_v, ref_psc, ref_psc_h, ref_asc, ref_refractory = _reference(
        x_seq,
        dense,
        state,
        params,
    )

    assert out.spikes is not None
    assert out.spike_events is not None
    assert out.state.psc_h is not None
    assert out.state.asc is not None
    assert out.state.refractory is not None
    torch.testing.assert_close(out.spikes, ref_spikes, atol=0, rtol=0)
    torch.testing.assert_close(out.state.v, ref_v, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.state.psc, ref_psc, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.state.psc_h, ref_psc_h, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.state.asc, ref_asc, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        out.state.refractory,
        ref_refractory,
        atol=1e-5,
        rtol=1e-5,
    )


def test_cuda_persistent_event_output_matches_dense_spikes():
    """Returned event buckets should contain exactly the fired neuron indices."""

    device = _require_cuda()
    x_seq = torch.tensor(
        [[[30.0, 32.0, 0.0, 0.0]], [[0.0, 0.0, 34.0, 36.0]]],
        device=device,
        dtype=torch.float32,
    )
    graph, _dense = _graph(device)
    state = _state(1, 4, device)
    out = _run_cuda_or_skip(
        _dense_to_events(x_seq),
        graph,
        state,
        PersistentSNNParams(window_size=2),
        return_mode="both",
    )

    assert out.spikes is not None
    assert out.spike_events is not None
    offsets = out.spike_events.offsets.detach().cpu()
    indices = out.spike_events.indices.detach().cpu()
    dense_spikes = out.spikes.detach().cpu()
    for bucket in range(offsets.numel() - 1):
        start = int(offsets[bucket].item())
        end = int(offsets[bucket + 1].item())
        got = torch.sort(indices[start:end]).values
        expected = torch.nonzero(
            dense_spikes.reshape(-1, dense_spikes.shape[-1])[bucket],
            as_tuple=False,
        ).flatten()
        torch.testing.assert_close(got, expected.to(got.dtype))


def test_cuda_persistent_soft_reset_preserves_surplus_voltage():
    """Soft reset should subtract threshold delta rather than clamp to reset."""

    device = _require_cuda()
    params = PersistentSNNParams(
        v_threshold=1.0,
        v_reset=0.0,
        v_rest=0.0,
        c_m=1.0,
        tau_ref=0.0,
        window_size=1,
    )
    x_seq = torch.tensor([[[2.0]]], device=device)
    events = _dense_to_events(x_seq)
    graph = EventCSRGraph(
        indptr=torch.tensor([0, 0], device=device, dtype=torch.int32),
        indices=torch.empty(0, device=device, dtype=torch.int32),
        weight=torch.empty(0, device=device, dtype=torch.float32),
        delay=None,
        shape=(1, 1),
    )
    state = _state(1, 1, device, params)
    out = _run_cuda_or_skip(
        events,
        graph,
        state,
        params,
        return_mode="dense",
    )
    ref_spikes, ref_v, *_ = _reference(
        x_seq,
        torch.zeros(1, 1, device=device),
        state,
        params,
    )

    torch.testing.assert_close(out.spikes, ref_spikes)
    torch.testing.assert_close(out.state.v, ref_v, atol=1e-5, rtol=1e-5)
    assert torch.all(out.state.v > params.v_reset)


def test_cuda_persistent_rejects_unsupported_v1_options():
    """Python dispatch should reject unsupported v1 options before JIT load."""

    device = _require_cuda()
    events = _dense_to_events(torch.ones(1, 1, 1, device=device))
    graph = EventCSRGraph(
        indptr=torch.tensor([0, 1], device=device, dtype=torch.int32),
        indices=torch.tensor([0], device=device, dtype=torch.int32),
        weight=torch.tensor([0.0], device=device, dtype=torch.float32),
        delay=torch.tensor([1], device=device, dtype=torch.int32),
        shape=(1, 1),
    )
    state = _state(1, 1, device)

    with pytest.raises(ValueError, match="hard_reset"):
        persistent_snn_forward(
            events,
            EventCSRGraph(
                indptr=graph.indptr,
                indices=graph.indices,
                weight=graph.weight,
                delay=None,
                shape=graph.shape,
            ),
            state,
            PersistentSNNParams(hard_reset=True),
            backend="cuda_persistent",
        )
    with pytest.raises(ValueError, match="psc_h"):
        persistent_snn_forward(
            events,
            EventCSRGraph(
                indptr=graph.indptr,
                indices=graph.indices,
                weight=graph.weight,
                delay=None,
                shape=graph.shape,
            ),
            PersistentSNNState(
                v=state.v,
                psc=state.psc,
                refractory=torch.zeros_like(state.v),
            ),
            backend="cuda_persistent",
        )
    with pytest.raises(ValueError, match="asc"):
        persistent_snn_forward(
            events,
            EventCSRGraph(
                indptr=graph.indptr,
                indices=graph.indices,
                weight=graph.weight,
                delay=None,
                shape=graph.shape,
            ),
            PersistentSNNState(
                v=state.v,
                psc=state.psc,
                psc_h=state.psc_h,
                refractory=state.refractory,
            ),
            backend="cuda_persistent",
        )
    with pytest.raises(ValueError, match="nonzero delay"):
        persistent_snn_forward(
            events,
            graph,
            state,
            backend="cuda_persistent",
        )
