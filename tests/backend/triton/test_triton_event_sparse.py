import pytest
import scipy.sparse
import torch

from btorch.backend.triton import (
    dense_spike_to_spike_list,
    post_span_spmm_from_spike_list,
    pre_span_spmm_from_spike_list,
)
from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.linear import SparseConn, SparseEventConn
from btorch.models.synapse import AlphaPSC, ExponentialPSC


def _build_dense_weight(device: str, dtype: torch.dtype) -> torch.Tensor:
    # Dense reference matrix with shape (n_pre, n_post). The test cases below
    # encode exactly the same connectivity into both the pre-span and
    # post-span padded CSR layouts so we can verify that every interface
    # produces the same postsynaptic current.
    return torch.tensor(
        [
            [0.2, 0.0, -0.5],
            [1.5, 0.3, 0.0],
            [0.0, -0.2, 0.8],
            [0.7, 0.0, 0.4],
        ],
        device=device,
        dtype=dtype,
    )


def _build_square_dense_weight(device: str, dtype: torch.dtype) -> torch.Tensor:
    # SparseConn is primarily used as a recurrent connection in this benchmark
    # path, so the integration tests cover the square n_neuron x n_neuron case.
    return torch.tensor(
        [
            [0.2, 0.0, -0.5, 0.1],
            [1.5, 0.3, 0.0, -0.4],
            [0.0, -0.2, 0.8, 0.0],
            [0.7, 0.0, 0.4, 0.2],
        ],
        device=device,
        dtype=dtype,
    )


def _build_pre_span_buffers(
    device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Row i lists all postsynaptic targets reached by presynaptic neuron i.
    row_length = torch.tensor([2, 2, 2, 2], device=device, dtype=torch.int64)
    ind = torch.tensor(
        [
            [0, 2],
            [0, 1],
            [1, 2],
            [0, 2],
        ],
        device=device,
        dtype=torch.int64,
    )
    weight = torch.tensor(
        [
            [0.2, -0.5],
            [1.5, 0.3],
            [-0.2, 0.8],
            [0.7, 0.4],
        ],
        device=device,
        dtype=dtype,
    )
    return row_length, ind, weight


def _build_post_span_buffers(
    device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # PostSpan in the GeNN sparse case uses the same presynaptic-row storage as
    # PreSpan. Only the traversal strategy changes.
    row_length = torch.tensor([2, 2, 2, 2], device=device, dtype=torch.int64)
    ind = torch.tensor(
        [
            [0, 2],
            [0, 1],
            [1, 2],
            [0, 2],
        ],
        device=device,
        dtype=torch.int64,
    )
    weight = torch.tensor(
        [
            [0.2, -0.5],
            [1.5, 0.3],
            [-0.2, 0.8],
            [0.7, 0.4],
        ],
        device=device,
        dtype=dtype,
    )
    return row_length, ind, weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sparse_event_conn_matches_dense_reference_for_both_modes():
    device = "cuda"
    dtype = torch.float32
    dense_weight = _build_dense_weight(device, dtype)
    spike = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    pre_row_length, pre_ind, pre_weight = _build_pre_span_buffers(device, dtype)
    post_row_length, post_ind, post_weight = _build_post_span_buffers(device, dtype)

    conn = SparseEventConn(
        in_features=dense_weight.shape[0],
        out_features=dense_weight.shape[1],
        event_mode="pre_span",
        device=device,
        dtype=dtype,
    )
    conn.set_pre_span_data(pre_row_length, pre_ind, pre_weight)
    conn.set_post_span_data(post_row_length, post_ind, post_weight)

    out_pre = conn.forward_events(spike, mode="pre_span")
    out_post = conn.forward_events(spike, mode="post_span")
    expected = spike @ dense_weight

    torch.testing.assert_close(out_pre, expected, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(out_post, expected, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("mode", ["pre_span", "post_span"])
def test_sparse_conn_forward_events_matches_dense_reference(mode: str):
    # SparseConn is the layer used by the RSNN benchmark. This test verifies
    # that its opt-in event interface preserves the same x @ W semantics as the
    # regular sparse forward, while reusing the existing learnable magnitude
    # parameter as the source of event weights.
    device = "cuda"
    dtype = torch.float32
    dense_weight = _build_square_dense_weight(device, dtype)
    conn = SparseConn(
        scipy.sparse.coo_array(dense_weight.cpu().numpy()),
        bias=None,
        enforce_dale=False,
        sparse_backend="native",
        device=device,
        dtype=dtype,
    )
    spike = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
        device=device,
        dtype=dtype,
    )

    out = conn.forward_events(spike, mode=mode)
    expected = conn(spike)

    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dense_spike_to_spike_list_matches_nonzero_reference():
    device = "cuda"
    spike = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )

    spike_count, spike_ind = dense_spike_to_spike_list(spike)

    expected_counts = torch.tensor([2, 3], device=device, dtype=torch.int32)
    torch.testing.assert_close(spike_count, expected_counts)
    for batch in range(spike.shape[0]):
        expected = torch.nonzero(spike[batch] > 0.5, as_tuple=False).flatten()
        actual = spike_ind[batch, : int(spike_count[batch].item())]
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_spike_list_span_kernels_match_dense_reference():
    device = "cuda"
    dtype = torch.float32
    dense_weight = _build_dense_weight(device, dtype)
    spike = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    row_length, ind, weight = _build_pre_span_buffers(device, dtype)
    spike_count, spike_ind = dense_spike_to_spike_list(spike)

    out_pre = pre_span_spmm_from_spike_list(
        spike_count,
        spike_ind,
        row_length,
        ind,
        weight,
        size_m=dense_weight.shape[1],
    )
    out_post = post_span_spmm_from_spike_list(
        spike_count,
        spike_ind,
        row_length,
        ind,
        weight,
        size_m=dense_weight.shape[1],
    )
    expected = spike @ dense_weight

    torch.testing.assert_close(out_pre, expected, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(out_post, expected, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sparse_event_conn_spike_list_matches_dense_reference_for_both_modes():
    device = "cuda"
    dtype = torch.float32
    dense_weight = _build_dense_weight(device, dtype)
    spike = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    row_length, ind, weight = _build_pre_span_buffers(device, dtype)
    spike_count, spike_ind = dense_spike_to_spike_list(spike)

    conn = SparseEventConn(
        in_features=dense_weight.shape[0],
        out_features=dense_weight.shape[1],
        event_mode="pre_span",
        device=device,
        dtype=dtype,
    )
    conn.set_pre_span_data(row_length, ind, weight)
    conn.set_post_span_data(row_length, ind, weight)

    out_pre = conn.forward_spike_list(spike_count, spike_ind, mode="pre_span")
    out_post = conn.forward_spike_list(spike_count, spike_ind, mode="post_span")
    expected = spike @ dense_weight

    torch.testing.assert_close(out_pre, expected, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(out_post, expected, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_exponential_psc_with_sparse_event_conn_matches_manual_reference():
    # This integration test verifies the exact seam we care about in the
    # project: ExponentialPSC should treat SparseEventConn as an event-driven
    # current source and add its output after exponential decay on each step.
    device = "cuda"
    dtype = torch.float32
    dt = 1.0
    tau_syn = 2.0
    dense_weight = _build_dense_weight(device, dtype)
    pre_row_length, pre_ind, pre_weight = _build_pre_span_buffers(device, dtype)
    conn = SparseEventConn(
        in_features=dense_weight.shape[0],
        out_features=dense_weight.shape[1],
        event_mode="pre_span",
        device=device,
        dtype=dtype,
    )
    conn.set_pre_span_data(pre_row_length, pre_ind, pre_weight)

    spike_seq = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0]],
        ],
        device=device,
        dtype=dtype,
    )

    with environ.context(dt=dt):
        synapse = ExponentialPSC(
            n_neuron=dense_weight.shape[1],
            tau_syn=tau_syn,
            linear=conn,
            latency=0.0,
            step_mode="m",
        )
        init_net_state(synapse, batch_size=1, device=device, dtype=dtype)
        out = synapse(spike_seq)

    decay = torch.exp(torch.tensor(-dt / tau_syn, device=device, dtype=dtype))
    psc = torch.zeros((1, dense_weight.shape[1]), device=device, dtype=dtype)
    expected_steps = []
    for spike_t in spike_seq:
        psc = psc * decay + spike_t @ dense_weight
        expected_steps.append(psc.clone())
    expected = torch.stack(expected_steps, dim=0)

    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("mode", ["pre_span", "post_span"])
def test_alpha_psc_sparse_conn_event_path_matches_default_no_grad(
    monkeypatch: pytest.MonkeyPatch, mode: str
):
    # AlphaPSC is what bench_rsnn.py uses. With the environment flag enabled
    # and gradients disabled, SparseConn should transparently switch to the
    # Triton event path without changing the PSC recurrence.
    device = "cuda"
    dtype = torch.float32
    dt = 1.0
    dense_weight = _build_square_dense_weight(device, dtype)
    weight_np = dense_weight.cpu().numpy()
    spike_seq = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0]],
        ],
        device=device,
        dtype=dtype,
    )

    with environ.context(dt=dt):
        baseline = AlphaPSC(
            n_neuron=dense_weight.shape[1],
            tau_syn=5.0,
            linear=SparseConn(
                scipy.sparse.coo_array(weight_np),
                bias=None,
                enforce_dale=False,
                sparse_backend="native",
                device=device,
                dtype=dtype,
            ),
            step_mode="m",
        )
        event = AlphaPSC(
            n_neuron=dense_weight.shape[1],
            tau_syn=5.0,
            linear=SparseConn(
                scipy.sparse.coo_array(weight_np),
                bias=None,
                enforce_dale=False,
                sparse_backend="native",
                device=device,
                dtype=dtype,
            ),
            step_mode="m",
        )
        init_net_state(baseline, batch_size=1, device=device, dtype=dtype)
        init_net_state(event, batch_size=1, device=device, dtype=dtype)

        monkeypatch.delenv("BTORCH_EVENT_SPARSE", raising=False)
        expected = baseline(spike_seq)

        monkeypatch.setenv("BTORCH_EVENT_SPARSE", "1")
        monkeypatch.setenv("BTORCH_EVENT_SPARSE_MODE", mode)
        with torch.no_grad():
            actual = event(spike_seq)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_alpha_psc_sparse_conn_event_path_falls_back_when_grad_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    # The event kernels are forward-only today. Even when the environment flag
    # is enabled, grad-enabled execution must keep using SparseConn.forward so
    # fwd+bwd benchmarks and training code still have a valid autograd graph.
    device = "cuda"
    dtype = torch.float32
    dense_weight = _build_square_dense_weight(device, dtype)
    conn = SparseConn(
        scipy.sparse.coo_array(dense_weight.cpu().numpy()),
        bias=None,
        enforce_dale=False,
        sparse_backend="native",
        device=device,
        dtype=dtype,
    )

    def fail_forward_events(*args, **kwargs):
        raise AssertionError("forward_events should not run with grad enabled")

    object.__setattr__(conn, "forward_events", fail_forward_events)
    with environ.context(dt=1.0):
        synapse = AlphaPSC(
            n_neuron=dense_weight.shape[1],
            tau_syn=5.0,
            linear=conn,
            step_mode="m",
        )
        init_net_state(synapse, batch_size=1, device=device, dtype=dtype)
        spike_seq = torch.tensor(
            [[[1.0, 0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0, 0.0]]],
            device=device,
            dtype=dtype,
        )
        monkeypatch.setenv("BTORCH_EVENT_SPARSE", "1")

        out = synapse(spike_seq)
    loss = out.sum()
    loss.backward()

    assert conn.magnitude.grad is not None
