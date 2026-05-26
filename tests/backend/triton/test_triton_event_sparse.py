import pytest
import torch

from btorch.backend.triton import post_span_spmm, pre_span_spmm
from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.linear import SparseEventConn
from btorch.models.synapse import ExponentialPSC


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
def test_pre_span_kernel_matches_dense_reference():
    device = "cuda"
    dtype = torch.float32
    dense_weight = _build_dense_weight(device, dtype)
    spike = torch.tensor(
        [[1.0, 0.0, 0.5, -1.0], [0.0, 2.0, -1.0, 0.25]],
        device=device,
        dtype=dtype,
    )
    row_length, ind, weight = _build_pre_span_buffers(device, dtype)

    out = pre_span_spmm(
        spike,
        row_length,
        ind,
        weight,
        size_m=dense_weight.shape[1],
        is_bool_float=False,
    )
    expected = spike @ dense_weight
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_post_span_kernel_matches_dense_reference():
    device = "cuda"
    dtype = torch.float32
    dense_weight = _build_dense_weight(device, dtype)
    spike = torch.tensor(
        [[1.0, 0.0, 0.5, -1.0], [0.0, 2.0, -1.0, 0.25]],
        device=device,
        dtype=dtype,
    )
    row_length, ind, weight = _build_post_span_buffers(device, dtype)

    out = post_span_spmm(
        spike,
        row_length,
        ind,
        weight,
        size_m=dense_weight.shape[1],
        is_bool_float=False,
    )
    expected = spike @ dense_weight
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sparse_event_conn_matches_dense_reference_for_both_modes():
    device = "cuda"
    dtype = torch.float32
    dense_weight = _build_dense_weight(device, dtype)
    spike = torch.tensor(
        [[1.0, 0.0, 0.5, -1.0], [0.0, 2.0, -1.0, 0.25]],
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
