import pytest
import torch

from btorch.models.linear import SparseLinear
from btorch.sparse import CSR, BinaryEvents, compact_events, event_sparse_mm


_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@_cuda
def test_compact_binary_events_matches_nonzero():
    events = BinaryEvents(
        torch.tensor(
            [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]],
            device="cuda",
        )
    )
    compact = compact_events(events)
    torch.testing.assert_close(
        compact.count,
        torch.tensor([2, 1], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(
        compact.indices[0, :2],
        torch.tensor([1, 3], device="cuda"),
    )
    torch.testing.assert_close(
        compact.indices[1, :1],
        torch.tensor([0], device="cuda"),
    )


@_cuda
@pytest.mark.parametrize("schedule", ["pre_span", "post_span"])
def test_irregular_padded_csr_uses_packed_values(schedule):
    matrix = CSR.from_edges(
        row=torch.tensor([0, 1, 1, 2], device="cuda"),
        col=torch.tensor([1, 0, 2, 1], device="cuda"),
        data=torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda"),
        shape=(3, 3),
    )
    layout = matrix.padded_csr_layout()
    assert matrix.data.numel() == 4
    assert layout.indices.numel() == 6

    events = BinaryEvents(torch.tensor([[1.0, 1.0, 0.0]], device="cuda"))
    with torch.no_grad():
        actual = event_sparse_mm(matrix, events, schedule=schedule)
    torch.testing.assert_close(
        actual,
        torch.tensor([[2.0, 1.0, 3.0]], device="cuda"),
    )


@_cuda
def test_grad_enabled_event_input_uses_numeric_fallback():
    matrix = CSR.from_edges(
        row=torch.tensor([0, 1], device="cuda"),
        col=torch.tensor([1, 0], device="cuda"),
        data=torch.tensor([2.0, 3.0], device="cuda"),
        shape=(2, 2),
    )
    linear = SparseLinear(matrix)
    events = BinaryEvents(torch.tensor([[1.0, 1.0]], device="cuda"))
    out = event_sparse_mm(linear.sparse_weight, events)
    out.sum().backward()
    values_param = dict(linear.sparse_weight.named_parameters())["data"]
    assert values_param.grad is not None
