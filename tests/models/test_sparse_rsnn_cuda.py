import numpy as np
import pytest
import scipy.sparse
import torch

from btorch.models.sparse_rsnn_cuda import (
    CyclicSparseRSNNCuda,
    interval_rsnn_reference,
)


def _connection() -> scipy.sparse.csr_array:
    """Build rows that exercise empty, full, boundary, and wrapped slices."""

    dense = np.asarray(
        [
            [0.0, 0.5, 0.0, 0.0, -0.2, 0.0],
            [0.1, 0.0, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.3, 0.4, 0.0],
            [0.7, 0.0, 0.0, 0.0, 0.0, -0.1],
            [0.0, 0.0, 0.6, 0.0, 0.0, 0.8],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return scipy.sparse.csr_array(dense)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("base_start", "active_count", "steps", "stride"),
    [
        (1, 2, 8, 1),
        # Starting at neuron five forces the active interval to wrap at the
        # first step and repeatedly exercises both CSR segments.
        (5, 3, 17, 5),
        # Full activity takes the no-search whole-row path.
        (0, 6, 8, 1),
    ],
)
def test_cyclic_sparse_rsnn_matches_torch_reference(
    base_start: int, active_count: int, steps: int, stride: int
):
    """The fused kernel preserves the reference recurrence final state."""

    connection = _connection()
    kernel = CyclicSparseRSNNCuda(connection, device="cuda:0")

    actual = kernel(
        base_start,
        active_count,
        steps,
        stride=stride,
        synchronize=True,
    )
    expected = interval_rsnn_reference(
        connection,
        base_start=base_start,
        active_count=active_count,
        steps=steps,
        stride=stride,
        device="cuda:0",
    )

    for actual_state, expected_state in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_state, expected_state, atol=1e-6, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_output_reuse_and_clone_contract():
    """Default outputs are reusable, while clone_outputs preserves
    snapshots."""

    kernel = CyclicSparseRSNNCuda(_connection(), device="cuda:0")
    first = kernel(0, 2, 8, synchronize=True)
    snapshot = kernel(0, 2, 8, clone_outputs=True, synchronize=True)
    second = kernel(1, 2, 8, synchronize=True)

    # Reusing output storage is the launch-path optimization used by repeated
    # benchmark samples. Callers that retain history request a clone instead.
    assert first[0].data_ptr() == second[0].data_ptr()
    assert snapshot[0].data_ptr() != second[0].data_ptr()
    assert not torch.equal(snapshot[0], second[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_invalid_interval_arguments_fail_before_launch():
    """Invalid activity descriptions fail without submitting CUDA work."""

    kernel = CyclicSparseRSNNCuda(_connection(), device="cuda:0")

    with pytest.raises(ValueError, match="base_start"):
        kernel(-1, 1, 8)
    with pytest.raises(ValueError, match="active_count"):
        kernel(0, 0, 8)
    with pytest.raises(ValueError, match="steps"):
        kernel(0, 1, 0)
