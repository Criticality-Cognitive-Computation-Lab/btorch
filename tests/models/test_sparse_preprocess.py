import itertools

import numpy as np
import pytest
import scipy.sparse
import torch

from btorch._sparse_config import SparseConfigRegistry, TritonSparseConfig
from btorch.models import _sparse_triton
from btorch.models._sparse_preprocess import preprocess_sparse
from btorch.models._sparse_triton import is_triton_available
from btorch.models.functional import prepare_sparse_modules, reset_net
from btorch.models.linear import SparseConn
from btorch.models.rnn import make_rnn


def _internal_coo(matrix: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact internal ``(destination, source)`` COO representation."""

    coo = scipy.sparse.coo_array(matrix).T
    coo.sum_duplicates()
    indices = torch.tensor(np.stack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    return indices, values


def test_triton_config_defaults_to_complete_optimization_set():
    """The Triton template enables all three independent optimizations."""

    registry = SparseConfigRegistry()
    config = registry.backend("triton")

    assert config.reorder is True
    assert config.block is True
    assert config.hash is True


def test_sparse_config_override_is_an_isolated_snapshot():
    """A module override does not mutate the global backend template."""

    registry = SparseConfigRegistry()
    template = registry.backend("triton")
    resolved = registry.resolve("triton", {"hash": False})

    assert template.hash is True
    assert resolved.hash is False
    assert resolved.block is True
    assert resolved.reorder is True


@pytest.mark.parametrize(
    ("reorder", "block", "hash_enabled"),
    itertools.product([False, True], repeat=3),
)
def test_preprocess_preserves_every_edge(reorder, block, hash_enabled):
    """Every optimization combination represents the original matrix exactly."""

    matrix = np.asarray(
        [
            [1.0, 0.0, 2.0, 0.0, 3.0],
            [0.0, 4.0, 0.0, 5.0, 0.0],
            [6.0, 7.0, 0.0, 8.0, 9.0],
            [0.0, 0.0, 10.0, 0.0, 11.0],
        ],
        dtype=np.float32,
    )
    indices, values = _internal_coo(matrix)
    config = TritonSparseConfig(
        reorder=reorder,
        block=block,
        hash=hash_enabled,
        block_sources=2,
        edge_block=4,
        long_fanout_threshold=3,
        hash_capacity=4,
        hash_min_edges=2,
    )

    layout = preprocess_sparse(indices, matrix.shape, config=config)

    assert sorted(layout.edge_permutation.tolist()) == list(range(values.numel()))
    reconstructed = torch.zeros(matrix.shape, dtype=torch.float32)
    reconstructed[
        layout.packed_source.long(), layout.packed_destination.long()
    ] = values[layout.edge_permutation]
    torch.testing.assert_close(reconstructed, torch.from_numpy(matrix))
    assert int(layout.task_indptr[-1]) == values.numel()
    assert torch.all(layout.task_indptr[1:] >= layout.task_indptr[:-1])


def test_triton_layout_rebuilds_after_loading_new_indices():
    """Non-persistent task metadata follows topology loaded from a checkpoint."""

    first = scipy.sparse.coo_array(
        np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    )
    second = scipy.sparse.coo_array(
        np.asarray([[0.0, 3.0], [4.0, 0.0]], dtype=np.float32)
    )
    source = SparseConn(first, enforce_dale=False, sparse_backend="triton")
    target = SparseConn(second, enforce_dale=False, sparse_backend="triton")

    target.load_state_dict(source.state_dict())

    packed_source = target.indices[1][target._triton_edge_permutation]
    packed_destination = target.indices[0][target._triton_edge_permutation]
    torch.testing.assert_close(
        target._triton_packed_source.long(), packed_source
    )
    torch.testing.assert_close(
        target._triton_packed_destination.long(), packed_destination
    )


def test_sparse_prepare_scope_reuses_one_weight_reorder():
    """The multi-step scope retains and then releases one packed weight tensor."""

    matrix = scipy.sparse.eye(4, dtype=np.float32, format="coo")
    model = SparseConn(matrix, enforce_dale=False, sparse_backend="triton")

    assert model._prepared_sparse_weight is None
    with prepare_sparse_modules(model):
        prepared = model._prepared_sparse_weight
        assert model._sparse_prepare_depth == 1
        assert prepared is not None
        expected = model.magnitude[model._triton_edge_permutation]
        torch.testing.assert_close(prepared, expected)
        reset_net(model)
        assert model._prepared_sparse_weight is prepared
        with prepare_sparse_modules(model):
            assert model._sparse_prepare_depth == 2
            assert model._prepared_sparse_weight is prepared
        assert model._sparse_prepare_depth == 1
        assert model._prepared_sparse_weight is prepared
    assert model._prepared_sparse_weight is None
    assert model._sparse_prepare_depth == 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="Triton sparse tests require a CUDA Triton environment.",
)
@pytest.mark.parametrize(
    ("reorder", "block", "hash_enabled"),
    itertools.product([False, True], repeat=3),
)
def test_triton_sparse_forward_and_backward(reorder, block, hash_enabled):
    """All eight Triton paths match dense forward and first-order gradients."""

    torch.manual_seed(42)
    dense_weight = torch.randn(8, 7, dtype=torch.float32)
    dense_weight[torch.rand_like(dense_weight) < 0.45] = 0.0
    sparse_weight = scipy.sparse.coo_array(dense_weight.numpy())
    model = SparseConn(
        sparse_weight,
        enforce_dale=False,
        sparse_backend="triton",
        sparse_config={
            "reorder": reorder,
            "block": block,
            "hash": hash_enabled,
            "block_sources": 4,
            "edge_block": 16,
            "long_fanout_threshold": 8,
            "hash_capacity": 32,
            "hash_max_probe": 8,
            "hash_min_edges": 2,
        },
        device="cuda",
    )
    x = torch.randn(3, 8, device="cuda", requires_grad=True)
    x.data[torch.rand_like(x) < 0.5] = 0.0
    dense_x = x.detach().clone().requires_grad_(True)
    dense_parameter = dense_weight.to("cuda").requires_grad_(True)

    actual = model(x)
    expected = dense_x @ dense_parameter
    actual.sum().backward()
    expected.sum().backward()

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x.grad, dense_x.grad, atol=1e-5, rtol=1e-5)

    sparse_source = model.indices[1]
    sparse_destination = model.indices[0]
    expected_sparse_grad = dense_parameter.grad[
        sparse_source, sparse_destination
    ]
    torch.testing.assert_close(
        model.magnitude.grad, expected_sparse_grad, atol=1e-5, rtol=1e-5
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="Triton sparse tests require a CUDA Triton environment.",
)
def test_triton_sparse_runs_inside_existing_multi_step_wrapper():
    """The existing RNN wrapper prepares and reuses the Triton workspace."""

    class SparseCell(torch.nn.Module):
        def __init__(self, connection):
            super().__init__()
            self.connection = connection

        def forward(self, x):
            return self.connection(x)

    dense_weight = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, -1.0, 3.0], [4.0, 0.0, 0.0]]
    )
    connection = SparseConn(
        scipy.sparse.coo_array(dense_weight.numpy()),
        enforce_dale=False,
        sparse_backend="triton",
        device="cuda",
    )
    rnn = make_rnn(SparseCell, unroll=2)(connection)
    x = torch.tensor(
        [
            [[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]],
            [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        device="cuda",
    )

    actual, _ = rnn.multi_step_forward(x)
    expected = x @ dense_weight.to("cuda")

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert connection._triton_workspace is not None
    assert connection._prepared_sparse_weight is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_triton_available(),
    reason="Triton sparse tests require a CUDA Triton environment.",
)
def test_triton_prepared_fast_path_is_cuda_graph_capturable(monkeypatch):
    """Graph warmup/capture/replay reuse one packed weight and workspace."""

    class SparseCell(torch.nn.Module):
        def __init__(self, connection):
            super().__init__()
            self.connection = connection

        def forward(self, x):
            return self.connection(x)

    dense_weight = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, -1.0, 3.0], [4.0, 0.0, 0.0]]
    )
    connection = SparseConn(
        scipy.sparse.coo_array(dense_weight.numpy()),
        enforce_dale=False,
        sparse_backend="triton",
        device="cuda",
    )
    rnn = make_rnn(SparseCell, unroll=2, cudagraph=True)(connection)
    x = torch.randn(4, 2, 3, device="cuda")
    calls = 0
    original_ensure = _sparse_triton.ensure_triton_workspace

    def counted_ensure(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_ensure(*args, **kwargs)

    monkeypatch.setattr(
        _sparse_triton, "ensure_triton_workspace", counted_ensure
    )
    with torch.inference_mode(), prepare_sparse_modules(rnn, batch_size=2):
        packed_weight = connection._prepared_sparse_weight
        workspace = connection._triton_workspace
        assert calls == 1
        first, _ = rnn.multi_step_forward(x)
        second, _ = rnn.multi_step_forward(x)
        assert connection._prepared_sparse_weight is packed_weight
        assert connection._triton_workspace is workspace
        assert calls == 1

    expected = x @ dense_weight.to("cuda")
    torch.testing.assert_close(first, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(second, expected, atol=1e-5, rtol=1e-5)
