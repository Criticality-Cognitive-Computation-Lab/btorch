import pytest
import scipy.sparse
import torch

from btorch.models.constrain import constrain_net
from btorch.models.linear import (
    available_sparse_backends,
    DenseConn,
    SparseConn,
    SparseConstrainedConn,
)
from tests.utils.compile import compile_or_skip


@pytest.mark.parametrize("backend", available_sparse_backends())
def test_equivalent_behavior(backend: str):
    """All connection classes match dense behavior for the same weights."""
    torch.manual_seed(42)

    # Create a small dense weight matrix
    W = torch.tensor([[1.0, 2.0, 0.0], [0.0, 3.0, -1.0], [2.0, 0.0, 1.0]])  # 3x3 matrix

    # Test inputs: single vector and batched vectors.
    x = torch.tensor([1.0, 2.0, 3.0])
    x_batch = torch.stack([x, x + 1.0], dim=0)

    # 1. Dense connection
    dense = DenseConn(3, 3, weight=W, bias=None)

    # 2. Sparse COO connection (convert dense to sparse)
    W_sparse = scipy.sparse.coo_array(W.numpy())

    # 3. Constrained sparse connection (each weight is its own group)
    # Create constraint matrix where each non-zero gets unique group ID
    constraint_data = []
    constraint_rows = []
    constraint_cols = []
    group_id = 1

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i, j] != 0:
                constraint_data.append(group_id)
                constraint_rows.append(i)
                constraint_cols.append(j)
                group_id += 1

    constraint = scipy.sparse.coo_array(
        (constraint_data, (constraint_rows, constraint_cols)), shape=W.shape
    )

    sparse_coo = SparseConn(
        W_sparse, bias=None, enforce_dale=False, sparse_backend=backend
    )
    constrained = SparseConstrainedConn(
        W_sparse, constraint, enforce_dale=False, bias=None, sparse_backend=backend
    )

    # Forward pass without batch.
    out_dense = dense(x)
    out_sparse = sparse_coo(x)
    out_constrained = constrained(x)

    # Check they're all the same for 1D input.
    torch.testing.assert_close(out_dense, out_sparse, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(out_dense, out_constrained, atol=1e-6, rtol=0.0)

    # Forward pass with batch.
    out_dense_batch = dense(x_batch)
    out_sparse_batch = sparse_coo(x_batch)
    out_constrained_batch = constrained(x_batch)

    # Check they're all the same for batched input.
    torch.testing.assert_close(out_dense_batch, out_sparse_batch, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        out_dense_batch, out_constrained_batch, atol=1e-6, rtol=0.0
    )


@pytest.mark.parametrize("backend", available_sparse_backends())
@pytest.mark.parametrize("enable_dale", [False, True])
def test_constraint_optimization(backend: str, enable_dale: bool):
    """Constraints and optional Dale's law."""
    torch.manual_seed(42)

    # Create weight matrix where some weights should be tied together
    W_sparse = scipy.sparse.coo_array(
        ([1.0, -2, -3, 1], ([0, 1, 0, 1], [0, 0, 1, 1])),
        shape=(2, 2),
    )

    # Create constraint matrix: positions (0,0) and (1,1) share group 1
    # positions (0,1) and (1,0) have separate groups
    constraint = scipy.sparse.coo_array(
        ([1, 2, 3, 1], ([0, 0, 1, 1], [0, 1, 0, 1])),  # groups: 1,2,3,1
        shape=(2, 2),
    )

    model = SparseConstrainedConn(
        W_sparse,
        constraint,
        enforce_dale=enable_dale,
        bias=None,
        sparse_backend=backend,
    )
    constrain_net(model)

    # Target output for a simple two-neuron input.
    x = torch.tensor([1.0, 1.0])
    x_batch = x[None, :]
    target = torch.tensor([10.0, 20.0])

    # Initial effective weights set the sign reference for Dale's law.
    magnitudes = model.magnitude[model._constraint_scatter_indices]
    effective_weights = model.initial_weight * magnitudes
    initial_signs = torch.sign(effective_weights)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for _ in range(10):  # Run enough steps to exercise constraints.
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        constrain_net(model)

    # Verify constraints and Dale's law after optimization.
    final_magnitudes = model.magnitude[model._constraint_scatter_indices]
    final_weights = model.initial_weight * final_magnitudes

    # Batch and non-batch forward results should align for the same inputs.
    out_single = model(x)
    out_batch = model(x_batch)
    torch.testing.assert_close(out_batch[0], out_single, atol=1e-6, rtol=0.0)

    # Constraint check: positions (0,0) and (1,1) share group 1.
    pos_00_effective = final_weights[0]
    pos_11_effective = final_weights[3]
    torch.testing.assert_close(pos_00_effective, pos_11_effective, atol=1e-6, rtol=0.0)

    # Dale's law: non-zero weights keep their initial sign.
    if enable_dale:
        final_signs = torch.sign(final_weights)
        for i in range(len(initial_signs)):
            if abs(model.initial_weight[i]) > 1e-8:
                initial_sign = initial_signs[i].item()
                final_sign = final_signs[i].item()
                assert initial_sign == final_sign or abs(final_weights[i]) < 1e-8, (
                    f"Dale's law violated at position {i}: "
                    f"initial_sign={initial_sign}, final_sign={final_sign}, "
                    f"initial_weight={model.initial_weight[i]:.6f}, "
                    f"final_weight={final_weights[i]:.6f}"
                )


@pytest.mark.parametrize("backend", available_sparse_backends())
def test_compile_matches_eager(backend: str):
    """Compiled forward matches eager output when compilation is supported."""
    torch.manual_seed(42)

    W = torch.tensor([[1.0, 2.0, 0.0], [0.0, 3.0, -1.0], [2.0, 0.0, 1.0]])
    W_sparse = scipy.sparse.coo_array(W.numpy())
    model = SparseConn(W_sparse, bias=None, enforce_dale=False, sparse_backend=backend)
    x = torch.tensor([1.0, 2.0, 3.0])
    x_batch = x[None, :]

    eager = model(x)
    compiled_model = compile_or_skip(model)
    compiled = compiled_model(x)

    torch.testing.assert_close(eager, compiled, atol=1e-6, rtol=0.0)

    eager_batch = model(x_batch)
    compiled_batch = compiled_model(x_batch)
    torch.testing.assert_close(eager_batch, compiled_batch, atol=1e-6, rtol=0.0)
