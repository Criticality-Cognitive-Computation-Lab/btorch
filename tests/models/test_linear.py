import pytest
import scipy.sparse
import torch

from btorch.models.constrain import constrain_net
from btorch.models.linear import (
    DenseConn,
    SparseConn,
    SparseConstrainedConn,
    available_sparse_backends,
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


@pytest.mark.parametrize("backend", available_sparse_backends())
def test_non_square_matrix(backend: str):
    """Test that sparse connections work correctly with non-square weight
    matrices."""
    torch.manual_seed(42)

    # Case 1: Wide matrix (more inputs than outputs): 4x2 matrix
    # Maps 4 input features to 2 output features (x @ W where W is 4x2)
    W_wide = torch.tensor(
        [[1.0, 0.0], [2.0, 3.0], [0.0, -1.0], [-1.0, 2.0]]
    )  # (4, 2) = (in_features, out_features)

    x_wide = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x_wide_batch = torch.stack([x_wide, x_wide + 1.0], dim=0)

    # Dense connection
    dense_wide = DenseConn(4, 2, weight=W_wide, bias=None)

    # Sparse COO connection
    W_wide_sparse = scipy.sparse.coo_array(W_wide.numpy())
    sparse_wide = SparseConn(
        W_wide_sparse, bias=None, enforce_dale=False, sparse_backend=backend
    )

    # Constrained sparse connection
    constraint_data = []
    constraint_rows = []
    constraint_cols = []
    group_id = 1
    for i in range(W_wide.shape[0]):
        for j in range(W_wide.shape[1]):
            if W_wide[i, j] != 0:
                constraint_data.append(group_id)
                constraint_rows.append(i)
                constraint_cols.append(j)
                group_id += 1

    constraint_wide = scipy.sparse.coo_array(
        (constraint_data, (constraint_rows, constraint_cols)), shape=W_wide.shape
    )
    constrained_wide = SparseConstrainedConn(
        W_wide_sparse,
        constraint_wide,
        enforce_dale=False,
        bias=None,
        sparse_backend=backend,
    )

    # Test wide matrix forward passes
    out_dense_wide = dense_wide(x_wide)
    out_sparse_wide = sparse_wide(x_wide)
    out_constrained_wide = constrained_wide(x_wide)

    assert out_dense_wide.shape == (2,), f"Expected (2,), got {out_dense_wide.shape}"
    torch.testing.assert_close(out_dense_wide, out_sparse_wide, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        out_dense_wide, out_constrained_wide, atol=1e-6, rtol=0.0
    )

    # Batched forward passes for wide matrix
    out_dense_wide_batch = dense_wide(x_wide_batch)
    out_sparse_wide_batch = sparse_wide(x_wide_batch)
    out_constrained_wide_batch = constrained_wide(x_wide_batch)

    assert out_dense_wide_batch.shape == (
        2,
        2,
    ), f"Expected (2, 2), got {out_dense_wide_batch.shape}"
    torch.testing.assert_close(
        out_dense_wide_batch, out_sparse_wide_batch, atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        out_dense_wide_batch, out_constrained_wide_batch, atol=1e-6, rtol=0.0
    )

    # Case 2: Tall matrix (more outputs than inputs): 2x4 matrix
    # Maps 2 input features to 4 output features (x @ W where W is 2x4)
    W_tall = torch.tensor([[1.0, 0.0, -1.0, 2.0], [2.0, 3.0, 0.0, -1.0]])  # (2, 4)

    x_tall = torch.tensor([1.0, 2.0])
    x_tall_batch = torch.stack([x_tall, x_tall + 1.0], dim=0)

    # Dense connection
    dense_tall = DenseConn(2, 4, weight=W_tall, bias=None)

    # Sparse COO connection
    W_tall_sparse = scipy.sparse.coo_array(W_tall.numpy())
    sparse_tall = SparseConn(
        W_tall_sparse, bias=None, enforce_dale=False, sparse_backend=backend
    )

    # Constrained sparse connection
    constraint_data = []
    constraint_rows = []
    constraint_cols = []
    group_id = 1
    for i in range(W_tall.shape[0]):
        for j in range(W_tall.shape[1]):
            if W_tall[i, j] != 0:
                constraint_data.append(group_id)
                constraint_rows.append(i)
                constraint_cols.append(j)
                group_id += 1

    constraint_tall = scipy.sparse.coo_array(
        (constraint_data, (constraint_rows, constraint_cols)), shape=W_tall.shape
    )
    constrained_tall = SparseConstrainedConn(
        W_tall_sparse,
        constraint_tall,
        enforce_dale=False,
        bias=None,
        sparse_backend=backend,
    )

    # Test tall matrix forward passes
    out_dense_tall = dense_tall(x_tall)
    out_sparse_tall = sparse_tall(x_tall)
    out_constrained_tall = constrained_tall(x_tall)

    assert out_dense_tall.shape == (4,), f"Expected (4,), got {out_dense_tall.shape}"
    torch.testing.assert_close(out_dense_tall, out_sparse_tall, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        out_dense_tall, out_constrained_tall, atol=1e-6, rtol=0.0
    )

    # Batched forward passes for tall matrix
    out_dense_tall_batch = dense_tall(x_tall_batch)
    out_sparse_tall_batch = sparse_tall(x_tall_batch)
    out_constrained_tall_batch = constrained_tall(x_tall_batch)

    assert out_dense_tall_batch.shape == (
        2,
        4,
    ), f"Expected (2, 4), got {out_dense_tall_batch.shape}"
    torch.testing.assert_close(
        out_dense_tall_batch, out_sparse_tall_batch, atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        out_dense_tall_batch, out_constrained_tall_batch, atol=1e-6, rtol=0.0
    )


def test_dense_conn_constraints():
    torch.manual_seed(42)
    in_features = 10
    out_features = 8

    # 1. Test Masking (float)
    density = 0.5
    model = DenseConn(in_features, out_features, mask=density, enforce_dale=False)
    assert model.mask is not None
    assert model.mask.shape == (out_features, in_features)

    constrain_net(model)
    # Check that mask is applied
    assert torch.all(model.weight.data[model.mask == 0] == 0)

    # 2. Test Dale's Law
    model_dale = DenseConn(in_features, out_features, enforce_dale=True)
    initial_weights = model_dale.weight.data.clone()

    # Flip some signs manually
    model_dale.weight.data *= -1
    constrain_net(model_dale)

    # Weights that were flipped should be 0 now (because
    # relu((W * initial_sign)) where W * initial_sign < 0)
    # Actually, if initial_sign was 1, and we flipped to -1, then (-1 * 1).relu() = 0.
    # If initial_sign was -1, and we flipped to 1, then (1 * -1).relu() = 0.
    assert torch.all(model_dale.weight.data == 0)  # All signs were flipped

    # If we set them to half the initial value (correct sign)
    model_dale.weight.data = initial_weights * 0.5
    constrain_net(model_dale)
    torch.testing.assert_close(model_dale.weight.data, initial_weights * 0.5)
