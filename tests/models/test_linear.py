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


def _forward_and_input_grad(
    model: torch.nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.clone().requires_grad_(True)
    output = model(x)
    output.sum().backward()
    return output, x.grad


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
    out_dense, grad_dense = _forward_and_input_grad(dense, x)
    out_sparse, grad_sparse = _forward_and_input_grad(sparse_coo, x)
    out_constrained, grad_constrained = _forward_and_input_grad(constrained, x)

    # Check they're all the same for 1D input.
    torch.testing.assert_close(out_dense, out_sparse, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(out_dense, out_constrained, atol=1e-6, rtol=0.0)

    torch.testing.assert_close(grad_dense, grad_sparse, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(grad_dense, grad_constrained, atol=1e-6, rtol=0.0)

    # Forward pass with batch.
    out_dense_batch, grad_dense_batch = _forward_and_input_grad(dense, x_batch)
    out_sparse_batch, grad_sparse_batch = _forward_and_input_grad(sparse_coo, x_batch)
    out_constrained_batch, grad_constrained_batch = _forward_and_input_grad(
        constrained, x_batch
    )

    # Check they're all the same for batched input.
    torch.testing.assert_close(out_dense_batch, out_sparse_batch, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        out_dense_batch, out_constrained_batch, atol=1e-6, rtol=0.0
    )

    torch.testing.assert_close(grad_dense_batch, grad_sparse_batch, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        grad_dense_batch, grad_constrained_batch, atol=1e-6, rtol=0.0
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
    out_single, grad_single = _forward_and_input_grad(model, x)
    out_batch, grad_batch = _forward_and_input_grad(model, x_batch)
    torch.testing.assert_close(out_batch[0], out_single, atol=1e-6, rtol=0.0)

    torch.testing.assert_close(grad_batch[0], grad_single, atol=1e-6, rtol=0.0)

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

    eager, eager_grad = _forward_and_input_grad(model, x)
    compiled_model = compile_or_skip(model)
    compiled, compiled_grad = _forward_and_input_grad(compiled_model, x)

    torch.testing.assert_close(eager, compiled, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(eager_grad, compiled_grad, atol=1e-6, rtol=0.0)

    eager_batch, eager_grad_batch = _forward_and_input_grad(model, x_batch)
    compiled_batch, compiled_grad_batch = _forward_and_input_grad(
        compiled_model, x_batch
    )
    torch.testing.assert_close(eager_batch, compiled_batch, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        eager_grad_batch, compiled_grad_batch, atol=1e-6, rtol=0.0
    )


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
    out_dense_wide, grad_dense_wide = _forward_and_input_grad(dense_wide, x_wide)
    out_sparse_wide, grad_sparse_wide = _forward_and_input_grad(sparse_wide, x_wide)
    out_constrained_wide, grad_constrained_wide = _forward_and_input_grad(
        constrained_wide, x_wide
    )

    assert out_dense_wide.shape == (2,), f"Expected (2,), got {out_dense_wide.shape}"
    torch.testing.assert_close(out_dense_wide, out_sparse_wide, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        out_dense_wide, out_constrained_wide, atol=1e-6, rtol=0.0
    )

    torch.testing.assert_close(grad_dense_wide, grad_sparse_wide, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        grad_dense_wide, grad_constrained_wide, atol=1e-6, rtol=0.0
    )

    # Batched forward passes for wide matrix
    out_dense_wide_batch, grad_dense_wide_batch = _forward_and_input_grad(
        dense_wide, x_wide_batch
    )
    out_sparse_wide_batch, grad_sparse_wide_batch = _forward_and_input_grad(
        sparse_wide, x_wide_batch
    )
    out_constrained_wide_batch, grad_constrained_wide_batch = _forward_and_input_grad(
        constrained_wide, x_wide_batch
    )

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

    torch.testing.assert_close(
        grad_dense_wide_batch, grad_sparse_wide_batch, atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        grad_dense_wide_batch, grad_constrained_wide_batch, atol=1e-6, rtol=0.0
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
    out_dense_tall, grad_dense_tall = _forward_and_input_grad(dense_tall, x_tall)
    out_sparse_tall, grad_sparse_tall = _forward_and_input_grad(sparse_tall, x_tall)
    out_constrained_tall, grad_constrained_tall = _forward_and_input_grad(
        constrained_tall, x_tall
    )

    assert out_dense_tall.shape == (4,), f"Expected (4,), got {out_dense_tall.shape}"
    torch.testing.assert_close(out_dense_tall, out_sparse_tall, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        out_dense_tall, out_constrained_tall, atol=1e-6, rtol=0.0
    )

    torch.testing.assert_close(grad_dense_tall, grad_sparse_tall, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        grad_dense_tall, grad_constrained_tall, atol=1e-6, rtol=0.0
    )

    # Batched forward passes for tall matrix
    out_dense_tall_batch, grad_dense_tall_batch = _forward_and_input_grad(
        dense_tall, x_tall_batch
    )
    out_sparse_tall_batch, grad_sparse_tall_batch = _forward_and_input_grad(
        sparse_tall, x_tall_batch
    )
    out_constrained_tall_batch, grad_constrained_tall_batch = _forward_and_input_grad(
        constrained_tall, x_tall_batch
    )

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

    torch.testing.assert_close(
        grad_dense_tall_batch, grad_sparse_tall_batch, atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        grad_dense_tall_batch, grad_constrained_tall_batch, atol=1e-6, rtol=0.0
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


@pytest.mark.parametrize("backend", available_sparse_backends())
def test_sparse_conn_get_sparse_matrix(backend: str):
    """get_sparse_matrix returns a usable sparse tensor with gradients."""
    torch.manual_seed(42)

    W = torch.tensor([[1.0, 2.0, 0.0], [0.0, 3.0, -1.0], [2.0, 0.0, 1.0]])
    W_sparse = scipy.sparse.coo_array(W.numpy())

    model = SparseConn(W_sparse, bias=None, enforce_dale=False, sparse_backend=backend)
    sp_mat = model.get_sparse_matrix()

    # Shape should match the original dense orientation.
    assert sp_mat.shape == (3, 3)
    assert sp_mat.layout == torch.sparse_coo

    # Dense reconstruction should match the original weights.
    torch.testing.assert_close(sp_mat.to_dense(), W, atol=1e-6, rtol=0.0)

    # A backward pass through the returned matrix should reach magnitude.
    loss = sp_mat.values().sum()
    loss.backward()
    assert model.magnitude.grad is not None


@pytest.mark.parametrize("backend", available_sparse_backends())
def test_sparse_constrained_conn_get_sparse_matrix(backend: str):
    """get_sparse_matrix reflects constraint magnitudes and preserves
    gradients."""
    torch.manual_seed(42)

    W_sparse = scipy.sparse.coo_array(
        ([1.0, -2.0, -3.0, 1.0], ([0, 1, 0, 1], [0, 0, 1, 1])),
        shape=(2, 2),
    )
    constraint = scipy.sparse.coo_array(
        ([1, 2, 3, 1], ([0, 0, 1, 1], [0, 1, 0, 1])),
        shape=(2, 2),
    )

    model = SparseConstrainedConn(
        W_sparse, constraint, enforce_dale=False, bias=None, sparse_backend=backend
    )
    sp_mat = model.get_sparse_matrix()

    assert sp_mat.shape == (2, 2)

    # Values should equal initial_weight * magnitude[scatter_indices].
    expected = model.initial_weight * model.magnitude[model._constraint_scatter_indices]
    torch.testing.assert_close(sp_mat.values(), expected, atol=1e-6, rtol=0.0)

    # Gradient should flow back to the learnable magnitude.
    loss = sp_mat.values().sum()
    loss.backward()
    assert model.magnitude.grad is not None


@pytest.mark.parametrize("backend", available_sparse_backends())
def test_get_sparse_matrix_non_square(backend: str):
    """get_sparse_matrix works for non-square sparse connections."""
    torch.manual_seed(42)

    # Wide matrix: 4 inputs -> 2 outputs.
    W = torch.tensor([[1.0, 0.0], [2.0, 3.0], [0.0, -1.0], [-1.0, 2.0]])
    W_sparse = scipy.sparse.coo_array(W.numpy())

    model = SparseConn(W_sparse, bias=None, enforce_dale=False, sparse_backend=backend)
    sp_mat = model.get_sparse_matrix()

    assert sp_mat.shape == (4, 2)
    torch.testing.assert_close(sp_mat.to_dense(), W, atol=1e-6, rtol=0.0)

    # Gradient should flow through the returned matrix.
    loss = sp_mat.values().sum()
    loss.backward()
    assert model.magnitude.grad is not None
