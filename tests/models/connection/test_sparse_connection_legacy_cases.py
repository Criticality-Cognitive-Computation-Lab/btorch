"""Scenarios of the former ``tests/models/test_linear.py`` sparse tests.

The cases were written for the removed ``SparseConn`` /
``SparseConstrainedConn`` layers and are kept as they were (same matrices,
same assertions), expressed with
:class:`~btorch.models.connection.SparseConnection`:

- plain per-edge weights: ``SparseConnection.from_adjacency(W)``;
- grouped weights:
  ``SparseConnection.from_adjacency(W, Synapse(weight=ConstrainedWeight(group=C)))``
  where ``C`` holds 1-based group ids in the same ``(src, dst)`` layout as
  ``W``.

The legacy tests were parametrised over ``sparse_backend``. Execution backends
are now chosen by the runtime, so that parameter is gone; the optional
``torch_sparse`` kernel is exercised explicitly in
:func:`test_torch_sparse_backend_matches_default`.
"""

import pytest
import scipy.sparse
import torch

from btorch.models.connection import ConstrainedWeight, SparseConnection, Synapse
from btorch.models.constrain import constrain_net
from btorch.models.linear import DenseConn
from btorch.sparse import runtime


def _forward_and_input_grad(
    model: torch.nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Output of ``model(x)`` and the gradient of its sum w.r.t.

    ``x``.
    """
    x = x.clone().requires_grad_(True)
    output = model(x)
    output.sum().backward()
    return output, x.grad


def _one_group_per_edge(W: torch.Tensor) -> scipy.sparse.coo_array:
    """Constraint matrix in which every non-zero of ``W`` is its own group.

    Group ids are 1-based and numbered in row-major order; with unit
    scales the constrained connection is then equivalent to the plain
    one.
    """
    rows, cols = torch.nonzero(W, as_tuple=True)
    group_ids = torch.arange(1, rows.numel() + 1)
    return scipy.sparse.coo_array(
        (group_ids.numpy(), (rows.numpy(), cols.numpy())), shape=W.shape
    )


def _dense_sparse_constrained(W: torch.Tensor):
    """The same ``(in, out)`` weights as dense, sparse and constrained
    layers."""
    W_sparse = scipy.sparse.coo_array(W.numpy())
    dense = DenseConn(W.shape[0], W.shape[1], weight=W, bias=None)
    sparse = SparseConnection.from_adjacency(W_sparse)
    constrained = SparseConnection.from_adjacency(
        W_sparse, Synapse(weight=ConstrainedWeight(group=_one_group_per_edge(W)))
    )
    return dense, sparse, constrained


def _assert_all_match_dense(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Outputs and input gradients of all three layers agree on ``x``."""
    dense, sparse, constrained = _dense_sparse_constrained(W)
    out_dense, grad_dense = _forward_and_input_grad(dense, x)
    for layer in (sparse, constrained):
        out, grad = _forward_and_input_grad(layer, x)
        torch.testing.assert_close(out_dense, out, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(grad_dense, grad, atol=1e-6, rtol=0.0)
    return out_dense


def test_equivalent_behavior():
    """All connection classes match dense behavior for the same weights."""
    torch.manual_seed(42)

    # A small dense weight matrix, (in_features, out_features) = (3, 3).
    W = torch.tensor([[1.0, 2.0, 0.0], [0.0, 3.0, -1.0], [2.0, 0.0, 1.0]])

    # Test inputs: single vector and batched vectors.
    x = torch.tensor([1.0, 2.0, 3.0])
    x_batch = torch.stack([x, x + 1.0], dim=0)

    # Forward pass without batch, then with batch.
    assert _assert_all_match_dense(W, x).shape == (3,)
    assert _assert_all_match_dense(W, x_batch).shape == (2, 3)


@pytest.mark.parametrize("enable_dale", [False, True])
def test_constraint_optimization(enable_dale: bool):
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

    # Dale's law of grouped weights is a flag of the weight module (it keeps
    # every group scale non-negative, so no edge can change sign).
    model = SparseConnection.from_adjacency(
        W_sparse,
        Synapse(weight=ConstrainedWeight(group=constraint, dale=enable_dale)),
    )
    constrain_net(model)

    # Target output for a simple two-neuron input.
    x = torch.tensor([1.0, 1.0])
    x_batch = x[None, :]
    target = torch.tensor([10.0, 20.0])

    # Initial effective weights set the sign reference for Dale's law.
    # ``model.weight()`` is ``base * scale[group]``, one value per edge slot.
    initial_signs = torch.sign(model.weight().detach())

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for _ in range(10):  # Run enough steps to exercise constraints.
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        constrain_net(model)

    # Verify constraints and Dale's law after optimization.
    final_weights = model.weight().detach()
    base = model.weight.base
    # The optimiser really moved the scales away from their initial value.
    assert not torch.allclose(model.weight.scale.detach(), torch.ones(3))

    # Batch and non-batch forward results should align for the same inputs.
    out_single, grad_single = _forward_and_input_grad(model, x)
    out_batch, grad_batch = _forward_and_input_grad(model, x_batch)
    torch.testing.assert_close(out_batch[0], out_single, atol=1e-6, rtol=0.0)

    torch.testing.assert_close(grad_batch[0], grad_single, atol=1e-6, rtol=0.0)

    # Constraint check: positions (0,0) and (1,1) share group 1. Both have a
    # base weight of 1, so their effective weights stay identical. The edge
    # slots are looked up through ``indices`` ([dst, src] per slot).
    dst, src = model.indices
    slot_00 = int(torch.nonzero((src == 0) & (dst == 0)))
    slot_11 = int(torch.nonzero((src == 1) & (dst == 1)))
    assert model.weight.group[slot_00] == model.weight.group[slot_11] == 0
    torch.testing.assert_close(
        final_weights[slot_00], final_weights[slot_11], atol=1e-6, rtol=0.0
    )

    # Dale's law: non-zero weights keep their initial sign.
    if enable_dale:
        assert bool((model.weight.scale >= 0).all())
        final_signs = torch.sign(final_weights)
        for i in range(len(initial_signs)):
            if abs(base[i]) > 1e-8:
                initial_sign = initial_signs[i].item()
                final_sign = final_signs[i].item()
                assert initial_sign == final_sign or abs(final_weights[i]) < 1e-8, (
                    f"Dale's law violated at position {i}: "
                    f"initial_sign={initial_sign}, final_sign={final_sign}, "
                    f"initial_weight={base[i]:.6f}, "
                    f"final_weight={final_weights[i]:.6f}"
                )


def test_compile_matches_eager():
    """Compiled forward matches eager output.

    ``fullgraph=True`` makes this a hard requirement: the connection must
    trace without graph breaks, and without the optional ``torch_sparse``
    package (the legacy layer only compiled cleanly with it).
    """
    torch.manual_seed(42)

    W = torch.tensor([[1.0, 2.0, 0.0], [0.0, 3.0, -1.0], [2.0, 0.0, 1.0]])
    W_sparse = scipy.sparse.coo_array(W.numpy())
    model = SparseConnection.from_adjacency(W_sparse)
    x = torch.tensor([1.0, 2.0, 3.0])
    x_batch = x[None, :]

    eager, eager_grad = _forward_and_input_grad(model, x)
    compiled_model = torch.compile(model, fullgraph=True)
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

    # The weight gradient flows through the compiled module too.
    model.zero_grad()
    compiled_model(x_batch).sum().backward()
    dst, src = model.indices
    torch.testing.assert_close(model.weight.value.grad, x[src], atol=1e-6, rtol=0.0)


def test_non_square_matrix():
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

    _, sparse_wide, constrained_wide = _dense_sparse_constrained(W_wide)
    for layer in (sparse_wide, constrained_wide):
        assert (layer.in_features, layer.out_features) == (4, 2)
    assert _assert_all_match_dense(W_wide, x_wide).shape == (2,)
    assert _assert_all_match_dense(W_wide, x_wide_batch).shape == (2, 2)

    # Case 2: Tall matrix (more outputs than inputs): 2x4 matrix
    # Maps 2 input features to 4 output features (x @ W where W is 2x4)
    W_tall = torch.tensor([[1.0, 0.0, -1.0, 2.0], [2.0, 3.0, 0.0, -1.0]])  # (2, 4)
    x_tall = torch.tensor([1.0, 2.0])
    x_tall_batch = torch.stack([x_tall, x_tall + 1.0], dim=0)

    _, sparse_tall, constrained_tall = _dense_sparse_constrained(W_tall)
    for layer in (sparse_tall, constrained_tall):
        assert (layer.in_features, layer.out_features) == (2, 4)
    assert _assert_all_match_dense(W_tall, x_tall).shape == (4,)
    assert _assert_all_match_dense(W_tall, x_tall_batch).shape == (2, 4)


def test_sparse_conn_get_sparse_matrix():
    """``to_sparse("src_dst")`` (formerly ``get_sparse_matrix``) returns a
    usable sparse matrix with gradients."""
    torch.manual_seed(42)

    W = torch.tensor([[1.0, 2.0, 0.0], [0.0, 3.0, -1.0], [2.0, 0.0, 1.0]])
    W_sparse = scipy.sparse.coo_array(W.numpy())

    model = SparseConnection.from_adjacency(W_sparse)
    sp_mat = model.to_sparse("src_dst")

    # Shape should match the original dense orientation.
    assert tuple(sp_mat.shape) == (3, 3)
    # A PyTorch COO tensor (what the legacy method returned) is one call away.
    assert sp_mat.to_torch("coo").layout == torch.sparse_coo

    # Dense reconstruction should match the original weights.
    torch.testing.assert_close(sp_mat.to_dense(), W, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        sp_mat.to_torch("coo").to_dense(), W, atol=1e-6, rtol=0.0
    )

    # A backward pass through the returned matrix should reach the weights.
    loss = sp_mat.values().sum()
    loss.backward()
    assert model.weight.value.grad is not None


def test_sparse_constrained_conn_get_sparse_matrix():
    """``to_sparse`` reflects the group scales and preserves gradients."""
    torch.manual_seed(42)

    W_sparse = scipy.sparse.coo_array(
        ([1.0, -2.0, -3.0, 1.0], ([0, 1, 0, 1], [0, 0, 1, 1])),
        shape=(2, 2),
    )
    constraint = scipy.sparse.coo_array(
        ([1, 2, 3, 1], ([0, 0, 1, 1], [0, 1, 0, 1])),
        shape=(2, 2),
    )

    model = SparseConnection.from_adjacency(
        W_sparse, Synapse(weight=ConstrainedWeight(group=constraint))
    )
    # Non-trivial scales, so the check below cannot pass with base weights.
    model.weight.set_scale(0, 2.0)
    model.weight.set_scale(2, -0.5)
    sp_mat = model.to_sparse("src_dst")

    assert tuple(sp_mat.shape) == (2, 2)

    # Values should equal base * scale[group], in edge-slot order.
    expected = model.weight.base * model.weight.scale[model.weight.group]
    torch.testing.assert_close(sp_mat.values(), expected, atol=1e-6, rtol=0.0)
    # Groups 1/2/3 sit at (0,0)+(1,1) / (0,1) / (1,0) of the (src, dst) matrix.
    torch.testing.assert_close(
        sp_mat.to_dense(), torch.tensor([[2.0, -3.0], [1.0, 2.0]])
    )

    # Gradient should flow back to the learnable scale.
    loss = sp_mat.values().sum()
    loss.backward()
    assert model.weight.scale.grad is not None


def test_get_sparse_matrix_non_square():
    """``to_sparse`` works for non-square sparse connections."""
    torch.manual_seed(42)

    # Wide matrix: 4 inputs -> 2 outputs.
    W = torch.tensor([[1.0, 0.0], [2.0, 3.0], [0.0, -1.0], [-1.0, 2.0]])
    W_sparse = scipy.sparse.coo_array(W.numpy())

    model = SparseConnection.from_adjacency(W_sparse)
    sp_mat = model.to_sparse("src_dst")

    assert tuple(sp_mat.shape) == (4, 2)
    torch.testing.assert_close(sp_mat.to_dense(), W, atol=1e-6, rtol=0.0)
    # The default orientation is the operator (out_features, in_features).
    assert tuple(model.to_sparse().shape) == (2, 4)

    # Gradient should flow through the returned matrix.
    loss = sp_mat.values().sum()
    loss.backward()
    assert model.weight.value.grad is not None


def test_sparse_conn_state_dict_roundtrip_loads_new_pattern():
    """``load_state_dict`` fully determines the layer's forward behaviour.

    The legacy layer once cached a native sparse tensor at construction
    (indices frozen into a plain attribute, outside the state dict). Loading a
    checkpoint whose connectivity *pattern* differs (but with the same number
    of non-zeros, so the tensor shapes match) updated the ``indices`` buffer
    while the cached tensor kept the old pattern, so the forward silently used
    the wrong wiring. ``SparseConnection`` does cache compressed execution
    layouts, so this is the regression test that they are rebuilt from the
    registered ``indices`` buffer whenever a checkpoint is loaded.
    """
    torch.manual_seed(0)
    # Two 4x3 matrices with 4 non-zeros each but different wiring.
    w_a = torch.tensor(
        [[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0], [4.0, 0, 0]],
    )
    w_b = torch.tensor(
        [[0, 1.0, 0], [0, 0, 2.0], [3.0, 0, 0], [0, 4.0, 0]],
    )
    saved = SparseConnection.from_adjacency(scipy.sparse.coo_array(w_b.numpy()))
    fresh = SparseConnection.from_adjacency(scipy.sparse.coo_array(w_a.numpy()))
    # A device/dtype move before loading re-creates the module's tensors, which
    # is when a cached layout stops aliasing the ``indices`` buffer.
    fresh = fresh.to(torch.float64)
    saved = saved.to(torch.float64)
    x = torch.randn(5, 4, dtype=torch.float64)
    # Sanity: the two layers really differ before loading.
    assert not torch.allclose(saved(x), fresh(x))

    fresh.load_state_dict(saved.state_dict())

    # After loading, the fresh layer reproduces the saved layer exactly and
    # matches the dense reference for the checkpointed matrix.
    torch.testing.assert_close(fresh(x), saved(x))
    torch.testing.assert_close(fresh(x), x @ w_b.double())
    # The introspection view follows the loaded wiring as well.
    torch.testing.assert_close(fresh.to_sparse("src_dst").to_dense(), w_b.double())


@pytest.mark.skipif(
    "torch_sparse" not in runtime.registry.available("csr_matvec", "cpu"),
    reason="optional torch_sparse backend is not installed",
)
def test_torch_sparse_backend_matches_default():
    """The optional ``torch_sparse`` kernel gives the same numbers.

    Backends are a runtime choice, not a constructor argument: the same
    module is run under :func:`btorch.sparse.runtime.use_backend`.
    """
    torch.manual_seed(42)
    W = torch.tensor([[1.0, 2.0, 0.0], [0.0, 3.0, -1.0], [2.0, 0.0, 1.0]])
    dense, sparse, constrained = _dense_sparse_constrained(W)
    x_batch = torch.randn(4, 3)

    out_dense, grad_dense = _forward_and_input_grad(dense, x_batch)
    for layer in (sparse, constrained):
        out_default, grad_default = _forward_and_input_grad(layer, x_batch)
        with runtime.use_backend("torch_sparse"):
            assert runtime.registry.name("csr_matvec", "cpu") == "torch_sparse"
            out_ts, grad_ts = _forward_and_input_grad(layer, x_batch)
        torch.testing.assert_close(out_ts, out_default, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(grad_ts, grad_default, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(out_ts, out_dense, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(grad_ts, grad_dense, atol=1e-6, rtol=0.0)
    # Leaving the context restores the automatic choice.
    assert runtime.registry.name("csr_matvec", "cpu") != "torch_sparse"
