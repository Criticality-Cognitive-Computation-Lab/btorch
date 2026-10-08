"""Model-level contract of :class:`SparseConnection`.

These tests were first written against the removed ``SparseConn`` /
``SparseConstrainedConn`` layers to pin down the behaviour a model relies on
(orientation, duplicate handling, batching, autograd, Dale's law,
``state_dict`` layout). They now assert the same contract on
:class:`~btorch.models.connection.SparseConnection`, plus the checkpoint
behaviour the legacy layers got wrong: everything that determines the forward
pass (edges, Dale signs, constraint base weights and groups) lives in the
``state_dict``, so loading a checkpoint into a layer built from a *different*
matrix reproduces the saved layer exactly.

Name mapping used throughout (legacy -> new):

- ``SparseConn(conn, enforce_dale=E)`` ->
  ``SparseConnection.from_adjacency(conn, Synapse(dale=E))``
- ``layer.magnitude`` / ``layer.initial_sign`` ->
  ``conn.weight.value`` / ``conn.weight.sign``
- ``SparseConstrainedConn(conn, constraint, enforce_dale=E)`` ->
  ``SparseConnection.from_adjacency(conn,
  Synapse(weight=ConstrainedWeight(group=constraint, dale=E)))``
- constrained ``magnitude`` / ``initial_weight`` / scatter indices ->
  ``conn.weight.scale`` / ``conn.weight.base`` / ``conn.weight.group``
"""

import numpy as np
import pytest
import scipy.sparse
import torch

from btorch.models.connection import ConstrainedWeight, SparseConnection, Synapse
from btorch.models.constrain import constrain_net


def _random_conn(n_src=7, n_dst=5, density=0.4, seed=0):
    """Random ``(n_src, n_dst)`` connection matrix and its dense form."""
    mat = scipy.sparse.random_array(
        (n_src, n_dst), density=density, random_state=seed, dtype=np.float32
    )
    # Signed weights, so Dale's-law tests see both signs.
    mat.data = mat.data - 0.5
    return mat.tocoo(), torch.tensor(mat.toarray())


def _dense_to_coo(dense: torch.Tensor) -> scipy.sparse.coo_array:
    """Dense ``(n_src, n_dst)`` tensor -> SciPy COO of its non-zeros."""
    return scipy.sparse.coo_array(dense.numpy())


# --------------------------------------------------------------- plain edges
def test_orientation_is_source_rows_destination_columns():
    """``from_adjacency`` takes ``(n_src, n_dst)`` and computes ``x @ conn``.

    The matrix is deliberately non-square and asymmetric: a transposed or
    doubly transposed implementation cannot pass.
    """
    # Edge 0 -> 2 with weight 3, edge 1 -> 0 with weight 5.
    conn = scipy.sparse.coo_array(([3.0, 5.0], ([0, 1], [2, 0])), shape=(2, 3))
    layer = SparseConnection.from_adjacency(conn)
    assert (layer.in_features, layer.out_features) == (2, 3)

    # Only source 0 is active: the current lands on destination 2.
    out = layer(torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(out, torch.tensor([0.0, 0.0, 3.0]))
    # Only source 1 is active: the current lands on destination 0.
    out = layer(torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(out, torch.tensor([5.0, 0.0, 0.0]))

    # ``orientation="post_pre"`` (and the bare constructor) take the operator
    # ``(n_post, n_pre)`` instead; the same edges need the transposed matrix.
    for other in (
        SparseConnection.from_adjacency(conn.T, orientation="post_pre"),
        SparseConnection(conn.T),
    ):
        assert (other.in_features, other.out_features) == (2, 3)
        torch.testing.assert_close(
            other(torch.tensor([1.0, 0.0])), torch.tensor([0.0, 0.0, 3.0])
        )


def test_duplicate_entries_are_summed_into_one_parameter():
    """Duplicate ``(src, dst)`` entries collapse to one trainable weight."""
    conn = scipy.sparse.coo_array(
        ([1.0, 2.0, 5.0], ([0, 0, 1], [1, 1, 0])), shape=(2, 3)
    )
    layer = SparseConnection.from_adjacency(conn)
    assert layer.weight.value.numel() == 2
    assert layer.nnz == 2
    x = torch.tensor([1.0, 10.0])
    torch.testing.assert_close(layer(x), torch.tensor([50.0, 3.0, 0.0]))


def test_stored_indices_are_destination_major_and_sorted():
    """``indices`` holds ``[dst, src]`` sorted by destination, then source.

    Checkpoints store this buffer, so its layout is part of the
    contract.
    """
    conn, dense = _random_conn()
    layer = SparseConnection.from_adjacency(conn)
    dst, src = layer.indices
    assert int(dst.max()) < conn.shape[1] and int(src.max()) < conn.shape[0]
    key = dst * conn.shape[0] + src
    assert bool((key[1:] > key[:-1]).all())
    # The weight of slot ``k`` belongs to edge ``src[k] -> dst[k]``.
    torch.testing.assert_close(layer.weight.value.detach(), dense[src, dst])
    # Without Dale's law the checkpoint is exactly edges + weights, plus the
    # ``layout`` buffer ([n_post, n_pre, n_receptor, n_delay]) that says which
    # populations the edge ids refer to.
    assert set(layer.state_dict()) == {"indices", "layout", "weight.value"}
    assert layer.layout.tolist() == [conn.shape[1], conn.shape[0], 1, 1]
    # With Dale's law the reference sign is persistent as well (the legacy
    # layer kept it out of the checkpoint, which broke cross-matrix loading).
    dale = SparseConnection.from_adjacency(conn, Synapse(dale=True))
    assert set(dale.state_dict()) == {
        "indices",
        "layout",
        "weight.value",
        "weight.sign",
    }


@pytest.mark.parametrize("lead", [(), (4,), (3, 4), (2, 3, 4)])
def test_leading_dimensions_are_preserved(lead):
    """Any number of leading (time / batch) dimensions is supported."""
    conn, dense = _random_conn()
    layer = SparseConnection.from_adjacency(conn)
    x = torch.randn(*lead, conn.shape[0])
    out = layer(x)
    assert out.shape == (*lead, conn.shape[1])
    torch.testing.assert_close(out, x @ dense, atol=1e-5, rtol=1e-5)


def test_bias_is_added_per_destination():
    conn, dense = _random_conn()
    bias = torch.arange(conn.shape[1], dtype=torch.float32)
    layer = SparseConnection.from_adjacency(conn, bias=bias)
    # The bias becomes a trainable parameter and part of the checkpoint.
    assert isinstance(layer.bias, torch.nn.Parameter)
    assert "bias" in layer.state_dict()
    x = torch.randn(3, conn.shape[0])
    torch.testing.assert_close(layer(x), x @ dense + bias, atol=1e-5, rtol=1e-5)


def test_gradients_match_dense_reference():
    """Gradients w.r.t.

    the input and every edge weight equal the dense ones.
    """
    conn, dense = _random_conn()
    layer = SparseConnection.from_adjacency(conn)
    x = torch.randn(3, conn.shape[0], requires_grad=True)
    target = torch.randn(3, conn.shape[1])

    ((layer(x) - target) ** 2).sum().backward()
    grad_x, grad_w = x.grad.clone(), layer.weight.value.grad.clone()

    dense = dense.clone().requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    ((x_ref @ dense - target) ** 2).sum().backward()

    torch.testing.assert_close(grad_x, x_ref.grad, atol=1e-5, rtol=1e-5)
    # ``indices`` is [dst, src]; the dense gradient is indexed [src, dst].
    dst, src = layer.indices
    torch.testing.assert_close(grad_w, dense.grad[src, dst], atol=1e-5, rtol=1e-5)


def test_to_sparse_exposes_effective_matrix_with_gradients():
    """``to_sparse("pre_post")`` is the connectome-oriented weight matrix.

    Replacement of the legacy ``get_sparse_matrix()``: the values are the
    effective weights, so a loss on them reaches the weight parameter.
    """
    conn, dense = _random_conn()
    layer = SparseConnection.from_adjacency(conn)
    mat = layer.to_sparse("pre_post")
    assert tuple(mat.shape) == conn.shape
    torch.testing.assert_close(mat.to_dense(), dense)
    # Without an argument the matrix comes back in the layout it was given
    # in (``from_adjacency`` -> ``pre_post``); the linear-algebra operator
    # ``(dst, src)`` is its transpose and has to be asked for by name.
    assert layer.orientation == "pre_post"
    torch.testing.assert_close(layer.to_sparse().to_dense(), dense)
    torch.testing.assert_close(layer.to_sparse("post_pre").to_dense(), dense.T)

    mat.values().sum().backward()
    torch.testing.assert_close(
        layer.weight.value.grad, torch.ones_like(layer.weight.value)
    )


def test_dale_is_opt_in():
    """``Synapse.dale`` defaults to False (the legacy default was True).

    Without it no sign is stored and ``constrain_net`` leaves sign flips
    alone, so migrated code has to ask for Dale's law explicitly.
    """
    conn = scipy.sparse.coo_array(([1.0, -2.0], ([0, 1], [0, 0])), shape=(2, 1))
    layer = SparseConnection.from_adjacency(conn)
    assert "weight.sign" not in layer.state_dict()
    with torch.no_grad():
        layer.weight.value.mul_(-1.0)
    constrain_net(layer)
    torch.testing.assert_close(layer.weight.value.detach(), torch.tensor([-1.0, 2.0]))


def test_dale_constraint_clamps_sign_flips_to_zero():
    """With ``dale=True`` a weight may shrink to zero but never flip sign."""
    conn = scipy.sparse.coo_array(
        ([1.0, -2.0, 3.0], ([0, 1, 2], [0, 0, 1])), shape=(3, 2)
    )
    layer = SparseConnection.from_adjacency(conn, Synapse(dale=True))
    before = layer.weight.value.detach().clone()
    torch.testing.assert_close(layer.weight.sign, torch.sign(before))

    # Push every weight across zero, as a large optimiser step would.
    # ``constrain_net`` reaches the weight module through ``layer.modules()``.
    with torch.no_grad():
        layer.weight.value.mul_(-1.0)
    constrain_net(layer)
    assert torch.all(layer.weight.value == 0)

    # A same-sign update is left untouched.
    with torch.no_grad():
        layer.weight.value.copy_(before * 0.5)
    constrain_net(layer)
    torch.testing.assert_close(layer.weight.value.detach(), before * 0.5)


# ------------------------------------------------------- constrained weights
def _constrained_layer(dale=False):
    """Four edges in three groups; edges (0,0) and (1,1) share group 1.

    The constraint matrix has the same ``(src, dst)`` orientation as the
    connection matrix and holds 1-based group ids (0 means "no entry").
    """
    conn = scipy.sparse.coo_array(
        ([1.0, -2.0, -3.0, 4.0], ([0, 1, 0, 1], [0, 0, 1, 1])), shape=(2, 2)
    )
    constraint = scipy.sparse.coo_array(
        ([1, 2, 3, 1], ([0, 1, 0, 1], [0, 0, 1, 1])), shape=(2, 2)
    )
    # Dale's law of a constrained weight is a property of the weight module:
    # ``Synapse.dale`` is ignored when ``weight`` already is a ``Weight``.
    weight = ConstrainedWeight(group=constraint, dale=dale)
    return SparseConnection.from_adjacency(conn, Synapse(weight=weight)), conn


def test_constrained_effective_weight_is_base_times_group_scale():
    """``w[e] = base[e] * scale[group[e]]``; scales start at 1."""
    layer, conn = _constrained_layer()
    assert layer.weight.scale.shape == (3,)
    torch.testing.assert_close(layer.weight.scale.detach(), torch.ones(3))
    # Base weights and groups are always persistent (the legacy layer only
    # saved ``magnitude`` unless ``persist_initial_weight`` was set).
    assert set(layer.state_dict()) == {
        "indices",
        "layout",
        "weight.scale",
        "weight.group",
        "weight.base",
    }
    # Only the per-group scale is trainable.
    assert [n for n, _ in layer.named_parameters()] == ["weight.scale"]

    # ``group`` and ``base`` are aligned with the edge slots of ``indices``.
    dense = torch.tensor(conn.toarray(), dtype=torch.float32)
    dst, src = layer.indices
    torch.testing.assert_close(layer.weight.base, dense[src, dst])
    expected_group = torch.tensor([[0, 2], [1, 0]])  # 0-based, [src, dst]
    assert torch.equal(layer.weight.group, expected_group[src, dst])

    x = torch.tensor([[1.0, 10.0]])
    torch.testing.assert_close(layer(x), x @ dense)

    # Doubling group 1 (0-based id 0) doubles edges (0,0) and (1,1) only.
    layer.weight.set_scale(0, 2.0)
    dense[0, 0] *= 2
    dense[1, 1] *= 2
    torch.testing.assert_close(layer(x), x @ dense)
    torch.testing.assert_close(layer.to_sparse("pre_post").to_dense(), dense)


def test_constrained_group_gradient_matches_dense_reference():
    """The gradient of a group scale is the sum over its edges of ``dL/dw[e] *
    base[e]``."""
    layer, conn = _constrained_layer()
    x = torch.randn(5, 2)
    target = torch.randn(5, 2)
    ((layer(x) - target) ** 2).sum().backward()

    dense = torch.tensor(conn.toarray(), dtype=torch.float32, requires_grad=True)
    ((x @ dense - target) ** 2).sum().backward()
    g = dense.grad * dense.detach()  # dL/dw * base, per edge
    expected = torch.stack([g[0, 0] + g[1, 1], g[1, 0], g[0, 1]])
    torch.testing.assert_close(layer.weight.scale.grad, expected, atol=1e-5, rtol=1e-5)


def test_constrained_dale_keeps_group_scales_non_negative():
    layer, _ = _constrained_layer(dale=True)
    with torch.no_grad():
        layer.weight.scale.copy_(torch.tensor([-1.0, 0.5, 2.0]))
    constrain_net(layer)
    torch.testing.assert_close(
        layer.weight.scale.detach(), torch.tensor([0.0, 0.5, 2.0])
    )

    # Without Dale's law a negative scale (a sign flip of the whole group) is
    # allowed to stay.
    free, _ = _constrained_layer(dale=False)
    with torch.no_grad():
        free.weight.scale.copy_(torch.tensor([-1.0, 0.5, 2.0]))
    constrain_net(free)
    torch.testing.assert_close(
        free.weight.scale.detach(), torch.tensor([-1.0, 0.5, 2.0])
    )


def test_constrained_missing_group_raises():
    """Every connection needs a constraint group."""
    conn = scipy.sparse.coo_array(([1.0, 2.0], ([0, 1], [0, 1])), shape=(2, 2))
    constraint = scipy.sparse.coo_array(([1], ([0], [0])), shape=(2, 2))
    with pytest.raises(ValueError, match="Constraint missing"):
        SparseConnection.from_adjacency(
            conn, Synapse(weight=ConstrainedWeight(group=constraint))
        )


def test_constrained_group_helpers():
    """``group_info`` / ``set_scale`` / ``weights_by_group`` inspect groups."""
    layer, _ = _constrained_layer()
    layer.weight.set_scale(1, 3.0)

    info = layer.weight.group_info(include_weights=True)
    assert list(info.columns) == [
        "group_id",
        "num_connections",
        "scale",
        "mean_base_weight",
        "std_base_weight",
    ]
    assert info["num_connections"].tolist() == [2, 1, 1]
    assert info["scale"].tolist() == [1.0, 3.0, 1.0]
    # Group 0 holds base weights 1 and 4; groups 1 / 2 hold -2 / -3.
    np.testing.assert_allclose(info["mean_base_weight"], [2.5, -2.0, -3.0])
    np.testing.assert_allclose(info["std_base_weight"], [1.5, 0.0, 0.0])

    by_group = layer.weight.weights_by_group()
    assert set(by_group) == {0, 1, 2}
    torch.testing.assert_close(by_group[0].detach(), torch.tensor([1.0, 4.0]))
    # Effective weight = base (-2) * scale (3).
    torch.testing.assert_close(by_group[1].detach(), torch.tensor([-6.0]))


# ------------------------------------------------------------- checkpointing
def test_state_dict_roundtrip_same_topology():
    """Saving and loading into a layer built from the same matrix is exact."""
    conn, _ = _random_conn()
    a = SparseConnection.from_adjacency(conn, Synapse(dale=True))
    with torch.no_grad():
        a.weight.value.mul_(0.3)
    b = SparseConnection.from_adjacency(conn, Synapse(dale=True))
    b.load_state_dict(a.state_dict())
    x = torch.randn(4, conn.shape[0])
    torch.testing.assert_close(a(x), b(x))
    # Dale's law still holds the loaded weights in place.
    constrain_net(b)
    torch.testing.assert_close(a(x), b(x))


def test_checkpoint_into_different_matrix_restores_edges_and_dale_signs():
    """A checkpoint fully determines the layer, whatever it was built from.

    The legacy layer kept the Dale reference sign outside the ``state_dict``.
    Loading a checkpoint into a layer built from another matrix (same number
    of edges, so the tensor shapes agree) then kept the *constructor's* signs:
    the next ``constrain_net`` zeroed every loaded weight whose sign differed.
    Now the edges and the signs both come from the checkpoint.
    """
    # Same 4 x 3 shape and 4 edges each, but different wiring and (for the
    # slots that will be compared) opposite signs.
    w_saved = torch.tensor(
        [[0, 1.0, 0], [0, 0, -2.0], [3.0, 0, 0], [0, -4.0, 0]],
    )
    w_fresh = torch.tensor(
        [[-1.0, 0, 0], [0, 2.0, 0], [0, 0, -3.0], [4.0, 0, 0]],
    )
    saved = SparseConnection.from_adjacency(_dense_to_coo(w_saved), Synapse(dale=True))
    fresh = SparseConnection.from_adjacency(_dense_to_coo(w_fresh), Synapse(dale=True))
    with torch.no_grad():
        saved.weight.value.mul_(0.5)  # "trained" weights, same signs
    x = torch.randn(5, 4)
    # Sanity: the two layers really differ before loading.
    assert not torch.equal(saved.indices, fresh.indices)
    assert not torch.equal(saved.weight.sign, fresh.weight.sign)
    assert not torch.allclose(saved(x), fresh(x))

    fresh.load_state_dict(saved.state_dict())

    # Edges, weights and reference signs are the checkpointed ones ...
    assert torch.equal(fresh.indices, saved.indices)
    assert torch.equal(fresh.weight.sign, saved.weight.sign)
    torch.testing.assert_close(fresh(x), saved(x))
    torch.testing.assert_close(fresh(x), x @ (w_saved * 0.5))
    # ... so Dale's law leaves the loaded weights alone (the legacy layer
    # zeroed the sign-mismatched ones here) ...
    constrain_net(fresh)
    torch.testing.assert_close(fresh(x), x @ (w_saved * 0.5))
    # ... and still clamps a later sign flip relative to the *loaded* signs.
    with torch.no_grad():
        fresh.weight.value.mul_(-1.0)
    constrain_net(fresh)
    assert torch.all(fresh.weight.value == 0)
    torch.testing.assert_close(fresh(x), torch.zeros(5, 3))

    # Gradients after loading follow the loaded wiring as well.
    saved.zero_grad()
    saved(x).sum().backward()
    dst, src = saved.indices
    torch.testing.assert_close(saved.weight.value.grad, x.sum(0)[src])


def test_checkpoint_into_different_matrix_restores_constraint_groups():
    """Constrained equivalent: base weights and groups come from the
    checkpoint.

    The legacy layer saved only ``magnitude`` by default, so loading into a
    layer built from other base weights / another grouping silently combined
    the saved scales with the wrong base and groups.
    """
    edges = ([0, 1, 0, 1], [0, 0, 1, 1])  # (src, dst) of the four edges

    def build(base, groups, edges=edges):
        conn = scipy.sparse.coo_array((base, edges), shape=(2, 2))
        constraint = scipy.sparse.coo_array((groups, edges), shape=(2, 2))
        return SparseConnection.from_adjacency(
            conn, Synapse(weight=ConstrainedWeight(group=constraint, dale=True))
        )

    # Saved: groups {(0,0),(1,1)}, {(1,0)}, {(0,1)}.
    saved = build([1.0, -2.0, -3.0, 4.0], [1, 2, 3, 1])
    # Fresh: other base weights and another grouping with the same number of
    # groups (so the ``scale`` shapes agree): {(0,0)}, {(1,0),(0,1)}, {(1,1)}.
    fresh = build([5.0, 6.0, 7.0, 8.0], [1, 2, 2, 3])
    with torch.no_grad():
        saved.weight.scale.copy_(torch.tensor([2.0, 0.5, 3.0]))
    x = torch.randn(5, 2)
    assert not torch.equal(saved.weight.group, fresh.weight.group)
    assert not torch.allclose(saved(x), fresh(x))

    fresh.load_state_dict(saved.state_dict())

    assert torch.equal(fresh.indices, saved.indices)
    assert torch.equal(fresh.weight.group, saved.weight.group)
    torch.testing.assert_close(fresh.weight.base, saved.weight.base)
    torch.testing.assert_close(fresh.weight.scale, saved.weight.scale)
    # Effective matrix: base * scale of the *saved* grouping.
    expected = torch.tensor([[1.0 * 2.0, -3.0 * 3.0], [-2.0 * 0.5, 4.0 * 2.0]])
    torch.testing.assert_close(fresh(x), saved(x))
    torch.testing.assert_close(fresh(x), x @ expected)
    # The scale gradient is accumulated over the loaded groups.
    fresh(x).sum().backward()
    saved(x).sum().backward()
    torch.testing.assert_close(fresh.weight.scale.grad, saved.weight.scale.grad)

    # A checkpoint with different *wiring* (not only other values) is
    # restored too: move one edge of the saved layer.
    rewired = build([1.0, -2.0, 4.0], [1, 2, 1], edges=([0, 1, 1], [0, 0, 1]))
    target = build([9.0, 9.0, 9.0], [2, 1, 2], edges=([0, 0, 1], [0, 1, 1]))
    target.load_state_dict(rewired.state_dict())
    torch.testing.assert_close(target(x), rewired(x))
    torch.testing.assert_close(target(x), x @ torch.tensor([[1.0, 0.0], [-2.0, 4.0]]))


# ------------------------------------------------------------ dtype / device
def test_scipy_float64_defaults_to_float32_weights():
    """SciPy matrices are float64 by NumPy convention, not by choice: the
    weights use the PyTorch default dtype unless ``dtype`` is given."""
    conn, dense = _random_conn()
    conn64 = conn.astype(np.float64)
    assert SparseConnection.from_adjacency(conn64).weight.value.dtype == torch.float32
    explicit = SparseConnection.from_adjacency(conn64, dtype=torch.float64)
    assert explicit.weight.value.dtype == torch.float64
    # Constrained base weights follow the same rule.
    constraint = scipy.sparse.coo_array(
        (np.ones(conn.nnz, dtype=np.int64), (conn.row, conn.col)), shape=conn.shape
    )
    constrained = SparseConnection.from_adjacency(
        conn64, Synapse(weight=ConstrainedWeight(group=constraint))
    )
    assert constrained.weight.base.dtype == torch.float32
    assert constrained.weight.scale.dtype == torch.float32


def test_dtype_and_device_follow_module():
    conn, dense = _random_conn()
    layer = SparseConnection.from_adjacency(conn, Synapse(dale=True)).to(torch.float64)
    # Floating-point state follows ``.to``; the integer edge list does not.
    assert layer.weight.value.dtype == torch.float64
    assert layer.weight.sign.dtype == torch.float64
    assert layer.indices.dtype == torch.long
    x = torch.randn(3, conn.shape[0], dtype=torch.float64)
    out = layer(x)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, x @ dense.double())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("how", ["constructor", "to"])
def test_cuda_placement(how):
    """``device=`` at construction and a later ``.to(device)`` are
    equivalent: all state lives on the device and the forward runs there."""
    conn, dense = _random_conn()
    bias = torch.arange(conn.shape[1], dtype=torch.float32)
    if how == "constructor":
        layer = SparseConnection.from_adjacency(
            conn, Synapse(dale=True), bias=bias, device="cuda"
        )
    else:
        layer = SparseConnection.from_adjacency(conn, Synapse(dale=True), bias=bias).to(
            "cuda"
        )
    for name, tensor in layer.state_dict().items():
        assert tensor.device.type == "cuda", name

    x = torch.randn(3, conn.shape[0], device="cuda", requires_grad=True)
    out = layer(x)
    assert out.device.type == "cuda"
    torch.testing.assert_close(
        out.cpu(), x.detach().cpu() @ dense + bias, atol=1e-5, rtol=1e-5
    )
    out.sum().backward()
    assert layer.weight.value.grad is not None and x.grad is not None

    # Moving back to the CPU re-plans execution for that device.
    cpu = layer.cpu()
    torch.testing.assert_close(
        cpu(x.detach().cpu()), x.detach().cpu() @ dense + bias, atol=1e-5, rtol=1e-5
    )
