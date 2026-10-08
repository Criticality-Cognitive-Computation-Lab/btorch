"""Behaviour of the legacy sparse connection layers that must not change.

These tests pin down the public contract of ``SparseConn`` and
``SparseConstrainedConn`` (orientation, duplicate handling, batching, autograd,
Dale's law, ``state_dict`` layout) so the layers can be re-implemented on top
of the new sparse core without silently changing model behaviour.
"""

import numpy as np
import pytest
import scipy.sparse
import torch

from btorch.models.constrain import constrain_net
from btorch.models.linear import SparseConn, SparseConstrainedConn


def _random_conn(n_src=7, n_dst=5, density=0.4, seed=0):
    """Random ``(n_src, n_dst)`` connection matrix and its dense form."""
    mat = scipy.sparse.random_array(
        (n_src, n_dst), density=density, random_state=seed, dtype=np.float32
    )
    # Signed weights, so Dale's-law tests see both signs.
    mat.data = mat.data - 0.5
    return mat.tocoo(), torch.tensor(mat.toarray())


def test_orientation_is_source_rows_destination_columns():
    """``conn`` is ``(n_src, n_dst)`` and the layer computes ``x @ conn``.

    The matrix is deliberately non-square and asymmetric: a transposed or
    doubly transposed implementation cannot pass.
    """
    # Edge 0 -> 2 with weight 3, edge 1 -> 0 with weight 5.
    conn = scipy.sparse.coo_array(([3.0, 5.0], ([0, 1], [2, 0])), shape=(2, 3))
    layer = SparseConn(conn, enforce_dale=False)
    assert (layer.in_features, layer.out_features) == (2, 3)

    # Only source 0 is active: the current lands on destination 2.
    out = layer(torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(out, torch.tensor([0.0, 0.0, 3.0]))
    # Only source 1 is active: the current lands on destination 0.
    out = layer(torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(out, torch.tensor([5.0, 0.0, 0.0]))


def test_duplicate_entries_are_summed_into_one_parameter():
    """Duplicate ``(src, dst)`` entries collapse to one trainable weight."""
    conn = scipy.sparse.coo_array(
        ([1.0, 2.0, 5.0], ([0, 0, 1], [1, 1, 0])), shape=(2, 3)
    )
    layer = SparseConn(conn, enforce_dale=False)
    assert layer.magnitude.numel() == 2
    x = torch.tensor([1.0, 10.0])
    torch.testing.assert_close(layer(x), torch.tensor([50.0, 3.0, 0.0]))


def test_stored_indices_are_destination_major_and_sorted():
    """``indices`` holds ``[dst, src]`` sorted by destination, then source.

    Checkpoints store this buffer, so its layout is part of the
    contract.
    """
    conn, _ = _random_conn()
    layer = SparseConn(conn, enforce_dale=False)
    dst, src = layer.indices
    assert int(dst.max()) < conn.shape[1] and int(src.max()) < conn.shape[0]
    key = dst * conn.shape[0] + src
    assert bool((key[1:] > key[:-1]).all())
    assert set(layer.state_dict()) == {"magnitude", "indices"}


@pytest.mark.parametrize("lead", [(), (4,), (3, 4), (2, 3, 4)])
def test_leading_dimensions_are_preserved(lead):
    """Any number of leading (time / batch) dimensions is supported."""
    conn, dense = _random_conn()
    layer = SparseConn(conn, enforce_dale=False)
    x = torch.randn(*lead, conn.shape[0])
    out = layer(x)
    assert out.shape == (*lead, conn.shape[1])
    torch.testing.assert_close(out, x @ dense, atol=1e-5, rtol=1e-5)


def test_bias_is_added_per_destination():
    conn, dense = _random_conn()
    bias = torch.arange(conn.shape[1], dtype=torch.float32)
    layer = SparseConn(conn, bias=bias, enforce_dale=False)
    x = torch.randn(3, conn.shape[0])
    torch.testing.assert_close(layer(x), x @ dense + bias, atol=1e-5, rtol=1e-5)


def test_gradients_match_dense_reference():
    """Gradients w.r.t.

    the input and every edge weight equal the dense ones.
    """
    conn, dense = _random_conn()
    layer = SparseConn(conn, enforce_dale=False)
    x = torch.randn(3, conn.shape[0], requires_grad=True)
    target = torch.randn(3, conn.shape[1])

    ((layer(x) - target) ** 2).sum().backward()
    grad_x, grad_w = x.grad.clone(), layer.magnitude.grad.clone()

    dense = dense.clone().requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    ((x_ref @ dense - target) ** 2).sum().backward()

    torch.testing.assert_close(grad_x, x_ref.grad, atol=1e-5, rtol=1e-5)
    # ``indices`` is [dst, src]; the dense gradient is indexed [src, dst].
    dst, src = layer.indices
    torch.testing.assert_close(grad_w, dense.grad[src, dst], atol=1e-5, rtol=1e-5)


def test_dale_constraint_clamps_sign_flips_to_zero():
    """With ``enforce_dale`` a weight may shrink to zero but never flip
    sign."""
    conn = scipy.sparse.coo_array(
        ([1.0, -2.0, 3.0], ([0, 1, 2], [0, 0, 1])), shape=(3, 2)
    )
    layer = SparseConn(conn, enforce_dale=True)
    before = layer.magnitude.detach().clone()

    # Push every weight across zero, as a large optimiser step would.
    with torch.no_grad():
        layer.magnitude.mul_(-1.0)
    constrain_net(layer)
    assert torch.all(layer.magnitude == 0)

    # A same-sign update is left untouched.
    with torch.no_grad():
        layer.magnitude.copy_(before * 0.5)
    constrain_net(layer)
    torch.testing.assert_close(layer.magnitude.detach(), before * 0.5)


def _constrained_layer(enforce_dale=False):
    """Four edges in three groups; edges (0,0) and (1,1) share group 1."""
    conn = scipy.sparse.coo_array(
        ([1.0, -2.0, -3.0, 4.0], ([0, 1, 0, 1], [0, 0, 1, 1])), shape=(2, 2)
    )
    constraint = scipy.sparse.coo_array(
        ([1, 2, 3, 1], ([0, 1, 0, 1], [0, 0, 1, 1])), shape=(2, 2)
    )
    return SparseConstrainedConn(conn, constraint, enforce_dale=enforce_dale), conn


def test_constrained_effective_weight_is_base_times_group_scale():
    """``w[e] = initial_weight[e] * magnitude[group[e]]``; magnitudes start at
    1."""
    layer, conn = _constrained_layer()
    assert layer.magnitude.shape == (3,)
    torch.testing.assert_close(layer.magnitude.detach(), torch.ones(3))
    assert set(layer.state_dict()) == {"magnitude", "indices"}

    x = torch.tensor([[1.0, 10.0]])
    dense = torch.tensor(conn.toarray(), dtype=torch.float32)
    torch.testing.assert_close(layer(x), x @ dense)

    # Doubling group 1 (0-based id 0) doubles edges (0,0) and (1,1) only.
    layer.set_group_magnitude(group_id=0, value=2.0)
    dense[0, 0] *= 2
    dense[1, 1] *= 2
    torch.testing.assert_close(layer(x), x @ dense)


def test_constrained_group_gradient_matches_dense_reference():
    """The gradient of a group scale is the sum over its edges of ``dL/dw[e] *
    initial_weight[e]``."""
    layer, conn = _constrained_layer()
    x = torch.randn(5, 2)
    target = torch.randn(5, 2)
    ((layer(x) - target) ** 2).sum().backward()

    dense = torch.tensor(conn.toarray(), dtype=torch.float32, requires_grad=True)
    ((x @ dense - target) ** 2).sum().backward()
    g = dense.grad * dense.detach()  # dL/dw * base, per edge
    expected = torch.stack([g[0, 0] + g[1, 1], g[1, 0], g[0, 1]])
    torch.testing.assert_close(layer.magnitude.grad, expected, atol=1e-5, rtol=1e-5)


def test_constrained_dale_keeps_group_scales_non_negative():
    layer, _ = _constrained_layer(enforce_dale=True)
    with torch.no_grad():
        layer.magnitude.copy_(torch.tensor([-1.0, 0.5, 2.0]))
    constrain_net(layer)
    torch.testing.assert_close(layer.magnitude.detach(), torch.tensor([0.0, 0.5, 2.0]))


def test_constrained_missing_group_raises():
    """Every connection needs a constraint group."""
    conn = scipy.sparse.coo_array(([1.0, 2.0], ([0, 1], [0, 1])), shape=(2, 2))
    constraint = scipy.sparse.coo_array(([1], ([0], [0])), shape=(2, 2))
    with pytest.raises(ValueError, match="Constraint missing"):
        SparseConstrainedConn(conn, constraint)


def test_state_dict_roundtrip_same_topology():
    """Saving and loading into a layer built from the same matrix is exact."""
    conn, _ = _random_conn()
    a = SparseConn(conn, enforce_dale=True)
    with torch.no_grad():
        a.magnitude.mul_(0.3)
    b = SparseConn(conn, enforce_dale=True)
    b.load_state_dict(a.state_dict())
    x = torch.randn(4, conn.shape[0])
    torch.testing.assert_close(a(x), b(x))
    # Dale's law still holds the loaded weights in place.
    constrain_net(b)
    torch.testing.assert_close(a(x), b(x))


def test_dtype_and_device_follow_module():
    conn, dense = _random_conn()
    layer = SparseConn(conn, enforce_dale=False).to(torch.float64)
    x = torch.randn(3, conn.shape[0], dtype=torch.float64)
    out = layer(x)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, x @ dense.double())
