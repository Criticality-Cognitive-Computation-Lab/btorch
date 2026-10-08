"""Products of sparse arrays with dense tensors against dense references.

``A @ x`` follows ``torch.matmul``: for ``A.shape == (M, N)`` the right
operand is a length-``N`` vector or a ``[..., N, K]`` stack of matrices.
``matvec`` / ``rmatvec`` instead apply the operator along the *last* axis of
``x``, with any number of leading sample dimensions.
"""

import numpy as np
import pytest
import torch

from btorch import sparse

from .helpers import (
    DEVICES,
    FORMATS,
    SHAPE,
    VARIANTS,
    build,
    dense_from_edges,
    random_edges,
)


M, N = SHAPE


def _case(fmt, variant, seed=0):
    """Sparse array in ``fmt`` and its independent dense ``[M, N]`` tensor."""
    rows, cols, values, dense = random_edges(variant, seed=seed)
    return build(fmt, rows, cols, values), torch.as_tensor(dense)


def _randn(*shape, seed=0):
    return torch.randn(*shape, dtype=torch.float64, generator=torch.manual_seed(seed))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_matmul_follows_torch_matmul(fmt, variant):
    """``A @ x``, ``A @ X``, ``x @ A`` and ``X @ A`` equal the dense result."""
    A, dense = _case(fmt, variant)
    if fmt == "coo":
        # COO keeps the entries as given: unsorted and with duplicates.
        assert A.properties.canonical is False
    # Vector operands.
    x, x_left = _randn(N), _randn(M)
    torch.testing.assert_close(A @ x, dense @ x)
    torch.testing.assert_close(x_left @ A, x_left @ dense)
    # Matrix operands: A @ X needs X = [N, K]; X @ A needs X = [K, M].
    X, X_left = _randn(N, 3), _randn(3, M)
    assert (A @ X).shape == (M, 3) and (X_left @ A).shape == (3, N)
    torch.testing.assert_close(A @ X, dense @ X)
    torch.testing.assert_close(X_left @ A, X_left @ dense)
    # Leading dimensions of the dense operand are batch dimensions.
    X, X_left = _randn(2, 4, N, 3), _randn(2, 4, 3, M)
    torch.testing.assert_close(A @ X, torch.matmul(dense, X))
    torch.testing.assert_close(X_left @ A, torch.matmul(X_left, dense))
    # The functional spelling is the same operation.
    torch.testing.assert_close(sparse.matmul(A, X), torch.matmul(dense, X))
    torch.testing.assert_close(sparse.matmul(X_left, A), torch.matmul(X_left, dense))


@pytest.mark.parametrize("lead", [(), (3,), (2, 3), (2, 1, 3)])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_matvec_and_rmatvec_accept_any_leading_dims(fmt, variant, lead):
    """``matvec`` maps ``[..., N] -> [..., M]``; ``rmatvec`` the reverse.

    This is the natural call for a batch of state vectors ``[B, N]`` (or
    ``[T, B, N]``): no transposing of ``x`` is needed, unlike ``A @ X``.
    """
    A, dense = _case(fmt, variant, seed=1)
    x, x_left = _randn(*lead, N), _randn(*lead, M)
    y = A.matvec(x)
    assert y.shape == (*lead, M)
    torch.testing.assert_close(y, torch.einsum("mn,...n->...m", dense, x))
    y = A.rmatvec(x_left)
    assert y.shape == (*lead, N)
    torch.testing.assert_close(y, torch.einsum("mn,...m->...n", dense, x_left))
    # Module-level functions and the transpose agree with the methods.
    torch.testing.assert_close(sparse.matvec(A, x), A.matvec(x))
    torch.testing.assert_close(sparse.rmatvec(A, x_left), A.T.matvec(x_left))
    assert A.T.shape == (N, M)
    torch.testing.assert_close(A.T.to_dense(), dense.T)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_non_canonical_compressed_storage(fmt):
    """CSR/CSC with unsorted minor indices and duplicates still sum."""
    # Three major slices (the middle one empty); the first stores minor
    # index 3 twice and out of order.
    pointer, minor = [0, 3, 3, 5], [3, 0, 3, 1, 0]
    values = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    major = np.array([0, 0, 0, 2, 2])
    if fmt == "csr":
        A = sparse.csr(pointer, minor, values, shape=(3, 4))
        dense = dense_from_edges(major, minor, values.numpy(), (3, 4))
    else:
        A = sparse.csc(pointer, minor, values, shape=(4, 3))
        dense = dense_from_edges(minor, major, values.numpy(), (4, 3))
    dense = torch.as_tensor(dense)
    assert A.nnz == 5  # stored as given
    torch.testing.assert_close(A.to_dense(), dense)
    x = torch.arange(1.0, A.shape[1] + 1)
    torch.testing.assert_close(A @ x, dense @ x)
    x = torch.arange(1.0, A.shape[0] + 1)
    torch.testing.assert_close(x @ A, x @ dense)
    # Converting to the other compressed format canonicalises.
    other = A.tocsc() if fmt == "csr" else A.tocsr()
    assert other.nnz == 4
    torch.testing.assert_close(other.to_dense(), dense)


@pytest.mark.parametrize("shape", [(3, 4), (0, 4), (3, 0), (0, 0)])
@pytest.mark.parametrize("fmt", FORMATS)
def test_zero_size_cases(fmt, shape):
    """No stored entries, no rows and no columns all behave like zeros."""
    m, n = shape
    empty = sparse.coo(torch.empty(2, 0, dtype=torch.long), torch.empty(0), shape)
    A = getattr(empty, f"to{fmt}")()
    dense = torch.zeros(shape)
    assert A.nnz == 0 and A.shape == shape and A.format == fmt
    assert torch.equal(A.to_dense(), dense)
    assert torch.equal(A @ torch.ones(n), dense @ torch.ones(n))
    assert torch.equal(torch.ones(m) @ A, torch.ones(m) @ dense)
    assert torch.equal(A @ torch.ones(n, 2), dense @ torch.ones(n, 2))
    assert torch.equal(A.matvec(torch.ones(6, n)), torch.zeros(6, m))
    assert torch.equal(A.rmatvec(torch.ones(6, m)), torch.zeros(6, n))
    # Interop of empty arrays keeps the shape too.
    assert A.to_scipy().shape == shape and A.to_scipy().nnz == 0
    assert tuple(A.to_torch().shape) == shape
    assert sparse.from_scipy(A.to_scipy()).shape == shape


def test_empty_python_lists_are_valid_indices():
    """``sparse.csr([0, 0, 0], [], [], shape)`` is the obvious empty matrix."""
    A = sparse.csr([0, 0, 0], [], torch.empty(0), shape=(2, 3))
    assert A.nnz == 0 and A.to_dense().tolist() == [[0.0] * 3] * 2


@pytest.mark.parametrize("fmt", FORMATS)
def test_shape_mismatch_is_reported(fmt):
    """A wrong operand length raises instead of gathering out of range."""
    A, _ = _case(fmt, "plain")
    with pytest.raises(ValueError, match=f"expected {N}"):
        A @ torch.ones(M)
    with pytest.raises(ValueError, match=f"expected {M}"):
        torch.ones(N) @ A
    with pytest.raises(ValueError, match=f"expected {N}"):
        A.matvec(torch.ones(3, M))


def test_constructors_validate_indices():
    """Out-of-range or malformed indices are caught at construction."""
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        sparse.coo(torch.tensor([[0], [3]]), torch.ones(1), shape=(2, 3))
    with pytest.raises(ValueError, match="pointer"):
        sparse.csr([0, 1], [0], torch.ones(1), shape=(2, 3))  # needs M + 1
    with pytest.raises(ValueError, match="indices"):
        sparse.csr([0, 1, 1], [3], torch.ones(1), shape=(2, 3))
    with pytest.raises(TypeError, match="integer"):
        sparse.coo(torch.tensor([[0.0], [1.0]]), torch.ones(1), shape=(2, 3))


# -------------------------------------------------------------------- dtype
@pytest.mark.parametrize("x_dtype", [torch.bool, torch.int64, torch.uint8])
@pytest.mark.parametrize("fmt", FORMATS)
def test_integer_and_bool_inputs_use_the_value_dtype(fmt, x_dtype):
    """Spike trains (bool/int) can be multiplied without an explicit cast."""
    A, dense = _case(fmt, "plain")
    A, dense = A.float(), dense.float()
    x = (torch.arange(N) % 3 == 0).to(x_dtype)
    y = A @ x
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, dense @ x.float())
    X = (torch.arange(4 * N).reshape(4, N) % 3 == 0).to(x_dtype)
    torch.testing.assert_close(A.matvec(X), X.float() @ dense.T)


@pytest.mark.parametrize("fmt", FORMATS)
def test_floating_dtypes(fmt):
    """Float32/float64 results match dense; mixed precision promotes."""
    A64, dense = _case(fmt, "duplicates")
    x64 = _randn(N)
    for dtype in (torch.float32, torch.float64):
        A, x = A64.to(dtype), x64.to(dtype)
        assert A.dtype == dtype and (A @ x).dtype == dtype
        torch.testing.assert_close(A @ x, dense.to(dtype) @ x)
    # Mixing precisions promotes like any elementwise torch operation.
    assert (A64 @ x64.float()).dtype == torch.promote_types(A64.dtype, torch.float32)
    torch.testing.assert_close(A64 @ x64.float(), dense @ x64.float().double())
    # Integer values with integer input stay integer.
    A_int = A64.with_values(torch.arange(1, A64.nnz + 1))
    x_int = torch.arange(N)
    assert (A_int @ x_int).dtype == torch.int64
    assert torch.equal(A_int @ x_int, A_int.to_dense() @ x_int)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", FORMATS)
def test_products_run_on_the_array_device(fmt, device):
    A, dense = _case(fmt, "duplicates")
    A, dense, x = A.to(device), dense.to(device), _randn(3, N).to(device)
    y = A.matvec(x)
    assert y.device.type == device
    torch.testing.assert_close(y, x @ dense.T)


# ---------------------------------------------------- dense entry dimensions
@pytest.mark.parametrize("variant", ["plain", "duplicates"])
@pytest.mark.parametrize("fmt", FORMATS)
def test_dense_entry_dimension(fmt, variant):
    """Every stored entry may carry a vector: ``shape == (M, N, D)``.

    Example: ``D`` receptor channels per synapse. ``matvec`` then returns one
    output per channel, ``[..., M, D]``.
    """
    n_dense = 3
    rows, cols, _, _ = random_edges(variant)
    values = _randn(len(rows), n_dense, seed=2)
    dense = torch.as_tensor(dense_from_edges(rows, cols, values.numpy(), SHAPE))
    coo = sparse.from_edges(rows, cols, values, (M, N, n_dense), dense_dim=1)
    A = getattr(coo, f"to{fmt}")()

    assert A.shape == (M, N, n_dense) and A.format == fmt
    assert (A.sparse_shape, A.dense_shape, A.dense_dim()) == (SHAPE, (n_dense,), 1)
    torch.testing.assert_close(A.to_dense(), dense)

    x, x_left = _randn(2, 4, N), _randn(2, 4, M)
    y = A.matvec(x)
    assert y.shape == (2, 4, M, n_dense)
    torch.testing.assert_close(y, torch.einsum("mnd,...n->...md", dense, x))
    torch.testing.assert_close(
        A.rmatvec(x_left), torch.einsum("mnd,...m->...nd", dense, x_left)
    )
    # A 1-D operand works with `@`; a matrix operand would be ambiguous.
    torch.testing.assert_close(A @ x[0, 0], torch.einsum("mnd,n->md", dense, x[0, 0]))
    with pytest.raises(NotImplementedError, match="matvec"):
        A @ _randn(N, 2)
    # torch COO has the same hybrid layout, so the round trip is lossless.
    back = sparse.from_torch(coo.coalesce().to_torch())
    assert back.dense_shape == (n_dense,)
    torch.testing.assert_close(back.to_dense(), dense)
