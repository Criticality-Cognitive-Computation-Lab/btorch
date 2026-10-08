"""Interop of ``btorch.sparse`` with SciPy and PyTorch sparse objects.

The contract under test: conversions keep shape, values, indices, dtype and
device; they keep the storage format when btorch has it (COO, CSR, CSC) and go
through COO otherwise; they never transpose and never densify; and values
that come from PyTorch keep their autograd history.
"""

import numpy as np
import pytest
import scipy.sparse
import torch

from btorch import sparse

from .helpers import DEVICES, FORMATS, SHAPE, build, random_edges


NATIVE = set(FORMATS)


def _scipy(fmt, kind="array", variant="plain", dtype=np.float32):
    """SciPy sparse object in format ``fmt`` and its dense reference.

    ``kind`` selects the modern ``sparray`` or the legacy ``spmatrix`` class.
    """
    rows, cols, values, dense = random_edges(variant, dtype=dtype)
    mat = scipy.sparse.coo_array((values, (rows, cols)), shape=SHAPE).asformat(fmt)
    if kind == "matrix":
        mat = getattr(scipy.sparse, f"{fmt}_matrix")(mat)
    return mat, dense


# --------------------------------------------------------------------- SciPy
@pytest.mark.parametrize("kind", ["array", "matrix"])
@pytest.mark.parametrize("fmt", ["coo", "csr", "csc", "bsr", "lil", "dok", "dia"])
def test_scipy_round_trip(fmt, kind):
    """SciPy -> btorch -> SciPy keeps shape, values, dtype and the format."""
    mat, dense = _scipy(fmt, kind)
    A = sparse.from_scipy(mat)

    # COO/CSR/CSC are stored natively; every other format goes through COO.
    assert A.format == (fmt if fmt in NATIVE else "coo")
    assert A.shape == SHAPE  # (5, 8): a transposed result would be (8, 5)
    assert A.dtype == torch.float32
    np.testing.assert_array_equal(A.to_dense().numpy(), dense)
    # Whatever order the entries arrived in, a later conversion to CSR must
    # agree with SciPy's own canonical CSR, index for index.
    ref = scipy.sparse.csr_array(dense)
    np.testing.assert_array_equal(A.tocsr().indptr.numpy(), ref.indptr)
    np.testing.assert_array_equal(A.tocsr().indices.numpy(), ref.indices)
    np.testing.assert_array_equal(A.tocsr().data.numpy(), ref.data)

    # Export back into the original SciPy format. The result is always a
    # modern sparse *array*, also for legacy matrix input.
    back = A.to_scipy(format=fmt)
    assert isinstance(back, scipy.sparse.sparray)
    assert back.format == fmt and back.shape == SHAPE and back.dtype == mat.dtype
    np.testing.assert_array_equal(back.toarray(), dense)


@pytest.mark.parametrize("kind", ["array", "matrix"])
@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_scipy_compressed_arrays_are_kept_verbatim(fmt, kind):
    """CSR/CSC pointer, index and data arrays are taken over unchanged."""
    mat, _ = _scipy(fmt, kind)
    A = sparse.from_scipy(mat)
    # SciPy names on the format-specific object ...
    np.testing.assert_array_equal(A.indptr.numpy(), mat.indptr)
    np.testing.assert_array_equal(A.indices.numpy(), mat.indices)
    np.testing.assert_array_equal(A.data.numpy(), mat.data)
    # ... and the PyTorch names for the same arrays.
    pointer = A.crow_indices() if fmt == "csr" else A.ccol_indices()
    minor = A.col_indices() if fmt == "csr" else A.row_indices()
    assert pointer is A.indptr and minor is A.indices and A.values() is A.data
    # SciPy often stores int32 indices; btorch indices are always int64.
    assert A.indptr.dtype == A.indices.dtype == torch.long
    back = A.to_scipy()
    np.testing.assert_array_equal(back.indptr, mat.indptr)
    np.testing.assert_array_equal(back.indices, mat.indices)


def test_scipy_coo_duplicates_and_order_are_kept_as_stored():
    """Unsorted COO with duplicates is not canonicalised behind your back.

    Duplicates mean "sum" (as in SciPy); they are only merged on
    request.
    """
    mat, dense = _scipy("coo", variant="duplicates")
    A = sparse.from_scipy(mat)
    assert A.nnz == mat.nnz and not A.properties.canonical
    np.testing.assert_array_equal(A.row.numpy(), mat.row)
    np.testing.assert_array_equal(A.col.numpy(), mat.col)
    np.testing.assert_array_equal(A.data.numpy(), mat.data)
    np.testing.assert_allclose(A.to_dense().numpy(), dense)
    # The export is just as literal: same entries, same order.
    back = A.to_scipy()
    assert back.nnz == mat.nnz
    np.testing.assert_array_equal(back.row, mat.row)

    # coalesce() is the explicit canonicalisation. SciPy's CSR conversion is
    # the independent reference for "row-major sorted, duplicates summed".
    ref = mat.tocsr()
    ref.sum_duplicates()
    canon = A.coalesce()
    assert canon.nnz == ref.nnz == mat.nnz - 4 and canon.properties.canonical
    np.testing.assert_array_equal(canon.row.numpy(), ref.tocoo().row)
    np.testing.assert_array_equal(canon.col.numpy(), ref.indices)
    np.testing.assert_allclose(canon.data.numpy(), ref.data)
    # tocsr() canonicalises the same way.
    csr = A.tocsr()
    np.testing.assert_array_equal(csr.indptr.numpy(), ref.indptr)
    np.testing.assert_array_equal(csr.indices.numpy(), ref.indices)
    np.testing.assert_allclose(csr.data.numpy(), ref.data)


def test_scipy_csr_with_duplicates_and_unsorted_columns():
    """A non-canonical SciPy CSR keeps its stored entries.

    Such a matrix can only be built from raw ``(data, indices, indptr)``.
    Row 0 stores column 2 twice and its columns are not sorted.
    """
    data = np.array([1.0, 2.0, 4.0, 8.0])
    mat = scipy.sparse.csr_array((data, [2, 0, 2, 1], [0, 3, 4]), shape=(2, 3))
    A = sparse.from_scipy(mat)
    assert A.format == "csr" and A.nnz == 4
    assert not A.properties.sorted and not A.properties.unique
    dense = np.array([[2.0, 0.0, 5.0], [0.0, 8.0, 0.0]])
    np.testing.assert_array_equal(A.to_dense().numpy(), dense)
    np.testing.assert_array_equal(A.to_scipy().indices, mat.indices)
    # Going through COO merges the duplicate and sorts the columns.
    canon = A.tocoo().tocsr()
    assert canon.indices.tolist() == [0, 2, 1] and canon.data.tolist() == [2, 5, 8]


def test_scipy_dok_in_arbitrary_insertion_order_converts_to_csr():
    """A DOK filled in arbitrary order still gives the right CSR."""
    mat = scipy.sparse.dok_array((3, 4))
    # Insertion order is neither row- nor column-major.
    mat[2, 1], mat[0, 3], mat[1, 0], mat[0, 0] = 1.0, 2.0, 3.0, 4.0
    A = sparse.from_scipy(mat)
    np.testing.assert_array_equal(A.to_dense().numpy(), mat.toarray())  # fine
    ref = mat.tocsr()
    ref.sort_indices()
    csr = A.tocsr()
    np.testing.assert_array_equal(csr.to_dense().numpy(), mat.toarray())
    np.testing.assert_array_equal(csr.indptr.numpy(), ref.indptr)
    np.testing.assert_array_equal(csr.indices.numpy(), ref.indices)
    np.testing.assert_array_equal(A.tocsc().to_dense().numpy(), mat.toarray())


@pytest.mark.parametrize("np_dtype", [np.float32, np.float64, np.int64, np.complex64])
def test_scipy_dtype_is_preserved_and_can_be_overridden(np_dtype):
    """The value dtype follows the SciPy array unless ``dtype`` is given."""
    mat = scipy.sparse.coo_array(
        (np.array([3, 5], dtype=np_dtype), ([0, 1], [2, 0])), shape=(2, 3)
    )
    A = sparse.from_scipy(mat)
    assert A.dtype == torch.from_numpy(mat.data).dtype
    assert A.to_scipy().dtype == np_dtype
    # An explicit dtype wins (kept complex here to avoid a lossy cast).
    target = torch.complex128 if A.dtype.is_complex else torch.float64
    assert sparse.from_scipy(mat, dtype=target).dtype == target


def test_dtype_argument_casts_integer_values():
    """``dtype`` must be honoured for integer-valued input as well."""
    mat = scipy.sparse.coo_array((np.array([3, 5]), ([0, 1], [2, 0])), shape=(2, 3))
    assert sparse.as_sparse(mat, dtype=torch.float32).dtype == torch.float32
    assert sparse.from_scipy(mat).to(torch.float32).dtype == torch.float32
    assert sparse.from_scipy(mat).double().dtype == torch.float64


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", FORMATS)
def test_scipy_device_is_explicit(fmt, device):
    """SciPy data is CPU data; ``device=`` puts values *and* indices there."""
    mat, dense = _scipy(fmt)
    A = sparse.from_scipy(mat, device=device)
    assert A.device.type == device
    index = A.indices() if fmt == "coo" else A.indices
    assert index.device.type == device
    # Export always lands in CPU memory again.
    np.testing.assert_array_equal(A.to_scipy().toarray(), dense)
    # .to() / .cpu() move the whole object.
    assert A.cpu().device.type == "cpu"
    np.testing.assert_array_equal(A.cpu().to_dense().numpy(), dense)


# --------------------------------------------------------------------- torch
def _torch_dense(device="cpu"):
    """Dense ``[4, 6]`` matrix with an empty row and an all-zero 2x3 block."""
    dense = torch.zeros(4, 6, device=device)
    dense[0, 1], dense[0, 4], dense[1, 0], dense[3, 5] = 1.0, 2.0, 3.0, 4.0
    return dense


def _to_layout(tensor, layout):
    """Dense or COO tensor -> torch sparse tensor with the named layout."""
    kwargs = {"blocksize": (2, 3)} if layout in ("bsr", "bsc") else {}
    return tensor.to_sparse(layout=getattr(torch, f"sparse_{layout}"), **kwargs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", ["coo", "csr", "csc", "bsr", "bsc"])
def test_torch_round_trip(layout, device):
    """Torch -> btorch -> torch keeps shape, values, dtype, device, layout."""
    dense = _torch_dense(device)
    t = _to_layout(dense, layout)
    A = sparse.from_torch(t)

    # Block layouts have no native btorch format and are stored as COO.
    assert A.format == (layout if layout in NATIVE else "coo")
    assert A.shape == (4, 6) and A.dtype == t.dtype and A.device == t.device
    torch.testing.assert_close(A.to_dense(), dense)

    # Native formats share the index and value tensors' contents exactly.
    if layout == "coo":
        assert torch.equal(A.indices(), t.coalesce().indices())
        assert torch.equal(A.values(), t.coalesce().values())
    elif layout == "csr":
        assert torch.equal(A.crow_indices(), t.crow_indices())
        assert torch.equal(A.col_indices(), t.col_indices())
    elif layout == "csc":
        assert torch.equal(A.ccol_indices(), t.ccol_indices())
        assert torch.equal(A.row_indices(), t.row_indices())

    # Export into the original layout (the default is the stored format).
    kwargs = {"blocksize": (2, 3)} if layout in ("bsr", "bsc") else {}
    back = A.to_torch(layout=layout, **kwargs)
    assert back.layout == t.layout and back.shape == t.shape
    assert back.dtype == t.dtype and back.device == t.device
    torch.testing.assert_close(back.to_dense(), dense)
    assert A.to_torch().layout == getattr(torch, f"sparse_{A.format}")
    # A torch layout object is accepted in place of its name.
    assert A.to_torch(layout=torch.sparse_csc).layout == torch.sparse_csc


def test_torch_uncoalesced_coo_is_coalesced():
    """PyTorch only exposes the entries of a coalesced COO tensor.

    ``from_torch`` therefore coalesces: duplicates are summed and the result
    is canonical, and both duplicates still receive a gradient.
    """
    values = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
    # (1, 2) is stored twice and (0, 1) comes last.
    t = torch.sparse_coo_tensor(torch.tensor([[1, 1, 0], [2, 2, 1]]), values, (2, 3))
    A = sparse.from_torch(t)
    assert A.nnz == 2 and A.properties.canonical
    assert A.indices().tolist() == [[0, 1], [1, 2]]
    assert A.values().tolist() == [4.0, 3.0]
    assert A.to_torch().is_coalesced()
    (A @ torch.tensor([1.0, 10.0, 100.0])).sum().backward()
    assert values.grad.tolist() == [100.0, 100.0, 10.0]


@pytest.mark.parametrize("layout", ["coo", "csr", "csc", "bsr", "bsc"])
def test_torch_values_keep_autograd_history(layout):
    """Gradients flow through ``from_torch`` back to the original values."""
    indices = torch.tensor([[0, 0, 1, 3], [1, 4, 0, 5]])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    t = _to_layout(torch.sparse_coo_tensor(indices, values, (4, 6)).coalesce(), layout)
    A = sparse.from_torch(t)
    assert A.requires_grad
    x = torch.arange(1.0, 7.0)
    (A @ x).sum().backward()
    # d/dv_e sum(A @ x) = x[col_e]
    torch.testing.assert_close(values.grad, x[indices[1]])


@pytest.mark.parametrize("fmt", FORMATS)
def test_to_torch_keeps_autograd_history(fmt):
    """The exported torch tensor is still connected to the btorch values."""
    rows, cols, values, _ = random_edges("plain")
    A = build(fmt, rows, cols, values)
    leaf = A.values().detach().requires_grad_()
    weight = torch.arange(40.0, dtype=torch.float64).reshape(SHAPE)
    (A.with_values(leaf).to_torch().to_dense() * weight).sum().backward()
    # Entry e sits at (row_e, col_e), so its gradient is weight[row_e, col_e].
    coo = A.tocoo()
    torch.testing.assert_close(leaf.grad, weight[coo.row, coo.col])
    # detach() cuts the history without touching the pattern.
    assert not A.with_values(leaf).detach().requires_grad
