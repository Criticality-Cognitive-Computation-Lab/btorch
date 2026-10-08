"""``as_sparse`` and representation access (``to<fmt>()`` vs ``as_<fmt>()``).

``as_sparse`` is the single entry point every higher-level API uses to accept
"a sparse matrix": a btorch array, a torch sparse tensor or a SciPy sparse
array/matrix. It never transposes and never sparsifies dense data.
"""

import numpy as np
import pytest
import scipy.sparse
import torch

from btorch import sparse

from .helpers import FORMATS, SHAPE, build, random_edges


# ------------------------------------------------------------- orientation
def _tiny(kind):
    """The matrix ``[[0, 0, 3], [5, 0, 0]]`` as different sparse objects."""
    coo = scipy.sparse.coo_array(([3.0, 5.0], ([0, 1], [2, 0])), shape=(2, 3))
    dense = torch.tensor(coo.toarray())
    return {
        "scipy_coo": lambda: coo,
        "scipy_csr": lambda: coo.tocsr(),
        "scipy_csc": lambda: coo.tocsc(),
        "scipy_matrix": lambda: scipy.sparse.csr_matrix(coo),
        "torch_coo": lambda: dense.to_sparse(),
        "torch_csr": lambda: dense.to_sparse_csr(),
        "torch_csc": lambda: dense.to_sparse_csc(),
        "btorch": lambda: sparse.from_edges([0, 1], [2, 0], [3.0, 5.0], (2, 3)),
    }[kind]()


@pytest.mark.parametrize(
    "kind",
    ["scipy_coo", "scipy_csr", "scipy_csc", "scipy_matrix"]
    + ["torch_coo", "torch_csr", "torch_csc", "btorch"],
)
def test_as_sparse_accepts_every_input_kind_and_never_transposes(kind):
    """``as_sparse`` is the one entry point; ``A @ x`` is plain linear algebra.

    ``A.shape == (2, 3)`` maps length-3 vectors to length-2 vectors. With the
    asymmetric matrix ``[[0, 0, 3], [5, 0, 0]]`` a transposed conversion
    fails on the shape and a doubly transposed one on the values.
    """
    obj = _tiny(kind)
    A = sparse.as_sparse(obj)
    assert isinstance(A, sparse.Sparse) and A.shape == (2, 3)
    assert sparse.asarray is sparse.as_sparse
    x = torch.tensor([1.0, 0.0, 2.0], dtype=A.dtype)
    assert (A @ x).tolist() == [6.0, 5.0]
    assert (torch.tensor([1.0, 2.0], dtype=A.dtype) @ A).tolist() == [10.0, 0.0, 3.0]
    assert A.T.shape == (3, 2) and (A.T @ x[:2]).tolist() == [0.0, 0.0, 3.0]
    # A btorch array is passed through untouched ...
    if kind == "btorch":
        assert sparse.as_sparse(obj) is obj
    # ... and `format=` converts only when the stored format differs.
    for fmt in FORMATS:
        B = sparse.as_sparse(obj, format=fmt, dtype=torch.float64)
        assert B.format == fmt and B.dtype == torch.float64
        assert B.to_dense().tolist() == [[0.0, 0.0, 3.0], [5.0, 0.0, 0.0]]
    assert sparse.as_sparse(A, format=A.format) is A


@pytest.mark.parametrize(
    "dense",
    [torch.eye(3), np.eye(3), [[1.0, 0.0]], scipy.sparse.eye_array(3).toarray()],
)
def test_dense_input_is_never_sparsified_implicitly(dense):
    """Dense data is rejected instead of being silently converted."""
    with pytest.raises(TypeError):
        sparse.as_sparse(dense)
    with pytest.raises(TypeError):
        sparse.from_scipy(dense)
    with pytest.raises(TypeError):
        sparse.from_torch(dense)


def test_as_sparse_rejects_unknown_formats():
    A = _tiny("btorch")
    with pytest.raises(ValueError, match="xyz"):
        sparse.as_sparse(A, format="xyz")


def test_as_sparse_bsr_format_gives_a_clear_error():
    """BSR is not a native format: asking for it explains so."""
    with pytest.raises((NotImplementedError, ValueError)):
        sparse.as_sparse(_tiny("btorch"), format="bsr")


# ------------------------------------------------- to* versus as_* access
@pytest.mark.parametrize("target", FORMATS)
@pytest.mark.parametrize("stored", FORMATS)
def test_to_converts_and_as_only_grants_access(stored, target):
    """``to<fmt>()`` may convert; ``as_<fmt>()`` never does.

    ``as_<fmt>()`` returns the object itself if it is already stored in that
    format and otherwise raises an error that names the converting call.
    """
    rows, cols, values, dense = random_edges("duplicates")
    A = build(stored, rows, cols, values)

    converted = getattr(A, f"to{target}")()
    assert converted.format == target and converted.shape == SHAPE
    np.testing.assert_allclose(converted.to_dense().numpy(), dense)

    if stored == target:
        assert getattr(A, f"as_{target}")() is A
        assert converted is A  # no copy when nothing has to change
    else:
        with pytest.raises(TypeError, match=rf"to{target}\(\)"):
            getattr(A, f"as_{target}")()
    # BSR is not a native format: access always fails, conversion is explicit
    # about not being implemented.
    with pytest.raises(TypeError, match="BSR"):
        A.as_bsr()
    with pytest.raises(NotImplementedError):
        A.tobsr((1, 1))


def test_format_specific_names_only_exist_on_the_specific_format():
    """A COO array has no ``indptr``; a CSR array has no ``row``/``col``."""
    rows, cols, values, _ = random_edges()
    coo, csr = build("coo", rows, cols, values), build("csr", rows, cols, values)
    assert not hasattr(coo, "indptr") and not hasattr(coo, "crow_indices")
    assert not hasattr(csr, "row") and not hasattr(csr, "ccol_indices")
    # Generic metadata is available on every format.
    for A in (coo, csr, csr.tocsc()):
        assert (A.ndim, A.nnz, A.dtype) == (2, len(values), torch.float64)
        assert (A.batch_shape, A.sparse_shape, A.dense_shape) == ((), SHAPE, ())
        assert (A.batch_dim(), A.sparse_dim(), A.dense_dim()) == (0, 2, 0)


def test_from_dense_is_the_explicit_sparsification():
    """``from_dense`` is explicit, keeps zeros out and stays differentiable."""
    dense = torch.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], requires_grad=True)
    A = sparse.from_dense(dense)
    assert A.format == "coo" and A.nnz == 3 and A.shape == (2, 3)
    torch.testing.assert_close(A.to_dense(), dense)
    A.values().sum().backward()
    torch.testing.assert_close(dense.grad, (dense != 0).float())
