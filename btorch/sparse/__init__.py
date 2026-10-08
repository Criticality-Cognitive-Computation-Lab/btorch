"""Sparse arrays for PyTorch with SciPy-like ergonomics.

``btorch.sparse`` is a numerical sparse-array API that is independent of
neuroscience connectivity:

>>> import torch
>>> from btorch import sparse
>>> A = sparse.from_edges(torch.tensor([0, 1]), torch.tensor([2, 0]),
...                       torch.tensor([3.0, 5.0]), shape=(2, 3))
>>> A @ torch.tensor([1.0, 0.0, 2.0])
tensor([6., 5.])
>>> A.tocsr().indptr
tensor([0, 1, 2])

Matrix products follow standard linear algebra (``A.shape == (M, N)`` maps
length-``N`` vectors to length-``M`` vectors) and conversions from PyTorch or
SciPy never transpose. Connection semantics (source/destination orientation,
synapses, receptors, delays) live in :mod:`btorch.models.connection`.
"""

from collections.abc import Sequence

from torch import Tensor

from .base import Sparse
from .conversion import as_sparse, from_dense, from_scipy, from_torch
from .coo import COO
from .csr import CSC, CSR
from .einsum import einsum
from .operator import (
    CompositeOperator,
    ConstantOperator,
    DiagonalOperator,
    ImplicitOperator,
    LinearOperator,
    LowRankOperator,
    ProductOperator,
    ScaledOperator,
    StructuredOperator,
    SumOperator,
    TransposedOperator,
    is_linear_operator,
)
from .ops import matmul, matvec, rmatvec, stack
from .properties import EdgeMap, Hints, Properties


asarray = as_sparse


def explain(connection, x: Tensor | None = None) -> str:
    """Describe how a connection is executed (debugging aid).

    Reports the input representation, the canonical state, and the
    representation, algorithm and backend the runtime selected. Model code
    must not depend on the content.

    Args:
        connection: A connection module such as
            :class:`~btorch.models.connection.SparseConnection`.
        x: Optional example input.
    """
    return connection.explain(x)


def coo(
    indices: Tensor,
    values: Tensor,
    shape: Sequence[int],
    *,
    batch_dim: int | None = None,
    dense_dim: int = 0,
) -> COO:
    """Create a COO array from ``[n_index, nnz]`` indices; see :class:`COO`."""
    return COO(indices, values, tuple(shape), batch_dim=batch_dim, dense_dim=dense_dim)


def csr(
    crow_indices: Tensor,
    col_indices: Tensor,
    values: Tensor,
    shape: Sequence[int],
    *,
    dense_dim: int = 0,
) -> CSR:
    """Create a CSR array; see :class:`CSR`.

    ``values`` of shape ``[G, nnz]`` with ``shape=(G, M, N)`` creates ``G``
    matrices that share one sparsity pattern.
    """
    return CSR(crow_indices, col_indices, values, tuple(shape), dense_dim=dense_dim)


def csc(
    ccol_indices: Tensor,
    row_indices: Tensor,
    values: Tensor,
    shape: Sequence[int],
    *,
    dense_dim: int = 0,
) -> CSC:
    """Create a CSC array; see :class:`CSC`."""
    return CSC(ccol_indices, row_indices, values, tuple(shape), dense_dim=dense_dim)


def from_edges(
    rows: Tensor,
    cols: Tensor,
    values: Tensor,
    shape: Sequence[int],
    *,
    dense_dim: int = 0,
) -> Sparse:
    """Create a sparse matrix from row and column coordinates.

    The result is format-agnostic: use ``tocsr()`` / ``tocoo()`` to request a
    specific format.

    Args:
        rows: ``[nnz]`` row of each entry.
        cols: ``[nnz]`` column of each entry.
        values: ``[*batch, nnz, *dense]`` values.
        shape: ``(*batch, M, N, *dense)``.
        dense_dim: Number of trailing dense dimensions of each entry.
    """
    import torch

    rows = torch.as_tensor(rows)
    cols = torch.as_tensor(cols, device=rows.device)
    return COO(torch.stack([rows, cols]), values, tuple(shape), dense_dim=dense_dim)


__all__ = [
    "COO",
    "CSC",
    "CSR",
    "CompositeOperator",
    "ConstantOperator",
    "DiagonalOperator",
    "EdgeMap",
    "Hints",
    "ImplicitOperator",
    "LinearOperator",
    "LowRankOperator",
    "ProductOperator",
    "Properties",
    "ScaledOperator",
    "Sparse",
    "StructuredOperator",
    "SumOperator",
    "TransposedOperator",
    "as_sparse",
    "asarray",
    "coo",
    "csc",
    "csr",
    "einsum",
    "explain",
    "from_dense",
    "from_edges",
    "from_scipy",
    "from_torch",
    "is_linear_operator",
    "matmul",
    "matvec",
    "rmatvec",
    "stack",
]
