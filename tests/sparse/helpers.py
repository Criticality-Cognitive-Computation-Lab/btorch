"""Shared builders for the ``btorch.sparse`` tests.

Every helper returns the sparse data *and* a dense reference that is computed
independently of ``btorch.sparse`` (a NumPy scatter-add), so the tests never
compare the package against itself.
"""

import numpy as np
import pytest
import torch

from btorch import sparse


# Matrices are deliberately non-square: a transposed result has the wrong
# shape and cannot pass by accident.
SHAPE = (5, 8)

# The kinds of entry lists a user may hand to a COO constructor.
VARIANTS = ("plain", "empty_rows", "duplicates", "unsorted")
FORMATS = ("coo", "csr", "csc")

# Use as ``@pytest.mark.parametrize("device", DEVICES)``.
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is not available"
        ),
    ),
]


def random_edges(variant="plain", shape=SHAPE, seed=0, dtype=np.float64):
    """Random entry list ``(rows, cols, values)`` and its dense matrix.

    Args:
        variant: ``"plain"`` is row-major sorted without duplicates;
            ``"empty_rows"`` additionally leaves rows 1 and ``M - 1`` and
            column 0 without entries; ``"duplicates"`` stores four
            coordinates twice (in shuffled order); ``"unsorted"`` shuffles
            the entries without duplicating any.
        shape: ``(M, N)``.
        seed: Seed of the NumPy generator.
        dtype: Value dtype.

    Returns:
        ``rows [E]``, ``cols [E]``, ``values [E]`` and the dense ``[M, N]``
        array in which duplicate entries are summed.
    """
    rng = np.random.default_rng(seed)
    m, n = shape
    flat = np.sort(rng.choice(m * n, size=(m * n) // 3, replace=False))
    rows, cols = flat // n, flat % n
    if variant == "empty_rows":
        keep = (rows != 1) & (rows != m - 1) & (cols != 0)
        rows, cols = rows[keep], cols[keep]
    if variant == "duplicates":
        rows = np.concatenate([rows, rows[:4]])
        cols = np.concatenate([cols, cols[:4]])
    if variant in ("duplicates", "unsorted"):
        order = rng.permutation(len(rows))
        rows, cols = rows[order], cols[order]
    values = rng.uniform(0.5, 2.0, size=len(rows)).astype(dtype)
    return rows, cols, values, dense_from_edges(rows, cols, values, shape)


def dense_from_edges(rows, cols, values, shape):
    """Dense reference ``[M, N, *dense]``: scatter-add ``values [E, *dense]``.

    Duplicate ``(row, col)`` pairs are summed, which is the meaning of
    duplicates in SciPy, PyTorch and ``btorch.sparse``.
    """
    values = np.asarray(values)
    dense = np.zeros((*shape, *values.shape[1:]), dtype=values.dtype)
    np.add.at(dense, (rows, cols), values)
    return dense


def build(fmt, rows, cols, values, shape=SHAPE):
    """Btorch array in format ``fmt`` from an entry list.

    The COO array stores the entries exactly as given (unsorted, with
    duplicates); CSR and CSC are obtained with ``tocsr()`` / ``tocsc()``.
    """
    coo = sparse.from_edges(
        torch.as_tensor(rows), torch.as_tensor(cols), torch.as_tensor(values), shape
    )
    return getattr(coo, f"to{fmt}")()
