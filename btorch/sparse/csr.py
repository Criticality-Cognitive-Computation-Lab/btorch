"""Compressed sparse row / column formats."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Sparse, _as_tensor, _check_index
from .properties import EdgeMap, Hints, Properties


class _Compressed(Sparse):
    """Shared implementation of CSR and CSC.

    ``pointer`` compresses the *major* axis (rows for CSR, columns for CSC)
    and ``minor`` holds the other coordinate of every entry.
    """

    _major_axis: int  # 0 for CSR (rows), 1 for CSC (columns)

    def __init__(
        self,
        pointer: Tensor,
        minor: Tensor,
        values: Tensor,
        shape: tuple[int, ...],
        *,
        dense_dim: int = 0,
        properties: Properties | None = None,
        hints: Hints | None = None,
        check: bool = True,
    ):
        values = _as_tensor(values)
        pointer = _check_index("pointer", _as_tensor(pointer, device=values.device))
        minor = _check_index("indices", _as_tensor(minor, device=values.device))
        shape = tuple(int(s) for s in shape)
        name = type(self).__name__
        if pointer.ndim != 1 or minor.ndim != 1:
            raise ValueError(
                f"{name} stores one sparsity pattern: the pointer and index "
                "arrays must be 1-D. Batch members that share the pattern only "
                "differ in `values` ([*batch, nnz]); use sparse.stack() for "
                "different patterns."
            )
        batch_dim = values.ndim - 1 - dense_dim
        if batch_dim < 0 or len(shape) != batch_dim + 2 + dense_dim:
            raise ValueError(
                f"shape {shape} is inconsistent with values of shape "
                f"{tuple(values.shape)} and dense_dim={dense_dim}; {name} has "
                "exactly two sparse dimensions."
            )
        n_major = shape[batch_dim + self._major_axis]
        n_minor = shape[batch_dim + 1 - self._major_axis]
        nnz = int(minor.shape[0])
        if pointer.shape[0] != n_major + 1:
            raise ValueError(
                f"pointer has length {pointer.shape[0]}, expected {n_major + 1}."
            )
        expected = (
            shape[:batch_dim]
            + (nnz,)
            + shape[len(shape) - dense_dim :] * bool(dense_dim)
        )
        if tuple(values.shape) != expected:
            raise ValueError(
                f"values has shape {tuple(values.shape)}, expected {expected}."
            )
        if check:
            if int(pointer[0]) != 0 or int(pointer[-1]) != nnz:
                raise ValueError("pointer must start at 0 and end at nnz.")
            if bool((pointer[1:] < pointer[:-1]).any()):
                raise ValueError("pointer must be non-decreasing.")
            if nnz and (int(minor.min()) < 0 or int(minor.max()) >= n_minor):
                raise ValueError(f"indices must be in [0, {n_minor}).")
        self._pointer = pointer
        self._minor = minor
        self._values = values
        self._shape = shape
        self._batch_dim = batch_dim
        self._dense_dim = dense_dim
        self._value_batch_dim = batch_dim
        # Expanded major coordinate of each entry. Computed eagerly: filling
        # it lazily would change object state during a ``torch.compile``
        # trace and force a recompilation on the next call.
        counts = pointer[1:] - pointer[:-1]
        self._major = torch.repeat_interleave(
            torch.arange(counts.shape[0], device=counts.device), counts
        )
        self.properties = properties or Properties()
        if self.properties.shared_pattern != bool(batch_dim):
            self.properties = self.properties.replace(shared_pattern=bool(batch_dim))
        self.hints = hints or Hints()

    @property
    def nnz(self) -> int:
        return int(self._minor.shape[0])

    def values(self) -> Tensor:
        """``[*batch, nnz, *dense]`` stored values (PyTorch name)."""
        return self._values

    @property
    def data(self) -> Tensor:
        """Stored values (SciPy name)."""
        return self._values

    @property
    def indptr(self) -> Tensor:
        """Pointer array of the compressed axis (SciPy name)."""
        return self._pointer

    @property
    def indices(self) -> Tensor:
        """Coordinate of each entry along the uncompressed axis (SciPy
        name)."""
        return self._minor

    def _major_indices(self) -> Tensor:
        """Expanded coordinate of each entry along the compressed axis."""
        return self._major

    def _like(self, cls, **changes):
        out = cls.__new__(cls)
        out.__dict__.update(self.__dict__)
        for key, value in changes.items():
            setattr(out, key, value)
        return out

    def with_values(self, values: Tensor):
        batch_dim = values.ndim - 1 - self._dense_dim
        out = type(self)(
            self._pointer,
            self._minor,
            values,
            tuple(values.shape[:batch_dim]) + self._shape[self._batch_dim :],
            dense_dim=self._dense_dim,
            properties=self.properties,
            hints=self.hints,
            check=False,
        )
        out._major = self._major
        return out

    def _map_tensors(self, index_fn, value_fn):
        return self._like(
            type(self),
            _pointer=index_fn(self._pointer),
            _minor=index_fn(self._minor),
            _values=value_fn(self._values),
            _major=index_fn(self._major),
        )

    def _swapped_shape(self) -> tuple[int, ...]:
        shape = list(self._shape)
        i = self._batch_dim
        shape[i], shape[i + 1] = shape[i + 1], shape[i]
        return tuple(shape)

    def tocoo(self):
        from .coo import COO

        row, col = self._major_indices(), self._minor
        props = self.properties
        if self._major_axis == 1:
            row, col = col, row
            props = props.replace(sorted=False)
        return COO(
            torch.stack([row, col]),
            self._values,
            self._shape,
            dense_dim=self._dense_dim,
            properties=props,
            hints=self.hints,
            check=False,
        )

    def _edge_index(self) -> tuple[None, Tensor, Tensor]:
        if self._major_axis == 0:
            return None, self._major_indices(), self._minor
        return None, self._minor, self._major_indices()


class CSR(_Compressed):
    """Compressed sparse row format (exactly two sparse dimensions).

    Batch members share one sparsity pattern: ``crow_indices`` and
    ``col_indices`` are stored once and ``values`` is ``[*batch, nnz,
    *dense]``. This is the natural layout for ensembles and parameter sweeps
    over a fixed topology.

    Args:
        pointer: ``[n_rows + 1]`` row pointers (``crow_indices``).
        minor: ``[nnz]`` column of each entry (``col_indices``).
        values: ``[*batch, nnz, *dense]`` stored values.
        shape: Logical shape ``(*batch, n_rows, n_cols, *dense)``.
        dense_dim: Number of trailing dense dimensions of each entry.
        properties: Known structural properties.
        hints: Performance expectations.
        check: Validate the pointer and index ranges.

    Examples:
        >>> import torch
        >>> from btorch import sparse
        >>> A = sparse.csr(torch.tensor([0, 1, 2]), torch.tensor([2, 0]),
        ...                torch.tensor([3.0, 5.0]), shape=(2, 3))
        >>> A.indptr
        tensor([0, 1, 2])
        >>> A @ torch.tensor([1.0, 0.0, 2.0])
        tensor([6., 5.])
    """

    format = "csr"
    _major_axis = 0

    def crow_indices(self) -> Tensor:
        """``[n_rows + 1]`` row pointers (PyTorch name)."""
        return self._pointer

    def col_indices(self) -> Tensor:
        """``[nnz]`` column of each entry (PyTorch name)."""
        return self._minor

    def row_indices(self) -> Tensor:
        """``[nnz]`` row of each entry, expanded from the pointers."""
        return self._major_indices()

    def tocsr(self, return_map: bool = False):
        """Return ``self`` (already CSR; entries are left as stored).

        Use :meth:`coalesce` to sort columns and merge duplicates.
        """
        if return_map:
            return self, EdgeMap.identity(self.nnz, self._minor.device)
        return self

    def coalesce(self, return_map: bool = False):
        """Canonical CSR: sorted columns, duplicates summed."""
        return self.tocoo().tocsr(return_map)

    def tocsc(self, return_map: bool = False):
        return self.tocoo().tocsc(return_map)

    def transpose(self) -> "CSC":
        """Transpose without copying: the same arrays read as CSC."""
        return self._like(
            CSC,
            _shape=self._swapped_shape(),
            properties=self.properties.replace(triangular=None),
        )


class CSC(_Compressed):
    """Compressed sparse column format (exactly two sparse dimensions).

    Args:
        pointer: ``[n_cols + 1]`` column pointers (``ccol_indices``).
        minor: ``[nnz]`` row of each entry (``row_indices``).
        values: ``[*batch, nnz, *dense]`` stored values.
        shape: Logical shape ``(*batch, n_rows, n_cols, *dense)``.
        dense_dim: Number of trailing dense dimensions of each entry.
        properties: Known structural properties (``sorted`` refers to
            column-major order).
        hints: Performance expectations.
        check: Validate the pointer and index ranges.
    """

    format = "csc"
    _major_axis = 1

    def ccol_indices(self) -> Tensor:
        """``[n_cols + 1]`` column pointers (PyTorch name)."""
        return self._pointer

    def row_indices(self) -> Tensor:
        """``[nnz]`` row of each entry (PyTorch name)."""
        return self._minor

    def col_indices(self) -> Tensor:
        """``[nnz]`` column of each entry, expanded from the pointers."""
        return self._major_indices()

    def tocsr(self, return_map: bool = False):
        return self.tocoo().tocsr(return_map)

    def tocsc(self, return_map: bool = False):
        """Return ``self`` (already CSC; entries are left as stored).

        Use :meth:`coalesce` to sort rows and merge duplicates.
        """
        if return_map:
            return self, EdgeMap.identity(self.nnz, self._minor.device)
        return self

    def coalesce(self, return_map: bool = False):
        """Canonical CSC: sorted rows, duplicates summed."""
        return self.tocoo().tocsc(return_map)

    def transpose(self) -> CSR:
        """Transpose without copying: the same arrays read as CSR."""
        return self._like(
            CSR,
            _shape=self._swapped_shape(),
            properties=self.properties.replace(triangular=None),
        )
