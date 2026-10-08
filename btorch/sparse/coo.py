"""Coordinate (COO) sparse format."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import Sparse, _as_tensor, _check_index
from .properties import EdgeMap, Hints, Properties


def _ravel(indices: Tensor, sizes: tuple[int, ...]) -> Tensor:
    """Row-major linear index of ``indices [len(sizes), E]``."""
    if math.prod(max(int(s), 1) for s in sizes) >= 2**63:
        raise OverflowError(
            f"The indexed shape {tuple(sizes)} has more than 2**63 positions; "
            "its entries cannot be linearised into int64 keys."
        )
    key = torch.zeros(indices.shape[1], dtype=torch.long, device=indices.device)
    for i, size in enumerate(sizes):
        key = key * size + indices[i]
    return key


def _unravel(key: Tensor, sizes: tuple[int, ...]) -> Tensor:
    out = []
    for size in reversed(sizes):
        out.append(key % size if size > 0 else key)
        key = torch.div(key, max(size, 1), rounding_mode="floor")
    return torch.stack(out[::-1]) if out else key.new_empty((0, key.shape[0]))


class COO(Sparse):
    """Coordinate format: one index tuple and one value per stored entry.

    COO supports any number of sparse dimensions. Entries may be unsorted and
    may contain duplicates; duplicates are summed by every operation, as in
    SciPy and PyTorch.

    Two kinds of batch are supported and may be combined:

    - *Shared pattern*: ``values`` has leading batch dimensions,
      ``[*batch, nnz, *dense]``, and every batch member uses the same indices.
    - *Different patterns*: the first index rows are batch coordinates, so each
      batch member can have its own number of entries. Build these with
      :func:`btorch.sparse.stack`.

    Args:
        indices: ``[n_index, nnz]`` integer coordinates. The rows are the
            batch coordinates (if any) followed by the sparse coordinates.
        values: ``[*value_batch, nnz, *dense]`` stored values.
        shape: Logical shape ``(*batch, *sparse, *dense)``.
        batch_dim: Total number of batch dimensions. Defaults to the number
            of leading value dimensions (a shared pattern).
        dense_dim: Number of trailing dense dimensions of each entry.
        properties: Known structural properties of the given indices.
        hints: Performance expectations.
        check: Validate index bounds.

    Examples:
        >>> import torch
        >>> from btorch import sparse
        >>> A = sparse.coo(torch.tensor([[0, 1], [2, 0]]),
        ...                torch.tensor([3.0, 5.0]), shape=(2, 3))
        >>> A @ torch.tensor([1.0, 0.0, 2.0])
        tensor([6., 5.])
    """

    format = "coo"

    def __init__(
        self,
        indices: Tensor,
        values: Tensor,
        shape: tuple[int, ...],
        *,
        batch_dim: int | None = None,
        dense_dim: int = 0,
        properties: Properties | None = None,
        hints: Hints | None = None,
        check: bool = True,
    ):
        values = _as_tensor(values)
        indices = _check_index("indices", _as_tensor(indices, device=values.device))
        shape = tuple(int(s) for s in shape)
        if indices.ndim != 2:
            raise ValueError(f"indices must be [n_index, nnz], got {indices.shape}.")
        n_index, nnz = indices.shape
        value_batch_dim = values.ndim - 1 - dense_dim
        if value_batch_dim < 0:
            raise ValueError(
                f"values with shape {tuple(values.shape)} cannot hold "
                f"{dense_dim} dense dimension(s) per entry."
            )
        if batch_dim is None:
            batch_dim = value_batch_dim
        index_batch_dim = batch_dim - value_batch_dim
        if index_batch_dim < 0:
            raise ValueError("batch_dim is smaller than the batch of `values`.")
        if len(shape) != value_batch_dim + n_index + dense_dim:
            raise ValueError(
                f"shape {shape} has {len(shape)} dimensions but indices/values "
                f"describe {value_batch_dim} value-batch + {n_index} indexed + "
                f"{dense_dim} dense."
            )
        if n_index - index_batch_dim < 1:
            raise ValueError("COO needs at least one sparse dimension.")
        if values.shape[value_batch_dim] != nnz:
            raise ValueError(
                f"values has {values.shape[value_batch_dim]} entries along its "
                f"entry axis but indices has {nnz}."
            )
        expected = (
            shape[:value_batch_dim] + (nnz,) + shape[len(shape) - dense_dim :]
            if dense_dim
            else shape[:value_batch_dim] + (nnz,)
        )
        if tuple(values.shape) != expected:
            raise ValueError(
                f"values has shape {tuple(values.shape)}, expected {expected}."
            )
        indexed = shape[value_batch_dim : value_batch_dim + n_index]
        if check and nnz > 0:
            lo = indices.amin(dim=1)
            hi = indices.amax(dim=1)
            for d, size in enumerate(indexed):
                if int(lo[d]) < 0 or int(hi[d]) >= size:
                    raise ValueError(
                        f"indices along indexed dimension {d} must be in "
                        f"[0, {size}), got range [{int(lo[d])}, {int(hi[d])}]."
                    )
        self._indices = indices
        self._values = values
        self._shape = shape
        self._batch_dim = batch_dim
        self._dense_dim = dense_dim
        self._value_batch_dim = value_batch_dim
        self.properties = properties or Properties()
        if self.properties.shared_pattern != bool(value_batch_dim):
            self.properties = self.properties.replace(
                shared_pattern=bool(value_batch_dim)
            )
        self.hints = hints or Hints()

    # ----------------------------------------------------------- accessors
    @property
    def nnz(self) -> int:
        return int(self._indices.shape[1])

    def indices(self) -> Tensor:
        """``[n_index, nnz]`` coordinates (PyTorch name)."""
        return self._indices

    def values(self) -> Tensor:
        """``[*value_batch, nnz, *dense]`` stored values (PyTorch name)."""
        return self._values

    @property
    def data(self) -> Tensor:
        """Stored values (SciPy name)."""
        return self._values

    @property
    def row(self) -> Tensor:
        """Row coordinate of each entry (two sparse dimensions only)."""
        self._require_matrix("row")
        return self._indices[-2]

    @property
    def col(self) -> Tensor:
        """Column coordinate of each entry (two sparse dimensions only)."""
        self._require_matrix("col")
        return self._indices[-1]

    def _require_matrix(self, what: str) -> None:
        if self.sparse_dim() != 2:
            raise ValueError(
                f"{what} needs exactly two sparse dimensions, this array has "
                f"{self.sparse_dim()}."
            )

    @property
    def _index_batch_dim(self) -> int:
        return self._batch_dim - self._value_batch_dim

    @property
    def _indexed_shape(self) -> tuple[int, ...]:
        """Sizes of the indexed (batch-coordinate + sparse) dimensions."""
        start = self._value_batch_dim
        return self._shape[start : start + self._indices.shape[0]]

    def _edge_index(self) -> tuple[Tensor | None, Tensor, Tensor]:
        """``(batch coordinates or None, row, col)`` for matrix products."""
        self._require_matrix("A matrix product")
        n_ib = self._index_batch_dim
        lead = self._indices[:n_ib] if n_ib else None
        return lead, self._indices[-2], self._indices[-1]

    # --------------------------------------------------------- construction
    def _like(self, indices: Tensor, values: Tensor, **changes) -> "COO":
        out = COO.__new__(COO)
        out.__dict__.update(self.__dict__)
        out._indices = indices
        out._values = values
        for key, value in changes.items():
            setattr(out, key, value)
        return out

    def with_values(self, values: Tensor) -> "COO":
        value_batch_dim = values.ndim - 1 - self._dense_dim
        tail = self._shape[self._value_batch_dim :]
        return COO(
            self._indices,
            values,
            tuple(values.shape[:value_batch_dim]) + tail,
            batch_dim=value_batch_dim + self._index_batch_dim,
            dense_dim=self._dense_dim,
            properties=self.properties,
            hints=self.hints,
            check=False,
        )

    def _map_tensors(self, index_fn, value_fn) -> "COO":
        return self._like(index_fn(self._indices), value_fn(self._values))

    # -------------------------------------------------------- canonical form
    def coalesce(self, return_map: bool = False) -> "COO" | tuple["COO", EdgeMap]:
        """Sort entries row-major and sum duplicates.

        Args:
            return_map: Also return the :class:`EdgeMap` from the old entries
                to the new ones, to move edge-aligned metadata consistently.

        Returns:
            The canonical COO, and the map if requested. The values of merged
            entries are summed, so gradients flow to every duplicate.
        """
        nnz = self.nnz
        if self.properties.canonical:
            edge_map = EdgeMap.identity(nnz, self._indices.device)
            return (self, edge_map) if return_map else self
        sizes = self._indexed_shape
        key = _ravel(self._indices, sizes)
        if self.properties.sorted:
            order = None
            sorted_key = key
        else:
            sorted_key, order = torch.sort(key, stable=True)
        unique_key, inverse = torch.unique_consecutive(sorted_key, return_inverse=True)
        if order is None:
            target = inverse
        else:
            target = torch.empty_like(inverse)
            target[order] = inverse
        edge_map = EdgeMap(target, int(unique_key.shape[0]))
        out = self._like(
            _unravel(unique_key, sizes),
            edge_map.sum(self._values, dim=self._value_batch_dim),
            properties=self.properties.replace(sorted=True, unique=True),
        )
        return (out, edge_map) if return_map else out

    def sum_duplicates(self) -> "COO":
        """SciPy-style alias of :meth:`coalesce` (returns a new array)."""
        return self.coalesce()

    # -------------------------------------------------------------- formats
    def tocoo(self) -> "COO":
        return self

    def _compressed(self, return_map: bool):
        """Canonical entries, their row pointers and the edge map."""
        self._require_matrix("CSR/CSC conversion")
        if self._index_batch_dim:
            raise NotImplementedError(
                "A batch of different sparsity patterns is stored as batched "
                "COO; compressed (CSR/CSC) storage for it is not implemented."
            )
        coo, edge_map = self.coalesce(return_map=True)
        n_rows = coo.sparse_shape[0]
        counts = torch.bincount(coo._indices[0], minlength=n_rows)
        pointer = torch.zeros(n_rows + 1, dtype=torch.long, device=counts.device)
        pointer[1:] = torch.cumsum(counts, 0)
        return coo, pointer, edge_map

    def tocsr(self, return_map: bool = False):
        """Convert to canonical CSR (sorted, duplicates summed).

        Args:
            return_map: Also return the :class:`EdgeMap` from the COO entries
                to the CSR entries.
        """
        from .csr import CSR

        coo, crow, edge_map = self._compressed(return_map)
        out = CSR(
            crow,
            coo._indices[1],
            coo._values,
            coo._shape,
            dense_dim=self._dense_dim,
            properties=coo.properties,
            hints=self.hints,
            check=False,
        )
        return (out, edge_map) if return_map else out

    def tocsc(self, return_map: bool = False):
        """Convert to canonical CSC (column-major, duplicates summed)."""
        result = self.transpose().tocsr(return_map)
        if return_map:
            return result[0].transpose(), result[1]
        return result.transpose()

    def transpose(self) -> "COO":
        self._require_matrix("transpose")
        idx = self._indices
        swapped = torch.cat([idx[:-2], idx[-1:], idx[-2:-1]])
        shape = list(self._shape)
        i = self._batch_dim
        shape[i], shape[i + 1] = shape[i + 1], shape[i]
        return self._like(
            swapped,
            self._values,
            _shape=tuple(shape),
            properties=self.properties.replace(sorted=False, triangular=None),
        )

    def to_dense(self) -> Tensor:
        n_vb = self._value_batch_dim
        sizes = self._indexed_shape
        dense = (
            self._shape[len(self._shape) - self._dense_dim :] if self._dense_dim else ()
        )
        key = _ravel(self._indices, sizes)
        flat = self._values.new_zeros(*self._shape[:n_vb], math.prod(sizes), *dense)
        flat = flat.index_add(n_vb, key, self._values)
        return flat.reshape(self._shape)
