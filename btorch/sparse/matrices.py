from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import scipy.sparse
import torch
import torch.utils._pytree as pytree

from .base import SparseTensorBase
from .initializers import Normal, initialize_values
from .layouts import PaddedCSRLayout
from .properties import EdgeAttributes, LogicalEdges, SparseProperties


# ── Helpers ───────────────────────────────────────────────────────────────────


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2 or shape[0] < 0 or shape[1] < 0:
        raise ValueError("shape must contain two non-negative integers.")
    return int(shape[0]), int(shape[1])


def _ensure_float(data: torch.Tensor) -> torch.Tensor:
    if not data.is_floating_point() and not data.is_complex():
        data = data.to(torch.get_default_dtype())
    return data


def _validate_attributes(attributes: EdgeAttributes, nnz: int) -> None:
    for name, label in attributes.labels.items():
        if label.numel() != nnz:
            raise ValueError(f"Label {name!r} must contain one value per edge.")


def _reorder_attributes(
    attributes: EdgeAttributes,
    order: torch.Tensor,
) -> EdgeAttributes:
    return EdgeAttributes(
        {name: label[order] for name, label in attributes.labels.items()}
    )


def _validate_properties(
    row: torch.Tensor,
    col: torch.Tensor,
    shape: tuple[int, int],
    properties: SparseProperties,
    *,
    storage_order: str,
) -> None:
    for name, degree in (
        ("max_fan_out", properties.max_fan_out),
        ("max_fan_in", properties.max_fan_in),
    ):
        if degree is not None and degree < 0:
            raise ValueError(f"{name} must be non-negative.")

    if properties.max_fan_out is not None:
        fan_out = torch.bincount(row, minlength=shape[0])
        if not torch.all(fan_out == properties.max_fan_out):
            raise ValueError(
                "max_fan_out does not match the number of edges for every row."
            )
    if properties.max_fan_in is not None:
        fan_in = torch.bincount(col, minlength=shape[1])
        if not torch.all(fan_in == properties.max_fan_in):
            raise ValueError(
                "max_fan_in does not match the number of edges for every col."
            )

    edge_key = row * shape[1] + col
    if properties.unique_edges and edge_key.numel() != torch.unique(edge_key).numel():
        raise ValueError("unique_edges=True requires unique row-col pairs.")

    if properties.sorted_indices and row.numel() > 1:
        if storage_order == "row":
            order_key = edge_key
        elif storage_order == "column":
            order_key = col * shape[0] + row
        else:
            raise ValueError(f"Unsupported storage order: {storage_order!r}.")
        if torch.any(order_key[1:] < order_key[:-1]):
            raise ValueError(
                "sorted_indices=True does not match the physical index ordering."
            )


def _canonicalize_edges(
    row: torch.Tensor,
    col: torch.Tensor,
    data: torch.Tensor,
    shape: tuple[int, int],
    labels: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    row = torch.as_tensor(row, dtype=torch.long, device=data.device).flatten()
    col = torch.as_tensor(col, dtype=torch.long, device=data.device).flatten()
    data = data.flatten()
    if row.numel() != col.numel() or row.numel() != data.numel():
        raise ValueError("row, col, and data must have equal lengths.")
    if row.numel():
        if row.min() < 0 or row.max() >= shape[0]:
            raise ValueError("row contains an out-of-range index.")
        if col.min() < 0 or col.max() >= shape[1]:
            raise ValueError("col contains an out-of-range index.")

    norm_labels: dict[str, torch.Tensor] = {
        name: torch.as_tensor(label, device=data.device).flatten()
        for name, label in (labels or {}).items()
    }
    for name, label in norm_labels.items():
        if label.numel() != row.numel():
            raise ValueError(f"Label {name!r} must contain one value per edge.")

    if not row.numel():
        return row, col, data, norm_labels

    key = row * shape[1] + col
    order = torch.argsort(key, stable=True)
    row = row[order]
    col = col[order]
    data = data[order]
    norm_labels = {name: label[order] for name, label in norm_labels.items()}
    key = key[order]

    unique_key, inverse = torch.unique_consecutive(key, return_inverse=True)
    if unique_key.numel() != key.numel():
        merged = torch.zeros(unique_key.numel(), device=data.device, dtype=data.dtype)
        merged.scatter_add_(0, inverse, data)
        first = torch.ones_like(key, dtype=torch.bool)
        first[1:] = key[1:] != key[:-1]
        for name, label in norm_labels.items():
            first_vals = label[first]
            if not torch.equal(first_vals[inverse], label):
                raise ValueError(
                    f"Duplicate edges must have identical {name!r} labels."
                )
            norm_labels[name] = first_vals
        data = merged
        row = torch.div(unique_key, shape[1], rounding_mode="floor")
        col = unique_key.remainder(shape[1])

    return row, col, data, norm_labels


def _coo_mm(row, col, data, shape, x):
    """Compute x @ W using COO data.

    x: (..., n_pre) → (..., n_post).
    """
    leading = x.shape[:-1]
    x2d = x.reshape(-1, shape[0])
    if data.numel() == 0:
        return torch.zeros(*leading, shape[1], device=x.device, dtype=x.dtype)
    indices = torch.stack([col, row])
    with torch.sparse.check_sparse_tensor_invariants(False):
        wt = torch.sparse_coo_tensor(
            indices,
            data,
            size=(shape[1], shape[0]),
            device=data.device,
            dtype=data.dtype,
            is_coalesced=False,
        )
    out = torch.sparse.mm(wt, x2d.T).T
    return out.reshape(*leading, shape[1])


def _ell_mm(indices, data, shape, x):
    """Compute x @ W for ELL layout."""
    leading = x.shape[:-1]
    n_pre, width = indices.shape
    n_post = shape[1]
    x2d = x.reshape(-1, n_pre)
    batch = x2d.shape[0]
    if width == 0 or n_pre == 0:
        return torch.zeros(*leading, n_post, device=x.device, dtype=x.dtype)
    gathered = x2d.unsqueeze(-1) * data.unsqueeze(0)  # (batch, n_pre, width)
    flat_cols = indices.unsqueeze(0).expand(batch, -1, -1)
    result = torch.zeros(batch, n_post, device=x.device, dtype=x.dtype)
    result.scatter_add_(1, flat_cols.reshape(batch, -1), gathered.reshape(batch, -1))
    return result.reshape(*leading, n_post)


# ── COO ───────────────────────────────────────────────────────────────────────


class COO(SparseTensorBase):
    """COO sparse matrix.

    row/col/data are (nnz,) tensors.
    """

    row: torch.Tensor
    col: torch.Tensor
    data: torch.Tensor

    def __init__(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        data: torch.Tensor,
        shape: tuple[int, int],
        properties: SparseProperties | None = None,
        attributes: EdgeAttributes | None = None,
        constraint: object | None = None,
        parameterization: object | None = None,
    ) -> None:
        shape = _validate_shape(shape)
        row = torch.as_tensor(row, dtype=torch.long).flatten()
        col = torch.as_tensor(col, dtype=torch.long).flatten()
        data = _ensure_float(torch.as_tensor(data).flatten())
        if row.numel():
            if row.min() < 0 or row.max() >= shape[0]:
                raise ValueError("row contains an out-of-range index.")
            if col.min() < 0 or col.max() >= shape[1]:
                raise ValueError("col contains an out-of-range index.")
        if row.numel() != col.numel() or row.numel() != data.numel():
            raise ValueError("row, col, and data must have equal lengths.")
        properties = properties or SparseProperties()
        attributes = attributes or EdgeAttributes()
        _validate_properties(
            row,
            col,
            shape,
            properties,
            storage_order="row",
        )
        _validate_attributes(attributes, row.numel())
        self.row = row
        self.col = col
        self.data = data
        self.shape = shape
        self.properties = properties
        self.attributes = attributes
        self.constraint = constraint
        self.parameterization = parameterization
        self._cache: dict = {}

    @classmethod
    def _new_unsafe(
        cls,
        *,
        row: torch.Tensor,
        col: torch.Tensor,
        data: torch.Tensor,
        shape: tuple[int, int],
        properties: SparseProperties,
        attributes: EdgeAttributes,
        constraint: object | None,
        parameterization: object | None,
    ) -> "COO":
        obj = object.__new__(cls)
        obj.row = row
        obj.col = col
        obj.data = data
        obj.shape = shape
        obj.properties = properties
        obj.attributes = attributes
        obj.constraint = constraint
        obj.parameterization = parameterization
        obj._cache = {}
        return obj

    @classmethod
    def from_edges(
        cls,
        row,
        col,
        data,
        shape: tuple[int, int],
        *,
        properties: SparseProperties | None = None,
        labels: Mapping[str, torch.Tensor] | None = None,
        constraint: object | None = None,
        parameterization: object | None = None,
        device=None,
        dtype=None,
    ) -> "COO":
        shape = _validate_shape(shape)
        data = _ensure_float(torch.as_tensor(data, device=device, dtype=dtype))
        src, dst, vals, nlabels = _canonicalize_edges(
            torch.as_tensor(row, device=device),
            torch.as_tensor(col, device=device),
            data,
            shape,
            labels,
        )
        requested = properties or SparseProperties()
        props = replace(requested, sorted_indices=True, unique_edges=True)
        _validate_properties(src, dst, shape, props, storage_order="row")
        return cls._new_unsafe(
            row=src,
            col=dst,
            data=vals,
            shape=shape,
            properties=props,
            attributes=EdgeAttributes(nlabels),
            constraint=constraint,
            parameterization=parameterization,
        )

    def to_coo(self) -> "COO":
        return self

    def to_csr(self) -> "CSR":
        n_pre = self.shape[0]
        order = torch.argsort(self.row, stable=True)
        source_s = self.row[order]
        destination_s = self.col[order]
        values_s = self.data[order]
        attributes = _reorder_attributes(self.attributes, order)
        counts = torch.bincount(source_s, minlength=n_pre)
        indptr = torch.cat(
            [torch.zeros(1, device=source_s.device, dtype=torch.long), counts.cumsum(0)]
        )
        return CSR._new_unsafe(
            indptr=indptr,
            indices=destination_s,
            data=values_s,
            shape=self.shape,
            properties=self.properties,
            attributes=attributes,
            constraint=self.constraint,
            parameterization=self.parameterization,
        )

    def to_csc(self) -> "CSC":
        return self.to_csr().to_csc()

    def to_ell(self) -> "ELL":
        return self.to_csr().to_ell()

    def mm(self, x: torch.Tensor) -> torch.Tensor:
        return _coo_mm(self.row, self.col, self.effective_values(), self.shape, x)

    def logical_edges(self) -> LogicalEdges:
        idx = torch.arange(self.row.numel(), device=self.row.device)
        return LogicalEdges(row=self.row, col=self.col, storage_index=idx)

    def nnz(self) -> int:
        return int(self.data.numel())

    def _trainable_tensors(self) -> list[tuple[str, torch.Tensor]]:
        return [("data", self.data)]

    def _structural_tensors(self) -> list[tuple[str, torch.Tensor]]:
        return [("row", self.row), ("col", self.col)]

    def _rebuild(self, parameters: dict, buffers: dict) -> "COO":
        if self.parameterization is not None:
            new_p = self.parameterization._rebuild(parameters, buffers)
            new_values = buffers["initial_weight"]
        else:
            new_values = parameters["data"]
        return COO._new_unsafe(
            row=buffers["row"],
            col=buffers["col"],
            data=new_values,
            shape=self.shape,
            properties=self.properties,
            attributes=self.attributes,
            constraint=self.constraint,
            parameterization=new_p if self.parameterization is not None else None,
        )

    def with_constraint(self, c: object) -> "COO":
        return COO._new_unsafe(
            row=self.row,
            col=self.col,
            data=self.data,
            shape=self.shape,
            properties=self.properties,
            attributes=self.attributes,
            constraint=c,
            parameterization=self.parameterization,
        )

    def with_parameterization(self, p: object) -> "COO":
        return COO._new_unsafe(
            row=self.row,
            col=self.col,
            data=self.data,
            shape=self.shape,
            properties=self.properties,
            attributes=self.attributes,
            constraint=self.constraint,
            parameterization=p,
        )

    def to_dense(self) -> torch.Tensor:
        dense = torch.zeros(self.shape, device=self.data.device, dtype=self.data.dtype)
        if self.data.numel():
            dense.index_put_(
                (self.row, self.col),
                self.effective_values(),
                accumulate=True,
            )
        return dense


# ── CSR ───────────────────────────────────────────────────────────────────────


class CSR(SparseTensorBase):
    """CSR sparse matrix.

    indptr: (n_rows+1,), indices/data: (nnz,).
    """

    indptr: torch.Tensor
    indices: torch.Tensor
    data: torch.Tensor

    def __init__(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        data: torch.Tensor,
        shape: tuple[int, int],
        properties: SparseProperties | None = None,
        attributes: EdgeAttributes | None = None,
        constraint: object | None = None,
        parameterization: object | None = None,
    ) -> None:
        shape = _validate_shape(shape)
        indptr = torch.as_tensor(indptr, dtype=torch.long).flatten()
        indices = torch.as_tensor(indices, dtype=torch.long).flatten()
        data = _ensure_float(torch.as_tensor(data).flatten())
        if indptr.shape != (shape[0] + 1,):
            raise ValueError("indptr must have shape (n_rows + 1,).")
        if indptr[0] != 0 or indptr[-1] != indices.numel():
            raise ValueError("indptr bounds do not match indices.")
        counts = indptr[1:] - indptr[:-1]
        if torch.any(counts < 0):
            raise ValueError("indptr must be non-decreasing.")
        if indices.numel() and (indices.min() < 0 or indices.max() >= shape[1]):
            raise ValueError("indices contain an out-of-range index.")
        if indices.numel() != data.numel():
            raise ValueError("indices and data must have the same length.")
        properties = properties or SparseProperties()
        attributes = attributes or EdgeAttributes()
        row = torch.repeat_interleave(
            torch.arange(shape[0], device=indptr.device),
            counts,
        )
        _validate_properties(
            row,
            indices,
            shape,
            properties,
            storage_order="row",
        )
        _validate_attributes(attributes, indices.numel())
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self._row = row
        self.shape = shape
        self.properties = properties
        self.attributes = attributes
        self.constraint = constraint
        self.parameterization = parameterization
        self._cache: dict = {}

    @classmethod
    def _new_unsafe(
        cls,
        *,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        data: torch.Tensor,
        shape: tuple[int, int],
        properties: SparseProperties,
        attributes: EdgeAttributes,
        constraint: object | None,
        parameterization: object | None,
        _row: torch.Tensor | None = None,
    ) -> "CSR":
        obj = object.__new__(cls)
        obj.indptr = indptr
        obj.indices = indices
        obj.data = data
        obj.shape = shape
        obj.properties = properties
        obj.attributes = attributes
        obj.constraint = constraint
        obj.parameterization = parameterization
        obj._cache = {}
        if _row is None:
            counts = indptr[1:] - indptr[:-1]
            _row = torch.repeat_interleave(
                torch.arange(shape[0], device=indptr.device), counts
            )
        obj._row = _row
        return obj

    @classmethod
    def from_edges(
        cls,
        row,
        col,
        data,
        shape: tuple[int, int],
        *,
        properties: SparseProperties | None = None,
        labels: Mapping[str, torch.Tensor] | None = None,
        constraint: object | None = None,
        parameterization: object | None = None,
        device=None,
        dtype=None,
    ) -> "CSR":
        shape = _validate_shape(shape)
        data = _ensure_float(torch.as_tensor(data, device=device, dtype=dtype))
        src, dst, vals, nlabels = _canonicalize_edges(
            torch.as_tensor(row, device=device),
            torch.as_tensor(col, device=device),
            data,
            shape,
            labels,
        )
        counts = torch.bincount(src, minlength=shape[0])
        indptr = torch.cat(
            [torch.zeros(1, device=src.device, dtype=torch.long), counts.cumsum(0)]
        )
        requested = properties or SparseProperties()
        props = replace(requested, sorted_indices=True, unique_edges=True)
        _validate_properties(src, dst, shape, props, storage_order="row")
        return cls._new_unsafe(
            indptr=indptr,
            indices=dst,
            data=vals,
            shape=shape,
            properties=props,
            attributes=EdgeAttributes(nlabels),
            constraint=constraint,
            parameterization=parameterization,
            _row=src,
        )

    @classmethod
    def from_scipy(
        cls,
        matrix: scipy.sparse.sparray,
        *,
        constraint: object | None = None,
        device=None,
        dtype=None,
    ) -> "CSR":
        coo = scipy.sparse.coo_array(matrix)
        coo.sum_duplicates()
        dtype = dtype or torch.get_default_dtype()
        return cls.from_edges(
            coo.row,
            coo.col,
            coo.data,
            shape=coo.shape,
            constraint=constraint,
            device=device,
            dtype=dtype,
        )

    def to_coo(self) -> "COO":
        return COO._new_unsafe(
            row=self._row,
            col=self.indices,
            data=self.data,
            shape=self.shape,
            properties=self.properties,
            attributes=self.attributes,
            constraint=self.constraint,
            parameterization=self.parameterization,
        )

    def to_csr(self) -> "CSR":
        return self

    def to_csc(self) -> "CSC":
        coo = self.to_coo()
        n_post = self.shape[1]
        key = coo.col * self.shape[0] + coo.row
        order = torch.argsort(key, stable=True)
        indices = coo.row[order]
        sorted_dst = coo.col[order]
        sorted_vals = self.data[order]
        attributes = _reorder_attributes(self.attributes, order)
        counts = torch.bincount(sorted_dst, minlength=n_post)
        indptr = torch.cat(
            [
                torch.zeros(1, device=self.indptr.device, dtype=torch.long),
                counts.cumsum(0),
            ]
        )
        return CSC._new_unsafe(
            indptr=indptr,
            indices=indices,
            data=sorted_vals,
            shape=self.shape,
            properties=replace(self.properties, sorted_indices=True),
            attributes=attributes,
            constraint=self.constraint,
            parameterization=self.parameterization,
        )

    def to_ell(self) -> "ELL":
        counts = self.indptr[1:] - self.indptr[:-1]
        if counts.numel() and counts.min() != counts.max():
            raise ValueError(
                "CSR cannot be converted to ELL: row lengths are not uniform."
            )
        coo = self.to_coo()
        n_pre = self.shape[0]
        if coo.row.numel() == 0:
            width = 0
            indices = torch.zeros(
                (n_pre, 0), dtype=torch.long, device=self.indptr.device
            )
            data = torch.zeros(
                (n_pre, 0), dtype=self.data.dtype, device=self.data.device
            )
        else:
            width = int((self.indptr[1:] - self.indptr[:-1]).max().item())
            indices = torch.zeros(
                (n_pre, width), dtype=torch.long, device=self.indptr.device
            )
            data = torch.zeros(
                (n_pre, width), dtype=self.data.dtype, device=self.data.device
            )
            slot = torch.arange(coo.row.numel(), device=self.indptr.device)
            slot = slot - self.indptr[coo.row]
            indices[coo.row, slot] = coo.col
            data[coo.row, slot] = self.data
        props = SparseProperties(
            max_fan_out=width,
            max_fan_in=self.properties.max_fan_in,
            sorted_indices=self.properties.sorted_indices,
            unique_edges=self.properties.unique_edges,
        )
        return ELL._new_unsafe(
            indices=indices,
            data=data,
            width=width,
            shape=self.shape,
            properties=props,
            attributes=self.attributes,
            constraint=self.constraint,
            parameterization=self.parameterization,
        )

    def mm(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]
        x2d = x.reshape(-1, self.shape[0])
        counts = self.indptr[1:] - self.indptr[:-1]
        row = torch.repeat_interleave(
            torch.arange(self.shape[0], device=self.indptr.device),
            counts,
        )
        contributions = x2d[:, row] * self.effective_values()
        result = torch.zeros(
            x2d.shape[0],
            self.shape[1],
            device=x.device,
            dtype=x.dtype,
        )
        result.scatter_add_(
            1,
            self.indices.expand(x2d.shape[0], -1),
            contributions,
        )
        return result.reshape(*leading, self.shape[1])

    def padded_csr_layout(self) -> PaddedCSRLayout:
        cached = self._cache.get("padded_csr")
        if cached is not None:
            return cached
        n_pre = self.shape[0]
        row_length = self.indptr[1:] - self.indptr[:-1]
        row_offset = self.indptr[:-1]
        row_stride = int(row_length.max().item()) if n_pre and row_length.numel() else 0
        indices = torch.zeros(
            (n_pre, row_stride), device=self.indices.device, dtype=torch.long
        )
        if self.indices.numel():
            coo_src = self.to_coo().row
            slot = torch.arange(coo_src.numel(), device=coo_src.device)
            slot = slot - row_offset[coo_src]
            indices[coo_src, slot] = self.indices
        layout = PaddedCSRLayout(
            row_length=row_length,
            row_offset=row_offset,
            indices=indices,
            row_stride=row_stride,
        )
        self._cache["padded_csr"] = layout
        return layout

    def logical_edges(self) -> LogicalEdges:
        n_pre = self.shape[0]
        counts = self.indptr[1:] - self.indptr[:-1]
        row = torch.repeat_interleave(
            torch.arange(n_pre, device=self.indptr.device), counts
        )
        idx = torch.arange(self.indices.numel(), device=self.indptr.device)
        return LogicalEdges(row=row, col=self.indices, storage_index=idx)

    def nnz(self) -> int:
        return int(self.indices.numel())

    def _trainable_tensors(self) -> list[tuple[str, torch.Tensor]]:
        return [("data", self.data)]

    def _structural_tensors(self) -> list[tuple[str, torch.Tensor]]:
        return [("indptr", self.indptr), ("indices", self.indices)]

    def _rebuild(self, parameters: dict, buffers: dict) -> "CSR":
        if self.parameterization is not None:
            new_p = self.parameterization._rebuild(parameters, buffers)
            new_values = buffers["initial_weight"]
        else:
            new_values = parameters["data"]
        return CSR._new_unsafe(
            indptr=buffers["indptr"],
            indices=buffers["indices"],
            data=new_values,
            shape=self.shape,
            properties=self.properties,
            attributes=self.attributes,
            constraint=self.constraint,
            parameterization=new_p if self.parameterization is not None else None,
        )

    def with_constraint(self, c: object) -> "CSR":
        return CSR._new_unsafe(
            indptr=self.indptr,
            indices=self.indices,
            data=self.data,
            shape=self.shape,
            properties=self.properties,
            attributes=self.attributes,
            constraint=c,
            parameterization=self.parameterization,
        )

    def with_parameterization(self, p: object) -> "CSR":
        return CSR._new_unsafe(
            indptr=self.indptr,
            indices=self.indices,
            data=self.data,
            shape=self.shape,
            properties=self.properties,
            attributes=self.attributes,
            constraint=self.constraint,
            parameterization=p,
        )

    def transpose(self) -> "CSC":
        return self.to_csc()

    def to_dense(self) -> torch.Tensor:
        return self.to_coo().to_dense()


# ── CSC ───────────────────────────────────────────────────────────────────────


class CSC(SparseTensorBase):
    """CSC sparse matrix.

    indptr: (n_cols+1,), indices/data: (nnz,).
    """

    indptr: torch.Tensor
    indices: torch.Tensor
    data: torch.Tensor

    def __init__(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        data: torch.Tensor,
        shape: tuple[int, int],
        properties: SparseProperties | None = None,
        attributes: EdgeAttributes | None = None,
        constraint: object | None = None,
        parameterization: object | None = None,
    ) -> None:
        shape = _validate_shape(shape)
        indptr = torch.as_tensor(indptr, dtype=torch.long).flatten()
        indices = torch.as_tensor(indices, dtype=torch.long).flatten()
        data = _ensure_float(torch.as_tensor(data).flatten())
        if indptr.shape != (shape[1] + 1,):
            raise ValueError("indptr must have shape (n_cols + 1,).")
        if indptr[0] != 0 or indptr[-1] != indices.numel():
            raise ValueError("indptr bounds do not match indices.")
        counts = indptr[1:] - indptr[:-1]
        if torch.any(counts < 0):
            raise ValueError("indptr must be non-decreasing.")
        if indices.numel() and (indices.min() < 0 or indices.max() >= shape[0]):
            raise ValueError("indices contain an out-of-range index.")
        if indices.numel() != data.numel():
            raise ValueError("indices and data must have the same length.")
        properties = properties or SparseProperties()
        attributes = attributes or EdgeAttributes()
        col = torch.repeat_interleave(
            torch.arange(shape[1], device=indptr.device),
            counts,
        )
        _validate_properties(
            indices,
            col,
            shape,
            properties,
            storage_order="column",
        )
        _validate_attributes(attributes, indices.numel())
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.shape = shape
        self.properties = properties
        self.attributes = attributes
        self.constraint = constraint
        self.parameterization = parameterization
        self._cache: dict = {}

    @classmethod
    def _new_unsafe(
        cls,
        *,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        data: torch.Tensor,
        shape: tuple[int, int],
        properties: SparseProperties,
        attributes: EdgeAttributes,
        constraint: object | None,
        parameterization: object | None,
    ) -> "CSC":
        obj = object.__new__(cls)
        obj.indptr = indptr
        obj.indices = indices
        obj.data = data
        obj.shape = shape
        obj.properties = properties
        obj.attributes = attributes
        obj.constraint = constraint
        obj.parameterization = parameterization
        obj._cache = {}
        return obj

    @classmethod
    def from_edges(
        cls,
        row,
        col,
        data,
        shape: tuple[int, int],
        *,
        properties: SparseProperties | None = None,
        labels: Mapping[str, torch.Tensor] | None = None,
        constraint: object | None = None,
        parameterization: object | None = None,
        device=None,
        dtype=None,
    ) -> "CSC":
        shape = _validate_shape(shape)
        data = _ensure_float(torch.as_tensor(data, device=device, dtype=dtype))
        src, dst, vals, nlabels = _canonicalize_edges(
            torch.as_tensor(row, device=device),
            torch.as_tensor(col, device=device),
            data,
            shape,
            labels,
        )
        # Re-sort by col (column) order
        key = dst * shape[0] + src
        order = torch.argsort(key, stable=True)
        indices = src[order]
        sorted_dst = dst[order]
        sorted_vals = vals[order]
        sorted_labels = {name: label[order] for name, label in nlabels.items()}
        counts = torch.bincount(sorted_dst, minlength=shape[1])
        indptr = torch.cat(
            [torch.zeros(1, device=src.device, dtype=torch.long), counts.cumsum(0)]
        )
        requested = properties or SparseProperties()
        props = replace(requested, sorted_indices=True, unique_edges=True)
        _validate_properties(
            indices,
            sorted_dst,
            shape,
            props,
            storage_order="column",
        )
        return cls._new_unsafe(
            indptr=indptr,
            indices=indices,
            data=sorted_vals,
            shape=shape,
            properties=props,
            attributes=EdgeAttributes(sorted_labels),
            constraint=constraint,
            parameterization=parameterization,
        )

    @classmethod
    def from_scipy(
        cls,
        matrix: scipy.sparse.sparray,
        *,
        constraint: object | None = None,
        device=None,
        dtype=None,
    ) -> "CSC":
        coo = scipy.sparse.coo_array(matrix)
        coo.sum_duplicates()
        dtype = dtype or torch.get_default_dtype()
        return cls.from_edges(
            coo.row,
            coo.col,
            coo.data,
            shape=coo.shape,
            constraint=constraint,
            device=device,
            dtype=dtype,
        )

    def to_coo(self) -> "COO":
        n_post = self.shape[1]
        counts = self.indptr[1:] - self.indptr[:-1]
        col = torch.repeat_interleave(
            torch.arange(n_post, device=self.indptr.device), counts
        )
        return COO._new_unsafe(
            row=self.indices,
            col=col,
            data=self.data,
            shape=self.shape,
            properties=self.properties,
            attributes=self.attributes,
            constraint=self.constraint,
            parameterization=self.parameterization,
        )

    def to_csr(self) -> "CSR":
        return self.to_coo().to_csr()

    def to_csc(self) -> "CSC":
        return self

    def to_ell(self) -> "ELL":
        return self.to_csr().to_ell()

    def mm(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]
        x2d = x.reshape(-1, self.shape[0])
        counts = self.indptr[1:] - self.indptr[:-1]
        col = torch.repeat_interleave(
            torch.arange(self.shape[1], device=self.indptr.device),
            counts,
        )
        contributions = x2d[:, self.indices] * self.effective_values()
        result = torch.zeros(
            x2d.shape[0],
            self.shape[1],
            device=x.device,
            dtype=x.dtype,
        )
        result.scatter_add_(
            1,
            col.expand(x2d.shape[0], -1),
            contributions,
        )
        return result.reshape(*leading, self.shape[1])

    def logical_edges(self) -> LogicalEdges:
        n_post = self.shape[1]
        counts = self.indptr[1:] - self.indptr[:-1]
        col = torch.repeat_interleave(
            torch.arange(n_post, device=self.indptr.device), counts
        )
        idx = torch.arange(self.indices.numel(), device=self.indptr.device)
        return LogicalEdges(row=self.indices, col=col, storage_index=idx)

    def nnz(self) -> int:
        return int(self.indices.numel())

    def _trainable_tensors(self) -> list[tuple[str, torch.Tensor]]:
        return [("data", self.data)]

    def _structural_tensors(self) -> list[tuple[str, torch.Tensor]]:
        return [("indptr", self.indptr), ("indices", self.indices)]

    def _rebuild(self, parameters: dict, buffers: dict) -> "CSC":
        if self.parameterization is not None:
            new_p = self.parameterization._rebuild(parameters, buffers)
            new_values = buffers["initial_weight"]
        else:
            new_values = parameters["data"]
        return CSC._new_unsafe(
            indptr=buffers["indptr"],
            indices=buffers["indices"],
            data=new_values,
            shape=self.shape,
            properties=self.properties,
            attributes=self.attributes,
            constraint=self.constraint,
            parameterization=new_p if self.parameterization is not None else None,
        )

    def with_constraint(self, c: object) -> "CSC":
        return CSC._new_unsafe(
            indptr=self.indptr,
            indices=self.indices,
            data=self.data,
            shape=self.shape,
            properties=self.properties,
            attributes=self.attributes,
            constraint=c,
            parameterization=self.parameterization,
        )

    def with_parameterization(self, p: object) -> "CSC":
        return CSC._new_unsafe(
            indptr=self.indptr,
            indices=self.indices,
            data=self.data,
            shape=self.shape,
            properties=self.properties,
            attributes=self.attributes,
            constraint=self.constraint,
            parameterization=p,
        )

    def transpose(self) -> "CSR":
        return self.to_csr()

    def to_dense(self) -> torch.Tensor:
        return self.to_coo().to_dense()


# ── ELL ───────────────────────────────────────────────────────────────────────


class ELL(SparseTensorBase):
    """ELL (ELLPACK) sparse matrix. Uniform fan-out per row row.

    indices: (n_rows, width) int64
    data:      (n_rows, width) float
    """

    indices: torch.Tensor
    data: torch.Tensor
    width: int

    def __init__(
        self,
        indices: torch.Tensor,
        data: torch.Tensor,
        shape: tuple[int, int],
        *,
        properties: SparseProperties | None = None,
        attributes: EdgeAttributes | None = None,
        constraint: object | None = None,
        parameterization: object | None = None,
    ) -> None:
        shape = _validate_shape(shape)
        indices = torch.as_tensor(indices, dtype=torch.long)
        data = _ensure_float(torch.as_tensor(data))
        if indices.ndim != 2 or indices.shape[0] != shape[0]:
            raise ValueError("indices must have shape (n_rows, width).")
        if data.shape != indices.shape:
            raise ValueError("data and indices must have identical shapes.")
        if indices.numel() and (indices.min() < 0 or indices.max() >= shape[1]):
            raise ValueError("indices contain an out-of-range index.")
        width = indices.shape[1]
        requested = properties or SparseProperties()
        if requested.max_fan_out is not None and requested.max_fan_out != width:
            raise ValueError("ELL max_fan_out must equal its physical width.")
        properties = replace(requested, max_fan_out=width)
        attributes = attributes or EdgeAttributes()
        row = (
            torch.arange(shape[0], device=indices.device)[:, None]
            .expand_as(indices)
            .reshape(-1)
        )
        col = indices.reshape(-1)
        _validate_properties(
            row,
            col,
            shape,
            properties,
            storage_order="row",
        )
        _validate_attributes(attributes, indices.numel())
        self.indices = indices
        self.data = data
        self.width = width
        self.shape = shape
        self.properties = properties
        self.attributes = attributes
        self.constraint = constraint
        self.parameterization = parameterization
        self._cache: dict = {}

    @classmethod
    def _new_unsafe(
        cls,
        *,
        indices: torch.Tensor,
        data: torch.Tensor,
        width: int,
        shape: tuple[int, int],
        properties: SparseProperties,
        attributes: EdgeAttributes,
        constraint: object | None,
        parameterization: object | None,
    ) -> "ELL":
        obj = object.__new__(cls)
        obj.indices = indices
        obj.data = data
        obj.width = width
        obj.shape = shape
        obj.properties = properties
        obj.attributes = attributes
        obj.constraint = constraint
        obj.parameterization = parameterization
        obj._cache = {}
        return obj

    @classmethod
    def from_indices(
        cls,
        indices,
        data,
        shape: tuple[int, int],
        *,
        properties: SparseProperties | None = None,
        attributes: EdgeAttributes | None = None,
        constraint: object | None = None,
        parameterization: object | None = None,
        device=None,
        dtype=None,
    ) -> "ELL":
        indices = torch.as_tensor(indices, device=device, dtype=torch.long)
        data = _ensure_float(torch.as_tensor(data, device=device, dtype=dtype))
        return cls(
            indices,
            data,
            shape,
            properties=properties,
            attributes=attributes,
            constraint=constraint,
            parameterization=parameterization,
        )

    @classmethod
    def random(
        cls,
        *,
        shape: tuple[int, int],
        fan_out: int,
        value=Normal(0.0, 1.0),
        seed: int = 0,
        allow_self: bool = True,
        constraint: object | None = None,
        device=None,
        dtype=None,
    ) -> "ELL":
        shape = _validate_shape(shape)
        max_conn = shape[1] - (0 if allow_self or shape[0] != shape[1] else 1)
        if fan_out < 0 or fan_out > max_conn:
            raise ValueError("fan_out is incompatible with the matrix shape.")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        rows = []
        for src in range(shape[0]):
            candidates = torch.randperm(shape[1], generator=generator, device=device)
            if not allow_self and shape[0] == shape[1]:
                candidates = candidates[candidates != src]
            rows.append(candidates[:fan_out])
        if rows:
            indices = torch.stack(rows)
        else:
            indices = torch.empty((0, fan_out), dtype=torch.long, device=device)
        vals = initialize_values(
            value,
            indices.shape,
            generator=generator,
            device=device,
            dtype=dtype or torch.get_default_dtype(),
        )
        props = SparseProperties(max_fan_out=fan_out)
        return cls._new_unsafe(
            indices=indices,
            data=vals,
            width=fan_out,
            shape=shape,
            properties=props,
            attributes=EdgeAttributes(),
            constraint=constraint,
            parameterization=None,
        )

    def to_coo(self) -> "COO":
        n_pre = self.shape[0]
        row = (
            torch.arange(n_pre, device=self.indices.device)[:, None]
            .expand_as(self.indices)
            .reshape(-1)
        )
        col = self.indices.reshape(-1)
        data = self.data.reshape(-1)
        return COO._new_unsafe(
            row=row,
            col=col,
            data=data,
            shape=self.shape,
            properties=self.properties,
            attributes=self.attributes,
            constraint=self.constraint,
            parameterization=self.parameterization,
        )

    def to_csr(self) -> "CSR":
        return self.to_coo().to_csr()

    def to_csc(self) -> "CSC":
        return self.to_coo().to_csc()

    def to_ell(self) -> "ELL":
        return self

    def mm(self, x: torch.Tensor) -> torch.Tensor:
        vals = self.effective_values()
        return _ell_mm(self.indices, vals.reshape(self.indices.shape), self.shape, x)

    def padded_csr_layout(self) -> PaddedCSRLayout:
        cached = self._cache.get("padded_csr")
        if cached is not None:
            return cached
        n_pre = self.shape[0]
        row_length = torch.full(
            (n_pre,), self.width, device=self.indices.device, dtype=torch.long
        )
        row_offset = (
            torch.arange(n_pre, device=self.indices.device, dtype=torch.long)
            * self.width
        )
        layout = PaddedCSRLayout(
            row_length=row_length,
            row_offset=row_offset,
            indices=self.indices,
            row_stride=self.width,
        )
        self._cache["padded_csr"] = layout
        return layout

    def logical_edges(self) -> LogicalEdges:
        n_pre = self.shape[0]
        row = (
            torch.arange(n_pre, device=self.indices.device)[:, None]
            .expand_as(self.indices)
            .reshape(-1)
        )
        col = self.indices.reshape(-1)
        idx = torch.arange(row.numel(), device=self.indices.device)
        return LogicalEdges(row=row, col=col, storage_index=idx)

    def nnz(self) -> int:
        return int(self.indices.numel())

    def _trainable_tensors(self) -> list[tuple[str, torch.Tensor]]:
        return [("data", self.data)]

    def _structural_tensors(self) -> list[tuple[str, torch.Tensor]]:
        return [("indices", self.indices)]

    def _rebuild(self, parameters: dict, buffers: dict) -> "ELL":
        if self.parameterization is not None:
            new_p = self.parameterization._rebuild(parameters, buffers)
            new_values = buffers["initial_weight"]
        else:
            new_values = parameters["data"]
        return ELL._new_unsafe(
            indices=buffers["indices"],
            data=new_values,
            width=self.width,
            shape=self.shape,
            properties=self.properties,
            attributes=self.attributes,
            constraint=self.constraint,
            parameterization=new_p if self.parameterization is not None else None,
        )

    def with_constraint(self, c: object) -> "ELL":
        return ELL._new_unsafe(
            indices=self.indices,
            data=self.data,
            width=self.width,
            shape=self.shape,
            properties=self.properties,
            attributes=self.attributes,
            constraint=c,
            parameterization=self.parameterization,
        )

    def with_parameterization(self, p: object) -> "ELL":
        return ELL._new_unsafe(
            indices=self.indices,
            data=self.data,
            width=self.width,
            shape=self.shape,
            properties=self.properties,
            attributes=self.attributes,
            constraint=self.constraint,
            parameterization=p,
        )

    def to_dense(self) -> torch.Tensor:
        return self.to_coo().to_dense()


# ── SparseTensor (auto-selecting) ─────────────────────────────────────────────


def _select_format(
    row: torch.Tensor,
    col: torch.Tensor,
    data: torch.Tensor,
    shape: tuple[int, int],
    properties: SparseProperties | None,
    attributes: EdgeAttributes | None,
    constraint: object | None,
) -> SparseTensorBase:
    properties = properties or SparseProperties(
        sorted_indices=True,
        unique_edges=True,
    )
    _validate_properties(
        row,
        col,
        shape,
        properties,
        storage_order="row",
    )
    if properties is not None and properties.max_fan_out is not None:
        fan_out = properties.max_fan_out
        n_pre = shape[0]
        indices = col.reshape(n_pre, fan_out)
        vals = data.reshape(n_pre, fan_out)
        return ELL._new_unsafe(
            indices=indices,
            data=vals,
            width=fan_out,
            shape=shape,
            properties=properties,
            attributes=attributes or EdgeAttributes(),
            constraint=constraint,
            parameterization=None,
        )
    if properties is not None and properties.max_fan_in is not None:
        # Use CSC for fixed fan-in
        n_post = shape[1]
        key = col * shape[0] + row
        order = torch.argsort(key, stable=True)
        indices = row[order]
        sorted_vals = data[order]
        sorted_attributes = _reorder_attributes(
            attributes or EdgeAttributes(),
            order,
        )
        counts = torch.bincount(col[order], minlength=n_post)
        indptr = torch.cat(
            [torch.zeros(1, device=row.device, dtype=torch.long), counts.cumsum(0)]
        )
        return CSC._new_unsafe(
            indptr=indptr,
            indices=indices,
            data=sorted_vals,
            shape=shape,
            properties=properties,
            attributes=sorted_attributes,
            constraint=constraint,
            parameterization=None,
        )
    # Default: CSR
    counts = torch.bincount(row, minlength=shape[0])
    indptr = torch.cat(
        [torch.zeros(1, device=row.device, dtype=torch.long), counts.cumsum(0)]
    )
    return CSR._new_unsafe(
        indptr=indptr,
        indices=col,
        data=data,
        shape=shape,
        properties=properties,
        attributes=attributes or EdgeAttributes(),
        constraint=constraint,
        parameterization=None,
    )


class SparseTensor(SparseTensorBase):
    """Auto-selecting sparse matrix wrapping a heuristically chosen format."""

    _inner: SparseTensorBase
    _cache_ref: dict

    def __init__(self, _inner: SparseTensorBase) -> None:
        self._inner = _inner
        self._cache_ref: dict = {}

    @classmethod
    def _new_unsafe(cls, inner: SparseTensorBase) -> "SparseTensor":
        obj = object.__new__(cls)
        obj._inner = inner
        return obj

    @classmethod
    def from_edges(
        cls,
        row,
        col,
        data,
        shape: tuple[int, int],
        *,
        properties: SparseProperties | None = None,
        attributes: EdgeAttributes | None = None,
        labels: Mapping[str, torch.Tensor] | None = None,
        constraint: object | None = None,
        device=None,
        dtype=None,
    ) -> "SparseTensor":
        shape = _validate_shape(shape)
        data = _ensure_float(torch.as_tensor(data, device=device, dtype=dtype))
        src, dst, vals, nlabels = _canonicalize_edges(
            torch.as_tensor(row, device=device),
            torch.as_tensor(col, device=device),
            data,
            shape,
            labels,
        )
        attrs = EdgeAttributes({**nlabels, **(attributes.labels if attributes else {})})
        inner = _select_format(src, dst, vals, shape, properties, attrs, constraint)
        return cls._new_unsafe(inner)

    @property
    def selected_format(self) -> type:
        return type(self._inner)

    @property
    def shape(self) -> tuple[int, int]:
        return self._inner.shape

    @property
    def properties(self) -> SparseProperties:
        return self._inner.properties

    @property
    def attributes(self) -> EdgeAttributes:
        return self._inner.attributes

    @property
    def constraint(self) -> object | None:
        return self._inner.constraint

    @property
    def parameterization(self) -> object | None:
        return self._inner.parameterization

    @property
    def _cache(self) -> dict:
        return self._inner._cache

    def mm(self, x: torch.Tensor) -> torch.Tensor:
        return self._inner.mm(x)

    def event_mm(self, events) -> torch.Tensor:
        return self._inner.event_mm(events)

    def to_coo(self) -> "COO":
        return self._inner.to_coo()

    def to_csr(self) -> "CSR":
        return self._inner.to_csr()

    def to_csc(self) -> "CSC":
        return self._inner.to_csc()

    def to_ell(self) -> "ELL":
        return self._inner.to_ell()

    def logical_edges(self) -> LogicalEdges:
        return self._inner.logical_edges()

    def nnz(self) -> int:
        return self._inner.nnz()

    def effective_values(self) -> torch.Tensor:
        return self._inner.effective_values()

    def prepare_as(self, format: type, **kwargs) -> None:
        self._inner.prepare_as(format, **kwargs)

    def as_bsr(self):
        return self._inner.as_bsr()

    def _trainable_tensors(self) -> list[tuple[str, torch.Tensor]]:
        return self._inner._trainable_tensors()

    def _structural_tensors(self) -> list[tuple[str, torch.Tensor]]:
        return self._inner._structural_tensors()

    def _rebuild(self, parameters: dict, buffers: dict) -> "SparseTensor":
        new_inner = self._inner._rebuild(parameters, buffers)
        return SparseTensor._new_unsafe(new_inner)

    def with_constraint(self, c: object) -> "SparseTensor":
        return SparseTensor._new_unsafe(self._inner.with_constraint(c))

    def with_parameterization(self, p: object) -> "SparseTensor":
        return SparseTensor._new_unsafe(self._inner.with_parameterization(p))

    def to_dense(self) -> torch.Tensor:
        return self._inner.to_dense()  # type: ignore[attr-defined]


# ── PyTree registration ───────────────────────────────────────────────────────


def _coo_flatten(W: COO):
    return [W.data], (
        W.row,
        W.col,
        W.shape,
        W.properties,
        W.attributes,
        W.constraint,
        W.parameterization,
    )


def _coo_unflatten(leaves, context):
    row, col, shape, properties, attributes, constraint, parameterization = context
    return COO._new_unsafe(
        row=row,
        col=col,
        data=leaves[0],
        shape=shape,
        properties=properties,
        attributes=attributes,
        constraint=constraint,
        parameterization=parameterization,
    )


def _csr_flatten(W: CSR):
    return [W.data], (
        W.indptr,
        W.indices,
        W.shape,
        W.properties,
        W.attributes,
        W.constraint,
        W.parameterization,
    )


def _csr_unflatten(leaves, context):
    (
        indptr,
        indices,
        shape,
        properties,
        attributes,
        constraint,
        parameterization,
    ) = context
    return CSR._new_unsafe(
        indptr=indptr,
        indices=indices,
        data=leaves[0],
        shape=shape,
        properties=properties,
        attributes=attributes,
        constraint=constraint,
        parameterization=parameterization,
    )


def _csc_flatten(W: CSC):
    return [W.data], (
        W.indptr,
        W.indices,
        W.shape,
        W.properties,
        W.attributes,
        W.constraint,
        W.parameterization,
    )


def _csc_unflatten(leaves, context):
    (
        indptr,
        indices,
        shape,
        properties,
        attributes,
        constraint,
        parameterization,
    ) = context
    return CSC._new_unsafe(
        indptr=indptr,
        indices=indices,
        data=leaves[0],
        shape=shape,
        properties=properties,
        attributes=attributes,
        constraint=constraint,
        parameterization=parameterization,
    )


def _ell_flatten(W: ELL):
    return [W.data], (
        W.indices,
        W.width,
        W.shape,
        W.properties,
        W.attributes,
        W.constraint,
        W.parameterization,
    )


def _ell_unflatten(leaves, context):
    indices, width, shape, properties, attributes, constraint, parameterization = (
        context
    )
    return ELL._new_unsafe(
        indices=indices,
        data=leaves[0],
        width=width,
        shape=shape,
        properties=properties,
        attributes=attributes,
        constraint=constraint,
        parameterization=parameterization,
    )


def _sparsetensor_flatten(W: SparseTensor):
    inner_leaves, inner_treespec = pytree.tree_flatten(W._inner)
    return inner_leaves, inner_treespec


def _sparsetensor_unflatten(leaves, inner_treespec):
    inner = pytree.tree_unflatten(leaves, inner_treespec)
    return SparseTensor._new_unsafe(inner)


pytree.register_pytree_node(COO, _coo_flatten, _coo_unflatten)
pytree.register_pytree_node(CSR, _csr_flatten, _csr_unflatten)
pytree.register_pytree_node(CSC, _csc_flatten, _csc_unflatten)
pytree.register_pytree_node(ELL, _ell_flatten, _ell_unflatten)
pytree.register_pytree_node(
    SparseTensor, _sparsetensor_flatten, _sparsetensor_unflatten
)
