from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .properties import LogicalEdges


if TYPE_CHECKING:
    from .base import SparseTensorBase


# ── Constraint types ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NonNegative:
    """All trainable values >= 0.

    Projection: clamp to zero.
    """


@dataclass(frozen=True)
class GroupedWeights:
    """Edges in the same group share one scalar.

    group_ids: (nnz,) int64.
    """

    group_ids: torch.Tensor


@dataclass(frozen=True)
class Symmetric:
    """W[i,j] = W[j,i].

    Requires reciprocal edges. Projection: average pairs.
    """


@dataclass(frozen=True)
class Bounded:
    """Clamp to [min_val, max_val]."""

    min_val: float
    max_val: float


@dataclass(frozen=True)
class BlockUnitNorm:
    """Each group of edges has unit L2 norm.

    block_ids: (nnz,) int64.
    """

    block_ids: torch.Tensor


# ── PreparedProjection types ──────────────────────────────────────────────────


@dataclass(frozen=True)
class GroupProjection:
    parameter_index: torch.Tensor  # (nnz,) int64 edge → group
    group_scale: torch.Tensor  # (n_groups,) float
    nonneg_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class NonNegativeProjection:
    pass  # projection is clamp_min(0)


@dataclass(frozen=True)
class SymmetricProjection:
    edge_pair_index: torch.Tensor  # (nnz,) int64 each edge → reciprocal index


@dataclass(frozen=True)
class BoundedProjection:
    min_val: float
    max_val: float


# ── prepare_projection ────────────────────────────────────────────────────────


def prepare_projection(constraint: object, edges: LogicalEdges) -> object:
    row = edges.row
    col = edges.col
    nnz = row.numel()
    device = row.device

    if isinstance(constraint, NonNegative):
        return NonNegativeProjection()

    if isinstance(constraint, GroupedWeights):
        group_ids = constraint.group_ids.to(device=device, dtype=torch.long)
        n_groups = int(group_ids.max().item()) + 1 if nnz else 0
        group_scale = torch.ones(n_groups, device=device)
        return GroupProjection(parameter_index=group_ids, group_scale=group_scale)

    if isinstance(constraint, Symmetric):
        pairs = list(zip(row.cpu().tolist(), col.cpu().tolist()))
        edge_set = {(s, d): i for i, (s, d) in enumerate(pairs)}
        reciprocal_indices = []
        for s, d in pairs:
            if (d, s) not in edge_set:
                raise ValueError("Symmetric constraint requires every reciprocal edge.")
            reciprocal_indices.append(edge_set[(d, s)])
        return SymmetricProjection(
            edge_pair_index=torch.tensor(
                reciprocal_indices, device=device, dtype=torch.long
            )
        )

    if isinstance(constraint, Bounded):
        return BoundedProjection(min_val=constraint.min_val, max_val=constraint.max_val)

    raise NotImplementedError(
        f"Constraint type {type(constraint).__name__} is not yet implemented."
    )


# ── project ───────────────────────────────────────────────────────────────────


def project(W: "SparseTensorBase", prepared: object) -> "SparseTensorBase":
    values = W.effective_values()

    if isinstance(prepared, NonNegativeProjection):
        new_values = values.clamp_min(0)

    elif isinstance(prepared, GroupProjection):
        group_id = prepared.parameter_index.to(device=values.device)
        scale = prepared.group_scale.to(device=values.device, dtype=values.dtype)
        n_groups = int(group_id.max().item()) + 1 if values.numel() else 0
        numerator = torch.zeros(n_groups, device=values.device, dtype=values.dtype)
        denominator = torch.zeros_like(numerator)
        numerator.scatter_add_(0, group_id, scale * values)
        denominator.scatter_add_(0, group_id, scale.square())
        coeff = numerator / denominator.clamp_min(torch.finfo(values.dtype).eps)
        if prepared.nonneg_mask is not None:
            nonneg = prepared.nonneg_mask.to(device=values.device)
            coeff[nonneg] = coeff[nonneg].clamp_min(0)
        new_values = scale * coeff[group_id]

    elif isinstance(prepared, SymmetricProjection):
        pair_idx = prepared.edge_pair_index.to(device=values.device)
        new_values = (values + values[pair_idx]) * 0.5

    elif isinstance(prepared, BoundedProjection):
        new_values = values.clamp(prepared.min_val, prepared.max_val)

    else:
        raise TypeError(f"Unknown PreparedProjection type: {type(prepared)}")

    return _replace_values(W, new_values)


def _replace_values(
    W: "SparseTensorBase",
    new_values: torch.Tensor,
) -> "SparseTensorBase":
    from .matrices import COO, CSC, CSR, ELL, SparseTensor

    if isinstance(W, SparseTensor):
        return SparseTensor._new_unsafe(_replace_values(W._inner, new_values))

    if isinstance(W, COO):
        return COO._new_unsafe(
            row=W.row,
            col=W.col,
            data=new_values,
            shape=W.shape,
            properties=W.properties,
            attributes=W.attributes,
            constraint=W.constraint,
            parameterization=W.parameterization,
        )
    if isinstance(W, CSR):
        return CSR._new_unsafe(
            indptr=W.indptr,
            indices=W.indices,
            data=new_values,
            shape=W.shape,
            properties=W.properties,
            attributes=W.attributes,
            constraint=W.constraint,
            parameterization=W.parameterization,
        )
    if isinstance(W, CSC):
        return CSC._new_unsafe(
            indptr=W.indptr,
            indices=W.indices,
            data=new_values,
            shape=W.shape,
            properties=W.properties,
            attributes=W.attributes,
            constraint=W.constraint,
            parameterization=W.parameterization,
        )
    if isinstance(W, ELL):
        return ELL._new_unsafe(
            indices=W.indices,
            data=new_values.reshape(W.indices.shape),
            width=W.width,
            shape=W.shape,
            properties=W.properties,
            attributes=W.attributes,
            constraint=W.constraint,
            parameterization=W.parameterization,
        )
    raise TypeError(f"Cannot replace values on {type(W).__name__}")


# ── constrain (functional, lazy-cached) ──────────────────────────────────────


def constrain(W: "SparseTensorBase") -> "SparseTensorBase":
    if W.constraint is None:
        return W
    if "projection" not in W._cache:
        W._cache["projection"] = prepare_projection(W.constraint, W.logical_edges())
    return project(W, W._cache["projection"])
