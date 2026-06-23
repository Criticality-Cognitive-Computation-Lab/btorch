from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import torch

from .events import EventRepresentation
from .properties import EdgeAttributes, LogicalEdges, SparseProperties


if TYPE_CHECKING:
    from .matrices import COO, CSC, CSR, ELL


class SparseTensorBase(ABC):
    shape: tuple[int, int]
    properties: SparseProperties
    attributes: EdgeAttributes
    constraint: object | None
    parameterization: object | None
    _cache: dict

    # --- Matrix operations ---

    def mm(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not implement mm()")

    def event_mm(self, events: EventRepresentation) -> torch.Tensor:
        from .operations import event_sparse_mm

        return event_sparse_mm(self, events)

    # --- Format conversion ---

    def to_coo(self) -> "COO":
        raise NotImplementedError(f"{type(self).__name__} does not implement to_coo()")

    def to_csr(self) -> "CSR":
        raise NotImplementedError(f"{type(self).__name__} does not implement to_csr()")

    def to_csc(self) -> "CSC":
        raise NotImplementedError(f"{type(self).__name__} does not implement to_csc()")

    def to_ell(self) -> "ELL":
        raise NotImplementedError(f"{type(self).__name__} does not implement to_ell()")

    def to_bsr(self, block_shape: tuple[int, int]) -> "object":
        raise NotImplementedError("BSR format is not yet implemented.")

    # --- Inspection ---

    def logical_edges(self) -> LogicalEdges:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement logical_edges()"
        )

    def nnz(self) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not implement nnz()")

    # --- Value access ---

    def effective_values(self) -> torch.Tensor:
        if self.parameterization is not None:
            return self.parameterization.effective_values()
        return self.data  # type: ignore[attr-defined]

    # --- Constraint / parameterization binding ---

    def with_constraint(self, c: object) -> "SparseTensorBase":
        raise NotImplementedError(
            f"{type(self).__name__} does not implement with_constraint()"
        )

    def with_parameterization(self, p: object) -> "SparseTensorBase":
        raise NotImplementedError(
            f"{type(self).__name__} does not implement with_parameterization()"
        )

    # --- Format cache ---

    def prepare_as(self, format: type, **kwargs) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement prepare_as()"
        )

    def as_bsr(self) -> "object":
        raise NotImplementedError("BSR format is not yet implemented.")

    # --- SparseParam integration ---

    def _trainable_tensors(self) -> list[tuple[str, torch.Tensor]]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _trainable_tensors()"
        )

    def _structural_tensors(self) -> list[tuple[str, torch.Tensor]]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _structural_tensors()"
        )

    def _rebuild(self, parameters: dict, buffers: dict) -> "SparseTensorBase":
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _rebuild()"
        )
