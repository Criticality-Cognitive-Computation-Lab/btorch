from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseProperties:
    max_fan_out: int | None = None
    max_fan_in: int | None = None
    sorted_indices: bool = False
    unique_edges: bool = True


class EdgeAttributes:
    """Uniform container for per-edge non-weight metadata."""

    __slots__ = ("_labels",)

    def __init__(self, labels: dict[str, torch.Tensor] | None = None) -> None:
        self._labels: dict[str, torch.Tensor] = dict(labels) if labels else {}

    @property
    def labels(self) -> dict[str, torch.Tensor]:
        return self._labels

    @classmethod
    def empty(cls) -> "EdgeAttributes":
        return cls()


@dataclass(frozen=True)
class LogicalEdges:
    row: torch.Tensor
    col: torch.Tensor
    storage_index: torch.Tensor
    local_index: torch.Tensor | None = None
