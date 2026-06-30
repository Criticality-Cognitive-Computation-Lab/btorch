from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BinaryEvents:
    """Explicit dense binary-event representation."""

    values: torch.Tensor
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.values.ndim < 1:
            raise ValueError("BinaryEvents values must have at least one dimension.")

    @property
    def size(self) -> int:
        return self.values.shape[-1]


@dataclass(frozen=True)
class SpikeListEvents:
    """Compact event indices with one padded list per batch item."""

    count: torch.Tensor
    indices: torch.Tensor
    size: int

    def __post_init__(self) -> None:
        if self.count.ndim != 1:
            raise ValueError("count must have shape (batch_size,).")
        if self.indices.ndim != 2:
            raise ValueError("indices must have shape (batch_size, capacity).")
        if self.indices.shape[0] != self.count.shape[0]:
            raise ValueError("count and indices batch dimensions must match.")
        if self.size <= 0:
            raise ValueError("size must be positive.")

    def to_dense(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = dtype or torch.get_default_dtype()
        dense = torch.zeros(
            (self.count.shape[0], self.size),
            device=self.indices.device,
            dtype=dtype,
        )
        slots = torch.arange(self.indices.shape[1], device=self.indices.device)
        valid = slots[None, :] < self.count[:, None]
        batch = torch.arange(self.count.shape[0], device=self.indices.device)
        batch = batch[:, None].expand_as(self.indices)
        dense[batch[valid], self.indices[valid]] = 1
        return dense


EventRepresentation = BinaryEvents | SpikeListEvents
