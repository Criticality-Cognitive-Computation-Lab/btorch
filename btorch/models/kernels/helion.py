"""State for the experimental HELION recurrent execution strategy.

The renderer and recurrent scheduler remain separate from this state holder.
Keeping the delay ring in a :class:`~btorch.models.base.MemoryModule` makes
chunk continuation, reset and serialization use the same state protocol as
the existing recurrent models.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..base import MemoryModule


class FutureCurrent(MemoryModule):
    """Store delay-indexed synaptic drive for a fused recurrent step.

    ``buffer[d, neuron, receptor]`` contains drive scheduled for a future
    timestep. ``cursor`` is the slot at the start of the current recurrent
    chunk. A persistent kernel derives each step's slot from
    ``(cursor + step_offset) % n_delay`` and commits the cursor once after the
    chunk. The neuron phase reads the old PSC state; propagation writes into
    future slots; the PSC phase consumes and clears the current slot.

    Args:
        n_neuron: Number of destination neurons.
        n_delay: Number of delay slots in the ring.
        n_receptor: Number of independent PSC receptors.
        dtype: Storage dtype for the drive ring. If omitted, the module uses
            the dtype supplied by :meth:`~btorch.models.base.MemoryModule.reset`
            or :meth:`~btorch.models.base.MemoryModule.init_state`.

    Notes:
        The ring is semantic recurrent state and is persistent in the module
        state dict. The fused plan must invalidate or reconstruct it when
        delayed weights or topology change while pending arrivals exist.
    """

    def __init__(
        self,
        n_neuron: int,
        n_delay: int = 1,
        n_receptor: int = 1,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dimensions = (int(n_delay), int(n_neuron), int(n_receptor))
        if any(value < 1 for value in dimensions):
            raise ValueError(
                "FutureCurrent dimensions must be positive: "
                f"n_delay={n_delay}, n_neuron={n_neuron}, "
                f"n_receptor={n_receptor}"
            )
        self.n_neuron = dimensions[1]
        self.n_delay = dimensions[0]
        self.n_receptor = dimensions[2]
        self.register_memory(
            "buffer",
            0.0,
            dimensions,
            dtype=dtype,
            persistent=True,
        )
        self.register_memory(
            "cursor",
            0,
            (1,),
            dtype=torch.int64,
            persistent=True,
        )

    @property
    def current(self) -> Tensor:
        """Return the complete delay ring."""
        return self.buffer

    @property
    def current_slot(self) -> Tensor:
        """Return the chunk-base cursor tensor used by compiled plans."""
        return self.cursor

    def slot(self, step_offset: int | Tensor) -> Tensor:
        """Return a delay-ring slot relative to the current chunk base.

        Args:
            step_offset: A Python integer or scalar tensor representing the
                offset within the active chunk.

        Returns:
            A scalar int64 tensor suitable for indexing the ring. The method
            is side-effect free and therefore safe to call from compiled
            recurrent setup code.
        """
        return torch.remainder(self.cursor + step_offset, self.n_delay)

    @torch.no_grad()
    def commit(self, steps: int | Tensor) -> None:
        """Commit one completed chunk by advancing its base cursor once."""
        if isinstance(steps, Tensor):
            if steps.numel() != 1 or steps.dtype not in (torch.int32, torch.int64):
                raise ValueError("steps must be a scalar integer tensor")
            value = steps.to(device=self.cursor.device, dtype=self.cursor.dtype)
        else:
            value = int(steps)
        if isinstance(value, int) and value < 0:
            raise ValueError("steps must be non-negative")
        self.cursor.add_(value).remainder_(self.n_delay)


__all__ = ["FutureCurrent"]
