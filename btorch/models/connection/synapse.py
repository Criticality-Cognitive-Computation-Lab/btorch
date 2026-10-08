"""Description of what a set of connections does to its targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .weight import ConstantWeight, EdgeWeight, Weight


@dataclass
class Synapse:
    """Synaptic properties of the edges of a connection.

    Weight, receptor, delay and plasticity are orthogonal: any combination is
    valid, and none of them prescribes how the connection is stored or
    executed. Per-edge arrays are aligned with the edges as supplied to the
    connection (the stored entries of the matrix, or the rule's edge list).

    Args:
        weight: ``None`` (use the matrix values, trainable), a number (the
            same fixed weight everywhere), a tensor (fixed per-edge weights),
            an ``nn.Parameter`` (trainable per-edge weights) or any
            :class:`~btorch.models.connection.Weight` module such as
            :class:`~btorch.models.connection.ConstrainedWeight`.
        delay: Transmission delay in time steps: ``None`` (no delay axis), an
            int (same delay everywhere) or a ``[n_edge]`` integer tensor.
        receptor: Receptor channel on the target: ``None`` (single channel),
            an int, or a ``[n_edge]`` integer tensor.
        n_delay: Number of delay bins (default: largest delay + 1).
        n_receptor: Number of receptor channels (default: largest id + 1).
        dale: Enforce Dale's law on the weights. For a :class:`Weight`
            module this switches its own ``dale`` flag on.
        plasticity: Reserved for online plasticity rules.

    Examples:
        >>> Synapse(weight=1.0)                                # doctest: +SKIP
        >>> Synapse(weight=nn.Parameter(w), delay=d, receptor=r)
        >>> Synapse(weight=ConstrainedWeight(group=g), dale=True)
    """

    weight: None | float | Tensor | Weight = None
    delay: None | int | Tensor = None
    receptor: None | int | Tensor = None
    n_delay: int | None = None
    n_receptor: int | None = None
    dale: bool = False
    plasticity: Any = None

    def make_weight(self) -> Weight:
        """Build the weight module described by :attr:`weight`."""
        w = self.weight
        if isinstance(w, Weight):
            if self.dale:
                if not hasattr(w, "dale"):
                    raise ValueError(f"{type(w).__name__} does not support Dale's law.")
                w.dale = True
            return w
        if w is None:
            return EdgeWeight(None, trainable=True, dale=self.dale)
        if isinstance(w, nn.Parameter):
            return EdgeWeight(w.data, trainable=w.requires_grad, dale=self.dale)
        if isinstance(w, Tensor) and w.ndim > 0:
            return EdgeWeight(w, trainable=False, dale=self.dale)
        return ConstantWeight(float(w))

    def routing(self, n_edge: int, device=None) -> tuple[Tensor, int, Tensor, int]:
        """Per-edge ``(receptor, n_receptor, delay, n_delay)`` as tensors."""

        def expand(value, count, what):
            if value is None:
                ids = torch.zeros(n_edge, dtype=torch.long, device=device)
                return ids, 1 if count is None else count
            ids = torch.as_tensor(value, device=device)
            if ids.is_floating_point():
                raise TypeError(f"{what} ids must be integers (time steps / channels).")
            ids = ids.to(torch.long).expand(n_edge).contiguous()
            if ids.numel() and int(ids.min()) < 0:
                raise ValueError(f"{what} ids must be non-negative.")
            largest = int(ids.max()) + 1 if ids.numel() else 1
            if count is None:
                count = largest
            elif largest > count:
                raise ValueError(
                    f"{what} id {largest - 1} is out of range for {count}."
                )
            return ids, count

        receptor, n_receptor = expand(self.receptor, self.n_receptor, "receptor")
        delay, n_delay = expand(self.delay, self.n_delay, "delay")
        return receptor, n_receptor, delay, n_delay
