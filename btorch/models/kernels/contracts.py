"""Declare immutable cell options and shared sequence history."""

from dataclasses import dataclass

import torch
from torch import Tensor


def pointwise_dtype(reference: Tensor, *operands: Tensor) -> torch.dtype:
    """Resolve floating model precision without widening low-precision
    inputs."""
    dtype = reference.dtype
    for value in operands:
        if value.is_floating_point():
            dtype = torch.promote_types(dtype, value.dtype)
    return dtype


@dataclass(frozen=True)
class PoissonOptions:
    """Describe immutable :class:`PoissonRandomSpike` behavior."""

    spiking: bool = True
    tau: float = 1.0
    rho: float = 1.0
    leak: float = 0.0
    k: float = 1.0


@dataclass(frozen=True)
class NeuronSpec:
    """Describe scalar cell choices without hiding tensor parameters.

    Args:
        model: Named neuron or PSC template.
        dt: Runtime scalar physical integration timestep; never a specialization key.
        hard_reset: Reset voltage directly instead of subtracting the gap.
        detach_reset: Stop only the spike contribution to reset derivatives.
        surrogate: A supported built-in deterministic surrogate name, or
            ``"poisson"`` when ``poisson`` supplies stochastic options.
        alpha: Non-trainable surrogate inverse half-width.
        damping: Non-trainable surrogate gradient damping.
        refractory: Include a refractory state and duration parameter.
        poisson: Stochastic surrogate options, or None for deterministic
            surrogates.
        surrogate_width: Optional explicit derivative cutoff in normalized
            surrogate-input units, independent of its alpha/damping.
    """

    model: str
    dt: float | Tensor = 1.0
    hard_reset: bool = False
    detach_reset: bool = False
    surrogate: str = "sigmoid"
    alpha: float = 2.0
    damping: float = 1.0
    refractory: bool = False
    poisson: PoissonOptions | None = None
    surrogate_width: float | None = None


def surrogate_support(spec: NeuronSpec) -> float | None:
    """Return known compact support in the surrogate's normalized input
    units."""
    natural = (
        2.0 / abs(spec.alpha) if spec.surrogate == "triangle" and spec.alpha else None
    )
    if spec.surrogate_width is None:
        return natural
    return (
        spec.surrogate_width if natural is None else min(spec.surrogate_width, natural)
    )


@dataclass(frozen=True)
class HistoryPlan:
    """Share records requested by users and needed by backward.

    Args:
        requested: Post-step state names or charged-state names to return.
        backward: Additional records needed by the sequence reverse pass.
    """

    requested: tuple[str, ...] = ()
    backward: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """Return stable unique record names without copying tensors."""
        return tuple(dict.fromkeys((*self.requested, *self.backward)))
