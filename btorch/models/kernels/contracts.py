"""Declare immutable cell options and shared sequence history."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class PoissonOptions:
    """Describe immutable :class:`PoissonRandomSpike` behavior."""

    spiking: bool = True
    tau: float = 1.0
    rho: float = 1.0
    leak: float = 0.0
    k: float = 1.0


@dataclass(frozen=True)
class CellSpec:
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
