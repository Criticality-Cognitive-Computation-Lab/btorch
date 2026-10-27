"""Give a deterministic spike surrogate explicit compact gradient support."""

import math

import torch

from .base import SurrogateFunctionBase


class Truncated(SurrogateFunctionBase):
    """Limit a surrogate derivative to a symmetric input window.

    The forward spike function is unchanged. The derivative is the wrapped
    surrogate inside ``abs(x) < width`` and exactly zero elsewhere. The width
    uses the surrogate input's units; neuron models normally pass normalized
    distance from their firing threshold, rather than raw voltage.

    Args:
        surrogate: Deterministic surrogate using the shared spike forward.
        width: Finite positive half-width of the derivative support.
    """

    def __init__(self, surrogate: SurrogateFunctionBase, width: float):
        # Parameters belong to the wrapped module, not a duplicated set of
        # alpha/damping attributes that could diverge after a user update.
        torch.nn.Module.__init__(self)
        if (
            not isinstance(surrogate, SurrogateFunctionBase)
            or type(surrogate).forward is not SurrogateFunctionBase.forward
        ):
            raise TypeError("Truncated requires the shared deterministic spike forward")
        if isinstance(width, bool) or not isinstance(width, (float, int)):
            raise TypeError("width must be a finite positive scalar")
        if not math.isfinite(width) or width <= 0:
            raise ValueError("width must be finite and positive")
        self.surrogate = surrogate
        self.width = float(width)

    @property
    def alpha(self):
        return self.surrogate.alpha

    @property
    def damping_factor(self):
        return self.surrogate.damping_factor

    @property
    def spiking(self):
        return self.surrogate.spiking

    @property
    def support(self) -> float:
        natural = self.surrogate.support
        return self.width if natural is None else min(self.width, natural)

    def primitive(self, x: torch.Tensor) -> torch.Tensor:
        return self.surrogate.primitive(x)

    def derivative(
        self, x: torch.Tensor, grad_output: torch.Tensor, damping_factor: float = 1.0
    ) -> torch.Tensor:
        gradient = self.surrogate.derivative(x, grad_output, damping_factor)
        return torch.where(x.abs() < self.width, gradient, 0.0)
