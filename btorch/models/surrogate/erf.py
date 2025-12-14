import math

import torch

from .base import SurrogateFunctionBase


_erf_primitive = torch.jit.script(
    lambda x, alpha, variance: 0.5
    * (1 + torch.erf(alpha * x / math.sqrt(2 * variance)))
)
_erf_derivative = torch.jit.script(
    lambda x, alpha, variance: (alpha / math.sqrt(2 * math.pi * variance))
    * torch.exp(-0.5 * (alpha * x) ** 2 / variance)
)


class Erf(SurrogateFunctionBase):
    """Error-function surrogate with configurable variance and damping."""

    def __init__(
        self,
        alpha: float = 1.0,
        variance: float = 1.0,
        damping_factor: float = 1.0,
        spiking: bool = True,
    ):
        super().__init__(alpha=alpha, damping_factor=damping_factor, spiking=spiking)
        self.variance = variance

    def primitive(self, x: torch.Tensor) -> torch.Tensor:
        return _erf_primitive(x, self.alpha, self.variance)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return _erf_derivative(x, self.alpha, self.variance)
