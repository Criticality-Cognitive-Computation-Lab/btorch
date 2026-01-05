import math

import torch

from btorch import jit

from .base import SurrogateFunctionBase


@jit.script
def _erf_primitive(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.special.erfc(-alpha * x) / 2.0


@jit.script
def _erf_derivative(
    x: torch.Tensor, grad_output: torch.Tensor, alpha: float, damping: float
) -> torch.Tensor:
    scale = alpha / math.sqrt(math.pi)
    grad = damping * scale * torch.exp(-((alpha * x) ** 2))
    return grad_output * grad


class Erf(SurrogateFunctionBase):
    """Error-function surrogate with configurable variance and damping."""

    def __init__(
        self,
        alpha: float = 2.0,
        variance: float | None = None,
        damping_factor: float = 1.0,
        spiking: bool = True,
    ):
        if variance is not None:
            alpha = 1.0 / math.sqrt(variance)
        super().__init__(alpha=alpha, damping_factor=damping_factor, spiking=spiking)

    def primitive(self, x: torch.Tensor) -> torch.Tensor:
        return _erf_primitive(x, self.alpha)

    def derivative(
        self,
        x: torch.Tensor,
        grad_output: torch.Tensor,
        damping_factor: float = 1.0,
    ) -> torch.Tensor:
        return _erf_derivative(x, grad_output, self.alpha, damping_factor)


def erf(
    x: torch.Tensor,
    alpha: float = 2.0,
    variance: float | None = None,
    damping_factor: float = 1.0,
    spiking: bool = True,
) -> torch.Tensor:
    return Erf(
        alpha=alpha,
        variance=variance,
        damping_factor=damping_factor,
        spiking=spiking,
    )(x)
