import math

import torch

from btorch import jit

from .base import SurrogateFunctionBase


@jit.script
def _atan_primitive(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return 0.5 + torch.atan(0.5 * math.pi * alpha * x) / math.pi


@jit.script
def _atan_derivative(
    x: torch.Tensor, grad_output: torch.Tensor, alpha: float, damping: float
) -> torch.Tensor:
    scale = 0.5 * math.pi * alpha
    grad = damping * alpha / (2.0 * (1 + (scale * x) ** 2))
    return grad_output * grad


class ATan(SurrogateFunctionBase):
    """Arctan surrogate matching SpikingJelly's alpha scaling."""

    def __init__(
        self, alpha: float = 2.0, damping_factor: float = 1.0, spiking: bool = True
    ):
        super().__init__(alpha=alpha, damping_factor=damping_factor, spiking=spiking)

    def primitive(self, x: torch.Tensor) -> torch.Tensor:
        return _atan_primitive(x, self.alpha)

    def derivative(
        self,
        x: torch.Tensor,
        grad_output: torch.Tensor,
        damping_factor: float = 1.0,
    ) -> torch.Tensor:
        return _atan_derivative(x, grad_output, self.alpha, damping_factor)


def atan(
    x: torch.Tensor,
    alpha: float = 2.0,
    damping_factor: float = 1.0,
    spiking: bool = True,
) -> torch.Tensor:
    return ATan(alpha=alpha, damping_factor=damping_factor, spiking=spiking)(x)
