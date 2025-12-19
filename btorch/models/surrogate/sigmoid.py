import torch

from btorch import jit

from .base import SurrogateFunctionBase


@jit.script
def _sigmoid_primitive(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.sigmoid(alpha * x)


@jit.script
def _sigmoid_derivative(x: torch.Tensor, alpha: float) -> torch.Tensor:
    sigma = torch.sigmoid(alpha * x)
    return alpha * sigma * (1 - sigma)


class Sigmoid(SurrogateFunctionBase):
    """Logistic surrogate derivative."""

    def primitive(self, x: torch.Tensor) -> torch.Tensor:
        return _sigmoid_primitive(x, self.alpha)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return _sigmoid_derivative(x, self.alpha)


def sigmoid(
    x: torch.Tensor,
    alpha: float = 1.0,
    damping_factor: float = 1.0,
    spiking: bool = True,
) -> torch.Tensor:
    return Sigmoid(alpha=alpha, damping_factor=damping_factor, spiking=spiking)(x)
