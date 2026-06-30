import math

import torch

from btorch import jit

from .base import SurrogateFunctionBase


# Internal scale: 2*ln(sqrt(2)+1)*alpha so that HWHM = 1/alpha.
# g(v) = 4*sigmoid(k*alpha*v)*(1-sigmoid(k*alpha*v))
# g(v_hw)=0.5 => v_hw=2*ln(sqrt(2)+1)/(k*alpha)=1/alpha => k=2*ln(sqrt(2)+1).
_SIG_K = 2.0 * math.log(math.sqrt(2.0) + 1.0)  # ≈ 1.7627


@jit.script
def _sigmoid_primitive(x: torch.Tensor, alpha: float) -> torch.Tensor:
    k: float = 2.0 * math.log(math.sqrt(2.0) + 1.0)  # 2*ln(sqrt(2)+1) ≈ 1.7627
    return torch.sigmoid(k * alpha * x)


@jit.script
def _sigmoid_derivative(
    x: torch.Tensor, grad_output: torch.Tensor, alpha: float, damping: float
) -> torch.Tensor:
    k: float = 2.0 * math.log(math.sqrt(2.0) + 1.0)  # 2*ln(sqrt(2)+1) ≈ 1.7627
    sigma = torch.sigmoid(k * alpha * x)
    grad = damping * 4.0 * sigma * (1.0 - sigma)
    return grad_output * grad


class Sigmoid(SurrogateFunctionBase):
    """Logistic surrogate gradient.

    Surrogate gradient: ``g(v) = 4·σ(k·alpha·v)·(1−σ(k·alpha·v))``
    where ``k = 2·ln(√2+1) ≈ 1.763``.

    ``alpha`` is the inverse half-width: HWHM = 1/alpha for any alpha.
    Peak at threshold is 1.0 when damping=1.
    """

    def __init__(
        self, alpha: float = 2.0, damping_factor: float = 1.0, spiking: bool = True
    ):
        super().__init__(alpha=alpha, damping_factor=damping_factor, spiking=spiking)

    def primitive(self, x: torch.Tensor) -> torch.Tensor:
        return _sigmoid_primitive(x, self.alpha)

    def derivative(
        self,
        x: torch.Tensor,
        grad_output: torch.Tensor,
        damping_factor: float = 1.0,
    ) -> torch.Tensor:
        return _sigmoid_derivative(x, grad_output, self.alpha, damping_factor)


def sigmoid(
    x: torch.Tensor,
    alpha: float = 2.0,
    damping_factor: float = 1.0,
    spiking: bool = True,
) -> torch.Tensor:
    return Sigmoid(alpha=alpha, damping_factor=damping_factor, spiking=spiking)(x)
