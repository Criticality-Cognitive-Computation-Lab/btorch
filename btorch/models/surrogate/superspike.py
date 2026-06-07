import math

import torch

from btorch import jit

from .base import SurrogateFunctionBase


# Internal scale: (sqrt(2)-1)*alpha so that HWHM = 1/alpha.
# g(v) = 1/(1 + (sqrt(2)-1)*alpha*|v|)^2
# g(v_hw) = 0.5  =>  (1+(sqrt(2)-1)*alpha*v_hw)^2 = 2  =>  v_hw = 1/alpha.
_SS_K = math.sqrt(2.0) - 1.0  # ≈ 0.4142


@jit.script
def _superspike_primitive(x: torch.Tensor, alpha: float) -> torch.Tensor:
    k: float = math.sqrt(2.0) - 1.0  # sqrt(2)-1 ≈ 0.4142
    k_alpha = k * alpha
    pos = 1.0 - 0.5 / (1.0 + k_alpha * x)
    neg = 0.5 / (1.0 - k_alpha * x)
    return torch.where(x >= 0.0, pos, neg)


@jit.script
def _superspike_derivative(
    x: torch.Tensor, grad_output: torch.Tensor, alpha: float, damping: float
) -> torch.Tensor:
    k: float = math.sqrt(2.0) - 1.0  # sqrt(2)-1 ≈ 0.4142
    grad = damping / (1.0 + k * alpha * x.abs()) ** 2
    return grad_output * grad


class SuperSpike(SurrogateFunctionBase):
    """SuperSpike surrogate gradient.

    Surrogate gradient: ``h(v) = 1 / (1 + (√2−1)·alpha·|v|)²``

    ``alpha`` is the inverse half-width: HWHM = 1/alpha for any alpha.
    Peak at threshold is 1.0 when damping=1.

    References:
        Zenke, F., & Ganguli, S. (2018). SuperSpike: Supervised learning in
        multi-layer spiking neural networks. *Neural Computation*, 30(6),
        1514–1541. https://doi.org/10.1162/neco_a_01086
    """

    def __init__(
        self, alpha: float = 2.0, damping_factor: float = 1.0, spiking: bool = True
    ):
        super().__init__(alpha=alpha, damping_factor=damping_factor, spiking=spiking)

    def primitive(self, x: torch.Tensor) -> torch.Tensor:
        return _superspike_primitive(x, self.alpha)

    def derivative(
        self,
        x: torch.Tensor,
        grad_output: torch.Tensor,
        damping_factor: float = 1.0,
    ) -> torch.Tensor:
        return _superspike_derivative(x, grad_output, self.alpha, damping_factor)


def superspike(
    x: torch.Tensor,
    alpha: float = 2.0,
    damping_factor: float = 1.0,
    spiking: bool = True,
) -> torch.Tensor:
    return SuperSpike(alpha=alpha, damping_factor=damping_factor, spiking=spiking)(x)
