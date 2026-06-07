import math

import torch

from btorch import jit

from .base import SurrogateFunctionBase


# Internal scale: alpha (replacing the previous pi*alpha/2) so that HWHM = 1/alpha.
# g(v) = 1/(1+(alpha*v)^2)  [Cauchy/Lorentz kernel]
# g(v_hw) = 0.5  =>  (alpha*v_hw)^2 = 1  =>  v_hw = 1/alpha.


@jit.script
def _atan_primitive(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return 0.5 + torch.atan(alpha * x) / math.pi


@jit.script
def _atan_approx_primitive(x: torch.Tensor, alpha: float) -> torch.Tensor:
    z = alpha * x
    approx_a = 0.28
    denom = 1.0 + approx_a * z * z
    atan_approx = z / denom
    return 0.5 + atan_approx / math.pi


@jit.script
def _atan_derivative(
    x: torch.Tensor, grad_output: torch.Tensor, alpha: float, damping: float
) -> torch.Tensor:
    grad = damping / (1.0 + (alpha * x) ** 2)
    return grad_output * grad


@jit.script
def _atan_approx_derivative(
    x: torch.Tensor, grad_output: torch.Tensor, alpha: float, damping: float
) -> torch.Tensor:
    z = alpha * x
    approx_a = 0.28
    denom = 1.0 + approx_a * z * z
    f_prime = (1.0 - approx_a * z * z) / (denom * denom)
    grad = damping * f_prime
    return grad_output * grad


class ATan(SurrogateFunctionBase):
    """Arctangent (Cauchy) surrogate gradient.

    Surrogate gradient: ``g(v) = 1 / (1 + (alpha·v)²)``

    ``alpha`` is the inverse half-width: HWHM = 1/alpha for any alpha.
    Peak at threshold is 1.0 when damping=1.
    """

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


class ATanApprox(SurrogateFunctionBase):
    """ATan surrogate with rational approximation.

    Surrogate gradient approximates ``1/(1+(alpha·v)²)`` via a rational
    function, avoiding the true arctangent in the primitive.

    ``alpha`` is the inverse half-width: HWHM ≈ 1/alpha (exact for ATan).
    Peak at threshold is 1.0 when damping=1.
    """

    def __init__(
        self, alpha: float = 2.0, damping_factor: float = 1.0, spiking: bool = True
    ):
        super().__init__(alpha=alpha, damping_factor=damping_factor, spiking=spiking)

    def primitive(self, x: torch.Tensor) -> torch.Tensor:
        return _atan_approx_primitive(x, self.alpha)

    def derivative(
        self,
        x: torch.Tensor,
        grad_output: torch.Tensor,
        damping_factor: float = 1.0,
    ) -> torch.Tensor:
        return _atan_approx_derivative(x, grad_output, self.alpha, damping_factor)


def atan(
    x: torch.Tensor,
    alpha: float = 2.0,
    damping_factor: float = 1.0,
    spiking: bool = True,
) -> torch.Tensor:
    return ATan(alpha=alpha, damping_factor=damping_factor, spiking=spiking)(x)


def atan_approx(
    x: torch.Tensor,
    alpha: float = 2.0,
    damping_factor: float = 1.0,
    spiking: bool = True,
) -> torch.Tensor:
    return ATanApprox(alpha=alpha, damping_factor=damping_factor, spiking=spiking)(x)
