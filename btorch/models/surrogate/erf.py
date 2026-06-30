import math

import torch

from btorch import jit

from .base import SurrogateFunctionBase


# Internal scale: sqrt(ln2)*alpha so that HWHM = 1/alpha.
# g(v) = exp(-ln2*(alpha*v)^2) = 2^{-(alpha*v)^2}
# g(v_hw) = 0.5  =>  ln2*(alpha*v_hw)^2 = ln2  =>  v_hw = 1/alpha.
_ERF_K = math.sqrt(math.log(2.0))  # ≈ 0.8326


@jit.script
def _erf_primitive(x: torch.Tensor, alpha: float) -> torch.Tensor:
    k: float = math.sqrt(math.log(2.0))  # sqrt(ln2) ≈ 0.8326
    return torch.special.erfc(-k * alpha * x) / 2.0


@jit.script
def _erf_derivative(
    x: torch.Tensor, grad_output: torch.Tensor, alpha: float, damping: float
) -> torch.Tensor:
    # g(v) = 2^{-(alpha*v)^2} = exp(-ln2*(alpha*v)^2)
    grad = damping * torch.exp(-math.log(2.0) * (alpha * x) ** 2)
    return grad_output * grad


class Erf(SurrogateFunctionBase):
    """Error-function (Gaussian) surrogate gradient.

    Surrogate gradient: ``g(v) = 2^{−(alpha·v)²} = exp(−ln2·(alpha·v)²)``

    ``alpha`` is the inverse half-width: HWHM = 1/alpha for any alpha.
    Peak at threshold is 1.0 when damping=1.

    The default ``alpha=4`` (HWHM=0.25) matches the Gaussian surrogate used
    in the large-scale V1 model of Chen et al. (2022), which uses
    ``gauss_std=0.28`` (mapping to ``alpha = 1/(0.28·√ln2) ≈ 4.3``) with
    ``damping_factor=0.5``.

    Parameters
    ----------
    alpha : float
        Inverse half-width.  ``variance = 1/alpha²`` sets the Gaussian
        variance of the gradient envelope; ``alpha = 1/sqrt(variance)``.
    variance : float, optional
        Convenience alternative to ``alpha``.  Sets ``alpha = 1/sqrt(variance)``,
        so the HWHM equals ``sqrt(variance)``.

    References:
        Chen, G., Scherr, F., & Maass, W. (2022). A data-based large-scale
        model for primary visual cortex enables brain-like robust and versatile
        visual processing. *Science Advances*, 8(44), eabq7592.
        https://doi.org/10.1126/sciadv.abq7592
    """

    def __init__(
        self,
        alpha: float = 4.0,
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
    alpha: float = 4.0,
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
