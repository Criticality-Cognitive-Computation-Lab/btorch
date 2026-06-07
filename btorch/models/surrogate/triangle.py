import torch

from btorch import jit

from .base import SurrogateFunctionBase


# Internal scale: alpha/2 so that HWHM = 1/alpha for any alpha.
# g(v) = (1 - |alpha*v/2|)+  =>  g(v_hw) = 0.5  =>  v_hw = 1/alpha.
_TRIANGLE_SCALE = 0.5


@jit.script
def _triangle_primitive(x: torch.Tensor, alpha: float) -> torch.Tensor:
    ax = 0.5 * alpha * x
    result = torch.where(
        ax < -1.0,
        torch.zeros_like(x),
        torch.where(
            ax < 0.0,
            ax + ax * ax / 2.0 + 0.5,
            torch.where(
                ax < 1.0,
                ax - ax * ax / 2.0 + 0.5,
                torch.ones_like(x),
            ),
        ),
    )
    return result


@jit.script
def _triangle_derivative(
    x: torch.Tensor, grad_output: torch.Tensor, alpha: float, damping: float
) -> torch.Tensor:
    v_scaled = 0.5 * alpha * x
    grad = damping * (1.0 - v_scaled.abs()).clamp(min=0.0)
    return grad_output * grad


class Triangle(SurrogateFunctionBase):
    """Triangular surrogate gradient.

    Surrogate gradient: ``g(v) = (1 − |alpha·v/2|)₊``

    ``alpha`` is the inverse half-width: HWHM = 1/alpha for any alpha.
    Peak at threshold is 1.0 when damping=1.
    """

    def __init__(
        self, alpha: float = 2.0, damping_factor: float = 1.0, spiking: bool = True
    ):
        super().__init__(alpha=alpha, damping_factor=damping_factor, spiking=spiking)

    def primitive(self, x: torch.Tensor) -> torch.Tensor:
        return _triangle_primitive(x, self.alpha)

    def derivative(
        self,
        x: torch.Tensor,
        grad_output: torch.Tensor,
        damping_factor: float = 1.0,
    ) -> torch.Tensor:
        return _triangle_derivative(x, grad_output, self.alpha, damping_factor)


def triangle(
    x: torch.Tensor,
    alpha: float = 2.0,
    damping_factor: float = 1.0,
    spiking: bool = True,
) -> torch.Tensor:
    return Triangle(alpha=alpha, damping_factor=damping_factor, spiking=spiking)(x)
