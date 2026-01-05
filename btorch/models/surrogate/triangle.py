import torch

from btorch import jit

from .base import SurrogateFunctionBase


@jit.script
def _triangle_primitive(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Antiderivative of the triangular surrogate gradient.

    Integrates the triangular function to produce a piecewise quadratic
    that smoothly transitions from 0 to 1, resembling a smoothed step.
    """
    ax = alpha * x
    # Piecewise quadratic: 0 for ax < -1, quadratic rise to 0.5 at ax=0,
    # quadratic descent from 0.5 to 1 at ax=1, then 1 for ax > 1
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
    v_scaled = alpha * x
    grad = (1.0 - v_scaled.abs()).clamp(min=0.0)
    grad = damping * grad * alpha
    return grad_output * grad


class Triangle(SurrogateFunctionBase):
    """Triangular surrogate gradient with optional damping."""

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
    alpha: float = 1.0,
    damping_factor: float = 1.0,
    spiking: bool = True,
) -> torch.Tensor:
    return Triangle(alpha=alpha, damping_factor=damping_factor, spiking=spiking)(x)
