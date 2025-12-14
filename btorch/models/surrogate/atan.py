import math

import torch

from .base import SurrogateFunctionBase


_atan_primitive = torch.jit.script(
    lambda x, alpha: 0.5 + torch.atan(alpha * x) / math.pi
)
_atan_derivative = torch.jit.script(lambda x, alpha: alpha / (1 + (alpha * x) ** 2))


class ATan(SurrogateFunctionBase):
    """Arctan surrogate (derivative ~ 1 / (1 + (alpha*x)^2))."""

    def primitive(self, x: torch.Tensor) -> torch.Tensor:
        return _atan_primitive(x, self.alpha)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return _atan_derivative(x, self.alpha)
