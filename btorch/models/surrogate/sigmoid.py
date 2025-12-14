import torch

from .base import SurrogateFunctionBase


_sigmoid_primitive = torch.jit.script(lambda x, alpha: torch.sigmoid(alpha * x))
_sigmoid_derivative = torch.jit.script(
    lambda x, alpha: alpha * torch.sigmoid(alpha * x) * (1 - torch.sigmoid(alpha * x))
)


class Sigmoid(SurrogateFunctionBase):
    """Logistic surrogate derivative."""

    def primitive(self, x: torch.Tensor) -> torch.Tensor:
        return _sigmoid_primitive(x, self.alpha)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return _sigmoid_derivative(x, self.alpha)
