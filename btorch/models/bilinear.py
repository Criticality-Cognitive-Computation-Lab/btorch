import torch
import torch.nn as nn
from torch import Tensor

from .constrain import HasConstraint


class SymmetricBilinear(nn.Bilinear, HasConstraint):
    in_features: int
    mask: Tensor | None
    initial_sign: Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mask: float | Tensor | None = None,
        enforce_dale: bool = False,
        device=None,
        dtype=None,
    ):
        """Symmetric Bilinear layer where both inputs are the same. Supports
        weight masking and Dale's Law enforcement.

        Args:
            in_features: Number of input features for both inputs.
            out_features: Number of output features.
            bias: Whether to use a bias term.
            mask: If float, a random binary mask with this density is generated.
                If Tensor, it must have shape (out_features, in_features, in_features).
            enforce_dale: If True, enforces weights to maintain their initial sign
                in the `constrain()` method.
            device: Torch device.
            dtype: Torch dtype.
        """
        super().__init__(
            in1_features=in_features,
            in2_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if mask is not None:
            if isinstance(mask, (int, float)):
                mask = (
                    torch.rand(self.weight.shape, device=device, dtype=dtype) < mask
                ).to(dtype=self.weight.dtype)
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        if enforce_dale:
            self.register_buffer("initial_sign", torch.sign(self.weight.data))
        else:
            self.initial_sign = None

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input, input)

    def constrain(self, *args, **kwargs):
        """Apply the weight mask and Dale's Law constraints to the weight
        matrix."""
        if self.mask is not None:
            self.weight.data *= self.mask
        if self.initial_sign is not None:
            self.weight.data = (
                self.weight.data * self.initial_sign
            ).relu() * self.initial_sign

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in1_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}, "
            f"mask={self.mask is not None}, "
            f"enforce_dale={self.initial_sign is not None}"
        )
