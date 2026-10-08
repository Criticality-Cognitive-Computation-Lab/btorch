from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from .base import ParamBufferMixin
from .constrain import HasConstraint


class LearnableScale(ParamBufferMixin, nn.Module):
    """Learnable scalar affine transform via softplus-parameterized scale/bias.

    This is the idiomatic way to apply learnable scaling and offset to
    noise generators or other modules: wrap them with ``LearnableScale``.

    trainable controls which parts are trainable:
      - True: both ``scale`` and ``bias`` (if present) are trainable
      - False: neither is trainable (stored as buffers)
      - "scale": only ``scale`` is trainable
      - "bias": only ``bias`` is trainable (requires ``bias`` to be provided)

    Args:
        scale: Initial scale value (softplus-ed internally, so always > 0).
        bias: Initial bias value (softplus-ed internally, so always > 0).
            If None, no bias term is applied.
        trainable: Which components are trainable.
    """

    def __init__(
        self,
        scale: float = 1.0,
        bias: float | None = None,
        trainable: bool | Literal["bias", "scale"] = True,
    ):
        ParamBufferMixin.__init__(self)
        nn.Module.__init__(self)

        if not (
            trainable is True or trainable is False or trainable in ("bias", "scale")
        ):
            raise ValueError("trainable must be a bool or one of 'bias' or 'scale'")
        if trainable == "bias" and bias is None:
            raise ValueError("trainable='bias' requires a non-None bias value")

        self.has_bias = bias is not None

        # Build trainable_param set for def_param
        if trainable is True:
            tp = {"scale", "bias"} if self.has_bias else {"scale"}
        elif trainable is False:
            tp = set()
        elif trainable == "scale":
            tp = {"scale"}
        else:
            tp = {"bias"}

        def _inv_softplus(y: float) -> torch.Tensor:
            # softplus(x) = log(1 + exp(x)); inverse is log(exp(y) - 1).
            # Clamp for y == 0 to avoid -inf.
            val = torch.exp(torch.tensor(float(y))) - 1.0
            return torch.log(val.clamp_min(1e-8))

        scale_init = _inv_softplus(scale)
        self.def_param(
            "scale",
            scale_init,
            sizes=(),
            trainable_param=tp,
            trainable_shape="scalar",
        )

        if self.has_bias:
            bias_init = _inv_softplus(bias)
            self.def_param(
                "bias",
                bias_init,
                sizes=(),
                trainable_param=tp,
                trainable_shape="scalar",
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.scale_value * x
        if self.has_bias:
            out = out + self.bias_value
        return out

    @property
    def scale_value(self) -> Tensor:
        return nn.functional.softplus(self.scale)

    @property
    def bias_value(self) -> Tensor:
        if not self.has_bias:
            raise AttributeError("bias_value is unavailable when bias=None")
        return nn.functional.softplus(self.bias)


class DenseConn(nn.Linear, HasConstraint):
    # Matrix product using y = x @ A.
    mask: Tensor | None
    initial_sign: Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: Tensor | None = None,
        bias: Tensor | None = None,
        mask: float | Tensor | None = None,
        enforce_dale: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Dense connection layer with optional weight masking and Dale's Law
        enforcement.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            weight: Initial weight matrix (in_features, out_features).
                Note: internally it is stored as (out_features, in_features)
                and transposed.
            bias: Initial bias vector (out_features,).
            mask: If float, a random binary mask with this density is generated.
                If Tensor, it must have shape (out_features, in_features).
            enforce_dale: If True, enforces weights to maintain their initial sign
                via ReLU in the `constrain()` method.
            device: Torch device.
            dtype: Torch dtype.
        """
        # if weight is given, in_features and out_features are ignored
        super().__init__(
            in_features, out_features, bias=bias is not None, device=device, dtype=dtype
        )
        if weight is not None:
            self.weight.data = weight.T
        if bias is not None:
            self.bias.data = bias

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

    def constrain(self, *args: Any, **kwargs: Any) -> None:
        """Apply the weight mask and Dale's Law constraints to the weight
        matrix."""
        if self.mask is not None:
            self.weight.data *= self.mask
        if self.initial_sign is not None:
            self.weight.data = (
                self.weight.data * self.initial_sign
            ).relu() * self.initial_sign
