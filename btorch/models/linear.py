from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from btorch.sparse.param import SparseLinear as _SparseLinearImpl

from .base import ParamBufferMixin
from .constrain import HasConstraint


class LearnableScale(ParamBufferMixin, nn.Module):
    """Positive learnable scalar scale and optional positive bias."""

    def __init__(
        self,
        scale: float = 1.0,
        bias: float | None = None,
        trainable: bool | Literal["bias", "scale"] = True,
    ) -> None:
        ParamBufferMixin.__init__(self)
        nn.Module.__init__(self)
        if not (
            trainable is True or trainable is False or trainable in ("bias", "scale")
        ):
            raise ValueError("trainable must be a bool, 'bias', or 'scale'.")
        if trainable == "bias" and bias is None:
            raise ValueError("trainable='bias' requires a bias value.")
        self.has_bias = bias is not None
        if trainable is True:
            trainable_param = {"scale", "bias"} if bias is not None else {"scale"}
        elif trainable is False:
            trainable_param = set()
        else:
            trainable_param = {trainable}

        def inverse_softplus(value: float) -> torch.Tensor:
            expm1 = torch.expm1(torch.tensor(float(value)))
            return torch.log(expm1.clamp_min(1e-8))

        self.def_param(
            "scale",
            inverse_softplus(scale),
            sizes=(),
            trainable_param=trainable_param,
            trainable_shape="scalar",
        )
        if bias is not None:
            self.def_param(
                "bias",
                inverse_softplus(bias),
                sizes=(),
                trainable_param=trainable_param,
                trainable_shape="scalar",
            )

    @property
    def scale_value(self) -> torch.Tensor:
        return nn.functional.softplus(self.scale)

    @property
    def bias_value(self) -> torch.Tensor:
        if not self.has_bias:
            raise AttributeError("bias_value is unavailable when bias=None.")
        return nn.functional.softplus(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.scale_value * x
        if self.has_bias:
            out = out + self.bias_value
        return out


class DenseConn(nn.Linear, HasConstraint):
    """Dense connection with optional projected mask and Dale sign."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool | torch.Tensor = True,
        *,
        weight: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        source_sign: torch.Tensor | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias is not False,
            device=device,
            dtype=dtype,
        )
        if weight is not None:
            if weight.shape != (in_features, out_features):
                raise ValueError("weight must have shape (in_features, out_features).")
            with torch.no_grad():
                self.weight.copy_(weight.T)
        if isinstance(bias, torch.Tensor):
            if bias.shape != (out_features,):
                raise ValueError("bias tensor must have shape (out_features,).")
            with torch.no_grad():
                self.bias.copy_(bias)
        if mask is not None:
            mask = torch.as_tensor(
                mask, device=self.weight.device, dtype=self.weight.dtype
            )
            if mask.shape == (in_features, out_features):
                mask = mask.T
            if mask.shape != self.weight.shape:
                raise ValueError("mask has an incompatible shape.")
        self.register_buffer("mask", mask)
        if source_sign is not None:
            source_sign = torch.as_tensor(
                source_sign, device=self.weight.device, dtype=self.weight.dtype
            )
            if source_sign.shape != (in_features,):
                raise ValueError("source_sign must have shape (in_features,).")
            source_sign = torch.sign(source_sign)
            if torch.any(source_sign == 0):
                raise ValueError("source_sign entries must be non-zero.")
        self.register_buffer("source_sign", source_sign)

    @torch.no_grad()
    def constrain(self, *args, **kwargs) -> None:
        if self.mask is not None:
            self.weight.mul_(self.mask)
        if self.source_sign is not None:
            sign = self.source_sign[None, :]
            self.weight.copy_((self.weight * sign).clamp_min(0) * sign)


class SparseLinear(_SparseLinearImpl):
    """Sparse connection. Wraps SparseTensorBase in SparseParam.

    Usage::
        W = CSR.from_edges(src, dst, vals, shape=(n_pre, n_post))
        layer = SparseConn(W, bias=True)
        out = layer(x)   # x: (..., n_pre) → (..., n_post)
    """


DenseLinear = DenseConn
SparseConn = SparseLinear


__all__ = [
    "LearnableScale",
    "DenseConn",
    "DenseLinear",
    "SparseLinear",
    "SparseConn",
]
