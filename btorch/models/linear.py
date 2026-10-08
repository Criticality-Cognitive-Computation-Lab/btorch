from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..sparse.operator import LinearOperator
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


class Linear(nn.Linear, LinearOperator, HasConstraint):
    """Apply a PyTorch-compatible dense linear transformation.

    ``Linear`` is both a standard :class:`torch.nn.Linear` module and a
    :class:`~btorch.sparse.LinearOperator`. Its weight always follows the
    PyTorch ``[out_features, in_features]`` convention. Receptor, delay, group,
    and Dale-law semantics belong to :class:`~btorch.models.connection.Synapse`
    and :class:`~btorch.models.connection.SparseConnection`, not this class.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        weight: Optional initial weight with shape
            ``[out_features, in_features]``.
        bias: ``True`` to initialize a bias, ``False`` or ``None`` for no bias,
            or an initial tensor with shape ``[out_features]``.
        mask: Optional dense mask with shape ``[out_features, in_features]``.
        device: Torch device.
        dtype: Torch dtype.
    """

    mask: Tensor | None
    _capabilities = frozenset({"matvec", "matmat", "rmatvec"})

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: Tensor | None = None,
        bias: bool | Tensor | None = True,
        mask: float | Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the dense operator."""
        if weight is not None:
            weight = torch.as_tensor(weight)
            if device is None:
                device = weight.device
            if dtype is None:
                dtype = (
                    weight.dtype
                    if weight.is_floating_point() or weight.is_complex()
                    else torch.get_default_dtype()
                )
        elif isinstance(bias, Tensor):
            if device is None:
                device = bias.device
            if dtype is None:
                dtype = (
                    bias.dtype
                    if bias.is_floating_point() or bias.is_complex()
                    else torch.get_default_dtype()
                )
        super().__init__(
            in_features,
            out_features,
            bias=bias is not False and bias is not None,
            device=device,
            dtype=dtype,
        )
        self._shape = (out_features, in_features)
        if weight is not None:
            weight = torch.as_tensor(weight, device=self.weight.device)
            if weight.shape != (out_features, in_features):
                raise ValueError(
                    "weight must have shape "
                    f"({out_features}, {in_features}), got {tuple(weight.shape)}."
                )
            with torch.no_grad():
                self.weight.copy_(weight.to(dtype=self.weight.dtype))
        if isinstance(bias, Tensor):
            bias = torch.as_tensor(
                bias,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            if bias.shape != (out_features,):
                raise ValueError(
                    f"bias must have shape ({out_features},), got {tuple(bias.shape)}."
                )
            with torch.no_grad():
                self.bias.copy_(bias)

        if mask is not None:
            if isinstance(mask, (int, float)):
                mask = (torch.rand_like(self.weight) < mask).to(dtype=self.weight.dtype)
            else:
                mask = torch.as_tensor(
                    mask,
                    device=self.weight.device,
                    dtype=self.weight.dtype,
                )
                if mask.shape != self.weight.shape:
                    raise ValueError(
                        "mask must have shape "
                        f"{tuple(self.weight.shape)}, got {tuple(mask.shape)}."
                    )
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(out_features, in_features)``."""
        return self.out_features, self.in_features

    def _matvec(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def _rmatvec(self, x: Tensor) -> Tensor:
        return nn.functional.linear(x, self.weight.T)

    def _matmat(self, x: Tensor) -> Tensor:
        return torch.matmul(self.weight, x)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the standard ``nn.Linear`` path along the last input axis."""
        return super().forward(x)

    def constrain(self, *args: Any, **kwargs: Any) -> None:
        """Apply the weight mask to the weight matrix."""
        if self.mask is not None and (
            hasattr(self, "weight_orig") or hasattr(self, "parametrizations")
        ):
            raise RuntimeError(
                "Linear mask constraints cannot be applied after an "
                "external weight parametrization or pruning transform. Apply "
                "btorch constraints before preparing pruning or quantization."
            )
        with torch.no_grad():
            if self.mask is not None:
                self.weight.mul_(self.mask)


__all__ = ["LearnableScale", "Linear"]
