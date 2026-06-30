from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn

from ..types import TensorLike


class VoltageRegularizer(nn.Module):
    """Voltage regularization loss for spiking neurons.

    Penalizes membrane potentials that deviate far from the reset/threshold
    range, encouraging neurons to stay near their operating regime.

    Args:
        v_threshold: Threshold voltage (default: 1.0).
        v_reset: Reset voltage (default: 0.0).
        voltage_cost: Coefficient for the regularization term (default: 1e-4).

    Example:
        >>> reg = VoltageRegularizer(v_threshold=1.0, v_reset=0.0)
        >>> loss = reg(voltages)
    """

    v_offset: torch.Tensor
    v_scale: torch.Tensor

    def __init__(
        self,
        v_threshold: float | TensorLike = 1.0,
        v_reset: float | TensorLike = 0.0,
        voltage_cost: float = 1e-4,
    ):
        super().__init__()
        self.voltage_cost = voltage_cost
        self.register_buffer("v_offset", torch.as_tensor(v_threshold))
        self.register_buffer("v_scale", torch.as_tensor(v_threshold - v_reset))

    def forward(
        self, voltages: Float[torch.Tensor, "... n_neuron"]
    ) -> Float[torch.Tensor, ""]:
        """Compute voltage regularization loss.

        Args:
            voltages: Membrane voltages of shape ``(..., n_neuron)``.

        Returns:
            Scalar regularization loss.
        """
        voltage_32 = (voltages.float() - self.v_offset) / self.v_scale

        v_pos = F.relu(voltage_32 - 1.0).pow(2)
        v_neg = F.relu(-voltage_32 - 1.0).pow(2)

        voltage_loss = torch.mean(torch.sum(v_pos + v_neg, dim=-1)) * self.voltage_cost
        return voltage_loss


class QuantileDistributionLoss(nn.Module):
    """Quantile distribution loss supporting pinball and huber modes.

    Supports arbitrary batch dimensions. Computes the loss between sorted
    predictions and sorted targets across quantiles.

    Args:
        loss_type: ``'pinball'`` or ``'huber_pinball'``.
        kappa: Smoothing parameter for huber loss.
        reduction: ``'mean'``, ``'sum'``, or ``'none'`` over batch elements.
        sorted: If True, assumes ``target`` is already sorted.

    Example:
        >>> loss_fn = QuantileDistributionLoss("huber_pinball", kappa=0.002)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        loss_type: Literal["pinball", "huber_pinball"] = "huber_pinball",
        kappa=0.002,
        reduction="mean",
        sorted: bool = False,
    ):
        super().__init__()
        assert loss_type in (
            "pinball",
            "huber_pinball",
        ), "loss_type must be 'pinball' or 'huber_pinball'"
        assert reduction in (
            "mean",
            "sum",
            "none",
        ), "reduction must be 'mean', 'sum', or 'none'"
        self.loss_type = loss_type
        self.kappa = kappa
        self.reduction = reduction
        self.sorted = sorted

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute quantile distribution loss between pred and target.

        Args:
            pred: Predictions of shape ``(..., N)``.
            target: Targets of shape ``(..., N)``.

        Returns:
            Scalar if ``reduction != 'none'``, else shape ``(...)``.
        """
        assert pred.shape == target.shape, "pred and target must have the same shape"
        assert pred.dim() >= 1, "Input must have at least one dimension"

        *batch_dims, N = pred.shape

        # Sort along last dimension
        pred_sorted, _ = torch.sort(pred, dim=-1)
        if not self.sorted:
            target, _ = torch.sort(target, dim=-1)

        # Quantiles: shape (N,)
        device = pred.device
        dtype = pred.dtype
        tau = (torch.arange(N, device=device, dtype=dtype) + 1) / N  # (N,)

        # Reshape tau for broadcasting: (1, 1, ..., N)
        tau_shape = (1,) * (pred.dim() - 1) + (N,)
        tau = tau.view(tau_shape)

        u = pred_sorted - target
        indicator = (u <= 0).float()
        weight = torch.abs(tau - indicator)

        if self.loss_type == "pinball":
            loss = weight * torch.abs(u)
        else:
            abs_u = torch.abs(u)
            quadratic = 0.5 * (u**2) / self.kappa
            linear = abs_u - 0.5 * self.kappa
            loss = torch.where(abs_u <= self.kappa, weight * quadratic, weight * linear)

        # loss shape: (..., N)
        # Average over last dim (quantiles)
        loss = loss.mean(dim=-1)  # shape (... batch dims)

        # Reduction over batch dims
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class FiringRateLoss(nn.Module):
    """Firing rate loss for matching population activity distributions.

    Uses :class:`QuantileDistributionLoss` internally to match the
    quantile distribution of firing rates (or spike counts).

    Args:
        target: Target firing rate distribution.
        input_type: ``'spike'`` or ``'firing_rate'``.
        n_neuron: Optional output neuron count for interpolation.
        loss_type: ``'pinball'`` or ``'huber'``.
        kappa: Smoothing parameter for huber loss.
        reduction: ``'sum'`` or ``'mean'``.
        rng: Seed for any random operations.
        sorted: If True, assumes ``target`` is already sorted.

    Example:
        >>> target_fr = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        >>> loss_fn = FiringRateLoss(target_fr, input_type="firing_rate")
        >>> loss = loss_fn(rates)
    """

    target: torch.Tensor

    def __init__(
        self,
        target: Float[TensorLike, "... M"],
        input_type: Literal["spike", "firing_rate"] = "spike",
        n_neuron: int | None = None,
        loss_type="huber",
        kappa=0.002,
        reduction: Literal["sum", "mean"] = "mean",
        rng: torch.Generator | int | None = None,
        sorted: bool = False,
    ):
        super().__init__()
        self.loss = QuantileDistributionLoss(loss_type, kappa, reduction)

        self.rng = torch.Generator().manual_seed(rng) if isinstance(rng, int) else rng
        self.input_type = input_type

        target_tensor = torch.as_tensor(target)
        if not sorted:
            target_tensor, _ = torch.sort(target_tensor)
        if n_neuron is not None:
            target_tensor = torch.nn.functional.interpolate(
                target_tensor, target_tensor.shape[:-1] + (n_neuron,), mode="bilinear"
            )

        self.n_neuron = target_tensor.shape[-1]

        self.register_buffer("target", target_tensor)

    def forward(self, x: Float[torch.Tensor, "... n_neuron"]):
        """Compute firing rate loss.

        Args:
            x: Spike tensor or firing rate tensor of shape ``(..., n_neuron)``.

        Returns:
            Scalar loss.
        """
        if self.input_type == "spike":
            x = x.mean(0)

        assert x.shape[-1] == 1 or (x.shape[-1] == self.n_neuron)

        return self.loss(x, self.target)
