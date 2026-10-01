"""Loss configuration and loss terms for the two-compartment fit."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor


@dataclass(frozen=True)
class FitLossConfig:
    """Loss weights and spike-matching settings for the two-compartment fit.

    This is the single public way to configure the loss: pass one instance as
    ``loss=`` to :func:`two_compartment_loss`,
    :func:`evaluate_two_compartment_fit`, :func:`evaluate_fit_across_sweeps`
    and :func:`fit_two_compartment_model`.

    Args:
        voltage_weight: Weight of the masked voltage reconstruction loss.
        spike_weight: Weight of the smoothed spike-train loss.
        spike_count_weight: Weight of the spike-count matching penalty.
        spike_timing_weight: Weight of the hard spike-timing event loss.
        spike_count_over_weight: Weight of over-prediction in the count loss
            (must be positive).
        spike_count_under_weight: Weight of under-prediction in the count loss
            (must be positive).
        sparsity_weight: Weight of the ``w_Ca`` sparsity penalty.
        spike_tau_ms: Smoothing constant (ms) of the spike-train loss.
        post_spike_mask_ms: Duration (ms) of the post-spike voltage mask.
        spike_match_window_ms: Tolerance window (ms) for matching predicted
            and true spikes.
        spike_miss_penalty_ms: Per-event penalty (ms) assigned to unmatched
            spikes in the hard timing loss. Defaults to the match window.
    """

    voltage_weight: float = 1.0
    spike_weight: float = 1.0
    spike_count_weight: float = 0.0
    spike_timing_weight: float = 0.0
    spike_count_over_weight: float = 1.0
    spike_count_under_weight: float = 1.0
    sparsity_weight: float = 1e-4
    spike_tau_ms: float = 10.0
    post_spike_mask_ms: float = 3.0
    spike_match_window_ms: float = 10.0
    spike_miss_penalty_ms: float | None = None


def mask_post_spike_voltage_samples(
    spike_true: Tensor,
    *,
    refractory_bins: int = 3,
) -> Tensor:
    """Mask out voltage samples immediately after true spikes."""
    if refractory_bins < 0:
        raise ValueError(
            f"refractory_bins must be non-negative, got {refractory_bins}."
        )
    mask = torch.ones_like(spike_true, dtype=torch.bool)
    spike_idx = spike_true > 0
    for offset in range(refractory_bins + 1):
        if offset == 0:
            mask = mask & ~spike_idx
            continue
        shifted = torch.zeros_like(spike_idx)
        shifted[offset:] = spike_idx[:-offset]
        mask = mask & ~shifted
    return mask


def exponential_filter_spike_train(
    spike_train: Tensor,
    *,
    tau_ms: float,
    dt: float,
) -> Tensor:
    """Apply a causal exponential filter to a spike train."""
    if tau_ms <= 0.0:
        raise ValueError(f"tau_ms must be positive, got {tau_ms}.")
    alpha = float(np.exp(-dt / tau_ms))
    filtered = torch.zeros_like(spike_train)
    filtered[0] = spike_train[0]
    for t in range(1, spike_train.shape[0]):
        filtered[t] = alpha * filtered[t - 1] + (1.0 - alpha) * spike_train[t]
    return filtered


def _to_1d_numpy(x: Tensor) -> np.ndarray:
    """Convert a `(T, B, N)` or compatible tensor to a 1D CPU numpy trace."""
    x_cpu = x.detach().cpu()
    if x_cpu.ndim == 3:
        return x_cpu[:, 0, 0].numpy()
    if x_cpu.ndim == 1:
        return x_cpu.numpy()
    raise ValueError(f"Expected a 1D or (T, B, N) tensor, got {tuple(x_cpu.shape)}.")


def _extract_spike_indices(spike_train: Tensor) -> np.ndarray:
    """Extract hard spike-event indices from a `(T, B, N)` spike tensor."""
    trace = _to_1d_numpy(spike_train)
    return np.flatnonzero(trace > 0.5).astype(np.int64)


def spike_timing_stats(
    spike_true: Tensor,
    spike_pred: Tensor,
    *,
    dt: float,
    match_window_ms: float = 10.0,
) -> dict[str, float]:
    """Match predicted and true spikes using a tolerance-window cost."""
    if match_window_ms <= 0.0:
        raise ValueError(f"match_window_ms must be positive, got {match_window_ms}.")

    true_idx = _extract_spike_indices(spike_true)
    pred_idx = _extract_spike_indices(spike_pred)
    true_count = int(true_idx.size)
    pred_count = int(pred_idx.size)

    if true_count == 0 and pred_count == 0:
        return {
            "matched_spikes": 0.0,
            "false_positive_spikes": 0.0,
            "false_negative_spikes": 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "timing_f1": 1.0,
            "mean_timing_error_ms": 0.0,
            "matched_fraction": 1.0,
        }

    if true_count == 0:
        return {
            "matched_spikes": 0.0,
            "false_positive_spikes": float(pred_count),
            "false_negative_spikes": 0.0,
            "precision": 0.0,
            "recall": 1.0,
            "timing_f1": 0.0,
            "mean_timing_error_ms": float(match_window_ms),
            "matched_fraction": 0.0,
        }

    if pred_count == 0:
        return {
            "matched_spikes": 0.0,
            "false_positive_spikes": 0.0,
            "false_negative_spikes": float(true_count),
            "precision": 1.0,
            "recall": 0.0,
            "timing_f1": 0.0,
            "mean_timing_error_ms": float(match_window_ms),
            "matched_fraction": 0.0,
        }

    distance_ms = np.abs(true_idx[:, None] - pred_idx[None, :]).astype(np.float64) * dt
    row_ind, col_ind = linear_sum_assignment(distance_ms)
    matched_distance = distance_ms[row_ind, col_ind]
    within_window = matched_distance <= match_window_ms
    matched_distance = matched_distance[within_window]
    matched_spikes = int(matched_distance.size)
    false_positive = pred_count - matched_spikes
    false_negative = true_count - matched_spikes
    precision = matched_spikes / max(pred_count, 1)
    recall = matched_spikes / max(true_count, 1)
    if precision + recall == 0.0:
        timing_f1 = 0.0
    else:
        timing_f1 = 2.0 * precision * recall / (precision + recall)

    mean_timing_error_ms = (
        float(matched_distance.mean()) if matched_spikes > 0 else float(match_window_ms)
    )
    return {
        "matched_spikes": float(matched_spikes),
        "false_positive_spikes": float(false_positive),
        "false_negative_spikes": float(false_negative),
        "precision": float(precision),
        "recall": float(recall),
        "timing_f1": float(timing_f1),
        "mean_timing_error_ms": mean_timing_error_ms,
        "matched_fraction": matched_spikes / max(true_count, 1),
    }


def spike_timing_loss(
    spike_true: Tensor,
    spike_pred: Tensor,
    *,
    dt: float,
    match_window_ms: float = 10.0,
    miss_penalty_ms: float | None = None,
) -> Tensor:
    """Compute a hard event-based spike-timing loss for global optimization."""
    penalty_ms = float(match_window_ms if miss_penalty_ms is None else miss_penalty_ms)
    stats = spike_timing_stats(
        spike_true,
        spike_pred,
        dt=dt,
        match_window_ms=match_window_ms,
    )
    matched = stats["matched_spikes"]
    false_positive = stats["false_positive_spikes"]
    false_negative = stats["false_negative_spikes"]
    total_events = max(matched + false_positive + false_negative, 1.0)
    total_error_ms = matched * stats["mean_timing_error_ms"] + penalty_ms * (
        false_positive + false_negative
    )
    return torch.as_tensor(
        total_error_ms / total_events,
        dtype=spike_pred.dtype,
        device=spike_pred.device,
    )


def two_compartment_loss(
    *,
    v_pred: Tensor,
    spike_pred: Tensor,
    v_true: Tensor,
    spike_true: Tensor,
    dt: float,
    w_Ca: Tensor | None = None,
    loss: FitLossConfig | None = None,
) -> dict[str, Tensor]:
    """Compute a composite fitting loss for the two-compartment model.

    Args:
        v_pred: Predicted voltage, shape ``[T, B, N]``.
        spike_pred: Predicted spikes, shape ``[T, B, N]``.
        v_true: Recorded voltage, same shape as ``v_pred``.
        spike_true: Recorded spikes, same shape as ``spike_pred``.
        dt: Simulation step in ms.
        w_Ca: Optional calcium-coupling weights for the sparsity penalty.
        loss: Loss weights and spike-matching settings. ``None`` uses the
            :class:`FitLossConfig` defaults.

    Returns:
        Dict with the weighted ``"total"`` and the unweighted components
        ``"voltage"``, ``"spike"``, ``"spike_count"``, ``"spike_timing"`` and
        ``"sparsity"``.

    Raises:
        ValueError: If the spike-count over/under weights are not positive.
    """
    c = FitLossConfig() if loss is None else loss
    refractory_bins = int(round(c.post_spike_mask_ms / dt))
    mask = mask_post_spike_voltage_samples(
        spike_true,
        refractory_bins=refractory_bins,
    )

    if mask.any():
        voltage_loss = F.mse_loss(v_pred[mask], v_true[mask])
    else:
        voltage_loss = torch.zeros((), device=v_pred.device, dtype=v_pred.dtype)

    spike_pred_smooth = exponential_filter_spike_train(
        spike_pred,
        tau_ms=c.spike_tau_ms,
        dt=dt,
    )
    spike_true_smooth = exponential_filter_spike_train(
        spike_true,
        tau_ms=c.spike_tau_ms,
        dt=dt,
    )
    spike_loss = F.smooth_l1_loss(spike_pred_smooth, spike_true_smooth)
    spike_count_pred = (spike_pred > 0.5).to(spike_pred.dtype).sum(dim=0)
    spike_count_true = spike_true.sum(dim=0)
    if c.spike_count_over_weight <= 0.0 or c.spike_count_under_weight <= 0.0:
        raise ValueError("Spike-count over/under weights must be positive.")
    spike_count_over = torch.relu(spike_count_pred - spike_count_true)
    spike_count_under = torch.relu(spike_count_true - spike_count_pred)
    spike_count_loss = (
        c.spike_count_over_weight * spike_count_over.square()
        + c.spike_count_under_weight * spike_count_under.square()
    ).mean()
    spike_timing_event_loss = spike_timing_loss(
        spike_true,
        spike_pred,
        dt=dt,
        match_window_ms=c.spike_match_window_ms,
        miss_penalty_ms=c.spike_miss_penalty_ms,
    )

    if w_Ca is None:
        sparsity_loss = torch.zeros((), device=v_pred.device, dtype=v_pred.dtype)
    else:
        sparsity_loss = w_Ca.abs().mean()

    total = (
        c.voltage_weight * voltage_loss
        + c.spike_weight * spike_loss
        + c.spike_count_weight * spike_count_loss
        + c.spike_timing_weight * spike_timing_event_loss
        + c.sparsity_weight * sparsity_loss
    )
    return {
        "total": total,
        "voltage": voltage_loss,
        "spike": spike_loss,
        "spike_count": spike_count_loss,
        "spike_timing": spike_timing_event_loss,
        "sparsity": sparsity_loss,
    }


def _loss_floats(losses: dict[str, Tensor]) -> dict[str, float]:
    """Convert a :func:`two_compartment_loss` result to float history entries.

    Keys are ``"<component>_loss"`` for every loss term, e.g. ``"total_loss"``
    and ``"spike_count_loss"``.
    """
    return {
        f"{name}_loss": float(value.detach().cpu()) for name, value in losses.items()
    }
