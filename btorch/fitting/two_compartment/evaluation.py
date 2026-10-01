"""Rollout and per-sweep evaluation for the two-compartment fit."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from btorch.models import environ, functional

from .data import AllenSweepBatch
from .loss import (
    FitLossConfig,
    _loss_floats,
    _to_1d_numpy,
    mask_post_spike_voltage_samples,
    spike_timing_stats,
    two_compartment_loss,
)


class TwoCompartmentModel(Protocol):
    """What the fitting and evaluation code needs from a neuron model.

    :class:`~btorch.models.neurons.two_compartment.TwoCompartmentGLIF`
    satisfies it. Besides the members below, the model must be a
    :class:`torch.nn.Module` that works with
    :func:`btorch.models.functional.init_net_state` and ``reset_net`` (hidden
    state buffers), and its fitted parameters are those with
    ``requires_grad=True`` (names must have bounds, see
    ``DEFAULT_TWO_COMPARTMENT_PARAM_BOUNDS``).

    Attributes:
        w_Ca: Calcium-coupling weights; their mean absolute value is the
            sparsity penalty of the loss.
    """

    w_Ca: Tensor

    def multi_step_forward(
        self,
        i_soma_seq: Tensor,
        i_apical_seq: Tensor | None = None,
        *,
        return_state: bool = False,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Run ``(T, *batch, n_neuron)`` inputs and return ``(spike, v,
        state)`` where ``state`` holds ``v_pre_spike``, ``i_a`` and ``i_bap``
        (called with ``return_state=True``)."""
        ...

    def parameters(self) -> Iterable[torch.nn.Parameter]: ...

    def named_parameters(self) -> Iterable[tuple[str, torch.nn.Parameter]]: ...


@dataclass
class FitEvaluation:
    """Evaluation artifacts for a fitted model on one sweep."""

    specimen_id: int
    sweep_number: int
    dt: float
    metrics: dict[str, float]
    traces: dict[str, np.ndarray]


def rollout_two_compartment(
    model: TwoCompartmentModel,
    i_soma: Tensor,
    i_apical: Tensor | None = None,
) -> dict[str, Tensor]:
    """Run a time-first rollout and collect fitting-relevant traces."""
    spike, voltage, state = model.multi_step_forward(
        i_soma,
        i_apical,
        return_state=True,
    )
    return {
        "spike": spike,
        "v": voltage,
        "v_pre_spike": state["v_pre_spike"],
        "i_a": state["i_a"],
        "i_bap": state["i_bap"],
    }


def _f1_score_from_binary_traces(
    spike_true: np.ndarray,
    spike_pred: np.ndarray,
) -> float:
    """Compute the binary spike F1 score for aligned traces."""
    true_mask = spike_true > 0.5
    pred_mask = spike_pred > 0.5
    tp = int(np.logical_and(true_mask, pred_mask).sum())
    fp = int(np.logical_and(~true_mask, pred_mask).sum())
    fn = int(np.logical_and(true_mask, ~pred_mask).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _prepare_sweep(
    model: TwoCompartmentModel,
    sweep: AllenSweepBatch,
    *,
    device: str | torch.device | None,
    dtype: torch.dtype,
    init_state: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Move a sweep to ``device``/``dtype`` and reset the model for it.

    Args:
        model: Model whose state is (re)initialised for the sweep batch size.
        sweep: Sweep to convert.
        device: Target device.
        dtype: Target dtype.
        init_state: Also call ``init_net_state`` (skip after the first sweep
            when the state buffers already exist).

    Returns:
        ``(i_soma, v_true, spike_true, i_apical)`` on the target device.
    """
    i_soma = sweep.i_soma.to(device=device, dtype=dtype)
    v_true = sweep.v_true.to(device=device, dtype=dtype)
    spike_true = sweep.spike_true.to(device=device, dtype=dtype)
    i_apical = None
    if sweep.i_apical is not None:
        i_apical = sweep.i_apical.to(device=device, dtype=dtype)

    batch_size = i_soma.shape[1]
    if init_state:
        functional.init_net_state(
            model, batch_size=batch_size, device=device, dtype=dtype
        )
    functional.reset_net(model, batch_size=batch_size, device=device, dtype=dtype)
    return i_soma, v_true, spike_true, i_apical


def evaluate_two_compartment_fit(
    model: TwoCompartmentModel,
    sweep: AllenSweepBatch,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    loss: FitLossConfig | None = None,
) -> FitEvaluation:
    """Evaluate a fitted model on a single sweep and collect metrics.

    Args:
        model: Fitted two-compartment neuron.
        sweep: Sweep to roll the model out on.
        device: Torch device; defaults to the model's device.
        dtype: Torch dtype used for the rollout.
        loss: Loss weights and spike-matching settings (defaults to
            :class:`FitLossConfig`).
    """
    config = FitLossConfig() if loss is None else loss
    i_soma, v_true, spike_true, i_apical = _prepare_sweep(
        model, sweep, device=device, dtype=dtype
    )

    with torch.no_grad():
        with environ.context(dt=float(sweep.dt)):
            rollout = rollout_two_compartment(model, i_soma, i_apical)

    losses = two_compartment_loss(
        v_pred=rollout["v"],
        spike_pred=rollout["spike"],
        v_true=v_true,
        spike_true=spike_true,
        dt=sweep.dt,
        w_Ca=model.w_Ca,
        loss=config,
    )
    timing_stats = spike_timing_stats(
        spike_true,
        rollout["spike"],
        dt=sweep.dt,
        match_window_ms=config.spike_match_window_ms,
    )
    refractory_bins = int(round(config.post_spike_mask_ms / sweep.dt))
    voltage_mask = mask_post_spike_voltage_samples(
        spike_true,
        refractory_bins=refractory_bins,
    )
    v_true_np = _to_1d_numpy(v_true)
    v_pred_np = _to_1d_numpy(rollout["v"])
    spike_true_np = _to_1d_numpy(spike_true)
    spike_pred_np = _to_1d_numpy(rollout["spike"])
    i_soma_np = _to_1d_numpy(i_soma)
    time_ms = np.arange(v_true_np.shape[0], dtype=np.float64) * float(sweep.dt)

    mask_np = _to_1d_numpy(voltage_mask.to(dtype=torch.float32)) > 0.5
    if mask_np.any():
        masked_error = v_pred_np[mask_np] - v_true_np[mask_np]
        voltage_rmse = float(np.sqrt(np.mean(masked_error**2)))
        target_masked = v_true_np[mask_np]
        ss_res = float(np.sum(masked_error**2))
        ss_tot = float(np.sum((target_masked - target_masked.mean()) ** 2))
        voltage_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    else:
        voltage_rmse = float("nan")
        voltage_r2 = float("nan")

    spike_mae = float(np.mean(np.abs(spike_pred_np - spike_true_np)))
    spike_count_true = int((spike_true_np > 0.5).sum())
    spike_count_pred = int((spike_pred_np > 0.5).sum())
    spike_count_error = abs(spike_count_pred - spike_count_true)
    f1_score = _f1_score_from_binary_traces(spike_true_np, spike_pred_np)

    metrics = {
        **_loss_floats(losses),
        "voltage_rmse": voltage_rmse,
        "voltage_r2": voltage_r2,
        "spike_mae": spike_mae,
        "spike_f1": f1_score,
        "spike_count_true": float(spike_count_true),
        "spike_count_pred": float(spike_count_pred),
        "spike_count_error": float(spike_count_error),
        "spike_timing_f1": timing_stats["timing_f1"],
        "spike_precision": timing_stats["precision"],
        "spike_recall": timing_stats["recall"],
        "matched_spikes": timing_stats["matched_spikes"],
        "false_positive_spikes": timing_stats["false_positive_spikes"],
        "false_negative_spikes": timing_stats["false_negative_spikes"],
        "mean_timing_error_ms": timing_stats["mean_timing_error_ms"],
    }
    return FitEvaluation(
        specimen_id=sweep.specimen_id,
        sweep_number=sweep.sweep_number,
        dt=sweep.dt,
        metrics=metrics,
        traces={
            "time_ms": time_ms,
            "i_soma": i_soma_np,
            "v_true": v_true_np,
            "v_pred": v_pred_np,
            "spike_true": spike_true_np,
            "spike_pred": spike_pred_np,
        },
    )


def evaluate_fit_across_sweeps(
    model: TwoCompartmentModel,
    sweeps: Iterable[AllenSweepBatch],
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    loss: FitLossConfig | None = None,
) -> tuple[list[FitEvaluation], dict[str, float]]:
    """Evaluate a fitted model on multiple sweeps and aggregate metrics.

    Args:
        model: Fitted two-compartment neuron.
        sweeps: Sweeps to evaluate; at least one is required.
        device: Torch device; defaults to the model's device.
        dtype: Torch dtype used for the rollouts.
        loss: Loss weights and spike-matching settings (defaults to
            :class:`FitLossConfig`).

    Returns:
        Per-sweep evaluations and the aggregate (``mean_*`` plus per regime
        ``silent_/lowrate_/spiking_`` means) metric dictionary.

    Raises:
        ValueError: If ``sweeps`` is empty.
    """
    evaluations = [
        evaluate_two_compartment_fit(
            model, sweep, device=device, dtype=dtype, loss=loss
        )
        for sweep in sweeps
    ]
    if not evaluations:
        raise ValueError("At least one sweep is required for evaluation.")

    metric_names = evaluations[0].metrics.keys()
    aggregate = {
        f"mean_{name}": float(np.mean([ev.metrics[name] for ev in evaluations]))
        for name in metric_names
    }
    group_filters = {
        "silent": [
            ev for ev in evaluations if float(ev.metrics["spike_count_true"]) == 0.0
        ],
        "spiking": [
            ev for ev in evaluations if float(ev.metrics["spike_count_true"]) > 0.0
        ],
        "lowrate": [
            ev
            for ev in evaluations
            if 0.0 < float(ev.metrics["spike_count_true"]) <= 5.0
        ],
    }
    aggregate["n_sweeps"] = float(len(evaluations))
    for group_name, group in group_filters.items():
        aggregate[f"n_{group_name}_sweeps"] = float(len(group))
        if not group:
            continue
        for name in metric_names:
            aggregate[f"{group_name}_mean_{name}"] = float(
                np.mean([ev.metrics[name] for ev in group])
            )
    return evaluations, aggregate
