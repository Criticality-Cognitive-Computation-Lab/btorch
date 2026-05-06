"""Allen-data fitting utilities for the two-compartment GLIF neuron."""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import optimize
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from btorch.models import environ, functional


matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class AllenSweepBatch:
    """Preprocessed Allen sweep tensors for fitting.

    Args:
        specimen_id: Allen specimen identifier.
        sweep_number: Sweep number within the NWB file.
        dt_ms: Simulation timestep in milliseconds.
        i_soma: Somatic current tensor shaped ``(T, B, N)``.
        v_true: Recorded voltage tensor shaped ``(T, B, N)``.
        spike_true: Binary spike tensor shaped ``(T, B, N)``.
        i_apical: Optional apical input tensor shaped ``(T, B, N)``.
        metadata: Auxiliary Allen metadata retained for bookkeeping.
    """

    specimen_id: int
    sweep_number: int
    dt_ms: float
    i_soma: Tensor
    v_true: Tensor
    spike_true: Tensor
    i_apical: Tensor | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class FitEvaluation:
    """Evaluation artifacts for a fitted model on one sweep."""

    specimen_id: int
    sweep_number: int
    dt_ms: float
    metrics: dict[str, float]
    traces: dict[str, np.ndarray]


@dataclass(frozen=True)
class TwoCompartmentFitStage:
    """Configuration for one stage of staged parameter fitting."""

    name: str
    trainable_params: frozenset[str]
    sweep_kind: Literal["all", "silent", "lowrate", "spiking", "countcal"]
    voltage_weight: float
    spike_weight: float
    spike_count_weight: float
    spike_timing_weight: float
    sparsity_weight: float
    spike_count_over_weight: float = 1.0
    spike_count_under_weight: float = 1.0
    param_bounds: dict[str, tuple[float, float]] | None = None


DEFAULT_TWO_COMPARTMENT_FIT_STAGES: tuple[TwoCompartmentFitStage, ...] = (
    TwoCompartmentFitStage(
        name="passive",
        trainable_params=frozenset({"E_L", "tau_s"}),
        sweep_kind="silent",
        voltage_weight=1.0,
        spike_weight=0.0,
        spike_count_weight=10.0,
        spike_timing_weight=0.0,
        spike_count_over_weight=8.0,
        spike_count_under_weight=1.0,
        sparsity_weight=0.0,
        param_bounds={
            "E_L": (-82.0, -65.0),
            "tau_s": (8.0, 60.0),
        },
    ),
    TwoCompartmentFitStage(
        name="threshold_reset",
        trainable_params=frozenset(
            {"tau_s", "E_L", "R_s", "v_threshold", "v_reset", "delta_T"}
        ),
        sweep_kind="countcal",
        voltage_weight=0.02,
        spike_weight=1.0,
        spike_count_weight=50.0,
        spike_timing_weight=12.0,
        spike_count_over_weight=2.0,
        spike_count_under_weight=5.0,
        sparsity_weight=0.0,
        param_bounds={
            "tau_s": (6.0, 40.0),
            "E_L": (-78.0, -66.0),
            "R_s": (0.04, 0.35),
            "v_threshold": (-50.0, -42.0),
            "v_reset": (-78.0, -58.0),
            "delta_T": (0.0, 12.0),
        },
    ),
    TwoCompartmentFitStage(
        name="spike_init",
        trainable_params=frozenset({"tau_s", "E_L", "R_s", "v_threshold", "delta_T"}),
        sweep_kind="spiking",
        voltage_weight=0.01,
        spike_weight=1.0,
        spike_count_weight=60.0,
        spike_timing_weight=15.0,
        spike_count_over_weight=2.0,
        spike_count_under_weight=6.0,
        sparsity_weight=0.0,
        param_bounds={
            "tau_s": (5.0, 35.0),
            "E_L": (-78.0, -64.0),
            "R_s": (0.05, 0.4),
            "v_threshold": (-52.0, -40.0),
            "delta_T": (0.5, 15.0),
        },
    ),
    TwoCompartmentFitStage(
        name="adaptation",
        trainable_params=frozenset({"tau_a", "tau_th", "delta_th"}),
        sweep_kind="spiking",
        voltage_weight=0.05,
        spike_weight=1.0,
        spike_count_weight=35.0,
        spike_timing_weight=10.0,
        spike_count_over_weight=2.0,
        spike_count_under_weight=4.0,
        sparsity_weight=0.0,
        param_bounds={
            "tau_a": (20.0, 250.0),
            "tau_th": (5.0, 250.0),
            "delta_th": (0.0, 15.0),
        },
    ),
    TwoCompartmentFitStage(
        name="coupling",
        trainable_params=frozenset({"w_Ca", "theta_Ca", "w_sa", "w_as"}),
        sweep_kind="all",
        voltage_weight=0.02,
        spike_weight=1.0,
        spike_count_weight=25.0,
        spike_timing_weight=10.0,
        spike_count_over_weight=3.0,
        spike_count_under_weight=2.0,
        sparsity_weight=2e-3,
        param_bounds={
            "w_Ca": (0.0, 1.0),
            "theta_Ca": (0.0, 8.0),
            "w_sa": (0.0, 1.2),
            "w_as": (0.0, 1.2),
        },
    ),
)


@dataclass(frozen=True)
class _ParameterSlice:
    """Packed parameter metadata for vector-based optimization."""

    name: str
    start: int
    stop: int
    shape: torch.Size


DEFAULT_TWO_COMPARTMENT_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    # Practical priors for mouse VISp layer-5 pyramidal-cell fitting with
    # current in pA and voltage in mV. These are intentionally conservative:
    # broad enough for real biological variability, but tight enough to stop
    # the optimizer from drifting into obviously implausible firing regimes.
    "tau_s": (5.0, 80.0),
    "R_s": (0.01, 0.5),
    "E_L": (-85.0, -55.0),
    "tau_a": (20.0, 400.0),
    "tau_th": (5.0, 400.0),
    "delta_th": (0.0, 20.0),
    "delta_T": (0.0, 10.0),
    "w_Ca": (0.0, 3.0),
    "theta_Ca": (-5.0, 15.0),
    "w_sa": (0.0, 3.0),
    "w_as": (0.0, 3.0),
    "v_threshold": (-55.0, -35.0),
    "v_reset": (-80.0, -50.0),
}


def _require_allensdk():
    try:
        from allensdk.core.cell_types_cache import CellTypesCache
    except ImportError as exc:
        raise ImportError(
            "AllenSDK is required for Allen Brain Cell Types access. "
            "Install it with `pip install allensdk` before using this "
            "pipeline."
        ) from exc
    return CellTypesCache


def get_cell_types_cache(
    manifest_file: str | Path | None = None,
    *,
    cache: Any | None = None,
):
    """Create or reuse an AllenSDK ``CellTypesCache`` instance."""
    if cache is not None:
        return cache
    CellTypesCache = _require_allensdk()
    if manifest_file is None:
        return CellTypesCache()
    return CellTypesCache(manifest_file=str(manifest_file))


def _matches_any(value: Any, expected: Sequence[str]) -> bool:
    text = str(value).lower()
    return any(item.lower() in text for item in expected)


def filter_mouse_visp_l5_pyramidal_cells(
    cells: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter Allen cell metadata to mouse VISp layer-5 pyramidal candidates.

    Notes:
        Allen metadata can vary across cache versions, so this filter accepts a
        few equivalent key names and uses ``spiny`` as a practical proxy for
        pyramidal morphology when an explicit class label is unavailable.
    """
    matched = []
    for cell in cells:
        species = cell.get("species") or cell.get("donor__species")
        area = (
            cell.get("structure_area_abbrev")
            or cell.get("structure_acronym")
            or cell.get("structure_parent__acronym")
        )
        layer = cell.get("structure_layer_name") or cell.get("cortical_layer")
        dendrite = cell.get("dendrite_type") or cell.get("cell_type")

        if not _matches_any(species, ["mouse", "mus musculus"]):
            continue
        if not _matches_any(area, ["visp", "primary visual cortex"]):
            continue
        if not _matches_any(layer, ["layer 5", "5"]):
            continue
        if not _matches_any(dendrite, ["spiny", "pyramidal"]):
            continue
        matched.append(cell)
    return matched


def query_mouse_visp_l5_pyramidal_cells(
    *,
    manifest_file: str | Path | None = None,
    cache: Any | None = None,
) -> list[dict[str, Any]]:
    """Query candidate mouse VISp layer-5 pyramidal cells from AllenSDK."""
    ctc = get_cell_types_cache(manifest_file, cache=cache)
    return filter_mouse_visp_l5_pyramidal_cells(ctc.get_cells())


def choose_current_clamp_sweeps(
    sweep_records: Sequence[dict[str, Any]],
    *,
    preferred_stimuli: Sequence[str] = ("long square",),
) -> list[dict[str, Any]]:
    """Select current-clamp sweeps appropriate for somatic fitting."""
    matched = []
    for sweep in sweep_records:
        stimulus_name = (
            sweep.get("stimulus_name")
            or sweep.get("ephys_stimulus", {}).get("description")
            or ""
        )
        stimulus_units = sweep.get("stimulus_units") or ""
        clamp_mode = sweep.get("clamp_mode") or sweep.get("stimulus_type_name") or ""

        is_current_clamp = _matches_any(stimulus_units, ["pa", "amp"]) or (
            clamp_mode and not _matches_any(clamp_mode, ["voltage clamp"])
        )
        if not is_current_clamp:
            continue
        if preferred_stimuli and not _matches_any(stimulus_name, preferred_stimuli):
            continue
        matched.append(sweep)
    return matched


def detect_spikes_from_voltage(
    voltage: Tensor,
    *,
    threshold: float = 0.0,
) -> Tensor:
    """Detect upward threshold crossings in a voltage trace."""
    above = voltage >= threshold
    shifted = torch.zeros_like(above)
    shifted[1:] = above[:-1]
    return (above & ~shifted).to(voltage.dtype)


def resample_trace(
    trace: np.ndarray | Tensor,
    *,
    source_dt_ms: float,
    target_dt_ms: float,
) -> Tensor:
    """Resample a 1D trace to a new timestep using linear interpolation."""
    if target_dt_ms <= 0.0:
        raise ValueError(f"target_dt_ms must be positive, got {target_dt_ms}.")
    trace_t = torch.as_tensor(trace, dtype=torch.float32)
    if trace_t.ndim != 1:
        raise ValueError(f"Expected a 1D trace, got shape {tuple(trace_t.shape)}.")
    if abs(source_dt_ms - target_dt_ms) < 1e-12:
        return trace_t

    duration_ms = max((trace_t.shape[0] - 1) * source_dt_ms, 0.0)
    target_steps = max(int(round(duration_ms / target_dt_ms)) + 1, 1)
    source_time = torch.linspace(0.0, duration_ms, trace_t.shape[0])
    target_time = torch.linspace(0.0, duration_ms, target_steps)

    np_resampled = np.interp(
        target_time.cpu().numpy(),
        source_time.cpu().numpy(),
        trace_t.cpu().numpy(),
    )
    return torch.from_numpy(np_resampled).to(dtype=torch.float32)


def load_allen_sweep(
    specimen_id: int,
    sweep_number: int,
    *,
    dt_ms: float = 0.5,
    manifest_file: str | Path | None = None,
    cache: Any | None = None,
    voltage_spike_threshold: float = 0.0,
    voltage_scale: float = 1e3,
    current_scale: float = 1e12,
) -> AllenSweepBatch:
    """Load one Allen ephys sweep and convert it to time-first torch tensors.

    Notes:
        AllenSDK current-clamp sweeps are typically returned in SI units
        (volts and amps). The two-compartment neuron parameters in this module
        use biologically familiar millivolt-scale voltages and pA-scale input
        magnitudes, so the default conversion is volts -> mV and amps -> pA.
    """
    ctc = get_cell_types_cache(manifest_file, cache=cache)
    dataset = ctc.get_ephys_data(specimen_id)
    sweep = dataset.get_sweep(sweep_number)

    response = np.asarray(sweep["response"], dtype=np.float32)
    stimulus = np.asarray(sweep["stimulus"], dtype=np.float32)
    sampling_rate = float(sweep["sampling_rate"])
    index_start, index_stop = sweep["index_range"]
    sl = slice(int(index_start), int(index_stop) + 1)

    source_dt_ms = 1000.0 / sampling_rate
    voltage = resample_trace(
        response[sl],
        source_dt_ms=source_dt_ms,
        target_dt_ms=dt_ms,
    )
    current = resample_trace(
        stimulus[sl],
        source_dt_ms=source_dt_ms,
        target_dt_ms=dt_ms,
    )
    voltage = voltage * float(voltage_scale)
    current = current * float(current_scale)
    spike_true = detect_spikes_from_voltage(
        voltage,
        threshold=voltage_spike_threshold,
    )

    return AllenSweepBatch(
        specimen_id=specimen_id,
        sweep_number=sweep_number,
        dt_ms=dt_ms,
        i_soma=current[:, None, None],
        v_true=voltage[:, None, None],
        spike_true=spike_true[:, None, None],
        i_apical=torch.zeros_like(current)[:, None, None],
        metadata={
            "sampling_rate_hz": sampling_rate,
            "voltage_scale": voltage_scale,
            "current_scale": current_scale,
        },
    )


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
    dt_ms: float,
) -> Tensor:
    """Apply a causal exponential filter to a spike train."""
    if tau_ms <= 0.0:
        raise ValueError(f"tau_ms must be positive, got {tau_ms}.")
    alpha = float(np.exp(-dt_ms / tau_ms))
    filtered = torch.zeros_like(spike_train)
    filtered[0] = spike_train[0]
    for t in range(1, spike_train.shape[0]):
        filtered[t] = alpha * filtered[t - 1] + (1.0 - alpha) * spike_train[t]
    return filtered


def _extract_spike_indices(spike_train: Tensor) -> np.ndarray:
    """Extract hard spike-event indices from a `(T, B, N)` spike tensor."""
    trace = _to_1d_numpy(spike_train)
    return np.flatnonzero(trace > 0.5).astype(np.int64)


def spike_timing_stats(
    spike_true: Tensor,
    spike_pred: Tensor,
    *,
    dt_ms: float,
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

    distance_ms = (
        np.abs(true_idx[:, None] - pred_idx[None, :]).astype(np.float64) * dt_ms
    )
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
    dt_ms: float,
    match_window_ms: float = 10.0,
    miss_penalty_ms: float | None = None,
) -> Tensor:
    """Compute a hard event-based spike-timing loss for global optimization."""
    penalty_ms = float(match_window_ms if miss_penalty_ms is None else miss_penalty_ms)
    stats = spike_timing_stats(
        spike_true,
        spike_pred,
        dt_ms=dt_ms,
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
    dt_ms: float,
    w_Ca: Tensor | None = None,
    voltage_weight: float = 1.0,
    spike_weight: float = 1.0,
    spike_count_weight: float = 0.0,
    spike_timing_weight: float = 0.0,
    spike_count_over_weight: float = 1.0,
    spike_count_under_weight: float = 1.0,
    sparsity_weight: float = 1e-4,
    spike_tau_ms: float = 10.0,
    post_spike_mask_ms: float = 3.0,
    spike_match_window_ms: float = 10.0,
    spike_miss_penalty_ms: float | None = None,
) -> dict[str, Tensor]:
    """Compute a composite fitting loss for the two-compartment model."""
    refractory_bins = int(round(post_spike_mask_ms / dt_ms))
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
        tau_ms=spike_tau_ms,
        dt_ms=dt_ms,
    )
    spike_true_smooth = exponential_filter_spike_train(
        spike_true,
        tau_ms=spike_tau_ms,
        dt_ms=dt_ms,
    )
    spike_loss = F.smooth_l1_loss(spike_pred_smooth, spike_true_smooth)
    spike_count_pred = (spike_pred > 0.5).to(spike_pred.dtype).sum(dim=0)
    spike_count_true = spike_true.sum(dim=0)
    if spike_count_over_weight <= 0.0 or spike_count_under_weight <= 0.0:
        raise ValueError("Spike-count over/under weights must be positive.")
    spike_count_over = torch.relu(spike_count_pred - spike_count_true)
    spike_count_under = torch.relu(spike_count_true - spike_count_pred)
    spike_count_loss = (
        spike_count_over_weight * spike_count_over.square()
        + spike_count_under_weight * spike_count_under.square()
    ).mean()
    spike_timing_event_loss = spike_timing_loss(
        spike_true,
        spike_pred,
        dt_ms=dt_ms,
        match_window_ms=spike_match_window_ms,
        miss_penalty_ms=spike_miss_penalty_ms,
    )

    if w_Ca is None:
        sparsity_loss = torch.zeros((), device=v_pred.device, dtype=v_pred.dtype)
    else:
        sparsity_loss = w_Ca.abs().mean()

    total = (
        voltage_weight * voltage_loss
        + spike_weight * spike_loss
        + spike_count_weight * spike_count_loss
        + spike_timing_weight * spike_timing_event_loss
        + sparsity_weight * sparsity_loss
    )
    return {
        "total": total,
        "voltage": voltage_loss,
        "spike": spike_loss,
        "spike_count": spike_count_loss,
        "spike_timing": spike_timing_event_loss,
        "sparsity": sparsity_loss,
    }


def rollout_two_compartment(
    model,
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


def _to_1d_numpy(x: Tensor) -> np.ndarray:
    """Convert a `(T, B, N)` or compatible tensor to a 1D CPU numpy trace."""
    x_cpu = x.detach().cpu()
    if x_cpu.ndim == 3:
        return x_cpu[:, 0, 0].numpy()
    if x_cpu.ndim == 1:
        return x_cpu.numpy()
    raise ValueError(f"Expected a 1D or (T, B, N) tensor, got {tuple(x_cpu.shape)}.")


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


def evaluate_two_compartment_fit(
    model,
    sweep: AllenSweepBatch,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    spike_tau_ms: float = 10.0,
    post_spike_mask_ms: float = 3.0,
    spike_count_weight: float = 0.0,
    spike_timing_weight: float = 0.0,
    spike_count_over_weight: float = 1.0,
    spike_count_under_weight: float = 1.0,
    spike_match_window_ms: float = 10.0,
    spike_miss_penalty_ms: float | None = None,
) -> FitEvaluation:
    """Evaluate a fitted model on a single sweep and collect metrics."""
    i_soma = sweep.i_soma.to(device=device, dtype=dtype)
    v_true = sweep.v_true.to(device=device, dtype=dtype)
    spike_true = sweep.spike_true.to(device=device, dtype=dtype)
    i_apical = None
    if sweep.i_apical is not None:
        i_apical = sweep.i_apical.to(device=device, dtype=dtype)

    functional.init_net_state(
        model,
        batch_size=i_soma.shape[1],
        device=device,
        dtype=dtype,
    )
    functional.reset_net(
        model,
        batch_size=i_soma.shape[1],
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        with environ.context(dt=float(sweep.dt_ms)):
            rollout = rollout_two_compartment(model, i_soma, i_apical)

    losses = two_compartment_loss(
        v_pred=rollout["v"],
        spike_pred=rollout["spike"],
        v_true=v_true,
        spike_true=spike_true,
        dt_ms=sweep.dt_ms,
        w_Ca=getattr(model, "w_Ca", None),
        spike_tau_ms=spike_tau_ms,
        post_spike_mask_ms=post_spike_mask_ms,
        spike_count_weight=spike_count_weight,
        spike_timing_weight=spike_timing_weight,
        spike_count_over_weight=spike_count_over_weight,
        spike_count_under_weight=spike_count_under_weight,
        spike_match_window_ms=spike_match_window_ms,
        spike_miss_penalty_ms=spike_miss_penalty_ms,
    )
    timing_stats = spike_timing_stats(
        spike_true,
        rollout["spike"],
        dt_ms=sweep.dt_ms,
        match_window_ms=spike_match_window_ms,
    )
    refractory_bins = int(round(post_spike_mask_ms / sweep.dt_ms))
    voltage_mask = mask_post_spike_voltage_samples(
        spike_true,
        refractory_bins=refractory_bins,
    )
    v_true_np = _to_1d_numpy(v_true)
    v_pred_np = _to_1d_numpy(rollout["v"])
    spike_true_np = _to_1d_numpy(spike_true)
    spike_pred_np = _to_1d_numpy(rollout["spike"])
    i_soma_np = _to_1d_numpy(i_soma)
    time_ms = np.arange(v_true_np.shape[0], dtype=np.float64) * float(sweep.dt_ms)

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
        "total_loss": float(losses["total"].detach().cpu()),
        "voltage_loss": float(losses["voltage"].detach().cpu()),
        "spike_loss": float(losses["spike"].detach().cpu()),
        "spike_count_loss": float(losses["spike_count"].detach().cpu()),
        "spike_timing_loss": float(losses["spike_timing"].detach().cpu()),
        "sparsity_loss": float(losses["sparsity"].detach().cpu()),
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
        dt_ms=sweep.dt_ms,
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
    model,
    sweeps: Iterable[AllenSweepBatch],
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    spike_tau_ms: float = 10.0,
    post_spike_mask_ms: float = 3.0,
    spike_count_weight: float = 0.0,
    spike_timing_weight: float = 0.0,
    spike_count_over_weight: float = 1.0,
    spike_count_under_weight: float = 1.0,
    spike_match_window_ms: float = 10.0,
    spike_miss_penalty_ms: float | None = None,
) -> tuple[list[FitEvaluation], dict[str, float]]:
    """Evaluate a fitted model on multiple sweeps and aggregate metrics."""
    evaluations = [
        evaluate_two_compartment_fit(
            model,
            sweep,
            device=device,
            dtype=dtype,
            spike_tau_ms=spike_tau_ms,
            post_spike_mask_ms=post_spike_mask_ms,
            spike_count_weight=spike_count_weight,
            spike_timing_weight=spike_timing_weight,
            spike_count_over_weight=spike_count_over_weight,
            spike_count_under_weight=spike_count_under_weight,
            spike_match_window_ms=spike_match_window_ms,
            spike_miss_penalty_ms=spike_miss_penalty_ms,
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


def plot_two_compartment_fit(
    evaluation: FitEvaluation,
    history: Sequence[dict[str, float | str]] | None = None,
    *,
    output_path: str | Path,
) -> Path:
    """Create a compact fit-quality figure with traces and loss history."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    time_ms = evaluation.traces["time_ms"]

    axes[0].plot(time_ms, evaluation.traces["i_soma"], color="tab:blue", lw=1.2)
    axes[0].set_ylabel("I soma")
    axes[0].set_title(
        f"Specimen {evaluation.specimen_id}, sweep {evaluation.sweep_number}"
    )

    axes[1].plot(time_ms, evaluation.traces["v_true"], label="Recorded", lw=1.5)
    axes[1].plot(time_ms, evaluation.traces["v_pred"], label="Predicted", lw=1.2)
    axes[1].set_ylabel("Voltage")
    axes[1].legend(loc="best")

    spike_true_t = evaluation.traces["spike_true"]
    spike_pred_t = evaluation.traces["spike_pred"]
    axes[2].eventplot(
        [time_ms[spike_true_t > 0.5], time_ms[spike_pred_t > 0.5]],
        lineoffsets=[1, 0],
        linelengths=0.8,
        colors=["tab:green", "tab:red"],
    )
    axes[2].set_yticks([0, 1], ["Pred", "True"])
    axes[2].set_ylabel("Spikes")

    if history:
        x = np.arange(len(history))
        axes[3].plot(x, [float(row["total_loss"]) for row in history], label="Total")
        axes[3].plot(
            x,
            [float(row["voltage_loss"]) for row in history],
            label="Voltage",
        )
        axes[3].plot(
            x,
            [float(row["spike_loss"]) for row in history],
            label="Spike",
        )
        axes[3].legend(loc="best")
        axes[3].set_xlabel("Optimization step")
    else:
        axes[3].axis("off")

    axes[3].set_ylabel("Loss")
    metrics_text = "\n".join(
        f"{name}: {value:.4f}" for name, value in evaluation.metrics.items()
    )
    fig.text(
        0.99,
        0.5,
        metrics_text,
        va="center",
        ha="right",
        fontsize=9,
        family="monospace",
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def save_fit_report(
    model,
    evaluations: Sequence[FitEvaluation],
    aggregate_metrics: dict[str, float],
    history: Sequence[dict[str, float | str]],
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save fitted parameters, metrics, history, and plots to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parameter_payload = {
        name: param.detach().cpu().reshape(-1).tolist()
        for name, param in model.named_parameters()
    }
    metrics_payload = {
        "aggregate": aggregate_metrics,
        "per_sweep": [
            {
                "specimen_id": ev.specimen_id,
                "sweep_number": ev.sweep_number,
                "dt_ms": ev.dt_ms,
                "metrics": ev.metrics,
            }
            for ev in evaluations
        ],
    }

    parameters_path = out_dir / "fitted_parameters.json"
    metrics_path = out_dir / "fit_metrics.json"
    history_path = out_dir / "fit_history.json"
    with parameters_path.open("w", encoding="utf-8") as f:
        json.dump(parameter_payload, f, indent=2)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(list(history), f, indent=2)

    last_plot = None
    primary_plot = None
    for index, evaluation in enumerate(evaluations):
        plot_path = out_dir / (
            f"fit_specimen_{evaluation.specimen_id}_"
            f"sweep_{evaluation.sweep_number}_{index}.png"
        )
        last_plot = plot_two_compartment_fit(
            evaluation,
            history=history,
            output_path=plot_path,
        )
        if (
            primary_plot is None
            and evaluation.metrics.get("spike_count_true", 0.0) > 0.0
        ):
            primary_plot = last_plot

    if primary_plot is None:
        primary_plot = last_plot

    return {
        "output_dir": out_dir,
        "parameters": parameters_path,
        "metrics": metrics_path,
        "history": history_path,
        "primary_plot": primary_plot if primary_plot is not None else out_dir,
        "plot": last_plot if last_plot is not None else out_dir,
    }


def _fit_sweeps_once(
    model,
    sweeps: Iterable[AllenSweepBatch],
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    voltage_weight: float = 1.0,
    spike_weight: float = 1.0,
    spike_count_weight: float = 0.0,
    spike_timing_weight: float = 0.0,
    spike_count_over_weight: float = 1.0,
    spike_count_under_weight: float = 1.0,
    sparsity_weight: float = 1e-4,
    spike_tau_ms: float = 10.0,
    post_spike_mask_ms: float = 3.0,
    spike_match_window_ms: float = 10.0,
    spike_miss_penalty_ms: float | None = None,
) -> dict[str, float]:
    """Evaluate the current model parameters on one or more sweeps."""
    total_loss = 0.0
    total_voltage = 0.0
    total_spike = 0.0
    total_spike_count = 0.0
    total_spike_timing = 0.0
    total_sparsity = 0.0
    sweep_count = 0

    with torch.no_grad():
        for sweep in sweeps:
            i_soma = sweep.i_soma.to(device=device, dtype=dtype)
            v_true = sweep.v_true.to(device=device, dtype=dtype)
            spike_true = sweep.spike_true.to(device=device, dtype=dtype)
            i_apical = None
            if sweep.i_apical is not None:
                i_apical = sweep.i_apical.to(device=device, dtype=dtype)

            functional.init_net_state(
                model,
                batch_size=i_soma.shape[1],
                device=device,
                dtype=dtype,
            )
            functional.reset_net(
                model,
                batch_size=i_soma.shape[1],
                device=device,
                dtype=dtype,
            )

            with environ.context(dt=float(sweep.dt_ms)):
                rollout = rollout_two_compartment(model, i_soma, i_apical)
                losses = two_compartment_loss(
                    v_pred=rollout["v"],
                    spike_pred=rollout["spike"],
                    v_true=v_true,
                    spike_true=spike_true,
                    dt_ms=sweep.dt_ms,
                    w_Ca=getattr(model, "w_Ca", None),
                    voltage_weight=voltage_weight,
                    spike_weight=spike_weight,
                    spike_count_weight=spike_count_weight,
                    spike_timing_weight=spike_timing_weight,
                    spike_count_over_weight=spike_count_over_weight,
                    spike_count_under_weight=spike_count_under_weight,
                    sparsity_weight=sparsity_weight,
                    spike_tau_ms=spike_tau_ms,
                    post_spike_mask_ms=post_spike_mask_ms,
                    spike_match_window_ms=spike_match_window_ms,
                    spike_miss_penalty_ms=spike_miss_penalty_ms,
                )

            total_loss += float(losses["total"].detach().cpu())
            total_voltage += float(losses["voltage"].detach().cpu())
            total_spike += float(losses["spike"].detach().cpu())
            total_spike_count += float(losses["spike_count"].detach().cpu())
            total_spike_timing += float(losses["spike_timing"].detach().cpu())
            total_sparsity += float(losses["sparsity"].detach().cpu())
            sweep_count += 1

    if sweep_count == 0:
        raise ValueError("At least one sweep is required for fitting.")

    return {
        "total_loss": total_loss / sweep_count,
        "voltage_loss": total_voltage / sweep_count,
        "spike_loss": total_spike / sweep_count,
        "spike_count_loss": total_spike_count / sweep_count,
        "spike_timing_loss": total_spike_timing / sweep_count,
        "sparsity_loss": total_sparsity / sweep_count,
    }


def _pack_trainable_parameters(
    model,
) -> tuple[np.ndarray, list[_ParameterSlice]]:
    """Pack trainable model parameters into a flat numpy vector."""
    vector_parts: list[np.ndarray] = []
    slices: list[_ParameterSlice] = []
    cursor = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        flat = param.detach().reshape(-1).cpu().numpy().astype(np.float64)
        vector_parts.append(flat)
        size = int(flat.size)
        slices.append(
            _ParameterSlice(
                name=name,
                start=cursor,
                stop=cursor + size,
                shape=param.shape,
            )
        )
        cursor += size

    if not slices:
        raise ValueError("The model has no trainable parameters to optimize.")

    return np.concatenate(vector_parts), slices


def _set_trainable_parameters(
    model,
    vector: np.ndarray,
    slices: Sequence[_ParameterSlice],
) -> None:
    """Copy a packed parameter vector into the model in-place."""
    params = dict(model.named_parameters())
    for spec in slices:
        value = vector[spec.start : spec.stop]
        target = params[spec.name]
        value_t = torch.as_tensor(
            value.reshape(spec.shape),
            device=target.device,
            dtype=target.dtype,
        )
        with torch.no_grad():
            target.copy_(value_t)


def _build_parameter_bounds(
    model,
    slices: Sequence[_ParameterSlice],
    param_bounds: dict[str, tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Expand per-parameter bounds to match the packed parameter vector."""
    merged_bounds = dict(DEFAULT_TWO_COMPARTMENT_PARAM_BOUNDS)
    if param_bounds is not None:
        merged_bounds.update(param_bounds)

    bounds: list[tuple[float, float]] = []
    for spec in slices:
        if spec.name not in merged_bounds:
            raise KeyError(
                f"Missing bounds for trainable parameter '{spec.name}'. "
                "Provide param_bounds to override the defaults."
            )
        lower, upper = merged_bounds[spec.name]
        if lower >= upper:
            raise ValueError(f"Invalid bounds for {spec.name}: ({lower}, {upper}).")
        bounds.extend([(float(lower), float(upper))] * (spec.stop - spec.start))
    return bounds


def _original_trainable_parameter_names(model) -> set[str]:
    """Return the names of parameters currently marked trainable."""
    return {name for name, param in model.named_parameters() if param.requires_grad}


def _set_trainable_parameter_names(
    model,
    trainable_names: set[str],
) -> dict[str, bool]:
    """Temporarily change which parameters are trainable."""
    previous = {}
    for name, param in model.named_parameters():
        previous[name] = bool(param.requires_grad)
        param.requires_grad_(name in trainable_names)
    return previous


def _restore_trainable_parameter_names(
    model,
    previous: dict[str, bool],
) -> None:
    """Restore the previous trainable-state map."""
    for name, param in model.named_parameters():
        if name in previous:
            param.requires_grad_(previous[name])


def _spike_count_for_sweep(sweep: AllenSweepBatch) -> float:
    """Return the total number of true spikes in a sweep."""
    return float(sweep.spike_true.sum().item())


def _safe_metric(metrics: dict[str, float], key: str, default: float = 0.0) -> float:
    """Return a finite metric value or a fallback."""
    value = float(metrics.get(key, default))
    return value if np.isfinite(value) else default


def _best_available_metric(
    metrics: dict[str, float],
    keys: Sequence[str],
    default: float,
) -> float:
    """Return the first finite metric among the requested keys."""
    for key in keys:
        if key in metrics:
            value = _safe_metric(metrics, key, default)
            if np.isfinite(value):
                return value
    return default


def _filter_sweeps_for_stage(
    sweeps: Sequence[AllenSweepBatch],
    sweep_kind: Literal["all", "silent", "lowrate", "spiking", "countcal"],
) -> list[AllenSweepBatch]:
    """Select stage-appropriate sweeps."""
    if sweep_kind == "all":
        return list(sweeps)

    if sweep_kind == "silent":
        selected = [sweep for sweep in sweeps if _spike_count_for_sweep(sweep) == 0.0]
        return selected if selected else list(sweeps)

    if sweep_kind == "lowrate":
        selected = [
            sweep for sweep in sweeps if 0.0 < _spike_count_for_sweep(sweep) <= 5.0
        ]
        if selected:
            return selected
        selected = [sweep for sweep in sweeps if _spike_count_for_sweep(sweep) > 0.0]
        return selected if selected else list(sweeps)

    if sweep_kind == "spiking":
        selected = [sweep for sweep in sweeps if _spike_count_for_sweep(sweep) > 0.0]
        return selected if selected else list(sweeps)

    if sweep_kind == "countcal":
        silent = [sweep for sweep in sweeps if _spike_count_for_sweep(sweep) == 0.0]
        lowrate = [
            sweep for sweep in sweeps if 0.0 < _spike_count_for_sweep(sweep) <= 5.0
        ]
        if lowrate:
            return silent + lowrate

        spiking = [sweep for sweep in sweeps if _spike_count_for_sweep(sweep) > 0.0]
        if spiking:
            spiking.sort(key=_spike_count_for_sweep)
            selected = list(silent)
            selected.extend(spiking[: max(1, min(2, len(spiking)))])
            return selected

        return silent if silent else list(sweeps)

    raise ValueError(f"Unknown sweep_kind: {sweep_kind}.")


def _annotate_stage_history(
    history: Sequence[dict[str, float | str]],
    *,
    stage_name: str,
) -> list[dict[str, float | str]]:
    """Prefix the recorded phase with the stage name."""
    annotated = []
    for row in history:
        new_row = dict(row)
        new_row["stage"] = stage_name
        new_row["phase"] = f"{stage_name}:{row['phase']}"
        annotated.append(new_row)
    return annotated


def _snapshot_parameter_values(model) -> dict[str, Tensor]:
    """Clone current parameter tensors for potential rollback."""
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def _restore_parameter_values(
    model,
    snapshot: dict[str, Tensor],
) -> None:
    """Restore parameters from a previously captured snapshot."""
    for name, param in model.named_parameters():
        if name in snapshot:
            with torch.no_grad():
                param.copy_(snapshot[name].to(device=param.device, dtype=param.dtype))


def _stage_objective_score(metrics: dict[str, float]) -> tuple[float, float, float]:
    """Rank stage outcomes by firing regime first, then timing, then
    voltage."""
    count_error = _best_available_metric(
        metrics,
        (
            "lowrate_mean_spike_count_error",
            "spiking_mean_spike_count_error",
            "mean_spike_count_error",
        ),
        float("inf"),
    )
    false_negative = _best_available_metric(
        metrics,
        (
            "lowrate_mean_false_negative_spikes",
            "spiking_mean_false_negative_spikes",
            "mean_false_negative_spikes",
        ),
        float("inf"),
    )
    timing_f1 = _best_available_metric(
        metrics,
        (
            "lowrate_mean_spike_timing_f1",
            "spiking_mean_spike_timing_f1",
            "mean_spike_timing_f1",
        ),
        0.0,
    )
    return (
        _safe_metric(metrics, "silent_mean_false_positive_spikes"),
        count_error,
        false_negative,
        -timing_f1,
        _safe_metric(metrics, "mean_voltage_rmse", float("inf")),
    )


def _fit_two_compartment_model_tbptt(
    model,
    sweeps: Iterable[AllenSweepBatch],
    *,
    lr: float = 1e-3,
    epochs: int = 10,
    chunk_size: int = 500,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    voltage_weight: float = 1.0,
    spike_weight: float = 1.0,
    spike_count_weight: float = 0.0,
    spike_timing_weight: float = 0.0,
    spike_count_over_weight: float = 1.0,
    spike_count_under_weight: float = 1.0,
    sparsity_weight: float = 1e-4,
    spike_tau_ms: float = 10.0,
    post_spike_mask_ms: float = 3.0,
    spike_match_window_ms: float = 10.0,
    spike_miss_penalty_ms: float | None = None,
) -> list[dict[str, float | str]]:
    """Fit the model to Allen sweeps with truncated BPTT.

    Notes:
        Each sweep is reset once at the beginning, then processed in
        ``chunk_size`` timesteps. ``functional.detach_net`` is called between
        chunks so the hidden state carries over while the graph stays bounded.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    initialized = False

    for epoch in range(epochs):
        for sweep in sweeps:
            i_soma = sweep.i_soma.to(device=device, dtype=dtype)
            v_true = sweep.v_true.to(device=device, dtype=dtype)
            spike_true = sweep.spike_true.to(device=device, dtype=dtype)
            i_apical = None
            if sweep.i_apical is not None:
                i_apical = sweep.i_apical.to(device=device, dtype=dtype)

            if not initialized:
                functional.init_net_state(
                    model,
                    batch_size=i_soma.shape[1],
                    device=device,
                    dtype=dtype,
                )
                initialized = True

            functional.reset_net(
                model,
                batch_size=i_soma.shape[1],
                device=device,
                dtype=dtype,
            )
            optimizer.zero_grad()

            with environ.context(dt=float(sweep.dt_ms)):
                for start in range(0, i_soma.shape[0], chunk_size):
                    if start > 0:
                        functional.detach_net(model)
                    stop = min(start + chunk_size, i_soma.shape[0])

                    rollout = rollout_two_compartment(
                        model,
                        i_soma[start:stop],
                        None if i_apical is None else i_apical[start:stop],
                    )
                    losses = two_compartment_loss(
                        v_pred=rollout["v"],
                        spike_pred=rollout["spike"],
                        v_true=v_true[start:stop],
                        spike_true=spike_true[start:stop],
                        dt_ms=sweep.dt_ms,
                        w_Ca=getattr(model, "w_Ca", None),
                        voltage_weight=voltage_weight,
                        spike_weight=spike_weight,
                        spike_count_weight=spike_count_weight,
                        spike_timing_weight=spike_timing_weight,
                        spike_count_over_weight=spike_count_over_weight,
                        spike_count_under_weight=spike_count_under_weight,
                        sparsity_weight=sparsity_weight,
                        spike_tau_ms=spike_tau_ms,
                        post_spike_mask_ms=post_spike_mask_ms,
                        spike_match_window_ms=spike_match_window_ms,
                        spike_miss_penalty_ms=spike_miss_penalty_ms,
                    )
                    losses["total"].backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    history.append(
                        {
                            "phase": "tbptt",
                            "epoch": float(epoch),
                            "specimen_id": float(sweep.specimen_id),
                            "sweep_number": float(sweep.sweep_number),
                            "chunk_start": float(start),
                            "total_loss": float(losses["total"].detach().cpu()),
                            "voltage_loss": float(losses["voltage"].detach().cpu()),
                            "spike_loss": float(losses["spike"].detach().cpu()),
                            "spike_count_loss": float(
                                losses["spike_count"].detach().cpu()
                            ),
                            "spike_timing_loss": float(
                                losses["spike_timing"].detach().cpu()
                            ),
                            "sparsity_loss": float(losses["sparsity"].detach().cpu()),
                        }
                    )
    return history


def _fit_two_compartment_model_global(
    model,
    sweeps: Iterable[AllenSweepBatch],
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    voltage_weight: float = 1.0,
    spike_weight: float = 1.0,
    spike_count_weight: float = 0.0,
    spike_timing_weight: float = 0.0,
    spike_count_over_weight: float = 1.0,
    spike_count_under_weight: float = 1.0,
    sparsity_weight: float = 1e-4,
    spike_tau_ms: float = 10.0,
    post_spike_mask_ms: float = 3.0,
    spike_match_window_ms: float = 10.0,
    spike_miss_penalty_ms: float | None = None,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    global_maxiter: int = 20,
    global_popsize: int = 8,
    local_maxiter: int = 50,
    seed: int | None = 0,
    polish: bool = True,
) -> list[dict[str, float | str]]:
    """Fit with bounded global search and optional local L-BFGS-B polish."""
    sweep_list = list(sweeps)
    x0, slices = _pack_trainable_parameters(model)
    bounds = _build_parameter_bounds(model, slices, param_bounds=param_bounds)
    history: list[dict[str, float | str]] = []

    def objective(x: np.ndarray) -> float:
        _set_trainable_parameters(model, x, slices)
        metrics = _fit_sweeps_once(
            model,
            sweep_list,
            device=device,
            dtype=dtype,
            voltage_weight=voltage_weight,
            spike_weight=spike_weight,
            spike_count_weight=spike_count_weight,
            spike_timing_weight=spike_timing_weight,
            spike_count_over_weight=spike_count_over_weight,
            spike_count_under_weight=spike_count_under_weight,
            sparsity_weight=sparsity_weight,
            spike_tau_ms=spike_tau_ms,
            post_spike_mask_ms=post_spike_mask_ms,
            spike_match_window_ms=spike_match_window_ms,
            spike_miss_penalty_ms=spike_miss_penalty_ms,
        )
        return metrics["total_loss"]

    global_result = optimize.differential_evolution(
        objective,
        bounds=bounds,
        maxiter=global_maxiter,
        popsize=global_popsize,
        seed=seed,
        polish=False,
        updating="deferred",
    )
    _set_trainable_parameters(model, global_result.x, slices)
    metrics = _fit_sweeps_once(
        model,
        sweep_list,
        device=device,
        dtype=dtype,
        voltage_weight=voltage_weight,
        spike_weight=spike_weight,
        spike_count_weight=spike_count_weight,
        spike_timing_weight=spike_timing_weight,
        spike_count_over_weight=spike_count_over_weight,
        spike_count_under_weight=spike_count_under_weight,
        sparsity_weight=sparsity_weight,
        spike_tau_ms=spike_tau_ms,
        post_spike_mask_ms=post_spike_mask_ms,
        spike_match_window_ms=spike_match_window_ms,
        spike_miss_penalty_ms=spike_miss_penalty_ms,
    )
    history.append(
        {
            "phase": "global",
            "epoch": 0.0,
            "specimen_id": -1.0,
            "sweep_number": -1.0,
            "chunk_start": -1.0,
            "total_loss": metrics["total_loss"],
            "voltage_loss": metrics["voltage_loss"],
            "spike_loss": metrics["spike_loss"],
            "spike_count_loss": metrics["spike_count_loss"],
            "spike_timing_loss": metrics["spike_timing_loss"],
            "sparsity_loss": metrics["sparsity_loss"],
        }
    )

    if not polish:
        return history

    local_result = optimize.minimize(
        objective,
        global_result.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(local_maxiter)},
    )
    _set_trainable_parameters(model, local_result.x, slices)
    metrics = _fit_sweeps_once(
        model,
        sweep_list,
        device=device,
        dtype=dtype,
        voltage_weight=voltage_weight,
        spike_weight=spike_weight,
        spike_count_weight=spike_count_weight,
        spike_timing_weight=spike_timing_weight,
        spike_count_over_weight=spike_count_over_weight,
        spike_count_under_weight=spike_count_under_weight,
        sparsity_weight=sparsity_weight,
        spike_tau_ms=spike_tau_ms,
        post_spike_mask_ms=post_spike_mask_ms,
        spike_match_window_ms=spike_match_window_ms,
        spike_miss_penalty_ms=spike_miss_penalty_ms,
    )
    history.append(
        {
            "phase": "local",
            "epoch": 1.0,
            "specimen_id": -1.0,
            "sweep_number": -1.0,
            "chunk_start": -1.0,
            "total_loss": metrics["total_loss"],
            "voltage_loss": metrics["voltage_loss"],
            "spike_loss": metrics["spike_loss"],
            "spike_count_loss": metrics["spike_count_loss"],
            "spike_timing_loss": metrics["spike_timing_loss"],
            "sparsity_loss": metrics["sparsity_loss"],
        }
    )
    return history


def _fit_two_compartment_model_staged(
    model,
    sweeps: Iterable[AllenSweepBatch],
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    voltage_weight: float = 1.0,
    spike_weight: float = 1.0,
    spike_count_weight: float = 0.0,
    spike_timing_weight: float = 0.0,
    spike_count_over_weight: float = 1.0,
    spike_count_under_weight: float = 1.0,
    sparsity_weight: float = 1e-4,
    spike_tau_ms: float = 10.0,
    post_spike_mask_ms: float = 3.0,
    spike_match_window_ms: float = 10.0,
    spike_miss_penalty_ms: float | None = None,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    global_maxiter: int = 20,
    global_popsize: int = 8,
    local_maxiter: int = 50,
    seed: int | None = 0,
    polish: bool = True,
    stages: Sequence[TwoCompartmentFitStage] | None = None,
) -> list[dict[str, float | str]]:
    """Fit the model in identifiable stages with bounded search."""
    sweep_list = list(sweeps)
    if not sweep_list:
        raise ValueError("At least one sweep is required for fitting.")

    active_stages = (
        tuple(stages) if stages is not None else DEFAULT_TWO_COMPARTMENT_FIT_STAGES
    )
    original_trainable = _original_trainable_parameter_names(model)
    history: list[dict[str, float | str]] = []
    _, best_metrics = evaluate_fit_across_sweeps(
        model,
        sweep_list,
        device=device,
        dtype=dtype,
        spike_count_weight=spike_count_weight,
        spike_timing_weight=spike_timing_weight,
        spike_count_over_weight=spike_count_over_weight,
        spike_count_under_weight=spike_count_under_weight,
        spike_match_window_ms=spike_match_window_ms,
        spike_miss_penalty_ms=spike_miss_penalty_ms,
    )
    best_score = _stage_objective_score(best_metrics)

    for stage_index, stage in enumerate(active_stages):
        stage_sweeps = _filter_sweeps_for_stage(sweep_list, stage.sweep_kind)
        stage_trainable = original_trainable.intersection(stage.trainable_params)
        if not stage_trainable:
            continue

        stage_bounds = dict(param_bounds or {})
        if stage.param_bounds is not None:
            stage_bounds.update(stage.param_bounds)

        snapshot = _snapshot_parameter_values(model)
        previous = _set_trainable_parameter_names(model, stage_trainable)
        try:
            stage_history = _fit_two_compartment_model_global(
                model,
                stage_sweeps,
                device=device,
                dtype=dtype,
                voltage_weight=stage.voltage_weight,
                spike_weight=stage.spike_weight,
                spike_count_weight=stage.spike_count_weight,
                spike_timing_weight=stage.spike_timing_weight,
                spike_count_over_weight=stage.spike_count_over_weight,
                spike_count_under_weight=stage.spike_count_under_weight,
                sparsity_weight=stage.sparsity_weight,
                spike_tau_ms=spike_tau_ms,
                post_spike_mask_ms=post_spike_mask_ms,
                spike_match_window_ms=spike_match_window_ms,
                spike_miss_penalty_ms=spike_miss_penalty_ms,
                param_bounds=stage_bounds,
                global_maxiter=global_maxiter,
                global_popsize=global_popsize,
                local_maxiter=local_maxiter,
                seed=None if seed is None else seed + stage_index,
                polish=polish,
            )
        finally:
            _restore_trainable_parameter_names(model, previous)

        _, stage_metrics = evaluate_fit_across_sweeps(
            model,
            sweep_list,
            device=device,
            dtype=dtype,
            spike_count_weight=spike_count_weight,
            spike_timing_weight=spike_timing_weight,
            spike_count_over_weight=spike_count_over_weight,
            spike_count_under_weight=spike_count_under_weight,
            spike_match_window_ms=spike_match_window_ms,
            spike_miss_penalty_ms=spike_miss_penalty_ms,
        )
        stage_score = _stage_objective_score(stage_metrics)

        if stage_score <= best_score:
            best_score = stage_score
            best_metrics = stage_metrics
            history.extend(
                _annotate_stage_history(stage_history, stage_name=stage.name)
            )
        else:
            _restore_parameter_values(model, snapshot)

    if not history:
        raise ValueError(
            "No staged fitting steps ran because no stage had " "trainable parameters."
        )

    return history


def fit_two_compartment_model(
    model,
    sweeps: Iterable[AllenSweepBatch],
    *,
    method: Literal["hybrid", "global", "tbptt", "staged"] = "hybrid",
    lr: float = 1e-3,
    epochs: int = 10,
    chunk_size: int = 500,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    voltage_weight: float = 1.0,
    spike_weight: float = 1.0,
    spike_count_weight: float = 0.0,
    spike_timing_weight: float = 0.0,
    spike_count_over_weight: float = 1.0,
    spike_count_under_weight: float = 1.0,
    sparsity_weight: float = 1e-4,
    spike_tau_ms: float = 10.0,
    post_spike_mask_ms: float = 3.0,
    spike_match_window_ms: float = 10.0,
    spike_miss_penalty_ms: float | None = None,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    global_maxiter: int = 20,
    global_popsize: int = 8,
    local_maxiter: int = 50,
    seed: int | None = 0,
    polish: bool = True,
    stages: Sequence[TwoCompartmentFitStage] | None = None,
) -> list[dict[str, float | str]]:
    """Fit the model with a robust strategy for poor initialization.

    Args:
        model: Two-compartment neuron instance to fit.
        sweeps: Allen or synthetic sweeps to fit against.
        method: ``"hybrid"`` first performs bounded global search, then runs a
            short TBPTT refinement. ``"global"`` stops after the search/polish
            stage. ``"tbptt"`` keeps the original truncated-BPTT workflow.
            ``"staged"`` fits parameter groups sequentially with stage-specific
            sweep subsets and spike-count-first weighting.
        lr: Learning rate for TBPTT refinement.
        epochs: Number of TBPTT epochs for ``method="tbptt"`` or the hybrid
            refinement stage.
        chunk_size: Truncated-BPTT chunk length.
        device: Torch device for evaluation and refinement.
        dtype: Torch dtype used during fitting.
        voltage_weight: Weight of the masked voltage reconstruction loss.
        spike_weight: Weight of the smoothed spike-train loss.
        spike_count_weight: Weight of the spike-count matching penalty.
        spike_timing_weight: Weight of the hard spike-timing event loss.
        sparsity_weight: Weight of the ``w_Ca`` sparsity penalty.
        spike_tau_ms: Smoothing constant for the spike-train loss.
        post_spike_mask_ms: Duration of the post-spike voltage mask.
        spike_match_window_ms: Tolerance window for matching predicted and
            true spikes.
        spike_miss_penalty_ms: Per-event penalty assigned to unmatched spikes
            in the hard spike-timing loss. Defaults to the match window.
        param_bounds: Optional per-parameter bounds override for the global
            search. Unspecified parameters use
            ``DEFAULT_TWO_COMPARTMENT_PARAM_BOUNDS``.
        global_maxiter: Differential-evolution outer iterations.
        global_popsize: Differential-evolution population size multiplier.
        local_maxiter: Maximum L-BFGS-B iterations after the global search.
        seed: Random seed for the global search.
        polish: If ``True``, run L-BFGS-B after the global search.
        stages: Optional staged-fit configuration. When omitted,
            ``DEFAULT_TWO_COMPARTMENT_FIT_STAGES`` is used.

    Returns:
        A history list with one row per fitting stage or TBPTT chunk.

    Notes:
        The default ``"hybrid"`` method is intended for real fitting runs where
        the starting parameters may be far from the biological regime. Global
        search handles the large basin-finding problem more robustly than pure
        BPTT, while the optional TBPTT stage can still fine-tune the result.
    """
    sweep_list = list(sweeps)
    if method == "tbptt":
        return _fit_two_compartment_model_tbptt(
            model,
            sweep_list,
            lr=lr,
            epochs=epochs,
            chunk_size=chunk_size,
            device=device,
            dtype=dtype,
            voltage_weight=voltage_weight,
            spike_weight=spike_weight,
            spike_count_weight=spike_count_weight,
            spike_timing_weight=spike_timing_weight,
            spike_count_over_weight=spike_count_over_weight,
            spike_count_under_weight=spike_count_under_weight,
            sparsity_weight=sparsity_weight,
            spike_tau_ms=spike_tau_ms,
            post_spike_mask_ms=post_spike_mask_ms,
            spike_match_window_ms=spike_match_window_ms,
            spike_miss_penalty_ms=spike_miss_penalty_ms,
        )

    if method == "staged":
        return _fit_two_compartment_model_staged(
            model,
            sweep_list,
            device=device,
            dtype=dtype,
            voltage_weight=voltage_weight,
            spike_weight=spike_weight,
            spike_count_weight=spike_count_weight,
            spike_timing_weight=spike_timing_weight,
            spike_count_over_weight=spike_count_over_weight,
            spike_count_under_weight=spike_count_under_weight,
            sparsity_weight=sparsity_weight,
            spike_tau_ms=spike_tau_ms,
            post_spike_mask_ms=post_spike_mask_ms,
            spike_match_window_ms=spike_match_window_ms,
            spike_miss_penalty_ms=spike_miss_penalty_ms,
            param_bounds=param_bounds,
            global_maxiter=global_maxiter,
            global_popsize=global_popsize,
            local_maxiter=local_maxiter,
            seed=seed,
            polish=polish,
            stages=stages,
        )

    history = _fit_two_compartment_model_global(
        model,
        sweep_list,
        device=device,
        dtype=dtype,
        voltage_weight=voltage_weight,
        spike_weight=spike_weight,
        spike_count_weight=spike_count_weight,
        spike_timing_weight=spike_timing_weight,
        spike_count_over_weight=spike_count_over_weight,
        spike_count_under_weight=spike_count_under_weight,
        sparsity_weight=sparsity_weight,
        spike_tau_ms=spike_tau_ms,
        post_spike_mask_ms=post_spike_mask_ms,
        spike_match_window_ms=spike_match_window_ms,
        spike_miss_penalty_ms=spike_miss_penalty_ms,
        param_bounds=param_bounds,
        global_maxiter=global_maxiter,
        global_popsize=global_popsize,
        local_maxiter=local_maxiter,
        seed=seed,
        polish=polish,
    )
    if method == "global":
        return history

    history.extend(
        _fit_two_compartment_model_tbptt(
            model,
            sweep_list,
            lr=lr,
            epochs=epochs,
            chunk_size=chunk_size,
            device=device,
            dtype=dtype,
            voltage_weight=voltage_weight,
            spike_weight=spike_weight,
            spike_count_weight=spike_count_weight,
            spike_timing_weight=spike_timing_weight,
            spike_count_over_weight=spike_count_over_weight,
            spike_count_under_weight=spike_count_under_weight,
            sparsity_weight=sparsity_weight,
            spike_tau_ms=spike_tau_ms,
            post_spike_mask_ms=post_spike_mask_ms,
            spike_match_window_ms=spike_match_window_ms,
            spike_miss_penalty_ms=spike_miss_penalty_ms,
        )
    )
    return history


__all__ = [
    "AllenSweepBatch",
    "DEFAULT_TWO_COMPARTMENT_PARAM_BOUNDS",
    "DEFAULT_TWO_COMPARTMENT_FIT_STAGES",
    "FitEvaluation",
    "TwoCompartmentFitStage",
    "choose_current_clamp_sweeps",
    "detect_spikes_from_voltage",
    "evaluate_fit_across_sweeps",
    "evaluate_two_compartment_fit",
    "exponential_filter_spike_train",
    "filter_mouse_visp_l5_pyramidal_cells",
    "fit_two_compartment_model",
    "get_cell_types_cache",
    "load_allen_sweep",
    "mask_post_spike_voltage_samples",
    "plot_two_compartment_fit",
    "query_mouse_visp_l5_pyramidal_cells",
    "resample_trace",
    "rollout_two_compartment",
    "save_fit_report",
    "spike_timing_loss",
    "spike_timing_stats",
    "two_compartment_loss",
]
