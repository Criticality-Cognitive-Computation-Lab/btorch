"""Fitting strategies for the two-compartment GLIF neuron."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import torch
from scipy import optimize
from torch import Tensor

from btorch.models import environ, functional

from .data import AllenSweepBatch, SweepKind
from .evaluation import (
    _prepare_sweep,
    evaluate_fit_across_sweeps,
    rollout_two_compartment,
)
from .loss import FitLossConfig, _loss_floats, two_compartment_loss


@dataclass(frozen=True)
class TwoCompartmentFitStage:
    """Configuration for one stage of staged parameter fitting."""

    name: str
    trainable_params: frozenset[str]
    sweep_kind: SweepKind
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


def _fit_sweeps_once(
    model,
    sweeps: Iterable[AllenSweepBatch],
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    loss: FitLossConfig,
) -> dict[str, float]:
    """Evaluate the current model parameters on one or more sweeps.

    Returns the per-sweep loss terms (see :func:`_loss_floats`) averaged over
    sweeps.
    """
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for sweep in sweeps:
            i_soma, v_true, spike_true, i_apical = _prepare_sweep(
                model, sweep, device=device, dtype=dtype
            )

            with environ.context(dt=float(sweep.dt)):
                rollout = rollout_two_compartment(model, i_soma, i_apical)
                losses = two_compartment_loss(
                    v_pred=rollout["v"],
                    spike_pred=rollout["spike"],
                    v_true=v_true,
                    spike_true=spike_true,
                    dt=sweep.dt,
                    w_Ca=getattr(model, "w_Ca", None),
                    loss=loss,
                )
            rows.append(_loss_floats(losses))

    if not rows:
        raise ValueError("At least one sweep is required for fitting.")

    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


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
    sweep_kind: SweepKind,
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
    loss: FitLossConfig,
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
            i_soma, v_true, spike_true, i_apical = _prepare_sweep(
                model,
                sweep,
                device=device,
                dtype=dtype,
                init_state=not initialized,
            )
            initialized = True
            optimizer.zero_grad()

            with environ.context(dt=float(sweep.dt)):
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
                        dt=sweep.dt,
                        w_Ca=getattr(model, "w_Ca", None),
                        loss=loss,
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
                            **_loss_floats(losses),
                        }
                    )
    return history


def _fit_two_compartment_model_global(
    model,
    sweeps: Iterable[AllenSweepBatch],
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    loss: FitLossConfig,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    global_maxiter: int = 20,
    global_popsize: int = 8,
    local_maxiter: int = 50,
    seed: int | None = 0,
    polish: bool = True,
) -> list[dict[str, float | str]]:
    """Fit with bounded global search and optional local L-BFGS-B polish."""
    # Resolve through the package namespace at call time so that
    # ``btorch.analysis.two_compartment_fit._fit_sweeps_once`` can be
    # monkeypatched (the characterization tests spy on it).
    from . import _fit_sweeps_once  # noqa: PLC0415

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
            loss=loss,
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
        loss=loss,
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
        loss=loss,
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
    loss: FitLossConfig,
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
        loss=loss,
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
                loss=replace(
                    loss,
                    voltage_weight=stage.voltage_weight,
                    spike_weight=stage.spike_weight,
                    spike_count_weight=stage.spike_count_weight,
                    spike_timing_weight=stage.spike_timing_weight,
                    spike_count_over_weight=stage.spike_count_over_weight,
                    spike_count_under_weight=stage.spike_count_under_weight,
                    sparsity_weight=stage.sparsity_weight,
                ),
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
            loss=loss,
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
            "No staged fitting steps ran because no stage had trainable parameters."
        )

    return history


def fit_two_compartment_model(
    model: torch.nn.Module,
    sweeps: Iterable[AllenSweepBatch],
    *,
    method: Literal["hybrid", "global", "tbptt", "staged"] = "hybrid",
    lr: float = 1e-3,
    epochs: int = 10,
    chunk_size: int = 500,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    loss: FitLossConfig | None = None,
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
        loss: Loss weights and spike-matching settings; ``None`` uses the
            :class:`FitLossConfig` defaults. In ``method="staged"`` the
            per-stage weights override the weight fields of this config while
            its spike-matching settings are kept.
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
    loss = FitLossConfig() if loss is None else loss
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
            loss=loss,
        )

    if method == "staged":
        return _fit_two_compartment_model_staged(
            model,
            sweep_list,
            device=device,
            dtype=dtype,
            loss=loss,
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
        loss=loss,
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
            loss=loss,
        )
    )
    return history
