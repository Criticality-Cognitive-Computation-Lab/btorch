"""Fit report writing for the two-compartment fit."""

import json
from collections.abc import Sequence
from pathlib import Path

import torch

from ._plots import plot_two_compartment_fit
from .evaluation import FitEvaluation


def save_fit_report(
    model: torch.nn.Module,
    evaluations: Sequence[FitEvaluation],
    aggregate_metrics: dict[str, float],
    history: Sequence[dict[str, float | str]],
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save fitted parameters, metrics, history, and plots to disk.

    Writes ``fitted_parameters.json`` (every named parameter, flattened),
    ``fit_metrics.json`` (``aggregate`` plus ``per_sweep`` entries),
    ``fit_history.json`` and one PNG per evaluation.

    Args:
        model: Fitted model; its ``named_parameters`` are serialised.
        evaluations: Per-sweep evaluations to plot and record.
        aggregate_metrics: Metrics averaged over sweeps.
        history: Rows returned by ``fit_two_compartment_model``.
        output_dir: Directory (created if missing).

    Returns:
        Dict with ``output_dir``, ``parameters``, ``metrics``, ``history``
        (JSON paths), ``plot`` (the last evaluation's figure) and
        ``primary_plot`` (the first evaluation with recorded spikes, else the
        last figure). Both plot entries fall back to ``output_dir`` when
        ``evaluations`` is empty.
    """
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
                "dt_ms": ev.dt,
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
