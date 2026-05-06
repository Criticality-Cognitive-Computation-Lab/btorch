"""Fit the two-compartment neuron to Allen Cell Types sweeps.

This example now uses a practical medium-sized dataset for one specimen:

1. query mouse VISp layer-5 pyramidal candidates,
2. pick representative long-square current-clamp sweeps,
3. split those sweeps into train and test sets,
4. fit :class:`btorch.models.neurons.TwoCompartmentGLIF` on the train set,
5. report train/test metrics and plots.

AllenSDK is required to run this script.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from btorch.analysis.two_compartment_fit import (
    DEFAULT_TWO_COMPARTMENT_PARAM_BOUNDS,
    choose_current_clamp_sweeps,
    evaluate_fit_across_sweeps,
    fit_two_compartment_model,
    load_allen_sweep,
    query_mouse_visp_l5_pyramidal_cells,
    save_fit_report,
)
from btorch.models.neurons import TwoCompartmentGLIF


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-file", type=str, default=None)
    parser.add_argument("--max-cells", type=int, default=1)
    parser.add_argument("--candidate-cells", type=int, default=8)
    parser.add_argument("--max-sweeps-per-cell", type=int, default=6)
    parser.add_argument("--test-fraction", type=float, default=0.33)
    parser.add_argument("--dt-ms", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--tbptt-refine-epochs", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tbptt-refine-lr", type=float, default=2e-4)
    parser.add_argument("--voltage-weight", type=float, default=1.0)
    parser.add_argument("--spike-weight", type=float, default=5.0)
    parser.add_argument("--spike-count-weight", type=float, default=20.0)
    parser.add_argument("--spike-timing-weight", type=float, default=10.0)
    parser.add_argument("--spike-match-window-ms", type=float, default=10.0)
    parser.add_argument("--sparsity-weight", type=float, default=1e-4)
    parser.add_argument(
        "--method",
        type=str,
        choices=("hybrid", "global", "tbptt", "staged"),
        default="staged",
    )
    parser.add_argument("--global-maxiter", type=int, default=20)
    parser.add_argument("--global-popsize", type=int, default=8)
    parser.add_argument("--local-maxiter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/two_compartment_fit",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser


@dataclass(frozen=True)
class SweepSummary:
    """Selection-time summary for one Allen sweep."""

    spike_count: int
    amplitude_pa: float
    stimulus_start_ms: float
    stimulus_stop_ms: float
    first_spike_ms: float | None
    last_spike_ms: float | None
    spike_span_ms: float
    onset_delay_ms: float | None
    is_onset_only: bool
    is_sustained_lowrate: bool
    is_sustained_highrate: bool


def _sample_evenly(items: list, count: int) -> list:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    target_positions = torch.linspace(0, len(items) - 1, steps=count)
    used: set[int] = set()
    selected_indices: list[int] = []
    for position in target_positions:
        candidate = int(round(position.item()))
        if candidate not in used:
            used.add(candidate)
            selected_indices.append(candidate)
            continue
        for radius in range(1, len(items)):
            left = candidate - radius
            right = candidate + radius
            if left >= 0 and left not in used:
                used.add(left)
                selected_indices.append(left)
                break
            if right < len(items) and right not in used:
                used.add(right)
                selected_indices.append(right)
                break
    selected_indices.sort()
    return [items[index] for index in selected_indices[:count]]


def _summarize_sweep(sweep) -> SweepSummary:
    """Extract sustained-spiking features used for sweep selection."""
    current = sweep.i_soma[:, 0, 0]
    current_abs = current.abs()
    amplitude_pa = float(current_abs.max().item())
    active = current_abs >= max(5.0, 0.1 * amplitude_pa)
    active_idx = torch.nonzero(active).flatten()
    if active_idx.numel() > 0:
        start_idx = int(active_idx[0].item())
        stop_idx = int(active_idx[-1].item())
    else:
        start_idx = 0
        stop_idx = int(current.shape[0] - 1)

    spikes = torch.nonzero(sweep.spike_true[:, 0, 0] > 0.5).flatten()
    spike_count = int(spikes.numel())
    if spike_count > 0:
        first_spike_ms = float(spikes[0].item() * sweep.dt_ms)
        last_spike_ms = float(spikes[-1].item() * sweep.dt_ms)
        spike_span_ms = last_spike_ms - first_spike_ms
        onset_delay_ms = first_spike_ms - float(start_idx * sweep.dt_ms)
    else:
        first_spike_ms = None
        last_spike_ms = None
        spike_span_ms = 0.0
        onset_delay_ms = None

    is_onset_only = (
        spike_count == 1
        or (spike_count == 2 and spike_span_ms < 20.0)
        or (
            spike_count > 0
            and onset_delay_ms is not None
            and onset_delay_ms < 25.0
            and spike_span_ms < 25.0
        )
    )
    is_sustained_lowrate = spike_count >= 2 and spike_span_ms >= 25.0
    is_sustained_highrate = spike_count >= 6 or spike_span_ms >= 80.0
    return SweepSummary(
        spike_count=spike_count,
        amplitude_pa=amplitude_pa,
        stimulus_start_ms=float(start_idx * sweep.dt_ms),
        stimulus_stop_ms=float(stop_idx * sweep.dt_ms),
        first_spike_ms=first_spike_ms,
        last_spike_ms=last_spike_ms,
        spike_span_ms=spike_span_ms,
        onset_delay_ms=onset_delay_ms,
        is_onset_only=is_onset_only,
        is_sustained_lowrate=is_sustained_lowrate,
        is_sustained_highrate=is_sustained_highrate,
    )


def _cell_richness_score(loaded: list) -> tuple[float, float, float]:
    """Rank cells by how informative their long-square sweeps are."""
    summaries = [_summarize_sweep(sweep) for sweep in loaded]
    sustained_count = sum(
        summary.is_sustained_lowrate or summary.is_sustained_highrate
        for summary in summaries
    )
    highrate_count = sum(summary.is_sustained_highrate for summary in summaries)
    max_spike_count = max((summary.spike_count for summary in summaries), default=0)
    return (float(sustained_count), float(highrate_count), float(max_spike_count))


def _select_representative_sweeps(loaded: list, total_count: int) -> list:
    summaries = {id(sweep): _summarize_sweep(sweep) for sweep in loaded}
    subthreshold = [sweep for sweep in loaded if summaries[id(sweep)].spike_count == 0]
    sustained_lowrate = [
        sweep for sweep in loaded if summaries[id(sweep)].is_sustained_lowrate
    ]
    sustained_highrate = [
        sweep
        for sweep in loaded
        if summaries[id(sweep)].is_sustained_highrate and sweep not in sustained_lowrate
    ]
    fallback_spiking = [
        sweep
        for sweep in loaded
        if summaries[id(sweep)].spike_count > 0
        and not summaries[id(sweep)].is_onset_only
        and sweep not in sustained_lowrate
        and sweep not in sustained_highrate
    ]

    sustained_lowrate.sort(
        key=lambda sweep: (
            summaries[id(sweep)].spike_count,
            summaries[id(sweep)].spike_span_ms,
            summaries[id(sweep)].amplitude_pa,
        )
    )
    sustained_highrate.sort(
        key=lambda sweep: (
            summaries[id(sweep)].spike_count,
            summaries[id(sweep)].spike_span_ms,
            summaries[id(sweep)].amplitude_pa,
        ),
        reverse=True,
    )
    fallback_spiking.sort(
        key=lambda sweep: (
            summaries[id(sweep)].spike_count,
            summaries[id(sweep)].spike_span_ms,
            summaries[id(sweep)].amplitude_pa,
        ),
        reverse=True,
    )
    subthreshold.sort(
        key=lambda sweep: summaries[id(sweep)].amplitude_pa,
        reverse=True,
    )

    target_subthreshold = min(len(subthreshold), max(1, total_count // 4))
    target_lowrate = min(
        len(sustained_lowrate),
        max(1 if sustained_lowrate else 0, total_count // 3),
    )
    target_highrate = min(
        len(sustained_highrate),
        max(1 if sustained_highrate else 0, total_count // 3),
    )
    selected: list = []
    selected_ids: set[int] = set()

    def extend_unique(candidates: list) -> None:
        for sweep in candidates:
            sweep_id = id(sweep)
            if sweep_id not in selected_ids:
                selected.append(sweep)
                selected_ids.add(sweep_id)

    extend_unique(_sample_evenly(subthreshold, target_subthreshold))
    extend_unique(_sample_evenly(sustained_lowrate, target_lowrate))
    extend_unique(_sample_evenly(sustained_highrate, target_highrate))

    remaining = total_count - len(selected)
    if remaining > 0:
        extend_unique(_sample_evenly(fallback_spiking, remaining))

    if len(selected) < total_count:
        extras = [
            sweep
            for sweep in loaded
            if id(sweep) not in selected_ids and not summaries[id(sweep)].is_onset_only
        ]
        extras.sort(
            key=lambda sweep: (
                summaries[id(sweep)].spike_count,
                summaries[id(sweep)].spike_span_ms,
                summaries[id(sweep)].amplitude_pa,
            ),
            reverse=True,
        )
        extend_unique(extras[: total_count - len(selected)])

    selected.sort(
        key=lambda sweep: (
            summaries[id(sweep)].is_sustained_highrate,
            summaries[id(sweep)].is_sustained_lowrate,
            summaries[id(sweep)].spike_count,
            summaries[id(sweep)].spike_span_ms,
            summaries[id(sweep)].amplitude_pa,
        ),
        reverse=True,
    )
    return selected[:total_count]


def _stratified_split_sweeps(sweeps: list, test_fraction: float) -> tuple[list, list]:
    if not sweeps:
        return [], []
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}.")

    summaries = {id(sweep): _summarize_sweep(sweep) for sweep in sweeps}
    highrate = [sweep for sweep in sweeps if summaries[id(sweep)].is_sustained_highrate]
    lowrate = [
        sweep
        for sweep in sweeps
        if summaries[id(sweep)].is_sustained_lowrate
        and not summaries[id(sweep)].is_sustained_highrate
    ]
    subthreshold = [sweep for sweep in sweeps if summaries[id(sweep)].spike_count == 0]
    residual_spiking = [
        sweep
        for sweep in sweeps
        if summaries[id(sweep)].spike_count > 0
        and sweep not in highrate
        and sweep not in lowrate
    ]

    def split_group(group: list) -> tuple[list, list]:
        if len(group) <= 1:
            return group, []
        test_count = max(1, int(math.ceil(len(group) * test_fraction)))
        test = _sample_evenly(group, test_count)
        train = [sweep for sweep in group if sweep not in test]
        return train, test

    train_highrate, test_highrate = split_group(highrate)
    train_lowrate, test_lowrate = split_group(lowrate)
    train_subthreshold, test_subthreshold = split_group(subthreshold)
    train_residual, test_residual = split_group(residual_spiking)
    train = train_highrate + train_lowrate + train_subthreshold + train_residual
    test = test_highrate + test_lowrate + test_subthreshold + test_residual

    if not train and test:
        train = test[:-1]
        test = test[-1:]
    return train, test


def _ensure_spiking_coverage(
    train: list,
    test: list,
    selected: list,
) -> tuple[list, list]:
    """Ensure both splits contain spiking sweeps when available."""
    spiking_selected = [
        sweep for sweep in selected if float(sweep.spike_true.sum().item()) > 0.0
    ]
    if len(spiking_selected) < 2:
        return train, test

    train_spiking = [
        sweep for sweep in train if float(sweep.spike_true.sum().item()) > 0.0
    ]
    test_spiking = [
        sweep for sweep in test if float(sweep.spike_true.sum().item()) > 0.0
    ]

    if not train_spiking and test_spiking:
        candidate = max(
            test_spiking,
            key=lambda sweep: float(sweep.spike_true.sum().item()),
        )
        test = [sweep for sweep in test if sweep is not candidate]
        train.append(candidate)
        train_spiking.append(candidate)

    if not test_spiking and train_spiking:
        candidate = min(
            train_spiking,
            key=lambda sweep: float(sweep.spike_true.sum().item()),
        )
        if len(train) > 1:
            train = [sweep for sweep in train if sweep is not candidate]
            test.append(candidate)

    return train, test


def _ensure_lowrate_coverage(
    train: list,
    test: list,
    selected: list,
) -> tuple[list, list]:
    """Ensure both splits contain low-rate sweeps when available."""
    lowrate_selected = [
        sweep for sweep in selected if 0.0 < float(sweep.spike_true.sum().item()) <= 5.0
    ]
    if len(lowrate_selected) < 2:
        return train, test

    train_lowrate = [
        sweep for sweep in train if 0.0 < float(sweep.spike_true.sum().item()) <= 5.0
    ]
    test_lowrate = [
        sweep for sweep in test if 0.0 < float(sweep.spike_true.sum().item()) <= 5.0
    ]

    if not train_lowrate and test_lowrate:
        candidate = min(
            test_lowrate,
            key=lambda sweep: float(sweep.spike_true.sum().item()),
        )
        test = [sweep for sweep in test if sweep is not candidate]
        train.append(candidate)
        train_lowrate.append(candidate)

    if not test_lowrate and train_lowrate and len(train) > 1:
        candidate = min(
            train_lowrate,
            key=lambda sweep: float(sweep.spike_true.sum().item()),
        )
        train = [sweep for sweep in train if sweep is not candidate]
        test.append(candidate)

    return train, test


def _ensure_sustained_coverage(
    train: list,
    test: list,
    selected: list,
) -> tuple[list, list]:
    """Ensure train and test both include sustained spiking sweeps when
    possible."""
    summaries = {id(sweep): _summarize_sweep(sweep) for sweep in selected}
    sustained_selected = [
        sweep
        for sweep in selected
        if summaries[id(sweep)].is_sustained_lowrate
        or summaries[id(sweep)].is_sustained_highrate
    ]
    if len(sustained_selected) < 2:
        return train, test

    train_sustained = [
        sweep
        for sweep in train
        if summaries.get(id(sweep), _summarize_sweep(sweep)).is_sustained_lowrate
        or summaries.get(id(sweep), _summarize_sweep(sweep)).is_sustained_highrate
    ]
    test_sustained = [
        sweep
        for sweep in test
        if summaries.get(id(sweep), _summarize_sweep(sweep)).is_sustained_lowrate
        or summaries.get(id(sweep), _summarize_sweep(sweep)).is_sustained_highrate
    ]

    if not train_sustained and test_sustained:
        candidate = max(
            test_sustained,
            key=lambda sweep: (
                summaries.get(id(sweep), _summarize_sweep(sweep)).spike_count,
                summaries.get(id(sweep), _summarize_sweep(sweep)).spike_span_ms,
            ),
        )
        test = [sweep for sweep in test if sweep is not candidate]
        train.append(candidate)

    if not test_sustained and train_sustained and len(train) > 1:
        candidate = min(
            train_sustained,
            key=lambda sweep: (
                summaries.get(id(sweep), _summarize_sweep(sweep)).spike_count,
                summaries.get(id(sweep), _summarize_sweep(sweep)).spike_span_ms,
            ),
        )
        train = [sweep for sweep in train if sweep is not candidate]
        test.append(candidate)

    return train, test


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cells = query_mouse_visp_l5_pyramidal_cells(manifest_file=args.manifest_file)
    if not cells:
        raise RuntimeError("No mouse VISp layer-5 pyramidal candidates were found.")

    candidate_payloads = []
    search_cells = cells[: max(args.max_cells, args.candidate_cells)]
    from btorch.analysis.two_compartment_fit import get_cell_types_cache

    cache = get_cell_types_cache(args.manifest_file)
    for cell in search_cells:
        specimen_id = int(cell.get("specimen_id") or cell.get("id"))
        sweep_records = cache.get_ephys_sweeps(specimen_id)
        chosen = choose_current_clamp_sweeps(sweep_records)
        loaded = [
            load_allen_sweep(
                specimen_id=specimen_id,
                sweep_number=int(sweep["sweep_number"]),
                dt_ms=args.dt_ms,
                manifest_file=args.manifest_file,
                cache=cache,
            )
            for sweep in chosen
        ]
        if not loaded:
            continue
        candidate_payloads.append((specimen_id, loaded, _cell_richness_score(loaded)))

    candidate_payloads.sort(key=lambda item: item[2], reverse=True)
    selected_payloads = candidate_payloads[: args.max_cells]

    train_sweeps = []
    test_sweeps = []
    for specimen_id, loaded, richness in selected_payloads:
        selected = _select_representative_sweeps(loaded, args.max_sweeps_per_cell)
        train_split, test_split = _stratified_split_sweeps(
            selected,
            args.test_fraction,
        )
        train_split, test_split = _ensure_spiking_coverage(
            train_split,
            test_split,
            selected,
        )
        train_split, test_split = _ensure_lowrate_coverage(
            train_split,
            test_split,
            selected,
        )
        train_split, test_split = _ensure_sustained_coverage(
            train_split,
            test_split,
            selected,
        )
        train_sweeps.extend(train_split)
        test_sweeps.extend(test_split)
        print(
            "Selected specimen",
            specimen_id,
            "richness=",
            richness,
            "train=",
            [sweep.sweep_number for sweep in train_split],
            "test=",
            [sweep.sweep_number for sweep in test_split],
        )

    if not train_sweeps:
        raise RuntimeError("No suitable current-clamp sweeps were found.")

    model = TwoCompartmentGLIF(
        n_neuron=1,
        tau_s=25.0,
        R_s=0.08,
        E_L=-72.0,
        tau_a=80.0,
        tau_th=50.0,
        delta_th=2.0,
        delta_T=2.0,
        w_Ca=0.0,
        theta_Ca=3.0,
        w_sa=0.0,
        w_as=0.0,
        v_threshold=-46.0,
        v_reset=-70.0,
        trainable_param={
            "tau_s",
            "R_s",
            "E_L",
            "tau_a",
            "tau_th",
            "delta_th",
            "delta_T",
            "w_Ca",
            "theta_Ca",
            "w_sa",
            "w_as",
            "v_threshold",
            "v_reset",
        },
    ).to(device)

    history = fit_two_compartment_model(
        model,
        train_sweeps,
        method=args.method,
        lr=args.lr,
        epochs=args.epochs,
        chunk_size=args.chunk_size,
        voltage_weight=args.voltage_weight,
        spike_weight=args.spike_weight,
        spike_count_weight=args.spike_count_weight,
        spike_timing_weight=args.spike_timing_weight,
        spike_match_window_ms=args.spike_match_window_ms,
        sparsity_weight=args.sparsity_weight,
        global_maxiter=args.global_maxiter,
        global_popsize=args.global_popsize,
        local_maxiter=args.local_maxiter,
        param_bounds=DEFAULT_TWO_COMPARTMENT_PARAM_BOUNDS,
        seed=args.seed,
        device=device,
    )
    if args.tbptt_refine_epochs > 0:
        tbptt_history = fit_two_compartment_model(
            model,
            train_sweeps,
            method="tbptt",
            lr=args.tbptt_refine_lr,
            epochs=args.tbptt_refine_epochs,
            chunk_size=args.chunk_size,
            voltage_weight=max(args.voltage_weight, 2.0),
            spike_weight=args.spike_weight,
            spike_count_weight=max(args.spike_count_weight, 25.0),
            spike_timing_weight=max(args.spike_timing_weight, 12.0),
            spike_match_window_ms=args.spike_match_window_ms,
            sparsity_weight=args.sparsity_weight,
            device=device,
        )
        history.extend(tbptt_history)
    train_evaluations, train_aggregate = evaluate_fit_across_sweeps(
        model,
        train_sweeps,
        device=device,
        spike_count_weight=args.spike_count_weight,
        spike_timing_weight=args.spike_timing_weight,
        spike_match_window_ms=args.spike_match_window_ms,
    )
    train_paths = save_fit_report(
        model,
        train_evaluations,
        train_aggregate,
        history,
        output_dir=Path(args.output_dir) / "train",
    )

    test_aggregate = {}
    test_paths = {}
    if test_sweeps:
        test_evaluations, test_aggregate = evaluate_fit_across_sweeps(
            model,
            test_sweeps,
            device=device,
            spike_count_weight=args.spike_count_weight,
            spike_timing_weight=args.spike_timing_weight,
            spike_match_window_ms=args.spike_match_window_ms,
        )
        test_paths = save_fit_report(
            model,
            test_evaluations,
            test_aggregate,
            [],
            output_dir=Path(args.output_dir) / "test",
        )

    final = history[-1]
    print("Finished fitting")
    print(f"  train_sweeps:   {len(train_sweeps)}")
    print(f"  test_sweeps:    {len(test_sweeps)}")
    print(f"  total_loss:    {final['total_loss']:.6f}")
    print(f"  voltage_loss:  {final['voltage_loss']:.6f}")
    print(f"  spike_loss:    {final['spike_loss']:.6f}")
    print(f"  spike_timing:  {final['spike_timing_loss']:.6f}")
    print(f"  sparsity_loss: {final['sparsity_loss']:.6f}")
    print("Train summary")
    for name, value in train_aggregate.items():
        print(f"  {name}: {value:.6f}")
    if test_aggregate:
        print("Test summary")
        for name, value in test_aggregate.items():
            print(f"  {name}: {value:.6f}")
    print("Train artifacts")
    for name, path in train_paths.items():
        print(f"  {name}: {path}")
    if test_paths:
        print("Test artifacts")
        for name, path in test_paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
