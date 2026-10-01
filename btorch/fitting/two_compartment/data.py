"""Allen sweep loading and resampling for the two-compartment fit."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from torch import Tensor


if TYPE_CHECKING:  # AllenSDK is an optional dependency
    from allensdk.core.cell_types_cache import CellTypesCache


@dataclass
class AllenSweepBatch:
    """Preprocessed Allen sweep tensors for fitting.

    Args:
        specimen_id: Allen specimen identifier.
        sweep_number: Sweep number within the NWB file.
        dt: Simulation timestep in milliseconds.
        i_soma: Somatic current tensor shaped ``(T, B, N)``.
        v_true: Recorded voltage tensor shaped ``(T, B, N)``.
        spike_true: Binary spike tensor shaped ``(T, B, N)``.
        i_apical: Optional apical input tensor shaped ``(T, B, N)``.
        metadata: Auxiliary Allen metadata retained for bookkeeping.
    """

    specimen_id: int
    sweep_number: int
    dt: float
    i_soma: Tensor
    v_true: Tensor
    spike_true: Tensor
    i_apical: Tensor | None = None
    metadata: dict[str, Any] | None = None


SweepKind = Literal["all", "silent", "lowrate", "spiking", "countcal"]


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
    cache: CellTypesCache | None = None,
) -> CellTypesCache:
    """Create or reuse an AllenSDK ``CellTypesCache`` instance.

    Args:
        manifest_file: Path of the cache manifest; ``None`` uses AllenSDK's
            default location. Ignored when ``cache`` is given.
        cache: An existing cache, returned unchanged.

    Returns:
        ``cache`` if provided, else a new ``CellTypesCache``.

    Raises:
        ImportError: If AllenSDK is not installed and no ``cache`` is given.
    """
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
    source_dt: float,
    target_dt: float,
) -> Tensor:
    """Resample a 1D trace to a new timestep using linear interpolation."""
    if target_dt <= 0.0:
        raise ValueError(f"target_dt must be positive, got {target_dt}.")
    trace_t = torch.as_tensor(trace, dtype=torch.float32)
    if trace_t.ndim != 1:
        raise ValueError(f"Expected a 1D trace, got shape {tuple(trace_t.shape)}.")
    if abs(source_dt - target_dt) < 1e-12:
        return trace_t

    duration_ms = max((trace_t.shape[0] - 1) * source_dt, 0.0)
    target_steps = max(int(round(duration_ms / target_dt)) + 1, 1)
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
    dt: float = 0.5,
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

    source_dt = 1000.0 / sampling_rate
    voltage = resample_trace(
        response[sl],
        source_dt=source_dt,
        target_dt=dt,
    )
    current = resample_trace(
        stimulus[sl],
        source_dt=source_dt,
        target_dt=dt,
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
        dt=dt,
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
