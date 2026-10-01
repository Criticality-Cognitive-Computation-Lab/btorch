"""Allen-data fitting utilities for the two-compartment GLIF neuron.

Layout:

- :mod:`.data`: Allen sweep loading, resampling and cell selection.
- :mod:`.loss`: :class:`FitLossConfig` and the voltage/spike loss terms.
- :mod:`.evaluation`: rollout and per-sweep evaluation.
- :mod:`.report`: fit report writer (JSON + figures).
- :mod:`.fit`: the staged / global / TBPTT fitting strategies.
"""

from ._plots import plot_two_compartment_fit
from .data import (
    AllenSweepBatch,
    SweepKind,  # noqa: F401
    choose_current_clamp_sweeps,
    detect_spikes_from_voltage,
    filter_mouse_visp_l5_pyramidal_cells,
    get_cell_types_cache,
    load_allen_sweep,
    query_mouse_visp_l5_pyramidal_cells,
    resample_trace,
)
from .evaluation import (
    FitEvaluation,
    _prepare_sweep,  # noqa: F401
    evaluate_fit_across_sweeps,
    evaluate_two_compartment_fit,
    rollout_two_compartment,
)
from .fit import (
    DEFAULT_TWO_COMPARTMENT_FIT_STAGES,
    DEFAULT_TWO_COMPARTMENT_PARAM_BOUNDS,
    TwoCompartmentFitStage,
    _fit_sweeps_once,  # noqa: F401
    _fit_two_compartment_model_global,  # noqa: F401
    fit_two_compartment_model,
)
from .loss import (
    FitLossConfig,
    _loss_floats,  # noqa: F401
    exponential_filter_spike_train,
    mask_post_spike_voltage_samples,
    spike_timing_loss,
    spike_timing_stats,
    two_compartment_loss,
)
from .report import save_fit_report


__all__ = [
    "AllenSweepBatch",
    "DEFAULT_TWO_COMPARTMENT_PARAM_BOUNDS",
    "DEFAULT_TWO_COMPARTMENT_FIT_STAGES",
    "FitEvaluation",
    "FitLossConfig",
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
