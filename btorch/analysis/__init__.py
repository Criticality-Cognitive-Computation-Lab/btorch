from .aggregation import (
    agg_by_neuron,
    agg_by_neuropil,
    agg_conn,
    build_group_frame,
    group_ecdf,
    group_summary,
    group_values,
)
from .branching import compute_branching_ratio
from .connectivity import HopDistanceModel, compute_ie_ratio
from .metrics import indices_to_mask, select_on_metric
from .spiking import (
    compute_raster,
    compute_spectrum,
    cv_temporal,
    fano,
    fano_population,
    fano_sweep,
    fano_temporal,
    firing_rate,
    isi_cv,
    isi_cv_population,
    kurtosis,
    kurtosis_population,
    local_variation,
)
from .statistics import (
    StatChoice,
    compute_log_hist,
    describe_array,
    use_percentiles,
    use_stats,
)
from .voltage import compute_voltage_overshoot, suggest_skip_timestep


__all__ = [
    "agg_by_neuropil",
    "agg_by_neuron",
    "agg_conn",
    "build_group_frame",
    "group_values",
    "group_summary",
    "group_ecdf",
    "compute_branching_ratio",
    "HopDistanceModel",
    "compute_ie_ratio",
    "indices_to_mask",
    "select_on_metric",
    # New simplified API
    "isi_cv",
    "fano",
    "kurtosis",
    "local_variation",
    # Population metrics
    "isi_cv_population",
    "fano_population",
    "kurtosis_population",
    # Temporal variants
    "cv_temporal",
    "fano_temporal",
    # Sweep functions
    "fano_sweep",
    # Utilities
    "firing_rate",
    "compute_raster",
    "compute_log_hist",
    "compute_spectrum",
    "describe_array",
    "suggest_skip_timestep",
    "compute_voltage_overshoot",
    "StatChoice",
    "use_stats",
    "use_percentiles",
]
