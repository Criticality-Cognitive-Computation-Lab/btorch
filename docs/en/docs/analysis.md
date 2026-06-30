# Analysis Module

The `btorch.analysis` module provides computational tools for neural data analysis.

## Core Modules

### `spiking.py`

Spike train analysis utilities with dual NumPy/PyTorch backend support.

| Function | Description |
|----------|-------------|
| `cv_from_spikes` | Coefficient of variation of ISIs per neuron |
| `fano_factor_from_spikes` | Fano factor (variance/mean of spike counts) |
| `kurtosis_from_spikes` | Kurtosis of spike count distribution |
| `local_variation_from_spikes` | Local Variation (LV) - rate-independent irregularity |
| `raster_plot` | Extract spike times/neuron indices for plotting |
| `firing_rate` | Convolve spikes to firing rates |
| `compute_spectrum` | Power spectrum via Welch method |

**Common Parameters:**

- `batch_axis`: Tuple of axis indices to aggregate over (e.g., `(1, 2)` for trials)
- `percentile`: Compute percentiles over neurons - `float`, `tuple[float, ...]`, or `None`

**Examples:**

```python
from btorch.analysis.spiking import cv_from_spikes, fano_factor_from_spikes

# NumPy input with batch aggregation across trials
cv, isi_total, isi_stats = cv_from_spikes(
    spike_data,           # shape: [T, B, N] 
    dt_ms=1.0,
    batch_axis=(1,),      # aggregate across batch dimension
    percentile=(0.1, 0.5, 0.9)  # compute 10th, 50th, 90th percentiles
)
# cv shape: [N] - per-neuron CV values
# isi_stats['percentile']: {'levels': (0.1, 0.5, 0.9), 'values': [...]}

# Torch GPU input
import torch
cv_gpu, _, _ = cv_from_spikes(
    torch.from_numpy(spike_data).cuda(),
    dt_ms=1.0,
    batch_axis=(1,)
)
# Returns GPU tensor, uses hybrid CPU/GPU for efficiency

# Fano factor with sliding windows
fano, info = fano_factor_from_spikes(
    spikes,
    window=100,       # window size in time steps
    overlap=50,       # overlap between windows
    percentile=0.9    # compute 90th percentile across neurons
)

# Sweep mode - compute for all window sizes
fano_sweep = fano_factor_from_spikes(
    spikes,
    sweep_window=True  # returns [T, ...] with FF for each window size
)

# Local Variation (LV) - less sensitive to rate changes than CV
lv, lv_stats = local_variation_from_spikes(
    spikes,
    dt_ms=1.0,
    percentile=(0.25, 0.75)
)
```

---

### `statistics.py`

General statistical utilities.

| Function | Description |
|----------|-------------|
| `describe_array` | Print descriptive statistics |
| `compute_log_hist` | Log-spaced histogram |
| `get_corr_stats` | Cross-correlation statistics for spike trains |

---

### `connectivity.py`

Network connectivity analysis.

| Function | Description |
|----------|-------------|
| `compute_ie_ratio` | Inhibitory/excitatory input ratio |
| `HopDistanceModel` | BFS-based hop distance computation |

**HopDistanceModel methods:**

- `compute_distances(seeds)` → DataFrame with hop distances
- `hop_statistics(seeds)` → Reachability statistics by hop
- `reconstruct_path(src, tgt)` → Shortest path

---

### `branching.py`

MR estimation from Wilting & Priesemann (2018).

| Function | Description |
|----------|-------------|
| `simulate_branching` | Simulate branching process |
| `simulate_binomial_subsampling` | Subsample spike trains |
| `MR_estimation` | Estimate branching ratio from spike counts |

---

### `aggregation.py`

Group-wise data aggregation.

| Function | Description |
|----------|-------------|
| `agg_by_neuron` | Aggregate by neuron type |
| `agg_by_neuropil` | Aggregate by neuropil region |
| `agg_conn` | Aggregate connectivity weights |
| `build_group_frame` | Convert `[N]` or `[..., N]` into long-format grouped values |
| `group_values` | Return grouped value arrays in deterministic group order |
| `group_summary` | Compute per-group descriptive statistics |
| `group_ecdf` | Compute per-group ECDF points for analysis/plotting |

---

### `voltage.py`

Voltage trace analysis.

| Function | Description |
|----------|-------------|
| `suggest_skip_timestep` | Suggest burn-in period |
| `voltage_overshoot` | Quantify voltage stability |

---

### `metrics.py`

Selection and masking utilities.

| Function | Description |
|----------|-------------|
| `indices_to_mask` | Convert indices to boolean mask |
| `select_on_metric` | Select neurons by metric (topk, any) |

---

## `dynamic_tools/` Subpackage

Advanced dynamical systems analysis tools.

| Module | Description |
|--------|-------------|
| `micro_scale.py` | ISI CV, burst detection, firing rate distribution |
| `complexity.py` | PCIst, representation alignment, gain-stability |
| `criticality.py` | Avalanche analysis, power-law fitting, DFA |
| `attractor_dynamics.py` | Phase space reconstruction, Kaplan-Yorke dimension |
| `lyapunov_dynamics.py` | Lyapunov exponent estimation |
| `ei_balance.py` | E/I balance metrics (ECI, lag correlation) |

### E/I Balance Analysis (`ei_balance.py`)

```python
from btorch.analysis.dynamic_tools.ei_balance import (
    compute_eci,
    compute_lag_correlation,
    compute_ei_balance_full
)

# Compute E/I cancellation index
eci, info = compute_eci(
    I_e,                  # excitatory current [T, B, N]
    I_i,                  # inhibitory current [T, B, N]
    I_ext=None,           # external current (optional)
    batch_axis=(1,),      # aggregate over trials
    percentile=0.9        # compute percentile over neurons
)

# Compute lag correlation between excitatory and inhibitory currents
peak_corr, corr_info = compute_lag_correlation(
    I_e,
    -I_i,                 # negative for inhibitory
    dt=1.0,
    max_lag_ms=30.0,      # maximum lag in milliseconds
    use_fft=True          # FFT-based for efficiency
)

# Full E/I balance analysis
metrics, info = compute_ei_balance_full(
    I_e, I_i,
    I_ext=None,
    dt=1.0,
    max_lag_ms=30.0,
    batch_axis=(1,)
)
# metrics: eci_mean, eci_median, track_corr_peak_mean, delay_ms_mean, etc.
```

---

## Usage Examples

```python
from btorch.analysis.spiking import firing_rate, fano_factor_from_spikes
from btorch.analysis.branching import MR_estimation

# Compute firing rates
fr = firing_rate(spikes, width=10, dt=0.1)

# Fano factor across windows
fano = fano_factor_from_spikes(spikes, window=100)

# Branching ratio estimation
result = MR_estimation(spike_counts)
print(f"Branching ratio: {result['branching_ratio']:.3f}")
```

---

## Backend Support

All spiking analysis functions support both NumPy and PyTorch:

- **NumPy**: Standard CPU-based computation
- **PyTorch**: GPU acceleration where beneficial
  - ISI-based metrics (CV, LV): Hybrid approach (GPU aggregation → CPU extraction → GPU return)
  - Count-based metrics (Fano, Kurtosis): Full GPU via cumulative sums

**Float16 Support:**

- Functions accept float16 inputs
- Internal accumulation uses float32 for numerical accuracy
- Returns follow input device placement
