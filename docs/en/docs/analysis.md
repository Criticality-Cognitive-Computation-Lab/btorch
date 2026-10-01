# Analysis Module

The `btorch.analysis` module provides computational tools for neural data analysis.

## Core Modules

### `spiking.py`

Spike train analysis utilities with dual NumPy/PyTorch backend support.

| Function | Description |
|----------|-------------|
| `isi_cv` | Coefficient of variation of ISIs per neuron |
| `fano` | Fano factor (variance/mean of spike counts) |
| `kurtosis` | Kurtosis of spike count distribution |
| `local_variation` | Local Variation (LV) - rate-independent irregularity |
| `compute_raster` | Extract spike times/neuron indices for plotting |
| `firing_rate` | Convolve spikes to firing rates |
| `compute_spectrum` | Power spectrum via Welch method |

**Common Parameters:**

- `batch_axis`: Tuple of axis indices to aggregate over (e.g., `(1, 2)` for trials)
- `percentile`: Compute percentiles over neurons - `float`, `tuple[float, ...]`, or `None`

**Examples:**

```python
from btorch.analysis import fano, fano_sweep, isi_cv, local_variation

# NumPy input with batch aggregation across trials
cv, info = isi_cv(
    spikes,               # shape: [T, B, N]
    dt=1.0,
    batch_axis=(1,),      # aggregate across batch dimension
    percentiles=(10, 50, 90),  # percentile levels in [0, 100]
)
# cv shape: [N] - per-neuron CV values
# info["cv_levels"], info["cv_percentiles"]: requested levels and their values

# Torch GPU input: returns GPU tensor, uses hybrid CPU/GPU for efficiency
import torch
cv_gpu, _ = isi_cv(torch.from_numpy(spikes).cuda(), dt=1.0, batch_axis=(1,))

# Fano factor with overlapping counting windows, aggregated to one number
ff, info = fano(spikes, window=100, overlap=50, stat="mean")

# Sweep over counting-window sizes 1..50
ff_sweep, info = fano_sweep(spikes, window=50)

# Local Variation (LV) - less sensitive to rate changes than CV
lv, info = local_variation(spikes, dt=1.0, percentiles=(25, 75))
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

### `two_compartment_fit.py`

Fitting and evaluation helpers for `TwoCompartmentGLIF` against Allen Cell Types
sweeps. All loss settings are carried by a single `FitLossConfig` dataclass that
is passed as `loss=` to `two_compartment_loss`, `evaluate_two_compartment_fit`,
`evaluate_fit_across_sweeps` and `fit_two_compartment_model` (`None` selects the
defaults).

```python
from btorch.analysis import (
    FitLossConfig,
    evaluate_fit_across_sweeps,
    fit_two_compartment_model,
)

loss = FitLossConfig(spike_count_weight=5.0, spike_match_window_ms=5.0)
history = fit_two_compartment_model(model, sweeps, method="hybrid", loss=loss)
evaluations, aggregate = evaluate_fit_across_sweeps(model, sweeps, loss=loss)
```

## `dynamic_tools/` Subpackage

Advanced dynamical systems analysis tools.

| Module | Description |
|--------|-------------|
| `micro_scale.py` | Firing rate distribution, SPIKE distance (ISI CV lives in `analysis.spiking.isi_cv`) |
| `fano.py` | Rate-compensated Fano factors (operational time, mean matching, model based) |
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
from btorch.analysis.spiking import firing_rate, fano
from btorch.analysis.branching import MR_estimation

# Compute firing rates
fr = firing_rate(spikes, width=10, dt=0.1)

# Fano factor across windows
ff, info = fano(spikes, window=100)

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
