# Analysis Module

The `btorch.analysis` module provides computational tools for neural data analysis.

## Core Modules

### `spiking.py`
Spike train analysis utilities.

| Function | Description |
|----------|-------------|
| `cv_from_spikes` | Coefficient of variation of ISIs per neuron |
| `fano_factor_from_spikes` | Fano factor (variance/mean of spike counts) |
| `kurtosis_from_spikes` | Kurtosis of spike count distribution |
| `raster_plot` | Extract spike times/neuron indices for plotting |
| `firing_rate` | Convolve spikes to firing rates |

---

### `statistics.py`
General statistical utilities.

| Function | Description |
|----------|-------------|
| `describe_array` | Print descriptive statistics |
| `compute_log_hist` | Log-spaced histogram |
| `compute_spectrum` | Power spectrum via Welch method |

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
| `micro_scale.py` | ISI CV, burst detection |
| `complexity.py` | Entropy, complexity measures |
| `criticality.py` | Avalanche analysis, power-law fitting |
| `attractor_dynamics.py` | Phase space reconstruction |
| `lyapunov_dynamics.py` | Lyapunov exponent estimation |

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
