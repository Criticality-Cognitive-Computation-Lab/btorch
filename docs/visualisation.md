# Visualisation Module

The `btorch.visualisation` module provides plotting functions for neural simulation analysis.

## Modules

### `timeseries.py`
Time-series visualization for spike and continuous data.

| Function | Description |
|----------|-------------|
| `plot_raster` | Spike raster with grouping, styling, events, regions, and tracks |
| `plot_traces` | Continuous traces (voltage, currents) |
| `plot_spectrum` | Frequency spectrum (Welch method) |
| `plot_grouped_spectrum` | Spectral analysis by neuron groups |
| `plot_log_hist` | Log-log histogram |
| `plot_neuron_traces` | Multi-panel neuron state plots (voltage, ASC, PSC) |

**Dataclasses:**
- `NeuronSpec`: Per-neuron styling (color, marker, linestyle)
- `SimulationStates`: Container for simulation data
- `TracePlotFormat`: Figure formatting options

---

### `dynamics.py`
Multiscale dynamics analysis visualization.

| Function | Description |
|----------|-------------|
| `plot_multiscale_fano` | Fano factor across time windows |
| `plot_dfa_analysis` | Detrended Fluctuation Analysis |
| `plot_isi_cv` | ISI Coefficient of Variation |
| `plot_avalanche_analysis` | Avalanche size/duration distributions |
| `plot_eigenvalue_spectrum` | Weight matrix eigenvalue spectrum |
| `plot_lyapunov_spectrum` | Lyapunov exponents spectrum |
| `plot_firing_rate_distribution` | Firing rate histogram |

**Dataclasses:**
- `DynamicsData`: Container for spike data
- `DynamicsPlotFormat`: Visualization mode (individual/grouped/distribution)
- `FanoFactorConfig`: Fano analysis parameters
- `DFAConfig`: DFA parameters

---

### `hexmap.py`
Hexagonal heatmap visualization using Plotly.

| Function | Description |
|----------|-------------|
| `hex_heatmap` | Interactive hex-grid heatmap with slider for time series |

---

### `neuropil.py`
Neuropil-level aggregated activity visualization.

| Function | Description |
|----------|-------------|
| `plot_agg_by_neuropil` | Aggregated traces by neuropil (wave or heatmap style) |
| `plot_neuropil_comparison` | Subplot grid comparing selected neuropils |

---

## Usage Examples

```python
from btorch.visualisation.timeseries import plot_raster, plot_neuron_traces, NeuronSpec

# Basic raster
plot_raster(spikes, dt=0.1, marker="|", markersize=5)

# Grouped raster with colors
plot_raster(
    spikes,
    neurons_df=df,
    group_by="cell_type",
    color={"excitatory": "red", "inhibitory": "blue"},
    show_separators=True,
    events=[100, 200],  # Event markers
    regions=[(50, 80)],  # Shaded regions
    show_tracks=True,
)

# Neuron traces with per-neuron styling
specs = [NeuronSpec(color="red"), NeuronSpec(color="blue")]
plot_neuron_traces(voltage=V, dt=0.1, neuron_specs=specs)
```
