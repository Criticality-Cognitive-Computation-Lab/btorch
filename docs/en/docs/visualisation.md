# Visualisation Module

The `btorch.visualisation` module provides plotting functions for neural simulation analysis.

## Modules

### `timeseries/` (package)
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
- `RasterStyle`, `RasterGrouping`, `GroupStripOptions`, `RatePanelOptions`, `RasterAnnotations`: Option groups for `plot_raster`
- `SpectrumGrouping`, `SpectrumStyle`: Option groups for `plot_grouped_spectrum`, e.g. `plot_grouped_spectrum(data, dt, SpectrumGrouping(neurons_df=df, group_by="type"), style=SpectrumStyle(show_traces=False))`

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

### `hex/static.py`
Static hex plots using matplotlib.

| Function | Description |
|----------|-------------|
| `scatter` | Hex scatter plot with color mapping |
| `quiver` | Quiver (vector field) plot on hex grid |
| `grid` | Hex grid with optional coordinate annotations |
| `draw_axes` | Overlay q/r/s axis arrows on hex plot |
| `compass` | Compass rose inset |
| `looming_stimulus` | Expanding stimulus sequence from center |

**Dataclasses** (`hex/options.py`): `HexGeometry` (layout, size, orientation, rotation), `HexColorMap` (cmap, vmin, vmax), `HexPatchStyle` (edge, alpha), `HexReference` (q/r/s axes, compass). They are passed to `scatter`/`quiver` as keyword-only `geometry=`, `color=`, `patch=` (`scatter` only) and `reference=`, e.g. `scatter(q, r, v, color=HexColorMap(cmap="RdBu_r", vmin=-1, vmax=1))`.

---

### `hex/interactive.py`
Interactive hex heatmaps using Plotly. Hex polygons drawn as SVG path
shapes in data coordinates — no pixel math, Plotly handles all scaling.

| Function | Description |
|----------|-------------|
| `heatmap` | Interactive hex-grid heatmap with animated slider for time series |

---

### `hex/animate.py`
Animated hex visualizations.

| Class | Description |
|-------|-------------|
| `HexScatter` | Animated hex scatter with matplotlib FuncAnimation |
| `HexQuiver` | Animated quiver plot for flow fields |

---

### `aggregation.py`
Grouped distribution and neuropil time-series visualization.

| Function | Description |
|----------|-------------|
| `plot_group_distribution` | Generic grouped plot API with `violin`, `box`, or `ecdf` |
| `plot_neuropil_timeseries_overview` | Aggregated neuropil overview in wave/heatmap style |
| `plot_neuropil_timeseries_panels` | Region-wise subplot grid for detailed comparison |

---

## Usage Examples

```python
from btorch.visualisation.timeseries import (
    GroupStripOptions,
    NeuronSpec,
    RasterAnnotations,
    RasterGrouping,
    RasterStyle,
    RatePanelOptions,
    SimulationStates,
    TracePlotFormat,
    plot_neuron_traces,
    plot_raster,
)

# Basic raster
plot_raster(spikes, dt=0.1, style=RasterStyle(marker="|", marker_size=5))

# Grouped raster with colors, strip and rate panel
plot_raster(
    spikes,
    style=RasterStyle(spike_color={"excitatory": "red", "inhibitory": "blue"}),
    grouping=RasterGrouping(
        neurons_df=df, group_key="cell_type", show_separators=True
    ),
    strip=GroupStripOptions(side="left"),  # colour strip beside the raster
    rate=RatePanelOptions(total=True, window_ms=10.0),
    annotations=RasterAnnotations(
        events=[100, 200],  # Event markers
        regions=[(50, 80)],  # Shaded regions
        show_tracks=True,
    ),
)

# Neuron traces with per-neuron styling
specs = [NeuronSpec(color="red"), NeuronSpec(color="blue")]
plot_neuron_traces(
    SimulationStates(voltage=V, dt=0.1),
    TracePlotFormat(neuron_specs=specs),
)
```

Raster options are grouped into small frozen dataclasses (`RasterStyle`,
`RasterGrouping`, `GroupStripOptions`, `RatePanelOptions`,
`RasterAnnotations`), all optional keyword arguments of `plot_raster`.

