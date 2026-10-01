# 可视化模块

`btorch.visualisation` 模块为神经仿真分析提供了绘图函数。

## 模块

### `timeseries/` (package)
脉冲和连续数据的时间序列可视化。

| 函数 | 描述 |
|----------|-------------|
| `plot_raster` | 具有分组、样式设置、事件、区域和轨道显示的脉冲光栅图 (Spike raster) |
| `plot_traces` | 连续轨迹（电压、电流） |
| `plot_spectrum` | 频谱（Welch 方法） |
| `plot_grouped_spectrum` | 按神经元组进行频谱分析 |
| `plot_log_hist` | 双对数直方图 |
| `plot_neuron_traces` | 多面板神经元状态图（电压、ASC、PSC） |

**数据类 (Dataclasses):**
- `NeuronSpec`: 单个神经元样式（颜色、标记、线型）
- `SimulationStates`: 仿真数据容器
- `TracePlotFormat`: 图形格式化选项
- `RasterStyle`、`RasterGrouping`、`GroupStripOptions`、`RatePanelOptions`、`RasterAnnotations`: `plot_raster` 的选项分组
- `SpectrumGrouping`、`SpectrumStyle`: `plot_grouped_spectrum` 的选项分组，例如 `plot_grouped_spectrum(data, dt, SpectrumGrouping(neurons_df=df, group_by="type"), style=SpectrumStyle(show_traces=False))`

---

### `dynamics.py`
多尺度动力学分析可视化。

| 函数 | 描述 |
|----------|-------------|
| `plot_multiscale_fano` | 跨时间窗口的 Fano 因子 |
| `plot_dfa_analysis` | 去趋势波动分析 (DFA) |
| `plot_isi_cv` | ISI 变异系数 |
| `plot_avalanche_analysis` | 雪崩规模/持续时间分布 |
| `plot_eigenvalue_spectrum` | 权重矩阵特征值谱 |
| `plot_lyapunov_spectrum` | Lyapunov 指数谱 |
| `plot_firing_rate_distribution` | 放电率直方图 |

**数据类 (Dataclasses):**
- `DynamicsData`: 脉冲数据容器
- `DynamicsPlotFormat`: 可视化模式（个体/分组/分布）
- `FanoFactorConfig`: Fano 分析参数
- `DFAConfig`: DFA 参数

---

### `hexmap.py`
使用 Plotly 的六边形热图可视化。

| 函数 | 描述 |
|----------|-------------|
| `hex_heatmap` | 带有时间序列滑块的交互式六边形网格热图 |

---

### `hex/static.py`
使用 matplotlib 的静态六边形绘图。

| 函数 | 描述 |
|----------|-------------|
| `scatter` | 带颜色映射的六边形散点图 |
| `quiver` | 六边形网格上的矢量场（流场）绘图 |
| `grid` | 带可选坐标标注的六边形网格 |
| `draw_axes` | 在六边形图上叠加 q/r/s 轴箭头 |
| `compass` | 指南针玫瑰嵌入图 |
| `looming_stimulus` | 从中心扩展的刺激序列 |

**数据类**（`hex/options.py`）：`HexGeometry`（布局、尺寸、朝向、旋转）、`HexColorMap`（cmap、vmin、vmax）、`HexPatchStyle`（边框、透明度）、`HexReference`（q/r/s 轴、指南针）。它们作为仅限关键字参数 `geometry=`、`color=`、`patch=`（仅 `scatter`）和 `reference=` 传给 `scatter`/`quiver`，例如 `scatter(q, r, v, color=HexColorMap(cmap="RdBu_r", vmin=-1, vmax=1))`。

---

### `hex/interactive.py`
使用 Plotly 的交互式六边形热图。六边形多边形以 SVG 路径形状绘制在数据坐标中——无数学像素计算，Plotly 处理所有缩放。

| 函数 | 描述 |
|----------|-------------|
| `heatmap` | 带动画滑块的交互式六边形网格热图 |

---

### `hex/animate.py`
动画六边形可视化。

| 类 | 描述 |
|-------|-------------|
| `HexScatter` | 使用 matplotlib FuncAnimation 的动画六边形散点图 |
| `HexQuiver` | 流场动画矢量图 |

---

### `aggregation.py`
分组分布和神经毯 (neuropil) 时间序列可视化。

| 函数 | 描述 |
|----------|-------------|
| `plot_group_distribution` | 通用分组绘图 API，支持 `violin`、`box` 或 `ecdf` |
| `plot_neuropil_timeseries_overview` | 波形/热图风格的聚合神经毯概览 |
| `plot_neuropil_timeseries_panels` | 用于详细对比的区域级子图网格 |

---

## 使用示例

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

# 基础光栅图
plot_raster(spikes, dt=0.1, style=RasterStyle(marker="|", marker_size=5))

# 带颜色、分组色条和发放率面板的分组光栅图
plot_raster(
    spikes,
    style=RasterStyle(spike_color={"excitatory": "red", "inhibitory": "blue"}),
    grouping=RasterGrouping(
        neurons_df=df, group_key="cell_type", show_separators=True
    ),
    strip=GroupStripOptions(side="left"),  # colour strip beside the raster
    rate=RatePanelOptions(total=True, window_ms=10.0),
    annotations=RasterAnnotations(
        events=[100, 200],  # 事件标记
        regions=[(50, 80)],  # 阴影区域
        show_tracks=True,
    ),
)

# 具有单个神经元样式的神经元轨迹
specs = [NeuronSpec(color="red"), NeuronSpec(color="blue")]
plot_neuron_traces(
    SimulationStates(voltage=V, dt=0.1),
    TracePlotFormat(neuron_specs=specs),
)
```

光栅图选项被分组为若干小型冻结数据类（`RasterStyle`、`RasterGrouping`、
`GroupStripOptions`、`RatePanelOptions`、`RasterAnnotations`），均为
`plot_raster` 的可选关键字参数。
