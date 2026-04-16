# 可视化模块

`btorch.visualisation` 模块为神经仿真分析提供了绘图函数。

## 模块

### `timeseries.py`
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

### `aggregation.py`
分组分布和神经毯 (neuropil) 时间序列可视化。

| 函数 | 描述 |
|----------|-------------|
| `plot_group_distribution` | 通用分组绘图 API，支持 `violin`、`box` 或 `ecdf` |
| `plot_group_violin` | 分组小提琴图便捷封装 |
| `plot_group_box` | 分组箱线图便捷封装 |
| `plot_group_ecdf` | 分组 ECDF 图便捷封装 |
| `plot_neuropil_timeseries_overview` | 波形/热图风格的聚合神经毯概览 |
| `plot_neuropil_timeseries_panels` | 用于详细对比的区域级子图网格 |

---

## 使用示例

```python
from btorch.visualisation.timeseries import plot_raster, plot_neuron_traces, NeuronSpec

# 基础光栅图
plot_raster(spikes, dt=0.1, marker="|", markersize=5)

# 带颜色的分组光栅图
plot_raster(
    spikes,
    neurons_df=df,
    group_by="cell_type",
    color={"excitatory": "red", "inhibitory": "blue"},
    show_separators=True,
    events=[100, 200],  # 事件标记
    regions=[(50, 80)],  # 阴影区域
    show_tracks=True,
)

# 具有单个神经元样式的神经元轨迹
specs = [NeuronSpec(color="red"), NeuronSpec(color="blue")]
plot_neuron_traces(voltage=V, dt=0.1, neuron_specs=specs)
```