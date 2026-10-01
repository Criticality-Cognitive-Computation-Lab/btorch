# 分析模块

`btorch.analysis` 模块提供了用于神经数据分析的计算工具。

## 核心模块

### `spiking.py`

支持 NumPy/PyTorch 双后端的脉冲序列分析工具。

| 函数 | 描述 |
|----------|-------------|
| `isi_cv` | 每个神经元的 ISI 变异系数 |
| `fano` | Fano 因子（脉冲计数的方差/均值） |
| `kurtosis` | 脉冲计数分布的峰度 |
| `local_variation` | 局部变异度 (LV) - 与速率无关的不规则性 |
| `compute_raster` | 提取用于绘图的脉冲时间/神经元索引 |
| `firing_rate` | 将脉冲卷积为发放率 |
| `compute_spectrum` | 通过 Welch 方法计算功率谱 |

**常用参数：**

- `batch_axis`: 用于聚合的轴索引元组（例如，用于 trial 的 `(1, 2)`）
- `percentile`: 计算神经元的百分位数 - `float`、`tuple[float, ...]` 或 `None`

**示例：**

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

通用统计工具。

| 函数 | 描述 |
|----------|-------------|
| `describe_array` | 打印描述性统计信息 |
| `compute_log_hist` | 对数间隔直方图 |
| `get_corr_stats` | 脉冲序列的互相关统计 |

---

### `connectivity.py`

网络连接分析。

| 函数 | 描述 |
|----------|-------------|
| `compute_ie_ratio` | 抑制性/兴奋性输入比例 |
| `HopDistanceModel` | 基于 BFS 的跳数距离计算 |

**HopDistanceModel 方法：**

- `compute_distances(seeds)` → 包含跳数距离的 DataFrame
- `hop_statistics(seeds)` → 按跳数分类的可达性统计
- `reconstruct_path(src, tgt)` → 最短路径

---

### `branching.py`

源自 Wilting & Priesemann (2018) 的 MR 估计。

| 函数 | 描述 |
|----------|-------------|
| `simulate_branching` | 模拟分支过程 |
| `simulate_binomial_subsampling` | 对脉冲序列进行二次采样 |
| `MR_estimation` | 从脉冲计数中估计分支比 |

---

### `aggregation.py`

分组数据聚合。

| 函数 | 描述 |
|----------|-------------|
| `agg_by_neuron` | 按神经元类型聚合 |
| `agg_by_neuropil` | 按神经毡区域聚合 |
| `agg_conn` | 聚合连接权重 |
| `build_group_frame` | 将 `[N]` 或 `[..., N]` 转换为长格式的分组值 |
| `group_values` | 按确定的分组顺序返回分组值数组 |
| `group_summary` | 计算每组的描述性统计信息 |
| `group_ecdf` | 计算用于分析/绘图的每组 ECDF 点 |

---

### `voltage.py`

电压轨迹分析。

| 函数 | 描述 |
|----------|-------------|
| `suggest_skip_timestep` | 建议预热期 |
| `voltage_overshoot` | 量化电压稳定性 |

---

### `metrics.py`

选择与掩码工具。

| 函数 | 描述 |
|----------|-------------|
| `indices_to_mask` | 将索引转换为布尔掩码 |
| `select_on_metric` | 按指标选择神经元 (topk, any) |

---

## `dynamic_tools/` 子包

高级动力系统分析工具。

| 模块 | 描述 |
|--------|-------------|
| `micro_scale.py` | 发放率分布、SPIKE 距离（ISI CV 见 `analysis.spiking.isi_cv`） |
| `fano.py` | 速率补偿的 Fano 因子（操作时间、均值匹配、模型法） |
| `complexity.py` | PCIst、表征对齐、增益稳定性 |
| `criticality.py` | 雪崩分析、幂律拟合、DFA |
| `attractor_dynamics.py` | 相空间重构、Kaplan-Yorke 维数 |
| `lyapunov_dynamics.py` | Lyapunov 指数估计 |
| `ei_balance.py` | E/I 平衡指标（ECI、滞后相关性） |

### E/I 平衡分析 (`ei_balance.py`)

```python
from btorch.analysis.dynamic_tools.ei_balance import (
    compute_eci,
    compute_lag_correlation,
    compute_ei_balance_full
)

# 计算 E/I 抵消指数
eci, info = compute_eci(
    I_e,                  # 兴奋性电流 [T, B, N]
    I_i,                  # 抑制性电流 [T, B, N]
    I_ext=None,           # 外部电流（可选）
    batch_axis=(1,),      # 跨 trial 聚合
    percentile=0.9        # 计算神经元的百分位数
)

# 计算兴奋性和抑制性电流之间的滞后相关性
peak_corr, corr_info = compute_lag_correlation(
    I_e,
    -I_i,                 # 抑制性取负值
    dt=1.0,
    max_lag_ms=30.0,      # 最大滞后（毫秒）
    use_fft=True          # 基于 FFT 以提高效率
)

# 全面的 E/I 平衡分析
metrics, info = compute_ei_balance_full(
    I_e, I_i,
    I_ext=None,
    dt=1.0,
    max_lag_ms=30.0,
    batch_axis=(1,)
)
# metrics: eci_mean, eci_median, track_corr_peak_mean, delay_ms_mean, 等。
```

---

## 使用示例

```python
from btorch.analysis.spiking import firing_rate, fano
from btorch.analysis.branching import MR_estimation

# 计算发放率
fr = firing_rate(spikes, width=10, dt=0.1)

# 跨窗口的 Fano 因子
ff, info = fano(spikes, window=100)

# 分支比估计
result = MR_estimation(spike_counts)
print(f"Branching ratio: {result['branching_ratio']:.3f}")
```

---

## 后端支持

所有脉冲分析函数均支持 NumPy 和 PyTorch：

- **NumPy**: 标准的基于 CPU 的计算
- **PyTorch**: 在有益的情况下使用 GPU 加速
    - 基于 ISI 的指标 (CV, LV)：混合方法（GPU 聚合 → CPU 提取 → GPU 返回）
    - 基于计数的指标 (Fano, Kurtosis)：通过累积和实现全 GPU 计算

**Float16 支持：**

- 函数接受 float16 输入
- 内部累加使用 float32 以保证数值精度
- 返回值与输入设备位置保持一致