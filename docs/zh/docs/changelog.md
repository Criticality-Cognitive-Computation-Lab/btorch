# 更新日志

btorch 的所有重要变更都将记录在此文件中。

本格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)，
并且本项目遵循 [语义化版本控制](https://semver.org/spec/v2.0.0.html)。

## [Unreleased]

### 新增
- `btorch.sparse`：SciPy/PyTorch 风格的稀疏数组 API（`Sparse`、`COO`、`CSR`、`CSC`、`sparse.from_edges`、`sparse.asarray`、`A @ x`、`sparse.stack`、`to_torch()` / `to_scipy()`）；采用标准矩阵方向，转换从不转置、也不稠密化。
- `btorch.models.connection`：`SparseConnection`（`from_adjacency`、`from_edges`、`from_hetersynapse`）、`Synapse`、`EdgeWeight`、`ConstantWeight`、`ConstrainedWeight`；受体与延迟作为边属性；支持网络批。
- `btorch.sparse.runtime`：执行层（注册算子、后端注册表、规划器）；编写模型时无需使用。
- 指南：[稀疏连接](guides/sparse_connectivity.md)。

### 变更
- **破坏性变更：** `btorch.models.linear` 中的 `SparseConn`、`SparseConstrainedConn`、`BaseSparseConn`、`SparseBackend`、`available_sparse_backends` 以及 `sparse_backend=` 参数已移除，没有兼容层。请改用 `SparseConnection.from_adjacency(conn, Synapse(dale=...))`；Dale 定律现需显式开启，`state_dict` 的键也已改变。迁移对照表见[稀疏连接指南](guides/sparse_connectivity.md)。
- **破坏性变更：** `memories_to_xarray` / `save_memories_to_xarray` 的选项归入 `DimLayout`、`SparseOptions` 与 `ZarrStoreOptions`（`btorch.io`）；`neuron_ids` 与 `partial_map` 变为仅关键字参数。
- **破坏性变更：** 在第 `t` 步送达的脉冲现在会影响第 `t` 步返回的 PSC，所有 PSC 类型一致（此前 `AlphaPSC`、`AlphaPSCBilleh`、`DualExponentialPSC` 多花一个 `dt`）。
- **破坏性变更：** `GLIF3.forward_exact_no_spike(x, t=None, v0=None, Iasc0=None, t_mode="homo")` 现为纯函数（不再更新状态）；逐元素原语 `exact_no_spike_at(x, t, v0, Iasc0)` 支持批量的异质时间与状态，例如用于迭代求根。`t_mode="heter"` 表示逐元素时间。
- **破坏性变更：** 绘图与拟合选项归入数据类（`TbpttConfig`、`GlobalSearchConfig`、`StagedConfig`、`FitLossConfig`、raster 选项），见[分析](analysis.md)与[可视化](visualisation.md)页面。
- **破坏性变更：** 六边形 `scatter` / `quiver` 使用 `HexGeometry`、`HexColorMap`、`HexPatchStyle`、`HexReference`；`plot_grouped_spectrum` 使用 `SpectrumGrouping` 与 `SpectrumStyle`。`quiver` 现在会遵循 `rotation_deg` 与颜色范围。
- **破坏性变更：** 包布局按层次调整。`btorch.datasets.noise` -> `btorch.models.noise`；`btorch.datasets.transforms` -> `btorch.utils.hex.augment`；`btorch.analysis.two_compartment_fit` -> `btorch.fitting.two_compartment`，`btorch.analysis.tuning` -> `btorch.fitting.tuning`（拟合相关名称不再从 `btorch.analysis` 重新导出）。`btorch.datasets` 包已移除。
- 分析函数命名规则：估计器/流水线统一为 `compute_*`。重命名：`branching_ratio` -> `compute_branching_ratio`，`get_slopes` -> `compute_lagged_slopes`，`get_continuous_spiking_rate` -> `compute_continuous_spiking_rate`，`voltage_overshoot` -> `compute_voltage_overshoot`。
- 移除 `plot_group_violin`、`plot_group_box`、`plot_group_ecdf`；请使用 `plot_group_distribution(..., kind="violin" | "box" | "ecdf")`。
- 较重的依赖现为可选扩展（`io`、`config`、`fast`、`gpu`、`analysis`、`viz`、`sparse`、`examples`、`all`），见[安装](installation.md)。
- `fano_population` / `kurtosis_population` 标注为 `StatsResult`；多输出的装饰器分析函数（`compute_lag_correlation`、E/I 平衡）标注为 `MultiStatsResult`（新增，位于 `btorch.analysis.statistics`）。
- `make_hetersynapse_constraint` 在同一处构造约束键；结果不变。

### 修复
- 稀疏连接的 Dale 符号与约束结构现保存在 `state_dict` 中；此前在 `load_state_dict` 之后会过期。
- 稀疏连接无需 `torch_sparse` 即可通过 `torch.compile(conn, fullgraph=True)` 编译。
- 纯 PyTorch 稀疏路径的反向传播在约 10 万神经元规模下不再内存不足。
- `make_hetersynapse_conn` 的延迟处理。
- `plot_multiscale_fano` 使用不支持的 `group_by` 时现抛出 `ValueError`，而非 `NameError`。
- 幂律缩放拟合在输入为常数时 `r_squared` 返回 `NaN`，不再除以零。

## [0.1.0]

### 新增
- **双室神经元**（`TwoCompartmentGLIF`）——具有非线性顶端平台电位、双向耦合和可选自适应阈值的体树突神经元。参见[教程](tutorials/mixed_neurons.md)。
- **混合神经元群体**（`MixedNeuronPopulation`）——在单个循环层中混合多种神经元类型（如 GLIF3 + TwoCompartmentGLIF），支持自动电流切片与脉冲拼接。
- **异构 RNN**（`HeteroRecurrentNN`）——`RecurrentNN` 的替代实现，接受 `MixedNeuronPopulation`。
- **六边形网格模块**（`btorch.utils.hex`）——坐标系统（axial、doubled、zigzag、flywire）、结构体数组数据类型、卷积层、眼渲染模型，以及带叠加层和指南针的 SVG 可视化。参见[六边形文档](hex.md)。
- **类型注解**——`btorch/py.typed`（PEP 561）以及 `btorch.analysis.spiking`、`btorch.models.neurons.two_compartment`、`btorch.utils.hex` 中完整的返回类型注解。
- **发布 CI**——GitHub Actions 工作流，当推送 `v*` 标签时构建分发包并通过可信发布上传至 PyPI（仅手动触发）。
- **Codecov**——配置文件，包含覆盖率阈值、标志管理和行内 PR 注解。

### 变更
- **代理梯度重构**——所有代理梯度导数现在对**任意** `alpha` 值均满足
  `g(v=0, damping_factor=1) == 1.0`（Zenke & Neftci 2021），
  且 `alpha = 1/HWHM` 在所有代理函数中统一成立。默认 `alpha` 值已更新。
  参见[代理梯度指南](concepts/surrogate_gradients.md) 了解迁移说明。
- **构建系统迁移至 uv**——`uv.lock` 取代 pip 锁文件；CI 使用 `uv sync` 配合 PyTorch CPU 索引。
- **文档迁移至 Zensicle**——使用 Zensicle + mkdocstrings 取代 mkdocs/myst/sphinx。英文和中文文档现在通过同一条流水线构建，并支持 AI 辅助翻译。
- **Conda 环境重命名**——`dev-requirements.yaml` → `environment.yml`。
- **RNN 类重命名**——清理了公开导出名称。

### 破坏性变更
所有代理梯度导数已重新归一化，使得对**任意** `alpha` 值均满足
`g(v=0, damping_factor=1) == 1.0`
（Zenke & Neftci 2021，*Neural Computation* 33(4)）。

此前，各导数均被缩放至在电压上积分为 1——其动机类比于概率密度函数。
但事实证明这并非正确的不变量：对稳定学习真正重要的是在**阈值处**的单位响应，
而非单位积分。

| 代理函数    | 旧峰值（v=0 处） | 施加因子   | 新峰值 |
|------------|----------------|-----------|-------|
| `Triangle`  | `alpha`        | `1/alpha` | 1     |
| `Sigmoid`   | `alpha/4`      | `4/alpha` | 1     |
| `Erf`       | `alpha/√π`     | `√π/alpha`| 1     |
| `ATan`      | `alpha/2`      | `2/alpha` | 1     |
| `ATanApprox`| `alpha/2`      | `2/alpha` | 1     |

`SuperSpike` 及 Heaviside 前向传播不受影响。

**迁移建议：** 使用上述任意代理函数训练的模型将产生不同的有效梯度幅值。
建议从头重新训练，或将现有 `damping_factor` 乘以旧峰值的倒数以保持幅值
（例如：`ATan` 在 `alpha=2` 时旧峰值为 1.0，无需调整；在 `alpha=4` 时
旧峰值为 2.0，需设置 `damping_factor=2.0`）。

所有代理梯度已重新参数化，使得 `alpha = 1/HWHM` 对所有代理梯度统一成立。
`g(v)` 的半高全宽（HWHM）现在精确等于 `1/alpha`（ATanApprox 因有理近似存在约 8% 误差）。

| 代理梯度    | 内部常数           | 新默认 α | 旧默认 α |
|------------|-------------------|-------------|-------------|
| `Triangle`  | k = 1/2           | 2.0         | 1.0 |
| `Sigmoid`   | k = 2ln(√2+1)≈1.763 | 2.0       | 1.0 |
| `Erf`       | k = √ln2≈0.833    | 4.0         | 2.0 |
| `ATan`      | k = 1（原为 π/2） | 2.0         | 2.0 |
| `ATanApprox`| k ≈ 1             | 2.0         | 2.0 |
| `SuperSpike`| k = √2−1≈0.414    | 2.0         | 4.0 |

**迁移建议：** 如果依赖之前的 `alpha` 值，在旧 `alpha` 处的梯度宽度现在有所不同。
建议通过搜索重新调整 `alpha`。

### 移除
- **移除 `pytorch_sparse` 硬依赖**——稀疏线性层现在默认使用 PyTorch 原生的
  `torch.sparse` 后端。`torch_sparse` 仍可作为可选安装用于大规模稀疏网络场景。
- Sphinx、myst-parser 和已废弃的 pip 锁文件。
- README 中的 AI 智能体提示章节（已替换为清晰的安装说明）。
