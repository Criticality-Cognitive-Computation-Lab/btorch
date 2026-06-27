# 更新日志

btorch 的所有重要变更都将记录在此文件中。

本格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)，
并且本项目遵循 [语义化版本控制](https://semver.org/spec/v2.0.0.html)。

## [Unreleased]

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
