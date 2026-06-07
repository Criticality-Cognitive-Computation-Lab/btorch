# 更新日志

btorch 的所有重要变更都将记录在此文件中。

本格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)，
并且本项目遵循 [语义化版本控制](https://semver.org/spec/v2.0.0.html)。

## [未发布]

### 新增
- **双室神经元**（`TwoCompartmentGLIF`）：具有非线性顶端平台电位、双向耦合和可选自适应阈值的体树突神经元。参见[教程](tutorials/mixed_neurons.md)。
- **混合神经元群体**（`MixedNeuronPopulation`）：在单个循环层中混合多种神经元类型（如 GLIF3 + TwoCompartmentGLIF），支持自动电流切片与脉冲拼接。
- **异构 RNN**（`HeteroRecurrentNN`）：`RecurrentNN` 的替代实现，接受 `MixedNeuronPopulation`。
- **GeNN/CUDA 后端**（`btorch.backend`）：基于 GeNN 的实验性脉冲仿真后端，面向 GPU 加速的大规模网络。

### 变更
- conda 环境文件由 `dev-requirements.yaml` 重命名为 `environment.yml`。

### 破坏性变更
所有代理梯度导数已重新归一化，使得对**任意** `alpha` 值均满足
`g(v=0, damping_factor=1) == 1.0`
（Zenke & Neftci 2021，*Neural Computation* 33(4)）。

此前，各导数均被缩放至在电压上积分为 1——其动机类比于概率密度函数。
但事实证明这并非正确的不变量：对稳定学习真正重要的是在**阈值处**的单位响应，
而非单位积分。Zenke & Neftci 的研究表明，正是这一属性使代理梯度学习在
不同网络配置下均具有良好的鲁棒性。

各受影响代理函数及其归一化因子如下：

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
此前，相同的 `alpha` 值在不同代理梯度中产生不同的梯度宽度，各代理梯度之间差异最大可达 4 倍。
现在相同的 `alpha` 在所有代理梯度中给出相同的梯度半宽度。

每种代理梯度通过在其内部参数中内嵌一个特定常数来实现这一点：

| 代理梯度    | 内部常数           | 新默认 alpha | 旧默认 alpha |
|------------|-------------------|-------------|-------------|
| `Triangle`  | k = 1/2           | 2.0         | 1.0 |
| `Sigmoid`   | k = 2ln(√2+1)≈1.763 | 2.0       | 1.0 |
| `Erf`       | k = √ln2≈0.833    | 4.0         | 2.0 |
| `ATan`      | k = 1（原为 π/2） | 2.0         | 2.0 |
| `ATanApprox`| k ≈ 1             | 2.0         | 2.0 |
| `SuperSpike`| k = √2−1≈0.414    | 2.0         | 4.0 |

**迁移建议：** 如果依赖之前的 `alpha` 值，在旧 `alpha` 处的梯度宽度现在有所不同。
建议通过搜索重新调整 `alpha`。
