# 替代梯度

脉冲神经元使用不连续的激活函数（当膜电压超过阈值时发出脉冲）。这种不连续性使得标准的时间反向传播（BPTT）无法实现，因为脉冲函数的梯度几乎处处为零。

替代梯度通过在反向传播期间用平滑近似替换真实梯度来解决这一问题。

## 可用的替代梯度

btorch 在 `btorch.models.surrogate` 中提供了多种替代梯度函数：

| 类 | 前向 | 反向 |
|-------|---------|----------|
| `Sigmoid` | Heaviside | Sigmoid 导数 |
| `ATan` | Heaviside | 反正切导数 |
| `Triangle` | Heaviside | 分段线性 |
| `Erf` | Heaviside | 高斯（误差函数）导数 |

## 用法

大多数神经元构造函数接受 `surrogate_function` 参数：

```python
from btorch.models.neurons import LIF
from btorch.models.surrogate import ATan

neuron = LIF(
    n_neuron=100,
    surrogate_function=ATan(),
)
```

如果未指定，通常会使用一个合理的默认值（通常是 `ATan`）。

## 选择替代梯度

- **ATan** — 平滑、行为良好的梯度；大多数任务的理想默认选择。
- **Sigmoid** — 远离阈值时梯度更强；有助于处理非常稀疏的活动。
- **Triangle** — 计算成本低；有界支撑。
- **Erf** — 非常平滑；有时有助于优化稳定性。

参阅 [Neurons API](../api/neurons.md) 了解构造函数详情。
