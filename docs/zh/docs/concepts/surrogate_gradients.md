# 代理梯度

脉冲神经元使用不连续的激活函数（当膜电位超过阈值时发放脉冲）。这种不连续性使得标准的时间反向传播（BPTT）无法执行，因为脉冲函数几乎处处梯度为零。

代理梯度通过在反向传播时用平滑近似代替真实梯度来解决这一问题。

## 可用的代理梯度

btorch 在 `btorch.models.surrogate` 中提供了多种代理梯度函数：

| 类名 | 代理梯度 `g(v)` | 默认 alpha |
|------|----------------|-----------|
| `ATan` | `1 / (1 + (α v)²)` | 2.0 |
| `ATanApprox` | `ATan` 的有理近似 | 2.0 |
| `Sigmoid` | `4 σ(k·α·v)(1 − σ(k·α·v))`，k = 2 ln(√2+1) | 2.0 |
| `Erf` | `2^{−(α v)²}` | 4.0 |
| `Triangle` | `(1 − \|α v\| / 2)₊` | 2.0 |
| `SuperSpike` | `(1 + (√2−1)·α·\|v\|)⁻²` | 2.0 |

所有默认值对应半宽度为 0.5（见下方 [alpha 约定](#约定二--alpha-约定hwhm--1alpha)），`Erf` 例外，其默认 alpha=4（HWHM=0.25）是为了匹配 Chen 等人（2022）的 V1 模型而校准的。

## 约定一 — 峰值归一化：`g(0) = 1`

所有 btorch 代理梯度均满足：

> **对于任意 `alpha`，`g(v=0, damping_factor=1) == 1.0`。**

这确保在不进行人为缩放的情况下，脉冲阈值处的有效梯度幅度始终为 1。切换代理梯度或改变 `alpha` 不会意外地重新缩放学习信号。

`damping_factor` 是**唯一**用于人为控制梯度幅度的参数。

该约定遵循 Zenke & Neftci（2021），他们通过实验证明，良好代理梯度的关键特性是在阈值处的单位响应，而非电位上的单位积分。此前，btorch（和大多数库）将每个导数缩放为对电位积分为 1——受到概率密度类比的启发——但这是错误的不变量。

> Zenke, F., & Neftci, E. O. (2021). *The remarkable robustness of surrogate
> gradient learning for instilling complex function in spiking neural networks.*
> Neural Computation, 33(4), 899–925.
> https://doi.org/10.1162/neco_a_01097

各代理梯度的归一化系数：

| 代理梯度 | 未归一化峰值 | 内嵌系数 | 归一化后的 `g(v)` |
|---------|------------|---------|-----------------|
| Triangle   | `alpha/2` | `2/alpha` | `(1 − \|αv\|/2)₊` |
| Sigmoid    | `alpha/4` | `4/alpha` | `4σ(k·αv)(1−σ)`，k=2ln(√2+1) |
| Erf        | `alpha/√π` | `√π/alpha` | `2^{−(αv)²}` |
| ATan       | `alpha/2` | `2/alpha` | `1/(1+(αv)²)` |
| ATanApprox | `alpha/2` | `2/alpha` | 有理近似 |
| SuperSpike | 1         | —        | `(1+(√2−1)α\|v\|)⁻²` |

## 约定二 — Alpha 约定：HWHM = 1/alpha

btorch 还对所有代理梯度中 `alpha` 的含义进行了标准化：

> **`alpha` 是半高全宽（HWHM）的倒数。对于任意代理梯度，当 `damping_factor = 1` 时，`g(1/alpha) = 0.5`。**

这意味着相同的 `alpha` 值对不同的代理梯度给出相同的梯度宽度。SpikingJelly 和大多数其他库不采用此约定——它们的 `alpha` 尺度在不同代理梯度之间相差最多 4 倍。

每种代理梯度通过在其内部参数中内嵌一个无理数常数来实现这一点：

| 代理梯度 | 内部参数 | HWHM 解析值 |
|---------|---------|------------|
| Triangle   | `alpha·v / 2` | `1/alpha`（精确） |
| Sigmoid    | `2ln(√2+1)·alpha·v` ≈ `1.763·alpha·v` | `1/alpha`（精确） |
| Erf        | `alpha·v` | `1/alpha`（精确，`g = 2^{−(αv)²}`） |
| ATan       | `alpha·v` | `1/alpha`（精确） |
| ATanApprox | `alpha·v` | `≈ 0.92/alpha`（近似，有理近似误差） |
| SuperSpike | `(√2−1)·alpha·v` ≈ `0.414·alpha·v` | `1/alpha`（精确） |

## 使用方法

大多数神经元构造函数接受 `surrogate_function` 参数：

```python
from btorch.models.neurons import LIF
from btorch.models.surrogate import ATan, Erf

# 默认 ATan，HWHM = 1/2 = 0.5
neuron = LIF(n_neuron=100, surrogate_function=ATan(alpha=2.0))

# 匹配 Chen 等人（2022）V1 模型的 Erf
neuron = LIF(n_neuron=100, surrogate_function=Erf(alpha=4.0, damping_factor=0.5))
```

## 选择代理梯度

- **ATan** — 柯西/洛伦兹核；平滑且具有多项式尾部。良好的通用默认选择。
- **ATanApprox** — ATan 的有理近似；避免 `atan` 调用的开销。
- **Sigmoid** — 指数尾部；在远离阈值处具有更强的梯度信号。
- **Triangle** — 紧支撑（在 `|v| > 2/alpha` 外为零）；计算成本低。
- **Erf** — 高斯尾部；次指数衰减，梯度非常局部。默认 alpha=4 匹配 V1 模型（Chen 等人 2022）。
- **SuperSpike** — 幂律（重）尾部；适用于不规则或稀疏活动（Zenke & Ganguli 2018）。

## 添加新的代理梯度

继承 `SurrogateFunctionBase` 并实现 `primitive` 和 `derivative`。提交前必须满足两个约定：

```python
import torch

x = torch.tensor(0.0, requires_grad=True)
MySurrogate(alpha=1.0, damping_factor=1.0)(x).backward()
assert abs(x.grad.item() - 1.0) < 1e-5, "峰值归一化失败"

x = torch.tensor(1.0, requires_grad=True)  # v = 1/alpha，此时 alpha=1
MySurrogate(alpha=1.0, damping_factor=1.0)(x).backward()
assert abs(x.grad.item() - 0.5) < 0.02, "HWHM 约定失败"
```

`tests/models/test_surrogate.py` 中的 `test_unit_gradient_at_threshold` 和 `test_consistent_hwhm` 测试会自动对所有内置代理梯度强制执行两个约定。

## 迁移指南

### 从 SpikingJelly 迁移

SpikingJelly 的 `alpha` 在不同代理梯度之间含义不一致——梯度宽度和阈值处的峰值均以各自特定的方式随 `alpha` 缩放。btorch 通过两个约定统一了这两点（峰值始终为 1，HWHM 始终为 1/alpha）。

要在移植时保持**相同的梯度宽度**，需按如下方式将 SpikingJelly 的 `alpha_sj` 转换为 btorch 的 `alpha_bt`：

| SJ 代理梯度 | SJ HWHM | btorch 对应 | 转换关系 |
|---|---|---|---|
| `Sigmoid(alpha_sj)` | `1.763/alpha_sj` | `Sigmoid` | `alpha_bt = 1.763 * alpha_sj` |
| `ATan(alpha_sj)` | `2/(π·alpha_sj)` | `ATan` | `alpha_bt = 2/π · alpha_sj ≈ 0.637 * alpha_sj` |
| `Triangle(alpha_sj)` | `1/alpha_sj` | `Triangle` | `alpha_bt = alpha_sj`（相同） |

要保持**相同的峰值幅度**，需将 `damping_factor` 设为原峰值：

| SJ 代理梯度 | SJ 峰值（v=0） | btorch `damping_factor` |
|---|---|---|
| `Sigmoid(alpha_sj)` | `alpha_sj / 4` | `alpha_sj / 4` |
| `ATan(alpha_sj)` | `alpha_sj / 2` | `alpha_sj / 2` |
| `Triangle(alpha_sj)` | `alpha_sj` | `alpha_sj` |

**示例**——将 SpikingJelly 的 `ATan(alpha=2)` 移植到 btorch：

```python
# SpikingJelly: HWHM = 2/(π·2) ≈ 0.318，峰值 = 2/2 = 1.0
# btorch 等效写法（保持宽度和幅度）：
import math
from btorch.models.surrogate import ATan
alpha_sj = 2.0
surrogate = ATan(alpha=2/math.pi * alpha_sj, damping_factor=alpha_sj/2)
# ATan(alpha≈0.637, damping_factor=1.0) — 峰值保持 1，HWHM 保持 0.318
```

### 从 braintools / brainstate 迁移

braintools 使用 JAX，内部缩放与 btorch 不同。以下是代理梯度的对应关系（令 btorch 的 `alpha_bt = 1/HWHM` 即可推算）：

| braintools 代理梯度 | bt HWHM | btorch 对应 | 转换关系 |
|---|---|---|---|
| `Sigmoid(alpha)` | `1.763/alpha` | `Sigmoid` | `alpha_bt = 1.763 * alpha` |
| `ATan(alpha)` | `2/(π·alpha)` | `ATan` | `alpha_bt = 2/π · alpha ≈ 0.637 * alpha` |
| `SuperSpike(alpha)` | `(√2−1)/alpha` | `SuperSpike` | `alpha_bt = (√2−1) * alpha ≈ 0.414 * alpha` |
| `PiecewiseQuadratic(alpha)` | `1/alpha` | `Triangle` | `alpha_bt = alpha`（形状相同，名称不同） |
| `PiecewiseExp(alpha)` | `ln2/alpha` | — | btorch 无精确对应 |
| `Erf(alpha)` | `√ln2/alpha` | `Erf` | `alpha_bt = √ln2 * alpha ≈ 0.833 * alpha` |

要保持**相同的峰值幅度**，需将 `damping_factor` 设为 braintools 的峰值：

| braintools 代理梯度 | bt 峰值（v=0） | btorch `damping_factor` |
|---|---|---|
| `Sigmoid(alpha)` | `alpha/4` | `alpha/4` |
| `ATan(alpha)` | `alpha/2` | `alpha/2` |
| `SuperSpike(alpha)` | `alpha/2` | `alpha/2` |
| `PiecewiseQuadratic(alpha)` | `alpha` | `alpha` |
| `PiecewiseExp(alpha)` | `alpha/2` | `alpha/2` |
| `Erf(alpha)` | `alpha/√π` | `alpha/√π` |

**示例**——将 braintools 的 `Erf(alpha=2)` 移植到 btorch：

```python
# braintools: HWHM = sqrt(ln2)/2 ≈ 0.416，峰值 = 2/sqrt(pi) ≈ 1.128
import math
from btorch.models.surrogate import Erf
alpha_bt_lib = 2.0
surrogate = Erf(
    alpha=math.sqrt(math.log(2)) * alpha_bt_lib,       # ≈ 1.665，HWHM = 0.416
    damping_factor=alpha_bt_lib / math.sqrt(math.pi),  # ≈ 1.128
)
```

## 参考文献

- Zenke, F., & Neftci, E. O. (2021). *The remarkable robustness of surrogate gradient learning for instilling complex function in spiking neural networks.* Neural Computation, 33(4), 899–925.
- Zenke, F., & Ganguli, S. (2018). *SuperSpike: Supervised learning in multi-layer spiking neural networks.* Neural Computation, 30(6), 1514–1541.
- Chen, G., Scherr, F., & Maass, W. (2022). *A data-based large-scale model for primary visual cortex enables brain-like robust and versatile visual processing.* Science Advances, 8(44), eabq7592.
