# 混合神经元种群与多输入 RNN

本指南展示如何构建在同一层中包含**多种神经元类型**的递归网络，以及如何向其中的一部分神经元提供**额外的输入流**（例如顶端/自上而下的驱动）。

## 你将学到

- 如何使用 `MixedNeuronPopulation` 将 `GLIF3`、`TwoCompartmentGLIF` 和其他神经元组合到同一层中。
- 如何使用 `ApicalRecurrentNN` 包裹种群，使顶端输入正确地进行时间展开。
- 当子种群具有不同的内部变量时，状态收集如何工作。

## 何时使用

当你的模型需要以下功能时，可以使用这些工具：

1. **异质细胞类型** — 例如 80% 的快速放电中间神经元 (`GLIF3`) 和 20% 的锥体细胞 (`TwoCompartmentGLIF`)。
2. **多室神经元** — 例如体细胞前馈输入 + 顶端自上而下输入。
3. **结构化状态读出** — 每个子种群通过点分状态名暴露其自身的电压/电流。

## 构建混合种群 RSNN

### 1. 创建子种群

每个子种群都是一个普通的 btorch 神经元模块。它们可以具有不同的参数集、不同的状态变量，甚至不同的 `step_mode`（只要包装器在你预期的模式下调用即可）。

```python
from btorch.models.neurons import GLIF3, TwoCompartmentGLIF
from btorch.models.neurons.mixed import MixedNeuronPopulation
from btorch.models.linear import DenseConn
from btorch.models.synapse import AlphaPSC
from btorch.models.rnn import ApicalRecurrentNN

n_neuron = 100
n_glif   = 80
n_tc     = 20

glif = GLIF3(n_neuron=n_glif, step_mode="s")
tc   = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")

mixed = MixedNeuronPopulation(
    [(n_glif, glif), (n_tc, tc)],
    step_mode="s",
)
```

`MixedNeuronPopulation` 沿最后一个（神经元）维度对输入电流进行切片，并将正确的切片分发给每个子模块。脉冲输出会重新拼接在一起，因此输出形状始终为 `(*batch, n_neuron)`。

### 2. 添加递归突触

突触看到的是**拼接后**的脉冲，因此连接矩阵必须为 `n_neuron x n_neuron`：

```python
conn = DenseConn(n_neuron, n_neuron, bias=None)
psc  = AlphaPSC(
    n_neuron=n_neuron,
    tau_syn=5.0,
    linear=conn,
    step_mode="s",
)
```

### 3. 用 `ApicalRecurrentNN` 包装

`ApicalRecurrentNN` 是 `RecurrentNN` 的子类，它接受第三个位置参数 `x_apical`。当你用时间序列调用它时，外部的展开循环会自动切片 `x_apical`，**前提是你以位置参数的形式传递它**：

```python
brain = ApicalRecurrentNN(
    neuron=mixed,
    synapse=psc,
    step_mode="m",          # 多步包装器
    unroll=4,
    update_state_names=(
        "neuron.group_0.v",
        "neuron.group_1.v",
        "neuron.group_1.i_a",
        "synapse.psc",
    ),
)
```

### 4. 初始化和运行

```python
from btorch.models import functional, environ

functional.init_net_state(brain, batch_size=4)

T = 100
x_soma   = torch.randn(T, 4, n_neuron)
x_apical = torch.randn(T, 4, n_neuron)   # 实际上只有 TC 切片被使用

with environ.context(dt=1.0):
    spikes, states = brain(x_soma, None, x_apical)

print(spikes.shape)                       # (T, 4, 100)
print(states["neuron.group_1.i_a"].shape) # (T, 4, 20)
```

## `MixedNeuronPopulation` 如何处理顶端切片

`TwoCompartmentGLIF` 期望两个参数：体细胞电流和顶端电流。`MixedNeuronPopulation` 知道这一点，并自动为每个 `TwoCompartmentGLIF` 子模块切片顶端张量。对于 `GLIF3`（以及任何其他单输入神经元），顶端切片将被简单忽略。

如果你**不需要**顶端输入，省略第三个参数，种群的行为将与标准的单输入神经元层相同：

```python
spikes, states = brain(x_soma)   # 每个组的 x_apical 均为 None
```

## 状态命名约定

由于子种群被注册为命名子模块（`group_0`、`group_1` 等），它们的状态将以点分前缀出现：

| 状态键 | 含义 | 形状 |
|-----------|---------|-------|
| `neuron.group_0.v` | GLIF3 膜电压 | `(T, batch, 80)` |
| `neuron.group_1.v` | TC 体细胞电压 | `(T, batch, 20)` |
| `neuron.group_1.i_a` | TC 顶端电流 | `(T, batch, 20)` |
| `synapse.psc` | 突触后电流 | `(T, batch, 100)` |

你可以对字典进行展开以便于访问：

```python
from btorch.utils.dict_utils import unflatten_dict
nested = unflatten_dict(states, dot=True)
nested["neuron"]["group_1"]["i_a"]   # (T, batch, 20)
```

## 命名分组

你可以为组指定显式名称，而不是自动命名：

```python
mixed = MixedNeuronPopulation({
    "fs": (80, GLIF3(n_neuron=80)),
    "pyr": (20, TwoCompartmentGLIF(n_neuron=20)),
}, step_mode="s")
```

状态键将变为 `neuron.fs.v`、`neuron.pyr.i_a` 等。

## 兼容性说明

- `torch.compile` — `MixedNeuronPopulation` 在子模块上使用 Python 循环，因此编译可能会在循环处产生图断点。如果这成为瓶颈，可以考虑将子种群融合到一个自定义模块中。
- 梯度检查点 — 通过 `RecurrentNNAbstract` 透明地工作，因为检查点区域是外部的 `multi_step_forward`，而不是单个神经元子模块。
- CPU 卸载 — 同样透明地工作；块输出在所有组完成完整的前向传递后被卸载。

## 另请参阅

- [`RecurrentNN`][btorch.models.rnn.RecurrentNN] — 标准单输入递归包装器。
- [`ApicalRecurrentNN`][btorch.models.rnn.ApicalRecurrentNN] — 顶端输入变体。
- [`MixedNeuronPopulation`][btorch.models.neurons.mixed.MixedNeuronPopulation] — 异质种群容器。
- [`TwoCompartmentGLIF`][btorch.models.neurons.two_compartment.TwoCompartmentGLIF] — 体细胞-顶端神经元。
