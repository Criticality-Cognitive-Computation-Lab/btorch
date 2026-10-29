# 稀疏连接

本页尚未翻译。完整内容见英文版指南 [Sparse Connectivity](https://criticality-cognitive-computation-lab.github.io/btorch/guides/sparse_connectivity/)。

要点：

- `btorch.models.linear` 中的 `SparseConn`、`SparseConstrainedConn` 及 `sparse_backend=` 参数已移除，没有兼容层。
- 数值稀疏数组 API 位于 `btorch.sparse`（`Sparse`、`COO`、`CSR`、`CSC`）。矩阵乘法遵循标准线性代数方向，转换从不转置、也不稠密化。
- 模型中使用 `btorch.models.connection` 的 `SparseConnection` 与 `Synapse`。
- 矩阵方向用两个字符串表示：`"pre_post"`（形状 `(n_pre, n_post)`，行为源神经元，`from_adjacency` 与 `FromSparse` 的默认值）和 `"post_pre"`（形状 `(n_post, n_pre)` 的算子，`SparseConnection(A)` 接受的形式）。对于方阵，方向写反不会报错，请用 `conn.find_edges(pre=i, post=j)` 核对一条已知的边。
- `conn.to_sparse()` 默认返回构建连接时所用的方向；`conn.pre` / `conn.post` 给出每条边的源与目标；`conn.set_edges_(slots, *, pre, post)` 的索引参数仅限关键字。
- 传入 `Synapse(weight=...)` 的 `Weight` 模块就是 `conn.weight`；优化器请使用 `conn.parameters()` 或 `conn.weight.value`。
- `torch.compile(mode="reduce-overhead")` 不得与重连（rewiring）或加载不同连接模式的检查点同时使用。

最小示例：

```python
import scipy.sparse
import torch
from btorch.models.connection import SparseConnection, Synapse

# 行为源神经元、列为目标神经元的连接矩阵（"pre_post"，from_adjacency 的默认方向）
weights = scipy.sparse.random(100, 100, density=0.1, format="csr", random_state=0)

conn = SparseConnection.from_adjacency(weights, Synapse(dale=True))
current = conn(torch.rand(32, 100))  # [..., n_pre] -> [..., n_post]
```

迁移对照表、方向一览表、权重参数化（`ConstrainedWeight`）、受体与延迟、`Projection` 与连接规则、E/I 循环网络示例、网络批、`torch.compile` 与 CUDA graph、检查点以及结构可塑性的说明均见英文版指南。

## Push 的复现性限制

CUDA `method="push"` 使用浮点原子累加；固定随机种子或 CUDA Graph 重放不保证
循环网络的状态、脉冲或梯度逐位一致。需要可复现的稀疏累加时选择 `method="pull"`；
不同后端及数学精度仍可能影响完整模型的数值一致性。

一次 RTX 5090 实测使用 btorch Izhikevich + ExponentialPSC：60,000 个神经元、
33,072,348 条合成边、batch=1、dt=1 ms，平均放电率约 5 Hz。两组 FP32 push
重复对照在第 100、500、1000 步观测到的最大电压差分别为 **0.00257、0.04715、
0.67584 mV**。这是指定时刻、两组对照中的最大值，不是累积最大值或通用误差上界。
一组从零起算 step 904 首次出现脉冲差异，1000 步共 12 个不同 spike bins
（总计 6000 万个）；另一组脉冲相同但物理状态不同。第一组第 1000 步全神经元
电压 MAE 为 0.000103 mV。窗口内因两侧 reset 时刻不同，瞬时最大电压差可接近
95 mV；长轨迹即使平均率相近，逐脉冲分歧仍可能很大。

固定脉冲输入时，重复传播最大差异为 1.91e-6 pA。FP64 连接累加和另一组可精确
求和的 FP32 权重对照均消除了该实验的全部 1000 步差异，支持舍入放大的解释，
但不能证明所有路径没有 race，也不意味着应该偷偷改变模型权重或 dtype。
完整实验条件及限制见英文指南。


## TileLang 稀疏后端

`tilelang` 与 `triton` 使用相同的分段 CSR pull、边梯度归约、块内 spike list
与持久 fanout、direct/hash 任务队列算法。TileLang 路径通过公开 JIT 内核接口
启动，前向与反向均不依赖 Triton 内核；自动选择仍优先使用可用的 Triton。

```python
from btorch.models import KernelOptions

options = KernelOptions(
    neuron="tilelang",
    connection="tilelang",
    connection_method="push",
)
```

将 `options` 传给 `RecurrentNN` 的 `backend` 参数。独立连接可用
`btorch.sparse.runtime.use_backend("tilelang")`，显式 push 则配合
`use_method("push")`。后端选择与传播方法独立；FP16/BF16/FP32/FP64、混合 dtype、
自动求导与 CUDA Graph 的契约不变。push 仍使用非确定性的浮点原子累加。

没有延迟或外部突触输入时，TileLang 神经元/PSC 模板在同一次启动中直接生成
块内脉冲列表并清零受体累加区；内部每步只需融合的神经元/PSC 生产者和 fanout
两个内核。独立连接、延迟源和外部突触输入仍由 TileLang packing 后再执行 fanout。
需要确定性的稀疏归约时选择 `connection_method="pull"`。
