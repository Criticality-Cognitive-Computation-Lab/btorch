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
