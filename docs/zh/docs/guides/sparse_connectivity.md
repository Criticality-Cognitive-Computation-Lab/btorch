# 稀疏连接

本页尚未翻译。完整内容见英文版指南 [Sparse Connectivity](https://criticality-cognitive-computation-lab.github.io/btorch/guides/sparse_connectivity/)。

要点：

- `btorch.models.linear` 中的 `SparseConn`、`SparseConstrainedConn` 及 `sparse_backend=` 参数已移除，没有兼容层。
- 数值稀疏数组 API 位于 `btorch.sparse`（`Sparse`、`COO`、`CSR`、`CSC`）。矩阵乘法遵循标准线性代数方向，转换从不转置、也不稠密化。
- 模型中使用 `btorch.models.connection` 的 `SparseConnection` 与 `Synapse`。

最小示例：

```python
import scipy.sparse
import torch
from btorch.models.connection import SparseConnection, Synapse

# 行为源神经元、列为目标神经元的连接矩阵
weights = scipy.sparse.random(100, 100, density=0.1, format="csr", random_state=0)

conn = SparseConnection.from_adjacency(weights, Synapse(dale=True))
current = conn(torch.rand(32, 100))  # [..., n_pre] -> [..., n_post]
```

迁移对照表、权重参数化（`ConstrainedWeight`）、受体与延迟、网络批、`torch.compile` 以及检查点的说明均见英文版指南。
