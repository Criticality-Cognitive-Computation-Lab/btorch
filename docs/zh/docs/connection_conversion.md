# 连接层转换

本页面记录了连接层的转换工具：

- `DenseConn`
- `SparseConn`
- `SparseConstrainedConn`

实现位于 `btorch.models.connection_conversion`。

## 概览

支持两种布局：

1. `base`：连接形状为 `(N_pre, N_post)`
2. `heter`：连接形状为 `(N_pre, N_post * n_receptor)`

支持的受体模式：

1. `neuron`
2. `connection`

默认情况下，`base -> heter` 使用不拆分（no-split）语义：每个非零基础边恰好映射到一个受体通道。

稠密转换直接在稠密张量上实现。

## API

### `convert_connection_layer`

```python
from btorch.models.connection_conversion import convert_connection_layer
```

在 `base` 和 `heter` 布局之间转换活跃的连接层实例。

常用输入：

- `target_layout`：`"base"` 或 `"heter"`
- `receptor_type_mode`：可选；仅在从 `neurons` 推断 base->heter 分配时需要
- `receptor_type_index`：包含 `receptor_index` 的受体通道表

对于 `base -> heter`：

- 默认不拆分：
    - 提供 `edge_receptor_assignment`，或者
    - 在 `neuron` 模式下，提供 `neurons` + `receptor_type_col`
- 可选拆分模式：
    - 设置 `allow_weight_split=True`
    - 提供带有 `weight_coeff` 的 `edge_receptor_weight`

### `convert_connection_layer_from_checkpoint`

```python
from btorch.models.connection_conversion import (
    convert_connection_layer_from_checkpoint,
)
```

从序列化权重（`state_dict`）以及调用者提供的拓扑结构进行转换。

拓扑结构有意由更高级别的 API 提供：

- 稀疏源：需要 `conn`
- 受限稀疏源：需要 `constraint`
- 受限稀疏源：
    - 如果提供了 `conn`，则它作为初始边权重的权威来源
    - 如果省略 `conn`，则 `state_dict` 必须同时提供 `initial_weight` 和 `indices`
- `source_class` 以类对象（`DenseConn`、`SparseConn` 或 `SparseConstrainedConn`）的形式传递

该函数重建源层，加载 `state_dict`，然后应用与 `convert_connection_layer` 相同的转换逻辑。

## 最小示例

```python
import pandas as pd
from btorch.models.connection_conversion import convert_connection_layer
from btorch.models.linear import SparseConn

layer = SparseConn(conn=conn_base, enforce_dale=False)
layer_heter = convert_connection_layer(
    layer,
    target_layout="heter",
    receptor_type_mode="connection",
    receptor_type_index=pd.DataFrame(
        [(0, "E"), (1, "I")],
        columns=["receptor_index", "receptor_type"],
    ),
    edge_receptor_assignment=edge_receptor_assignment,
)
```

## 注意事项

- 在 `allow_weight_split=False` 时传递拆分表会引发 `ValueError`。
- 对于受限转换，`group_policy` 控制受体扩展后的组是独立的（`"independent"`）还是共享的（`"shared"`）。
- 有意不支持跨家族的输出覆盖（例如 `DenseConn` -> 稀疏，或稀疏 -> `DenseConn`）。