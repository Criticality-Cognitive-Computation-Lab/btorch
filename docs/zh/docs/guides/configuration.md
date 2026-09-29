# OmegaConf 配置指南

btorch 在 [`btorch.utils.conf`](../api/utils.md) 中提供了以数据类优先的配置工具。本指南介绍如何构建配置、从文件和 CLI 参数加载配置，以及如何从启动脚本向工作进程转发选项。

## 为什么选择数据类优先配置

- **类型安全**：配置字段具有类型，并在加载时进行检查。
- **单一事实来源**：默认值存在于 Python 数据类中，而非 YAML 文件中。
- **可组合**：嵌套数据类可以清晰地区分通用设置和任务特定设置。
- **CLI 友好**：OmegaConf 自动解析 `key=value` 覆盖。

## 快速开始

最简单的模式是加载一个数据类并将其与 CLI 覆盖合并：

```python
from dataclasses import dataclass
from btorch.utils.conf import load_config

@dataclass
class Config:
    lr: float = 1e-3
    epochs: int = 100

cfg = load_config(Config)
print(cfg.lr)  # 被命令行上的 `lr=0.01` 覆盖
```

运行方式：

```bash
python train.py lr=0.01 epochs=200
```

## 组合模式

实际项目通常具有嵌套配置。以下是 btorch 自身示例中使用的模式：

```python
from dataclasses import dataclass, field

@dataclass
class CommonConf:
    output_path: str = "outputs"
    seed: int = 42

@dataclass
class SolverConf:
    lr: float = 1e-3
    max_iter: int = 1000

@dataclass
class ArgConf:
    common: CommonConf = field(default_factory=CommonConf)
    solver: SolverConf = field(default_factory=SolverConf)
```

CLI 覆盖使用点号表示法：

```bash
python train.py common.seed=123 solver.lr=0.005
```

## 从文件加载

如果你传递 `config_path=path/to/config.yaml`，`load_config` 会先在默认值之上合并 YAML，然后再应用 CLI 覆盖：

```bash
python train.py config_path=base.yaml solver.lr=0.005
```

优先级顺序：**数据类默认值 → 配置文件 → CLI 参数**。

## 使用 `_type_` 进行变体选择

OmegaConf 支持数据类联合类型。你可以在不添加手动 `mode: str` 字段的情况下切换变体：

```python
from dataclasses import dataclass, field

@dataclass
class AdamConf:
    lr: float = 1e-3

@dataclass
class SGDConf:
    lr: float = 1e-2
    momentum: float = 0.9

@dataclass
class TrainConf:
    optimizer: AdamConf | SGDConf = field(default_factory=AdamConf)
```

在运行时切换：

```bash
python train.py optimizer="{_type_: SGDConf, lr: 0.01, momentum: 0.95}"
```

## 启动器 → 工作进程选项转发

运行参数扫描或分布式任务时，启动脚本通常需要向各个工作进程转发 CLI 覆盖。`to_dotlist` 将 OmegaConf 容器转换为 `key=value` 字符串列表：

```python
from btorch.utils.conf import load_config, to_dotlist

@dataclass
class BatchConf:
    single: ArgConf = field(default_factory=ArgConf)
    max_workers: int = 4

cfg, cli_cfg = load_config(BatchConf, return_cli=True)

# 转发除工作进程 ID 外的所有内容
dotlist = to_dotlist(
    cli_cfg.single,
    use_equal=True,
    exclude={"common.id"},
)

# 构建工作进程命令
cmd = ["python", "worker.py"] + dotlist + [f"common.id={worker_id}"]
```

## 与训练脚本集成

Fashion-MNIST 示例（`examples/fmnist_lightning.py`）使用嵌套的 `NetworkConfig` 数据类来管理神经元和突触超参数，并通过 `load_config` 加载。你可以在自己的模型中使用相同的模式：

```python
@dataclass
class NetworkConfig:
    n_neuron: int = 256
    dt: float = 1.0

cfg = load_config(NetworkConfig)
with environ.context(dt=cfg.dt):
    # ... 训练循环
```

## 工具参考

| 函数 | 用途 |
|----------|---------|
| [`load_config`](../api/utils.md) | 加载数据类 + 文件 + CLI |
| [`to_dotlist`](../api/utils.md) | 将配置展平为 CLI 字符串 |
| [`diff_conf`](../api/utils.md) | 计算两个配置之间的结构化差异 |
| [`get_dotkey`](../api/utils.md) | 通过点路径读取嵌套字段 |
| [`set_dotkey`](../api/utils.md) | 通过点路径写入嵌套字段 |

## 稀疏后端配置

稀疏连接可以在模块级通过 `sparse_backend` 选择后端，也可以先选择全局
后端配置模板。Triton 后端使用事件驱动的稀疏矩阵乘法，适合稀疏脉冲输入：

```python
from btorch import config
from btorch.models.linear import SparseConn

triton_config = config.sparse.backend("triton")
triton_config.reorder = True
triton_config.block = True
triton_config.hash = True

connection = SparseConn(connectivity, sparse_backend="triton")
```

Triton 配置默认开启三项独立优化：源神经元重排（`reorder`）、边分块
（`block`）和局部哈希聚合（`hash`）。也可以针对单个模块覆盖模板，而不影响
其他模块：

```python
connection = SparseConn(
    connectivity,
    sparse_backend="triton",
    sparse_config={"reorder": False, "block": True, "hash": False},
)
```

配置在构造模块时复制。之后修改全局模板只会影响新建模块，不会改变已经创建
的连接。后端需要安装 Triton，并且当前只接受 CUDA 上的 `float32` 输入和
权重；公开的 `SparseConn` 路径会在内部整理连续布局。CPU、`float16` 和
`bfloat16` 应使用其他稀疏后端。

对于直接在 Python 时间循环中反复调用 `SparseConn` 的代码，应使用
[`prepare_sparse_modules`](../api/models.md) 在循环外预打包权重和工作区：

```python
import torch

from btorch.models.functional import prepare_sparse_modules

with torch.inference_mode(), prepare_sparse_modules(
    connection, batch_size=batch_size
):
    currents = torch.stack([connection(spikes[t]) for t in range(timesteps)])
```

`RecurrentNN` 和 `make_rnn` 会在多步执行期间自动完成同样的准备，因此通常
不需要手动包裹。输入支持任意前导批量维度，例如 `(batch, neurons)` 或
`(time, batch, neurons)`；实现会将前导维度展平为批量桶后再执行稀疏乘法。
Triton 后端也提供一阶 autograd backward，可对输入和可学习稀疏权重求梯度。
