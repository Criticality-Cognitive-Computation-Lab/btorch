# Triton 事件驱动稀疏后端

本教程介绍 btorch 新增的 Triton 事件驱动稀疏实现：如何选择后端、如何在
多步时间循环中复用准备好的工作区，以及如何解释一次与 cudagraph 回归实验
同规模的基准结果。

## 适用范围

Triton 后端面向 CUDA 上的稀疏脉冲网络。它根据输入中的非零脉冲筛选活跃
源任务，然后对对应边执行稀疏聚合，因此输入活动率较低时更容易获得收益。
当前实现有以下明确边界：

- 需要可导入的 Triton 和 CUDA 设备。
- 输入与稀疏权重当前必须是 CUDA 上的 `float32` 张量；公开的 `SparseConn`
  路径会在内部整理输入的连续布局，底层 Triton 算子则要求传入连续张量。
- CPU、`float16` 和 `bfloat16` 不支持 Triton 路径；代码会直接报错，避免
  静默使用错误的内核。
- 连接拓扑在模块构造时确定；权重可以继续作为可学习参数更新。

Triton 后端**支持批量和一阶反向传播**。`SparseConn` 接受任意前导维度，
例如 `(batch, n_source)`、`(time, batch, n_source)` 或其他
`(..., n_source)` 形状；内部只将前导维度展平为批量桶。自定义 autograd
路径会计算输入梯度和稀疏权重梯度。因此不应在使用批量输入或训练模式时
额外绕过 Triton 后端；不支持的是设备或 dtype 边界，而不是 batch/backward。

## 安装与最小用法

在 CUDA PyTorch 环境中安装项目及其依赖后，构造一个 scipy 稀疏连接并明确
选择 Triton：

```python
import numpy as np
import scipy.sparse
import torch

from btorch.models.linear import SparseConn

n_source = 1024
n_destination = 1024
rng = np.random.default_rng(0)
source = np.repeat(np.arange(n_source), 16)
destination = rng.integers(0, n_destination, size=source.size)
weight = rng.normal(0.0, 0.1, size=source.size).astype(np.float32)
connectivity = scipy.sparse.coo_array(
    (weight, (source, destination)),
    shape=(n_source, n_destination),
)

connection = SparseConn(
    connectivity,
    enforce_dale=False,
    sparse_backend="triton",
    device="cuda",
    dtype=torch.float32,
)

spikes = (torch.rand(8, 4, n_source, device="cuda") < 0.01).float()
currents = connection(spikes)
assert currents.shape == (8, 4, n_destination)
```

如果系统未安装 Triton，或输入不在 CUDA/`float32` 条件内，后端会抛出明确
异常。需要回退时省略 `sparse_backend="triton"`，让 btorch 使用可用的默认
稀疏后端。

## 在时间循环中复用工作区

直接在 Python 循环中多次调用 `SparseConn` 时，建议在整个循环外使用
`prepare_sparse_modules`。它会预打包重复使用的边权重，并按照批量大小分配
任务队列：

```python
from btorch.models.functional import prepare_sparse_modules

batch_size = 4
timesteps = 100
spikes = (
    torch.rand(timesteps, batch_size, n_source, device="cuda") < 0.01
).float()

with torch.inference_mode(), prepare_sparse_modules(
    connection, batch_size=batch_size
):
    currents = torch.stack([connection(spikes[t]) for t in range(timesteps)])
```

`RecurrentNN` 和 `make_rnn` 的多步路径会自动准备其内部的稀疏连接。直接
调用这些包装器时不需要再套一层；手动 Python 时间循环才需要显式使用该
上下文管理器。

训练路径可以继续使用 autograd。例如，批量输入的单步反向传播可写成：

```python
x = torch.randn(4, n_source, device="cuda", requires_grad=True)
target = torch.randn(4, n_destination, device="cuda")
prediction = connection(x)
loss = torch.nn.functional.mse_loss(prediction, target)
loss.backward()

assert x.grad is not None
```

如果需要在一个多步训练作用域内显式准备模块，应让使用该前向图的计算和
反向传播保持在合适的生命周期内；权重更新后下一次作用域会重新打包必要的
权重。推理场景则可以像上面的示例一样使用 `torch.inference_mode()`。

## 后端配置

可以通过全局配置模板设置 Triton 优化，也可以在单个模块上覆盖：

```python
from btorch import config

triton = config.sparse.backend("triton")
triton.reorder = True
triton.block = True
triton.hash = True

connection = SparseConn(
    connectivity,
    sparse_backend="triton",
    sparse_config={"hash": False},
)
```

三个开关彼此独立：`reorder` 控制源任务重排，`block` 控制边分块，`hash`
控制局部哈希聚合。关闭全部开关会选择按源分桶的 direct-atomic 基线。
详细字段及其默认值见[配置指南](../guides/configuration.md)。

## 规模匹配基准

### 实验条件与 provenance

下面的结果来自 `benchmarks/sparse_rnn/benchmark_compile_triton_proxy.py`。
它使用 RTX 5090、PyTorch `2.11.0+cu129`，并对原生稀疏路径和 Triton
路径使用相同的网络规模、输入活动历史和时间循环：

- 32768 个神经元；每个投影每个神经元采样 512 条边。
- 每个投影合并重复项后为 16,646,991 条边；共有两个循环投影。
- `T=1000`，`unroll=8`，固定活动率 `0.2999054%`。
- `float32`，`torch.compile(mode="default")`，不使用 scan。
- 模型内 cudagraph 和 `torch._inductor.config.triton.cudagraphs` 均关闭。

这是用于隔离后端和编译影响的**规模匹配 proxy**，不是原始应用拓扑的复现：
原始应用拓扑在该实验中不可用。因此这些数字不应表述为原应用端到端结果。

### 测量结果

| 后端与执行方式 | 时间 | 相对同后端 eager |
| --- | ---: | ---: |
| Native eager | 881.1417 ms | 1.00000x |
| Native compiled | 903.7372 ms | 0.97500x |
| Triton eager | 199.6218 ms | 1.00000x |
| Triton compiled | 120.3175 ms | 1.65913x |

跨后端比较如下：

| 比较 | 加速比 |
| --- | ---: |
| Triton eager / Native eager | 4.414x |
| Triton compiled / Native eager | 7.323x |
| Triton compiled / Native compiled | 7.511x |

首次编译时间为 Native `2.3538 s`、Triton `4.4449 s`。在相同输入上，
两个后端的最大 eager-versus-compiled 绝对误差均为 `1.49e-8`。

这些时间应理解为该脚本中稳态执行路径的比较；首次编译和构建开销单独报告，
没有被当作稳态加速的一部分。实际收益仍取决于活动率、拓扑、时间步数、权重
更新频率和 GPU。尤其是当活动率较高或循环很短时，事件筛选和 kernel 调度
开销可能抵消稀疏计算收益。

### 复现命令

在仓库根目录运行：

```bash
micromamba run -n ml-py312 \
  python benchmarks/sparse_rnn/benchmark_compile_triton_proxy.py \
  --backend native \
  --output benchmarks/sparse_rnn/results/rtx5090_compile_proxy_native.json

micromamba run -n ml-py312 \
  python benchmarks/sparse_rnn/benchmark_compile_triton_proxy.py \
  --backend triton \
  --output benchmarks/sparse_rnn/results/rtx5090_compile_proxy_triton.json
```

运行前确认当前 Python 环境能导入 CUDA PyTorch、Triton、SciPy 和 btorch。
若要比较编译开关，请保持网络规模、随机种子、活动历史和 cudagraph 设置
不变，只修改明确标注的变量。

## 常见问题

### 批量输入会失败吗？

不会。`SparseConn` 支持 `(..., n_source)`，批量维度会被展平后送入同一
个 Triton 内核。使用 `prepare_sparse_modules` 时，应传入实际 batch size，
使工作区队列容量正确分配。

### 可以训练吗？

可以。当前实现包含一阶 autograd backward，支持输入和稀疏权重的梯度。仍需
遵守 CUDA 连续 `float32` 限制；不满足该限制时应在构造模型时选择其他后端，
而不是等待运行时隐式转换。

### 为什么 torch.compile 可能没有收益？

`torch.compile` 只改变编译调度，不会自动改变稀疏算法或输入活动率。上面的
proxy 中 Native compiled 为 `903.7372 ms`，反而略慢于 Native eager 的
`881.1417 ms`；Triton compiled 则为 `120.3175 ms`，因为它同时使用事件驱动
稀疏内核和编译后的调用路径。比较时应把后端、活动率、首次编译、cudagraph
设置和时间循环长度分别记录。
