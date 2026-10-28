# 示例库

`examples/` 目录包含可运行的脚本，演示了 btorch 的核心模式。本页面对每个示例进行了注解，并链接到相关的测试和 API 文档。

## 两个 RSNN 示例：训练、推理与 SONATA

| 脚本 | 模型 | 默认运行 |
|---|---|---|
| `rsnn.py` | GLIF3 + 异质 AlphaPSC，延迟与 Dale 约束 | 256 个神经元，CPU，推理和训练，目标 7.5 Hz |
| `rsnn_izhikevich.py` | Izhikevich + 异质双指数 PSC | 60,000 个神经元、33,072,348 条边，CUDA/Triton，CUDA Graph 推理，目标 5 Hz |

有 CUDA GPU 时，在仓库根目录直接运行：

```bash
python -m examples.rsnn_izhikevich
```

无需额外设置。脚本复用 btorch 的 `FixedIndegree` 采样器生成精确边数，校准持续异质
输入，运行 5000 步、每步 1 ms，并输出相同记录策略下 push eager、pull CUDA Graph、
push CUDA Graph 的计时。首个计时调用包含首块 JIT 和 Graph 设置；目标发放率的
校准不计入模拟总耗时。构图、校准、CPU
校验/传输以及写入记录文件单独计时，保留磁盘缓存。公共 replay 的输入、状态和返回
结果所有权复制计入 replay 时间。基准固定网络、共享校准和策略顺序，因此只描述该
合成电路，不构成进程冷启动的单独结论，也不保证其他硬件达到相同结果。

`--chunk-size 200` 限制 CUDA Graph 的预热和捕获长度。默认
`--chunk-mode split` 使用 `torch.split`，413 步分为 200、200、13；
`--chunk-mode chunk` 使用 `torch.chunk`，分块数为 `ceil(T / chunk_size)`，
同一输入分为 138、138、137 步。重复的块复用 Graph，较短的最后一块直接
eager 执行。在 Python API 中，`chunk_size=None` 关闭外层分块。

默认保存 `fig/rsnn_izhikevich/recordings.pt`（本配置约 6.7 GiB），旁边的 JSON
保存配置、实测率和计时。六个张量形状均为 `[time, batch, neuron]`：

| 键 | 含义 | 单位 |
|---|---|---|
| `v` | reset 后膜电位 | mV |
| `u` | Izhikevich 恢复状态 | pA |
| `spike` | 发射的二值脉冲 | 无量纲 |
| `psc` | 本步突触更新后的总带符号 PSC | pA |
| `epsc` | receptor 0 的兴奋性电流 | pA |
| `ipsc` | receptor 1 的抑制性电流 | pA |

`psc = epsc + ipsc`，IPSC 为负。遵循 `RecurrentNN` 的前一 PSC 调度，step t
记录的电流驱动 step t+1 神经元。EPSC/IPSC 是同一份传回 CPU 的 receptor 历史的
视图，不额外启动逐步 kernel；三种执行策略都开启全部记录。完整历史及独立返回的
Graph 输出需要较多显存，默认配置在 32 GiB RTX 5090 验证；小显存设备请显式缩小规模。

```python
import torch

traces = torch.load("fig/rsnn_izhikevich/recordings.pt", weights_only=True, mmap=True)
v = traces["v"][:, 0]  # CPU 上的 [time, neuron]
```

`--no-compare` 跳过计时对比，`--no-cudagraph` 将最终推理改为 eager，
`--recordings PATH` 指定保存位置。对比过程会独立测试三种策略，与最终推理选择分开。
**FP32 push 不保证逐位复现**，CUDA Graph 也不会消除这一限制；相近平均率不能代替
脉冲、状态或梯度一致性。见[1000 步量化误差](guides/sparse_connectivity.md#push)。

小规模训练：

```bash
python -m examples.rsnn
python -m examples.rsnn_izhikevich --neurons 256 --density .05 --steps 256 --mode train --compile
```

`--mode both` 先推理再训练；训练阶段关闭显式推理 capture，`--compile` 使用 Torch
的 `reduce-overhead` 训练图机制。训练保留现有 functional BPTT、checkpoint 和 CPU
offload 行为；chunk 化 CUDA Graph 重放只用于推理。每个 epoch 重置物理 memories，
优化 `--target-rate-hz` 和电压惩罚。
GLIF 在 optimizer 更新后通过 `constrain_net(network)` 保持 Dale 符号，Izhikevich
权重不受符号约束。`--edges E` 可覆盖 density，指定精确总边数。

模型组合直接放在各脚本的 `build_model`；公共 CLI/执行逻辑位于 `rsnn_common.py`。
通用校准复用 `btorch.analysis.compute_input_scale`，恢复 memories、记录及 Graph 配置，
不改变模型参数。校准窗口最多 512 步，完整推理再检查实测率；不保证单神经元、稳态或
训练参数更新后的放电率。持续推理只初始化一次并保留 memories；`clear_cudagraph()`
清除缓存图但不重置状态，改变 native 记录策略会自动使旧 capture 失效。

神经元/PSC 与连接后端独立选择。`--method push` 显式请求 spike-list SpMSpV，
`pull` 使用目标驱动遍历；auto 可在展开算子维度均小于 4000 时内部使用 dense matmul，
外部仍保持稀疏接口。

两个示例都有 SONATA 导入/导出：

```bash
pip install 'btorch[io]' libsonata
python -m examples.rsnn --export-sonata /tmp/glif-circuit
python -m examples.rsnn --import-sonata /tmp/glif-circuit/circuit_config.json
python -m examples.rsnn_izhikevich --neurons 256 --density .05 --steps 256 --no-compare --export-sonata /tmp/izh-circuit
python -m examples.rsnn_izhikevich --steps 256 --no-compare --import-sonata /tmp/izh-circuit/circuit_config.json
```

导入恢复网络、梯度设置和 timestep；规模、边数、density、seed、`--dt` 只配置新网络。
执行后端由调用者选择，校准只缩放输入，不改变导入的权重。导出保存更新后的模型参数，
不保存 optimizer 或运行状态。见 [SONATA 映射](guides/sonata_io.md)。

## `rsnn_brain.py` — 脑-环境交互

扩展了基础 RSNN，增加了脑-环境交互网络。演示了感觉运动任务中的 `NeuronEmbedMapLayer` 和 `DetectionWindow`。

**关键模式：**
- 神经元空间之间的嵌入映射
- 基于脉冲的事件检测窗口
- `RecurrentNN` 之外的多模块组合

## `fmnist.py` — Fashion-MNIST 训练（纯 PyTorch）

使用基于 GLIF3 的 RSNN 进行 Fashion-MNIST 分类的完整训练流程，包含稀疏循环连接和电压正则化。

**关键模式：**
- 用于稀疏循环权重的 `SparseConnection`
- 具有异构时间常数的 `AlphaPSCBilleh` 突触
- 用于膜电压正则化的 `VoltageRegularizer`
- 每批次手动调用 `reset_net` 的训练循环
- 与 `state_dict()` 一起保存 `memories_rv`

**另请参阅：**
- [教程 2：训练 SNN](tutorials/training.md)
- API: [`btorch.models.regularizer`](api/models.md)

## `fmnist_lightning.py` — PyTorch Lightning 集成

将相同的 Fashion-MNIST 模型重构为 PyTorch Lightning `LightningModule`。展示了 btorch 状态管理如何融入 Lightning 训练生命周期。

**关键模式：**
- 带 `reset_net` 的 Lightning `training_step`
- 用于神经元参数字典缩放的 `scale_state_`
- 带状态记录的验证循环

## `delayed_synapse_demo.py` — 突触延迟

演示使用 `Delay` 和 `DelayedSynapse` 实现异构突触延迟。

**关键模式：**
- 作为滚动脉冲缓冲区的 `Delay`
- 用于历史记录 + 线性变换的 `DelayedSynapse`
- 用于延迟感知连接矩阵的 `expand_conn_for_delays`

**另请参阅：**
- API: [`btorch.models.history`](api/models.md)
- 测试: [`tests/connectome/test_delay_expansion.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/tests/connectome/test_delay_expansion.py)

## 异构 E/I 完整工作流

[异构多受体 E/I RSNN 教程](tutorials/heterogeneous_ei_rsnn.md)展示语义受体边、受体特异 PSC、Dale 约束、训练循环和通道检查。

## 测试即示例

许多 `tests/` 文件包含简洁、经过验证的使用模式，非常适合作为文档：

- `tests/models/test_mem_load_save.py` — 检查点和状态恢复
- `tests/models/test_compile.py` — 带 `dt` 上下文的 `torch.compile`
- `tests/utils/test_conf.py` — OmegaConf 模式
- `tests/visualisation/*.py` — 几乎每个绘图函数

**提示：** 将测试代码改编为文档时，将 `fig_path` / `save_fig` 调用替换为标准 matplotlib 模式（例如 `plt.show()` 或返回图形对象）。
