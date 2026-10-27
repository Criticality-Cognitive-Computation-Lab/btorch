# 示例库

`examples/` 目录包含可运行的脚本，演示了 btorch 的核心模式。本页面对每个示例进行了注解，并链接到相关的测试和 API 文档。

## 两个 RSNN 示例：推理、训练和 SONATA

| 脚本 | 神经元 / PSC | 连接 |
|---|---|---|
| `rsnn.py` | GLIF3 + AlphaPSC | 两个异质 receptor 通道，0–3 个时间步的传播延迟 |
| `rsnn_izhikevich.py` | Izhikevich + DualExponentialPSC | 两个不同上升/衰减时间的 receptor 通道，无额外传播延迟 |

在仓库根目录运行任一 CPU 示例：

```bash
python -m examples.rsnn
python -m examples.rsnn_izhikevich
```

默认使用 256 个神经元，先推理，再训练三个 epoch。每个脚本的 `build_model`
直接展示神经元、异质 PSC、状态记录和 `RecurrentNN` 的组合。
`rsnn_common.py` 共享稀疏连接、状态初始化和执行循环；
`rsnn_sonata.py` 提供显式的网络映射。

输入和脉冲的形状均为 `[time, batch, neuron]`，示例的 batch size 为 1。
时间单位为 ms，电压为 mV，输入电流为 pA。receptor 0 初始为兴奋性，
receptor 1 初始为抑制性。GLIF 示例在每次 optimizer 更新后调用
`constrain_net(network)`，通过 Dale 约束保持每个源神经元的 E/I 符号；
Izhikevich 示例的可训练权重没有符号约束。PSC 按
`post * 2 + receptor` 展平，每个神经元使用同一对不同的 receptor 时间常数，
神经元自身参数也随神经元变化。

用 `--mode inference` 或 `--mode train` 可分别运行。每个训练 epoch
视为独立试验并重新初始化物理状态，目标是合成的 20 Hz 放电率及电压惩罚，
用于演示神经元参数和循环权重的梯度，不是科学模型拟合任务。
连续推理时应只初始化一次，在连续输入块之间保留 memories。

原生 CUDA 执行可独立选择神经元 / PSC 和连接后端：

```bash
python -m examples.rsnn --device cuda --neuron-backend triton --sparse-backend triton --compile
python -m examples.rsnn_izhikevich --device cuda --neuron-backend triton --sparse-backend triton --method push --compile
```

`--method push` 显式使用 spike-list 传播，`pull` 使用相反的遍历方向。
auto 模式在少于 4,000 个神经元时可使用稠密乘法，同时保持稀疏接口。
`--compile` 支持训练；显式 `--cudagraph` 要求 `--device cuda --mode inference`。

安装可选 IO 依赖后，可导出和导入两个网络：

```bash
pip install 'btorch[io]' libsonata
python -m examples.rsnn --export-sonata /tmp/glif-circuit
python -m examples.rsnn --import-sonata /tmp/glif-circuit/circuit_config.json
python -m examples.rsnn_izhikevich --export-sonata /tmp/izh-circuit
python -m examples.rsnn_izhikevich --import-sonata /tmp/izh-circuit/circuit_config.json
```

导入恢复网络、梯度设置及时间步；`--neurons`、`--density`、`--seed` 和 `--dt`
仅配置新构建的网络。执行后端仍由调用者选择。训练后的导出包含更新的参数，
不保存 optimizer 或运行中的 memories。参见 [SONATA 映射说明](guides/sonata_io.md)。

较大的 Izhikevich 配置为 `--neurons 50000 --density 0.05`。
构建使用稀疏边数组，但该规模的完整状态记录和训练仍需要大量内存；
默认的小规模配置用于学习。

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
