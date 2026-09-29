# 示例库

`examples/` 目录包含生产级质量的脚本，演示了 btorch 的核心模式。本页面对每个示例进行了注解，并链接到相关的测试和 API 文档。

## `rsnn.py` — 最小化 RSNN

一个自包含的循环脉冲神经网络演示，使用 `GLIF3`、`AlphaPSC` 和 `RecurrentNN`。它包含模拟循环（带光栅图生成）和虚拟训练循环。

**关键模式：**
- `functional.init_net_state` 和 `functional.reset_net`
- `environ.context(dt=1.0)`
- 用于状态记录的 `update_state_names`
- `btorch.visualisation` 中的 `plot_raster` 和 `plot_neuron_traces`

**另请参阅：**
- [教程 1：构建 RSNN](tutorials/building_rsnn.md)
- [教程 2：训练 SNN](tutorials/training.md)

## `rsnn_brain.py` — 脑-环境交互

扩展了基础 RSNN，增加了脑-环境交互网络。演示了感觉运动任务中的 `NeuronEmbedMapLayer` 和 `DetectionWindow`。

**关键模式：**
- 神经元空间之间的嵌入映射
- 基于脉冲的事件检测窗口
- `RecurrentNN` 之外的多模块组合

## `fmnist.py` — Fashion-MNIST 训练（纯 PyTorch）

使用基于 GLIF3 的 RSNN 进行 Fashion-MNIST 分类的完整训练流程，包含稀疏循环连接和电压正则化。

**关键模式：**
- 用于稀疏循环权重的 `SparseConn`
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
- 用于神经元参数缩放的 `scale_net` / `unscale_net`
- 带状态记录的验证循环

## `delayed_synapse_demo.py` — 突触延迟

演示使用 `SpikeHistory` 和 `DelayedSynapse` 实现异构突触延迟。

**关键模式：**
- 作为滚动脉冲缓冲区的 `SpikeHistory`
- 用于历史记录 + 线性变换的 `DelayedSynapse`
- 用于延迟感知连接矩阵的 `expand_conn_for_delays`

**另请参阅：**
- API: [`btorch.models.history`](api/models.md)
- 测试: [`tests/connectome/test_delay_expansion.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/tests/connectome/test_delay_expansion.py)

## `triton_sparse_manual_loop.py` — 手动 Triton 稀疏时间循环

演示如何将直接重复调用 `SparseConn` 的时间循环包裹在
`prepare_sparse_modules` 中。该作用域会在整个循环开始时准备边权重和 Triton
工作区；`RecurrentNN` 与 `make_rnn` 已经会自动执行这一步。

**关键模式：**
- 使用 `SparseConn(..., sparse_backend="triton")` 选择事件驱动稀疏后端
- 在完整时间循环外只准备一次稀疏模块
- 使用 CUDA `float32` 推理
- 输入可以包含批量维度；训练时支持一阶 autograd backward

对应脚本：[`examples/triton_sparse_manual_loop.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/examples/triton_sparse_manual_loop.py)

## 编译与基准复现

要复现教程中的规模匹配 proxy，请运行：

```bash
python benchmarks/sparse_rnn/benchmark_compile_triton_proxy.py \
  --backend triton --output /tmp/triton_proxy.json
```

该脚本比较原生稀疏实现和 Triton 稀疏实现的 eager 与
`torch.compile(mode="default")` 路径。完整实验条件、结果和限制见
[Triton 事件驱动稀疏教程](tutorials/triton_sparse.md)。

## 测试即示例

许多 `tests/` 文件包含简洁、经过验证的使用模式，非常适合作为文档：

- `tests/models/test_mem_load_save.py` — 检查点和状态恢复
- `tests/models/test_compile.py` — 带 `dt` 上下文的 `torch.compile`
- `tests/utils/test_conf.py` — OmegaConf 模式
- `tests/visualisation/*.py` — 几乎每个绘图函数

**提示：** 将测试代码改编为文档时，将 `fig_path` / `save_fig` 调用替换为标准 matplotlib 模式（例如 `plt.show()` 或返回图形对象）。
