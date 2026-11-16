# Neuron 与 connection 后端

分别选择 neuron/PSC dynamics 和 sparse propagation：

```python
from btorch.models import KernelOptions
from btorch.models.rnn import RecurrentNN

rnn = RecurrentNN(neuron, synapse, backend=KernelOptions(
    neuron="triton", connection="auto", merge_synapse=False,
))
```

字符串只选择 cell 后端。内置名称为 `torch`、`triton`、`tilelang`、`cupy`；
外部注册的名称也使用同一个参数。GPU 包都是可选依赖：Triton 可通过
`pip install 'btorch[gpu]'` 安装，TileLang、CuPy 请按 CUDA 环境单独安装。
Cell 后端复用现有 sparse runtime，不重复实现 sparse 算法。

注册的 cell 序列算子需要 PyTorch 2.4 或更新版本。
训练支持一阶 BPTT；`create_graph=True` 会明确报错，手写 kernel VJP 不提供高阶导数。

首批模型包括 IF、LIF、GLIF3、Izhikevich，以及 ExponentialPSC、AlphaPSC、
AlphaPSCBilleh、DualExponentialPSC。六种确定性的内置 surrogate 保持 btorch
本身的归一化导数。暂不支持的自定义 surrogate 或可训练超参数会明确报错。

独立神经元/PSC 序列调用复用模型元数据、已校验的布局和预备执行内核。每次调用读取当前状态与参数张量；修改 `dt` 不会重新编译内核。Triton 和 CuPy 还可跨序列长度复用内核；TileLang 目前按序列长度特化。

## CUDA Graph 推理

`RecurrentNN` 默认使用 `cudagraph=None`：兼容的 CUDA 推理自动使用图重放。
推理请使用 `torch.no_grad()` 或 `torch.inference_mode()`；需要梯度的调用仍走
普通 autograd。原生后端显式设置 `cudagraph=True` 时，分别捕获有界 chunk 的
forward 和 reverse。每个 chunk 的反向历史独立保存，状态梯度跨 chunk 传播，
不隐式 detach。检查点重算和 CPU offload 在捕获边界外执行，较短的末块走 eager。
通用 Torch runner 的手动捕获仍仅支持推理。

`cudagraph=False` 选择 eager 提交，也可配合
`torch.compile(model, mode="reduce-overhead")` 使用 Inductor 对合适区域的捕获。
图中断和仍被反向引用的 chunk 输出可能限制 Graph Trees 重放；它不保证整次训练
迭代捕获。不要将 reduce-overhead 嵌套到手动捕获内。

预热与捕获由 `chunk_size=200` 限定。等长 chunk 复用同一张图，较短的末块使用
eager。每次调用开始时读取当前 memory，包括 reset 和原地修改。各块结果写入
独立持有的完整序列记录，后续调用不会覆盖它。`cpu_offload=True` 将记录放到
CPU；否则完整记录所需的 GPU 存储仍与序列长度成正比。训练中已为反向保存到
CPU 的记录复用同一数值快照，保留传输梯度和原地修改检查，避免再次传输。

## 准备与自动调优

适用的 GPU kernel 在准备阶段自动比较两个 CUDA kernel 配置。Triton 和 TileLang
使用各自的原生 autotuner，CuPy 使用其 CUDA-event profiler。准备完成后直接
调用选定的可执行 kernel；首次调用包含编译和调优，应与预热后的运行时间分开测量。

Triton 和 TileLang 自己保存调优结果。复用相同执行配置时不重复搜索，也不分配
试运行存储。已缓存的调优器不持有试运行张量；已准备的 kernel 只持有实际运行的参数。

调优使用独立的可写存储，保留别名和 stride。小规模计算、超过试运行内存预算的
计算，以及设备队列尚未填充的消费者保留启发式配置；不改变 spike-list 的 block
归属或队列容量。TileLang 的累加阶段使用公开 profiler，每次试运行前恢复缓冲区。

CUDA Graph capture 前只需预热实际的 RecurrentNN chunk；首次编译和调优须在
capture 外完成。重复 capture 不启动搜索，
而是复用已选配置或启发式配置。配置选择缓存在进程内，后端编译缓存仍可跨进程复用。

## 运行 RSNN 示例

`examples/rsnn.py` 展示含传播延迟的 GLIF3 + 异质 AlphaPSC；
`examples/rsnn_izhikevich.py` 展示 Izhikevich + 异质 DualExponentialPSC。
两者均支持推理、训练和 SONATA 导入/导出。GLIF 示例默认是小规模 CPU 网络；
Izhikevich 示例默认使用 60,000 神经元、33,072,348 条边和 CUDA Graph。

```bash
python -m examples.rsnn --device cuda --neuron-backend triton --sparse-backend triton --compile
python -m examples.rsnn_izhikevich --device cuda --neuron-backend triton --sparse-backend triton --method push --compile
```

`--method push` 显式选择 spike-list 传播；`--method pull` 选择 pull。
`--cudagraph` 要求 CUDA，支持原生后端的 BPTT chunk 重放。
`--no-cudagraph --compile` 切换为 Inductor 训练路径。长序列可使用
`--grad-checkpoint --cpu-offload`，不截断跨 chunk 的梯度。
参见 [示例说明](../examples.md) 了解张量形状、状态初始化及模型映射。

## 梯度与精度

序列入口检查 autograd 状态，以及实际参与运算的输入、初始状态、连接权重和
模型浮点参数。Training 路径保留 backward 所需历史；inference 只保留最终状态
和用户请求的记录。`.train()`、`.eval()` 不参与分流。即使 RNN 参数冻结，可训练
encoder 的梯度仍会穿过 RNN；`no_grad()`、`inference_mode()` 强制 inference。

Torch、Triton 和 TileLang 保留显式提供的 FP16/BF16 神经元和 PSC 状态精度；
混合浮点操作数遵循 Torch 类型提升规则，准备阶段不将状态或参数强制转换为 FP32。
缺少低精度数学接口的操作仅临时提升该次运算，并立即转回结果。
时间方向的伴随梯度和参数归约可使用 FP32 累加，但不提升前向轨迹或 checkpoint 重放的精度。
公共输出序列跟随输入 dtype。CuPy renderer 的标量局部变量目前仍使用 FP32。
单模型时间循环按提升后的模型 dtype 绑定混合初始状态，确保循环寄存器类型稳定；
统一的 FP16/BF16 模型仍保留原精度。CUDA Graph 复用还要求状态 shape、dtype 和 stride
不变；仅修改状态值不会使捕获失效。
时间在第一维，通常是 `[T,B,N]`；初始状态没有时间维。Adaptation 和 receptor 轴
显式保留，整数拓扑、receptor 和 delay 不求梯度。

每次调用读取标量 `environ.dt`；修改它不会重新生成 kernel 或重新编译序列计算图，
backward 使用对应 forward 的 dt。CUDA graph replay 保留捕获值，除非更新其设备输入。

## 状态与记录

使用 `init_state` 或 `functional.init_net_state` 初始化。最终状态保持可微，并在
连续调用间保留计算图，直到显式 reset 或 detach。RSNN 返回的状态记录可参与
附加 loss；每次调用独立保存反向数据，支持两次 forward 后统一 backward。

显式 kernel 选项使用有状态 PSC 序列递推。原有 `backend="torch"` 的 PSC 多步
卷积从静息状态计算，由 `kernel_len` 截断 impulse response，不写回状态；需要
有状态 Torch reference 时使用 `backend=KernelOptions(neuron="torch")`。

`merge_synapse=False` 保留独立 receptor 状态，可查询 EPSC/IPSC。显式 merge 仅
适用于兼容的线性 dynamics，以及兼容的冻结参数或真正共享的参数对象。独立可训练
参数不能只因当前数值相等而合并。合并状态无法还原独立 receptor 记录；不兼容
请求会报错。为新模型开启 merge 前，请确认已测试的支持范围。

首版序列集成支持 identity、dense `nn.Linear` 和 `SparseConnection`；这条路径
暂不支持自定义 connection module 或 sparse `member_pattern_batch`。Receptor
merge 是 heterogeneous RSNN 的选项，独立 neuron/PSC 调用会拒绝该选项。
外部 cell backend 通过注册的 callback 与相同 connection 组合，不会自动获得
内置 paired cell fusion。

参见 [外部 kernel 指南](oot_kernels.md)、
[训练教程](../tutorials/kernel_training.md) 和 [sparse 指南](sparse_connectivity.md)。

`ConductancePSC` 的独立调用可使用原生后端。连接 COBA 网络时使用
`ConductanceRecurrentNN`；该类只接受 `ConductancePSC` 类型，且网络中所有
模块必须使用纯 `backend="torch"`。目前不支持原生后端的循环 COBA 执行。

较长训练序列可结合 checkpoint 与 CPU offload：

```python
network = RecurrentNN(
    neuron, synapse, chunk_size=200, chunk_mode="split",
    grad_checkpoint=True, cpu_offload=True,
    update_state_names=("neuron.v",),
)
```

Checkpoint 在反向时重算私有激活；CPU offload 将保存的私有张量和返回的历史
移至主机内存，参数和调用方输入保留在原设备。返回的历史仍可参与梯度计算。
开启 offload 后，完整轨迹占用的 CPU 内存随序列长度增长，GPU 记录缓冲区
则由 chunk 大小限制。模型权重和调用方的 GPU 输入仍驻留在 GPU。
只选择需要的记录，或通过 `update_state_names=()` 省略状态轨迹。


### 数学精度

`KernelOptions(fast_math=False)` 为内置神经元、PSC 和反向重放选择 libdevice
指数函数与精确舍入除法。默认 `True` 使用近似 FP32 `exp`；指数 Euler 始终
复用 libdevice `expm1`，避免相消，也不引入额外的近似实现。FP64 始终使用
libdevice。Triton 的 libdevice 路径可能将 FP32 非规格化数归零，包括提升到
FP32 的极小 BF16 输入；`fast_math=False` 不保证保留非规格化数。
保留 half/BF16 存储类型，仅在超越函数的 GPU ABI 要求时提升计算精度。
Torch 参考路径保留 PyTorch 原生运算。

```python
options = KernelOptions(neuron="triton", fast_math=False, enable_fp_fusion=False)
```

`enable_fp_fusion` 单独控制编译器乘加收缩。数学模式不保证长序列逐位一致，也不改变
稀疏累加顺序。ALIF/ELIF 的后端字符串简写选择 `fast_math=False`，显式选项优先。
修改这些选项会使准备好的内核和 CUDA Graph 缓存失效。

自定义 Triton/TileLang 函数可复用
`btorch.models.kernels.backends.math_triton` 或 `math_tilelang` 中的
`exp(x, fast_math=True)`、`expm1(x)`。设计遵循
[SGLang PR #48942](https://github.com/sgl-project/sglang/pull/48942) 的显式快速/精确
数学模式，不依赖全局编译器 fast-math 开关。CuPy 显式选择指数指令，两个模式的
`expm1` 均使用 libdevice。
