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

## 运行 RSNN 示例

`examples/rsnn.py` 展示含传播延迟的 GLIF3 + 异质 AlphaPSC；
`examples/rsnn_izhikevich.py` 展示 Izhikevich + 异质 DualExponentialPSC。
两者均支持推理、训练和 SONATA 导入/导出，默认在 CPU 上运行小规模网络。

```bash
python -m examples.rsnn --device cuda --neuron-backend triton --sparse-backend triton --compile
python -m examples.rsnn_izhikevich --device cuda --neuron-backend triton --sparse-backend triton --method push --compile
```

`--method push` 显式选择 spike-list 传播；`--method pull` 选择 pull。
`--compile` 支持训练，显式 `--cudagraph` 要求 `--mode inference --device cuda`。
参见 [示例说明](../examples.md) 了解张量形状、状态初始化及模型映射。

## 梯度与精度

序列入口检查 autograd 状态，以及实际参与运算的输入、初始状态、连接权重和
模型浮点参数。Training 路径保留 backward 所需历史；inference 只保留最终状态
和用户请求的记录。`.train()`、`.eval()` 不参与分流。即使 RNN 参数冻结，可训练
encoder 的梯度仍会穿过 RNN；`no_grad()`、`inference_mode()` 强制 inference。

当前内置 kernel 支持 FP32、FP16、BF16 输入，cell 状态与累加使用 FP32，输出序列跟随输入 dtype。
这是目前实现的限制，精度与类型提升策略尚未确定。
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
