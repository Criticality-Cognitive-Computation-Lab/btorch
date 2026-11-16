# 编写自定义 LIF kernel

在安装了 Triton 的 CUDA 设备上运行：

```bash
python -m examples.oot_cell_backend
```

示例只展示一件事：替换 LIF 前向实现，同时保留常规 btorch Module 接口和已有的
替代梯度反向实现。按调用顺序分为三部分：

1. `lif_kernel` 是手写 Triton kernel。每个 lane 在整个序列上更新一个神经元，
   输出 spike、最终电压及请求的记录。
2. `forward` 将 btorch 的张量回调适配到该 kernel。状态顺序为 `(v,)`，参数顺序为
   `(v_threshold, v_reset, c_m, tau)`。广播参数和带 stride 的输入保留为视图，
   返回张量不与输入共享存储。
3. `register_example_backend` 将前向实现注册为 `tutorial_lif`，并复用 btorch
   的 Triton LIF 反向实现。随后通过 `LIF(..., backend="tutorial_lif")` 选择它，
   无需修改状态初始化或训练代码。

```python
from btorch.models.neurons import LIF
from examples.oot_cell_backend import register_example_backend

register_example_backend()
neuron = LIF(129, tau=4.0, step_mode="m", backend="tutorial_lif", device="cuda")
```

复用该反向实现的前提是自定义前向遵守相同的 LIF 方程和 reset 约定。改变方程时
必须提供匹配的反向。示例明确不支持不应期和随机变体，输入为
`(time, batch, neuron)`。

`run_example` 先通过 `check_backend_parity` 对照已有 Torch 模型检查 spike、电压记录和
梯度，包括编译和 CUDA Graph 重放，再使用常规 LIF Module，对输入缩放参数计算
spike/电压 loss 的梯度。
示例不再混合 RSNN、连接布局和源码生成流程。

需要共享方程模板而非手写前向时，参见
[kernel 生成](../design/kernel_generation.md)。回调顺序、能力声明、compile 和
CUDA Graph 检查参见[扩展契约](../guides/oot_kernels.md)。在线学习使用独立示例
`examples/oot_stdp.py`，见[可塑性指南](../guides/plasticity.md#out-of-tree-native-learning-templates)。
