# Out-of-tree cell kernel

外部 Python 包可以注册 cell 实现，无需 fork btorch。注册只选择数值执行方式；
btorch 继续管理有状态 Module API、autograd 分流和完整序列的算子边界。

```python
from btorch.models.kernels import register_cell_backend

register_cell_backend(
    "lab_cell",
    forward=forward,
    backward=backward,
    supports=lambda spec, device: spec.model == "lif" and device.type == "cuda",
)
neuron = LIF(128, step_mode="m", backend="lab_cell", device="cuda")
```

名称必须是标识符。重复注册报错，除非显式 `replace=True`；内置名称不可替换。
替换前需完成尚未执行的 backward；若 backend 或 model 注册已改变，旧 forward
对应的 backward 会明确报错。替换后需重建编译 callable 和独立捕获图；RSNN 的
graph runner 会依据 backend revision 使缓存失效。仅提供 forward 的实现
只用于 inference，需要梯度时在入口报错。不支持的模型或设备不会静默回退 Torch。

## Tensor callback 契约

`ModelTemplate` 固定声明 state、parameter 和 history 的名称与顺序。实际参与
运算的参数均显式作为 tensor 输入，保留计算图。Callback 使用 tuple：

```python
def forward(x_seq, states, parameters, spec, history_names):
    return output_seq, final_states, histories


def backward(x_seq, initial_states, parameters, spec, history_names, histories,
             grad_output, grad_final_states, grad_histories, *, needs_input_grad=None):
    return grad_x, grad_initial_states, grad_parameters
```

Forward 返回 FP32 输出、状态、记录；序列 wrapper 将公共输出转为输入 dtype。
返回结果必须符合 custom operator 的非 alias 契约。`history_names` 是用户记录和
反向需求的并集，不应另行保存全部 residual；wrapper 复用 PSC 输出和对应历史的
存储。Backward 同时包含输出、最终状态与记录的梯度。初始状态冻结时仍需传播
时间递归中的 state adjoint；`needs_input_grad` 裁剪返回值，不截断递归。

新模型使用 `register_model_template`，必须提供具名声明、状态扩展维度、记录
形状来源及可信的单步 forward/backward reference。注册后端名称本身不会定义
新的模型方程。

## 生成与验证

`generate_cell_source(backend, spec, n_lanes, n_steps, history_names, ...)` 返回
可检查的 DSL 源码和有序 buffer 声明，与内置执行共享模型片段。生成不执行编译；
CUDA graph capture 前需 warmup。外部包可通过 `resolve_cell_backend` 复用内置
callback 实现，再独立调整调度。

公共验证工具不依赖 pytest：

```python
from btorch.models.kernels.testing import ParityCase, check_backend_parity

report = check_backend_parity("lab_cell", [
    ParityCase(x_seq, states, parameters, spec, history),
])
report.assert_close()
```

按 kernel 支持的形状、初始状态、参数广播、dtype 和模型选项创建 case。工具会
对照现有 btorch Python 模型，比较输出、最终状态、请求的记录，以及输入、状态
和参数梯度。对于 btorch 之外的新模型，通过 `ParityCase.golden` 提供其 Python 实现。

使用非零初始状态，并请求附加 loss 会用到的记录。检查长序列及自己支持的每种
surrogate/reset 选项。报告指出失败张量、坐标及首次 spike 分歧的时间步。
Spike 必须严格一致；连续张量使用可配置的容差。

每个 case 默认还检查：

- 空序列、单步序列，以及数值相同的非连续输入。
- 所有已声明的状态与 charged-state 记录，即使提供的 case 没有请求它们。
- 仅输入、仅状态、仅参数求梯度；全部冻结；下游 readout 可训练；
  `no_grad` 和 `inference_mode`。
- 两段调用保持状态梯度连续，以及两次 forward 后一次联合 backward。

定位单个问题时，可设置
`ParityOptions(check_edge_cases=False, check_continuation=False)`。工具保留你提供
的 dtype 和参数数值；需要自行为声称支持的每种精度、形状、广播、长序列和模型
选项添加 case。

`ParityOptions(compile=True, capture=True)` 额外检查 fullgraph 与 warmup 后的
CUDA replay。开启梯度检查时，capture 还会将重放的前向、反向以及输入、状态和参数
梯度与 eager Torch autograd 比较。CPU capture 显式记为 skip，请检查 `report.skipped`；skip 不等于
已经支持。没有执行数值检查或存在失败时，`assert_close` 会报错。

`torch.library.opcheck` 检查 custom operator 的 schema、fake/meta、autograd 注册
和编译集成，不证明梯度数值正确。`torch.autograd.gradcheck` 使用有限差分，适合
平滑 PSC dynamics；它不能验证刻意替换的 spike surrogate 导数。Spike 梯度仍需
对照现有 btorch Module 的 autograd 路径。参见
[PyTorch 算子测试文档](https://docs.pytorch.org/docs/stable/library.html#testing-custom-ops)。

参见 [可运行的 OOT 教程](../tutorials/oot_kernels.md)。Sparse 扩展继续使用
[已有 sparse registry](sparse_connectivity.md)；cell 注册不会重定义拓扑、delay
布局或 sparse 算法。
