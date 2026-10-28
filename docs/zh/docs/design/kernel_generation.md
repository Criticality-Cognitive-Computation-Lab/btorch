# 生成内核的模型契约

btorch 使用同一份模型定义生成独立 neuron/PSC 内核与融合 neuron–PSC 步骤。
Triton、CuPy 和 TileLang 各自负责存储访问与执行语法。

使用已有模型时，通过 `KernelOptions` 选择后端，见
[内核后端](../guides/kernel_backends.md)。本文面向向生成器添加模型方程的开发者。
若要接入独立实现的内核，见[外部内核](../guides/oot_kernels.md)。

## 声明模型

使用 `btorch.models.kernels.templates` 中的 `register_model_template`
注册 `ModelTemplate`。内置定义位于 `btorch/models/kernels/definitions/`。

| 声明 | 用途 |
| --- | --- |
| `state_names`、`parameter_names` | 确定张量参数顺序 |
| `history_names`、`history_sources` | 声明支持的记录及其状态形状来源 |
| `state_extra_dims`、`parameter_extra_dims` | 描述 batch、neuron 维度之外的分量轴 |
| `fragments` | 提供前向与反向标量阶段 |
| `module_factory` | 构造用于数值校验的独立 Torch 模型 |

Module 通过类属性 `kernel_model` 选择注册定义。改变方程的子类必须显式声明
自己的绑定，不会自动使用继承的绑定。物理状态仍通过 `register_memory` 注册，
保留普通 Module 的状态初始化与续算接口。

## 同时定义前向与反向

Fragment 构造函数返回 `BackendFragments`。前向阶段使用 Python 表达式语法
编写标量赋值；反向阶段计算输入、初始状态及参与参数的 VJP，包括记录状态上
的损失。可参考内置定义：`izhikevich.py` 包含电压与适应状态，`alpha.py`
包含相互耦合的 PSC 状态。

若某个记录的伴随必须在反向阶段之间加入，在相应位置插入
`RecordCotangent(record_name)`。其他状态记录的伴随在反向步骤边界加入。
当记录与输出对应同一物理状态时，这个顺序尤其重要。

请显式提供反向方程。仅追踪前向操作无法指定 spike 替代梯度或 detached-reset
语义。使用原始 Torch 模型校验生成的 VJP，包括非零初始状态与记录上的损失。

## 准备与渲染

模型定义描述数学语义；准备阶段绑定张量布局、预计算系数、残差存储及编译内核。
在反向中重建状态的模型声明 `checkpoint_state`、`checkpoint_interval` 和
`replay_state`；需要预计算系数的模型声明 `prepared_parameters` 与
`prepare_parameters`。

独立内核与融合内核为共享标量阶段设置各自的命名空间。后端渲染器提供张量访问、
数学函数与源码外壳，不包含模型方程。编译扩展前，可通过
`generate_cell_source` 或 `generate_fused_source` 查看生成源码。

Triton 和 CuPy 序列内核在调用时接收时间步数，只改变序列长度可复用生成源码
及编译内核。其余形状、步长、dtype 和模型选项仍用于特化。当前 TileLang 包装
使用静态循环与缓冲区长度，因此仍按序列长度特化。CUDA Graph 仍按形状捕获。
