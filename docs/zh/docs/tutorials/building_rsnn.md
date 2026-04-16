# 教程 1：构建 RSNN

**作者：** btorch 贡献者  
**基于：** [`examples/rsnn.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/examples/rsnn.py)

本教程介绍如何使用 btorch 构建一个最小化的循环脉冲神经网络（RSNN）。

## 你将学到什么

- 如何组合神经元（`GLIF3`）、突触（`AlphaPSC`）和循环包装器（`RecurrentNN`）。
- 如何初始化和重置网络状态。
- 如何在前向传递期间记录内部状态变量（电压、电流）。
- `dt` 上下文如何控制 ODE 积分。

## 基本模块

一个最小化的 RSNN 需要三个组件：

1. **神经元模型** — 定义膜电压如何演化以及何时发放脉冲。
2. **突触模型** — 将脉冲转换为突触后电流。
3. **连接层** — 定义神经元之间的权重矩阵。

```python
import torch
import torch.nn as nn
from btorch.models import environ, functional, glif, rnn, synapse
from btorch.models.linear import DenseConn


class MinimalRSNN(nn.Module):
    def __init__(self, num_input, num_hidden, num_output, device="cpu"):
        super().__init__()

        # 1. 输入投影
        self.fc_in = nn.Linear(num_input, num_hidden, bias=False, device=device)

        # 2. 脉冲神经元
        neuron_module = glif.GLIF3(
            n_neuron=num_hidden,
            v_threshold=-45.0,
            v_reset=-60.0,
            c_m=2.0,
            tau=20.0,
            tau_ref=2.0,
            k=[0.1, 0.2],
            asc_amps=[1.0, -2.0],
            step_mode="s",   # 单步定义
            backend="torch",
            device=device,
        )

        # 3. 循环连接
        conn = DenseConn(num_hidden, num_hidden, bias=None, device=device)

        # 4. 突触
        psc_module = synapse.AlphaPSC(
            n_neuron=num_hidden,
            tau_syn=5.0,
            linear=conn,
            step_mode="s",
        )

        # 5. 循环包装器（多步）
        self.brain = rnn.RecurrentNN(
            neuron=neuron_module,
            synapse=psc_module,
            step_mode="m",
            update_state_names=("neuron.v", "synapse.psc"),
        )

        # 6. 输出读取
        self.fc_out = nn.Linear(num_hidden, num_output, bias=False, device=device)

    def forward(self, x):
        x = self.fc_in(x)                 # (T, Batch, num_input) -> (T, Batch, N)
        spike, states = self.brain(x)     # spike: (T, Batch, N)
        rate = spike.mean(dim=0)          # (Batch, N)
        out = self.fc_out(rate)           # (Batch, num_output)
        return out
```

## 初始化状态

在第一次前向传递之前，调用 `init_net_state` 来注册和初始化记忆缓冲区：

```python
model = MinimalRSNN(num_input=20, num_hidden=64, num_output=5)
functional.init_net_state(model, batch_size=4, device="cpu")
```

## 运行前向传递

将前向传递包装在 `environ.context(dt=...)` 块中：

```python
environ.set(dt=1.0)
inputs = torch.rand((100, 4, 20))  # (T, Batch, input_dim)

functional.reset_net(model, batch_size=4)
with environ.context(dt=1.0):
    out = model(inputs)  # (Batch, num_output)
```

## 查看记录的状态

`update_state_names` 告诉 `RecurrentNN` 要保存哪些变量。返回的 `states` 字典使用点号表示法：

```python
with environ.context(dt=1.0):
    spike, states = model.brain(inputs)

print(states["neuron.v"].shape)     # (T, Batch, N)
print(states["synapse.psc"].shape)  # (T, Batch, N)
```

你可以将其展开以便更方便地访问：

```python
from btorch.utils.dict_utils import unflatten_dict
nested = unflatten_dict(states, dot=True)
nested["neuron"]["v"][:, 0, :]   # 批次 0 的电压
```

## 下一步

在 [教程 2：训练 SNN](training.md) 中，我们将添加损失函数、训练循环和检查点保存。
