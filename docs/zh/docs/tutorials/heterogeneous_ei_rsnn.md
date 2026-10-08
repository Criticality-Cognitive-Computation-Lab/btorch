# 教程：异构多受体 E/I RSNN

本教程构建一个带四类受体通道的 E/I 循环脉冲网络：E→E、E→I、I→E 和 I→I。连接使用一张语义边表，受体通道通过 `Synapse(receptor=...)` 表达；`HeterSynapsePSC` 为每个通道维护独立的 Alpha PSC，再把通道电流汇总到目标神经元。

这里不把 `n_neuron * n_receptor` 的物理展开矩阵当作模型定义。那只是 `SparseConnection` 为后端执行生成的布局。这样检查点、权重和边查询仍然使用神经元级的 `pre`、`post` 和 `receptor`。

## 完整示例

```python
import torch
from torch import nn
import pandas as pd

from btorch.models import environ, functional
from btorch.models.connection import SparseConnection, Synapse
from btorch.models.constrain import constrain_net
from btorch.models.neurons import GLIF3
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import AlphaPSC, HeterSynapsePSC


class HeterogeneousEIRSNN(nn.Module):
    """E/I RSNN with semantic receptor-labelled recurrent edges."""

    def __init__(self, n_input, n_exc, n_inh, n_output, device):
        super().__init__()
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_neuron = n_exc + n_inh
        neuron_type = torch.cat(
            [
                torch.zeros(n_exc, dtype=torch.long),
                torch.ones(n_inh, dtype=torch.long),
            ]
        )

        source = torch.arange(self.n_neuron).repeat_interleave(self.n_neuron)
        target = torch.arange(self.n_neuron).repeat(self.n_neuron)
        keep = source != target
        pre = source[keep].to(device)
        post = target[keep].to(device)
        pre_type = neuron_type[pre.cpu()].to(device)
        post_type = neuron_type[post.cpu()].to(device)

        # Receptor order: E->E, E->I, I->E, I->I.
        receptor = pre_type * 2 + post_type
        generator = torch.Generator(device=device).manual_seed(0)
        magnitude = 0.04 + 0.02 * torch.rand(
            pre.numel(), generator=generator, device=device
        )
        initial_weight = torch.where(pre_type == 0, magnitude, -magnitude)

        receptor_index = pd.DataFrame(
            {
                "receptor_index": [0, 1, 2, 3],
                "pre_receptor_type": ["E", "E", "I", "I"],
                "post_receptor_type": ["E", "I", "E", "I"],
            }
        )

        self.connection = SparseConnection.from_edges(
            pre,
            post,
            n_pre=self.n_neuron,
            n_post=self.n_neuron,
            values=initial_weight,
            synapse=Synapse(
                receptor=receptor,
                n_receptor=4,
                dale=True,
            ),
            device=device,
        )
        self.receptor_index = receptor_index

        tau_by_receptor = torch.tensor(
            [5.0, 8.0, 6.0, 10.0],
            device=device,
        ).repeat(self.n_neuron)
        psc = HeterSynapsePSC(
            n_neuron=self.n_neuron,
            n_receptor=4,
            receptor_type_index=receptor_index,
            linear=self.connection,
            base_psc=AlphaPSC,
            tau_syn=tau_by_receptor,
            step_mode="s",
        )
        neuron = GLIF3(
            n_neuron=self.n_neuron,
            v_threshold=-45.0,
            v_reset=-60.0,
            c_m=2.0,
            tau=20.0,
            k=[1.0 / 80.0],
            asc_amps=[-0.2],
            tau_ref=2.0,
            step_mode="s",
            backend="torch",
            device=device,
        )
        self.brain = RecurrentNN(
            neuron=neuron,
            synapse=psc,
            step_mode="m",
            update_state_names=("neuron.v", "synapse.psc"),
        )
        self.input = nn.Linear(n_input, self.n_neuron, bias=False, device=device)
        self.output = nn.Linear(self.n_neuron, n_output, device=device)

    def forward(self, x):
        spikes, states = self.brain(self.input(x))
        logits = self.output(spikes.mean(dim=0))
        return logits, spikes, states


torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, time_steps = 6, 24
model = HeterogeneousEIRSNN(
    n_input=12,
    n_exc=6,
    n_inh=4,
    n_output=3,
    device=device,
).to(device)
model = model.to(device=device, dtype=torch.float32)
functional.init_net_state(
    model,
    batch_size=batch_size,
    device=device,
    dtype=torch.float32,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
inputs = torch.randn(time_steps, batch_size, 12, device=device)
targets = torch.randint(0, 3, (batch_size,), device=device)

environ.set(dt=1.0)
model.train()
for _ in range(3):
    functional.reset_net(model, batch_size=batch_size)
    optimizer.zero_grad()
    with environ.context(dt=1.0):
        logits, spikes, states = model(inputs)
        loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    constrain_net(model)

assert logits.shape == (batch_size, 3)
assert spikes.shape == (time_steps, batch_size, 10)
assert states["neuron.v"].shape == spikes.shape
assert model.connection.n_receptor == 4
edge_weight = model.connection.weight()
assert bool((edge_weight[model.connection.pre < model.n_exc] >= 0).all())
assert bool((edge_weight[model.connection.pre >= model.n_exc] <= 0).all())
print(f"loss={loss.item():.4f}, edges={model.connection.nnz}")
```

## 受体索引表

`receptor_index` 是模型旁边的语义元数据，不会被编码进物理矩阵。四个通道的索引约定为：

| `receptor_index` | 源类型 | 目标类型 | `tau_syn` |
| ---: | --- | --- | ---: |
| 0 | E | E | 5.0 |
| 1 | E | I | 8.0 |
| 2 | I | E | 6.0 |
| 3 | I | I | 10.0 |

边上的 `receptor` 是 `[n_edge]` 的整数张量。`HeterSynapsePSC` 的输出先按 `post * n_receptor + receptor` 排列，再沿最后一个受体轴求和，所以 `RecurrentNN` 仍然看到 `[batch, n_neuron]` 的总突触电流。

## Dale 约束

教程把所有从 E 神经元发出的边初始化为正，把所有从 I 神经元发出的边初始化为负，并设置 `Synapse(dale=True)`。`EdgeWeight` 保存每条边的参考符号；每次优化器更新后调用 `constrain_net(model)`，确保符号不会改变。

如果需要不同的权重共享策略，可以把 `ConstrainedWeight` 传给 `Synapse(weight=...)`。受体和延迟仍然是边属性，不需要创建 `HeterogeneousLinear` 或其他专用线性层。

## 读取通道电流

前向之后，可以读取未汇总的通道 PSC：

```python
psc = model.brain.synapse
ee_current = psc.get_psc(receptor_type=("E", "E"))
ie_current = psc.get_psc(receptor_type=("I", "E"))
assert ee_current.shape == ie_current.shape == (batch_size, model.n_neuron)
```

也可以检查当前语义边和权重：

```python
edges = model.connection.edge_table()
assert {"pre", "post", "receptor", "weight"}.issubset(edges)
current_weight = model.connection.weight()
```

## 加入异构延迟

延迟与受体使用同一套语义边模型。为每条边增加 `delay`，并设置 `n_delay`：

```python
synapse = Synapse(
    receptor=receptor,
    delay=delay_steps,
    n_receptor=4,
    n_delay=3,
    dale=True,
)
```

`SparseConnection` 会声明 `n_delay`，`HeterSynapsePSC` 自动创建 `SpikeHistory`，将延迟历史送入连接。输入仍然是未展开的 `[batch, n_neuron]` 脉冲；历史和扁平索引由 PSC 与连接内部管理。

## 与物理展开数据互操作

如果数据来自旧的 `make_hetersynapse_conn`，可以使用：

```python
conn = SparseConnection.from_hetersynapse(
    expanded_matrix,
    receptor_type_index=receptor_index,
    n_delay=3,
)
```

这只是适配入口。新建模型时，优先使用本教程中的 `from_edges` 与 `Synapse(receptor=..., delay=...)`，因为它们保留了神经元级拓扑和边属性。

## 下一步

要比较稀疏连接的 planner、task queue 和 CUDA Graph，请参阅[稀疏连接指南](../guides/sparse_connectivity.md)。
