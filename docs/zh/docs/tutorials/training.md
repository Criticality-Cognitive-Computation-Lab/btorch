# 教程 2：训练 SNN

**作者：** btorch 贡献者  
**基于：** [`examples/fmnist.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/examples/fmnist.py)、[`skills/btorch-snn-modelling/references/training_example.md`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/skills/btorch-snn-modelling/references/training_example.md)

本教程介绍如何使用 btorch 训练脉冲神经网络，包括状态初始化、`dt` 环境、检查点保存和截断时间反向传播。

## 网络设置

我们复用 [教程 1](building_rsnn.md) 中的 RSNN 模式，但采用稀疏循环连接和 Fashion-MNIST 示例中使用的 Billeh alpha-PSC 突触：

```python
import torch
from btorch.models import environ, functional
from btorch.models.neurons import GLIF3
from btorch.models.synapse import AlphaPSCBilleh
from btorch.models.linear import SparseConn
from btorch.models.rnn import RecurrentNN
from btorch.models.init import uniform_v_
from btorch.models.regularizer import VoltageRegularizer

# 创建一个任意稀疏矩阵作为示例
from tests.utils.conn import build_sparse_mat  # 来自测试套件的辅助函数
weights, _, _ = build_sparse_mat(n_e=80, n_i=20, i_e_ratio=1.0)
conn = SparseConn(conn=weights)

neuron = GLIF3(
    n_neuron=100,
    v_threshold=-45.0,
    v_reset=-60.0,
    c_m=2.0,
    tau=20.0,
    k=[1.0 / 80],
    asc_amps=[-0.2],
    tau_ref=2.0,
    detach_reset=False,
    step_mode="s",
    backend="torch",
)

# AlphaPSCBilleh 在初始化时需要 dt
environ.set(dt=1.0)
psc = AlphaPSCBilleh(
    n_neuron=100,
    tau_syn=torch.cat([torch.ones(80) * 5.8, torch.ones(20) * 6.5]),
    linear=conn,
    step_mode="s",
)

model = RecurrentNN(
    neuron=neuron,
    synapse=psc,
    step_mode="m",
    update_state_names=("neuron.v", "synapse.psc"),
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

## 初始化并随机化状态

```python
# 1. 注册记忆缓冲区
functional.init_net_state(model, batch_size=32, device=device)

# 2. 随机化膜电压并将其存储为重置值
uniform_v_(model.neuron, set_reset_value=True)
```

`set_reset_value=True` 很重要：它告诉 `reset_net` 在每个批次开始时将电压恢复到这些随机化值。

## 训练循环

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
voltage_reg = VoltageRegularizer(-45.0, -60.0, voltage_cost=1.0)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        x, target = batch  # x: (T, Batch, input_dim)
        x = x.to(device)
        target = target.to(device)

        # 每个批次前重置状态
        functional.reset_net(model, batch_size=x.shape[1])

        optimizer.zero_grad()

        with environ.context(dt=1.0):
            spikes, states = model(x)

            # spikes: (T, Batch, N) -> 速率编码
            rate = spikes.mean(dim=0)  # (Batch, N)
            task_loss = criterion(rate, target)

            # 电压正则化
            v_loss = voltage_reg(states["neuron.v"])
            loss = task_loss + 0.1 * v_loss

        loss.backward()
        optimizer.step()
```

## 检查点保存

动态缓冲区被排除在 `state_dict()` 之外。要完全恢复模型，需要同时保存记忆重置值和权重：

```python
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "memories_rv": functional.named_memory_reset_values(model),
    }, path)

def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # 加载权重（动态键已被排除）
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # 恢复记忆重置值
    if "memories_rv" in ckpt:
        functional.set_memory_reset_values(model, ckpt["memories_rv"])
    if "hidden_states" in ckpt:
        functional.set_hidden_states(model, ckpt["hidden_states"])

    return ckpt["epoch"]
```

## 截断时间反向传播

对于长序列，你可以使用 `detach_net` 将 BPTT 分成多个片段：

```python
chunk_size = 50
for t in range(0, T, chunk_size):
    functional.detach_net(model)
    # 注意：此处不要调用 reset_net；状态应在片段之间保持
    spikes, states = model(x[t:t+chunk_size])
    loss = criterion(spikes.mean(0), target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

`detach_net` 在当前状态值处断开计算图，防止梯度流回更早的片段。

## 关键要点

1. **始终重置状态** — 在每个新批次前使用 `functional.reset_net`。
2. **始终包装前向传递** — 在 `environ.context(dt=...)` 中运行。
3. **保存 `memories_rv`** — 保存检查点时一并保存；`state_dict()` 不包含动态状态。
4. **使用 `detach_net`** — 用于长序列的截断时间反向传播。

有关常见错误和故障排除，请参阅 [FAQ](../faq.md)。
