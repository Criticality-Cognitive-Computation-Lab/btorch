# btorch 中的有状态模块

与标准 PyTorch 模块不同，许多 btorch 组件携带随时间演化的内部状态。这种模式对脉冲神经网络（SNN）至关重要，因为膜电压、突触电流和脉冲历史必须在各前向步骤之间保持。

## MemoryModule

btorch 状态管理的核心是 [`MemoryModule`][btorch.models.base.MemoryModule]。它在 `nn.Module` 基础上扩展了：

- **`register_memory`**：声明一个随时间变化的缓冲区（例如膜电压 `v`）。
- **`_memories`**：当前状态值的字典。
- **`_memories_rv`**：重置值，用于在新批次或新试验开始时恢复状态。

当你调用 `functional.init_net_state(model, batch_size=4)` 时，btorch 会遍历模块树，并将每个 `MemoryModule` 的缓冲区初始化为请求的形状。

## 典型生命周期

```python
from btorch.models import functional

# 1. 一次性初始化状态缓冲区
functional.init_net_state(model, batch_size=4, device="cuda")

# 2. 应用选定的初始化方法
init.uniform_v_(model.neuron, save_reset_values=False)

# 3. 在每个批次 / 试验前重置
functional.reset_net(model, batch_size=4)

# 4. 运行前向传递
with environ.context(dt=1.0):
    spikes, states = model(x)

# 5. 为截断时间反向传播分离梯度
functional.detach_net(model)
```

## 状态名使用点号表示

`RecurrentNN` 使用点号表示法记录状态：

```python
states = {
    "neuron.v":       torch.Tensor,  # (T, Batch, N)
    "neuron.Iasc":    torch.Tensor,  # (T, Batch, N, n_asc)
    "synapse.psc":    torch.Tensor,  # (T, Batch, N)
}
```

你可以使用 `btorch.utils.dict_utils.unflatten_dict` 将其展开：

```python
from btorch.utils.dict_utils import unflatten_dict
nested = unflatten_dict(states, dot=True)
# nested["neuron"]["v"]  -> (T, Batch, N)
```

## 为什么 `state_dict()` 不够用

动态缓冲区（膜电压、突触电流）被有意排除在 `state_dict()` 之外，因为它们的形状依赖于批次大小，并在运行时重建。要保存训练好的模型的检查点，请保存：

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
}
```

如有需要，也可以选择保存：

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "memories_rv": functional.named_memory_reset_values(model),  # 如果重置值是随机的
    "hidden_states": functional.named_hidden_states(model),      # 如果你需要神经元状态
}
```

参阅 [`functional`][btorch.models.functional] 了解完整的状态管理 API。
