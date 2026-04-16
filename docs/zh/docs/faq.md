# 常见问题 / 常见陷阱

本页面收集了使用 btorch 时最常见的错误及其修复方法。内容来源于 [`btorch-snn-modelling` 技能](skills.md) 和测试套件。

## 1. 忘记 `dt` 上下文

**症状：** `KeyError: 'dt is not found in the context.'`

**修复：** 将每个前向传递包装在 `environ.context(dt=...)` 中：

```python
from btorch.models import environ

with environ.context(dt=1.0):
    spikes, states = model(x)
```

详见 [`dt` 环境](concepts/dt_environment.md)。

## 2. 未在批次之间重置状态

**症状：** 前一批次的状态泄漏到当前批次，导致训练或验证结果不稳定。

**修复：** 在每个新批次前调用 `reset_net`：

```python
from btorch.models import functional

functional.reset_net(model, batch_size=x.shape[1])
```

要实现确定性重置，请先初始化随机电压：

```python
from btorch.models.init import uniform_v_
uniform_v_(model.neuron, set_reset_value=True)
```

## 3. 错误的状态名（点号表示法）

**症状：** 访问 `states` 时出现 `KeyError`，或 `update_state_names` 未记录预期的变量。

**修复：** 使用与模块层次结构匹配的点号名称：

```python
model = RecurrentNN(
    neuron=neuron,
    synapse=psc,
    update_state_names=("neuron.v", "synapse.psc"),
)
```

你可以通过以下方式检查有效名称：

```python
print(functional.named_hidden_states(model).keys())
```

## 4. 检查点中缺少记忆重置值

**症状：** 加载检查点后，神经元重置为出厂默认值，而非训练后的初始化值。

**修复：** 显式保存和恢复 `_memories_rv`：

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "memories_rv": functional.named_memory_reset_values(model),
}

# 加载
model.load_state_dict(ckpt["model_state_dict"], strict=False)
functional.set_memory_reset_values(model, ckpt["memories_rv"])
```

完整示例请参阅 [教程 2：训练 SNN](tutorials/training.md)。

## 5. OmegaConf `_type_` 用法和 CLI 语法

**症状：** 在命令行传递变体配置时出现 `TypeError` 或 `ValidationError`。

**修复：** 使用 `_type_` 键来切换联合变体：

```bash
python train.py optimizer="{_type_: SGDConf, lr: 0.01, momentum: 0.95}"
```

或使用嵌套键：

```bash
python train.py optimizer._type_=SGDConf optimizer.lr=0.01
```

完整模式请参阅 [配置指南](guides/configuration.md)。

## 6. `torch.compile` 与动态缓冲区

**症状：** `torch.compile` 在基于循环缓冲区的历史记录模型上失败。

**修复：** 为训练编译时，在 `SpikeHistory` 或 `DelayedSynapse` 中设置 `use_circular_buffer=False`：

```python
from btorch.models.history import SpikeHistory
history = SpikeHistory(n_neuron=100, max_delay_steps=5, use_circular_buffer=False)
```

这以牺牲内存效率为代价换取完全的 `torch.compile` 兼容性。
