# 快速入门

## 安装

由于 `btorch` 尚未发布到 PyPI，请从源码安装：

```bash
git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
cd btorch
pip install -e . --config-settings editable_mode=strict
```

## 神经元基础用法

创建并运行一个简单的 LIF 神经元：

```python
import torch
from btorch.models.neurons import LIF

# 创建单个 LIF 神经元
neuron = LIF(n_neuron=100, tau=20.0, v_threshold=1.0)

# 运行 100 个时间步
spikes = []
for t in range(100):
    # 随机输入电流
    x = torch.randn(100) * 0.5
    spike = neuron(x)
    spikes.append(spike)

# 将脉冲堆叠为张量 [time, neurons]
spike_train = torch.stack(spikes)
```

## 使用异构参数

神经元支持为每个神经元设置独立的参数：

```python
import torch
from btorch.models.neurons import LIF

# 每个神经元都有自己的时间常数
taus = torch.rand(100) * 30 + 10  # 范围在 10-40 ms

neuron = LIF(n_neuron=100, tau=taus, v_threshold=1.0)
```

## 基础分析用法

分析脉冲序列：

```python
import numpy as np
from btorch.analysis import isi_cv, fano, firing_rate

# 生成示例脉冲数据 [time, batch, neurons]
spike_data = np.random.rand(1000, 10, 50) > 0.95

# ISI（脉冲间隔）的变异系数
cv, isi_total, isi_stats = isi_cv(spike_data, dt_ms=1.0)

# Fano 因子（脉冲计数的方差/均值）
fano_values, fano_stats = fano(spike_data, window_ms=100, dt_ms=1.0)

# 通过卷积计算发放率
rates = firing_rate(spike_data, dt_ms=1.0, smooth_ms=50)
```

## 形状约定

- 输入形状：`(*batch, n_neuron)`，其中 `*batch` 可以是任意维数
- `n_neuron` 以元组形式存储；使用 `.size` 获取神经元总数
- 对于多维批次设置，请使用 `init_net_state(..., batch_size=(...))`