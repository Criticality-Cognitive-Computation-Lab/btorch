# Btorch 文档

**Btorch** 是一个用于类脑研究的启发式 Torch 库，提供有状态的神经元模型、连接组工具和分析工具。

## 概览

Btorch 提供：

- **神经元模型**：兼容 torch.compile 的 LIF、ALIF、GLIF3、Izhikevich 神经元
- **连接组工具**：稀疏连接矩阵、兼容 Flywire 的数据处理
- **分析**：脉冲序列分析、动态指标、统计工具
- **替代梯度**：用于脉冲神经网络的自定义梯度函数

## 核心特性

- 异构神经元参数
- 针对有状态模块的增强型形状/数据类型检查
- 兼容 torch.compile 和 ONNX
- 支持梯度检查点和截断时间反向传播
- 支持稀疏连接矩阵

## 安装

从源码安装：

```bash
git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
cd btorch
pip install -e . --config-settings editable_mode=strict
```