---
hide:
  - navigation
  - toc
---

# Btorch

**一个用于神经形态研究的大脑启发式 Torch 库。**

![Btorch Overview](assets/images/btorch-overview.png)
*[@msy79lucky](https://github.com/msy79lucky) 绘制

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } __神经元模型__

    ---

    兼容 `torch.compile` 的 LIF、ALIF、GLIF3 和 Izhikevich 神经元，
    支持异构参数。

    [:octicons-arrow-right-24: 快速开始](quickstart.md)

-   :material-graph-outline:{ .lg .middle } __连接组工具__

    ---

    稀疏连接矩阵、延迟扩展，
    以及兼容 Flywire 的数据处理。

    [:octicons-arrow-right-24: 连接转换](connection_conversion.md)

-   :material-chart-line:{ .lg .middle } __分析与可视化__

    ---

    脉冲序列分析、动态指标，
    以及面向大规模模拟的绘图工具。

    [:octicons-arrow-right-24: 分析](analysis.md)

-   :material-book-open-variant:{ .lg .middle } __教程__

    ---

    逐步指导构建 RSNN、训练 SNN，
    以及使用数据类优先配置。

    [:octicons-arrow-right-24: 教程](tutorials/training.md)

</div>

## 安装

从源码安装：

```bash
git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
cd btorch
pip install -e . --config-settings editable_mode=strict
```

## 核心特性

- **有状态模块**：为脉冲神经元提供内置的记忆管理
- **形状安全**：针对标量和异构参数增强数据类型和维度处理
- **`torch.compile` 就绪**：兼容 PyTorch 2.x 编译
- **稀疏连接**：对大型稀疏矩阵提供一流支持
- **截断时间反向传播**：轻松截断长序列的梯度
