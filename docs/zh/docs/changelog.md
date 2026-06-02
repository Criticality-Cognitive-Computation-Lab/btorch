# 更新日志

btorch 的所有重要变更都将记录在此文件中。

本格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)，
并且本项目遵循 [语义化版本控制](https://semver.org/spec/v2.0.0.html)。

## [未发布]

### 新增
- **双室神经元**（`TwoCompartmentGLIF`）：具有非线性顶端平台电位、双向耦合和可选自适应阈值的体树突神经元。参见[教程](tutorials/mixed_neurons.md)。
- **混合神经元群体**（`MixedNeuronPopulation`）：在单个循环层中混合多种神经元类型（如 GLIF3 + TwoCompartmentGLIF），支持自动电流切片与脉冲拼接。
- **异构 RNN**（`HeteroRecurrentNN`）：`RecurrentNN` 的替代实现，接受 `MixedNeuronPopulation`。
- **GeNN/CUDA 后端**（`btorch.backend`）：基于 GeNN 的实验性脉冲仿真后端，面向 GPU 加速的大规模网络。

### 变更
- conda 环境文件由 `dev-requirements.yaml` 重命名为 `environment.yml`。
