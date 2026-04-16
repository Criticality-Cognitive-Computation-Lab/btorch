# 技能参考

btorch 内置了编码规范使用模式的技能。如果你使用 AI 智能体，可以显式调用这些技能（例如"使用 btorch-snn-modelling 技能"）以获得经过验证的、具备上下文的帮助。

本页面总结了每个技能涵盖的内容，并链接到相关的源文件和示例。

## btorch-snn-modelling

**何时调用：** 每当你使用 btorch 构建或训练脉冲神经网络时。

**涵盖内容：**

- **有状态模块** — `MemoryModule`、`init_net_state`、`reset_net`、检查点保存
- **`dt` 环境** — `environ.context(dt=...)` 的用法
- **训练循环** — 纯 PyTorch 和 Lightning 集成
- **检查点** — 与 `state_dict()` 一起保存/加载 `memories_rv`
- **截断时间反向传播** — 用于长序列的 `detach_net`
- **常见陷阱** — 忘记 `dt`、错误的状态名、缺少重置

**关键参考：**

- 技能源文件：[`skills/btorch-snn-modelling/SKILL.md`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/skills/btorch-snn-modelling/SKILL.md)
- 完整训练循环：[`skills/btorch-snn-modelling/references/training_example.md`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/skills/btorch-snn-modelling/references/training_example.md)
- 纯 PyTorch 示例：[`examples/fmnist.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/examples/fmnist.py)
- Lightning 示例：[`examples/fmnist_lightning.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/examples/fmnist_lightning.py)
- 测试：[`tests/models/test_mem_load_save.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/tests/models/test_mem_load_save.py)

## omegaconf-config

**何时调用：** 当你需要结构化配置、CLI 覆盖或启动器到工作进程的选项转发时。

**涵盖内容：**

- **数据类优先配置** — 默认值存在于 Python 中，而非 YAML 中
- **组合** — 用于通用 + 任务特定设置的嵌套数据类
- **变体选择** — 使用 `_type_` 的数据类联合类型
- **选项转发** — 用于生成工作进程的 `to_dotlist`
- **差异工具** — 用于比较配置的 `diff_conf`

**关键参考：**

- 技能源文件：[`skills/omegaconf-config/SKILL.md`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/skills/omegaconf-config/SKILL.md)
- 工具：[`btorch/utils/conf.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/btorch/utils/conf.py)
- 测试：[`tests/utils/test_conf.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/tests/utils/test_conf.py)
- 指南：[配置指南](guides/configuration.md)

## 如何引用技能

在提示智能体时，请明确说明：

> "使用 btorch-snn-modelling 技能帮助我编写一个带截断时间反向传播的训练循环。"

> "使用 omegaconf-config 技能来设置一个带启动器到工作进程转发的批量参数扫描。"
