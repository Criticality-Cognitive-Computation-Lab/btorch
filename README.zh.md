# Btorch

<h4 align="center">
    <p>
        <a href="README.md">English</a> |
        <b>简体中文</b>
    </p>
</h4>

面向神经形态和计算神经科学研究的大脑启发式可微分 PyTorch 工具包。

如果你需要以下功能，请使用 `btorch`：

- 循环脉冲神经网络（SNN）建模
- 具有显式记忆管理的有状态神经元/突触模块
- 对稀疏/连接组风格网络结构的实用支持
- 原生 PyTorch 训练特性（`torch.compile`、检查点保存、截断时间反向传播）
- 稳定的运行时性能和 ONNX 导出支持
- 通过 SONATA 进行连接组导入/导出，以及即将推出的灵活网络定义

深受 [brainstate](https://github.com/chaobrain/brainstate) 影响，
从 [spikingjelly](https://github.com/fangwei123456/spikingjelly) 演化而来。
我们感谢这两个库的开发人员带来的启发。

**相较于 spikingjelly 的增强**：

- 异构参数
- 增强的 `register_memory` 形状和数据类型检查
- `torch.compile` 兼容性
- 梯度检查点和截断时间反向传播
- 稀疏连接矩阵
- 更多神经元和突触模型
- 具有静态大小并由 torch buffer 管理的记忆状态
  - 易于导出 onnx（注意：稀疏矩阵不受 onnx 支持）

## 🤖 面向 AI 智能体 / 编程助手

**将以下提示复制到你的编程助手中：**

```text
为该仓库安装 `btorch`。

在运行命令之前，先询问用户四件事：
1. 用户想要 `conda`/`micromamba` 安装还是优先 `pip` 安装？
2. 应该使用什么环境名称？（默认：`ml-py312`）
3. 是否想从 https://github.com/alexfanqi/omegaconf 安装 fork 版本的 omegaconf？（默认：是）
4. 是否需要可选的稀疏后端？它适合大型稀疏网络；提示：与 `torch.compile` 一起使用时需要 torch 2.8+。旧平台可能无法支持这么高的 PyTorch 版本。

然后按照对应的路径进行。

路径 A - Conda 或 Micromamba（推荐）：
- 使用用户提供的环境名称从 `dev-requirements.yaml` 创建环境。
- 激活环境。（conda 环境已包含 `pytorch_sparse`。）
- 如果用户想要 fork 版 omegaconf：`pip install git+https://github.com/alexfanqi/omegaconf.git`
- 运行：`pip install -e . --config-settings editable_mode=strict`

路径 B - 优先 Pip：
- 创建并激活虚拟环境。
- 如果用户想要 fork 版 omegaconf：`pip install git+https://github.com/alexfanqi/omegaconf.git`
- 可选：安装 `torch_sparse` 以获得更好的稀疏性能。
  使用与 torch/CUDA 版本匹配的 PyG 预编译 wheel：
  `pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-<version>+<cuda>.html`。
- 如果 PyG wheel 不可用，告诉用户并跳过`torch_sparse`。
- 运行：`pip install -e . --config-settings editable_mode=strict`

安装后，使用以下命令验证：
- `python -c "import btorch; print(btorch.__version__)"`

报告：
- 选择的安装路径
- 环境名称
- 是否选择 fork 版 omegaconf
- `torch_sparse` 选择和兼容性提示
- 安装/验证输出
- 任何需要的后续操作
```

安装说明请参阅 [docs/installation.md](docs/zh/docs/installation.md)。  
开发工作流和贡献指南请参阅 [docs/development.md](docs/zh/docs/development.md)。

## 文档

**在线文档：** [https://criticality-cognitive-computation-lab.github.io/btorch/](https://criticality-cognitive-computation-lab.github.io/btorch/)

文档使用 **MkDocs Material** 和 **mkdocstrings** 从文档字符串自动生成 API 文档构建而成。

本地构建：

```bash
python scripts/docs.py command=build-all
```

生成的站点写入 `site/`。

预览特定语言：

```bash
python scripts/docs.py command=live language=en
```

如果需要干净重建：

```bash
rm -rf site/
python scripts/docs.py command=build-all
```

## 技能

`skills/` 目录包含与 AI 智能体一起使用 btorch 时的使用模式和技巧。这些内容仅供参考，可能并不代表所有用例的最佳配置。

## 路线图

- [x] 支持多维批次大小和神经元
- [ ] 更清晰的连接组导入、网络参数管理和操作库
  - [ ] 支持完整 SONATA 格式（包括 [BlueBrain](https://github.com/openbraininstitute/libsonata.git) 和 [AIBS](https://github.com/AllenInstitute/sonata) 变体）
  - [ ] 像 [neuroarch](https://github.com/fruitflybrain/neuroarch.git) 一样灵活且易于集成。考虑使用 DuckDB
- [ ] 验证数值精度。与 Neuron 和 Brainstate 对齐
- [ ] 支持有状态函数和纯函数之间的自动转换
  - 类似于 [torchopt](https://github.com/metaopt/torchopt) 中的 make_functional
  - [ ] 考虑迁移到纯记忆状态而非 register_memory。梯度检查点 + torch.compile 与修改 self 存在冲突
- [ ] GPU 上的稀疏矩阵乘法优化
- [ ] 大规模多设备训练和模拟
  - [ ] 与 [torchtitan](https://github.com/pytorch/torchtitan.git) 集成大规模训练支持
  - [ ] 工作负载分布和平衡
- [ ] 兼容 [neurobench](https://github.com/NeuroBench/neurobench.git)、[Tonic](https://tonic.readthedocs.io/en/latest/)
- [ ] [NIR](https://github.com/neuromorphs/NIR.git) 导入和导出

## 设计与开发原则

- 为有状态模块提供坚实基础
- 可用性优先于性能，简单优先于易用，可定制性优先于抽象
  - 网络模型遵循单文件/文件夹原则
  - 参见 [Diffusers 的理念](https://github.com/huggingface/diffusers/blob/main/PHILOSOPHY.md)
  - 正在努力使当前实现与这些原则对齐

## 贡献者

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/alexfanqi"><img src="https://avatars.githubusercontent.com/u/8381176?s=100" width="100" height="100" alt="alexfanqi"/><br /><sub><b>alexfanqi</b></sub></a><br /><a href="https://github.com/Criticality-Cognitive-Computation-Lab/btorch/commits?author=alexfanqi" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/CFXTGJD"><img src="https://avatars.githubusercontent.com/u/97458246?s=100" width="100" height="100" alt="CFXTGJD"/><br /><sub><b>CFXTGJD</b></sub></a><br /><a href="https://github.com/Criticality-Cognitive-Computation-Lab/btorch/commits?author=CFXTGJD" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gaozh0814"><img src="https://avatars.githubusercontent.com/u/158576844?s=100" width="100" height="100" alt="gaozh0814"/><br /><sub><b>gaozh0814</b></sub></a><br /><a href="https://github.com/Criticality-Cognitive-Computation-Lab/btorch/commits?author=gaozh0814" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/msy79lucky"><img src="https://avatars.githubusercontent.com/u/166973717?s=100" width="100" height="100" alt="msy79lucky"/><br /><sub><b>msy79lucky</b></sub></a><br /><a href="https://github.com/Criticality-Cognitive-Computation-Lab/btorch/commits?author=msy79lucky" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yulaugh"><img src="https://avatars.githubusercontent.com/u/175782476?s=100" width="100" height="100" alt="yulaugh"/><br /><sub><b>yulaugh</b></sub></a><br /><a href="https://github.com/Criticality-Cognitive-Computation-Lab/btorch/commits?author=yulaugh" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
