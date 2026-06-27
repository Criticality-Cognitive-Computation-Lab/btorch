# Btorch

<h4 align="center">
    <p>
        <a href="README.md">English</a> |
        <b>简体中文</b>
    </p>
</h4>

<p align="center">
  <a href="https://pypi.org/project/btorch/"><img src="https://img.shields.io/pypi/v/btorch?label=PyPI" alt="PyPI 版本"></a>
  <a href="https://pypi.org/project/btorch/"><img src="https://img.shields.io/pypi/pyversions/btorch" alt="Python 版本"></a>
  <a href="https://github.com/Criticality-Cognitive-Computation-Lab/btorch/actions/workflows/ci.yml"><img src="https://github.com/Criticality-Cognitive-Computation-Lab/btorch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/Criticality-Cognitive-Computation-Lab/btorch"><img src="https://codecov.io/gh/Criticality-Cognitive-Computation-Lab/btorch/branch/main/graph/badge.svg" alt="覆盖率"></a>
  <a href="https://criticality-cognitive-computation-lab.github.io/btorch/"><img src="https://img.shields.io/badge/docs-live-brightgreen" alt="文档"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="许可证"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

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

## 安装

### pip / uv

从 PyPI 安装最新发布版：

```bash
pip install btorch
```

或

```bash
uv pip install btorch
```

### conda / mamba

```bash
conda env create -n ENV_NAME -f https://github.com/Criticality-Cognitive-Computation-Lab/btorch/raw/refs/heads/main/environment.yml
```

或

```bash
mamba env create -n ENV_NAME -f https://github.com/Criticality-Cognitive-Computation-Lab/btorch/raw/refs/heads/main/environment.yml
```

### 从版本控制安装

Btorch 迭代快速。如需最新的未发布改动，直接从仓库安装：

```bash
pip install git+https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
```

Gitee 镜像：

```bash
pip install git+https://gitee.com/alexfanqi/btorch.git
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

`skills/` 目录包含与 AI 智能体一起使用 btorch 时的使用模式和技巧。
通过 `npx skills` 安装：

```bash
npx skills add https://github.com/Criticality-Cognitive-Computation-Lab/btorch/tree/main/skills/btorch-snn-modelling
```

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
