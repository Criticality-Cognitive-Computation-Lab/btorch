# 安装

由于 `btorch` 尚未在 PyPI 或 Conda-forge 上发布，必须从源码安装。这种方式也便于快速开发，因为对代码的任何修改都会立即生效。

## 1. 克隆仓库

```bash
git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
cd btorch
```

## 2. 环境配置

我们建议使用 `conda` 或 `micromamba` 以及提供的环境文件：

```bash
# 使用 Conda
conda env create -n ml-py312 --file=dev-requirements.yaml

# 或使用 Micromamba
micromamba env create -n ml-py312 -f dev-requirements.yaml
```

### Fork 版 OmegaConf（可选但推荐）

本仓库支持来自 `https://github.com/alexfanqi/omegaconf` 的增强版 OmegaConf。该 fork 版本通过增加对 dataclass unions、`Literal` 和 `Sequence` 类型的支持（参见 [omegaconf#144](https://github.com/omry/omegaconf/issues/144)，[omegaconf#1233](https://github.com/omry/omegaconf/pull/1233)），缩小了与 Tyro 的功能差距，同时保留了 OmegaConf 的单一事实来源配置优先级：dataclass 默认值 → 配置文件 → CLI 覆盖。`omegaconf-config` 功能需要此 fork 版本。安装方法如下：

```bash
pip install git+https://github.com/alexfanqi/omegaconf.git
```

### 关于 `pip` 和 `pytorch_sparse` 的说明

如果你倾向于直接使用 `pip`，从源码或默认 pypi 安装 `pytorch_sparse` 可能会遇到困难。我们建议从 [PyG 仓库](https://data.pyg.org/whl/) 使用与你的 PyTorch 和 CUDA 版本匹配的预编译 wheel 文件：

```bash
# 以 CUDA 12.8 对应的 PyTorch 2.8.0 为例
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

## 3. 以可编辑模式安装

最后，以可编辑模式安装 `btorch`，以确保你的本地修改能立即生效：

```bash
pip install -e . --config-settings editable_mode=strict
```