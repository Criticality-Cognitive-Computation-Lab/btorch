# 安装

## pip / uv

从 PyPI 安装最新发布版：

```bash
pip install btorch
```

或

```bash
uv pip install btorch
```

### CUDA 支持

`btorch` 依赖 PyTorch。PyPI 默认提供 CPU 版本的 torch。如需 CUDA，请**先**安装对应计算平台版本的 PyTorch，再安装 `btorch`：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu1xx
pip install btorch
```

或使用 `uv`：

```bash
uv pip install torch --torch-backend auto
uv pip install btorch
```

详见 [PyTorch Get Started](https://pytorch.org/get-started/locally/) 页面了解其他 CUDA / ROCm 版本。

## conda / mamba

`environment.yml` 通过 conda-forge 和 PyG 通道捆绑了含 CUDA 的 PyTorch、`pytorch_sparse` 及所有依赖：

```bash
conda env create -n btorch -f https://github.com/Criticality-Cognitive-Computation-Lab/btorch/raw/refs/heads/main/environment.yml
conda activate btorch
```

或使用 `mamba`：

```bash
mamba env create -n btorch -f https://github.com/Criticality-Cognitive-Computation-Lab/btorch/raw/refs/heads/main/environment.yml
mamba activate btorch
```

## 从版本控制安装

Btorch 迭代快速。如需最新的未发布改动，直接从仓库安装：

```bash
pip install git+https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
```

Gitee 镜像：

```bash
pip install git+https://gitee.com/alexfanqi/btorch.git
```

### 可编辑安装（开发）

克隆仓库并以可编辑模式安装：

```bash
git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
cd btorch
pip install -e . --config-settings editable_mode=strict
```

若使用 `uv`，克隆并同步 lockfile：

```bash
git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
cd btorch
uv sync --group dev
source .venv/bin/activate
pip install -e . --config-settings editable_mode=strict
```

CUDA + `uv`，先安装对应后端版本的 torch：

```bash
uv venv .venv-cuda
uv pip install torch --torch-backend auto --python .venv-cuda/bin/python
uv pip install -e . --python .venv-cuda/bin/python
```

## 可选：`torch_sparse` 后端

稀疏线性层默认使用 PyTorch 原生 `torch.sparse` 后端。安装 `torch_sparse` 可在大型稀疏网络上获得更好性能。
使用与 PyTorch 和 CUDA 版本匹配的 [PyG 仓库](https://data.pyg.org/whl/) 预编译 wheel：

```bash
# 以 PyTorch 2.7 + CUDA 12.6 为例
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
```

若未安装 `torch_sparse`，层将静默回退到原生后端。

## 验证安装

```bash
python -c "import btorch; import torch; print(btorch.__version__, torch.__version__, torch.cuda.is_available())"
```