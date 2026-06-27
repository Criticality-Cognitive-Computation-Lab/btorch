# Installation

## pip / uv

Install the latest released `btorch` from PyPI:

```bash
pip install btorch
```

or

```bash
uv pip install btorch
```

### CUDA support

`btorch` depends on PyTorch. PyPI ships CPU-only torch by default. For CUDA,
install PyTorch with the right compute platform **first**, then add `btorch`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu1xx
pip install btorch
```

Or with `uv`:

```bash
uv pip install torch --torch-backend auto
uv pip install btorch
```

See the [PyTorch Get Started](https://pytorch.org/get-started/locally/) page for
other CUDA / ROCm variants.

## conda / mamba

`environment.yml` bundles PyTorch with CUDA, `pytorch_sparse`, and all heavy
dependencies via conda-forge and PyG channels:

```bash
conda env create -n btorch -f https://github.com/Criticality-Cognitive-Computation-Lab/btorch/raw/refs/heads/main/environment.yml
conda activate btorch
```

Or with `mamba`:

```bash
mamba env create -n btorch -f https://github.com/Criticality-Cognitive-Computation-Lab/btorch/raw/refs/heads/main/environment.yml
mamba activate btorch
```

## Install from source control

Btorch is fast evolving. For the latest unreleased changes, install directly from
the repository:

```bash
pip install git+https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
```

Gitee mirror alternative:

```bash
pip install git+https://gitee.com/alexfanqi/btorch.git
```

### Editable install (development)

Clone the repo and install in editable mode:

```bash
git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
cd btorch
pip install -e . --config-settings editable_mode=strict
```

If you use `uv`, clone and sync the lockfile:

```bash
git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
cd btorch
uv sync --group dev
source .venv/bin/activate
pip install -e . --config-settings editable_mode=strict
```

For CUDA with `uv`, install torch with the right backend first:

```bash
uv venv .venv-cuda
uv pip install torch --torch-backend auto --python .venv-cuda/bin/python
uv pip install -e . --python .venv-cuda/bin/python
```

## Optional: `torch_sparse` backend

Sparse linear layers default to PyTorch's native `torch.sparse` backend.
Install `torch_sparse` for better performance on large sparse networks.
Use prebuilt wheels from the [PyG repository](https://data.pyg.org/whl/)
matching your PyTorch and CUDA version:

```bash
# Example for PyTorch 2.7 with CUDA 12.6
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
```

If `torch_sparse` is absent, layers fall back to the native backend silently.

## Verify

```bash
python -c "import btorch; import torch; print(btorch.__version__, torch.__version__, torch.cuda.is_available())"
```
