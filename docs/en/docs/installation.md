# Installation

As `btorch` is not yet published to PyPI or Conda-forge, it must be installed from source. This approach also allows for rapid development, as any modifications to the code are immediately available.

## 1. Clone the Repository

```bash
git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
cd btorch
```

## 2. Set Up the Environment

We recommend using `conda` or `micromamba` with the provided environment file:

```bash
# Using Conda
conda env create -n ml-py312 --file=dev-requirements.yaml

# or using Micromamba
micromamba env create -n ml-py312 -f dev-requirements.yaml
```

### Forked OmegaConf (Optional but Recommended)

This repository supports an enhanced fork of OmegaConf from `https://github.com/alexfanqi/omegaconf`. The fork narrows the feature gap with Tyro by adding support for dataclass unions, `Literal`, and `Sequence` types (see [omegaconf#144](https://github.com/omry/omegaconf/issues/144), [omegaconf#1233](https://github.com/omry/omegaconf/pull/1233)), while preserving OmegaConf's single-source-of-truth config priority: dataclass defaults → config file → CLI overrides. The forked version is required for the `omegaconf-config` skill. To install it:

```bash
pip install git+https://github.com/alexfanqi/omegaconf.git
```

### Note on `pip` and `pytorch_sparse`

If you prefer using `pip` directly, installing `pytorch_sparse` from source or default pypi can be challenging. We recommend using prebuilt wheels from the [PyG repository](https://data.pyg.org/whl/) that match your PyTorch and CUDA installation:

```bash
# Example for PyTorch 2.8.0 with CUDA 12.8
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

## 3. Install in Editable Mode

Finally, install `btorch` in editable mode to ensure your local changes are reflected immediately:

```bash
pip install -e . --config-settings editable_mode=strict
```
