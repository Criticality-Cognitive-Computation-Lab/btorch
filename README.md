# Btorch

Heavily influenced by [brainstate](https://github.com/chaobrain/brainstate).
Evolved from [spikingjelly](https://github.com/fangwei123456/spikingjelly).
We thank the developers of both libraries for the inspirations.

**Enhancement from spikingjelly**:

- heterogenous parameters
- enhanced check of shape and dtype of register_memory
- torch.compile compatibility
- gradient checkpoint and truncated BPTT
- Sparse connectivity matrix
- More neuron and synapse models
- Memory state with static size and managed by torch buffer
  - onnx export is easy (note: sparse matrix is not supported by onnx)

## Installation

As `btorch` is not yet published to PyPI or Conda-forge, it must be installed from source. This approach also allows for rapid development, as any modifications to the code are immediately available.

### 1. Clone the Repository

```bash
git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
cd btorch
```

### 2. Set Up the Environment

We recommend using `conda` or `micromamba` with the provided environment file:

```bash
# Using Conda
conda env create -n ml-py312 --file=dev-requirements.yaml

# or using Micromamba
micromamba env create -n ml-py312 -f dev-requirements.yaml
```

#### Note on `pip` and `pytorch_sparse`

If you prefer using `pip` directly, installing `pytorch_sparse` from source or default pypi can be challenging. We recommend using prebuilt wheels from the [PyG repository](https://data.pyg.org/whl/) that match your PyTorch and CUDA installation:

```bash
# Example for PyTorch 2.8.0 with CUDA 12.8
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

### 3. Install in Editable Mode

Finally, install `btorch` in editable mode to ensure your local changes are reflected immediately:

```bash
pip install -e . --config-settings editable_mode=strict
```

## Development

Install precommit hooks for auto formatting.

PR without precommit formatting will not be accepted!

```{bash}
pre-commit install --install-hooks
```

Highly recommended to use [jaxtyping](https://docs.kidger.site/jaxtyping/) to mark expected array shape,
see [good example of using jaxtyping](https://fullstackdeeplearning.com/blog/posts/rwkv-explainer)

### run the tests

```bash
ruff check .
pytest tests
python -m sphinx.cmd.build docs docs/_build/html
```

## Documentation

Documentation is generated with **Sphinx** using API auto-generation from
docstrings (autodoc + autosummary).

Build locally with:

```bash
python -m sphinx.cmd.build docs docs/_build/html
```

The generated site is written to `docs/_build/html/`.

If you want a clean rebuild:

```bash
rm -rf docs/_build docs/api/generated
python -m sphinx.cmd.build docs docs/_build/html
```

## TODO List

- [x] support multi-dim batch size and neuron
- [ ] cleaner connectome import, network param management and manipulation lib
  - [ ] compat with bmtk
  - [ ] support full SONATA format (both [BlueBrain](https://github.com/openbraininstitute/libsonata.git) and [AIBS](https://github.com/AllenInstitute/sonata) variants)
  - [ ] flexible like [neuroarch](https://github.com/fruitflybrain/neuroarch.git) and tiny to integrate. thinking about using DuckDB
- [ ] verify numerical accuracy. align with Neuron and Brainstate
- [ ] support automatic conversion between stateful and pure functions
  - similar to make_functional in [torchopt](https://github.com/metaopt/torchopt)
  - [ ] consider migrate to pure memory states instead of register_memory. gradient checkpointing + torch.compile struggles with mutating self
- [ ] integrate large-scale training support with [torchtitan](https://github.com/pytorch/torchtitan.git)
- [ ] compat with [neurobench](https://github.com/NeuroBench/neurobench.git), [Tonic](https://tonic.readthedocs.io/en/latest/)
- [ ] [NIR](https://github.com/neuromorphs/NIR.git) import and export

## Design and Development Principles

- provide solid foundation of stateful Modules
- usability over performance, simple over easy, and customizability over abstractions
  - single file/folder principle on network model
  - see [Diffusers' philosophy](https://github.com/mreraser/diffusers/blob/fix-contribution.md/PHILOSOPHY.md)
  - WIP to align current implementation with these principles
