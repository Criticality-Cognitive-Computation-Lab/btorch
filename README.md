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

Until the package is published to PyPI/conda, install from source. The upside is that edits are immediately usable.

1) Clone:

```bash
git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
cd btorch
```

2) Create the dev environment (choose one):

```bash
conda env create -n ml-py312 --file=dev-requirements.yaml
# or
micromamba env create -n ml-py312 -f dev-requirements.yaml
```

3) Install in editable mode so local changes are importable right away:

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

Highly encouraged to put your prototyping work under [braintools-examples](https://github.com/Criticality-Cognitive-Computation-Lab/btorch-examples.git).
In light of the nature of fast changing prototyping common in machine learning, and to avoid duplicated work scattered among multiple branches,
we need to share "just-work" implementation as early as possible and iterate fast.

### run the tests

```bash
ruff check .
pytest tests
mkdocs build --strict
```

## Documentation

Docs are scaffolded with MkDocs (Material). Edit content under `docs/` and build locally with:

```bash
mkdocs serve
```

## TODO List

- [ ] support multi-dim batch size and neuron
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
