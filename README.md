# Btorch

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="README.zh.md">简体中文</a>
    </p>
</h4>

<p align="center">
  <a href="https://pypi.org/project/btorch/"><img src="https://img.shields.io/pypi/v/btorch?label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/btorch/"><img src="https://img.shields.io/pypi/pyversions/btorch" alt="Python versions"></a>
  <a href="https://github.com/Criticality-Cognitive-Computation-Lab/btorch/actions/workflows/ci.yml"><img src="https://github.com/Criticality-Cognitive-Computation-Lab/btorch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/Criticality-Cognitive-Computation-Lab/btorch"><img src="https://codecov.io/gh/Criticality-Cognitive-Computation-Lab/btorch/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://criticality-cognitive-computation-lab.github.io/btorch/"><img src="https://img.shields.io/badge/docs-live-brightgreen" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

Brain-inspired differentiable PyTorch toolkit for neuromorphic and computational
neuroscience research.

Use `btorch` if you need:

- Recurrent SNN modelling
- stateful neuron/synapse modules with explicit memory handling
- practical support for sparse/connectome-style network structure
- torch native training features (`torch.compile`, checkpointing,
  truncated BPTT)
- solid runtime performance and ONNX export support
- connectome import/export via SONATA, and flexible network definition coming soon  

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

### pip/uv

Install the latest released package from PyPI:

```bash
pip install btorch
```

or

```bash
uv pip install btorch
```

### conda or mamba

```bash
conda env create -n ENV_NAME -f https://github.com/Criticality-Cognitive-Computation-Lab/btorch/raw/refs/heads/main/environment.yml
```

### Install from source control

Btorch is fast evolving. If you want the latest unreleased changes, install directly from the repository:

```bash
pip install git+https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
```

Gitee mirror alternative:

```bash
pip install git+https://gitee.com/alexfanqi/btorch.git
```

For setup instructions, see [docs/installation.md](docs/en/docs/installation.md).  
For development workflow and contributing guidelines, see [docs/development.md](docs/en/docs/development.md).

## Documentation

**Live docs:** [https://criticality-cognitive-computation-lab.github.io/btorch/](https://criticality-cognitive-computation-lab.github.io/btorch/)

Documentation is built with **Zensicle** and **mkdocstrings** for API
auto-generation from docstrings.

Build locally:

```bash
python scripts/docs.py command=build-all
```

The generated site is written to `site/`.

Preview a specific language:

```bash
python scripts/docs.py command=live language=en
```

If you want a clean rebuild:

```bash
rm -rf site/
python scripts/docs.py command=build-all
```

## Skills

The `skills/` directory contains usage patterns and tips for using btorch with
AI agents. Install them with `npx skills`:

```bash
npx skills add https://github.com/Criticality-Cognitive-Computation-Lab/btorch/tree/main/skills/btorch-snn-modelling
```

## Road Map

- [x] support multi-dim batch size and neuron
- [ ] cleaner connectome import, network param management and manipulation lib
  - [ ] support full SONATA format (both [BlueBrain](https://github.com/openbraininstitute/libsonata.git) and [AIBS](https://github.com/AllenInstitute/sonata) variants)
  - [ ] flexible like [neuroarch](https://github.com/fruitflybrain/neuroarch.git) and tiny to integrate. thinking about using DuckDB
- [ ] verify numerical accuracy. align with Neuron and Brainstate
- [ ] support automatic conversion between stateful and pure functions
  - similar to make_functional in [torchopt](https://github.com/metaopt/torchopt)
  - [ ] consider migrate to pure memory states instead of register_memory. gradient checkpointing + torch.compile struggles with mutating self
- [ ] sparse matrix multiplication optimisation on GPU
- [ ] large scale multi-device training and simulation
  - [ ] integrate large-scale training support with [torchtitan](https://github.com/pytorch/torchtitan.git)
  - [ ] work distribution and balancing
- [ ] compat with [neurobench](https://github.com/NeuroBench/neurobench.git), [Tonic](https://tonic.readthedocs.io/en/latest/)
- [ ] [NIR](https://github.com/neuromorphs/NIR.git) import and export

## Design and Development Principles

- provide solid foundation of stateful Modules
- usability over performance, simple over easy, and customizability over abstractions
  - single file/folder principle on network model
  - see [Diffusers' philosophy](https://github.com/huggingface/diffusers/blob/main/PHILOSOPHY.md)
  - WIP to align current implementation with these principles

## Contributors

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
