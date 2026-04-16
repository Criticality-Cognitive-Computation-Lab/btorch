# Btorch

Brain-inspired differentiable PyTorch toolkit for neuromorphic and computational
neuroscience research.

Use `btorch` if you need:

- Recurrent SNN modelling
- stateful neuron/synapse modules with explicit memory handling
- practical support for sparse/connectome-style network structure
- torchn native training features (`torch.compile`, checkpointing,
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

## 🤖 For AI Agents / Coding Assistants

**Copy and paste this prompt into your coding assistant:**

```text
Install `btorch` for this repository.

Before running commands, ask the user three things:
1. Does the user want `conda`/`micromamba` setup or `pip`-first setup?
2. Which environment name should be used? (default: `ml-py312`)
3. Do you want to install the forked version of omegaconf from https://github.com/alexfanqi/omegaconf? (default: yes)
   - Optional but recommended. It narrows the feature gap with Tyro, including dataclass union type, `Literal`, `Sequence`. Most importantly, it allows single source of truth, dataclass centric config and domain models.
   - If yes, install with: `pip install git+https://github.com/alexfanqi/omegaconf.git`
   - If no, the standard PyPI version will be used (some features may not work).

Then follow the matching path.

Path A - Conda or Micromamba (recommended):
- Create env from `dev-requirements.yaml` using the user-provided env name.
- Activate the environment.
- If user wants forked omegaconf: `pip install git+https://github.com/alexfanqi/omegaconf.git`
- Run: `pip install -e . --config-settings editable_mode=strict`

Path B - Pip-first:
- Create and activate a virtual environment.
- If user wants forked omegaconf: `pip install git+https://github.com/alexfanqi/omegaconf.git`
- If `torch_scatter`/`torch_sparse` fail from PyPI, install wheels that match
  both the installed PyTorch version and CUDA version from:
  `https://data.pyg.org/whl/` (for example,
  `https://data.pyg.org/whl/torch-<torch_version>+cu<cuda_version>.html`).
- Run: `pip install -e . --config-settings editable_mode=strict`

After install, verify with:
- `python -c "import btorch; print(btorch.__version__)"`

Report:
- chosen setup path
- environment name
- forked omegaconf choice
- install/verification output
- any follow-up actions needed
```

For setup instructions, see [docs/installation.md](docs/installation.md).  
For development workflow and contributing guidelines, see [docs/development.md](docs/development.md).

## Documentation

Documentation is built with **MkDocs Material** and **mkdocstrings** for API
auto-generation from docstrings.

Build locally:

```bash
python scripts/docs.py build-all
```

The generated site is written to `site/`.

Preview a specific language:

```bash
python scripts/docs.py live --language en
```

If you want a clean rebuild:

```bash
rm -rf site/
python scripts/docs.py build-all
```

## Skills

The `skills/` directory contains usage patterns and tips for using btorch with AI agent. These are provided as reference and may not represent optimal configurations for every use case.

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
  - see [Diffusers' philosophy](https://github.com/mreraser/diffusers/blob/fix-contribution.md/PHILOSOPHY.md)
  - WIP to align current implementation with these principles

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/alexfanqi"><img src="https://avatars.githubusercontent.com/u/8381176?s=100" width="100px;" height="100px;" alt="alexfanqi"/><br /><sub><b>alexfanqi</b></sub></a><br /><a href="https://github.com/Criticality-Cognitive-Computation-Lab/btorch/commits?author=alexfanqi" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/CFXTGJD"><img src="https://avatars.githubusercontent.com/u/97458246?s=100" width="100px;" height="100px;" alt="CFXTGJD"/><br /><sub><b>CFXTGJD</b></sub></a><br /><a href="https://github.com/Criticality-Cognitive-Computation-Lab/btorch/commits?author=CFXTGJD" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gaozh0814"><img src="https://avatars.githubusercontent.com/u/158576844?s=100" width="100px;" height="100px;" alt="gaozh0814"/><br /><sub><b>gaozh0814</b></sub></a><br /><a href="https://github.com/Criticality-Cognitive-Computation-Lab/btorch/commits?author=gaozh0814" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/msy79lucky"><img src="https://avatars.githubusercontent.com/u/166973717?s=100" width="100px;" height="100px;" alt="msy79lucky"/><br /><sub><b>msy79lucky</b></sub></a><br /><a href="https://github.com/Criticality-Cognitive-Computation-Lab/btorch/commits?author=msy79lucky" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yulaugh"><img src="https://avatars.githubusercontent.com/u/175782476?s=100" width="100px;" height="100px;" alt="yulaugh"/><br /><sub><b>yulaugh</b></sub></a><br /><a href="https://github.com/Criticality-Cognitive-Computation-Lab/btorch/commits?author=yulaugh" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
