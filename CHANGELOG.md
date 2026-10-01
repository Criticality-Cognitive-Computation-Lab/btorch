# Changelog

All notable changes to btorch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **CUDA graph capture for RNNs** (`RecurrentNN(cudagraph=True)`) — collapses the
  many small per-step launches of the recurrent time loop into a single graph
  replay, a one-flag speedup for launch-bound inference. Guarded to inference
  (capture cannot record autograd) and composes with `cpu_offload` and
  `torch.compile`. Capture lives in `btorch.models.cudagraph` (`CudaGraphRunner`).
  See [`examples/train_cudagraph.py`](examples/train_cudagraph.py) for inference and
  a by-hand whole-step training capture, and
  `benchmarks/rnn/test_compile_strategies.py` for the compile-strategy comparison.
- **`MemoryModule.reset(inplace=True)`** — resets hidden state into the existing
  buffers (host-free `zero_()` fast path) so state tensors keep fixed addresses,
  as CUDA graph capture requires.
- **`set_hidden_states(inplace=True)` and `named_hidden_states(clone=True)`** —
  `inplace` copies state into the existing buffers (fixed addresses, for restoring
  state in a captured inference graph); `clone` snapshots state decoupled from the
  live buffers (e.g. a start state to restore each step).

### Changed

- **Surrogate autograd functions are `torch.func`-compatible** — `_SurrogateAutograd`
  and the Poisson random spike function use `setup_context`/`jvp`/
  `generate_vmap_rule`, so `torch.func.jvp/vjp/vmap` work through them.
- **Analysis failures no longer return error strings or fake zeros** —
  `compare_fano_methods` and `calculate_gain_stability_sensitivity` return NaN and
  log a warning; `save_yaml` fails explicitly; `fano_operational_time` rejects a
  non-zero `overlap` instead of ignoring it.
- `FiringRateLoss` now defaults to a valid `loss_type` (`"huber_pinball"`).

### Fixed

- `is_broadcastable` now checks "first broadcasts to second" on shapes only.
- `MemoryModule.init_state`/`reset` no longer leak one memory's dtype/persistent/
  batch-size override into later memories; explicit `persistent=False` is honored.
- `load_config` now actually loads the config file found via `search_path` (it
  resolved the path and then loaded the original relative one), and raises
  `FileNotFoundError`/`ValueError` instead of a bare `assert`.
- `exp_euler_step` no longer returns NaN when the linear term is exactly zero.
- `init_net_state`/`reset_net` no longer raise `AttributeError` for plain (non-
  compiled) modules; they warn as intended.
- `SupportScaleState` guards now actually raise (`enforce="assert"` used to assert a
  non-empty string), and `scale_state` marks the module as scaled.
- `calculate_pcist` returns NaN with a warning (not a fake `0.0`) when the SVD fails.
- Argument validation in `btorch.analysis` and `connectome.augment` raises
  `ValueError` instead of a bare `assert` (which `python -O` strips).
- `make_hetersynapse_conn` docstring default for `n_delay_bins` matches the signature.
- `scale_state_` returns `(scale, zeropoint)` on its early-return paths too.
- `btorch.config` no longer imports `distutils` (removed in Python 3.12).
- `calculate_gain_stability_sensitivity` imported a nonexistent `model` package.
- Declared `h5py`, `pyyaml` and `fastdtw`; dropped unused `spikingjelly` and
  `typing-extensions`; aligned the pandas minimum across manifests.

### Removed

- `btorch.config.SPARSE_BACKEND` (no consumers).
- `btorch.io.dict_to_xarray`, `xarray_to_dict`, `save_dict_to_xarray`,
  `load_dict_from_xarray` (unused aliases of the `memories_*` functions).
- `btorch.models.parametrize` (unused; superseded by `constrain`).
- `maxslopes` argument of `branching_ratio`.
- `stateful` argument of `PoissonNoiseLayer` (it was accepted but ignored).

- Validation of user input in `btorch.models` (`MemoryModule`, regularizers, scale,
  linear), `PoissonNoiseLayer` and `connectome.connection` raises `ValueError`/
  `KeyError`/`TypeError` instead of a bare `assert`; `memories_to_xarray` rejects a
  `partial_map` entry with no neuron dimensions instead of ignoring it.
- `SupportScaleState` supports `enforce="repeated"`; `scale_func`/`unscale_func`
  return their input under `enforce="ignore"`.

### Internal

- `plot_raster`/`plot_neuron_traces` are split into private helpers (public
  signatures unchanged) and pinned by characterization tests; the two-compartment
  fit routines share one `FitLossConfig` and a `_prepare_sweep` helper, and the
  plotting function lives in `btorch/analysis/_two_compartment_plots.py`
  (re-exported from `two_compartment_fit`).

- `RecurrentNN` eager and CUDA-graph multi-step paths share chunking/offload
  helpers; the CUDA-graph compatibility rule lives in
  `RecurrentNNAbstract._cudagraph_incompatibilities()`; the hidden-state setters
  in `btorch.models.functional` no longer use callback parameters.
- Library `print()` calls became `warnings.warn`; broad `except Exception`
  blocks in analysis were narrowed to the expected exceptions.

## [0.1.0]

### Added
- **Two-compartment neuron** (`TwoCompartmentGLIF`) — soma-apical neuron with
  nonlinear apical plateau, bidirectional coupling, and optional adaptive
  threshold. See the [mixed neuron tutorial](docs/en/docs/tutorials/mixed_neurons.md).
- **Mixed neuron population** (`MixedNeuronPopulation`) — heterogeneous
  recurrent layer mixing multiple neuron types (e.g. GLIF3 + TwoCompartmentGLIF)
  with automatic current slicing and spike concatenation.
- **Hex grid module** (`btorch.utils.hex`) — coordinate systems (axial, doubled,
  zigzag, flywire), struct-of-arrays data types, convolution layers, eye
  rendering models, and SVG-based visualisation with overlays and compasses.
  See the [hex docs](docs/en/docs/hex.md).
- **Type annotations** — `btorch/py.typed` (PEP 561) and full return-type
  annotations across `btorch.analysis.spiking`, `btorch.models.neurons.two_compartment`,
  and `btorch.utils.hex`.
- **Release CI** — GitHub Actions workflow to build distributions on `v*` tags
  and publish to PyPI via trusted publishing (manual trigger only).
- **Codecov** — configuration file with coverage thresholds, flag management,
  and inline PR annotations.

### Changed
- **Build system migrated to uv** — `uv.lock` replaces pip lockfiles; CI
  uses `uv sync` with the PyTorch CPU index.
- **Documentation migrated to Zensicle** — replaced mkdocs/myst/sphinx with
  Zensicle + mkdocstrings. English and Chinese docs now built from the same
  pipeline with AI-assisted translation.
- **Conda environment** renamed from `dev-requirements.yaml` to `environment.yml`.

### Breaking Changes
All surrogate gradient derivatives have been renormalised so that
`g(v=0, damping_factor=1) == 1.0` for **any** value of `alpha`
(Zenke & Neftci 2021, *Neural Computation* 33(4)).

Previously, each derivative was scaled so that it integrated to 1 over
voltage — an analogy to probability densities. This turns out to be the
wrong invariant: what matters for stable learning is a unit response *at the
threshold*, not a unit integral.

| Surrogate   | Old peak (at v=0) | Factor applied | New peak |
|-------------|-------------------|----------------|----------|
| `Triangle`  | `alpha`           | `1/alpha`      | 1        |
| `Sigmoid`   | `alpha/4`         | `4/alpha`      | 1        |
| `Erf`       | `alpha/√π`        | `√π/alpha`     | 1        |
| `ATan`      | `alpha/2`         | `2/alpha`      | 1        |
| `ATanApprox`| `alpha/2`         | `2/alpha`      | 1        |

`SuperSpike` and the Heaviside forward pass are unaffected.

**Migration:** models trained with the above surrogates will see different
effective gradient magnitudes. Either retrain from scratch or multiply your
existing `damping_factor` by the inverse of the old peak to preserve magnitude
(e.g. for `ATan` at `alpha=2`, old peak 1.0, no change; at `alpha=4`, old
peak 2.0, set `damping_factor=2.0`).

All surrogate gradients have been reparametrised so that `alpha = 1/HWHM`
universally. The half-width at half-maximum of `g(v)` is now exactly `1/alpha`
for every surrogate (ATanApprox within ~8% due to rational approximation).

| Surrogate   | Internal constant  | New default α | Old default α |
|-------------|-------------------|---------------|---------------|
| `Triangle`  | k = 1/2           | 2.0           | 1.0 |
| `Sigmoid`   | k = 2ln(√2+1)≈1.763 | 2.0         | 1.0 |
| `Erf`       | k = √ln2≈0.833    | 4.0           | 2.0 |
| `ATan`      | k = 1 (was π/2)   | 2.0           | 2.0 |
| `ATanApprox`| k ≈ 1             | 2.0           | 2.0 |
| `SuperSpike`| k = √2−1≈0.414    | 2.0           | 4.0 |

**Migration:** if you relied on previous `alpha` values, the gradient width
at your old `alpha` is now different. Divide your old `alpha` by the constant
shown above to reproduce the old half-width. Retuning `alpha` with a sweep is
recommended.

### Removed
- **`pytorch_sparse` hard dependency** — sparse linear layers now default to
  PyTorch's native `torch.sparse` backend. `torch_sparse` remains available as
  an optional install for large-scale sparse network workloads.
- Sphinx, myst-parser, and obsolete pip lockfiles.
- AI agent prompt section from README (replaced with clean install instructions).
