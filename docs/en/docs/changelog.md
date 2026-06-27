# Changelog

All notable changes to btorch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

### Added
- **Two-compartment neuron** (`TwoCompartmentGLIF`) — soma-apical neuron with
  nonlinear apical plateau, bidirectional coupling, and optional adaptive
  threshold. See the [mixed neuron tutorial](tutorials/mixed_neurons.md).
- **Mixed neuron population** (`MixedNeuronPopulation`) — heterogeneous
  recurrent layer mixing multiple neuron types (e.g. GLIF3 + TwoCompartmentGLIF)
  with automatic current slicing and spike concatenation.
- **Heterogeneous RNN** (`HeteroRecurrentNN`) — replacement for `RecurrentNN`
  that accepts a `MixedNeuronPopulation`.
- **Hex grid module** (`btorch.utils.hex`) — coordinate systems (axial, doubled,
  zigzag, flywire), struct-of-arrays data types, convolution layers, eye
  rendering models, and SVG-based visualisation with overlays and compasses.
  See the [hex docs](hex.md).
- **Type annotations** — `btorch/py.typed` (PEP 561) and full return-type
  annotations across `btorch.analysis.spiking`, `btorch.models.neurons.two_compartment`,
  and `btorch.utils.hex`.
- **Release CI** — GitHub Actions workflow to build distributions on `v*` tags
  and publish to PyPI via trusted publishing (manual trigger only).
- **Codecov** — configuration file with coverage thresholds and inline PR
  annotations.

### Changed
- **Surrogate gradients reworked** — all surrogate derivatives now satisfy
  `g(v=0, damping_factor=1) == 1.0` for any `alpha` (Zenke & Neftci 2021),
  and `alpha = 1/HWHM` universally across all surrogates. Default `alpha` values
  updated. See the [surrogate gradients guide](concepts/surrogate_gradients.md)
  for migration instructions.
- **Build system migrated to uv** — `uv.lock` replaces pip lockfiles; CI
  uses `uv sync` with the PyTorch CPU index.
- **Documentation migrated to Zensicle** — replaced mkdocs/myst/sphinx with
  Zensicle + mkdocstrings. English and Chinese docs now built from the same
  pipeline with AI-assisted translation.
- **Conda environment** renamed from `dev-requirements.yaml` to `environment.yml`.
- **RNN classes renamed** — public export names cleaned up.

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
