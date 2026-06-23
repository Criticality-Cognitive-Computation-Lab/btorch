# Changelog

All notable changes to btorch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- **Two-compartment neuron** (`TwoCompartmentGLIF`): soma-apical neuron with nonlinear apical plateau, bidirectional coupling, and optional adaptive threshold. See [tutorial](tutorials/mixed_neurons.md).
- **Mixed neuron population** (`MixedNeuronPopulation`): single recurrent layer mixing multiple neuron types (e.g. GLIF3 + TwoCompartmentGLIF) with automatic current slicing and spike concatenation.
- **Heterogeneous RNN** (`HeteroRecurrentNN`): drop-in replacement for `RecurrentNN` that accepts a `MixedNeuronPopulation`.

### Changed
- Conda environment file renamed from `dev-requirements.yaml` to `environment.yml`.

### Breaking Changes
All surrogate gradient derivatives have been renormalised so that
`g(v=0, damping_factor=1) == 1.0` for **any** value of `alpha`
(Zenke & Neftci 2021, *Neural Computation* 33(4)).

Previously, each derivative was scaled so that it integrated to 1 over
voltage — motivated by an analogy to probability densities. This turns out
to be the wrong invariant: what matters for stable learning is a unit
response *at the threshold*, not a unit integral. Zenke & Neftci show that
this is the property that makes surrogate gradient learning robust across
network configurations.

The affected surrogates and their normalisation factors are:

| Surrogate   | Old peak (at v=0) | Factor applied | New peak |
|-------------|-------------------|----------------|----------|
| `Triangle`  | `alpha`           | `1/alpha`      | 1        |
| `Sigmoid`   | `alpha/4`         | `4/alpha`      | 1        |
| `Erf`       | `alpha/√π`        | `√π/alpha`     | 1        |
| `ATan`      | `alpha/2`         | `2/alpha`      | 1        |
| `ATanApprox`| `alpha/2`         | `2/alpha`      | 1        |

`SuperSpike` and the Heaviside forward pass are unaffected.

**Migration:** models trained with any of the above surrogates will see
different effective gradient magnitudes. Either retrain from scratch or
multiply your existing `damping_factor` by the inverse of the old peak
(e.g. for `ATan` at `alpha=2`, old peak was 1.0 so no change; at `alpha=4`,
old peak was 2.0, set `damping_factor=2.0` to preserve magnitude).

All surrogate gradients have been reparametrised so that `alpha = 1/HWHM`
universally. The half-width at half-maximum of `g(v)` is now exactly `1/alpha`
for every surrogate (ATanApprox within ~8% due to rational approximation).
Previously, equal `alpha` produced different gradient widths depending on the
surrogate—differences of up to 4× between surrogates. Now the same `alpha`
gives the same gradient half-width across all surrogates.

Each surrogate absorbs a surrogate-specific constant into its internal argument
to enforce this:

| Surrogate   | Internal constant  | Default alpha | Old default alpha |
|-------------|-------------------|---------------|-------------------|
| `Triangle`  | k = 1/2           | 2.0           | 1.0 |
| `Sigmoid`   | k = 2ln(√2+1)≈1.763 | 2.0         | 1.0 |
| `Erf`       | k = √ln2≈0.833    | 4.0           | 2.0 |
| `ATan`      | k = 1 (was π/2)   | 2.0           | 2.0 |
| `ATanApprox`| k ≈ 1             | 2.0           | 2.0 |
| `SuperSpike`| k = √2−1≈0.414    | 2.0           | 4.0 |

**Migration:** if you relied on the previous `alpha` values, the gradient
width at your old `alpha` is now different. Divide your old `alpha` by the
constant shown above to reproduce the old half-width (e.g. old `Triangle`
with `alpha=1` had HWHM=1; new `Triangle` needs `alpha=1` to get HWHM=1,
but the shape formula changed from `(1−|αv|)₊` to `(1−|αv|/2)₊`).
Retuning `alpha` with a sweep is recommended.
