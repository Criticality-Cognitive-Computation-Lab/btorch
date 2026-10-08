# Changelog

All notable changes to btorch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `btorch.sparse`: sparse arrays with a SciPy/PyTorch-like API (`Sparse`, `COO`,
  `CSR`, `CSC`, `sparse.from_edges`, `sparse.asarray`, `A @ x`, `sparse.stack`,
  `to_torch()` / `to_scipy()`); standard matrix orientation, conversions never
  transpose or densify.
- `btorch.models.connection`: `SparseConnection` (`from_adjacency`, `from_edges`,
  `from_hetersynapse`), `Synapse`, `EdgeWeight`, `ConstantWeight`,
  `ConstrainedWeight`; receptors and delays as edge attributes; batches of
  networks. Orientations are named `"pre_post"` (default of `from_adjacency`
  and `FromSparse`) and `"post_pre"`. Accessors `conn.pre`, `conn.post`,
  `conn.find_edges(pre, post)`, `conn.orientation`; `Projection.weight`,
  `edge_table()`, `to_sparse()`, `n_delay`, `n_receptor`.
- CUDA graph capture protocol: `conn.capture_version`,
  `conn.capture_incompatibility()`, `btorch.models.cudagraph.capture_versions`
  / `capture_incompatibilities`.
- `btorch.sparse.runtime`: execution layer (registered operators, backend
  registry, planner); not needed to write models. Triton pull backend
  (`csr_matvec`, `edge_grad`) as the default on CUDA when Triton is installed.
- Connection rules and `Projection`: NEST-style construction
  (`Projection(pre, post, rule, synapse)`; `OneToOne`, `AllToAll`,
  `FixedIndegree`, `FixedOutdegree`, `PairwiseBernoulli`, `DistanceDependent`,
  `FromEdges`, `FromSparse`).
- `btorch.sparse.operator`: matrix-free linear operators (`ConstantOperator`,
  `DiagonalOperator`, `LowRankOperator`, `ImplicitOperator`, lazy composites)
  and `StructuredConnection` / `ImplicitConnection` / `HybridConnection`.
- `sparse.einsum` (experimental): one N-D sparse operand with dense tensors,
  dense output.
- `HardDeepR`: fixed-slot hard Deep Rewiring for `SparseConnection`
  (`attach(optimizer)`); soft Deep R is not implemented.
- Guide: [Sparse Connectivity](guides/sparse_connectivity.md).

### Changed
- **Breaking** (relative to earlier revisions of the unreleased connection
  API; no aliases): orientation strings are `"pre_post"` / `"post_pre"`
  (`"src_dst"` / `"dst_src"` are gone); `FromSparse` defaults to `"pre_post"`;
  `conn.to_sparse()` defaults to the orientation the connection was built
  from; `set_edges_(slots, *, pre, post, ...)` is keyword-only;
  `HardDeepROptions.candidate` is called as `candidate(pre, post)`.
- **Breaking:** weights are bound by identity: a `Weight` module passed in
  `Synapse(weight=...)` is `conn.weight` (one module per connection); an
  `nn.Parameter` is adopted as `conn.weight.value` when the edge order is
  kept, otherwise copied with a warning. `Synapse(weight=<number>, dale=True)`
  and `Synapse(plasticity=...)` raise.
- **Breaking:** the planner distinguishes `"push"` (Triton on CUDA; not
  bitwise reproducible) from `"adaptive-push"` (reference backend; cannot be
  captured in a CUDA graph). `conn.value_version` now changes on optimizer
  steps. `btorch.sparse.runtime.__all__` no longer lists `ops` and
  `kernels_aten`. See the
  [guide](guides/sparse_connectivity.md#performance-hints-and-explain).
- **Breaking:** `SparseConn`, `SparseConstrainedConn`, `BaseSparseConn`,
  `SparseBackend`, `available_sparse_backends` and the `sparse_backend=` argument
  are removed from `btorch.models.linear` without a compatibility layer. Use
  `SparseConnection.from_adjacency(conn, Synapse(dale=...))`; Dale's law is now
  opt-in and the `state_dict` keys changed; checkpoints of the removed layers
  are refused with a migration message. See the
  [migration table](guides/sparse_connectivity.md#migration-from-sparseconn).
- **Breaking:** a spike delivered at step `t` now affects the PSC returned at step
  `t` for every PSC type (`AlphaPSC`, `AlphaPSCBilleh`, `DualExponentialPSC`
  previously took one extra `dt`).
- **Breaking:** `GLIF3.forward_exact_no_spike(x, t=None, v0=None, Iasc0=None,
  t_mode="homo")` is a pure function (it no longer updates state); the elementwise
  primitive `exact_no_spike_at(x, t, v0, Iasc0)` supports batched heterogeneous
  times and states, e.g. for iterative root finding. `t_mode="heter"` selects
  per-element times.
- **Breaking:** plotting and fitting options are grouped in dataclasses
  (`TbpttConfig`, `GlobalSearchConfig`, `StagedConfig`, `FitLossConfig`, raster
  options); see the [analysis](analysis.md) and [visualisation](visualisation.md)
  pages.
- **Breaking:** the options of `memories_to_xarray` / `save_memories_to_xarray`
  are grouped into `DimLayout`, `SparseOptions` and `ZarrStoreOptions`
  (`btorch.io`); `neuron_ids` and `partial_map` are keyword-only.
- **Breaking:** hex `scatter` / `quiver` take `HexGeometry`, `HexColorMap`,
  `HexPatchStyle` and `HexReference`; `plot_grouped_spectrum` takes
  `SpectrumGrouping` and `SpectrumStyle`. `quiver` now honours `rotation_deg` and
  the colour limits.
- **Breaking:** package layout follows layers. `btorch.datasets.noise` ->
  `btorch.models.noise`; `btorch.datasets.transforms` ->
  `btorch.utils.hex.augment`; `btorch.analysis.two_compartment_fit` ->
  `btorch.fitting.two_compartment` and `btorch.analysis.tuning` ->
  `btorch.fitting.tuning` (the fitting names are no longer re-exported from
  `btorch.analysis`). The `btorch.datasets` package is removed.
- Analysis naming rule: estimators/pipelines are `compute_*`. Renamed
  `branching_ratio` -> `compute_branching_ratio`, `get_slopes` ->
  `compute_lagged_slopes`, `get_continuous_spiking_rate` ->
  `compute_continuous_spiking_rate`, `voltage_overshoot` ->
  `compute_voltage_overshoot`.
- Removed `plot_group_violin`, `plot_group_box` and `plot_group_ecdf`; use
  `plot_group_distribution(..., kind="violin" | "box" | "ecdf")`.
- Heavy dependencies are optional extras (`io`, `config`, `fast`, `gpu`,
  `analysis`, `viz`, `sparse`, `examples`, `all`); see
  [installation](installation.md).
- `fano_population` / `kurtosis_population` are annotated as `StatsResult`;
  multi-output decorated analysis functions (`compute_lag_correlation`, E/I
  balance) are annotated as `MultiStatsResult` (new, in
  `btorch.analysis.statistics`).
- `make_hetersynapse_constraint` builds its constraint key in one place; results
  are unchanged.

### Fixed
- Dale signs and constraint structure of sparse connections are saved in the
  `state_dict`; they were stale after `load_state_dict`.
- `torch.compile(conn, fullgraph=True)` works for sparse connections without
  `torch_sparse`.
- The backward pass of the PyTorch-only sparse path no longer runs out of memory
  at about 100k neurons.
- `make_hetersynapse_conn` delay handling.
- `plot_multiscale_fano` with an unsupported `group_by` now raises `ValueError`
  instead of `NameError`.
- The power-law scaling fit reports `r_squared = NaN` for constant input instead
  of dividing by zero.

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
