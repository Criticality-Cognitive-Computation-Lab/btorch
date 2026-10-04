# Changelog

All notable changes to btorch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `tests/test_pipeline_e2e.py` — end-to-end example test chaining connectome ->
  sparse conn -> LIF/ExponentialPSC RNN -> spike analysis -> xarray round trip.
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

- **Breaking (visualisation):** `plot_raster` and `plot_neuron_traces` no longer
  take dozens of flat keyword arguments.
  - `plot_raster(spikes, *, dt, times, ax, title, xlabel, ylabel, style, grouping,
    strip, rate, annotations)` takes the new frozen dataclasses `RasterStyle`
    (`spike_color`, `marker`, `marker_size`, `neuron_specs`), `RasterGrouping`
    (`neurons_df`, `group_key`, `group_sort`, `sort_neurons`, `show_separators`,
    `separator_style`), `GroupStripOptions` (`show`, `color_key`, `cmap`, `layout`,
    `legend`, `label_mode`, `side`; replaces `show_group_strip`, `group_color_key`,
    `strip_cmap`, `group_strip_kwargs`, `group_strip_legend`, `group_label_mode`,
    `group_strip_side`), `RatePanelOptions` (`total`, `per_group`, `window_ms`;
    replaces `rate`, `group_rate`, `rate_window_ms`) and `RasterAnnotations`
    (`events`, `regions`, `show_tracks`, `event_kwargs`, `region_kwargs`), all
    exported from `btorch.visualisation.timeseries`. Every argument after
    `spikes` is keyword-only.
  - `plot_neuron_traces(states, format=None)` now takes only a `SimulationStates`
    (which gained `psc_labels`) and an optional `TracePlotFormat`; the plain
    `voltage=`, `dt=`, ... and `neuron_indices=`, ... keyword interface and the
    unused `neurons_df` argument are removed. `voltage` is a required field of
    `SimulationStates`.
  - The group-rate and strip error messages now name the option fields.
  - `_select_trace_neurons` no longer reseeds numpy's global RNG (same sample).
- `btorch.analysis.two_compartment_fit` is now a package (`data`, `loss`,
  `evaluation`, `report`, `fit`); all previously importable names are unchanged.
  `compute_ra` returns NaN (with a warning) instead of 0.0 for all-zero spike
  activity, matching the dynamic_tools failure-return convention. Docstrings of
  `btorch.io.serialization` and several analysis functions now document `Raises`
  and failure returns.
- **Breaking (optional dependencies):** `h5py`, `hdf5plugin`, `omegaconf` and
  `pyyaml` are no longer core dependencies. `h5py`/`hdf5plugin`
  (`btorch.utils.hdf5_utils`) moved into `btorch[io]`; `omegaconf`/`pyyaml`
  (`btorch.utils.conf`, `btorch.utils.yaml_utils`) form the new `btorch[config]`
  extra. Importing these modules without the library raises an `ImportError` with a
  `pip install "btorch[...]"` hint; `btorch.utils.file` (`fig_path`, `save_fig`) no
  longer needs OmegaConf (`cfg` accepts a `FigPathConfig` or any mapping). New
  extras `fast` (numba, polars) and `gpu` (triton, used by `do_bench(timing_method=
  "gpu")`) declare the existing silent-fallback accelerators; all are in `all`.
  The optional-import policy is documented in `btorch.utils._optional`.
- **Breaking (analysis / io review follow-up):**
  - `fit_two_compartment_model` no longer takes the flat method options `lr`,
    `epochs`, `chunk_size`, `param_bounds`, `global_maxiter`, `global_popsize`,
    `local_maxiter`, `seed`, `polish`, `stages`. Use the new frozen dataclasses
    `TbpttConfig(lr, epochs, chunk_size)` via `tbptt=`,
    `GlobalSearchConfig(param_bounds, maxiter, popsize, local_maxiter, seed,
    polish)` via `search=` and `StagedConfig(stages)` via `staged=`; a config the
    chosen `method` does not use raises `ValueError`. The private back-ends take
    only their own config. The model contract is now the exported
    `TwoCompartmentModel` protocol: `w_Ca` is read directly instead of
    `getattr(model, "w_Ca", None)`, so a model without `w_Ca` raises
    `AttributeError`.
  - `isi_cv_population`: the pooled-ISI helpers always return an ISI array (empty
    for fewer than two spikes) instead of a `(nan, {})` tuple in that case; with
    `stat=None` the result is therefore an empty array rather than `nan`.
    `compute_stats_batch` returns NaN for every stat of an empty input (no
    warnings or reduction errors). Docs and annotation now state that the default
    result is the scalar population CV.
  - `compute_lyapunov_exponent` (complexity) -> `compute_lyapunov_exponent_from_spikes`
    (the 1-D series estimator stays `compute_max_lyapunov_exponent`).
  - `analysis.tuning.get_fi_vi_curve` -> `compute_fi_vi_curve` (it builds and
    simulates a network) with typed parameters.
  - `get_continuous_spiking_rate` now returns a `torch.Tensor` for tensor input
    (was always `np.ndarray`); `compute_gain_stability_sensitivity` and
    `compute_lyapunov_exponent_from_spikes` convert internally.
  - `get_slopes` returns the `LaggedSlopes` named tuple (still unpackable as the
    former 7-tuple) with every member and NaN/0 sentinel documented;
    `compute_structural_eigenvalue_outliers` returns the `EigenvalueOutliers`
    TypedDict (`max_eigenvalue` is now documented) and `spectral_radius` is
    `float | None`; `compute_avalanche_statistics` returns `AvalancheStatistics`.
  - `fano_model_based` raises `ValueError` for `window <= overlap` (was a
    `ZeroDivisionError`/nonsense windows), matching `fano_mean_matching`.
  - `firing_rate(batch_axis=...)` is typed with the shared `BatchAxis` alias
    (`int | tuple[int, ...] | None`) like the other statistics; decorated
    statistics are annotated `StatsResult` (`tuple[Any, dict]`).
  - The duplicate `_to_numpy` helpers in `analysis.aggregation` and `io.serialization`
    are replaced by `btorch.utils.array.to_numpy(strict=True)` and the new
    `to_numpy_or_sparse` (keeps SciPy sparse, converts sparse torch tensors to
    SciPy COO, which previously called a nonexistent `Tensor.to_scipy`).
  - `memories_to_xarray` was split into `_expand_partial`, `_resolve_var_dims`,
    `_encode_variable` and `_root_id_entry` helpers (same output);
    `fano_model_based`, `fano_mean_matching` and `compute_avalanche_statistics`
    were split into shared window/prepare/compute helpers (outputs verified
    identical to the previous implementation).
- **Breaking (analysis / utils / io cleanup):**
  - Two-compartment fitting: `fit_two_compartment_model`, `two_compartment_loss`,
    `evaluate_two_compartment_fit` and `evaluate_fit_across_sweeps` no longer take
    the loose loss keywords (`voltage_weight`, `spike_weight`, `spike_count_weight`,
    `spike_timing_weight`, `spike_count_over_weight`, `spike_count_under_weight`,
    `sparsity_weight`, `spike_tau_ms`, `post_spike_mask_ms`,
    `spike_match_window_ms`, `spike_miss_penalty_ms`); pass one
    `loss=FitLossConfig(...)` instead (`FitLossConfig` is now exported from
    `btorch.analysis`). The evaluate helpers now honour every config field, so
    `total_loss` in their metrics uses the real weights (previously only a subset
    was forwarded and the other weights silently stayed at their defaults).
  - `btorch.analysis.dynamic_tools.spiking` -> `btorch.analysis.dynamic_tools.fano`
    (it only holds the rate-compensated Fano factor methods).
  - `btorch.analysis.dynamic_tools.micro_scale.compute_cv_isi` removed; use
    `btorch.analysis.isi_cv` (the single ISI-CV implementation). Neurons with
    exactly two spikes now give NaN instead of CV = 0 (a CV needs two ISIs).
  - `micro_scale.compute_spike_distance` returns NaN (was 0.0) when fewer than two
    neurons are given.
  - `firing_rate(axis=...)` -> `firing_rate(batch_axis=...)`.
  - `plot_micro_dynamics` lost its unused `ax` parameter.
  - Decorated analysis functions (`isi_cv`, `fano`, `kurtosis`, `local_variation`,
    `cv_temporal`, `fano_temporal`, `isi_cv_population`, `compute_eci`,
    `compute_lag_correlation`, `compute_ei_balance`) no longer accept (and
    silently ignore) arbitrary `**kwargs`; unknown keywords raise `TypeError`.
  - `use_stats` / `use_percentiles` are typed with `StatsDecorated` /
    `PercentilesDecorated` protocols (added kwargs and the `(*values, info)`
    return are visible to type checkers). `compute_stat` / `compute_stats_batch`
    share one implementation: `compute_stat` on a multi-element torch result now
    returns the tensor instead of raising, and `compute_stats_batch` raises
    `ValueError` for an unknown stat name.
  - Internal `_compute_stat`, `_compute_stats_batch` and `_compute_eci` forwarders
    removed.
  - `utils.conf.get_dotkey` no longer masks an `AttributeError` raised inside a
    property getter (it propagates); only genuinely missing segments return
    `default`.
  - Hetero spelling unified to `hetersynapse` in `connectome.connection`
    internals (`hetero_conn_df` -> `hetersynapse_conn_df`, column
    `post_hetero` -> `post_hetersynapse`) and in tests/docs.
- **New:** `btorch.utils.array.to_numpy` (shared tensor/array-like -> NumPy helper,
  replacing duplicate private `_to_numpy` copies in `visualisation`).
- **Breaking (models cleanup):**
  - `btorch.models.dlif` moved to `btorch.models.neurons.dlif`
    (`DendriticLIF`, `DLIF`, `DBNN` keep their names; still exported from
    `btorch.models` and `btorch.models.neurons`).
  - `MemoryModule.memories_rv` is now a read-only property (its setter duplicated
    `set_memories_rv()`).
  - Removed the deprecated `scale.SupportScaleState` and its wrappers
    `functional.scale_net`, `unscale_net`, `scale_state`, `unscale_state`
    (no users). `scale.scale_state_` is unchanged.
  - `environ.DEFAULT_SETTINGS` is now private and copy-on-write (lock-free,
    consistent reads under concurrent `set`); use `environ.set()`/`get()`/`all()`,
    and the new `environ.unset()` to drop a default.
  - `MemoryModule._batch_dim_detect` -> `_detect_batch_shape`; unused
    `_batch_dim_exist` removed. `RecurrentNNAbstract._process_small_chunk` ->
    `_run_unroll_block`, `_process_large_chunk_impl` -> `_run_chunk_steps`.
  - `MemoryModule.single_step_forward` (and `RecurrentNNAbstract`'s) now raise
    `NotImplementedError` instead of silently returning `None`;
    `RecurrentNNAbstract._detect_loop_args` raises `ValueError` rather than
    `assert` when no time dim can be inferred.
  - `BaseSparseConn.sparse_tensor` (native-backend cache) removed; see Fixed.
  - `constrain_net`'s parameter is named `net` (was `mod`).
- **Breaking: numerical results change — PSC spike timing.** For every PSC type a
  spike delivered to `single_step_forward` at step `t` now already changes the
  PSC returned at step `t` (first response at the delivery step, delay 0, as
  `ExponentialPSC` always did). `AlphaPSC`, `AlphaPSCBilleh` and
  `DualExponentialPSC` previously responded one `dt` late (`psc[0] == 0` for a
  spike at step 0). New impulse responses (`get_kernel` and
  `multi_step_forward` match `single_step_forward` exactly):
  - `ExponentialPSC`: `k[t] = a^t`, `a = exp(-dt/tau)` (unchanged).
  - `AlphaPSC`: `k[t] = g_max (t+1)(1-a) a^t` (was `t (1-a) a^(t-1)`, `k[0]=0`).
  - `AlphaPSCBilleh`: `k[t] = (t+1) (e/tau) a^(t+1)` (was `t (e/tau) a^t`); the
    unit-amplitude peak now falls at kernel index `tau-1`.
  - `DualExponentialPSC`: `k[t] = A' (a_d^(t+1) - a_r^(t+1))` (was
    `A' (a_d^t - a_r^t)`, `k[0]=0`).
  `DelayedPSC(max_delay_steps=d)` still shifts the base response by exactly `d`
  steps. `DBNN`/`DLIF`-style cells and any model using these PSCs fire one step
  earlier than before.
- **Breaking:** neuron constructors no longer have mutable / instance default
  arguments. `trainable_param`, `surrogate_function` (and GLIF3 `k`,
  `asc_amps`) default to `None` and are built per neuron, so two default
  neurons no longer share one surrogate `nn.Module` / one set.
- **Breaking:** `tau_ref` now defaults to `None` (refractory disabled, no
  `tau_ref` buffer or `refractory` state) in every neuron; `ELIF` and `GLIF3`
  previously defaulted to `0.0` (behaviourally identical, but it allocated
  state and a `tau_ref` entry in `state_dict`). Pass `tau_ref=0.0` explicitly
  to keep the old state layout.
- **Breaking:** `StepModule.supported_step_mode` is now a property, consistent
  with `supported_backends` (no call parentheses).
- `environ.set()` now really sets process-global defaults visible from every
  thread; `environ.context()` remains a thread-local override stack that wins
  over the defaults inside its `with` block. Previously defaults set in one
  thread were invisible to others. `environ.DEFAULT` is replaced by
  `environ.DEFAULT_SETTINGS`.
- **Breaking: heavy dependencies are now optional extras.** `plotly`, `xarray`,
  `zarr`, `numcodecs`, `powerlaw`, `nolds` and `fastdtw` are no longer installed
  with `pip install btorch`. Install `btorch[io]` (xarray, zarr, numcodecs),
  `btorch[analysis]` (powerlaw, nolds, fastdtw), `btorch[viz]` (networkx, plotly)
  or `btorch[all]`. They are imported lazily; using a feature without its extra
  raises an `ImportError` with the matching install command.
- **Breaking: analysis / io / visualisation / utils API made consistent.** No
  aliases are kept; update call sites.
  - Time step: `dt_ms` -> `dt` (milliseconds) in `isi_cv`, `isi_cv_population`,
    `cv_temporal`, `local_variation`, `fano_operational_time`,
    `compare_fano_methods`, and in `btorch.analysis.two_compartment_fit`
    (`AllenSweepBatch.dt_ms`, `FitEvaluation.dt_ms`, `load_allen_sweep`,
    `exponential_filter_spike_train`, `spike_timing_stats`, loss helpers;
    `resample_trace(source_dt_ms, target_dt_ms)` -> `(source_dt, target_dt)`).
    The `dt_ms` key in the saved fit report JSON is unchanged.
  - Spike input: `spike` (`fano`, `kurtosis`, `fano_population`,
    `kurtosis_population`, `fano_temporal`, `fano_sweep`) and `spike_data`
    (`isi_cv`, `isi_cv_population`, `cv_temporal`, `local_variation`, all
    `btorch.analysis.dynamic_tools.spiking` functions) -> `spikes`.
  - `batch_axis` is annotated `int | tuple[int, ...] | None` everywhere and an
    `int` is now accepted by `fano_temporal`, `local_variation`,
    `fano_operational_time`, `fano_mean_matching` and `fano_model_based`.
    `fano_operational_time(overlap=...)` is annotated `int | None`.
  - `btorch.analysis.dynamic_tools`: `calculate_*` -> `compute_*`
    (`calculate_ra`, `calculate_pcist`, `calculate_lyapunov_exponent`,
    `calculate_gain_stability_sensitivity`, `calculate_dfa`,
    `calculate_kaplan_yorke_dimension`,
    `calculate_structural_eigenvalue_outliers`, `calculate_fr_distribution`,
    `calculate_cv_isi`, `calculate_spike_distance`).
  - Save helpers are object-first: `save_dict_to_hdf5(folder_or_filename, data, ...)`
    -> `save_dict_to_hdf5(data, folder_or_file, ...)` (matching `save_yaml`,
    `save_memories_to_xarray`); the path parameter of the HDF5 helpers is now
    `folder_or_file`; `save_yaml(args, ...)` -> `save_yaml(obj, ...)`;
    `save_memories_to_xarray(data, ...)` -> `save_memories_to_xarray(memories, ...)`.
  - `save_memories_to_xarray` gains `force_sparse` (forwarded to
    `memories_to_xarray`); it sits before `compression_level` positionally.
  - Hex plots: one shared `btorch.utils.hex.CoordFormat` literal is used for
    `coord_format` in static/interactive `scatter`, `quiver`, `heatmap` and the
    animation classes; the interactive resolver now supports every format.
  - `make_hetersynapse_conn` and `neuron_subset_to_conn_mat` gained
    `typing.overload` signatures keyed on `return_dict` / `return_mode`.

- **Surrogate autograd functions are `torch.func`-compatible** — `_SurrogateAutograd`
  and the Poisson random spike function use `setup_context`/`jvp`/
  `generate_vmap_rule`, so `torch.func.jvp/vjp/vmap` work through them.
- **Analysis failures no longer return error strings or fake zeros** —
  `compare_fano_methods` and `calculate_gain_stability_sensitivity` return NaN and
  log a warning; `save_yaml` fails explicitly; `fano_operational_time` rejects a
  non-zero `overlap` instead of ignoring it.
- `FiringRateLoss` now defaults to a valid `loss_type` (`"huber_pinball"`).
- **Breaking (models/connectome review cleanup):**
  - `btorch.models.functional.set_memory_reset_values(mod, hidden_states, ...)`: the
    argument is now `reset_values` (it sets reset values, not hidden states).
    `MemoryModule.set_memories_rv` now defaults to `strict=True`, like the
    function and `set_reset_value` (was `False`). The function now also accepts a
    whole-module entry such as `{"neuron": {"v": 0.0}}` (it raised `AttributeError`)
    and an unknown name raises `KeyError`.
  - `RecurrentNN.get_grad_history()` returns a copy; mutating the result no longer
    changes the module's history.
  - `PoissonRandomSpike.derivative(x, damping_factor)` now has the base-class
    signature `derivative(x, grad_output, damping_factor)`.
  - `step_mode`/`backend` annotations use the shared `btorch.models.base.StepMode`
    and `Backend` literals everywhere (typing only).
  - `tests/benchmark/test_ode.py` -> `tests/benchmark/test_ode_bench.py`.

### Fixed
- `import btorch` no longer fails when the package is not installed (e.g. imported from a
  source checkout): `__version__` falls back to `"9999"` (same convention as xarray) instead
  of raising `PackageNotFoundError`.
- **`make_hetersynapse_conn` delay expansion** crashed with a length mismatch (or
  silently shifted delays between connections) whenever `delay_col` was combined
  with duplicated (pre, post) pairs, `receptor_type_mode="connection"`,
  `dropna="filter"` or `return_dict=True`. Delays are now applied per edge, so every
  combination of `delay_col`, receptor mode and `return_dict` works.
- **Breaking:** `make_hetersynapse_conn(..., receptor_type_col=None, return_dict=True)`
  ignored `return_dict` and returned a sparse array (contradicting the overload); it
  now raises `ValueError`, as there are no receptor types to key a dict by.
- New `GLIF3.exact_no_spike_at(x, t, v0, Iasc0)`: the elementwise closed-form primitive (no time
  axis; every batch element and neuron may have its own `t`, `x`, `v0`, `Iasc0` and
  parameters). Pure, differentiable and `torch.compile` friendly, meant for iterative root
  finding such as the time at which `v` reaches the threshold.
- **Breaking:** `GLIF3.forward_exact_no_spike(x, t, v0, Iasc0)` has an explicit
  shape contract: `t` has the time axis first and `t_mode` says how it is read -- `"homo"`
  (default) a scalar or 1-D `(n_time,)` grid shared by everything, `"heter"` a grid whose
  trailing axes broadcast against the state so times can differ per batch element and
  neuron (was `dt` with four accepted shape unions, guessed from shapes), the batch shape comes from the state, outputs are
  `(n_time, *state[, n_Iasc])`, and it is a pure function: it never writes the module
  state (it used to overwrite it with the whole time series whenever `v0` and `Iasc0`
  were omitted). Assign `neuron.v, neuron.Iasc = v[-1], Iasc[-1]` to continue from it. The `tau == 1/k` branch no longer risks NaN gradients.

- `SparseConn`/`SparseConstrainedConn` (native backend) cached a sparse tensor
  at construction whose indices were not in the state dict; after a `.to()` /
  `.double()` followed by `load_state_dict` with a different wiring the forward
  used stale connectivity. The sparse tensor is now built from the `indices`
  buffer on each call.
- `IF.neuronal_charge` referenced a non-existent `self.V` and raised
  `AttributeError`; it now uses `self.v`.
- `MemoryModule.set_reset_value(name, ResetValue, strict=False)` for a new name
  skipped the sizes / `has_batch` validation of the normal registration path;
  it is now validated (and copied instead of aliased).
- `is_broadcastable` now checks "first broadcasts to second" on shapes only.
- `MemoryModule.init_state`/`reset` no longer leak one memory's dtype/persistent/
  batch-size override into later memories; explicit `persistent=False` is honored.
- `load_config` now actually loads the config file found via `search_path` (it
  resolved the path and then loaded the original relative one), and raises
  `FileNotFoundError`/`ValueError` instead of a bare `assert`.
- `exp_euler_step` no longer returns NaN when the linear term is exactly zero.
- `compute_lyapunov_exponent` (analysis.dynamic_tools.complexity) raised `ValueError` on any
  spike train: it passed a 2D rate to nolds and put `dt` in the `emb_dim` slot. It now
  uses the mean population rate with default embedding parameters.
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
- **Two-compartment fitting** (`btorch.analysis.two_compartment_fit`): the
  global-search objective now uses the full `FitLossConfig` (`spike_tau_ms`,
  `post_spike_mask_ms`, match window, miss penalty) instead of only the weights,
  and staged fitting ranks stages with the same smoothing / mask settings.
  Fit results with non-default settings change intentionally.
- **`plot_raster`**: `show_group_strip=True` honours an explicit per-neuron
  `spike_color`; `group_rate` without `group_key` raises `ValueError` instead of
  drawing an empty rate panel; the strip legend no longer reads `A / A` when
  there are no subgroups; an all-zero spike matrix with a strip and
  `neuron_specs` no longer fails.
- **`plot_neuron_traces`**: a batched 3D `psc` next to 3D `voltage` is no longer
  misdetected as multi-component (and `(time, batch, neurons, n_psc)` now works);
  `separate_figures=True` honours `neuron_specs`; `dt` defaults to `None`
  (taken from `states`, else 1.0), so an explicit `dt=1.0` is no longer silently
  overridden by `states.dt`.
- **`memories_to_xarray`** raises `ValueError` when `neuron_ids` cannot fill the
  neuron dims instead of silently dropping `root_id`; removed the dead `partial`
  argument of `_infer_dim_counts` and the unused `unique_val_dims`.
- **`do_bench`** in duration mode always takes at least one sample (a tiny `rep`
  used to return NaN from an empty sample list).
- **`compute_avalanche_statistics`** returns the same keys on the failure path
  (`avg_size_by_duration`, `gamma_stats=None`); failure semantics are documented.
- The `plot_neuron_traces`/`plot_raster` characterization fixture is now the
  tracked `tests/visualisation/timeseries_characterization.json` (was `.golden`
  because of the `*.json` ignore rule), records numpy/matplotlib versions and
  the test skips when they differ.

### Removed

- `btorch.config.SPARSE_BACKEND` (no consumers).
- **Breaking:** `btorch.models.connection_conversion` (`convert_connection_layer`,
  `convert_connection_layer_from_checkpoint`) and its docs page; the module is to be
  rewritten. `ReceptorTypeMode` now lives only in `btorch.connectome.connection`.
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

- New dedicated tests for the `Erf`, `Triangle`, `SuperSpike` and `PoissonRandomSpike`
  surrogates (`tests/models/test_surrogate_functions.py`).
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
- `benchmarks/numpy_model.py` became the test reference
  `tests/models/numpy_reference.py` (pure-numpy LIF/GLIF3, all PSC types and a
  recurrent network), with step-by-step float64 parity tests in
  `tests/models/test_numpy_parity.py`. The manual `benchmarks/draw_glif.py` and
  `benchmarks/vis_glif.py` demos were removed; their useful checks
  (`GLIF3.forward_exact_no_spike` vs the discrete simulation, bfloat16 gradient
  finiteness) now live in `tests/models/neurons/test_glif.py`.

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

### Packaging and utils cleanup
- New `btorch[examples]` extra (torchvision, seaborn, tqdm); `btorch[all]` is now
  a self-referencing extra so it cannot drift from the others.
- `torch>=2.3` is now required (`torch.compiler.is_compiling`).
- `btorch.utils.conf.diff_conf_records` and `btorch.utils.bench.do_bench` were
  split into small helpers with unchanged behaviour.
