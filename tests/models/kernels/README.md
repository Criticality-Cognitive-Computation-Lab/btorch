# Kernel test structure

See the [coverage map](COVERAGE.md) for model oracles and numerical failure mechanisms.

Use `helpers.py` for backend names (`GPU_BACKENDS`), optional CUDA availability
checks, `NewEmptyShapes`, and whole-sequence output/VJP comparison against the
actual Module. `assert_module_sequence` delegates to the canonical
`btorch.models.kernels.testing` comparator; its `SequenceParityOptions` keeps
separate forward and VJP tolerances, and callers with parameter views can supply
their backing-storage roots. Helpers coordinate existing Torch and btorch code
rather than implement another simulator.

The `reference_backend` fixture registers the real Torch sequence callbacks
once, on demand. Use it when a CPU test needs to enter the same recurrent
operator boundary as native GPU backends. Plain `backend="torch"` remains the
actual Module schedule oracle. Importing a test should not register an
otherwise unused reference backend.

Keep model-specific drives, time units, reset/delivery expectations, gradients,
and numerical regression values next to their tests. Use selected scenario tables instead of crossing independent
model/backend/layout options. Cover shared equations and state/gradient behavior
once, and choose additional cases for features that differ between renderers.
Do not merge distinct physical edge cases merely to shorten the files.

Use `require_cuda_backend` for optional backend availability, including mixed
Torch/GPU integration matrices. Collect only supported combinations (for
example, AlphaPSCBilleh has a fixed unit timestep); invalid-input behavior
belongs in validation regressions rather than successful-parity cases that
skip at runtime. Test in-tree contracts directly: do not probe alternative
attribute names or skip because an implemented internal field was renamed.
The sparse task tests use concrete prepared-state/workspace dataclasses and
exercise topology invalidation plus concurrent-workspace release.

| Area | Modules |
|---|---|
| Math policy and stable integration | `test_precision.py`, `test_*_exp_euler.py`, `test_dual_exponential_numerics.py`, `test_alif_reverse_association.py` |
| Parameter storage and fused addressing | `test_parameter_layout.py`, `test_fused_parameter_layout.py` |
| Public operators and reverse-output contracts | `test_optional_cotangents.py`, `test_reverse_output_pruning.py` |
| Recurrent memory, records, checkpoints | `test_inference_storage.py`, `test_recurrent_*.py`, `test_residual_plans.py` |
| Stochastic trajectories, saved rasters, capture | `test_poisson_native.py`, `test_poisson_capture.py`, `test_recurrent_poisson.py` |
| Toolchain/cache identity and foreign storage | `test_source_cache.py`, `test_tilelang_toolchain_cache.py`, `test_cupy_interop.py`, `test_device_context.py`, `test_benchmark_capture.py` |
| Timestep assertions and model policy | `test_dt_validation.py`, `test_model_default_policy.py` |
| Low-activity sparse routing | `test_low_rate_recurrent.py`; product/layout cases live in `tests/sparse/test_firing_rate_hints.py` |

Conductance tests deliberately retain three boundaries: plain PSC sequence
semantics (`tests/models/test_conductance_sequence.py`), native standalone
kernels (`test_conductance_sequence.py` here), and the network scheduler
(`tests/models/rnn/test_conductance.py`). These are complementary contracts,
not interchangeable golden implementations.

## Routine and full validation

`pytest tests/models/kernels` runs selected numerical/state/gradient and renderer
cases, together with focused regressions. Each selected table explains which
physical or rendering feature it exercises. `validation_cases` marks the other
combinations `full_validation`; those combinations use the same oracle and
assertions, and are excluded by the default pytest marker expression.

For a change, first run its affected files or `-k` selection. Run the routine
kernel suite once after implementation settles. Use the exhaustive matrix for
an explicitly requested full validation, rather than after every local edit:

```bash
pytest tests/models/kernels/test_low_rate_recurrent.py
pytest tests/models/kernels --durations=20
pytest tests -o addopts='' --durations=20  # includes routine and exhaustive cases
```

Do not weaken tolerances or replace numerical checks with forwarding/attribute
checks when selecting cases. Keep distinct failure mechanisms, such as a
noncontiguous input, outstanding tapes, AMP, delay state and checkpoint/offload.
