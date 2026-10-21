# Kernel test structure

Use `helpers.py` for the backend parameter matrix (`GPU_BACKENDS`), optional
CUDA availability checks, the common cell input layouts, `NewEmptyShapes`, and
whole-sequence output/VJP comparison against the actual Module. Numerical
options for that oracle are grouped in `SequenceParityOptions`; callers with
parameter views can supply their backing-storage roots. Helpers coordinate
existing Torch and btorch code rather than implement another simulator.

The `reference_backend` fixture registers the real Torch sequence callbacks
once, on demand. Use it when a CPU test needs to enter the same recurrent
operator boundary as native GPU backends. Plain `backend="torch"` remains the
actual Module schedule oracle. Importing a test should not register an
otherwise unused reference backend.

Keep model-specific drives, time units, reset/delivery expectations, gradients,
and numerical regression values next to their tests. Parameterize independent
model/backend/layout choices; avoid creating one copy of a test per backend.
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
