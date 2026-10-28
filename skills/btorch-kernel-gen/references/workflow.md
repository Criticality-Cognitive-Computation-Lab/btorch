# btorch scientific and upstream requirements

External tool setup and source attribution are in [README](../README.md). This reference specifies model, validation and contribution requirements.

# Model and kernel checklist

Complete the first list before generating a kernel, then repeat the matching
checks against every candidate kernel. Mark unavailable facilities as NOT_RUN
with the environment reason; do not treat absence as a pass.

## Before generation

- [ ] Check the Python model's eager correctness and its declared
  `torch.compile` and CUDA-graph support.
- [ ] Check that a new stateful model follows :class:`MemoryModule`: persistent
  state is registered through its memory interface, reset/detach/checkpoint
  behavior is defined, and both the current memories and their reset values
  (`_memories` and `_memories_rv`) describe the intended state.
- [ ] Check that each primitive neuron or PSC supplies
  `single_step_forward` and `multi_step_forward`. The latter must state its
  continuous initial/final-state contract; it is not merely a convenience
  Python loop.
- [ ] For an RSNN/RNN, check whether existing :class:`RecurrentNN` composition
  expresses the required topology and scheduling. Reuse it when it does; a
  manual temporal loop needs a concrete unsupported scheduling or model reason.

## After generation

- [ ] Compare eager candidate outputs, requested records, final state and
  gradients with the confirmed golden across repeated stateful calls.
- [ ] Check the candidate through its intended primitive
  `single_step_forward` and `multi_step_forward` paths, including their
  initial/final-state contracts.
- [ ] Check the candidate in its intended :class:`RecurrentNN` composition;
  verify that no manual loop has bypassed the requested network scheduler.
- [ ] Validate the declared `torch.compile` support and, on CUDA, CUDA-graph
  replay correctness. Compare outputs, records and final state with eager
  execution; compiled/captured runs must preserve continuation semantics.

# Inference, training forward and backward

| Path | Computation | Retained information |
|---|---|---|
| Inference | Outputs and state updates without a derivative path through this invocation | Final states and requested records; no backward-only tape/residual |
| Training forward | Outputs and state updates with a derivative path | Requested records plus backward-required information, shared or recomputed when appropriate |
| Backward | VJP from outputs, final states and returned records to differentiable inputs, initial states and parameters | The matching invocation's independent context |

`forward` alone means positive-time computation; call a residual-producing path training forward. Inference still maintains membrane, PSC, delay and other model state. It need not be the same algorithm as training forward.

Primitive neuron/PSC `multi_step_forward` is a continuous sequence boundary, including registered kernel/op dispatch and its documented initial/final-state semantics. RecurrentNN owns network composition and uses the planned primitive sequence path where the selected backend supports it. Preserve both layers. Do not automatically wrap a primitive in `make_rnn` to obtain sequence execution or records. A record adapter must exercise the intended primitive sequence path and expose its actual histories. The existing Torch PSC convolution path is stateless; do not assume it has the stateful continuation semantics of other paths.

Select at the whole-sequence boundary from `torch.is_grad_enabled()` and the actual participating tensors' `requires_grad`. `no_grad`/`inference_mode` select inference; `.train()`/`.eval()` do not. A frozen network still needs a derivative path if its upstream encoder does. Frozen weights do not remove the transpose needed for input gradients; initial states without requested gradients do not remove temporal adjoint propagation.

Use Torch's native custom_op/register_fake/register_autograd interfaces. Backward may call another custom operator, including EventProp after event-driven forward. Each invocation owns its context so two forwards can precede one backward. Saved tensors follow Torch version/lifetime rules. A separate inference name is insufficient evidence: inspect saved tensors, returned auxiliaries, implementation allocations and measured memory.

For identical forward/gradient semantics, golden is the actual btorch Python Module or confirmed user model. opcheck checks registration; gradcheck checks numerical derivatives when applicable. A hard threshold with surrogate/STE has a deliberately different derivative: compare its declared VJP instead. EventProp needs an oracle appropriate to the event model, not automatic equality to discrete surrogate BPTT. Quantized forward compares its quantized golden and separately reports floating-point accuracy. Stochastic pathwise comparison needs aligned logical random inputs; identical seeds alone do not guarantee identical random consumption.


# Scientific validation and readable state

Use one real model/network factory at single-neuron, connected-pair and complete-network scales. Do not build a second simplified simulator. Tests in `tests/models/neurons/test_neurons.py` demonstrate GLIF3 activities and behavioral assertions; retain every relevant adaptation channel rather than plotting only one.

Describe every state with its Module path, axes, dtype/device, units and logical observation time. Distinguish previous, charged and reset state if the model exposes them. Show receptor/sign, emission/arrival timestamps, delay-ring cursor, and relevant RNG/plastic state. Tiny cases expose complete values; larger cases expose chosen neuron/edge/time slices and raw tensor artifacts. State inspection must not mutate the model or RNG. The generic response helpers expose recorded tensors; event timestamps, model-specific assertions and stochastic/plasticity oracles must be supplied by the model adapter and validation cases. They are not generated automatically.

Use no-input, impulse, step-on/off and sustained drive. Set expected ranges/attractors from the model, not a universal convergence requirement. Network limits may include zero/mixed/maximum delays, ring wrap, simultaneous events, threshold neighborhoods, empty/one-step input, nonzero initial state and chunk boundaries around events. Select relevant cases from the task rather than forcing every mechanism on every model.

Activity figures are required at all three scales. Include input, spikes, membrane/threshold, adaptation and PSC as applicable; network raster/rates and transient/delay zooms. Compare candidate/golden with a first-divergence window and event context. Figures carry case/source/parameter/time metadata; numerical assertions remain separate from human inspection.

Reference [NESTML](https://nestml.readthedocs.io/en/latest/) for explicit state, units, dynamics and event handling. Its [STDP windows](https://nestml.readthedocs.io/en/latest/tutorials/stdp_windows/stdp_windows.html) motivate pre/post timing, traces, delay and weight-response checks; its [stochastic GL tutorial](https://nestml.readthedocs.io/en/latest/tutorials/gl_model/gl_model_tutorial.html) motivates repeated-trial activity and statistical response checks. Reuse these validation ideas with Torch models rather than introducing a DSL.

Stochastic comparisons use a declared pathwise random input or statistical contract across trials. Plastic cases can examine pre-only/post-only, both orders, ties, repeated events, bounds and trace decay according to the actual rule. Preserve time-of-weight-read/update semantics. Model execution owns these mechanisms; the harness does not supply a replacement plasticity engine.

Feedback distinguishes PASS, FAIL, BLOCKED, UNSUPPORTED and NOT_RUN. Include reproducible commands, first differing time/state/index, expected/actual/tolerance and raw data/figure paths. Hypotheses are labeled hypotheses. No missing required capability is silently accepted.


# Performance sweep

Use this contract for kernels whose cost varies with neuron count, connection
density and activity. It selects workloads and gates acceptance; it is not a
second profiling workflow. Use ncu-report-skill for collection, diagnosis and
reports, and Humanize for the implementation/review loop.

Use anchors `N={1e3, 1e5}`, `density={0.1%, 5%, 10%}` and target rate
`{1, 5, 20 Hz}`. Density is the fraction of nonzero recurrent edges,
`density = nnz / (N * N)`, and `nnz` is the primary workload axis. `N_small` is
launch/occupancy-bound and `N_large` is the largest size that fits on the target
device at 10% density; treat them as a search space, not a mandatory factorial.
Activity is hard to control, so calibrate input drive toward the target and
record the measured rate `realized_hz = 1000 * spikes / (N * T * dt_ms)`.

Screen the grid with a light initial profile, then use that profile and the
current bottleneck to choose 3-5 representative points covering both `N`
regimes and the density/activity operating plane. Treat the set as
provisional: re-select points as profiling evidence and failures change the
bottleneck. Optimize only the current representative points, then verify the
candidate on the remaining combinations for correctness, latency, memory,
`nnz` and realized activity. Keep one declared baseline and report small and
large workloads separately.

The optional final report has three speedup curves with at least five points
each: versus neuron count, versus density/`nnz`, and versus realized firing
rate. Ask the user before running it.


# btorch upstream delivery

For SONATA IO, or network-definition serialization without a specified IO format/backend, use [sonata-io](../../sonata-io/SKILL.md). The network-specific adapter owns traversal and reconstruction, calls shared file helpers, and validates both reader interoperability and model round-trip behavior. No common network-definition protocol is assumed.
Upstream shared SONATA helpers/interface and reusable checks only; keep network-specific SONATA implementations in the user's project/OOT package.

OOT users choose their source and output layout. Apply the following only when preparing a btorch contribution.

Read the target checkout's AGENTS.md, actual modules, tests and `.pre-commit-config.yaml`. Put neurons in the existing neuron package; use corresponding synapse/connection/network modules and kernel/parity organization. Follow existing public export and optional dependency patterns. Reusable execution/state/plotting functions belong in btorch; harness scripts orchestrate them.

Organize tests by the corresponding model/kernel/visualisation functionality. Include intended uses and meaningful edge cases with explanatory comments; do not add tests that merely assert assignments or argument passing. Use the actual model as the scientific baseline.

Use `btorch.utils.file.fig_path` and `save_fig` for contribution figures. Pass `file=__file__` from the calling test so a shared helper does not assign figures to its own location. Use distinct parameterized names, record returned paths and close figures. Tests/benchmark paths follow the existing utilities. Do not impose this output convention on OOT workspaces.

Run `pre-commit install --install-hooks` and `pre-commit run --files <contribution files>` in the target checkout. Apply formatting and rerun until passing, then rerun relevant tests on formatted code. Use the target btorch configuration, not an independently invented formatter setup. Run appropriate Ruff and bilingual docs builds; never edit autogenerated API pages.

Prepare a scoped branch/worktree without unrelated edits. Keep one implementation for OOT and in-tree, adapting imports/registration rather than regenerating equations. Verify both in fresh processes. Contribute models, kernels or both as the task requires; every backend is not mandatory.

After changing a native operator schema, verify compiled execution with an isolated compiler cache as well as the existing cache. Report stale-cache failures separately; do not delete shared caches or present a cache-isolated pass as proof that old artifacts remain compatible.

Include source/license attribution, three-scale evidence, parity, suitable benchmark results and reproduction commands in PR materials. Large traces/profiles/build caches are external artifacts, not source files. By default publish a GitHub Draft PR or GitLab Draft MR using the target remote/default branch; resume by updating the same draft. User instructions can override destination or readiness. Do not repeatedly ask for publication permission or automatically merge. If destination/authentication is unavailable, retain prepared local work and report the concrete blocker.
