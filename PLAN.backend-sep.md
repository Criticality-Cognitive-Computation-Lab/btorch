# backend-sep implementation plan

This is the active cell-backend plan. The sparse design in `PLAN.md` remains
historical context. Start from main on `backend-sep`; reuse existing sparse
kernels, routing, delay/receptor layouts and caches. This document specifies the
target API and acceptance criteria, not a claim that every item is implemented.

## Architecture and invariants

Use explicit ModelTemplate declarations, fixed charge/adapt/fire/reset stages,
handwritten reverse stages and BackendFragments. Generate code directly; do not
parse Torch IR or build a megakernel. Neuron and PSC dynamics can share a cell
kernel; connection propagation remains an independent existing kernel. No new
heavy mandatory dependencies.

Preserve the RNN recurrence: neuron reads previous PSC, current spike goes
through connection, then PSC advances. Fused scheduling uses first neuron,
alternating connection and PSC/neuron, and final PSC: T connection launches and
T+1 cell launches for nonempty T. Records align with logical time.

Support IF, LIF, GLIF3, Izhikevich and ExponentialPSC, AlphaPSC, AlphaPSCBilleh,
DualExponentialPSC. Start with LIF + AlphaPSC. Preserve six built-in deterministic
surrogate formulas, voltage normalization, reset/detach and refractory semantics.
Non-trainable scalar surrogate hyperparameters only in the first version.
No arbitrary custom surrogate, online plasticity or higher-order gradients.

## End-user API

One argument chooses both independent backends:

```python
from btorch.models import KernelOptions
from btorch.models.rnn import RecurrentNN

rnn = RecurrentNN(
    neuron=neuron,
    synapse=synapse,
    backend=KernelOptions(
        cell="triton", connection="auto", merge_synapse=False,
    ),
)
with torch.inference_mode():
    spikes, records = rnn(x_seq)
spikes, records = rnn(encoder(x_seq))
criterion(readout(spikes), target).backward()
```

`backend: str | KernelOptions`; a string changes only `cell`. Built-ins are
`torch`, `triton`, `tilelang`, `cupy`; registered out-of-tree names are accepted.
`connection` remains `auto`, `triton` or `aten`, using the sparse registry.
Do not add a separate `kernel_options` or user-maintained simulation/training flag.
Existing Torch defaults and existing constructor/return contracts remain usable.
Explicit sequence options provide stateful PSC recurrence; document the existing
plain Torch PSC convolution's stateless/truncated behavior instead of silently
changing it.

At the entire sequence entry choose training iff grad is enabled and any
participating input, initial state, connection, neuron or PSC float tensor
requires grad. `train()` / `eval()` have no effect on this choice. Preserve the
upstream encoder path with frozen RNN parameters; a trainable readout alone does
not require RNN backward. No automatic BPTT detach.

Precision/promotion and AMP policy remain pending review of established kernel
and simulator practice; the current FP32 coercion is an implementation limitation,
not the target contract. Default tensor layout `[T,B,N]`, explicit receptor axis and
preserved multidimensional batch/neuron axes at Module boundaries.

## Out-of-tree extension API

Publish these from `btorch.models.kernels`; user packages need no btorch fork:

```python
from btorch.models.kernels import register_cell_backend

register_cell_backend(
    "my_lab_lif",
    forward=my_forward,
    backward=my_backward,
    supports=lambda spec, device: spec.model == "lif" and device.type == "cuda",
)
neuron = LIF(..., step_mode="m", backend="my_lab_lif")
```

The registration name identifies an implementation, not a new run mode.
Validate names and callable signatures where practical, reject duplicate
registration unless explicit `replace=True`, and reject unsupported model/device
choices with a useful error. A simulation-only backend may omit backward but
must fail clearly when differentiable execution is needed. Never silently fall
back to Torch for a requested custom backend.

Forward callback contract:

```python
forward(x_seq, states, parameters, spec, history_names)
# -> (output_seq, final_states, histories)
```

Backward callback contract:

```python
backward(
    x_seq, initial_states, parameters, spec, history_names, histories,
    grad_output, grad_final_states, grad_histories,
    needs_input_grad=None,
)
# -> (grad_x, grad_initial_states, grad_parameters)
```

Tensor tuples have ModelTemplate's declared order. History tensors follow
history_names. Every participating trainable float tensor is an explicit input;
packing does not detach. Integer delay/receptor/topology tensors do not require
grad. The btorch sequence wrapper owns grad dispatch, fake/meta, autograd
registration, output ordering, and Module state writeback. A backend owns pure
tensor numerical execution. Each call independently retains its reverse-pass
records, including two forwards followed by one backward.

Expose named template declarations and renderer fragments so an external package
can generate a backend from supported btorch equations without duplicating their
semantics. Add a deterministic generation example with explicit generated-source
output, static spec/cache key and eager warmup before CUDA capture. Do not hide
JIT, run `eval` on user tensor data, or require a global monkey patch. Extension
to a new model needs an explicit template with shape/history declarations and a
an existing Python implementation for parity; registering a name alone is insufficient.

Sparse out-of-tree implementations continue to use the existing sparse registry;
cell registration must not copy or replace sparse algorithms. Document both
registries' boundaries and a complete custom-cell + existing-sparse example.

## Public parity tool for generated and external kernels

Provide `btorch.models.kernels.testing` as an optional developer helper. It
should not require pytest at runtime; it returns a structured report and offers
an assertion method. Planned usage:

```python
from btorch.models.kernels.testing import ParityCase, check_backend_parity

report = check_backend_parity(
    "my_lab_lif",
    cases=[ParityCase(x_seq=x_seq, states=states, parameters=parameters,
                      spec=spec, history=history)],
)
report.assert_close()
```

Use independent cloned inputs/state/parameters and the same initial values,
precision and loss cotangents for existing btorch Python Modules and candidate backend.
Keep both forward and backward tests: spike outputs alone are insufficient.
Report named tensors, maximum absolute/relative error, failing coordinates and
first spike-divergence timestep. Tolerances depend explicitly on dtype; include
low-precision long sequences and do not hide spike mismatches in a broad float
error tolerance.

Core matrix:

- spikes/output sequence, every final state, requested post-step and charged
  history, EPSC/IPSC for unmerged receptors;
- input, participating parameter and initial-state gradients; losses on output,
  final states and records, including reset-detach and refractory branches;
- all parameters frozen but upstream input trainable; input frozen but parameter
  or initial state trainable; only downstream trainable; all frozen;
- eval with gradients, no_grad and inference_mode; simulation does not allocate
  backward-only records; requested/backward history reuse storage;
- reset/detach, zero/single/long sequence, chunk continuation, checkpoint restore,
  two forwards then unified backward, multiple batch/neuron axes;
- scalar and per-neuron broadcast parameters, FP32/FP16/BF16, all six surrogates,
  each advertised neuron/PSC model and supported option;
- custom-backend registration success, duplicate and unsupported-device/model
  failures, simulation-only backward refusal, external module state writeback;
- dense/sparse parity, receptor/delay layout, delay-ring initial-state gradients,
  sparse transpose for frozen weights, edge-gradient time/sample reduction;
- static `torch.compile(fullgraph=True)` forward AND backward; CUDA graph replay
  after warmup; no tensor-value dispatch, capture-time compilation or stale
  registration caches.

An unavailable optional GPU/backend/compiler case is an explicit skip in the
report, never a pass. Core numerical failures cause assertion failure. Compile
and capture checks are requested explicitly and return distinct statuses so
users know what their generated kernel actually supports. Add a faulty OOT
backend regression demonstrating that wrong state/parameter/record gradients
are caught, and a successful external package example exercising the same public
API as real users. API parity tests validate meaningful execution, never merely
argument forwarding or attribute assignment.

## History, autograd and merge contracts

HistoryPlan unions user records and reverse-pass requirements; store shared
references, not duplicate residuals. Recover previous state from initial state
and history and locally recompute cheap charged values/derivatives; do not replay
the full forward. Always propagate temporal state adjoints, even when the initial
state is frozen. Trim final returned/input parameter gradients by needs_input_grad.
Avoid `[T,B,E]` edge-gradient storage or dense sparse-weight gradients; use the
existing sparse transpose and edge reduction kernels.

`merge_synapse=False` retains full receptor dynamics and queryable EPSC/IPSC.
`True` merges only compatible linear dynamics with frozen equal parameters or
actual shared parameter identity. Independently trainable parameters are not
compatible just because current values match. Merge changes cell drive/state,
not sparse topology, receptor/delay expansion or connection kernels. Explicit
incompatible merge or per-receptor recording/query requests raise. Save merge
metadata in checkpoints; switching modes requires reset; independent states
cannot be recovered from a merged state.

Retain chunk/checkpoint/offload and normal BPTT. Complete-sequence registered
operators have fake/meta and autograd implementations and bypass the old disabled
time loop for static fullgraph compile. GPU execution can contain multiple
kernels. Prepare code and route caches before CUDA graph capture.

## Bilingual documentation and executable tutorials

Update both `docs/en/docs/` and `docs/zh/docs/` in the same change, with matching
coverage and navigation. Keep Python code, comments and docstrings in English;
translate explanatory prose for Chinese pages. Do not manually edit generated
`api/` pages in either language.

- A backend guide explains KernelOptions, supported model/options/precision,
  sequence shapes, autograd-based dispatch, stateful continuation, history losses,
  receptor merge restrictions, independent sparse routing and optional installs.
- An OOT guide documents register_cell_backend, callback tensor ordering,
  simulation-only backends, declared capabilities, generation/source inspection,
  warmup/cache lifecycle and structured parity reports with actionable failures.
- An executable tutorial builds encoder -> sparse RSNN -> readout, runs simulation
  and training/eval with gradients, records PSC/state losses, resets/detaches and
  demonstrates chunk/checkpoint continuation.
- An executable OOT tutorial generates or registers a custom cell, verifies it
  with the public parity tool, uses it through the normal neuron/RSNN API, and
  demonstrates detection and diagnosis of an intentionally wrong derivative.
- Update existing building_rsnn and training tutorials with links and a working
  backend selection example. Place reusable runnable examples under `examples/`
  rather than duplicating an untested implementation in documentation snippets.
- Tutorial/API parity tests execute the published public workflow on Torch and
  available optional backends, checking output/state/gradient behavior. Keep code
  and both language explanations synchronized; report optional backend skips.
- Update README.md and README.zh.md for new installation or workflow instructions.
  Run `python scripts/docs.py command=build-all` for both language sites.

## Test-writing and review requirements from AGENTS.md

Re-read repository AGENTS.md before authoring/reviewing tests. Its requirements
are mandatory:

> In tests, include thorough explanatory comments so they are understandable and
> can serve as examples.

> Tests shall include only intended usage examples and edge cases. Do NOT write
> superficial tests that only check argument passing, attribute assignment, and
> trivial things alike.

Use English comments explaining the user scenario, expected numerical/state
semantics, and the subtle error each regression detects. Explain why surrogate
gradient comparisons use the existing Python Module autograd rather than finite
differences of a discontinuous spike output. Exercise real encoder/RSNN/readout,
initial-state/parameter optimization, receptor/delay, persistence and extension
workflows. Never count registry-name checks, mocked argument forwarding, or
attribute assignment as feature acceptance evidence. Duplicate-registration and
unsupported-mode tests are useful only as extension-boundary edge cases beside
actual registered-kernel execution and parity validation.

Review each test for an independently derived expected result or trusted existing
reference, meaningful assertions covering outputs AND relevant gradients/state,
clear tolerances/precision, and capability-specific skips. Do not mirror an
implementation formula and assert it against itself. Each owner runs targeted
pytest and root runs affected existing model/sparse regressions, Ruff and
pre-commit formatting. Keep unavailable environment checks distinct from passes.

## Parallel work and cross-review

1. Templates/reference: model declarations, exact Torch recurrence, handwritten
   VJP, sequence operators, history planning and targeted numerical tests.
2. GPU: shared fragments and real Triton/TileLang/CuPy codegen, sequence forward
   and backward, precision tests and benchmark tooling.
3. Integration: stateful Module adapters, whole-sequence RSNN boundary, existing
   dense/sparse/heter/delay routing, merge and continuation/checkpoint tests.
4. Root: public options/registry, public OOT parity helper, generation/API docs,
   integration review and final regression verification.

Templates author reviews GPU formulas; GPU author reviews reference/operator
registration; integration author reviews ABI, state and sparse routing. Root
resolves findings. Keep ownership boundaries and send actionable review findings;
do not approve an untested placeholder as a complete backend.

## Delivery order and acceptance evidence

1. LIF + AlphaPSC complete numerical vertical slice, three backend forward and
   handwritten backward, every participating floating parameter gradient.
2. Independent connections and RSNN with dense/sparse/heter/delay; compare
   existing API/timing, state restoration and BPTT.
3. Remaining supported neurons and PSCs; merge, static fullgraph, CUDA replay,
   checkpoint/offload and their failure/edge cases.
4. OOT registration, generation example and public parity reports, including
   intentional bad-kernel detection and advertised-capability matrix.
5. User docs; targeted pytest plus sparse/model regressions, `ruff check .` and
   pre-commit formatting. Record GPU/dependency versions and actual test scope.
6. Build both documentation languages and execute bilingual tutorial workflows.
7. Benchmark Torch, existing kernels and three new backends: forward, backward,
   memory, cold compilation and steady execution, eager versus capture; report
   first spike divergence for low-precision long trajectories.

Track remaining gaps explicitly. Do not claim speedups or compile/capture/API
support from source inspection alone; provide executed evidence.

## Implementation evidence (2026-10-10)

The historical counts below were reported by earlier sessions. They must not
be presented as current-session verification. The initial audit could not
access CUDA; the later GPU rerun is recorded in REVIEW.kernel-gen.md.

- Implemented the shared templates, three generated GPU backends and explicit
  first-order VJPs for IF/LIF/GLIF3/Izhikevich and all four existing PSCs.
- Implemented sequence operators, stateful module adapters, independent sparse
  connections, heterogeneous delays, paired built-in cell execution and OOT
  registration/generation/parity APIs. Parallel cross-review fixed low-precision
  drive handling, history duplication, checkpoint merge restoration and graph
  invalidation after topology/backend changes.
- CPU model/sparse regression: 2,023 passed, 634 skipped, 4 expected failures.
  GPU kernel suites: 343 passed. Subsequent checkpoint regression: 1 passed;
  three-backend capture/rewire/backend-switch regression: 3 passed.
- Executable OOT tutorial: Triton fullgraph/capture 115 numerical comparisons;
  TileLang and CuPy capture 96 each. All include AlphaPSC and sparse RSNN training.
- Repository Ruff checks pass. Both documentation languages build successfully
  with `PYTHONPATH=.`; existing documentation link warnings remain.
- Benchmark measurements and their small-workload limitations are recorded in
  `benchmarks/models/CELL_BACKENDS.md`; they are not a general speedup claim.
- Initial connection scope is Identity/Linear/SparseConnection, excluding sparse
  member-pattern batches and custom connection modules. Merge is a heterogeneous
  RSNN option. OOT callbacks compose with connections but do not automatically
  use the built-in paired fusion. These boundaries are documented in both languages.

## Follow-up parallel review evidence

- Three fresh reviewers audited autograd/OOT lifecycle, state/checkpoint/offload,
  and GPU/benchmark coverage. No major obstacle was found.
- Replaced registrations between forward/backward now raise at the reverse
  operator boundary, including compiled backward. Merged checkpoint loading
  retains the reset requirement. First-order-only gradients are explicitly
  enforced instead of silently returning detached higher-order results.
- Public parity capture checks now independently compare replayed training
  outputs and all input/state/parameter gradients, not just simulation.
  Faulty state, parameter and record-loss VJPs have negative regression cases.
- Expanded CPU regression: 2,036 passed, 676 skipped, 4 expected failures;
  subsequent first-order refusal and faulty-VJP cases passed targeted checks.
  GPU kernel suites with a fresh isolated compiler cache: 399 passed.
- Checkpoint/offload combinations and external backend replacement recapture
  passed on CUDA. Six-surrogate refractory/reset state-loss matrix: 72 passed.
- Added independent eager parity and captured complete training to cell
  measurements, plus an executed sparse heter/delay RSNN comparison against the
  historical module baseline. Reproduction and actual first-call/device/memory
  numbers are in `benchmarks/models/RSNN_BACKENDS.md`. Disk-cache warmth and the
  small concurrent workload prevent cold-compile or general speedup claims.

## Parity correction: existing Python models are authoritative

The earlier handwritten Torch recurrence was an incorrect numerical baseline.
It has been removed. `golden.py` binds input/state/parameter tensors to existing
btorch Python Modules; `torch_backend.py` adapts those Modules to the sequence
ABI and uses local Torch autograd for its VJP. GPU fragments remain independent
implementations to be checked against the existing Python models.

The public OOT checker now adds empty/single-step/noncontiguous trajectories,
all declared records, differentiable chunk continuation and two live forward
graphs with one backward by default. Tests deliberately break stride indexing,
charged records, parameter gradients and state continuation to verify rejection.
Model-specific shapes, dtypes, surrogates, reset options and long sequences
remain explicit user cases rather than undocumented synthetic parameter changes.

Removed 30 self-comparison tests and added PyTorch `opcheck` for custom-operator
registration plus `gradcheck` for smooth PSC derivatives. Surrogate spike VJPs
remain compared to the existing Python model's autograd.

Executed after correction: CPU models/sparse 2,023 passed, 682 skipped and 4
expected failures; GPU kernels 390 passed; final OOT/tutorial checks 24 passed.
The complete Izhikevich + heterogeneous AlphaPSC + mixed-delay comparison
against the original RSNN passes all three GPU backends, including initial
v/u/psc/h/delay-history and floating parameter gradients. Both documentation
languages build. Cell and RSNN benchmark workflows were rerun with original
Python Module baselines; older measurement tables are historical, not updated
performance claims for this corrected Torch adapter.
