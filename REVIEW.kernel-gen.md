# Kernel delivery audit

This audit compares delivered claims with code, assertions and reproducible
counterexamples. It identifies misleading outcomes and acceptance weaknesses;
it does not infer intent from a bug. Three independent reviewers examined prose,
golden/parity, and state/evidence handling.

## Findings and corrections

- Repeated source branding in skill descriptions, phase headings and tutorials
  added no implementation evidence. Removed it; source links and concrete prompt
  adaptations remain together in the README.
- The GLIF3 example called finite-value and spike-presence checks an "expected
  response". Its output and documentation now explicitly call these smoke
  checks. They do not establish adaptation dynamics, PSC sign/arrival timing,
  delay correctness or generated-kernel parity.
- Tutorial OOT/parity snippets used undefined package, candidate and golden
  names. Replaced them with the actual loadable affine template and ordinary
  Torch affine oracle, explicitly identified as an API example. Added a runnable
  initialized MemoryModule sequence example. A subsequent sequence-boundary
  review replaced its wrapper with the primitive's own multi-step entrypoint.
- A report could exclude a failed temporal check from required checks, disable
  the three activity scales, or supply source text as a figure. The gate now
  rejects all explicit FAIL results, requires the three scales and checks image
  format. Regression cases exercise these bypasses.
- An unordered provenance digest accepted swapped model/helper contents. The
  digest now preserves declared source/artifact order while allowing unchanged
  directory relocation; a regression exercises the swap.
- A period-13 cotangent accepted a wrong backward that exchanged indices 0 and
  13. Both parity interfaces now use locally seeded nonperiodic probes without
  changing global RNG, and a deliberately wrong-backward regression rejects it.
- Comparing only after moving outputs to CPU hid device mismatches. Floating
  tolerance also accepted a one-unit error in a large integer output. Device is
  now checked before transfer; integer and boolean outputs compare exactly.
- Native parity always requested a grad-enabled run, even for inference-only
  kernels, and its label could imply training execution with frozen inputs.
  Validation contexts are now selectable; frozen grad-enabled runs explicitly
  report that they do not establish training-kernel execution.
- Historical GPU pass counts in the plan were not reproduced in this session.
  They are marked historical, not current verification. CUDA was inaccessible
  in the initial sandboxed run; the subsequent GPU rerun is recorded separately
  below. GPU performance cannot be inferred from numerical test results.

## Remaining limits

- The gate records suite results and checks integrity. A suite can still lie
  about execution; hashes do not prove scientific correctness. Image parsing
  does not prove a figure depicts the declared run. Reviewer text is a record,
  not human authentication. Review actual assertions, traces and figures.
- Generic response plots show supplied tensors; they do not automatically infer
  receptor/neuron identities, units, event arrival times or first-divergence
  windows. Model adapters and cases must supply those diagnostics.
- Stochastic, plastic, quantized and alternative-backward cases require declared
  independent oracles. The bundled examples do not implement or validate all
  those mechanisms. A flexible interface is not evidence of algorithm coverage.
- A sampled VJP is not proof of the complete Jacobian. Use appropriate additional
  cases and Torch gradcheck for smooth derivatives; finite-difference gradcheck
  cannot certify a deliberately different surrogate derivative.
- Parity numerical success must be read alongside unavailable/skipped checks;
  it does not establish an unexecuted GPU or capture capability.
- Cell parity's default tolerance follows input precision, including FP32 state
  outputs of low-precision-input cases. This can be too permissive for some
  states. Declare tighter task-specific tolerances; per-output budgets remain
  an improvement to the checker, not an implemented guarantee.

## Substantiated implementation

Built-in cell golden execution constructs existing btorch Python Modules and
uses their single-step path for the stateful cell ABI. Network examples reuse
RecurrentNN; primitive examples use their own multi-step entrypoint. Explicit
single-step reference diagnostics are not primitive sequence-op parity. State
tests include circular delay cursor restoration, aliases,
chunk continuation and parameter/input gradients. Native templates have fake
and autograd registration; the affine template's broadcasting and MemoryModule
initialization/continuation were executed during this audit. These are scoped
facts, not a claim that every generated kernel is correct.

## GPU rerun after permission update (2026-10-11)

Environment: three RTX 5090 GPUs, Torch 2.11.0, CUDA driver reporting 13.2;
Triton, TileLang and CuPy are installed.

The first kernel-suite run produced 409 passed and 6 failed. Three fullgraph
backward failures loaded old Inductor artifacts with an obsolete custom-op
schema. All three passed with a separate empty compiler cache; no existing
cache was deleted. The other three failures were a real regression in the new
parity probe: constructing a CUDA Generator inside graph capture. The checker
now prepares cotangents before capture and reuses them for the reference,
warmup and replay. All three captured-training tests pass after this fix.

An independent GPU run of OOT/native tests passed 38 cases. The loadable affine
template's broadcast case passed Torch opcheck and gradcheck on CUDA; an actual
CPU/CUDA output-device mismatch was rejected. These correctness checks do not
measure performance or reproduce historical benchmark timings.

Final rerun, with an isolated Inductor cache: **455 passed, no skips or
failures** across `tests/models/kernels`, GLIF/Izhikevich neuron tests, memory
save/load tests and the harness evidence tests. Modified-file pre-commit/Ruff
checks pass. This covers the existing three-backend numerical, state/parameter
gradient, sparse heter/delay, fullgraph and CUDA graph regression cases.

## Primitive sequence versus network scheduling audit

Three reviewers checked core execution, documentation and the response helper.
Primitive neuron/PSC multi-step entrypoints own continuous-sequence semantics;
RecurrentNN owns network scheduling. An override can internally loop without
being a GPU kernel, but a wrapper must not silently bypass that override.

Corrected the harness's automatic make_rnn fallback. Standalone primitives now
use their own multi_step_forward, including empty-input semantics. Requested
primitive histories require an explicit adapter and are rejected before state
advances when no adapter is supplied. GLIF single-step reference diagnostics
are explicitly identified as such. The MemoryModule template now implements
its own continuous sequence entrypoint. Relevant regression tests pass (112).

Corrected the unconditional PSC convolution-equivalence claim in
docs/en/docs/concepts/dt_environment.md, both mixed-neuron tutorials, and overly
broad test explanations. Torch PSC convolution is from-rest/stateless; the
stateful sequence kernel and chunk continuation are different contracts.

The core dispatch issue was reproduced: make_rnn's step scheduler in
btorch/models/rnn.py calls rnn_cell as a Module, and RecurrentNN similarly calls
its neuron/synapse Modules. MixedNeuronPopulation does the same for children.
These calls obey each child's step_mode, so a child in mode m can interpret a
single-step batch dimension as time. Reproduction: make_rnn(LIF(1,
step_mode="m")), batch size 2, input shape [3, 2, 1], returns [3, 2, 2, 1].
It is now fixed by a scoped single-step Module call in make_rnn, RecurrentNN,
ApicalRecurrentNN and MixedNeuronPopulation. The child default is restored in
a finally block, and ordinary Module calls preserve hooks/custom forward.
Primitive multi-step entrypoints remain independent.

Five dispatch regressions pass: batched shape/state/gradients and split BPTT,
neuron/PSC recurrence, mixed children, hooks/custom forward and exception
restoration. Expanded RNN/mixed/kernel-integration testing produced 267 passed
and two failures in existing CUDA graph topology assertions that assume
RecurrentNN has no capture_version participant. This dispatch patch does not
change capture_version; the failures remain reported rather than rewriting
their assertions to obtain a green run.

The existing golden's single-step rollout for the stateful cell ABI is valid.
Replacing it with the stateless PSC convolution would invalidate nonzero-state
and continuation comparisons. Normal RecurrentNN scheduler/unroll tests and
its description as a network multi-step layer are also valid.

## Development delivery checks (2026-10-12)

Excluded local external-skill installations from the contribution; the harness
README retains all three dependency links. Corrected the review classification
of the explicitly reference-only GLIF3 diagnostic adapter. Consolidated imports
in the combined surrogate test file to satisfy the repository formatter/linter.

Current ml-py312 verification: 474 passed across tests/models/kernels,
tests/utils/test_kernel_gen_evidence.py and tests/models/test_surrogate.py.
Contribution-file pre-commit checks pass. These checks do not complete the
pending lowering/performance decisions in REVIEW.queue.md.

## Parallel audit fixes (2026-10-12)

Three fresh agents implemented API/validation, Delay continuation, and fused
reverse generation, then independently reviewed each other's changes. The
integrated kernel/PSC/Delay/harness suite passed 670 cases. Contribution-file
pre-commit checks pass. The additional independently registered OOT VJP case
also passed in the API agent's targeted run.

The bilingual docs build was attempted and failed during English API collection
of btorch.analysis.dynamic_tools.spiking; Chinese build was not reached. No
success is claimed for the docs build. General lowering and precision/performance
work remains listed in REVIEW.queue.md.
