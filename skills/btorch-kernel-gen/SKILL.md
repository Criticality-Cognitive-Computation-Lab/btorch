---
name: btorch-kernel-gen
description: Validate stateful btorch models and generated kernels, and prepare contributions.
---

# btorch model and kernel validation

Use Humanize, KernelWiki and ncu-report-skill directly for planning/review, research and profiling. Setup is in [README](README.md). This supplement adds btorch-specific modeling, scientific validation and upstream requirements; it does not replace those skills or implement another agent loop. Read [workflow](references/workflow.md) as needed.

The [phase prompts](prompts/) specify the task contract: Python golden, shapes/dtypes, states, timestep/event order, representative/full cases and gradient semantics.
For SONATA interchange, or network-definition serialization with no IO format/backend specified, use [sonata-io](../sonata-io/SKILL.md): the agent writes network-specific traversal/mapping using shared btorch helpers and calls.
Complete the [model and kernel checklist](references/workflow.md#model-and-kernel-checklist) before generation and again for each candidate kernel.

Before generating a kernel for a new/changed model, run the actual model at single-neuron, connected-pair and network scales. Mandatory activity figures show input, spikes, membrane and every relevant adaptation/PSC channel. Model expectations govern resting, impulse, sustained input, step-off, delay and limits. Human confirmation of these results establishes the golden. State inspection exposes units, logical time, delay/RNG/plastic state without advancing the model.

Performance acceptance is separate from correctness. For cost that varies with neuron count, connection density or activity, use the sweep anchors and verification gates in [Performance sweep](references/workflow.md#performance-sweep); record realized firing rate and use one declared baseline. Activity targets are approximate. The three-curve speedup report is optional; ask before running it.

Start from [GLIF3/RecurrentNN](assets/templates/glif3_rsnn.py), [MemoryModule](assets/templates/stateful_module.py) or [native operators](assets/templates/native_operator.py). Preserve primitive neuron/PSC multi_step_forward as continuous sequence kernel/op boundaries. Reuse RecurrentNN for network scheduling when sufficient; do not replace primitive multi-step execution with make_rnn. Reusable execution/state/plotting belongs in btorch. NESTML informs explicit state/event and scientific checks.

Treat each `ModelTemplate` as the source of truth for state, parameter, record,
forward and reverse fragments. Reuse those fragments through the primitive and
fused paths and render them with the backend-specific renderer; do not add a
model-specific branch to central dispatch for a new neuron or PSC. A module
adapter declares its template binding explicitly, and a local extension test
must exercise the Python golden, generated source and at least one runtime
backend.

Distinguish inference (stateful, no backward-only tape), training forward and backward. Torch owns autograd/registration. Custom models, mega kernels, sparse layouts, quantization and different backward algorithms remain valid; compare declared semantic oracles. Report first divergence with trace/figure evidence, and do not count missing checks as passes.

OOT directories are unrestricted. Upstream follows target btorch files/tests, fig_path/save_fig and pre-commit formatting, then reruns relevant tests. Verify OOT/in-tree behavior in fresh processes. Default requested delivery is a GitHub Draft PR or GitLab Draft MR, without automatic merge.

For kernel parity, import `ModelParityCase`, `ModelParityOptions` and
`check_model_parity` from `btorch.models.kernels.testing`; native binding and
loading helpers remain in `btorch.models.kernels.native`. Prefer a small set of
named cases, each with an independent physical or gradient oracle. Preserve
distinct checks for noncontiguous inputs, state continuation, records,
trainability and compiler/capture behavior. Do not expand routine validation
into a Cartesian model/backend/dtype/options matrix or repeat the same oracle
under every combination. Run the full collection only after shared dispatch,
registration, autograd or source-generation changes.

Coverage audits must enumerate every built-in template and fragment metadata
declaration, plus each local extension registration, and map them to named
source, output/state and VJP checks. An unlisted declaration is a coverage gap,
not an implicit pass.
