---
name: kernel-gen
description: Validate stateful btorch models and generated kernels, and prepare contributions.
---

# btorch model and kernel validation

Use Humanize, KernelWiki and ncu-report-skill directly for planning/review, research and profiling. Setup is in [README](README.md). This supplement adds btorch-specific modeling, scientific validation and upstream requirements; it does not replace those skills or implement another agent loop. Read [workflow](references/workflow.md) as needed.

The [phase prompts](prompts/) specify the task contract: Python golden, shapes/dtypes, states, timestep/event order, representative/full cases and gradient semantics.
For SONATA interchange, or network-definition serialization with no IO format/backend specified, use [sonata-io](../sonata-io/SKILL.md): the agent writes network-specific traversal/mapping using shared btorch helpers and calls.
Complete the [model and kernel checklist](references/workflow.md#model-and-kernel-checklist) before generation and again for each candidate kernel.

Before generating a kernel for a new/changed model, run the actual model at single-neuron, connected-pair and network scales. Mandatory activity figures show input, spikes, membrane and every relevant adaptation/PSC channel. Model expectations govern resting, impulse, sustained input, step-off, delay and limits. Human confirmation of these results establishes the golden. State inspection exposes units, logical time, delay/RNG/plastic state without advancing the model.

Start from [GLIF3/RecurrentNN](assets/templates/glif3_rsnn.py), [MemoryModule](assets/templates/stateful_module.py) or [native operators](assets/templates/native_operator.py). Preserve primitive neuron/PSC multi_step_forward as continuous sequence kernel/op boundaries. Reuse RecurrentNN for network scheduling when sufficient; do not replace primitive multi-step execution with make_rnn. Reusable execution/state/plotting belongs in btorch. NESTML informs explicit state/event and scientific checks.

Distinguish inference (stateful, no backward-only tape), training forward and backward. Torch owns autograd/registration. Custom models, mega kernels, sparse layouts, quantization and different backward algorithms remain valid; compare declared semantic oracles. Report first divergence with trace/figure evidence, and do not count missing checks as passes.

OOT directories are unrestricted. Upstream follows target btorch files/tests, fig_path/save_fig and pre-commit formatting, then reruns relevant tests. Verify OOT/in-tree behavior in fresh processes. Default requested delivery is a GitHub Draft PR or GitLab Draft MR, without automatic merge.
