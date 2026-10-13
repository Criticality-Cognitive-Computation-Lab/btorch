# btorch kernel-gen harness

Use Humanize for planning and review, KernelWiki for implementation research, and ncu-report-skill for profiling. This bundle supplements them with btorch model/scientific validation, templates and upstream requirements. It does not provide a replacement planning, review or optimization framework. Start with [SKILL.md](SKILL.md); detailed guidance is in [workflow.md](references/workflow.md) and the phase prompts.

## External tools

Reuse installed tools whenever available. These are external dependencies, **not Git submodules and not vendored into btorch**:

| Dependency | Installation and use |
|---|---|
| [Humanize](https://github.com/PolyArch/humanize) | Install its Claude plugin following upstream instructions; use `/humanize:gen-plan` and `/humanize:start-rlcr-loop`. |
| [KernelWiki](https://github.com/DongyunZou/KernelWiki) | Install/expose its skill through the agent's supported discovery; use it for relevant implementation research. |
| [ncu-report-skill](https://github.com/DongyunZou/ncu-report-skill) | Install/expose its skill and follow its profiler/report setup; apply hardware-specific guidance to supported hardware. |

For example, Humanize's documented Claude installation uses:

```text
/plugin marketplace add PolyArch/humanize
/plugin install humanize@PolyArch
```

KernelWiki and ncu-report-skill may be cloned into an external agent skill directory and discovered from there; follow their upstream setup for actual dependencies. Do not add global configuration from this harness without the user's request. Record installed revisions/tool versions with run results. If a tool is unavailable, report the missing dependency; do not silently claim its workflow was executed.

## Run model validation

Install btorch's normal development dependencies. The example uses actual GLIF3/AlphaPSC/SparseConnection Modules through RecurrentNN, not a replacement temporal simulator:

```bash
python examples/kernel_generation.py --output run/figures
python skills/kernel-gen/scripts/check_model.py verify --suite examples/kernel_generation.py:run_suite --output run/report.json
```

The example covers no-input, impulse and step-on/off at single, pair and network scales. Its assertions are smoke checks for finite values, pre-input silence and spike presence. They do not validate adaptation dynamics, PSC signs, propagation delay, gradients or kernel parity. `verify` records suite-reported results and hashes evidence; it does not independently prove those results or authenticate a human reviewer. Inspect the actual suite, traces and figures before confirming a new model. Only the human who has inspected the results and figures should run:

```bash
python skills/kernel-gen/scripts/check_model.py confirm --report run/report.json --reviewer YOUR_NAME --output run/confirmation.json
python skills/kernel-gen/scripts/check_model.py gate --report run/report.json --confirmation run/confirmation.json
```

## Generate, optimize and contribute

Use [Phase 0](prompts/phase0-model.md), then [Phase 1](prompts/phase1-correctness.md), [Phase 2](prompts/phase2-optimization.md) and [Phase 3](prompts/phase3-delivery.md). [Templates](assets/templates/) start from existing btorch models, MemoryModule and native Torch operators. [Execution semantics](references/workflow.md) separates inference, training forward and backward. [Scientific validation](references/workflow.md) includes NESTML references and mandatory activity plots.

OOT output directories are unrestricted. For [upstream](references/workflow.md), follow btorch file/test organization, fig_path/save_fig and the target checkout's pre-commit formatting. Publishing defaults to a GitHub Draft PR or GitLab Draft MR; it does not merge automatically. Run `python skills/kernel-gen/scripts/check_model.py --help` for model evidence verify/confirm/gate commands; upstream uses normal Git and host CLI tools.

See the [English tutorial](../../docs/en/docs/tutorials/kernel_generation.md) or [中文教程](../../docs/zh/docs/tutorials/kernel_generation.md), and the input-prompt provenance below.

For SONATA import/export, or network-definition serialization without an IO format/backend specified, invoke [sonata-io](../sonata-io/SKILL.md). Each network supplies its own traversal and model mapping; shared repository helpers handle file layout and edge indices. See the [Izhikevich RSNN round-trip](../../examples/sonata_izhikevich.py) and [IO guide](../../docs/en/docs/guides/sonata_io.md).

## Source prompts and adaptations

The phase prompts are adaptations of task prompts supplied to agents, not their response logs or final implementations:

- [Contest prompts/README.md](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest/blob/main/prompts/README.md): draft → Humanize gen-plan → start-rlcr-loop.
- [gdn-prefill/phase1.md](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest/blob/main/prompts/gdn-prefill/phase1.md): explicit workload/state contract, first correct implementation and representative validation.
- [gdn-prefill/phase2.md](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest/blob/main/prompts/gdn-prefill/phase2.md): ranked optimization directions with NCU and before/after measurements. Its five-iteration budget is adjustable here.
- [gdn-prefill/phase3.md](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest/blob/main/prompts/gdn-prefill/phase3.md): measured shape specialization and complete workload evaluation.
- [KDA prompts/basic-flow.md](https://github.com/mit-han-lab/kernel-design-agents/blob/main/prompts/basic-flow.md): explicit objective, validation/evaluation commands and promotion evidence.

Local source inspection used these files under `/data1/fanqixuan/src/skills/mlsys2026-flashinfer-contest` and `/data1/fanqixuan/src/skills/kda`. Btorch-specific additions are Torch golden/state handling, three-scale mandatory activity plots, independent execution/gradient contracts and upstream formatting. KDA first-party documentation is CC BY 4.0 and code Apache-2.0; third-party material retains its own license. Check attribution/license when reusing implementations; do not infer a license for a source that lacks one.

## btorch templates

- `assets/templates/glif3_rsnn.py` instantiates GLIF3, AlphaPSC and SparseConnection; RecurrentNN owns sequence execution at all three scales.
- `assets/templates/stateful_module.py` subclasses MemoryModule, uses `register_memory("q", 0.0, n_neuron)` and supplies both `single_step_forward` and a continuous `multi_step_forward`; the latter is the sequence kernel/op boundary. RecurrentNN owns network scheduling, not a replacement primitive sequence implementation. Its cumulative-drive toy does not duplicate a built-in neuron equation.
- `assets/templates/native_operator.py` shows independent native Torch inference/training operators, fake registration and a registered backward VJP. Replace the toy calculation with the confirmed model's kernel; only implement paths the user requests.

Scripts are limited to model phase/state/evidence checks. Upstream uses ordinary Git worktrees, the target btorch pre-commit configuration and tests, followed by `gh pr create --draft --body-file pr.md` or the installed `glab mr create --draft` equivalent.
