# Generate and contribute a kernel

The kernel-gen harness starts with an actual Torch model, validates its activity, then checks kernel correctness, profiles candidates and validates the full workload. You can start with a model description, an existing Python model or an OOT kernel. A custom neuron, fused network, sparse layout, quantized forward or alternative backward algorithm can use the same workflow.

## Set up external tools

Use the installed [Humanize](https://github.com/PolyArch/humanize), [KernelWiki](https://github.com/DongyunZou/KernelWiki) and [ncu-report-skill](https://github.com/DongyunZou/ncu-report-skill). They are external dependencies, not repository submodules. Reuse existing installations; otherwise follow `skills/btorch-kernel-gen/README.md` and each upstream's installation instructions. Expose their skills and `skills/btorch-kernel-gen` through your agent's supported discovery.

In the task workspace, write `docs/draft.md`, invoke `/humanize:gen-plan`, then `/humanize:start-rlcr-loop`. Phase 1 finds a correct implementation; Phase 2 explores measured optimization directions; Phase 3 validates the complete workload and justified specialization. Hardware and budget are task-specific.

## Validate the Python model first

Start with `skills/btorch-kernel-gen/assets/templates/glif3_rsnn.py`: it constructs actual GLIF3, AlphaPSC and SparseConnection Modules and delegates sequence execution to RecurrentNN. The same factory builds one neuron, an upstream/downstream pair and a larger network. `stateful_module.py` shows a new model's register_memory pattern; it does not duplicate a built-in neuron.

Run no-input, impulse, sustained-input and step-off protocols, then connection/delay and network cases. Produce an activity figure at all three scales. Include every relevant adaptation/receptor channel, input, spikes, membrane and PSC. Inspect state and event timing as well as outputs. See `tests/models/neurons/test_neurons.py` for GLIF3 examples.

A newly generated or changed model becomes golden only after you confirm the actual results and figures. An agent should not issue that confirmation on your behalf. Existing btorch Python models are the golden for matching built-in semantics; there is no second hand-written reference simulator.

[NESTML](https://nestml.readthedocs.io/en/latest/) provides useful examples of explicit state, units and events. Its [STDP windows](https://nestml.readthedocs.io/en/latest/tutorials/stdp_windows/stdp_windows.html) and [stochastic model](https://nestml.readthedocs.io/en/latest/tutorials/gl_model/gl_model_tutorial.html) tutorials inform additional plasticity/statistical checks without changing your Torch model.

```python
# MemoryModule template: registered state and a single-step rule.
self.register_memory("q", 0.0, n_neuron)

def single_step_forward(self, x):
    self.q = self.q + environ.get("dt") * self.gain * x
    return self.q
```


For the MemoryModule toy, initialize its state and call its continuous sequence entrypoint:

```python
import runpy
import torch
from btorch.models import environ, functional

CumulativeDrive = runpy.run_path(
    "skills/btorch-kernel-gen/assets/templates/stateful_module.py"
)["CumulativeDrive"]
model = CumulativeDrive(3, device="cpu", dtype=torch.float32)
x_seq = torch.ones(10, 2, 3, dtype=torch.float32)
functional.init_net_state(model, batch_size=2, device="cpu", dtype=x_seq.dtype)
with environ.context(dt=1.0):
    output_seq = model.multi_step_forward(x_seq)
```

Primitive neuron/PSC `multi_step_forward` owns the continuous sequence semantics and may dispatch a registered kernel/operator. RecurrentNN composes network timing; its current step scheduler mainly calls single-step methods, and can later consume primitive multi-step operators. Do not use `make_rnn` as a replacement for a primitive sequence entrypoint. Requested state histories require an explicit adapter exposing that entrypoint’s actual records. In particular, check each PSC path’s initial/final-state contract: the existing Torch convolution path is stateless, so it is not interchangeable with stateful continuation.

## Separate the three execution paths

| Path | Responsibility |
|---|---|
| Inference / inference | Compute outputs and update state; retain requested records, but no backward-only tape |
| Training forward | Compute outputs/final state and retain or recompute the information backward needs |
| Backward | Compute VJPs for inputs, initial states and parameters using that invocation's context |

`.train()` and `.eval()` do not choose the kernel path. Whole-invocation autograd demand does: `no_grad()`/`inference_mode()` select inference; upstream trainable tensors still require training forward when network parameters are frozen. Inference is stateful, even though it has no backward tape.

Use native Torch custom operators, fake/meta and autograd registration. Backward can invoke a different operator or algorithm. `native_operator.py` is a runnable registration template, with separate inference/training entrypoints and `KERNEL_BINDINGS`; its affine computation is an API example, not a neuroscience golden.

The following two snippets run from the repository root in one Python process. They only validate the affine API template; they do not validate a neuron or RSNN kernel.

```python
from btorch.models.kernels.native import load_kernel_package

import torch

bindings = load_kernel_package("skills/btorch-kernel-gen/assets/templates/native_operator.py")
x = torch.randn(5, 2, 3, requires_grad=True)
gain = torch.ones(3, requires_grad=True)
result = bindings["affine"](x, gain)
```

Parameters and initial states must be explicit participating tensor inputs. Your binding defines the tensor output/state/record contract; your Module adapter handles write-back.

## Check model and gradient parity

```python
from btorch.models.kernels.native import (
    ModelParityCase, ModelParityOptions, check_model_parity,
)

report = check_model_parity(
    [ModelParityCase(
        "broadcast_gain", lambda x, gain: x * gain,
        bindings["affine"], (x, gain),
    )],
    options=ModelParityOptions(atol=1e-5, rtol=1e-5),
)
report.assert_close()
```

For an inference-only kernel, select only validation contexts with `ModelParityOptions(modes=("inference", "no_grad"), check_gradients=False)`. These options select tests, not runtime dispatch. A grad-enabled run with frozen inputs is not evidence that the training kernel executed.

Each execution starts from independent copies of the same explicit inputs, parameters and initial states. Outputs may include final states and user records, so their gradients participate too. Supply representative and edge cases, including delay wrap, chunk continuation, shared parameters and multiple forwards before backward.

Use Torch `opcheck` for registration and `gradcheck` where numerical derivatives apply. A surrogate/STE is not the finite-difference derivative of a hard threshold. For EventProp or another gradient definition, supply `gradient_oracle`; do not compare incompatible gradients. Quantized and stochastic models need their declared reference and accuracy/statistical contract. Missing validation is reported as missing, not passed.

## Record and gate the run

A suite callable returns `checks`, `figures`, `source_files`, `required_checks` and execution/gradient `metadata`. Checks identify `single`, `pair` or `network`, status and concrete detail. Required checks should cover the selected task, not unrelated features.

`plot_response` aligns input, raster and state panels, separates adaptation/receptor components, and keeps neuron colors consistent. Pass `ResponsePlotOptions(units=..., time_unit="ms", batch_index=0)` for explicit units and trial selection. The example uses the same units for plots and JSON evidence. Unknown state layouts retain all indexed values, without guessing neuron identities; values are not clipped or normalized for appearance.

```bash
python skills/btorch-kernel-gen/scripts/check_model.py verify --suite examples/kernel_generation.py:run_suite --output run/report.json
# Run this only after you, the user, have inspected and confirmed a new model:
python skills/btorch-kernel-gen/scripts/check_model.py confirm --report run/report.json --reviewer YOUR_NAME --output run/confirmation.json
python skills/btorch-kernel-gen/scripts/check_model.py gate --report run/report.json --confirmation run/confirmation.json
```

Keep candidate and optimization records with the planning/review tool; this supplement only supplies model validation and upstream helpers. Keep correctness, profile and before/after performance evidence. Measure inference, training forward and backward separately, including memory and compile/capture costs. OOT output paths are yours to choose.

The evidence gate checks declared results and file integrity. It does not establish scientific correctness, authenticate a reviewer, or prove that a supplied figure depicts the run. Review the suite assertions, actual traces and figures. The bundled GLIF3 example performs only finite-value, resting-silence and spike-presence smoke checks; it does not test delay, adaptation dynamics or kernel parity.

## Prepare and publish upstream

Bring the same implementation into a dedicated target btorch worktree; preserve unrelated changes. Follow existing model/kernel/public-export and tests organization. Only for upstream, use `fig_path()`/`save_fig()` for figures, with `file=__file__` from the calling test so a shared helper does not change their location.

The contribution must be formatted with the target btorch `.pre-commit-config.yaml`: install hooks, run `pre-commit run --files ...`, apply fixes and repeat until passing. Rerun relevant tests after formatting, plus Ruff and bilingual docs checks. New tests should explain meaningful usage and edge cases, not merely assert assignments.

After evidence and formatting checks, use normal Git and GitHub/GitLab CLI tools to commit selected files, push and create or update the requested Draft PR/MR. The supplement has no publishing automation. Include activity figures, parity, benchmark/profile summaries, reproduction commands and source/license attribution. It does not automatically merge. If destination or authentication is unavailable, retain the prepared local contribution and report that blocker.

For example, adapting paths and the target branch to your checkout:

```bash
git worktree add -b contribution/glif3-kernel ../btorch-contribution origin/main
cd ../btorch-contribution
pre-commit install --install-hooks
pre-commit run --files btorch/models/neurons/my_model.py tests/models/neurons/test_my_model.py
pytest tests/models/neurons/test_my_model.py
git add btorch/models/neurons/my_model.py tests/models/neurons/test_my_model.py
git commit -m "Add validated neuron model"
git push -u origin contribution/glif3-kernel
gh pr create --draft --title "Add validated neuron model" --body-file pr.md
# GitLab alternative, using your installed glab version's description option:
glab mr create --draft --title "Add validated neuron model" --description "$(cat pr.md)"
```

The scripts only check scientific evidence. Invoke Humanize directly for plans/review. Validate and measure declared execution paths; an inference-only kernel does not need an unrequested backward. Model approval tracks model/case/figure/raw-data changes, not timestamps or benchmark metrics alone.
