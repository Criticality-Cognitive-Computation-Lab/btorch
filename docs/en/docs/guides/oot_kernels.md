# Out-of-tree cell kernels

An external Python package can register a cell implementation without a btorch
fork. Registration selects numerical execution; btorch continues to own the
stateful Module API, autograd dispatch and complete-sequence operator boundary.

```python
from btorch.models.kernels import register_cell_backend

register_cell_backend(
    "lab_cell",
    forward=forward,
    backward=backward,
    supports=lambda spec, device: spec.model == "lif" and device.type == "cuda",
)
neuron = LIF(128, step_mode="m", backend="lab_cell", device="cuda")
```

Names must be identifiers. Duplicates raise unless `replace=True`; built-in names
cannot be replaced. Finish outstanding backward calls before replacement;
pending backward calls reject changed backend or model registrations. Recreate
compiled callables and standalone captured graphs after replacement. RSNN's graph
runner invalidates its cache when the backend revision changes.
A forward-only implementation is inference-only and rejects training at
entry. Unsupported model/device choices raise without a Torch fallback.

## Tensor callback contract

`ModelTemplate` declares state, parameter and history names in a fixed order.
All participating parameters are explicit tensors, preserving their autograd
connections. Callbacks use tuples:

```python
def forward(x_seq, states, parameters, spec, history_names):
    return output_seq, final_states, histories


def backward(x_seq, initial_states, parameters, spec, history_names, histories,
             grad_output, grad_final_states, grad_histories, *, needs_input_grad=None):
    return grad_x, grad_initial_states, grad_parameters
```

Forward returns FP32 output/state/history; the sequence wrapper casts public
output to input dtype. Every result must obey the non-aliasing custom-operator
contract. `history_names` is the union required by the caller and backward, not
an invitation to allocate every residual. The wrapper shares PSC output storage
with its corresponding history. Backward includes cotangents from outputs,
final states and user records. A frozen initial state still needs temporal state
adjoints; `needs_input_grad` trims returned gradients, not the recurrence itself.

For a new model, `register_model_template` requires named declarations, state
shape extensions, history shape sources and trusted one-step forward/backward
callbacks. Registering a backend name alone does not define new equations.
To generate the same model for multiple backends, see
[generated kernel model contracts](../design/kernel_generation.md).

## Generate and validate

`generate_cell_source(backend, spec, n_lanes, n_steps, history_names, ...)` returns
inspectable direct DSL source and ordered buffer declarations. It uses the same
model fragments as built-in execution. Generation does not compile kernels;
warm up before CUDA graph capture. `resolve_cell_backend` exposes a built-in
callback implementation for reuse by an external package.

The public developer helper needs no pytest dependency:

```python
from btorch.models.kernels.testing import (
    ParityCase, ParityOptions, check_backend_parity,
)

report = check_backend_parity(
    "lab_cell",
    [ParityCase(x_seq, states, parameters, spec, history)],
    options=ParityOptions(modes=("training", "inference")),
)
report.assert_close()
```

Create cases using the shapes, initial states, parameter broadcasts, dtypes and
model options your kernel supports. The helper compares your implementation with
the existing btorch Python model, including output, final states, requested records
and input/state/parameter gradients. For a new model outside btorch, provide its
Python implementation through `ParityCase.golden`.

Include nonzero initial states and request records used by auxiliary losses.
Check long trajectories and each surrogate/reset option you support. Reports name
the failing tensor and coordinate and identify the first divergent spike timestep.
Spikes must match exactly; continuous tensors use configurable tolerances.

The comparator does not manufacture an edge-case matrix. Add explicit named cases
for the behaviors your kernel supports: empty and single-step sequences,
noncontiguous drives, declared state or charged records, input/state/parameter
gradients, frozen inputs, `no_grad` and `inference_mode`, differentiable chunk
continuation, and multiple outstanding forwards. Keep each case tied to the
physical or compiler behavior it is meant to catch. The helper keeps your
supplied dtype and parameter values; add cases for each precision, shape,
broadcast, long sequence and model option you advertise.

`ParityOptions(compile=True, capture=True)` requests fullgraph and warmed-up CUDA
replay checks. CPU capture is an explicit skip. Inspect `report.skipped`; a skip
is not evidence of support. Empty/failed reports fail `assert_close`.
With gradient checks enabled, capture also compares replayed forward/backward
outputs and input/state/parameter gradients against eager Torch autograd.

Use `torch.library.opcheck` for custom-operator schema, fake/meta, autograd
registration and compiler integration. It does not establish numerical gradient
correctness. `torch.autograd.gradcheck` uses finite differences and is appropriate
for smooth PSC dynamics; it cannot validate an intentionally substituted spike
surrogate derivative. Spike gradients remain compared to the existing btorch
Module autograd path. See the [PyTorch operator testing documentation](https://docs.pytorch.org/docs/stable/library.html#testing-custom-ops).

See [the executable OOT tutorial](../tutorials/oot_kernels.md). Sparse extensions
continue to use the existing sparse registry described in
[the sparse guide](sparse_connectivity.md); cell registration does not redefine
connection topology, delay layout or sparse algorithms.
