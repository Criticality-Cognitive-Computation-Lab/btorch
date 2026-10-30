# Generated kernel model contracts

btorch generates standalone neuron/PSC kernels and fused neuron–PSC steps from
the same model definition. Triton, CuPy and TileLang render that definition using
their own storage and execution syntax.

For an existing model, select its backend through `KernelOptions`; see
[kernel backends](../guides/kernel_backends.md). The contracts below are for
developers adding model equations to the generator. To supply a separately
implemented kernel, use [out-of-tree kernels](../guides/oot_kernels.md).

## Define the model

Register a `ModelTemplate` with `register_model_template` from
`btorch.models.kernels.templates`. Built-in definitions live in
`btorch/models/kernels/definitions/`.

| Declaration | Purpose |
| --- | --- |
| `state_names`, `parameter_names` | Fix the tensor argument order |
| `history_names`, `history_sources` | Name supported records and their state shape sources |
| `state_extra_dims`, `parameter_extra_dims` | Describe component axes beyond batch and neuron dimensions |
| `fragments` | Supply forward and reverse scalar stages |
| `module_factory` | Construct the independent Torch model used for numerical validation |

The Module opts into its registered definition with a `kernel_model` class
attribute. A subclass that changes the equations must declare its own binding;
an inherited binding is not used automatically. Continue to register physical
state through `register_memory` so state initialization and continuation retain
the ordinary Module interface.

## Define forward and reverse together

The fragment builder returns `BackendFragments`. Forward stages contain scalar
assignments in Python expression syntax. Reverse stages compute VJPs for the
input, initial state and participating parameters, including losses on recorded
states. Use the existing definitions as examples: `izhikevich.py` has voltage
and adaptation state; `alpha.py` has coupled PSC states.

If a record adjoint must enter between reverse stages, insert
`RecordCotangent(record_name)` at that position. Otherwise state-record
adjoints join at the reverse-step boundary. This ordering matters when a record
and the output refer to the same physical state.

Supply reverse equations explicitly. Tracing forward operations alone does not
specify a spike surrogate derivative or detached-reset semantics. Validate the
generated VJPs against the original Torch model, including nonzero initial
states and losses on records.

Sequence boundaries use public `torch.library` schemas with fake kernels and
explicit first-order autograd. Implementations must return fresh, non-aliasing
outputs and leave inputs unchanged. Validate these contracts with
`torch.library.opcheck` alongside numerical full-VJP comparisons.

## Prepare and render

Model definitions describe the mathematics. Preparation binds tensor layouts,
computed coefficients, residual storage and compiled kernels. Models that
reconstruct state during backward declare `checkpoint_state`,
`checkpoint_interval` and `replay_state`; models that precompute coefficients
declare `prepared_parameters` and `prepare_parameters`.

Primitive forward and reverse sequences reuse tensor-free layout and launch
plans, binding current inputs, saved tensors and cotangents and allocating
independent results on each invocation. Saved-tensor hooks still control tape
storage; restored tensor layouts participate in reverse preparation. Changed
layouts, dtypes or model options require a new plan; the runtime `dt` remains dynamic.
Warm the intended layouts before CUDA Graph capture.

Recurrent preparation also binds the repeated stage's argument slots and
storage layout. Paired stages reuse tensor-free plans across chunks, binding
current states, parameters, coefficients and runtime `dt` on each invocation.
Cached plans retain no input, recording or backward tensors.
Pointer-only runtimes use the original tensor storage;
TileLang exposes its flat buffer ABI. Changing sequence rows must not inherit
the first row's alignment specialization. Prepared Triton launches resolve the
executable once and retain profiler hooks. A recurrent invocation binds its
current CUDA stream once; the next invocation selects its stream afresh.
Triton reuses the compiled argument ABI without repeating JIT preparation.
CuPy recurrent calls also bind their invocation's stream through its public API;
standalone calls continue to select the caller's current stream.
Prepared pointer arguments retain their owning tensors. Triton and CuPy interior stages
update positional slots with byte offsets computed from the actual time stride;
state views are materialized at checkpoint and final-state boundaries. The
neuron/PSC template metadata supplies the state ordering for every model.
TileLang retains prepared tensor views and launches through its public kernel
call interface; execution-backend argument and stream handling remain TileLang's
responsibility.

Native paired reverse stages reuse model/layout plans across calls and bind
stream-specific argument slots once per trajectory layout. Gradient working
sets remain invocation-owned; changed cotangent or restored tensor layouts
select another prepared variant. Native CSR stages likewise reuse tensor-free
launch recipes, while backend autotuners retain their execution configurations.

The cuSPARSE route prepares CSR tensors, transposed values and dense workspaces
once per recurrent invocation. Standalone products keep their one-off execution
path. Both paths retain explicit input and stored-edge weight VJPs, including
checkpoint replay and CPU offload; preparation never caches trainable values
across invocations. Capture training preparation with the backward operation so
value refreshes replay after optimizer updates.

Spike-list propagation can bind a complete trajectory during preparation and
consume rows by timestep index, reusing the producer's device queue. This
retains the original storage and avoids constructing and validating a new
tensor view at every interior step.

Primitive and fused generation apply separate namespaces to the shared scalar
stages. Backend renderers provide tensor access, math intrinsics and source
wrappers; they do not define model equations. Inspect generated code with
`generate_cell_source` or `generate_fused_source` before compiling an extension.

Triton and CuPy sequence kernels accept the timestep count at invocation, so
changing only the sequence length reuses their generated source and compiled
kernel. Trailing shapes, strides, dtypes and model options still specialize the
kernel. The current TileLang wrapper uses static loop and buffer extents and
therefore specializes on sequence length. CUDA Graphs remain shape-specific.
