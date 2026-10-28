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

## Prepare and render

Model definitions describe the mathematics. Preparation binds tensor layouts,
computed coefficients, residual storage and compiled kernels. Models that
reconstruct state during backward declare `checkpoint_state`,
`checkpoint_interval` and `replay_state`; models that precompute coefficients
declare `prepared_parameters` and `prepare_parameters`.

Primitive and fused generation apply separate namespaces to the shared scalar
stages. Backend renderers provide tensor access, math intrinsics and source
wrappers; they do not define model equations. Inspect generated code with
`generate_cell_source` or `generate_fused_source` before compiling an extension.
