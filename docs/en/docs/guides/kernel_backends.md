# Neuron and connection backends

Select neuron/PSC dynamics independently from sparse propagation:

```python
from btorch.models import KernelOptions
from btorch.models.rnn import RecurrentNN

rnn = RecurrentNN(neuron, synapse, backend=KernelOptions(
    neuron="triton", connection="auto", connection_method="push",
    merge_synapse=False,
    enable_fp_fusion=False, precise_division=True,
))
```

A backend string selects only the neuron/PSC implementation. `torch`, `triton`,
`tilelang` and `cupy` are built in; externally registered names work through the
same argument. GPU packages are optional. Install Triton with `pip install
'btorch[gpu]'`; install TileLang or CuPy separately for the CUDA environment you
use. Cell backends reuse the existing sparse runtime; selecting a cell backend
does not migrate or duplicate sparse algorithms.

`neuron` selects the implementation shared by neuron and PSC dynamics;
`connection` selects `"triton"`, `"cusparse"`, `"torch"`, or `"auto"`.
`"cusparse"` requests native CUDA CSR products through PyTorch, for FP32/FP64;
its weight VJP uses the shared PyTorch sampled edge-gradient implementation.
`"torch"` is the general reference, including CPU and gather/scatter fallbacks.
`connection_method` independently
requests `"pull"` (destination-driven) or `"push"` (spike-list SpMSpV).
The default `"auto"` inherits the sparse connection's method and activity hints.
An explicit request overrides hints and raises if the device, dtype, batching,
determinism setting or backend cannot implement it. Currently explicit push
requires CUDA, Triton, FP32, unbatched weights and nondeterministic accumulation.
Dense connections are unaffected by this sparse-only option.

For a standalone sparse connection, use `SparseConnection(..., method="push")`
or assign `connection.method = "pull"` between runs. For a runtime override:

```python
from btorch.sparse import runtime

with runtime.use_method("pull"):
    output, records = rnn(inputs)
```

A thread-local runtime scope takes precedence over RSNN `connection_method`, which takes
precedence over the connection's own `method`; `"auto"` inherits the enclosing
choice. Configure runtime scopes outside compiled execution, like backend
scopes. Scoped method changes are guarded by `torch.compile` and select a new
route as needed. Compiled native RSNNs bind explicit kernel options at trace time;
the standalone connection's default plan is not modified by that binding.
Changing RSNN options invalidates CUDA graph capture. Eligible native Triton
recurrence emits forward spike lists directly; standalone products and
unsupported compositions use the device compaction path.

The training/inference example in `examples/rsnn_izhikevich.py` supports the
50,000-neuron/2,500-indegree reference size through `--neurons 50000 --density 0.05`,
heterogeneous dual-exponential PSCs, and explicit Triton spike-list push.
Connectivity is constructed from sparse edge arrays and moved to the selected
device. For a short run and a backend comparison:

```bash
python -m examples.rsnn_izhikevich --neurons 256 --steps 32 --mode inference --device cuda --neuron-backend triton --method push --cudagraph
python -m examples.rsnn_izhikevich --neurons 256 --steps 32 --device cuda --neuron-backend triton --method pull --sparse-backend cusparse
```

The example prints realized firing rate. Connectivity density is independent
of firing rate and does not act as a low-activity hint.

Registered cell sequence operators require PyTorch 2.4 or newer.
Training supports first-order BPTT. `create_graph=True` is explicitly rejected;
these handwritten kernel VJPs do not provide higher-order derivatives.

Kernel integration follows these shared boundaries:

| Responsibility | Implementation |
|---|---|
| Public model parameters, state and backend options | Neuron/PSC Modules and `KernelOptions` |
| Model-to-tensor adaptation and validation | `models/kernels/adapters.py` |
| Model equations and supported state/parameter layouts | `models/kernels/templates.py` and `contracts.py` |
| Native cell source generation and paired neuron/PSC fusion | `models/kernels/backends/engine.py` and `fused.py` |
| Sequence registration, autograd and recurrent scheduling | `models/kernels/operators.py` and `recurrent_operator.py` |
| Sparse route selection and active-work propagation | `sparse/runtime/planner.py`, `kernels_triton_spikelist.py`, `kernels_triton_push.py` and `kernels_triton_tasks.py` |

New cell implementations should extend the shared templates or the public
backend registration API rather than duplicate Module adaptation, recurrent
scheduling or sparse kernels. Benchmark-specific wrappers belong in
`benchmarks/`; reusable runtime code belongs in the package. Cell inference,
training forward and backward share a semantic contract but retain their own
storage requirements. Connection propagation retains its own route, prepared
workspace and gradient implementation.

The model set is IF, LIF, ALIF, ELIF, GLIF3, Izhikevich and ExponentialPSC, AlphaPSC,
AlphaPSCBilleh, DualExponentialPSC. Native surrogate templates are Sigmoid,
ATan, ATanApprox, Erf, SuperSpike, Triangle, and PoissonRandomSpike. Set the
Module's `surrogate_function` normally; the adapter selects the matching
template and its non-trainable alpha/damping values. Custom Python surrogates
remain available with Torch, but require an explicit native implementation to
use the GPU cell kernels. Unsupported native surrogates and trainable surrogate
hyperparameters raise instead of silently changing semantics.

Recurrent gradient algorithms are independent of kernel selection. Enable
exact compact-support execution with `gradient=BPTT(sparse=True)`:

```python
from btorch.models import BPTT, KernelOptions, RecurrentNN
from btorch.models.surrogate import Sigmoid, Truncated

neuron.surrogate_function = Truncated(Sigmoid(alpha=2.0), width=0.4)
network = RecurrentNN(
    neuron, synapse,
    gradient=BPTT(sparse=True),
    backend=KernelOptions(neuron="triton", connection_method="push"),
)
```

`Truncated` explicitly changes the derivative to zero outside
`abs(surrogate_input) < width`, including the boundary; forward spikes are
unchanged. Width uses the neuron's normalized threshold-distance units, not
raw voltage. Triangle already has compact support and needs no wrapper.
`BPTT(sparse=True)` exploits that support without changing the chosen derivative,
reset gradients, or physical-state adjoints. The default `BPTT()` evaluates the
full gradient. An unbounded surrogate is never silently truncated.

Generated Triton neuron templates emit a compact near-threshold list directly
in their existing forward launch, using the same block-local compaction helper
as forward spike lists. Every block writes its count, including zero for a
silent block; preallocated capacity covers all neurons. There is no standalone
backward-activity compaction kernel or host-side count read.

The training tape keeps the near-threshold list separately from a combined
list containing near-threshold **or firing** neurons. Single-sample joint
input/weight reverse consumes the combined list: firing neurons outside the
surrogate window can still contribute weight gradients. Input-only reverse
consumes the near-threshold list, including across samples. Both reuse the
existing source-major CSR reduction kernel; a small output kernel materializes
zero adjoints for inactive rows and combines the existing 512-edge row partials.
The partition grid depends on topology shapes, not live activity, and programs
stride over additional segments of unusually long rows. Physical-state adjoints
remain complete.

Batched trainable weights retain the existing masked, sample-sharing fused
reverse because separate per-sample list traversal would add an edge-gradient
pass or require competing updates to shared parameters. That choice uses only
static shapes and gradient requirements, never per-step profiling. Other native
renderers retain the support mask and their existing execution paths.
Delay queues and external synaptic inputs currently retain full propagation
adjoints because their dependencies extend beyond the surrogate. Checkpointing,
CPU offload and compiled training keep the same public state interface;
`torch.compile(mode="reduce-overhead")` can capture training, while the explicit
`cudagraph=True` option remains for inference.

For longer training sequences, combine checkpointing with CPU offload:

```python
network = RecurrentNN(
    neuron, synapse, chunk_size=200, chunk_mode="split",
    grad_checkpoint=True, cpu_offload=True,
    update_state_names=("neuron.v",),
)
```

Checkpointing recomputes private activations during backward. CPU offload moves
saved private tensors and returned histories to host memory while parameters
and caller inputs remain on their original devices. Returned histories remain
differentiable. With offload enabled, full traces use CPU memory proportional
to sequence length; GPU recording buffers stay bounded by chunk size. Model
weights and caller-owned GPU inputs remain resident. Select only the records
you need, or use `update_state_names=()` to omit state traces.

Surrogate primitives and derivatives are plain Torch functions, compatible
with the existing autograd and `torch.compile` paths. The former `btorch.jit`
shim and `BTORCH_JIT` environment option have been removed; there is no
TorchScript cold/warmed execution phase.

## Gradients and precision

At sequence entry, btorch checks grad mode and all participating floating inputs,
initial states, connection weights and model parameters. Training runs retain the
history needed for backward; inference retains final state and requested records.
`.train()` and `.eval()` do not select this path. A frozen RNN still propagates a
trainable encoder's gradients. `no_grad()` and `inference_mode()` select inference.
When inference record selection omits pre-spike voltage or adaptation, their final
module states still update without retaining a full temporal history. Requesting
those records returns their complete trajectories like any other public record.

Primitive reverse materializes only the input, initial-state and parameter VJPs
requested by autograd. Freezing the drive or initial memories omits their gradient
buffers and kernel stores. The reverse still propagates the internal state
adjoints needed for trainable parameters. A trainable encoder keeps its input VJP
even when every neuron and PSC parameter is frozen.

Standalone GLIF3 training on the built-in GPU backends saves adaptation
checkpoints and replays the forward recurrence during backward when no full
`Iasc` history is needed. Requesting `Iasc` records retains that full history
instead. This saves training memory at the cost of extra backward computation;
it does not change returned values or add storage to inference. Native recurrent
LIF+ExponentialPSC training uses bounded checkpoint replay when physical state
records are omitted, without refractory state, delay or receptor merging. Other
recurrent configurations retain physical-state histories. Unrequested pre-spike memories are
kept only as differentiable final values; requesting their full records retains
the corresponding trajectories.

Current built-in kernels accept FP32, FP16 or BF16 inputs and use FP32 state and
accumulation; output sequences follow input dtype. This is an implementation
limitation, not a settled precision/promotion policy. Time is first, normally
`[T,B,N]`; initial state has
no time axis. Model adaptation and receptor axes remain explicit. Integer topology,
receptors and delays receive no gradients.

Native cell pointer bindings retain input and parameter storage dtype and
strides, including sliced or transposed views with nonzero offsets. Register
loads perform the calculation-precision conversion inside the kernel. They do
not require an FP32 contiguous copy of each input. Low-precision training keeps
the producer's existing spike buffer as a private residual, independently of
the public raster's dtype.

Use standard Torch AMP without converting the model or its memories:

```python
scaler = torch.amp.GradScaler("cuda")  # normally needed for FP16, not BF16
with torch.autocast("cuda", dtype=torch.float16):
    spikes, records = network(inputs)
    loss = loss_fn(spikes)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

For native recurrent execution, dense linear connections use the selected
FP16/BF16 autocast dtype. The differentiable master-weight conversion happens
once at trajectory entry; backward and checkpoint replay retain that forward
precision even when run outside autocast. Neuron/PSC parameters and memories
are not implicitly downcast. Sparse pull and push keep their operand-promotion
policy under autocast; an FP32 edge tensor with BF16 spikes therefore produces
FP32 current. Autocast does not itself request lower-precision sparse weights.
Delay history retains its incoming storage dtype, matching `Delay.update()`.

Native Poisson surrogate lowering is currently narrower: inputs, physical
states, participating parameters, and explicit uniform tensors must all be
FP32. Invalid reduced-precision calls fail before the default Torch generator
is advanced. This restriction applies to sampled and non-spiking Poisson modes;
plain Torch Modules retain their normal dtype behavior, and deterministic
native surrogates retain the FP16/BF16 support described above.

`KernelOptions(enable_fp_fusion=False)` asks built-in GPU cell kernels to preserve
separate FP32 rounding for operations that a compiler could otherwise contract,
such as a multiply followed by an add. The default is `True` for the historical
throughput-oriented behavior. This setting controls contraction only: it does not
enable or disable approximate exponential, division or other general fast-math
transformations. It is a kernel compilation choice, not part of the neuron or PSC
equations, and changing it invalidates compiled and captured kernel routes.
Surrogate functions use ordinary Torch operations without TorchScript.
Native ATan backward follows the same square/add expression; its contraction
is controlled by this option rather than an unconditional fused operation.

`KernelOptions(precise_division=True)` separately requests accurately rounded
floating-point division from built-in GPU cell backends. Its default is `False`,
which preserves each backend's existing division implementation when an explicit
options object is supplied. Plain backend strings use accurately rounded
division for ALIF/ELIF; these new families need it to preserve scalar parameter
VJPs in the validated configurations. Other families retain their historical
string defaults. Explicit `KernelOptions` always take precedence. This is useful
for long nonlinear recurrences where small division errors accumulate, but it can
reduce throughput. It does not promise bitwise equality in general and does not
control exponential or square-root approximations, denormal handling, or
flush-to-zero behavior. Division precision and floating-point contraction are
independent options and neither changes `NeuronSpec` or the model equations.

`KernelOptions(precise_exp=True)` requests accurate exponential evaluation.
It is opt-in and defaults to `False`; no model shorthand enables it. It affects
exponentials independently of division precision and multiply-add contraction.
Use it when the extra accuracy matters for the workload. For example:

```python
options = KernelOptions(neuron="triton", precise_exp=True)
```

The cell benchmark exposes the same choice through `--precise-exp` and records
it with the measurements.

GLIF3, AlphaPSC and DualExponentialPSC state kernels follow the Module's stable exponential-Euler update
using `expm1`, including zero adaptation rates and the infinite time-constant
limit. This avoids subtracting nearly equal exponential values for small updates.
The compilation options above still apply; matching equations does not guarantee
bitwise equality across backends or long recurrent trajectories.

**Breaking — corrected mathematical sensitivities:** exponential Euler now
evaluates `expm1(dt * L) / L` with a continuous series near zero in both Torch
and generated kernels. Its derivative with respect to `L` at zero is
`dt**2 / 2`; the previous zero branch incorrectly returned zero for this term,
and the quotient backward lost accuracy near zero. Parameter gradients can
therefore change in these limits. The differential equations and event order
are unchanged, and accurate exp remains opt-in. Separately,
`GLIF3.forward_exact_no_spike()` now uses `v_rest` for its equilibrium when it
differs from `v_reset`, matching the membrane equation.

TileLang cache identities include the selected NVCC and ptxas versions and the
CUDA device-library contents. Switching CUDA toolchains therefore compiles a
separate artifact rather than reusing a binary produced by another toolkit.
The fingerprint is computed once per process. Select the toolkit before
starting Python; changing compiler installations inside a running process is
not supported. CuPy and Triton do not probe this TileLang toolchain.

Each call reads scalar `environ.dt`. Changing it does not specialize generated
kernel source or recompile the sequence graph; backward uses that call's dt.
CUDA graph replay keeps the captured value unless its device input is updated.

## State and records

Initialize modules with `init_state` or `functional.init_net_state`. Final state
remains differentiable across calls until you explicitly reset or detach it.
Returned RSNN records can contribute an auxiliary loss. Each invocation owns the
saved data it needs, so two forwards followed by one backward are supported.

Native recurrent chunks prepare paired neuron/PSC kernel bindings and supported
Triton sparse workspaces once per invocation. Interior kernels write spikes and
requested histories directly into their sequence destinations. Working storage
belongs to that invocation; it does not overwrite earlier windows or the inputs
saved for backward. Final-state ownership and chunk-boundary transfers remain
explicit, preserving the existing `MemoryModule` reset/detach interface.

Paired backward likewise prepares its kernel and static parameter bindings once
per trajectory, alternating gradient destinations so incoming adjoints remain
readable and writing interior input gradients directly into their sequence.
Checkpoint replay reuses two bounded block buffers. Parameter-gradient
reductions and replay retain their existing numerical semantics.
Explicit synaptic inputs and delay updates use
invocation-owned forward buffers rather than allocating new tensors each step.

This uses the existing `RecurrentNN` options for chunking, gradient checkpointing
and CPU offload. Explicit `cudagraph=True` remains inference-only; supported
compiled training capture uses `torch.compile(mode="reduce-overhead")`. There is
no additional user-facing workspace or preparation API.

Explicit kernel options select a stateful PSC sequence recurrence. The historical
plain `backend="torch"` PSC multi-step convolution starts from rest, truncates its
impulse response at `kernel_len`, and does not update state; use
`backend=KernelOptions(neuron="torch")` for the stateful reference path.

With `ConductancePSC(step_mode="m")`, pass `v_seq` by keyword to compute current
from a voltage sequence:

```python
current_seq = conductance_psc(z_seq, v_seq=v_seq)
```

The leading voltage dimension must have length `T` or `1`; length one broadcasts
the voltage over time, and the remaining dimensions broadcast to the
post-synaptic batch and neuron shape. This call updates the registered
conductance and delay state, so separate calls continue the sequence.

Backend selection affects the no-voltage sequence contract:

- Plain `backend="torch"` with omitted `v_seq` keeps the historical stateless
  convolution from rest and its `kernel_len` behavior.
- `backend=KernelOptions(neuron="torch")` selects the stateful adapter reference.
- CuPy, Triton, and TileLang reuse the existing exponential PSC operator to
  produce the conductance sequence, then apply `g * (E - v_seq)` as an ordinary
  Torch expression. Omitting voltage emits the stateful CUBA conductance.

Native backends support standalone single-step and sequence calls with dense
or sparse connections, delay continuation, gradients and CUDA Graph replay.
Native recurrent COBA execution is not supported.

Use `ConductanceRecurrentNN` for a connected Torch COBA network:

```python
from btorch.models.rnn import ConductanceRecurrentNN

network = ConductanceRecurrentNN(
    neuron,
    conductance_psc,
    backend="torch",
    update_state_names=("neuron.v", "synapse.psc"),
)
spike_seq, states = network(input_sequence)
```

At step `t`, this schedule computes current from the conductance and post-reset
voltage left by step `t - 1`, advances the neuron, then updates conductance from
the emitted spike. A recurrent spike therefore affects membrane current starting
on the following step. Dense and sparse connections, delays, chunk continuation,
checkpointing, input modules, and public state records use the inherited
`RecurrentNN` machinery.

The class requires the exact `ConductancePSC` type and plain `backend="torch"`
on every backend-bearing module in the network, including input transforms. A
`KernelOptions(neuron="torch")` value is also rejected because kernel options
select an execution implementation, while this class defines an explicit
scientific schedule. Native recurrent COBA remains unsupported pending validated
recurrent kernels. The distinct 002HIP conductance models are not configurations
or lowerings of this schedule.

`torch.compile(network)` uses the inherited fixed-unroll scheduler and supports
the default spiking surrogate, state continuation, and training gradients. The
default btorch surrogates define a custom forward-mode `jvp`, so PyTorch Dynamo
currently rejects the stricter `fullgraph=True` mode for ordinary `RecurrentNN`
and `ConductanceRecurrentNN` alike; this is not specific to conductance
scheduling.

The explicit `cudagraph=True` runner is inference-only. Under grad mode it
rejects trainable inputs, parameters, and buffers, including initial membrane
and conductance states; use `torch.no_grad()` or `torch.inference_mode()` for
inference capture. The conductance schedule recaptures after replacing model
parameter/buffer storage or changing a Python/CPU scalar timestep. An in-place
CUDA tensor timestep update uses the existing graph because device work reads
its live storage.

`merge_synapse=False` retains individual receptor states for EPSC/IPSC queries.
Explicit merge is restricted to compatible linear dynamics with frozen compatible
parameters or actual shared parameter identity. Independent trainable parameters
cannot be merged because their current values happen to match. Independent
receptor records cannot be recovered from merged state; incompatible requests
raise. Consult the tested support scope before enabling merge for a new model.

The initial sequence integration supports identity, dense `nn.Linear` and
`SparseConnection` connections. Custom connection modules and sparse
`member_pattern_batch` are not supported by this path. Receptor merging is a
heterogeneous RSNN option; standalone neuron/PSC calls reject it. External cell
backends compose with the same connections through their registered callbacks;
they do not automatically acquire the built-in paired cell fusion.

See [the external-kernel guide](oot_kernels.md),
[the training tutorial](../tutorials/kernel_training.md) and
[the sparse guide](sparse_connectivity.md).

## Low-activity recurrent propagation

Cell kernels and sparse propagation are selected independently. To reuse
Btorch's Triton spike-list SpMSpV implementation in a recurrent
network, construct its `SparseConnection` with
`Hints(firing_rate_hz=4.0)` (from `btorch.sparse`). A supplied rate strictly below
5 Hz requests source-driven propagation on CUDA; the native recurrent sequence
operator carries the prepared connection route into each timestep. Higher
rates use the existing density policy. Deterministic execution and unsupported
layouts retain their fallback behavior. The hint never caps actual spikes.

The initial neuron and fused PSC/next-neuron Triton templates emit active IDs
and local degree prefixes directly when there are no delays or external
synaptic inputs. The existing packed fanout consumes these invocation-owned
lists, giving two interior forward launches: propagation and the fused
producer. This capability is shared by model templates, not specific to one
neuron or synapse equation. Explicit push uses the same path.

For composed/delayed sources or other neuron renderers, one Triton kernel
combines output clearing, compaction and the local degree scan before the same
fanout. There is no global CUB scan. Only active-source fanout is read, and
source-major weights are prepared once per trajectory. Public spike rasters
and training residuals retain their existing representation. See
[sparse connectivity](sparse_connectivity.md) for hint precedence, recording
tradeoffs and numerical behavior.
