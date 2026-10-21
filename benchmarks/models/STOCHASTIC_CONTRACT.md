# Stochastic ALIF and ELIF native contract

Status: native FP32 stochastic contract and qualification record. This document
does not make the distinct 002HIP neuron equations a btorch golden.

The executable scientific oracle is the existing btorch Python path:

- `btorch/models/surrogate/poisson_random.py:13-52` for stochastic forward,
  backward, and JVP;
- `btorch/models/neurons/alif.py:193-247` for ALIF charge, adaptation, fire,
  reset, and refractory order;
- `btorch/models/neurons/alif.py:382-397` for the ELIF voltage charge;
- `btorch/models/base.py:1206-1234` for single-step and multi-step scheduling.

`tests/models/neurons/test_stochastic_alif.py` already establishes CPU Module
examples for whole-call versus chunk continuation, final global RNG state,
nontrivial zero/one spike rasters, two forwards before one VJP, and the
`detach_reset` gradient branch. Native work must extend those examples rather
than replace them with duplicated recurrence equations.

Current evidence: Slurm 1266 passes 18 CPU/CUDA actual-Module tests, including
whole/chunk continuation, reset VJPs, modes, deterministic non-spiking behavior,
and three changed-input CUDA Graph replays with exact events, physical state,
RNG advancement and final generator state. These tests establish the existing
Module contract; they do not establish native stochastic coverage.

## PoissonRandomSpike semantics

Let `q` be the tensor passed to `PoissonRandomSpike`. In btorch ALIF and ELIF,
the firing input is the dimensionless normalized voltage

```text
q = (v_charge - v_threshold) / (v_threshold - v_reset).
```

For `spiking=True`, the Module evaluates

```text
rate = exp(rho * q) / tau
u = rand_like(rate)
spike = cast(rate > u, q.dtype)
```

There is one uniform draw for every element of `q`. For finite `tau > 0`, the
conditional event probability is

```text
P(spike = 1 | q) = min(max(exp(rho * q) / tau, 0), 1).
```

The implementation does not clamp `rate`; the probability expression above is
only the consequence of comparison with `u in [0, 1)`. Equality uses strict
`>`. A rate of one therefore fires for every ordinary Torch uniform sample.
NaN compares false. Negative or zero `tau`, non-finite `rho`, and a zero
`v_threshold - v_reset` denominator are outside the first native
qualification set; this is a scope restriction, not a proposed Module
validation change.

`tau` is a per-fire-call probability divisor. It is not `environ.dt`, and the
forward does not use `1 - exp(-lambda * dt)`. `rho` scales a dimensionless
normalized voltage. Changing `environ.dt` changes the neuron integration and
refractory counter, but does not directly rescale the stochastic probability.

The declared pathwise derivative ignores `tau`, `rho`, the uniform sample, and
the sampled event:

```text
d spike / d q := damping_factor * (k if q >= 0 else leak).
```

The `q == 0` branch uses `k`. Gradients are returned only for `q`; the five
configuration values are ordinary Python options and receive no VJP. The
normalization that creates `q` remains in autograd, so voltage, threshold, and
reset gradients include its numerator and denominator derivatives.

For `spiking=False`, no random draw occurs. Forward is the deterministic
piecewise-linear primitive `(k if q >= 0 else leak) * q`, while backward still
includes `damping_factor`. `.train()` and `.eval()` do not alter either mode.
An actual Module probe confirmed identical train/eval spikes and final RNG
state under restored CPU RNG, and no RNG advancement in `spiking=False` mode.

## Draw and event order

For one btorch ALIF or ELIF step, the observable order is:

1. Charge `v` from the previous `v`, previous `g_k`, and current input.
2. Decay `g_k` with exponential Euler.
3. Form normalized `q` and call `PoissonRandomSpike`, consuming one
   `rand_like(q)` draw over the full batch and neuron shape.
4. If refractory state exists, multiply the sampled spike by detached
   `(refractory == 0)`.
5. Optionally store charged `v_pre_spike`, reset voltage, add `dg_k * spike_d`,
   and update refractory state.

The draw happens before refractory masking. Refractory lanes therefore consume
random values even though their emitted spike is zero. Removing those draws or
drawing only active lanes changes every later event and the global RNG state.
`detach_reset=True` detaches the spike used by voltage reset, adaptation
increment, and refractory update; it does not detach the returned spike or its
surrogate derivative.

Python multi-step execution loops over time and invokes `rand_like` once per
step with shape `[*batch, *n_neuron]`, on the charged voltage's device and
dtype. Whole-call and chunked execution must have identical events, final
physical state, and final RNG state when they begin from the same model and RNG
state. The existing CPU test covers this for ALIF and ELIF, including
refractory state.

A local CPU probe found that one `[T, *batch, *n_neuron]` draw happened to be
bitwise equal to stacking T per-step draws for contiguous float16, float32, and
float64 tensors. This is informative only. CUDA generator counter reservation,
capture, tensor layout, and dtype must be validated independently; the native
contract must not infer sequence-wide equivalence from the CPU result.

## Minimal native random-input design

The first native implementation should keep random generation outside the cell
kernel and make the random values an explicit private operator input:

```text
uniform_seq: [T, *batch, *n_neuron], same device as q and FP32
```

The kernel performs only `rate > uniform_seq[t]`. Uniforms have no gradient.
Backward and checkpoint replay reuse the forward uniforms or authoritative
saved spikes and never draw again. This gives one deterministic cell program
for CuPy, Triton, and TileLang and prevents backward recomputation from
advancing user RNG state.

Two invocation modes are useful and must remain explicit:

### Default Module-compatible global RNG

When the caller does not provide uniforms, the adapter constructs
`uniform_seq` using T Torch calls with the same logical shape, dtype, device,
and time order as the Python Module:

```text
stack(rand_like(step_reference) for t in range(T))
```

This deliberately preserves the current default generator ownership and call
schedule. It also preserves draws for refractory lanes. The final global RNG
state is observable and is part of pathwise parity. A faster single-call draw
may replace this only after CUDA whole/chunk, dtype, layout, eager, compile, and
graph-capture tests prove identical values and generator continuation.

Pre-generating the draws is eligible only when the supported cell and connection
schedule contains no other interleaved RNG consumers. A stochastic connection,
dropout layer, or user callback would require preserving its original draw order
as well; those compositions are excluded from the first lowering slice.

The adapter, not the native kernel, owns these draws. Consequently a chunked
call naturally advances the same global generator between chunks, and the
operator ABI needs no opaque generator-state output.

### Explicit logical uniforms

A lower-level validation/capture path may supply `uniform_seq` directly. This
path does not read or advance Torch's global generator. It is the preferred
way to test exact pathwise parity across backends, changed random inputs during
CUDA Graph replay, checkpoint replay, and two forwards before one backward.
Values must be in `[0, 1)` for parity with `rand_like`; shape, dtype, and device
must match the native compute contract.

The qualified native stochastic surface is FP32 only. Sequence inputs, initial
physical states, participating parameters, and explicit uniforms are rejected
when they use FP16 or BF16, before a default random draw occurs. This prevents
reduced-precision `rand_like` values from silently selecting a different event
raster even when the generator counter advances by the same amount. Plain Torch
Modules remain unrestricted, and deterministic non-Poisson native paths retain
their existing reduced-precision behavior. The restriction also applies to
`spiking=False`, which consumes no RNG but shares the same native Poisson model
contract.

An owned native seed/offset generator is excluded from the first slice. Such a
design would need to specify Philox counter reservation, global versus explicit
generator ownership, chunk continuation, graph replay advancement, stream
ordering, and compatibility with Torch's exact call schedule. It is not needed
to validate the neuron equations.

## Native training and inference

Training forward returns spikes and final physical states plus only requested
histories. The stochastic residual is either the explicit uniform sequence or
the emitted spike sequence; it is not a second hidden RNG history. Reverse must
preserve all existing cotangent surfaces:

- spike output and requested charged/post-reset histories;
- final `v`, `g_k`, and optional refractory state;
- input sequence, initial physical states, and trainable neuron parameters;
- attached and detached reset branches;
- two forwards before one backward with independent autograd contexts.

The surrogate VJP uses the saved or replayed `q` sign and the configured
`leak`, `k`, and `damping_factor`. It must not differentiate the probability
formula or regenerate random values. If physical states are checkpointed,
block replay carries adjoints across blocks and uses the original events; no
state adjoint may be detached at a block boundary.

Inference remains stateful and consumes the same random draws as training, but
must expose no backward-only tape, physical training history, or saved
surrogate input. Requested public histories and the explicit `uniform_seq`
input are legitimate sequence-sized tensors. `torch.no_grad()`,
`torch.inference_mode()`, and `.eval()` do not disable stochastic firing.

Empty-T behavior is excluded until the eager ALIF/ELIF Module defines a golden
result; its current Python loop stacks an empty list.

## 002HIP comparison and scheduling scope

The inspected external sources are:

- `/data1/fanqixuan/src/002HIP/model/ALIF.py:86-214` for neuron scheduling;
- `/data1/fanqixuan/src/002HIP/model/ALIF.py:252-390` for its Poisson surrogate;
- `/data1/fanqixuan/src/002HIP/model/ELIF.py:60-119` for ELIF.

The standalone 002HIP Poisson function has the same stochastic comparison and
piecewise `leak`/`k` backward as btorch when btorch uses
`damping_factor=1`. It receives an unnormalized voltage difference in the
legacy neuron, while btorch receives normalized `q`. Exact surrogate-only
comparison is eligible when both implementations receive the same logical
input and explicit uniforms.

The legacy ALIF `torch.where` expression evaluates its stochastic surrogate
for the whole tensor before selecting refractory zeros, so its refractory lanes
also consume draws. Its neuron recurrence still differs from btorch: implicit
`dt=1` explicit Euler, unnormalized firing input, explicit-Euler adaptation,
and different `detach_reset` treatment of the adaptation increment. Its
advertised CuPy multi-step body is commented out. It is eligible only as an
unchanged Torch source for narrowly mapped external cases, not as a native
kernel oracle or performance baseline.

The legacy ELIF is further excluded from sequence scheduling parity. It omits
adaptation, hard-codes a `0.1` voltage step, clamps refractory voltage to rest,
returns `(spike, v)` from single step, and its multi-step override does not
return the superclass result. A caller-provided Poisson surrogate can still be
tested in an isolated single-step case, but the class is not an executable
whole-sequence golden for btorch ELIF.

## Validation gate

Before native generation, add actual-Module tests for:

- exact spike/state/final-RNG parity for whole versus chunked calls on each
  target device and supported dtype;
- draw count and shape with refractory lanes present;
- train/eval and no-grad/inference-mode stochastic behavior;
- pathwise VJPs from random spike, history, and final-state cotangents under
  restored global RNG and under explicit uniforms;
- `spiking=False` deterministic primitive behavior and zero RNG consumption;
- CUDA Graph replay with changed explicit uniforms and, separately, default
  Torch RNG advancement rather than a frozen captured raster;
- inference saved-tensor inspection proving absence of backward-only tape;
- noncontiguous/broadcast parameters and batched neuron axes;
- finite positive `tau`, finite `rho`, both signs of `q`, probabilities below
  one, and saturated probability-one lanes.

Acceptance is pathwise equality to the actual Module under controlled draws,
not similarity of empirical firing rates. Statistical tests may supplement but
cannot replace exact event, continuation, RNG-state, and VJP checks.
