# Conductance and legacy dual-exponential contract

Source audit: 2026-10-19. This records source qualification and the standalone
native functional gate; it is not performance evidence. No 002HIP source was
modified. The inspected 002HIP checkout is `/data1/fanqixuan/src/002HIP` at HEAD
`f64cd8e002af0e4285125e54d275cf7eae6ef0bf`, with pre-existing working-tree
changes.

Inspected source identities:

```text
1ea72e3a4501312845ba80fc0577f7741c83b8575c6ac3cb4549465415192f85  002HIP/model/connection.py
37b5ad842a491bbaa19107b9fe16975ee8dc1f9919d170f22e28847f83995ddf  002HIP/model/connection_kernel.py
b0b03c02af49484a662ff9ebed512a93dc1b1352e5dbbd18f9dd8bab0bf30828  002HIP/model/connection_kernel_sparse.py
f0b127df2657a93164f79afe02ebffb8cfd851987a6709ecdd6724be25442311  btorch/models/synapse.py
9225c80862022beed8c9a4f3115e12d77754bf79e19bab6bb341ef0ae9413c2b  btorch/models/kernels/adapters.py
```

These identities preserve the source provenance of the original audit. They
are historical snapshots, not hashes of the current working tree.

## Decisions

| Capability | Classification | Decision before native work |
| --- | --- | --- |
| Existing btorch `ConductancePSC` | Valid single-step and sequence `MemoryModule`; standalone native composition is functionally qualified, and the explicit public Torch-only recurrent schedule is implemented with human golden confirmation pending | The exact-class standalone adapter reuses the existing exponential state operator and applies `g * (E - v)` as an ordinary Torch operation. `ConductanceRecurrentNN` fixes previous-post-reset voltage observation and next-step recurrent event delivery. Keep conductance outside the general native recurrent layout until this public schedule is confirmed and a coupled lowering is separately validated. |
| 002HIP `ConductanceBasedExponentialPSC` | Different scientific model and currently not an executable golden | First evaluate existing HeterSynapsePSC with compatible PSC dynamics and per-receptor voltage readout. Map integration/event semantics explicitly; propose a new Module only if a distinct scientific law remains necessary. The audited source is not a valid golden as shipped. |
| 002HIP `ConductanceBasedDoubleExponentialPSC` | Separate per-edge, hard-coded-`dt=0.1` model; currently incomplete | Evaluate existing HeterSynapsePSC plus DualExponentialPSC dynamics and voltage readout before proposing a new model. Its inactive custom dispatch and differing normalization/event timing remain qualification issues. |
| 002HIP legacy `DualExponentialPSC` | Not the same state, integration, normalization, or event-time contract as btorch `DualExponentialPSC` | Direct parameter copying is invalid. A restricted mathematical mapping exists with transformed time constants, amplitude, state, and one input-delay step; otherwise parity is impossible and a separate Module is required. |

This follows the kernel workflow gate: the actual Python `MemoryModule`, its
registered state, three-scale activity figures, continuation, and gradients
must be confirmed before any CuPy/Triton/TileLang generation
(`.agents/skills/kernel-gen/SKILL.md`; `references/workflow.md:1-75`;
`prompts/phase0-model.md:1-7`).

## Heterogeneous receptor composition comes first

`HeterSynapsePSC` already owns the per-postsynaptic-neuron/receptor state
layout, receptor-specific dynamics and delayed connection expansion. A
per-edge implementation is not by itself a missing scientific model: edges
with compatible linear dynamics and reversal potential can be aggregated into
those receptor channels. Reuse this structure before proposing new Modules.
Legacy integration, event timing, normalization and external-drive differences
still require their own mapping; receptor grouping alone does not prove parity.

The current composition is incomplete for voltage-dependent current:
`HeterSynapsePSC.single_step_forward(z)` invokes its base PSC without voltage
and then sums receptor outputs. With `base_psc=ConductancePSC`, that returns
summed conductance, because the base PSC's no-voltage contract returns `g`.
The multistep API also supplies no voltage. The added Torch conductance
recurrent wrapper currently rejects HeterSynapsePSC through its exact-class
guard. These are integration/backend coverage gaps, not evidence that btorch
needs a separate per-edge conductance model.

A CPU check with two receptor conductances `[1, 1]` and reversals `[0, -80]`
returns `2` through the current HeterSynapsePSC call. The existing base
`current_charge(v)` instead gives total current `40` at `v=-60`, and `0` at
`v=-40`. Correct COBA composition must apply each receptor's voltage-dependent
readout before receptor reduction, and recurrent execution must provide the
chosen observed membrane voltage. Preserve independent receptor memories,
delays and gradients through that readout.

## Existing btorch ConductancePSC

`btorch/models/synapse.py:426-571` defines one post-synaptic conductance state,
stored in the registered `psc` memory inherited from `BasePSC`
(`synapse.py:101-135`). For step `t`, with `a = exp(-dt / tau_syn)`, its actual
single-step contract is

```text
g_t = a * g_{t-1} + linear(z_t)
I_t = g_t * (E - v_t)       when v_t is supplied
I_t = g_t                   when v_t is omitted
```

The input event therefore changes the returned current in its delivery step.
The sign convention is positive excitatory current for `E > v`; this is
asserted by `tests/models/test_synapse.py:41-69`. `tau_syn` and `E` are buffers;
`g` is the `psc` memory and follows `MemoryModule` initialization, reset,
detach, checkpoint, device, and dtype handling (`base.py:682-704,800-904,
915-1003`). The linear layer owns dense/sparse weights and their gradients.

The explicit `multi_step_forward(z_seq, v_seq=v_seq)` path is the stateful
sequence golden (`synapse.py:486-571`). Its leading voltage dimension is
`T` or `1`, with remaining dimensions broadcast to the post-synaptic shape.
The plain `backend="torch"` voltage-aware path calls the authoritative
`single_step_forward(z_t, v_t)` once per step. The exact-class standalone
adapter uses the unchanged native `exponential` operator for the same
conductance trajectory, then evaluates `g_t * (E - v_t)` as an ordinary Torch
expression (`kernels/adapters.py:230-245`). Both update registered conductance
and delay state and preserve whole/chunk continuation and autograd semantics.
An explicit-voltage empty sequence advances no state and uses the actual COBA
expression to preserve dtype promotion and voltage/state VJPs.

Backend selection deliberately preserves two no-voltage contracts. Plain
`backend="torch"` retains the historical stateless convolution from rest,
including `kernel_len` truncation and no mutation of `psc` or delay history
(`synapse.py:501-527`). `KernelOptions(neuron="torch")` and the CuPy, Triton, and
TileLang cell backends select the stateful exact-class adapter for both single
and multi-step CUBA/COBA calls. Adding optional `v_seq` did not alter existing
plain-Torch calls and is not a Breaking model change.

The CPU Module contract is exercised in
`tests/models/test_conductance_sequence.py:87-250` and the 16-check validation
harness. Standalone native composition is functionally qualified by
`tests/models/kernels/test_conductance_sequence.py`: all 21 cases passed on
CuPy, Triton, and TileLang in Slurm run 1287, covering dense/sparse connections,
delay and nonzero state, CUBA/COBA single and split sequences, full VJPs, empty
chunks, inference tape absence, and changed-input CUDA Graph training replay.
This is functional evidence only; it establishes no performance parity.

`ConductancePSC` remains absent from the general `_model_layout`
(`kernels/adapters.py:14-51`). Exact subclasses and native `RecurrentNN`
preparation are rejected before state advances. The public
`ConductanceRecurrentNN` schedule below now fixes voltage observation to the
previous post-reset value and recurrent event delivery to the following step.
The remaining work is human confirmation of that golden and, only afterward, a
separately validated native coupled lowering. In inference, retain final
registered state and requested records only. Do not allocate or save `T`-length
conductance/voltage histories for backward. Training may save or recompute the
information required for the VJP. This is the same separation enforced by
`tests/models/kernels/test_inference_storage.py:27-121` and
`kernels/operators.py:600-649`.

## 002HIP conductance exponential

The intended Python recurrence is visible at
`002HIP/model/connection.py:399-442`. Its state is per edge because the event
drive is `z * weight0`, and `F` is required to have the weight matrix shape
(`connection.py:417,435-441`):

```text
q_d = 1 - 1 / tau_d                    # implicit dt = 1 Euler
q_r = 1 - 1 / tau_r
g_t[e] = q_d * g_{t-1}[e] + h_{t-1}[e]
h_t[e] = q_r * h_{t-1}[e] + z_t[pre(e)] * W[e]
I_t[e] = (g_t[e] + F[e]) * (v_t[post(e)] - E[e])
```

`g` is read after its update, while the current event is added to `h` only
afterward. A spike at `t` first changes `g` and current at `t+1`; `F` contributes
at every step. The sign is the opposite algebraic convention from btorch:
`v - E`, not `E - v`. Summation over incoming edges is not performed by this
class. The source does not define whether `v_t` is pre-charge, charged, or reset
voltage; the only recurrent container that would establish this order is
commented out (`connection.py:544-603`).

That sign and missing-container statement is scoped to the modified checkout
identified at the top of this document. Historical commit
`cf8f78e81eeb904409f03c3ba0fca2f92195e5f9` instead has an active recurrent
container: its PSC returns conductance, and the container computes
`-G * (v - E)` before incoming-edge reduction, which is algebraically
`G * (E - v)`. The two revisions must not be combined into one claimed
executable contract.

This is not a configuration of btorch `ConductancePSC`: it has two Euler
states rather than one exponential state, one-step event latency rather than
delivery-step response, per-edge state and `F`, and the opposite current sign.
Aggregating conductance before multiplying by voltage is equivalent only when
all incoming edges for a post neuron share the same reversal potential; a
general per-edge `E`/`F` contract requires receptor/edge channels through the
current calculation.

The checked-out class is not executable as a reference:

- Its constructor calls `reset_states()` without the required `z` argument
  (`connection.py:66-75,428`), after requiring `F.shape == weight0.shape`.
  Shipped examples also omit the required `reversal_potential` argument
  (`test_connection_0322/test_coba_psc.py:51-55` and
  `test_net/net_test1_torch_coba.py:26-29`).
- The inherited Torch multistep loop calls `single_step_forward(z)` although
  the base single-step method requires `(z, v)`, and the conductance class does
  not override either call (`connection.py:95-106,444-446`).
- The CUDA wrappers return `G = g + F`, not current, and accept no voltage or
  reversal potential (`connection.py:450-481`; dense kernel
  `connection_kernel.py:610-764`; sparse kernel
  `connection_kernel_sparse.py:641-805`). They therefore cannot be an oracle
  for COBA current or voltage gradients.
- Dense and sparse CUDA do not implement one recurrence. Dense injects `wz`,
  while sparse injects `(1/tau_d)(1/tau_r) wz`
  (`connection_kernel.py:634-642` versus
  `connection_kernel_sparse.py:668-680`). Both read `z_seq` without the current
  time offset; dense also sizes the weight loop by flattened `neuron_num`, which
  folds batch into neuron count (`connection_kernel.py:719-735`).
- Their backwards omit the connection transpose/weights, use a gradient scale
  inconsistent with dense forward, ignore gradients triggered only by `h_init`,
  weights, `F`, and time constants, and return fewer gradients than forward
  inputs (`connection_kernel.py:685-690,700-703,806`;
  `connection_kernel_sparse.py:723-728,738-740,847`). With
  `save_spike_as_bool_in_neuron_kernel`, they also boolean-pack non-boolean
  conductance output (`connection_kernel.py:749-755`).

Accordingly, the source can inform a proposed model but cannot be called the
golden. Before implementation, create a real btorch `MemoryModule` with
registered `g` and `h` memories and explicit reset values, define edge/receptor
axes, broadcasting of `tau_d`, `tau_r`, `F`, `E`, `v`, and `W`, define incoming
edge reduction, and define whether batch lanes own independent edge states. The
audited edge update is `z * weight0` and does not use `BasePSC.linear` bias, so
the first exact model must require zero bias unless a different bias contract is
explicitly proposed and validated. Validate no-input, impulse, sustained input,
step-off, nonzero initial state, two independent batch lanes, and whole/chunk
continuation. If the historical recurrent container is the target, its stored
post-synaptic current `I` and the neuron memories are continuation roots in
addition to `g` and `h`; nonzero initial `I` must be exercised. Required figures
must show input, voltage, `g`, `h`, `F`, per-receptor current, reduced/stored
current, and spikes at single-neuron, connected-pair, and network scales. Human
confirmation freezes that Module as the golden; only then should a native
lowering be designed.

`ConductanceBasedDoubleExponentialPSC` is also separate. It uses edge states
`exp_rise`/`exp_decay`, hard-coded factors `exp(-0.1/tau)`, normalization by
`tau_d - tau_r`, and `(g + F) * (v - E)`
(`connection.py:485-540`). Its dispatch is commented out, its edge states are
not registered memories, and they are undefined until `reset_states(z)` is
called. The ELIF example omits the required reversal argument
(`model/ELIF.py:233-250`). It is neither executable evidence nor a lowering of
the preceding Euler cascade.

## Legacy current-based DualExponentialPSC

002HIP's legacy class at historical main commit
`cf8f78e81eeb904409f03c3ba0fca2f92195e5f9` (`model/connection.py:105-226`)
implements an Euler cascade, not btorch's difference of independently decayed
traces. This historical Torch Module is executable and is the source for the
separate qualification harness; it must not be confused with the broken
current checkout or its CUDA implementations. With implicit `dt = 1`,
`q_d = 1 - 1/tau_decay`, `q_r = 1 - 1/tau_rise`, `D = E_syn - v_r`, and the
complete projected drive `x_t = W z_t + bias`:

```text
g_t = q_d * g_{t-1} + h_{t-1}
h_t = q_r * h_{t-1} + g_max * x_t
I_t = D * g_t
```

The event first affects `I` at `t+1`. In contrast, btorch
`DualExponentialPSC` uses exponential factors and delivery-step output
(`btorch/models/synapse.py:714-794`):

```text
r_t = exp(-dt / tau_rise_b) * r_{t-1} + x_t
d_t = exp(-dt / tau_decay_b) * d_{t-1} + x_t
psc_t = a * (q_d * d_t - q_r * r_t)
```

Directly copying `tau_decay`, `tau_rise`, or `g_max` is therefore wrong.
The currently chosen finite, positive-time-constant mapping requires
`0 < q_d < 1`, `0 < q_r < 1`, `q_d != q_r`, `g_max != 0`, and `D`
representable by btorch's per-output amplitude:

```text
tau_decay_b = -dt / log(q_d)
tau_rise_b  = -dt / log(q_r)
a           = D * g_max / (q_d - q_r)
r_init      = h_init / (g_max * q_r)
d_init      = ((q_d - q_r) * g_init + h_init) / (g_max * q_d)
x_b[t]      = x_legacy[t - 1]             # explicit one-step delay, zero before t=0
```

Set btorch's explicit `A` so that
`a = (tau_decay_b - tau_rise_b) * A /
(tau_rise_b * tau_decay_b)`. These equations give an algebraic output and input
VJP identity for the intended legacy recurrence, including arbitrary mapped
initial `g/h`. An FP64 CPU substitution through the actual btorch Module agreed
with the algebraic recurrence to `2.7e-15` for outputs, `5.4e-15` for final
states, and `2.2e-14` for input, state, and parameter VJPs. This is an algebraic
probe rather than the historical-Module evidence. Btorch's stored
`g_rise/g_decay` remain transformed coordinates, not legacy state records.

The restricted mapping is now functionally qualified against the actual public
historical Module `model.connection.DualExponentialPSC(backend="torch")` at
`cf8f78e81eeb904409f03c3ba0fca2f92195e5f9`. The harness used the real,
unchanged SpikingJelly `MemoryModule` base, not a copied recurrence. Slurm 1295
passed the btorch Torch path and all three native backends at `T=32, N=16`;
Slurm 1296 passed all four at `T=128, N=64`; and Slurm 1297 added explicit
chunk-continuation VJPs for input, weight, bias, and the original legacy `g/h`
roots, with all four paths passing at `atol=rtol=2e-5`. At the longer size, the
largest native forward/final-state error was `6.676e-6`, the largest whole-run
VJP error was `1.073e-6`, and the largest chunk-continuation VJP error was
`1.193e-6`. The machine-readable evidence is
`run/parity/002hip-dual-exponential-chunk-vjp.json`. These are functional
results and make no performance claim.

The one-step delay applies to the complete projected drive `x = W z + bias`.
Btorch's ordinary connection delay occurs before `linear`, so it does not delay
the bias; exact reuse therefore requires zero bias or an explicit delay of the
projected drive. Let `r_t` and `d_t` be btorch's stored traces after producing
legacy output `t`, and let the separate pending state be `p_t = x_t`. The
corresponding legacy final state is:

```text
g_t = g_max * (q_d * d_t - q_r * r_t) / (q_d - q_r)
h_t = g_max * (q_r * r_t + p_t)
```

For a chunk beginning at legacy step `s`, the btorch cell consumes
`[p_{s-1}, x_s, ..., x_{s+L-2}]` and returns `p_{s+L-1} = x_{s+L-1}` as the
new pending state; at the global start, `p_{-1} = 0`. Full-BPTT continuation
must carry all three tensors without detaching them. In particular, the final
`h` cotangent has a direct path through `p_t` to the final input, weight, bias,
and `g_max`. Resetting, dropping, or detaching that pending projected drive
changes output timing, the final-state transform, or its VJP.

The public btorch `A -> a` expression also makes `q = 1` unsupported by this
mapping: it requires `tau_b = inf`, for which the constructor evaluates an
`inf / inf` factor and stores `a = NaN`. Non-positive `q` has no supported real
exponential-Euler representation, and `q_d == q_r` is the repeated-pole
singularity of the current dual-trace parameter form. Values near zero, one, or
equal poles are ill-conditioned even before those exact boundaries. The stated
coordinate choice divides by `g_max`, so it excludes `g_max == 0`; that case is
not mathematically impossible, because an adapter could introduce a separate
nonzero coordinate scale and scale its delayed input by `g_max`. Consumers that
require legacy `g/h` under their original names still need the explicit state
transform. These restrictions and the visible projected-drive delay must be
part of any adapter contract rather than inferred from class names.

The historical 002HIP CUDA and sparse implementations remain outside this
qualification. Dense and sparse CUDA forward repeatedly read the first input
time row (`connection_kernel.py:32-43` and
`connection_kernel_sparse.py:35-50`), and dense uses transposed/square-only
weight indexing. Backward does not apply `W^T`, ignores weight/parameter
gradients and `h_init`-only activation, computes suspect initial-state
gradients, and returns the wrong number of argument gradients
(`connection_kernel.py:80-94,103-105,214`;
`connection_kernel_sparse.py:87-101,110-112,224`). The archived Torch sparse
branch also falls through to its dense `linear`, so it is not sparse evidence.
The qualification therefore compares mapped btorch execution and Torch
autograd with the actual historical dense Torch Module, never with those CUDA
kernels. Current 002HIP HEAD is a separate broken API: its public call raises
the expected narrow `TypeError` because `current_charge()` accepts no voltage
argument while its caller supplies one. That failure is recorded only as an
exclusion and is not treated as evidence for the historical revision.

## Promotion gate

The existing btorch `ConductancePSC` has a validated Torch golden and
functionally qualified standalone native composition. This does not qualify a
native recurrent COBA schedule, performance parity, either distinct 002HIP
conductance model, or a transformed legacy adapter. Those promotions require:

1. Confirmed state/axis/unit/dtype/device tables, voltage observation time,
   event order, `dt`, sign, reduction, initialization, per-edge versus reduced
   storage, and independent batch-lane continuation. Dense and sparse topology
   must name the same physical edge order and reduction.
2. Actual-Module figures and numerical assertions at single-neuron,
   connected-pair, and network scales, including impulse placement away from
   `t=0`, mixed reversal potentials entering one post-synaptic neuron, nonzero
   initial edge states, nonzero initial recurrent `I` when that container is the
   target, two independent batch lanes, and chunk boundaries on both sides of
   event delivery.
3. Torch-autograd oracles for inputs/events, every initial edge state, initial
   recurrent `I` and neuron memories when applicable, voltage, connection
   weights, reversal potential, `F`, and trainable time constants/amplitudes.
   Losses must include cotangents on outputs and every final carried state so
   continuation-only VJP paths are covered. Unsupported gradients must be
   declared rather than silently zeroed. The first exact 002HIP conductance
   validation uses zero connection bias because the audited edge recurrence
   bypasses the inherited biased linear projection.
4. Separate inference and training contracts. Inference retains only final
   registered state and requested public histories; it creates no autograd
   graph and stores no backward-only sequence residuals.
5. Only after human confirmation: minimal native implementation, then
   CuPy/Triton/TileLang parity, first-divergence reports, compile/CUDA-graph
   checks, and Slurm GPU performance runs. Slurm scripts and large artifacts
   remain outside source control.

Slurm 1299 expands historical qualification to scalar, per-neuron and strided
roots for `tau_decay`, `tau_rise`, `g_max`, `E_syn` and `v_r`, together with
input, weight, bias and initial g/h. All twelve backend/layout cases pass
whole/chunk VJPs at T=128/B=1/N=64 with unchanged atol=rtol=2e-5. Unused strided
backing slots have zero gradients. Changed-value forwards preserve the original
root storage and rebuild mapped coefficients/state/amplitude before execution.
Native absolute parameter-VJP error reaches 5.723e-5 in this wider matrix but
passes the combined absolute/relative gate. Report:
`run/parity/002hip-dual-exponential-parameter-vjp.json`. This does not extend
qualification to singular poles, omitted external paths or performance.

## Public recurrent COBA golden candidate

`btorch.models.rnn.ConductanceRecurrentNN` now exposes the explicit Torch-only
schedule previously confined to the activity harness. It composes the actual
neuron and exact `ConductancePSC` Modules:

```text
I_t = g_(t-1) * (E - v_(t-1, post-reset))
z_t = neuron(x_t + I_t)
g_t = ConductancePSC(z_t + x_syn_t)
```

The synapse call advances its connection/delay/conductance state; its returned
current is not delivered again in the same step. Emitted recurrent events first
influence membrane current on the following step. Existing `RecurrentNN` keeps
its CUBA behavior. Network and backend-bearing descendants must use plain
`"torch"` in this initial candidate. Native coupled execution is not delivered.

Slurm 1307 passes 85 targeted recurrent/standalone conductance tests. The public
activity harness passes all 16 checks at single, pair and network scales in
`run/parity/conductance-public-validation/report.md`. Root inspection found
finite, readable input/spike/voltage/conductance/current/reversal/drive-force
channels. These results are submitted for human golden confirmation; they are
not approval or native-kernel evidence.

The new class inherits state recording, checkpointing and continuation. Its
capture version tracks non-memory parameter/buffer identities and scalar or
Tensor timestep identity, while excluding advancing recurrent memories. Tests
exercise changed-input replay, parameter/buffer replacement recapture, in-place
updates, timestep changes and pre-callback rejection of unsupported backends.

### Native readout contract after confirmation

Reuse exponential conductance-state dynamics, but keep the recurrent readout
identity separate from the exponential state layout. `E` is invocation data,
with its original scalar/broadcast/strided storage and gradients preserved.
For incoming current cotangent `a_t`, the extra coupling VJPs are:

```text
dL/dg_(t-1) += a_t * (E - v_(t-1))
dL/dE       += a_t * g_(t-1)
dL/dv_(t-1) += -(a_t * g_(t-1))
```

The voltage cotangent also contains the neuron's ordinary state path. Preserve
the actual Module graph's reduction and addition order; long recurrent VJPs
must be diagnosed with complete boundary cotangents rather than partial fused
inputs. Inference reads E/g/v in registers and retains only final state and
requested public records, with no backward-only coupling buffers or arguments.

Do not select the existing CUBA LIF+Exponential checkpoint plan merely because
conductance dynamics reuse the `exponential` cell layout. Replay must itself
apply the voltage-aware current; use an explicit readout-aware replay contract
or retain the full-state training fallback until that path is qualified.
