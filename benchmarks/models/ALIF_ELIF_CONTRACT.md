# ALIF and ELIF semantic contract

Status: pre-kernel contract. This document fixes the Python semantics that a
future native CuPy, Triton or TileLang implementation must preserve. It does
not qualify a backend or change either scientific model.

The btorch Python Modules are the golden implementations:

- `btorch/models/neurons/alif.py:ALIF`
- `btorch/models/neurons/alif.py:ELIF`
- `btorch/models/base.py:BaseNode`
- `btorch/models/ode.py:exp_euler_step`

The inspected 002HIP files define a separate compatibility reference:

- `/data1/fanqixuan/src/002HIP/model/ALIF.py:StochasticALIFNode`
- `/data1/fanqixuan/src/002HIP/model/ELIF.py:ELIFNode`

Their class names do not imply identical equations or event semantics. Native
btorch parity is measured against the btorch Modules first. An external 002HIP
comparison is accepted only for a configuration whose mapping and exclusions
are stated explicitly below.

## Shapes, parameters and state

Input and output use time-leading sequences for multi-step execution:
`x_seq` and `spike_seq` have shape `[T, *batch, *n_neuron]`. A single step has
shape `[*batch, *n_neuron]`. Parameters may be scalar or broadcast over the
neuron axes, following the existing `def_param` behavior.

| Item | ALIF | ELIF |
| --- | --- | --- |
| Core continuation state | `v`, `g_k`, and optional `refractory` | Same as ALIF |
| Optional diagnostic state | `v_pre_spike` when `pre_spike_v=True`; non-persistent | Same as ALIF |
| Registered reset values | Constructor-time snapshots of `v_reset` and `g_k_init`; `refractory=0` | Same as ALIF |
| Shared parameters | `v_threshold`, `v_reset`, `c_m`, `g_leak`, `E_leak`, `E_k`, `tau_adapt`, `dg_k`; optional `tau_ref` | All ALIF parameters |
| Additional parameters | None | `delta_T`, `v_T` |

`g_k_init` configures the reset value of the `g_k` memory; it is not a model
parameter. The `v` reset value likewise captures the constructor argument before
`v_reset` is registered as a parameter or buffer. Memory reset values are
detached snapshots: later optimization or replacement of a trainable `v_reset`
does not automatically change the value restored by `reset()`. `tau_ref=None`
removes both the refractory state and its parameter. Any name selected through
`trainable_param` must retain the Module's parameter gradient semantics. Initial
`v`, `g_k` and optional `refractory` are independent sequence inputs for
state-continuation and initial-state VJP checks even though ordinary Module use
initializes them as buffers.

State dtype/device are chosen when the Module state is initialized and are not
converted from each input automatically. Native acceptance cases require
Module-compatible input, state and parameter dtype/device. State and history
outputs retain the complete batch and neuron axes; an implementation must not
flatten away those logical axes.

## btorch ALIF recurrence

Let `v0`, `g0` and, when enabled, `r0` be the states at the start of step `t`.
Let `dt` be the positive finite scalar from `btorch.models.environ`. Define the
exponential-Euler factor

```text
phi(A, dt) = dt                              when A == 0
             expm1(dt * A) / A               otherwise.
```

This is also a backward contract. At exactly `A == 0`, `torch.where` selects
the constant `dt` branch, so Torch autograd gives `d phi / d A = 0`; it does not
differentiate the smooth limiting extension, whose derivative would be
`dt**2 / 2`. Native VJPs must preserve the implemented branch derivative.

The step is evaluated in this order.

1. Charge voltage using the previous adaptation conductance:

   ```text
   Fv = (-g_leak * (v0 - E_leak) - g0 * (v0 - E_k) + x_t) / c_m
   Av = (-g_leak - g0) / c_m
   v_charge = v0 + phi(Av, dt) * Fv
   ```

2. Decay adaptation independently of the new spike:

   ```text
   Ag = -1 / tau_adapt
   g_decay = g0 + phi(Ag, dt) * (-g0 / tau_adapt)
           = g0 * exp(-dt / tau_adapt)
   ```

3. Fire from charged voltage. The surrogate input is normalized:

   ```text
   q = (v_charge - v_threshold) / (v_threshold - v_reset)
   raw_spike = surrogate(q)
   spike_t = raw_spike                                      # no refractory state
   spike_t = raw_spike * detach(r0 == 0)                    # refractory enabled
   ```

4. Reset voltage, add the spike-triggered adaptation increment, and update the
   refractory counter. `spike_d` is `detach(spike_t)` only when
   `detach_reset=True`.

   ```text
   v1 = v_charge - (v_charge - v_reset) * spike_d           # hard reset
   v1 = v_charge - (v_threshold - v_reset) * spike_d        # soft reset
   g1 = g_decay + dg_k * spike_d
   r1 = relu(r0 + spike_d * tau_ref - dt)                   # when enabled
   ```

An input delivered at step `t` affects `v_charge` and the emitted spike at that
same step. The spike-triggered `g_k` increment is post-event state and first
affects voltage at step `t + 1`. Refractory status is tested from `r0`; voltage
and `g_k` continue evolving while firing is blocked. ALIF does not clamp
refractory voltage to `v_reset`.

## btorch ELIF recurrence

ELIF inherits every ALIF state, event, refractory and reset rule. Only the
voltage charge changes. With

```text
exp_term = g_leak * delta_T * exp((v0 - v_T) / delta_T)
```

the Module supplies the following derivative and local linear term to
exponential Euler:

```text
Fv = (-g_leak * (v0 - E_leak) - g0 * (v0 - E_k)
      + exp_term + x_t) / c_m
Av = (-g_leak - g0 + exp_term / delta_T) / c_m
v_charge = v0 + phi(Av, dt) * Fv
```

This is the exact behavior of `ELIF.dV` plus `exp_euler_step`. The exponential
ODE is nonlinear, so this is a local exponential-Euler step, not an analytic
closed-form solution over the whole timestep. A backend must preserve this
evaluation and its derivatives rather than replacing it with explicit Euler or
another exponential-neuron discretization.

## Spike forward and surrogate derivatives

The default btorch surrogate is `Sigmoid(alpha=2, damping_factor=1,
spiking=True)`. Its forward is the hard event

```text
H(q) = 1 when q >= 0, else 0.
```

Its declared backward with respect to `q` is

```text
k_sig = 2 * log(sqrt(2) + 1)
sigma = sigmoid(k_sig * alpha * q)
dH/dq := damping_factor * 4 * sigma * (1 - sigma).
```

The normalization in `q` remains inside autograd. Gradients to charged voltage,
`v_threshold` and `v_reset` therefore include derivatives of both numerator and
denominator. `detach_reset=True` detaches only spike use in reset, adaptation
increment and refractory update; it does not detach the returned spike or the
charge path.

btorch also provides `PoissonRandomSpike`. For uniform `u` with the same shape
as `q`, its stochastic forward and declared derivative are

```text
rate = exp(rho * q) / tau
spike = (rate > u)
dspike/dq := damping_factor * (k if q >= 0 else leak).
```

The rate is not clamped to `[0, 1]`. This stochastic surrogate is not part of
the first native ALIF/ELIF scope. Adding it requires an explicit logical random
input or an RNG state contract covering continuation, chunking, CUDA Graph
replay and backward recomputation. Equal seeds alone are insufficient evidence
of pathwise parity.

## Outputs, histories and continuation

`single_step_forward` returns `spike_t`. `multi_step_forward` returns only
`spike_seq` and leaves the Module at `(v_T, g_k_T[, refractory_T])`. Repeated
sequence calls continue from those final states.

Native sequence qualification must compare:

| Observable | Logical time |
| --- | --- |
| `spike` | Event emitted after charge/adaptation decay and before reset |
| `v` history | Post-reset `v1` |
| `g_k` history | Post-decay, post-spike `g1` |
| `refractory` history | Post-update `r1` when enabled |
| `v_pre_spike` | Charged `v_charge` captured immediately before reset when requested |
| Final states | The last post-step values, usable as the next call's initial states |

Histories are optional outputs; requesting none must not allocate public history
storage. Training may retain or recompute independent backward residuals, but
those are not public histories. Inference must retain no backward-only tape.

The current eager `BaseNode.multi_step_forward` stacks a Python list and does
not define an empty-`T` result. Empty-sequence behavior must be decided and
matched by the eager golden before it becomes a native acceptance case; a
backend must not silently invent a divergent convention.

Backward qualification covers VJPs from spikes, requested histories and final
states to `x_seq`, initial states and every selected trainable parameter.
Optional output/final/history cotangents, two forwards before one backward,
state continuation and detach-reset branches are separate required cases.

## 002HIP ALIF compatibility profile

The legacy ALIF recurrence uses implicit `dt=1` explicit Euler:

```text
v_charge = v0 + (-g_leak * (v0 - E_leak) + x_t
                  - g0 * (v0 - E_k)) / C
q_hip = v_charge - v_peak
spike = 0 if r0 > 0 else PoissonRandomSpike(q_hip)
g1 = g0 - g0 / tau_u + dg_k * spike
v1 = v_rest                              # hard reset on spike
v1 = v_charge - v_peak * spike           # nominal soft reset
r1 = relu(r0 + spike * t_ref - 1)
```

Its order is charge, stochastic fire, combined adaptation decay/increment,
then reset. As in btorch, the new spike adaptation does not affect voltage until
the following step. The decay differs: legacy explicit Euler uses
`1 - 1/tau_u`, whereas btorch uses `exp(-dt/tau_adapt)`.

Legacy `detach_reset=True` detaches the spike only inside voltage and
refractory reset. Its adaptation update already consumed the attached spike, so
the `dg_k * spike` path remains differentiable. btorch uses `spike_d` for its
adaptation increment as well and therefore detaches that path. This excludes
detach-reset training parity even in a case with aligned forward values.

The legacy constructor forces its own `PoissonRandomSpike` and applies it to
the unnormalized voltage difference. A restricted stochastic mapping with
positive constant `D = v_peak - v_rest` can align forward probability and the
piecewise voltage derivative algebraically:

```text
v_threshold = v_peak
v_reset = v_rest
rho_btorch = rho_hip * D
k_btorch = k_hip * D
leak_btorch = leak_hip * D
tau_btorch = tau_hip
```

because `q_btorch = q_hip / D`. This does not remove integration differences,
does not cover trainable threshold/reset derivatives, and still requires the
same logical random draws. It is therefore a case mapping, not whole-model
equivalence.

The legacy ALIF single-step output is only the spike. Its multi-step method
returns `(spike_seq, post-reset v_seq)` and exposes neither `g_k` nor
`refractory` history. The advertised CuPy body is commented out; backend names
alone are not evidence of an executable custom kernel. The nominal
`v_rest=None` soft-reset path must also be execution-qualified because the
constructor subsequently calls `torch.tensor(v_rest)`.

## 002HIP ELIF compatibility profile

The legacy ELIF constructor bypasses `StochasticALIFNode.__init__`. It has only
`v` and `refractory` state and no `g_k`, `E_k`, `tau_u` or `dg_k` adaptation.
It accepts a caller-provided surrogate and uses this explicit update:

```text
v_charge = v0 + 0.1 * (
    -g_leak * (v0 - E_leak) + x_t
    + g_leak * ELIF_delT * exp((v0 - ELIF_VT) / ELIF_delT)
) / C
v_charge = v_rest if r0 > 0 else v_charge
spike = 0 if r0 > 0 else surrogate(v_charge - v_peak)
v1 and r1 follow the inherited ALIF reset with an implicit counter step of 1
```

Consequently it is not btorch `ELIF` under renamed parameters: it lacks
adaptation, uses hard-coded voltage `dt=0.1`, decrements refractory by `1`, and
clamps voltage to rest while refractory. Its single-step method returns
`(spike, post-reset v)`.

The legacy `multi_step_forward` override does not return the superclass result,
and the inherited ALIF loop assumes a tensor where ELIF single-step returns a
tuple. Treat legacy ELIF multi-step/custom-backend comparison as unsupported
until an unchanged executable reference path is demonstrated. Do not repair the
external source inside the parity harness.

## Parameter-name mapping

| 002HIP | btorch | Qualification note |
| --- | --- | --- |
| `v_peak` | `v_threshold` | Same event level; btorch normalizes surrogate input |
| `v_rest` | `v_reset` | Equivalent for hard reset; soft formulas coincide only when reset is zero or are mapped explicitly |
| `C` | `c_m` | Same role |
| `g_k_reset` | `g_k_init` | Reset/initial state value, not a trainable parameter in btorch |
| `tau_u` | `tau_adapt` | Explicit-Euler versus exponential decay |
| `t_ref` | `tau_ref` | Counter decrement is `1` in 002HIP and `dt` in btorch |
| `ELIF_delT` | `delta_T` | Same exponential scale, different integrator and state model |
| `ELIF_VT` | `v_T` | Same exponential reference voltage |
| `E_leak`, `g_leak`, `E_k`, `dg_k` | Same names | Equation role matches for ALIF, subject to integration/order contract |
| `act_tau`, `act_rho`, `leak`, `k` | `PoissonRandomSpike` options | Requires normalization scaling and controlled random draws |

## Scope and acceptance

The first implementation scope is deterministic btorch ALIF and ELIF with the
existing six supported deterministic surrogates, scalar or broadcast FP32
parameters, optional refractory state, hard/soft reset, detach-reset, requested
records, continuation, inference and training VJPs. The actual btorch Module
and Torch autograd are the oracle.

The following are excluded from an initial parity claim:

- stochastic Poisson native execution and statistical equivalence;
- changing ALIF/ELIF equations, timestep rules or event ordering to resemble
  002HIP;
- claiming legacy ELIF multi-step or custom-kernel performance before an
  unchanged executable reference passes correctness checks;
- online plasticity, connection topology, delay/receptor routing and rewiring;
- dual-exponential PSC qualification, which is a separate contract. The legacy
  implementation does exist in `/data1/fanqixuan/src/002HIP/model/connection.py`
  with dense and sparse CuPy kernels; it must not be inferred from
  `model/synapse.py` alone.

No performance result is accepted before forward trajectories, every requested
history, final continuation state and declared VJPs pass the fixed numerical
gate. Stochastic cases additionally require pathwise draws or a separately
declared statistical test. Missing or non-executable reference paths are
reported as `UNSUPPORTED` or `NOT_RUN`, never as parity passes.
