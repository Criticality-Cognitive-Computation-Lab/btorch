# The `dt` Environment

btorch neuron models are defined by ordinary differential equations (ODEs). To solve these ODEs numerically, the solver needs a time-step size `dt`. Rather than threading `dt` through every constructor and forward call, btorch uses a lightweight computation environment similar to BrainPy.

## Setting `dt`

The recommended pattern is a context manager:

```python
from btorch.models import environ

with environ.context(dt=1.0):
    spikes, states = model(x)
```

This scopes `dt` to the forward pass and avoids accidental global state leaks.

## Global Default

You can also set a global default (useful in notebooks or scripts):

```python
environ.set(dt=1.0)
```

Any module that calls `environ.get("dt")` will fall back to this value when no active context exists.

`environ.set` stores **process-global** defaults, visible from every thread. `environ.context` is a **thread-local** override stack: inside its `with` block it wins over the global default, and other threads keep seeing the default.

## Forgetting `dt` Is a Common Pitfall

If `dt` is not set, neuron forward passes may raise a `KeyError`. The error message explicitly tells you how to fix it:

```
KeyError: 'dt is not found in the context.
You can set it by `with environ.context(dt=value)` locally
or `environ.set(dt=value)` globally.'
```

## Decorator Usage

`environ.context` also works as a function decorator:

```python
@environ.context(dt=1.0)
def forward(model, x):
    return model(x)
```

## Timing Convention

Neurons and post-synaptic current (PSC) models use these timing conventions:

- input current at step `t` affects the returned neuron state at step `t`;
- ExponentialPSC and alpha PSCs respond to a delivered spike immediately;
- DualExponentialPSC injects equal increments into its rise and decay states.
  Their difference is zero at injection, so its impulse response begins at
  step `t + 1`: `k[s] = a * (exp(-s*dt/tau_decay) - exp(-s*dt/tau_rise))`.
  Torch, convolution and native fused kernels share this convention;
- a spike's reset and adaptation (`g_k`, `Iasc`, `u`, ...) are applied at the end
  of step `t` and first act at step `t + 1`;
- `DelayedPSC(psc, max_delay_steps=d)` shifts the PSC response by exactly `d`
  further steps.

For example the unit-weight impulse response of `AlphaPSC` is
`k[t] = (t + 1)(1 - a) a^t`. The Torch convolution `multi_step_forward`
matches a single-step rollout's outputs from rest when `kernel_len` covers
the compared time range. This path does not read or update existing PSC/delay
state; output parity does not establish final-state or chunk-continuation
equivalence. Stateful sequence kernels have their own initial/final-state
contract at the primitive multi-step entrypoint.

See [`environ`][btorch.models.environ] for the full environment API.
