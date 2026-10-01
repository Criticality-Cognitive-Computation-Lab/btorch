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

All neurons and post-synaptic current (PSC) models share one convention:

- an input (current, or a spike delivered to a PSC) at step `t` already affects
  the returned `v` / spike / `psc` of step `t`;
- a spike's reset and adaptation (`g_k`, `Iasc`, `u`, ...) are applied at the end
  of step `t` and first act at step `t + 1`;
- `DelayedPSC(psc, max_delay_steps=d)` shifts the PSC response by exactly `d`
  further steps.

For example the unit-weight impulse response of `AlphaPSC` is
`k[t] = (t + 1)(1 - a) a^t`, so `multi_step_forward` (convolution with `k`) equals
stepping `single_step_forward` exactly.

See [`environ`][btorch.models.environ] for the full environment API.
