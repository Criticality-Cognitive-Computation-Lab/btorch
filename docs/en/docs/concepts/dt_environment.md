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

See [`environ`][btorch.models.environ] for the full environment API.
