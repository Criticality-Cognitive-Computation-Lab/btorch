# Surrogate Gradients

Spiking neurons use a discontinuous activation (a spike is emitted when the membrane voltage crosses a threshold). This discontinuity makes standard backpropagation through time (BPTT) impossible because the gradient of the spike function is zero almost everywhere.

Surrogate gradients solve this by replacing the true gradient with a smooth approximation during the backward pass.

## Available Surrogates

btorch provides several surrogate gradient functions in `btorch.models.surrogate`:

| Class | Forward | Backward |
|-------|---------|----------|
| `Sigmoid` | Heaviside | Sigmoid derivative |
| `ATan` | Heaviside | Arc-tangent derivative |
| `Triangle` | Heaviside | Piecewise linear |
| `Erf` | Heaviside | Gaussian (error function) derivative |

## Usage

Most neuron constructors accept a `surrogate_function` argument:

```python
from btorch.models.neurons import LIF
from btorch.models.surrogate import ATan

neuron = LIF(
    n_neuron=100,
    surrogate_function=ATan(),
)
```

If not specified, a sensible default (usually `ATan`) is used.

## Choosing a Surrogate

- **ATan** — Smooth, well-behaved gradients; good default for most tasks.
- **Sigmoid** — Stronger gradient far from threshold; can help with very sparse activity.
- **Triangle** — Computationally cheap; bounded support.
- **Erf** — Very smooth; sometimes helps with optimization stability.

See the [Models API](../api/models.md) for constructor details.
