# Scaling and Normalization Patterns

## Neuron Scaling

Scale voltages for numerical stability and faster gradient convergence:

```python
# Register scaling buffers
self.register_buffer("neuron_scale", v_threshold - v_reset)
self.register_buffer("neuron_zeropoint", v_reset)

# Scale input: x_eff = x / neuron_scale
scaled_input = input_current / self.neuron_scale
```

## Scale Modes

| Mode | Description |
|------|-------------|
| `average` | Global mean across all neurons |
| `cell_type_average` | Per-cell-type averaging |
| `per_neuron` | Individual per-neuron scaling |

## Input Scaling with Rheobase

Rheobase is calculated from neuron parameters (v_threshold, v_reset, c_m, tau). When neuron parameters are scaled, rheobase is **automatically scaled** too.

```python
from btorch.models.neurons.glif import get_rheobase

rheobase = get_rheobase(v_threshold, v_rest, c_m, tau)
# If v_threshold/v_reset were scaled, rheobase is already scaled
```

**Critical**: Rheobase scaling and neuron scaling interact. Rheobase is calculated from scaled neuron parameters, so it's already scaled. When both `multiply_input_by_rheobase` and `scale_input` are enabled, avoid double-scaling:

```python
# rheobase is already scaled (derived from scaled v_threshold, v_reset)
if multiply_input_by_rheobase:
    # Don't also apply neuron_scale - rheobase already includes it
    x = rheobase * x
else:
    if scale_input:
        x = x / neuron_scale
```

## LearnableScale Module

Used for calibrating input and output data ranges. RSNN performance depends heavily on proper input/output scales.

```python
from btorch.models.linear import LearnableScale

self.input_scaling = LearnableScale(scale=1.0, bias=None)
scaled_input = self.input_scaling(raw_input)

# Freeze after calibration
self.input_scaling.requires_grad_(False)
```

## Dale's Law Constraint

Enforce E/I separation after optimizer step:

```python
from btorch.models import constrain

# After optimizer.step()
constrain.constrain_net(model)
# Or specific layer
constrain.constrain_(model.fc1.weight, self.e_mask, self.i_mask)
```
