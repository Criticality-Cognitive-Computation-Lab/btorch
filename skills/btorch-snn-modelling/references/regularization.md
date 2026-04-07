# Regularization Patterns

## Purpose

Voltage and firing rate regularization enforce biorealistic regime:
- Clamp voltage near threshold (promotes spikes for gradient backprop)
- Avoid excessive firing (prevents wrong gradient credit assignment to overactive neurons)
- Keep network in trainable, stable regime

## Calibration

Regularization losses must be balanced against task loss. Scale them so they don't dominate.

### Quick Calibration

Run a few samples, visualize, tune:

```python
# 1. Run forward, collect traces
spikes, states = model(x)
v = states["neuron.v"]  # (T, batch, n_neuron)

# 2. Plot voltage traces and raster
# Check: voltage staying near threshold? firing rate in [2-30] Hz?

# 3. Tune lambdas so reg losses are ~10-20% of task loss
loss = task_loss + 0.1 * v_loss + 0.01 * rate_loss
```

Can calibrate jointly with input/output `LearnableScale`.

### Precise Calibration (Tedious)

Match gradient magnitudes on trainable params:

```python
# Backward each loss separately, compare param grad magnitudes
task_loss.backward(retain_graph=True)
task_grad_norm = sum(p.grad.norm() for p in model.parameters())

model.zero_grad()
v_loss.backward(retain_graph=True)
v_grad_norm = sum(p.grad.norm() for p in model.parameters())

# Set lambda so v_grad_norm ~ 0.1 * task_grad_norm
```

## Voltage Regularization

```python
from btorch.models.regularizer import VoltageRegularizer

self.voltage_reg = VoltageRegularizer(
    v_threshold, v_reset, voltage_cost=1.0
)

v_loss = self.voltage_reg(states["neuron.v"])
loss = task_loss + voltage_lambda * v_loss
```

Keeps voltage near threshold for spike generation.

## Firing Rate Regularization

```python
# Simple: keep in [2, 30] Hz range
actual_rate = states["neuron.spike"].mean() * 1000.0 / dt
rate_loss = (
    torch.clamp(2.0 - actual_rate, min=0.0) ** 2 +
    torch.clamp(actual_rate - 30.0, min=0.0) ** 2
).mean()

loss = task_loss + rate_lambda * rate_loss
```

Prevents silent or excessively firing neurons.

## Trade-offs

- Too tight constraints hurt training (network can't explore)
- Voltage and rate regularization can fight each other
  - High voltage threshold promotes spikes
  - Low firing rate penalizes spikes
  - If this happens, reduce or disable one
- Start with loose constraints, tighten gradually
