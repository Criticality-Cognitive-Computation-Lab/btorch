# Tutorial 1: Building an RSNN

**Author:** btorch contributors  
**Based on:** [`examples/rsnn.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/examples/rsnn.py)

This tutorial walks through building a minimal Recurrent Spiking Neural Network (RSNN) with btorch.

## What You Will Learn

- How to compose a neuron (`GLIF3`), a synapse (`AlphaPSC`), and a recurrent wrapper (`RecurrentNN`).
- How to initialize and reset network state.
- How to record internal state variables (voltage, current) during a forward pass.
- How `dt` context governs ODE integration.

## The Building Blocks

A minimal RSNN needs three things:

1. **Neuron model** — defines how membrane voltage evolves and when spikes fire.
2. **Synapse model** — transforms spikes into post-synaptic currents.
3. **Connection layer** — defines the weight matrix between neurons.

```python
import torch
import torch.nn as nn
from btorch.models import environ, functional, glif, rnn, synapse
from btorch.models.linear import DenseConn


class MinimalRSNN(nn.Module):
    def __init__(self, num_input, num_hidden, num_output, device="cpu"):
        super().__init__()

        # 1. Input projection
        self.fc_in = nn.Linear(num_input, num_hidden, bias=False, device=device)

        # 2. Spiking neuron
        neuron_module = glif.GLIF3(
            n_neuron=num_hidden,
            v_threshold=-45.0,
            v_reset=-60.0,
            c_m=2.0,
            tau=20.0,
            tau_ref=2.0,
            k=[0.1, 0.2],
            asc_amps=[1.0, -2.0],
            step_mode="s",   # single-step definition
            backend="torch",
            device=device,
        )

        # 3. Recurrent connection
        conn = DenseConn(num_hidden, num_hidden, bias=None, device=device)

        # 4. Synapse
        psc_module = synapse.AlphaPSC(
            n_neuron=num_hidden,
            tau_syn=5.0,
            linear=conn,
            step_mode="s",
        )

        # 5. Recurrent wrapper (multi-step)
        self.brain = rnn.RecurrentNN(
            neuron=neuron_module,
            synapse=psc_module,
            step_mode="m",
            update_state_names=("neuron.v", "synapse.psc"),
        )

        # 6. Output readout
        self.fc_out = nn.Linear(num_hidden, num_output, bias=False, device=device)

    def forward(self, x):
        x = self.fc_in(x)                 # (T, Batch, num_input) -> (T, Batch, N)
        spike, states = self.brain(x)     # spike: (T, Batch, N)
        rate = spike.mean(dim=0)          # (Batch, N)
        out = self.fc_out(rate)           # (Batch, num_output)
        return out
```

## Initializing State

Before the first forward pass, call `init_net_state` to register and initialize memory buffers:

```python
model = MinimalRSNN(num_input=20, num_hidden=64, num_output=5)
functional.init_net_state(model, batch_size=4, device="cpu")
```

## Running a Forward Pass

Wrap the forward pass in an `environ.context(dt=...)` block:

```python
environ.set(dt=1.0)
inputs = torch.rand((100, 4, 20))  # (T, Batch, input_dim)

functional.reset_net(model, batch_size=4)
with environ.context(dt=1.0):
    out = model(inputs)  # (Batch, num_output)
```

## Inspecting Recorded States

`update_state_names` tells `RecurrentNN` which variables to save. The returned `states` dict uses dot notation:

```python
with environ.context(dt=1.0):
    spike, states = model.brain(inputs)

print(states["neuron.v"].shape)     # (T, Batch, N)
print(states["synapse.psc"].shape)  # (T, Batch, N)
```

You can unflatten the dict for easier access:

```python
from btorch.utils.dict_utils import unflatten_dict
nested = unflatten_dict(states, dot=True)
nested["neuron"]["v"][:, 0, :]   # voltage of batch 0
```

## What Comes Next

In [Tutorial 2: Training an SNN](training.md), we add a loss function, a training loop, and checkpointing.
