---
name: btorch-snn-modelling
description: Patterns for building SNNs with btorch. Use when working with spiking neurons, synaptic dynamics, time-stepped simulations, or implementing custom stateful modules.
---

# btorch SNN Patterns

## Overview

btorch provides stateful neuron/synapse modules and RNN wrappers for neuromorphic modeling.

## Clarify Architecture First

Before writing code, confirm the intended architecture when it is unclear from the user request or existing project config:

- neuron model (for example: GLIF3, LIF, ALIF)
- synapse model (for example: AlphaPSC, AlphaPSCBilleh, ExponentialPSC)
- connection pattern (for example: SparseConn, DenseConn, E/I split, receptor split)

If these are already explicit in user requirements/config, proceed without asking.

## Core Pattern

```python
from btorch.models import environ, functional
from btorch.models.neurons import GLIF3
from btorch.models.synapse import AlphaPSCBilleh
from btorch.models.linear import SparseConn
from btorch.models.rnn import RecurrentNN
from btorch.models.init import uniform_v_

# Build RSNN
conn = SparseConn(conn=weights)

neuron = GLIF3(n_neuron=100, ...)
psc = AlphaPSCBilleh(n_neuron=100, linear=conn, ...)

brain = RecurrentNN(
    neuron=neuron,
    synapse=psc,
    step_mode="m",
    update_state_names=("neuron.v", "synapse.psc"),
)

# Initialize and run
functional.init_net_state(brain, batch_size=4)
uniform_v_(brain.neuron, set_reset_value=True)

with environ.context(dt=1.0):
    spikes, states = brain(x)
functional.reset_net_state(brain)
```

## Key Patterns

### dt Environment

```python
# Always use context manager
with environ.context(dt=1.0):
    spikes, states = model(x)
```

### State Reset

```python
for batch in dataloader:
    functional.reset_net(model, batch_size=batch_size)
    uniform_v_(model.neuron, set_reset_value=True)

    with environ.context(dt=1.0):
        spikes, states = model(x)
```

`set_reset_value=True` stores the uniform voltages as memory reset values. Each `reset_net()` will deterministically reset to these values.

Random init per epoch (`set_reset_value=False`, then `uniform_v_` each batch) acts as regularization (network learns to work from different init states), but can hurt performance.

### Truncated BPTT

```python
for t in range(0, T, chunk_size):
    functional.detach_net(model, state_names=["neuron.v", "synapse.psc"])
    # no functional.reset_net() here - we want to keep the state across chunks
    spikes, states = model(x[t:t+chunk_size])
    loss.backward()
```

### Checkpointing

```python
# Save
checkpoint = {
    "model_state_dict": model.state_dict(),
    "memories_rv": functional.named_memory_reset_values(model),
    "hidden_states": functional.named_hidden_states(model),
}

# Load
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
functional.set_memory_reset_values(model, checkpoint["memories_rv"])
```

## Heterogeneous Modelling

| Topic | Core Idea | Reference |
|-------|-----------|-----------|
| **Delays** | Expand rows with `expand_conn_for_delays`, buffer spikes with `SpikeHistory`, matmul via `get_flattened` | [references/heter_delays.md](references/heter_delays.md) |
| **Receptor-split synapses** | Expand columns with `make_hetersynapse_conn`, run dynamics with `HeterSynapsePSC` | [references/heter_synapses.md](references/heter_synapses.md) |
| **Group-aware weight stacking** | Build per-receptor sparse matrices, stack with `stack_hetersynapse`, map weights with `map_weight_to_conn` | [references/heter_syn_weights.md](references/heter_syn_weights.md) |

## RNN Architecture Choices

| Approach | Use When | Compile? |
|----------|----------|----------|
| `RecurrentNN` | Standard neuron+synapse | Yes |
| Custom `AbstractRNN` | Non-standard architectures | Yes |
| `make_rnn` | Quick prototyping | No |

## Common Pitfalls

1. **Forgetting dt context** - Always wrap forward in `environ.context(dt=...)`
2. **Not resetting state** - Call `reset_net()` before each training batch
3. **Wrong state names** - Use dot notation: `"neuron.v"`
4. **Missing memory reset values** - Dynamic states aren't in `state_dict()`

## References

- [references/training_example.md](references/training_example.md) - Full training loop with checkpointing
- [references/custom_modules.md](references/custom_modules.md) - Custom neuron/synapse patterns
- [references/scaling.md](references/scaling.md) - Neuron scaling and normalization
- [references/regularization.md](references/regularization.md) - Regularization patterns
- [references/performance.md](references/performance.md) - torch.compile, gradient checkpointing, chunked computation
- [references/heter_delays.md](references/heter_delays.md) - Heterogeneous synaptic delays
- [references/heter_synapses.md](references/heter_synapses.md) - Receptor-split (heterogeneous) synapses
- [references/heter_syn_weights.md](references/heter_syn_weights.md) - Group-aware weight stacking and assignment
