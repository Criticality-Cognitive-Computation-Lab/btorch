# Stateful Modules in btorch

Unlike standard PyTorch modules, many btorch components carry internal state that evolves over time. This pattern is essential for spiking neural networks (SNNs), where membrane voltages, synaptic currents, and spike histories must persist between forward steps.

## MemoryModule

At the heart of btorch's state management is [`MemoryModule`][btorch.models.base.MemoryModule]. It extends `nn.Module` with:

- **`register_memory`**: Declares a time-varying buffer (e.g., membrane voltage `v`).
- **`_memories`**: A dictionary of current state values.
- **`_memories_rv`**: Reset values used to restore state at the start of a new batch or trial.

When you call `functional.init_net_state(model, batch_size=4)`, btorch walks the module tree and initializes every `MemoryModule` buffer to the requested shape.

## Typical Lifecycle

```python
from btorch.models import functional

# 1. Initialize state buffers once
functional.init_net_state(model, batch_size=4, device="cuda")

# 2. Apply chosen inialisation method
init.uniform_v_(model.neuron, save_reset_values=False)

# 3. Reset before each batch / trial
functional.reset_net(model, batch_size=4)

# 4. Run forward
with environ.context(dt=1.0):
    spikes, states = model(x)

# 5. Detach for truncated BPTT
functional.detach_net(model)
```

## State Names Are Dotted

`RecurrentNN` records states using dot notation:

```python
states = {
    "neuron.v":       torch.Tensor,  # (T, Batch, N)
    "neuron.Iasc":    torch.Tensor,  # (T, Batch, N, n_asc)
    "synapse.psc":    torch.Tensor,  # (T, Batch, N)
}
```

You can unflatten them with `btorch.utils.dict_utils.unflatten_dict`:

```python
from btorch.utils.dict_utils import unflatten_dict
nested = unflatten_dict(states, dot=True)
# nested["neuron"]["v"]  -> (T, Batch, N)
```

## Why state_dict() Is Not Enough

Dynamic buffers (membrane voltages, synaptic currents) are intentionally excluded from `state_dict()` because their shapes depend on batch size and they are reconstructed at runtime. To checkpoint a trained model, save:

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
}
```

Optionally, also save if needed:

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "memories_rv": functional.named_memory_reset_values(model),  # if reset values are randomized
    "hidden_states": functional.named_hidden_states(model),      # if you need neuron state
}
```

See [`functional`][btorch.models.functional] for the full state-management API.
