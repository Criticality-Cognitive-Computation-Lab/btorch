# Custom Module Patterns

## MemoryModule Base

All stateful modules inherit from `MemoryModule`:

```python
from btorch.models.base import MemoryModule

class MyNeuron(MemoryModule):
    def __init__(self, n_neuron):
        super().__init__()
        self.n_neuron = (n_neuron,)
        
        # (name, default_value, shape)
        self.register_memory("v", 0.0, n_neuron)
        self.register_memory("refractory", 0, n_neuron)
    
    def forward(self, x):
        self.v = self.v + x  # Persists across calls
        return self.v
```

## Custom State Initialization

When the default `MemoryModule` behavior doesn't fit (e.g., state shape depends on runtime `batch_size`), override `init_state()` and `reset()`:

```python
class CustomSynapse(MemoryModule):
    def __init__(self, n_neuron):
        super().__init__()
        self.n_neuron = (n_neuron,)
        self.register_memory("psc", 0.0, n_neuron)
        # Don't register dynamic-shaped memory here
    
    def init_state(self, batch_size=None, dtype=None, device=None, **kwargs):
        # Initialize standard memories
        super().init_state(batch_size, dtype, device)
        
        # Handle dynamic-shaped buffer manually
        sizes = (self.latency_steps + 1, batch_size, self.n_neuron[0])
        self.register_buffer(
            "delay_buffer", torch.zeros(sizes, dtype=dtype, device=device)
        )
    
    def reset(self, batch_size=None, dtype=None, device=None, **kwargs):
        # Reset standard memories
        super().reset(batch_size, dtype, device)
        
        # Custom reset for dynamic buffer
        if hasattr(self, "delay_buffer"):
            self.delay_buffer.zero_()
```

## ParamBufferMixin for Trainable Parameters

Use when parameters can be trainable or fixed:

```python
from btorch.models.base import MemoryModule, ParamBufferMixin

class CustomNeuron(MemoryModule, ParamBufferMixin):
    def __init__(self, n_neuron, trainable_param=set()):
        super().__init__()
        self.n_neuron = (n_neuron,)
        self.trainable_param = trainable_param
        
        # Parameter if in trainable_param, else buffer
        self.def_param("tau", 20.0, sizes=n_neuron)
        self.def_param("v_threshold", 1.0, sizes=n_neuron)
        
        # Memory (state, not parameter)
        self.register_memory("v", 0.0, n_neuron)
```

## Memory vs Buffer vs Parameter

| Type | Method | Use For | In state_dict |
|------|--------|---------|---------------|
| Memory | `register_memory` | Dynamic state (v, psc) | No |
| Buffer | `register_buffer` | Fixed tensors, shapes | Yes |
| Parameter | `def_param` | Trainable/fixed params | Yes |
| nn.Parameter | Always trainable | Trainable params | Yes |

**Key distinction**: Memory gets reset to stored reset value; not in state_dict.
