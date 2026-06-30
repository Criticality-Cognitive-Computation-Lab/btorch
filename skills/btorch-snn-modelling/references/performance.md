# Performance Optimization

## torch.compile

`RecurrentNN` is designed to work with `torch.compile`. Wrap your model after construction:

```python
from btorch.models.rnn import RecurrentNN

brain = RecurrentNN(
    neuron=neuron,
    synapse=psc,
    step_mode="m",
)

# Compile for faster training
brain = torch.compile(brain)
```

**Note**: Requires torch >= 2.8. torch.compile 2.7 has known segfault issues with torch_sparse.

**Long sequences**: torch.compile can be slow with very long sequences due to excessive graph unrolling. Use the `unroll` parameter to limit timesteps per compilation unit:

```python
brain = RecurrentNN(
    neuron=neuron,
    synapse=psc,
    step_mode="m",
    unroll=8,  # Compile unrolls 8 steps at a time (default)
)
```

Reduce `unroll` if compilation is slow; increase if you need faster forward pass and have VRAM.

## Gradient Checkpointing

Enable gradient checkpointing in `RecurrentNN` to trade compute for memory:

```python
brain = RecurrentNN(
    neuron=neuron,
    synapse=psc,
    step_mode="m",
    grad_checkpoint=True,  # Saves GPU memory
)
```

Use when training large models that don't fit in GPU memory.

## Chunked Computation with CPU Offload

For very long sequences in simulation, use chunked computation. Setting `chunk_size` automatically enables `cpu_offload`:

```python
brain = RecurrentNN(
    neuron=neuron,
    synapse=psc,
    step_mode="m",
)

# Enable chunked computation - chunk_size automatically enables cpu_offload
brain.chunk_size = 128  # Process 128 timesteps at a time
```

Or configure at simulation time:

```python
def sim(self, x, dt, chunk_size=None):
    if chunk_size is not None:
        self.brain.cpu_offload = True
        self.brain.chunk_size = chunk_size
    else:
        self.brain.cpu_offload = False
        self.brain.chunk_size = None
    
    with environ.context(dt=dt):
        output, states = self(x)
    return output, states
```

**How it works**:

- Sequence is split into chunks of `chunk_size`
- Each chunk is processed on GPU
- Intermediate states are offloaded to CPU between chunks (via `cpu_offload`)
- Reduces peak GPU memory usage significantly
- Only for simulation (inference), not training

## Memory Optimization Summary

| Technique | Trade-off | Use When |
|-----------|-----------|----------|
| `torch.compile` | Compile time | Production training (not very long timesteps) |
| `unroll` | Limits graph size | Long timesteps with compile |
| `grad_checkpoint` | Recomputation | Long timesteps in training, OOM errors |
| `chunk_size` | CPU-GPU transfer overhead | Long timesteps in simulation |

## When to Use Each

**Training** (gradients needed):
```python
brain = RecurrentNN(
    neuron=neuron,
    synapse=psc,
    step_mode="m",
    grad_checkpoint=True,  # Save memory during backward
)
brain = torch.compile(brain)
```

**Simulation** (inference only, long sequences):
```python
brain = RecurrentNN(
    neuron=neuron,
    synapse=psc,
    step_mode="m",
)
brain.chunk_size = 128  # Process long sequences in chunks (enables cpu_offload)
```
