# Tutorial: Triton Event-Driven Sparse Connections

This tutorial shows how to use btorch's Triton event-driven sparse backend for
CUDA workloads with low activity. The backend groups active source neurons into
work items and evaluates only the affected sparse tasks. It is intended for
large recurrent networks where most input entries are zero at each timestep.

The backend is selected per connection with `sparse_backend="triton"`. A
`RecurrentNN` or `make_rnn` wrapper prepares Triton sparse modules around its
multi-step loop automatically. Code that calls `SparseConn` directly in a time
loop should use `prepare_sparse_modules` explicitly so packed weights and the
CUDA workspace are reused across timesteps.

## Requirements and Scope

The Triton backend currently has these runtime requirements:

- a CUDA device and a CUDA-enabled PyTorch installation;
- the `triton` Python package;
- `torch.float32` inputs and sparse weights.

The implementation supports arbitrary leading dimensions in the input. For
example, `(batch, neurons)` and `(time, batch, neurons)` are flattened to a
two-dimensional launch internally and restored on return. First-order
autograd backward is also supported for the input and learnable sparse
connection magnitudes. Use the native or `torch_sparse` backend when a
different dtype or CPU execution is required.

## Build a Triton Sparse Connection

`SparseConn` accepts a SciPy sparse matrix with shape
`(num_source, num_destination)`:

```python
import numpy as np
import scipy.sparse
import torch

from btorch.models.linear import SparseConn

n_neuron = 1024
fanout = 16
rng = np.random.default_rng(42)

source = np.repeat(np.arange(n_neuron), fanout)
destination = rng.integers(0, n_neuron, size=source.size)
weight = rng.normal(scale=0.1, size=source.size).astype(np.float32)
connectivity = scipy.sparse.coo_array(
    (weight, (source, destination)),
    shape=(n_neuron, n_neuron),
)

connection = SparseConn(
    connectivity,
    enforce_dale=False,
    sparse_backend="triton",
    device="cuda",
    dtype=torch.float32,
)

spikes = (torch.rand(4, n_neuron, device="cuda") < 0.01).float()
current = connection(spikes)
assert current.shape == (4, n_neuron)
```

The default Triton configuration enables source reordering, source blocking,
and destination hashing. These options can be overridden per module or through
the global configuration registry. See the [configuration guide](../guides/configuration.md)
for the available options.

## Reuse Preparation in a Time Loop

Use `prepare_sparse_modules` around the complete loop when calling a sparse
connection directly. Pass the runtime batch size so the reusable workspace has
enough queue capacity:

```python
from btorch.models.functional import prepare_sparse_modules

timesteps = 100
batch_size = 4
spikes = (
    torch.rand(timesteps, batch_size, n_neuron, device="cuda") < 0.01
).float()

with torch.inference_mode(), prepare_sparse_modules(
    connection,
    batch_size=batch_size,
):
    currents = torch.stack([connection(spikes[t]) for t in range(timesteps)])

assert currents.shape == (timesteps, batch_size, n_neuron)
```

Preparation is nested safely and is released when the context exits. The
`RecurrentNN.multi_step_forward` path already creates this scope, so no extra
wrapper is needed when the connection is part of an RNN model.

## Batch Dimensions and Backward

Leading dimensions are supported as long as the final dimension is the source
neuron dimension:

```python
x = torch.randn(2, 3, n_neuron, device="cuda", requires_grad=True)
y = connection(x)
loss = y.square().mean()
loss.backward()

assert y.shape == (2, 3, n_neuron)
assert x.grad is not None
assert connection.magnitude.grad is not None
```

This is first-order backward. The Triton forward kernel is followed by a
PyTorch-based backward implementation for the input and packed edge weights,
so training is supported, although backward may have a different performance
profile from forward. The explicit preparation scope also works in grad mode;
the packed weight then retains the graph needed by backward.

## Configuration

```python
from btorch import config

triton = config.sparse.backend("triton")
triton.reorder = True
triton.block = True
triton.hash = True

# New SparseConn instances use this template.
connection = SparseConn(connectivity, sparse_backend="triton")

# Override only this module.
connection = SparseConn(
    connectivity,
    sparse_backend="triton",
    sparse_config={"reorder": False, "block": True, "hash": False},
)
```

Backend configuration is copied when `SparseConn` is constructed. Updating the
global template later affects newly created modules, not existing modules.

## Compile and Benchmark

The Triton backend can be used with `torch.compile` for the surrounding model.
The benchmark below compares a native sparse path with this backend at a
scale-matched proxy size:

```bash
micromamba run -n ml-py312 \
  python benchmarks/sparse_rnn/benchmark_compile_triton_proxy.py \
  --backend native \
  --output benchmarks/sparse_rnn/results/rtx5090_compile_proxy_native.json

micromamba run -n ml-py312 \
  python benchmarks/sparse_rnn/benchmark_compile_triton_proxy.py \
  --backend triton \
  --output benchmarks/sparse_rnn/results/rtx5090_compile_proxy_triton.json
```

### Measured Scale-Matched Proxy

The following measurements were collected on an RTX 5090 with PyTorch
2.11.0+cu129. This is a scale-matched proxy, not the original application
topology: the original topology was unavailable in the benchmark environment.
Both paths used the same generated topology and input history.

| Setting | Value |
| --- | --- |
| Neurons | 32,768 |
| Sampled edges | 512 per neuron per projection |
| Coalesced edges | 16,646,991 per projection |
| Recurrent projections | 2 |
| Sequence length | `T=1000` |
| Unroll | 8 |
| Activity | Fixed 0.2999054% |
| Dtype | `float32` |
| Compiler | `torch.compile(mode="default")` |
| CUDA graphs | Disabled in the model and `torch._inductor` |
| Scan | Not used |

The timings below are steady-state forward-loop timings. First compile time is
reported separately because it is setup cost, not a steady-state timestep
measurement.

| Backend | Eager | Compiled | Compiled / eager |
| --- | ---: | ---: | ---: |
| Native sparse | 881.1417 ms | 903.7372 ms | 0.97500x |
| Triton event-driven sparse | 199.6218 ms | 120.3175 ms | 1.65913x |

Relative to the native path, Triton eager was 4.414x faster than native eager,
Triton compiled was 7.323x faster than native eager, and Triton compiled was
7.511x faster than native compiled. First compile took 2.3538 s for native and
4.4449 s for Triton. The maximum eager-versus-compiled absolute error was
`1.49e-8` for both backends.

These results show the benefit of the event-driven kernel for this low-activity
proxy, but they do not predict performance for every topology or activity
rate. Measure with the actual graph, dtype, sequence length, batch size, and
activity distribution used by the application. Include preparation and compile
costs separately when comparing short runs.

## Choosing a Backend

- Use `triton` for CUDA `float32` workloads with low activity and repeated
  sparse calls.
- Use `native` for a dependency-free PyTorch fallback or CPU execution.
- Use `torch_sparse` when that package is already part of the deployment and
  its workload-specific performance is preferable.

See [`examples/triton_sparse_manual_loop.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/examples/triton_sparse_manual_loop.py)
for a complete manual-loop example and the sparse preprocessing tests for
validated forward, batch, backward, and CUDA graph usage.
