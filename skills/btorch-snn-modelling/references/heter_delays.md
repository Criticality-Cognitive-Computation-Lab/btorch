# Heterogeneous Synaptic Delays

Use `expand_conn_for_delays` to expand the **row** dimension of a sparse connection matrix so each pre-neuron gets `n_delay_bins` virtual rows. Pair it with `SpikeHistory` to buffer past spikes and `history.get_flattened(n_delays)` to feed the expanded matrix.

## Decision Table

| Use Case | Component |
|----------|-----------|
| Simulation only (memory-efficient) | `SpikeHistory(..., use_circular_buffer=True)` |
| Training with `torch.compile` | `SpikeHistory(..., use_circular_buffer=False)` or `DelayedSynapse(..., use_circular_buffer=False)` |
| Delays + receptor split together | `make_hetersynapse_conn(..., delay_col="delay_steps", n_delay_bins=5)` |

## Basic Delay Pattern

```python
import numpy as np
import scipy.sparse
import torch

from btorch.connectome.connection import expand_conn_for_delays
from btorch.models.history import SpikeHistory
from btorch.models.linear import SparseConn

conn = scipy.sparse.coo_array(([5.0], ([0], [1])), shape=(2, 2))
delays = np.array([2])  # one delay per non-zero entry

conn_d = expand_conn_for_delays(conn, delays, n_delay_bins=5)
linear = SparseConn(conn_d, enforce_dale=False)

history = SpikeHistory(n_neuron=2, max_delay_steps=5)
history.init_state(batch_size=1)

for t in range(T):
    spike = neuron(input[t])  # shape (batch, n_neuron)
    history.update(spike)
    z_delayed = history.get_flattened(n_delays=5)  # (batch, n_neuron * n_delays)
    psc = linear(z_delayed)  # (batch, n_neuron)
```

## Convenience Wrapper

```python
from btorch.models.history import DelayedSynapse

synapse = DelayedSynapse(n_neuron=100, linear=linear, max_delay_steps=5)
synapse.init_state(batch_size=4)
psc = synapse(spike)  # updates history and computes PSC
```

## Delays + Heterosynapse Combined

```python
from btorch.connectome.connection import make_hetersynapse_conn

conn, receptor_idx = make_hetersynapse_conn(
    neurons=neurons_df,
    connections=conn_df,
    receptor_type_col="EI",
    receptor_type_mode="neuron",
    delay_col="delay_steps",
    n_delay_bins=5,
)
# conn shape: (n_neurons * n_delay_bins, n_neurons * n_receptors)
```

## Common Pitfalls

1. **Mismatched delay sizes** — `n_delay_bins`, `max_delay_steps`, and the `n_delays` argument to `get_flattened` must match. Mismatches silently truncate or raise `IndexError`.
2. **Forgetting to call `init_state`** — `SpikeHistory` and `DelayedSynapse` require `init_state(batch_size=...)` before the first `update()`.
3. **Wrong buffer mode for training** — `use_circular_buffer=True` is efficient for simulation but can break `torch.compile`. Use `False` when compiling the model for training.
4. **`dt` needed at instantiation** — Because `dt` is read from `environ`, modules that use delays (e.g., `DelayedSynapse` wrapped inside a synapse model) may need `with environ.context(dt=1.0):` even during model construction or `init_net_state()` calls. If `dt` is missing, instantiation or state initialization can raise.
