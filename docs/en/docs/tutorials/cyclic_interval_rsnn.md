# Cyclic-Interval RSNN Inference

`CyclicIntervalRSNN` is a CUDA inference model for a narrow recurrent
workload: at each timestep, spikes occupy one contiguous interval of neuron
IDs, and that interval advances cyclically by a fixed stride. It owns the full
current, voltage, reset, and sparse-aggregation recurrence, so it is not a
`SparseConn` backend and does not support arbitrary activity patterns,
autograd, or CPU execution.

## Create the Model

The connection follows Btorch's `(source, destination)` convention. The model
transposes it internally into destination-major CSR before moving the topology
to CUDA.

```python
import scipy.sparse

from btorch.models.rnn import CyclicIntervalRSNN

connection = scipy.sparse.eye(4096, dtype="float32", format="csr")
model = CyclicIntervalRSNN(
    connection,
    device="cuda",
    strategy="hybrid_rangegate",
)
```

## Run a Sequence

```python
spikes, voltage, current = model(
    base_start=0,
    active_count=64,
    steps=128,
    stride=31,
    synchronize=True,
)
```

Calls return only the final state. By default, output storage is reused across
calls; set `clone_outputs=True` when retaining more than one result. Launches
use PyTorch's current CUDA stream, so `synchronize=True` is only needed when a
host-side result is immediately required.

## Strategies

All strategies preserve the same recurrence and interval contract:

- `"hybrid_rangegate"` is the default. It uses precomputed per-row spans to
  skip inactive rows and sum entire covered rows.
- `"cyclic_interval"` is the baseline that finds interval boundaries with CSR
  binary searches.
- `"ed_int32_natural"` derives range gates directly from the int32 CSR index
  endpoints.

Use `SparseConn` with the native, `torch_sparse`, or Triton backend when the
input activity is not a contiguous cyclic interval or when gradients are
required.
