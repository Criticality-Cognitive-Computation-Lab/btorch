# Heterogeneous Synapses

Use `make_hetersynapse_conn` to expand the **column** dimension of a connection matrix so each post-neuron gets a block of columns per receptor type. Use `HeterSynapsePSC` to run synaptic dynamics on the expanded space and sum across receptors per neuron.

## Decision Table

| Situation | Mode / Component |
|-----------|------------------|
| Receptor types are neuron properties (e.g., E/I) | `receptor_type_mode="neuron"` |
| Receptor types are connection properties (cotransmission) | `receptor_type_mode="connection"` |
| Group-constrained training (e.g., per-cell-type magnitudes) | `SparseConstrainedConn.from_hetersynapse` |
| Inspect PSC for one receptor pair/type | `hetero_psc.get_psc(receptor_type=...)` |

## Neuron-Mode Heterosynapse

```python
from btorch.connectome.connection import make_hetersynapse_conn
from btorch.models.synapse import HeterSynapsePSC, AlphaPSC
from btorch.models.linear import SparseConn
from btorch.models import environ, functional

conn, receptor_idx = make_hetersynapse_conn(
    neurons=neurons_df,
    connections=conn_df,
    receptor_type_col="EI",
    receptor_type_mode="neuron",  # yields E->E, E->I, I->E, I->I
)
n_receptor = len(receptor_idx)
n_neuron = len(neurons_df)

linear = SparseConn(conn, enforce_dale=False)

with environ.context(dt=1.0):
    psc = HeterSynapsePSC(
        n_neuron=n_neuron,
        n_receptor=n_receptor,
        receptor_type_index=receptor_idx,
        linear=linear,
        base_psc=AlphaPSC,
        tau_syn=5.0,
    )
    functional.init_net_state(psc, batch_size=4)
    out = psc(spike)  # summed PSC across receptors

# Inspect per-receptor contribution
psc_ee = psc.get_psc(receptor_type=("E", "E"), psc=psc.base_psc.psc)
```

## Connection-Mode Heterosynapse

Use `receptor_type_mode="connection"` when receptor types are properties of the connections themselves (e.g., cotransmission). The `receptor_type_index` will have a single `receptor_type` column, and `get_psc` accepts a string:

```python
psc_conn = psc.get_psc(receptor_type="glutamate", psc=psc.base_psc.psc)
```

## Constrained Connection from Heterosynapse

```python
from btorch.connectome.connection import make_hetersynapse_constrained_conn
from btorch.models.linear import SparseConstrainedConn

conn, constraint, receptor_idx = make_hetersynapse_constrained_conn(
    neurons=neurons_df,
    connections=conn_df,
    cell_type_col="cell_type",
    receptor_type_col="EI",
    receptor_type_mode="neuron",
    constraint_mode="cell_and_receptor",
)
linear = SparseConstrainedConn.from_hetersynapse(
    conn=conn,
    constraint=constraint,
    receptor_type_index=receptor_idx,
    enforce_dale=True,
)
```

## Common Pitfalls

1. **Wrong `get_psc` argument type** — In neuron mode, `get_psc` expects a tuple `(pre_type, post_type)`. In connection mode, it expects a string. Passing the wrong type raises `ValueError`.
2. **Mismatched linear output size** — The `linear` layer must map to `n_neurons * n_receptor` (or use the expanded conn directly). `SparseConn` handles this automatically when initialized with the expanded matrix.
3. **Confusing `return_dict=True`** — `make_hetersynapse_conn(..., return_dict=True)` returns an `OrderedDict` of per-receptor matrices. If you want a single stacked matrix, use `return_dict=False` (default).
