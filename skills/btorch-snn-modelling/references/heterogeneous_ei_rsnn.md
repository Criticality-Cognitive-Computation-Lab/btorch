# Semantic-Edge Multi-Receptor E/I RSNN

Represent E/I receptor routing directly on the edges:

```python
connection = SparseConnection.from_edges(
    pre,
    post,
    n_pre=n_neuron,
    n_post=n_neuron,
    synapse=Synapse(
        receptor=receptor_id,
        n_receptor=4,
        dale=True,
    ),
    values=signed_initial_weight,
)
psc = HeterSynapsePSC(
    n_neuron=n_neuron,
    n_receptor=4,
    receptor_type_index=receptor_index,
    linear=connection,
    base_psc=AlphaPSC,
    tau_syn=tau_by_receptor.repeat(n_neuron),
)
```

Use a receptor index table with `receptor_index`,
`pre_receptor_type`, and `post_receptor_type` columns for neuron-mode E/I
channels. Inspect a channel with `psc.get_psc(("I", "E"))`; inspect current
edge weights with `connection.weight()`. After `optimizer.step()`, call
`constrain_net(model)` to enforce Dale's law.

For new code, do not multiply population dimensions by the number of
receptors or delays. Those flattened dimensions are derived by the connection
and PSC. Use `from_hetersynapse` only to import physically expanded matrices
from older connectome pipelines. See [the English tutorial](../../../docs/en/docs/tutorials/heterogeneous_ei_rsnn.md)
for a complete runnable workflow.
