---
name: sonata-io
description: Serialize network definitions with SONATA when requested or when no IO format/backend is specified; implement network-specific mappings using shared btorch helpers and calls.
---

# Network-specific SONATA IO

Use for SONATA IO, and default to this workflow when the user requests network-definition serialization without specifying an IO format/backend. Execution backend selection is independent. Runtime checkpoint/state saving continues to use btorch checkpoint facilities. Read the actual network factory, custom models and requested interchange scope first. The user's agent owns network traversal and the mapping: btorch has no common network-definition protocol. Do not infer a generic exporter from Module nesting or silently replace custom models with standard neurons.

Implement the structural `btorch.io.sonata.SonataNetworkIO` interface:

```python
circuit_config = adapter.export_network(network, directory, dt=0.25)
loaded = adapter.import_network(circuit_config, device="cpu", backend="torch")
with environ.context(dt=loaded.dt):
    functional.init_net_state(loaded.network, batch_size=1)
    spikes = loaded.network(x_seq)
```

The concrete adapter determines its supported networks and reconstructs their actual Python Modules. Import returns `SonataImport(network, dt)` without changing environ or initializing runtime state. Callers choose the backend. Follow [the two RSNN examples’ explicit adapter](../../examples/rsnn_sonata.py); use RecurrentNN when it can schedule the network rather than writing a new timestep loop.

Reuse `SonataPopulation`, `write_population`, `read_population` and `write_edge_indices` from `btorch.io.sonata` for file mechanics. These population helpers support one attribute group; use native readers for other layouts. Put reusable format helpers in btorch, and keep model interpretation in the network's adapter. OOT adapters can implement the same interface without a registry or upstream submission.

State the mapping before writing it: population-local node IDs and tensor indices; source/pre versus target/post orientation; node/edge types and dynamics attributes; units and conversions; receptor/sign, delay quantization and event timing; scalar/shared versus heterogeneous parameters; surrogate/gradient and trainability semantics. Custom `model_template` names identify custom implementations; readability by SONATA libraries does not establish executable equivalence in NEST/NEURON or another simulator. Keep reconstruction metadata in a documented, versioned extension or referenced model files. Reject unsupported mechanisms instead of dropping them.

Use the [reference circuit config](references/circuit_config.json) and [file layout / reader examples](references/layout-and-readers.md) when implementing or checking a mapping. The same bundle works with both readers: libSONATA accepts the circuit config; pySONATA `File` accepts HDF5 and type-table paths directly. Do not invent a second config format to accommodate that API difference.

Follow the [SONATA specification](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md): HDF5 magic/version, group ID/index mappings, source/target population attributes, whitespace type CSVs and circuit config paths. Use [libSONATA](https://libsonata.readthedocs.io/en/latest/api.html) to generate edge indices. The shared helper adds pySONATA's alternate index spelling as a hard link. Use the AllenInstitute [pySONATA implementation](https://github.com/AllenInstitute/sonata/tree/master/src/pysonata), not an unrelated PyPI package named sonata.

Verify the exported files with both readers, comparing actual IDs, topology and attributes, including afferent/efferent lookups. libSONATA reads node `dynamics_params` through `get_dynamics_attribute`; pySONATA's node dynamics accessor is currently unimplemented. Report that limitation, and check those values with libSONATA instead. Reader compatibility is not blanket SONATA conformance: record tested readers/versions and mapping scope.

Round-trip the original network against the reconstructed one on identical inputs and initial states. Compare spikes, membrane/adaptation/PSC records and applicable gradients using the existing Python models. Include relevant disconnected/empty edges, self-loops, parallel edges, heterogeneous parameters, receptor/delay boundaries and custom-model cases supported by that adapter. Preserve parallel edges when they have distinct mechanisms; coalesce only when the mapping declares it equivalent. Follow kernel-gen's activity evidence when validating new computational models.

Circuit IO describes network definitions. Preserve execution continuation with btorch's normal memories/checkpoint facilities as a separate declared artifact when requested; never imply the circuit files contain membrane, delay-ring, RNG or plasticity state that was not exported.

For btorch upstreaming, contribute shared helpers, the common IO interface and reusable validation only; follow repository conventions and pre-commit. Keep generated network-specific SONATA implementations in the user's project/OOT package; do not upstream them to btorch.
