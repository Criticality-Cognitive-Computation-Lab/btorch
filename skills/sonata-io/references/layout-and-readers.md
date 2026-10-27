# Reference circuit layout and reader entry points

Use [circuit_config.json](circuit_config.json) as a minimal circuit-file reference,
not as a complete btorch reconstruction descriptor. Generate a runnable bundle:

```bash
python -m examples.rsnn --neurons 8 --steps 32 --export-sonata /tmp/glif-circuit
```

The adapter adds its versioned `btorch` mapping extension to the config. Copying
only this reference JSON will not preserve model shapes, trainability or timestep.
For Izhikevich, the population is `izhikevich`; use that example's exported config.

```text
circuit_config.json
node_types.csv     # whitespace-separated; not comma-separated
edge_types.csv
nodes.h5
  @magic = uint32(0x0A7A), @version = uint32[0, 1]
  /nodes/glif3/
    node_id, node_type_id, node_group_id, node_group_index
    0/dynamics_params/<scalar component columns>
edges.h5
  @magic = uint32(0x0A7A), @version = uint32[0, 1]
  /edges/recurrent/
    source_node_id  @node_population = "glif3"
    target_node_id  @node_population = "glif3"
    edge_type_id, edge_group_id, edge_group_index
    0/{syn_weight, receptor, delay, dale_sign}  # dale_sign only with Dale constraints
    indices/{source_to_target,target_to_source}/
      node_id_to_ranges
      node_id_to_range  # hard link to the plural dataset
      range_to_edge_id
```

Group IDs are zero in this helper's layout; group indices map population rows to
attribute rows. General SONATA permits multiple groups. Do not treat this example
layout as a restriction of the standard.

Example type files (composite node mechanism owns neuron and PSC dynamics):

```text
node_type_id model_type model_template
0 point_neuron btorch:GLIF3AlphaPSC
```

```text
edge_type_id model_template nsyns
0 btorch:AlphaPSC 1
1 btorch:AlphaPSC 1
```

## Reader differences

There are not two mandatory competing circuit formats. These tested APIs use
different entry points into the same bundle:

```python
from pathlib import Path
import libsonata
from sonata.circuit import File  # AllenInstitute pySONATA

root = Path("/tmp/glif-circuit")
circuit = libsonata.CircuitConfig.from_file(str(root / "circuit_config.json"))
nodes = circuit.node_population("glif3")
edges = circuit.edge_population("recurrent")
dynamics = nodes.get_dynamics_attribute("neuron_tau_0", nodes.select_all())
incoming = edges.afferent_edges([0])

# pySONATA File takes HDF5 + type CSV paths directly, not a circuit-config path.
py_nodes = File(str(root / "nodes.h5"), str(root / "node_types.csv")).nodes["glif3"]
py_edges = File(str(root / "edges.h5"), str(root / "edge_types.csv")).edges["recurrent"]
incoming_rows = list(py_edges.get_target(0))
outgoing_rows = list(py_edges.get_source(0))
```

Dynamics column names are adapter-defined: inspect `btorch.layouts` in the actual
exported config rather than assuming the illustrative `neuron_tau_0` spelling above.
libSONATA writes `node_id_to_ranges`; the tested AllenInstitute pySONATA reader
uses `node_id_to_range`. `write_edge_indices` creates a hard link for the latter;
do not duplicate index storage. pySONATA's node dynamics accessor is unimplemented
in the tested release: validate those values using libSONATA.

Tested: libsonata 0.2.2, h5py 3.16.0, AllenInstitute pySONATA distribution 0.0.1.
Install the official implementation, not an unrelated package sharing its name.
These are circuit configs, not simulator run configs: inputs, reports, runtime
memories and optimizer state are separate concerns.

References: [SONATA specification](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md),
[libSONATA API](https://libsonata.readthedocs.io/en/latest/api.html),
[pySONATA File source](https://github.com/AllenInstitute/sonata/blob/master/src/pysonata/sonata/circuit/file.py).

For constrained weights, store their reference signs separately from current
weights: a projected zero weight still needs its original sign to resume training.
