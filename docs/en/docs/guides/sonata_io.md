# Network-specific SONATA IO

The two [RSNN examples](../examples.md) share an explicit adapter in
`examples/rsnn_sonata.py`. It follows `SonataNetworkIO` structurally and reuses
`SonataPopulation`, `write_population`, `read_population`, and
`write_edge_indices` from `btorch.io.sonata`. There is no generic Module walker
or model registry: this adapter reconstructs exactly these two concrete circuits.

```python
from pathlib import Path
from examples.rsnn import build_model
from examples.rsnn_common import ExampleOptions, initialize
from examples.rsnn_sonata import RSNNSonataIO
from btorch.models import environ

network = build_model(ExampleOptions(neurons=32))
adapter = RSNNSonataIO()
circuit_config = adapter.export_network(network, Path("/tmp/glif-circuit"), dt=1.0)
loaded = adapter.import_network(circuit_config, device="cpu", backend="torch")
with environ.context(dt=loaded.dt):
    initialize(loaded.network, "cpu")
    spikes, records = loaded.network(x_seq)
```

## Mapping scope

- One population, local node IDs equal tensor positions; source=pre, target=post.
- GLIF3 + heterogeneous AlphaPSC: two signed receptor channels and discrete
  delay bins, including zero and the final bin. Dale reference signs are
  preserved separately from current weights, including weights projected to zero.
- Izhikevich + heterogeneous DualExponentialPSC: signed receptor channels and
  distinct rise/decay times, with no additional transmission delay.
- Node templates `btorch:GLIF3AlphaPSC` and
  `btorch:IzhikevichDualExponentialPSC` denote the **composite point mechanisms**.
  Their scalar dynamics columns encode both neuron and receptor-filter
  parameters. These are custom model namespaces, not simulator built-ins.
- Edge type IDs identify receptor channels. Parallel source/target pairs with
  distinct receptor/delay mechanisms remain distinct. Signed weights are pA;
  voltage mV, time ms, capacitance pF, and conductance nS. Additional coefficient
  units are declared in the config extension. Optional `dale_sign` edge columns
  retain the projection constraint across export/import.
- SONATA `delay = delay_bin * dt` in ms. Import rejects fractional-bin,
  negative, or out-of-range delays. This is additional transmission delay;
  RecurrentNN still consumes previous-step PSC.
- The versioned `btorch.mapping = "example-rsnn-v2"` extension records original
  scalar/shared/per-neuron/adaptation-mode/receptor tensor shapes, parameter
  ownership/trainability, surrogate cutoff, BPTT setting, reset and record policy.
  An edit inconsistent with shared scalar metadata is rejected.

HDF5 scalar columns retain heterogeneous values; config metadata retains their
Torch broadcast shapes and trainability. Cross-field parameter aliases, custom
models, batched connectivity, extra input modules and arbitrary receptor-table
metadata are rejected. Import neither initializes memories nor changes environ;
execution backends are caller-selected. Circuit IO does not save membrane,
adaptation, PSC, delay-ring or optimizer continuation state. Use normal btorch
checkpoints for those.

## Run and verify

```bash
pip install 'btorch[io]' libsonata
python -m examples.rsnn --export-sonata /tmp/glif-circuit
python -m examples.rsnn_izhikevich --export-sonata /tmp/izh-circuit
python -m pytest tests/io/test_sonata.py
```

The files use SONATA HDF5 magic/version, group ID/index mappings, population
endpoint attributes, whitespace type CSVs and portable circuit config paths.
Tests compare inference spikes, membrane/adaptation/PSC histories and complete
input/parameter gradients after round-trip. They cover isolated nodes,
self-loops, mechanism-distinct parallel edges, empty edges, heterogeneous
parameters and delay boundaries. Sparse gradient reductions admit normal FP32
rounding; state/activity comparisons remain strict.

Both libSONATA and AllenInstitute pySONATA independently read IDs, topology,
receptors, weights and indexed incoming/outgoing edges. libSONATA checks every
node dynamics component; pySONATA's node dynamics accessor is unimplemented.
Tested reader distributions: libsonata 0.2.2, h5py 3.16.0 and the AllenInstitute
pySONATA implementation (distribution version 0.0.1). Install that implementation
rather than relying on an unrelated package with the same name:

```bash
pip install 'git+https://github.com/AllenInstitute/sonata.git#subdirectory=src/pysonata'
```

Reader compatibility does not establish executable equivalence in another
simulator. See the [SONATA specification](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md)
and [libSONATA API](https://libsonata.readthedocs.io/en/latest/api.html).
For a custom circuit, follow [the SONATA IO skill](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/dev/skills/sonata-io/SKILL.md)
and provide its own explicit mapping.
