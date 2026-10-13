# Network-specific SONATA IO

A network supplies its own SONATA adapter. btorch does not infer a network definition by walking arbitrary Modules. User agents implement traversal, model construction, ID/edge mapping and units; shared `btorch.io.sonata` helpers handle single-group HDF5 populations and native edge-index generation.

The common `SonataNetworkIO` calling interface is structural: no inheritance or registration is required. `export_network(network, directory, dt=...)` returns a circuit config path; `import_network(config, ...)` returns `SonataImport(network, dt)`. Import neither sets environ nor initializes state.

```python
from examples.sonata_izhikevich import IzhikevichSonataIO, make_example
from btorch.models import environ, functional

adapter = IzhikevichSonataIO()
config = adapter.export_network(make_example(), directory, dt=0.25)
loaded = adapter.import_network(config, device="cpu", backend="torch")
with environ.context(dt=loaded.dt):
    functional.init_net_state(loaded.network, batch_size=1)
    spikes, records = loaded.network(x_seq)  # [time, batch, neuron]
```

Run the complete export/import/activity comparison:

```bash
pip install 'btorch[io]' libsonata
python examples/sonata_izhikevich.py --directory ./sonata-izh
```

The example maps one Izhikevich population, AlphaPSC, plain sparse per-edge weights and scalar/per-neuron parameters. It preserves reset/surrogate settings, trainability and requested records in a versioned `btorch` config extension. Nodes use population-local IDs; edge source is pre, target is post. Parameter units are explicit. Edge delay is zero additional transmission delay; the existing RecurrentNN schedule still consumes previous-step PSC. Other models, receptors, delays, parameter sharing and custom mechanisms need their own mapping. Unsupported configurations raise rather than silently lose information.

The files follow the [SONATA specification](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md): HDF5 populations/dynamics, whitespace type tables and JSON circuit config. Native indices come from [libSONATA](https://libsonata.readthedocs.io/en/latest/api.html); hard-link aliases accommodate pySONATA's index spelling. Tests read topology, attributes and edge lookups with both libraries, and compare original/restored activity and gradients. Install official [pySONATA](https://github.com/AllenInstitute/sonata/tree/master/src/pysonata) for that optional check:

```bash
pip install 'git+https://github.com/AllenInstitute/sonata.git#subdirectory=src/pysonata'
PYTHONPATH=. python -m pytest tests/io/test_sonata.py
```

pySONATA's node dynamics accessor is unimplemented; those values are checked with libSONATA. Reader compatibility does not make `btorch:Izhikevich` an executable model in every simulator. Circuit files hold network definitions; membrane/PSC/delay state and optimizer continuation use separate btorch checkpoints.

For agent-generated custom networks, invoke [sonata-io](../../../../skills/sonata-io/SKILL.md) when SONATA is requested or network-definition serialization has no specified IO format/backend. Keep the same calls and shared helpers; the network's agent supplies the specific implementation. Upstream shared helpers/interface and reusable checks only; generated network-specific implementations stay in the user's project/OOT package.
