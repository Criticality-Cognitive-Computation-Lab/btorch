# Examples Gallery

The `examples/` directory contains production-quality scripts that demonstrate core btorch patterns. This page annotates each example and links to the relevant tests and API docs.

## `rsnn.py` — Minimal RSNN

A self-contained demonstration of a recurrent spiking neural network using `GLIF3`, `AlphaPSC`, and `RecurrentNN`. It includes both a simulation loop (with raster-plot generation) and a dummy training loop.

**Key patterns:**
- `functional.init_net_state` and `functional.reset_net`
- `environ.context(dt=1.0)`
- `update_state_names` for state recording
- `plot_raster` and `plot_neuron_traces` from `btorch.visualisation`

**See also:**
- [Tutorial 1: Building an RSNN](tutorials/building_rsnn.md)
- [Tutorial 2: Training an SNN](tutorials/training.md)

## `rsnn_brain.py` — Brain-Environment Interaction

Extends the basic RSNN with brain-environment interaction networks. Demonstrates `NeuronEmbedMapLayer` and `DetectionWindow` for sensory-motor tasks.

**Key patterns:**
- Embedding maps between neuron spaces
- Detection windows for spike-based event detection
- Multi-module composition beyond `RecurrentNN`

## `fmnist.py` — Fashion-MNIST Training (Plain PyTorch)

A full training pipeline for Fashion-MNIST classification using a GLIF3-based RSNN with sparse recurrent connectivity and voltage regularization.

**Key patterns:**
- `SparseConn` for sparse recurrent weights
- `AlphaPSCBilleh` synapse with heterogeneous time constants
- `VoltageRegularizer` for membrane-voltage regularization
- Manual training loop with `reset_net` per batch
- Checkpointing `memories_rv` alongside `state_dict()`

**See also:**
- [Tutorial 2: Training an SNN](tutorials/training.md)
- API: [`btorch.models.regularizer`](api/models.md)

## `fmnist_lightning.py` — PyTorch Lightning Integration

The same Fashion-MNIST model factored into a PyTorch Lightning `LightningModule`. Shows how btorch state management fits into the Lightning training lifecycle.

**Key patterns:**
- Lightning `training_step` with `reset_net`
- `scale_net` / `unscale_net` for neuron-parameter scaling
- Validation loop with state recording

## `delayed_synapse_demo.py` — Synaptic Delays

Demonstrates heterogeneous synaptic delays using `SpikeHistory` and `DelayedSynapse`.

**Key patterns:**
- `SpikeHistory` as a rolling spike buffer
- `DelayedSynapse` for history + linear transformation
- `expand_conn_for_delays` for delay-aware connection matrices

**See also:**
- API: [`btorch.models.history`](api/models.md)
- Tests: [`tests/connectome/test_delay_expansion.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/tests/connectome/test_delay_expansion.py)

## Tests as Examples

Many `tests/` files contain concise, validated usage patterns that are ideal for documentation:

- `tests/models/test_mem_load_save.py` — Checkpointing and state restoration
- `tests/models/test_compile.py` — `torch.compile` with `dt` context
- `tests/utils/test_conf.py` — OmegaConf patterns
- `tests/visualisation/*.py` — Nearly every plotting function

**Tip:** When adapting test code for documentation, replace `fig_path` / `save_fig` calls with standard matplotlib patterns (e.g., `plt.show()` or returning the figure).
