# Examples Gallery

The `examples/` directory contains runnable scripts that demonstrate core btorch patterns. This page annotates each example and links to the relevant tests and API docs.

## Two RSNN examples: training, inference and SONATA

| Script | Neuron/PSC | Connectivity |
|---|---|---|
| `rsnn.py` | GLIF3 + AlphaPSC | Two heterogeneous receptor channels; 0–3 delay bins |
| `rsnn_izhikevich.py` | Izhikevich + DualExponentialPSC | Two heterogeneous rise/decay channels; no extra delay |

Start from the repository root with either CPU example:

```bash
python -m examples.rsnn
python -m examples.rsnn_izhikevich
```

Defaults run inference and three training epochs on 256 neurons. Each script
shows its neuron, heterogeneous PSC, recorded states and `RecurrentNN` composition
in `build_model`. `rsnn_common.py` shares sparse connectivity, memory initialization
and the execution loop; `rsnn_sonata.py` contains the explicit circuit mapping.

Inputs and spikes have shape `[time, batch, neuron]`; examples use batch size 1.
Time is ms, voltage mV and input current pA. Receptor 0 starts excitatory and
receptor 1 inhibitory. GLIF uses Dale constraints: `constrain_net(network)`
after each optimizer update preserves each source neuron's E/I sign.
Izhikevich keeps unconstrained trainable weights. PSC lanes
use `post * 2 + receptor`, so each neuron receives the same pair of distinct
receptor time constants. Neuron parameters also vary across neurons.

Use `--mode inference` or `--mode train` independently. Training resets physical
memories before each independent trial and fits a synthetic 20 Hz target plus a
voltage penalty. This demonstrates gradients through neuron parameters and
recurrent weights; it is not a scientific fitting task. For continuous inference,
initialize once and retain memories between successive input chunks.

Native CUDA execution selects neuron/PSC and connection backends independently:

```bash
python -m examples.rsnn --device cuda --neuron-backend triton --sparse-backend triton --compile
python -m examples.rsnn_izhikevich --device cuda --neuron-backend triton --sparse-backend triton --method push --compile
```

`--method push` explicitly selects spike-list propagation; `pull` selects the
opposite traversal. Auto mode can use dense multiplication below 4,000 neurons
while preserving the sparse public interface. `--compile` supports training;
explicit `--cudagraph` requires `--device cuda --mode inference`.

To export and import either circuit, install the optional IO dependencies:

```bash
pip install 'btorch[io]' libsonata
python -m examples.rsnn --export-sonata /tmp/glif-circuit
python -m examples.rsnn --import-sonata /tmp/glif-circuit/circuit_config.json
python -m examples.rsnn_izhikevich --export-sonata /tmp/izh-circuit
python -m examples.rsnn_izhikevich --import-sonata /tmp/izh-circuit/circuit_config.json
```

Import restores circuit/gradient settings and its timestep; `--neurons`,
`--density`, `--seed` and `--dt` configure newly constructed circuits only.
Execution backends remain caller-selected. Export after training writes updated
parameters, not optimizer or runtime memories. See [the SONATA mapping](guides/sonata_io.md).

The larger Izhikevich configuration is `--neurons 50000 --density 0.05`.
Construction uses sparse edge arrays. Full state recording and training at that
size still consume substantial memory; the defaults are intended for learning.

## `rsnn_brain.py` — Brain-Environment Interaction

Extends the basic RSNN with brain-environment interaction networks. Demonstrates `NeuronEmbedMapLayer` and `DetectionWindow` for sensory-motor tasks.

**Key patterns:**
- Embedding maps between neuron spaces
- Detection windows for spike-based event detection
- Multi-module composition beyond `RecurrentNN`

## `fmnist.py` — Fashion-MNIST Training (Plain PyTorch)

A full training pipeline for Fashion-MNIST classification using a GLIF3-based RSNN with sparse recurrent connectivity and voltage regularization.

**Key patterns:**
- `SparseConnection` for sparse recurrent weights
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
- `scale_state_` for scaling neuron-parameter dicts
- Validation loop with state recording

## `delayed_synapse_demo.py` — Synaptic Delays

Demonstrates synaptic delays using `Delay`, `DelayedPSC`, and `HeterSynapsePSC`.

**Key patterns:**
- `Delay` as a rolling spike buffer
- `DelayedPSC` for adding scalar delays to any PSC model
- `HeterSynapsePSC(..., max_delay_steps=...)` for delay + receptor combinations
- `expand_conn_for_delays` for delay-aware connection matrices

**See also:**
- API: [`btorch.models.history`](api/models.md) and [`btorch.models.synapse`](api/models.md)
- Tests: [`tests/connectome/test_delay_expansion.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/tests/connectome/test_delay_expansion.py)

## Tests as Examples

Many `tests/` files contain concise, validated usage patterns that are ideal for documentation:

- `tests/models/test_mem_load_save.py` — Checkpointing and state restoration
- `tests/models/test_compile.py` — `torch.compile` with `dt` context
- `tests/utils/test_conf.py` — OmegaConf patterns
- `tests/visualisation/*.py` — Nearly every plotting function

## Heterogeneous E/I Workflow

The [heterogeneous multi-receptor E/I tutorial](tutorials/heterogeneous_ei_rsnn.md)
builds a complete recurrent network with semantic receptor-labelled edges,
receptor-specific PSC kinetics, Dale's law, training, and channel inspection.
