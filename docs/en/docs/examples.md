# Examples Gallery

The `examples/` directory contains runnable scripts that demonstrate core btorch patterns. This page annotates each example and links to the relevant tests and API docs.

## Two RSNN examples: training, inference and SONATA

| Script | Neuron/PSC | Default execution |
|---|---|---|
| `rsnn.py` | GLIF3 + heterogeneous AlphaPSC, delays, Dale constraints | 256 neurons, CPU, inference and training, 7.5 Hz target |
| `rsnn_izhikevich.py` | Izhikevich + heterogeneous DualExponentialPSC | 60,000 neurons, 33,072,348 edges, CUDA/Triton, CUDA Graph inference, 5 Hz target |

Run the Izhikevich example from the repository root, with a CUDA GPU available:

```bash
python -m examples.rsnn_izhikevich
```

No extra parameters are needed. The script constructs the exact edge count using
btorch's `FixedIndegree` sampler, calibrates sustained heterogeneous current,
and runs 5000 steps at 1 ms. It prints matched **push eager**, **pull CUDA Graph**,
and **push CUDA Graph** timings, with the same recordings enabled in every case.
Compilation, capture, calibration, saving and analysis are outside the warm timings;
public replay's input/state copies and output ownership copies are included.
One RTX 5090 run measured 5.035 Hz, 338 ms push eager, 1571 ms pull Graph,
and 74 ms push Graph: 4.6x over push eager and 21.3x over pull Graph.
These measurements depend on hardware and circuit activity, and are not a
promise for every GPU or input. This is the example's synthetic circuit.

The default archive is `fig/rsnn_izhikevich/recordings.pt` (about 6.7 GiB for
this configuration), with adjacent JSON run metadata and timings. Each tensor
has shape `[time, batch, neuron]`:

| Key | Meaning | Units |
|---|---|---|
| `v` | Post-reset membrane voltage | mV |
| `u` | Izhikevich recovery state | pA |
| `spike` | Emitted binary spike raster | dimensionless |
| `psc` | Total signed PSC after this step's synaptic update | pA |
| `epsc` | Signed receptor-0 excitatory current | pA |
| `ipsc` | Signed receptor-1 inhibitory current | pA |

`psc = epsc + ipsc`; IPSC is negative. The current recorded at step `t` drives
neuron step `t+1`, following `RecurrentNN`'s previous-PSC schedule. EPSC/IPSC
are views of one transferred receptor history, not extra per-step kernels.
All six recordings remain enabled during the comparisons. Full histories and
owned graph outputs require substantial memory; the default was verified on a
32 GiB RTX 5090. Choose a smaller circuit explicitly on a smaller device.

```python
import torch

traces = torch.load("fig/rsnn_izhikevich/recordings.pt", weights_only=True, mmap=True)
v = traces["v"][:, 0]  # [time, neuron]; loaded on CPU
```

Use `--no-compare` to skip comparisons, `--no-cudagraph` to select eager final
inference, and `--recordings PATH` to choose the archive location. Comparisons
explicitly exercise all three execution policies independently of the final
inference choice. **FP32 push is not bitwise reproducible**, including under
CUDA Graph replay; similar population rates do not imply identical spike,
state or gradient trajectories. See [the quantified 1000-step errors](guides/sparse_connectivity.md#measured-recurrent-reproducibility-of-fp32-push).

For smaller training examples:

```bash
python -m examples.rsnn
python -m examples.rsnn_izhikevich --neurons 256 --density .05 --steps 256 --mode train --compile
```

`--mode both` runs inference and training. Inference capture is disabled for
training; `--compile` uses Torch's training graph machinery. Each epoch starts
from fresh physical memories and fits `--target-rate-hz` plus a voltage penalty.
This demonstrates gradients through neuron parameters and recurrent weights;
it is not a scientific fitting task. Changing neuron count or density changes
the generated circuit; `--edges E` explicitly sets an exact total instead.
GLIF preserves the sign of each source's weights with `constrain_net(network)`
after each optimizer update; Izhikevich weights remain unconstrained.

Both factories show the actual neuron, PSC, records and `RecurrentNN`
composition. `rsnn_common.py` shares CLI/execution plumbing; model-independent
calibration is `btorch.analysis.compute_input_scale`, which restores initialized
memories, recording policy and graph settings without changing parameters.
Calibration measures the population/time mean on at most 512 steps; full
inference separately checks the measured rate. This does not guarantee each
neuron's rate, stationary activity, or the rate after optimizer updates.

Neuron/PSC and connection backends are independent. `--method push` explicitly
requests spike-list SpMSpV; `pull` selects destination-driven execution. Auto
can use dense multiplication below 4000 expanded operator dimensions while
retaining the sparse public interface. For continuous inference, initialize
once and retain physical memories across chunks. `clear_cudagraph()` releases
cached captures without resetting those memories; changing recording policy
invalidates existing native captures automatically.

Both examples support explicit SONATA circuit import/export:

```bash
pip install 'btorch[io]' libsonata
python -m examples.rsnn --export-sonata /tmp/glif-circuit
python -m examples.rsnn --import-sonata /tmp/glif-circuit/circuit_config.json
python -m examples.rsnn_izhikevich --neurons 256 --density .05 --steps 256 --no-compare --export-sonata /tmp/izh-circuit
python -m examples.rsnn_izhikevich --steps 256 --no-compare --import-sonata /tmp/izh-circuit/circuit_config.json
```

Import restores circuit/gradient settings and timestep. Size, edge count,
density, seed and `--dt` configure new circuits only; execution backends remain
caller-selected. Calibration changes the input scale, not imported weights.
Export writes updated parameters, not optimizer state or runtime memories.
See [the SONATA mapping](guides/sonata_io.md).

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
