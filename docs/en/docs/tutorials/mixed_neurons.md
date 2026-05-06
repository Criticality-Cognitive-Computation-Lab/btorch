# Mixed Neuron Populations and Apical RNNs

This guide shows how to build recurrent networks that contain **multiple neuron types** in the same layer, and how to feed **additional input streams** (e.g. apical / top-down drive) into a subset of those neurons.

## What You Will Learn

- How to combine `GLIF3`, `TwoCompartmentGLIF`, and other neurons in one layer with `MixedNeuronPopulation`.
- How to wrap the population in `ApicalRecurrentNN` so that apical inputs are correctly time-unrolled.
- How state collection works when sub-populations have different internal variables.

## When to Use This

Use these tools when your model needs:

1. **Heterogeneous cell types** — e.g. 80 \% fast-spiking interneurons (`GLIF3`) and 20 \% pyramidal cells (`TwoCompartmentGLIF`).
2. **Multi-compartment neurons** — e.g. somatic feed-forward input + apical top-down input.
3. **Structured state read-out** — each sub-population exposes its own voltages / currents via dotted state names.

## Building a Mixed-Population RSNN

### 1. Create the Sub-Populations

Each sub-population is an ordinary btorch neuron module.  They can have different parameter sets, different state variables, and even different `step_mode` (as long as the wrapper is called in the mode you intend).

```python
from btorch.models.neurons import GLIF3, TwoCompartmentGLIF
from btorch.models.neurons.mixed import MixedNeuronPopulation
from btorch.models.linear import DenseConn
from btorch.models.synapse import AlphaPSC
from btorch.models.rnn import ApicalRecurrentNN

n_neuron = 100
n_glif   = 80
n_tc     = 20

glif = GLIF3(n_neuron=n_glif, step_mode="s")
tc   = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")

mixed = MixedNeuronPopulation(
    [(n_glif, glif), (n_tc, tc)],
    step_mode="s",
)
```

`MixedNeuronPopulation` slices the last dimension of the input current and dispatches the correct slice to each child.  Spikes are concatenated back together, so the output shape is always `(*batch, n_neuron)`.

### 2. Add Recurrent Synapses

The synapse sees the **concatenated** spikes, so the connection matrix must be `n_neuron x n_neuron`:

```python
conn = DenseConn(n_neuron, n_neuron, bias=None)
psc  = AlphaPSC(
    n_neuron=n_neuron,
    tau_syn=5.0,
    linear=conn,
    step_mode="s",
)
```

### 3. Wrap in `ApicalRecurrentNN`

`ApicalRecurrentNN` is a subclass of `RecurrentNN` that accepts a third positional argument `x_apical`.  When you call it with a time sequence, the outer unroll loop slices `x_apical` automatically **as long as you pass it positionally**:

```python
brain = ApicalRecurrentNN(
    neuron=mixed,
    synapse=psc,
    step_mode="m",          # multi-step wrapper
    unroll=4,
    update_state_names=(
        "neuron.group_0.v",
        "neuron.group_1.v",
        "neuron.group_1.i_a",
        "synapse.psc",
    ),
)
```

### 4. Initialise and Run

```python
from btorch.models import functional, environ

functional.init_net_state(brain, batch_size=4)

T = 100
x_soma   = torch.randn(T, 4, n_neuron)
x_apical = torch.randn(T, 4, n_neuron)   # only the TC slice is actually used

with environ.context(dt=1.0):
    spikes, states = brain(x_soma, None, x_apical)

print(spikes.shape)                       # (T, 4, 100)
print(states["neuron.group_1.i_a"].shape) # (T, 4, 20)
```

## How `MixedNeuronPopulation` Handles the Apical Slice

`TwoCompartmentGLIF` expects two arguments: somatic current and apical current.  `MixedNeuronPopulation` knows this and automatically slices the apical tensor for every `TwoCompartmentGLIF` child.  For `GLIF3` (and any other single-input neuron) the apical slice is simply ignored.

If you do **not** need apical input, omit the third argument and the population behaves like a standard single-input neuron layer:

```python
spikes, states = brain(x_soma)   # x_apical is None for every group
```

## State Naming Convention

Because sub-populations are registered as named children (`group_0`, `group_1`, …), their states appear with dotted prefixes:

| State key | Meaning | Shape |
|-----------|---------|-------|
| `neuron.group_0.v` | GLIF3 membrane voltage | `(T, batch, 80)` |
| `neuron.group_1.v` | TC somatic voltage | `(T, batch, 20)` |
| `neuron.group_1.i_a` | TC apical current | `(T, batch, 20)` |
| `synapse.psc` | Post-synaptic current | `(T, batch, 100)` |

You can unflatten the dictionary for easier access:

```python
from btorch.utils.dict_utils import unflatten_dict
nested = unflatten_dict(states, dot=True)
nested["neuron"]["group_1"]["i_a"]   # (T, batch, 20)
```

## Named Groups

Instead of auto-naming, you can give groups explicit names:

```python
mixed = MixedNeuronPopulation({
    "fs": (80, GLIF3(n_neuron=80)),
    "pyr": (20, TwoCompartmentGLIF(n_neuron=20)),
}, step_mode="s")
```

State keys then become `neuron.fs.v`, `neuron.pyr.i_a`, etc.

## Compatibility Notes

- `torch.compile` — `MixedNeuronPopulation` uses a Python loop over children, so compilation may graph-break on the loop.  If this is a bottleneck, consider fusing the sub-populations into a single custom module.
- Gradient checkpointing — works transparently through `RecurrentNNAbstract` because the checkpointed region is the outer `multi_step_forward`, not the individual neuron children.
- CPU offloading — also works transparently; chunk outputs are offloaded after the full forward pass through all groups.

## See Also

- [`RecurrentNN`][btorch.models.rnn.RecurrentNN] — standard single-input recurrent wrapper.
- [`ApicalRecurrentNN`][btorch.models.rnn.ApicalRecurrentNN] — apical-input variant.
- [`MixedNeuronPopulation`][btorch.models.neurons.mixed.MixedNeuronPopulation] — heterogeneous population container.
- [`TwoCompartmentGLIF`][btorch.models.neurons.two_compartment.TwoCompartmentGLIF] — soma-apical neuron.
