# Write a custom LIF kernel

Run the example on a CUDA device with Triton installed:

```bash
python -m examples.oot_cell_backend
```

The example shows one task: replace LIF's forward implementation while keeping
its normal btorch Module interface and existing surrogate-gradient backward.
It contains three parts, in call order:

1. `lif_kernel` is the handwritten Triton kernel. One lane updates one neuron
   across the sequence, producing spikes, final voltage and requested records.
2. `forward` adapts btorch's tensor callback to that kernel. State order is
   `(v,)`; parameter order is `(v_threshold, v_reset, c_m, tau)`. Broadcast
   parameters and strided drives remain views. Returned tensors have separate
   storage from the inputs.
3. `register_example_backend` registers this forward as `tutorial_lif` and
   reuses btorch's Triton LIF backward. `LIF(..., backend="tutorial_lif")`
   then selects it without changing state initialization or training code.

```python
from btorch.models.neurons import LIF
from examples.oot_cell_backend import register_example_backend

register_example_backend()
neuron = LIF(129, tau=4.0, step_mode="m", backend="tutorial_lif", device="cuda")
```

Reusing that backward is valid because the custom forward implements the same
LIF equations and reset convention. Changing the equations requires a matching
backward. The example explicitly excludes refractory and stochastic variants;
it accepts `(time, batch, neuron)` input.

`run_example` first checks spikes, voltage records and gradients against the existing
Torch model with `check_backend_parity`, including compilation and CUDA Graph
replay. It then differentiates a spike/voltage
loss through a trainable input scale using the ordinary LIF Module. No RSNN,
connection layout or source-generation workflow is mixed into this example.

For shared equation templates rather than a handwritten forward, see
[kernel generation](../design/kernel_generation.md). For callback ordering,
capabilities, compile and graph checks, see the
[extension contract](../guides/oot_kernels.md). Online learning has a separate
example, `examples/oot_stdp.py`, described in the
[plasticity guide](../guides/plasticity.md#out-of-tree-native-learning-templates).
