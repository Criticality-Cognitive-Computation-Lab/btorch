# Cell and connection backends

Select neuron/PSC dynamics independently from sparse propagation:

```python
from btorch.models import KernelOptions
from btorch.models.rnn import RecurrentNN

rnn = RecurrentNN(neuron, synapse, backend=KernelOptions(
    cell="triton", connection="auto", merge_synapse=False,
))
```

A backend string selects only the cell implementation. `torch`, `triton`,
`tilelang` and `cupy` are built in; externally registered names work through the
same argument. GPU packages are optional. Install Triton with `pip install
'btorch[gpu]'`; install TileLang or CuPy separately for the CUDA environment you
use. Cell backends reuse the existing sparse runtime; selecting a cell backend
does not migrate or duplicate sparse algorithms.

Registered cell sequence operators require PyTorch 2.4 or newer.
Training supports first-order BPTT. `create_graph=True` is explicitly rejected;
these handwritten kernel VJPs do not provide higher-order derivatives.

The initial model set is IF, LIF, GLIF3, Izhikevich and ExponentialPSC, AlphaPSC,
AlphaPSCBilleh, DualExponentialPSC. The six deterministic built-in surrogate
functions retain btorch's normalized derivatives. Custom surrogates and trainable
surrogate hyperparameters raise instead of silently changing semantics.

## Gradients and precision

At sequence entry, btorch checks grad mode and all participating floating inputs,
initial states, connection weights and model parameters. Training runs retain the
history needed for backward; inference retains final state and requested records.
`.train()` and `.eval()` do not select this path. A frozen RNN still propagates a
trainable encoder's gradients. `no_grad()` and `inference_mode()` select inference.

Current built-in kernels accept FP32, FP16 or BF16 inputs and use FP32 state and
accumulation; output sequences follow input dtype. This is an implementation
limitation, not a settled precision/promotion policy. Time is first, normally
`[T,B,N]`; initial state has
no time axis. Model adaptation and receptor axes remain explicit. Integer topology,
receptors and delays receive no gradients.

Each call reads scalar `environ.dt`. Changing it does not specialize generated
kernel source or recompile the sequence graph; backward uses that call's dt.
CUDA graph replay keeps the captured value unless its device input is updated.

## State and records

Initialize modules with `init_state` or `functional.init_net_state`. Final state
remains differentiable across calls until you explicitly reset or detach it.
Returned RSNN records can contribute an auxiliary loss. Each invocation owns the
saved data it needs, so two forwards followed by one backward are supported.

Explicit kernel options select a stateful PSC sequence recurrence. The historical
plain `backend="torch"` PSC multi-step convolution starts from rest, truncates its
impulse response at `kernel_len`, and does not update state; use
`backend=KernelOptions(cell="torch")` for the stateful reference path.

`merge_synapse=False` retains individual receptor states for EPSC/IPSC queries.
Explicit merge is restricted to compatible linear dynamics with frozen compatible
parameters or actual shared parameter identity. Independent trainable parameters
cannot be merged because their current values happen to match. Independent
receptor records cannot be recovered from merged state; incompatible requests
raise. Consult the tested support scope before enabling merge for a new model.

The initial sequence integration supports identity, dense `nn.Linear` and
`SparseConnection` connections. Custom connection modules and sparse
`member_pattern_batch` are not supported by this path. Receptor merging is a
heterogeneous RSNN option; standalone neuron/PSC calls reject it. External cell
backends compose with the same connections through their registered callbacks;
they do not automatically acquire the built-in paired cell fusion.

See [the external-kernel guide](oot_kernels.md),
[the training tutorial](../tutorials/kernel_training.md) and
[the sparse guide](sparse_connectivity.md).
