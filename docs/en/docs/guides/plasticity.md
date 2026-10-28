# Online synaptic plasticity

`Synapse.plasticity` attaches a local learning rule to a `SparseConnection`.
The connection retains canonical weight ownership; plasticity owns trace
memories. Neither the rule nor its equations select a neuron model or sparse
backend.

```python
import torch
from btorch.models.connection import EdgeWeight, PairSTDP, SparseConnection, Synapse

connection = SparseConnection.from_edges(
    torch.tensor([0, 1]), torch.tensor([1, 0]), 2, 2,
    Synapse(
        weight=EdgeWeight(torch.tensor([0.3, 0.4]), trainable=False),
        delay=torch.tensor([0, 2]),
        plasticity=PairSTDP(
            a_plus=0.01, a_minus=0.012,
            tau_plus=20.0, tau_minus=20.0, w_min=0.0, w_max=1.0,
        ),
    ),
)
```

Amplitudes, time constants and bounds may be scalars or input-edge arrays.
They follow the same canonical sorting as weights, receptors and integer
delays. Duplicate edges with identical routing must be coalesced before adding
a rule; merging independent plastic synapses changes their learning semantics.

For edge `e = (i, j, receptor, delay)`, let `a_e(t) = z_i(t-delay)` be its
actual arrival and `z_j(t)` its postsynaptic spike. Each step uses this order:

1. Update the neuron using the preceding PSC, then propagate actual arrivals
   with the current weights, following `RecurrentNN`'s existing schedule.
2. Decay old pre/post traces by `exp(-dt/tau_plus)` and `exp(-dt/tau_minus)`.
3. Compute `delta = -a_minus * a_e * post_trace_old + a_plus * z_j * pre_trace_old`.
4. Add the combined delta and clip once to the weight bounds.
5. Add current arrivals/postsynaptic spikes to their respective traces.

Simultaneous spikes do not pair with one another. They can both pair with
strictly earlier events. The resulting weight is visible to the next
propagation. Receptor channels share the target spike but may have different
trace parameters. Edge-local traces preserve heterogeneous time constants.

For standalone connections, the caller supplies actual delay-expanded arrivals
and calls `connection.update_plasticity(arrivals, post_spikes, dt=dt)` **after**
`connection(arrivals)`. Arrival layout is `[..., source * n_delay + delay]`.
The method does not advance a second delay buffer. In a supported `RecurrentNN`,
its PSC owns delay buffering and the recurrent scheduler invokes learning.

Online plasticity mutates weights under `torch.no_grad()` or
`torch.inference_mode()`. It is separate from BPTT and optimizer learning;
grad-enabled execution and gradient checkpointing raise before stepping.
The eager reference and native online path support one sample and unconstrained
`EdgeWeight`.
Shared-weight minibatch reductions, structural rewiring, nested `DelayedPSC`,
apical/conductance recurrence and CuPy plasticity are not supported.
Unsupported combinations raise
explicit errors rather than silently ignoring learning.

Trace memories are persistent buffers and follow normal module device/dtype,
reset and detach operations. Reset clears traces, **not learned weights**.
For a full running-network checkpoint, initialize neuron/PSC/delay memories
with `functional.init_net_state(..., persistent=True)` before simulation;
then save/load the complete `state_dict`. Continue chunks without resetting.
`compute_input_scale` also restores learned weights between calibration trials
and on exit, so calibration does not consume the network's learning state.

## Optimized recurrent execution

Select the neuron/PSC renderer independently of sparse propagation:

```python
from btorch.models import KernelOptions

network.backend = KernelOptions(neuron="tilelang", connection="triton")
# Or neuron="triton". Both renderers execute the same STDP scalar statements.
with torch.no_grad():
    spikes, records = network(inputs)
```

The native online trajectory reuses the existing sparse route and paired
PSC/neuron updates, adding one independent canonical-edge STDP kernel per
step. That kernel combines trace decay, LTD, LTP, clipping and trace increments.
It is a baseline, **not a HELION single-kernel timestep**. TileLang renders
neuron, PSC and plasticity here; the existing sparse propagation may use
Triton. Static connections allocate no plasticity traces or update kernels.

Weights stay in their canonical module. A route with another CSR edge order
keeps a refreshed execution view; static packed weight snapshots are bypassed
for online learning. Explicit trace and weight histories are optional through
`update_state_names`, for example `synapse.linear.plasticity.pre_trace`,
`synapse.linear.plasticity.post_trace` and `synapse.linear.weight.value`.
As with other recurrent memories, `update_state_names=None` records traces at
every timestep. Select the neuron/PSC histories you need, or use
`update_state_names=()` for simulation without histories; final traces remain
available on the plasticity module.

`torch.compile` supports the online inference operator through preallocated
output tensors with declared mutation. `cudagraph=True` uses the existing
split chunk runner (`chunk_size=200` by default); full chunks share a capture
and the remainder uses a shape-specific capture. Warmup/capture restores both
trace state and canonical weights before real execution. Online STDP still
requires `no_grad` and cannot be combined with gradient checkpointing or BPTT.
Without STDP, existing gradient/VJP and checkpoint paths remain available.
