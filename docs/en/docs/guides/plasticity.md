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

`trace_scope="connection"` owns independent traces per canonical synaptic edge.
This controls learning-state ownership and is independent of `merge_synapse`,
which combines compatible linearly additive PSC states.

With scalar time constants, `PairSTDP(..., trace_scope="neuron")` instead
owns one pre trace per nonempty `(source, delay)` segment and one post trace
per neuron. Amplitudes and bounds may remain edge-specific. Shared traces
update once per event stream, including isolated postsynaptic neurons.
Restore checkpoints into the same trace layout. Eager and separate native
execution support this layout. The separate baseline requires
`recurrent="standard", spike_selection="auto"`. HELION also supports shared
traces: edges read old segment/post traces before the propagation barrier;
domain owners then advance traces once inside the same timestep kernel.
Zero-delay fusion reuses source-to-segment pointers; delayed incoming traversal
uses the prepared canonical-edge inverse. The standard two-kernel event
schedule also supports shared traces without delays: its edge readers consume
current traces while domain owners write separate next buffers.

For weight dependence, use
`PowerPairSTDP(PowerPairSTDPOptions(mu_plus=1.0, mu_minus=1.0), ...)`.
Its pair traces and event order are unchanged. For normalized
`q = clip((weight - w_min) / (w_max - w_min), 0, 1)`, multiply LTP by
`(1-q)**mu_plus` and LTD by `q**mu_minus`. Zero exponents recover additive
pairing; unit exponents give multiplicative dependence. Amplitudes remain in
absolute weight units. Equal bounds fix the weight while traces continue.
Bounds follow numerical order even on negative intervals: LTP moves toward
`w_max`, reducing inhibitory magnitude there. Receptors do not change this sign.
Power pairing has eager, separate native and HELION implementations in
Triton and TileLang. Its normalized weight law reuses the same trace bindings,
edge ownership and lazy materialization as additive pairing.

For nearest-neighbour pairing, use
`NearestPairSTDP(..., pairing="symmetric")`. Each event pairs with the nearest
strictly earlier opposite event; current events replace their own traces.
An earlier spike may participate in several later pairs.
`pairing="restricted"` additionally consumes opposite eligibility after each
event, allowing each spike to contribute at most once to LTP and once to LTD.
Coincident events first use old snapshots, then establish fresh eligibility
on both sides. Nonzero amplitudes replace traces; within-step spike
multiplicities have no distinct timestamps. The pairing policy is saved with
the memories and restored by `state_dict`. The eager reference and separate
Triton/TileLang updates share the pair weight law and explicit trace policy.
HELION uses the same policy with disjoint edge writers and lazy silent decay.

For standalone connections, the caller supplies actual delay-expanded arrivals
and calls `connection.update_plasticity(arrivals, post_spikes, dt=dt)` **after**
`connection(arrivals)`. Arrival layout is `[..., source * n_delay + delay]`.
The method does not advance a second delay buffer. In a supported `RecurrentNN`,
its PSC owns delay buffering and the recurrent scheduler invokes learning.

Online plasticity mutates weights under `torch.no_grad()` or
`torch.inference_mode()`. It is separate from BPTT and optimizer learning;
grad-enabled execution and gradient checkpointing raise before stepping.
The eager reference and native online path support one sample and unconstrained
`EdgeWeight` with independently writable edge slots. Nonoverlapping strided
views are supported; broadcast weights that alias multiple edges are rejected.
Shared-weight minibatch reductions, structural rewiring, nested `DelayedPSC`,
apical recurrence and CuPy plasticity are not supported.
Unsupported combinations raise
explicit errors rather than silently ignoring learning.

Trace memories are persistent buffers and follow normal module device/dtype,
reset and detach operations. Reset clears traces, **not learned weights**.
For a full running-network checkpoint, initialize neuron/PSC/delay memories
with `functional.init_net_state(..., persistent=True)` before simulation;
then save/load the complete `state_dict`. Continue chunks without resetting.
`compute_input_scale` also restores learned weights between calibration trials
and on exit, so calibration does not consume the network's learning state.

## Modulated eligibility

`ModulatedSTDP` stores pair eligibility separately from learned weights. Supply
the teaching/reward signal explicitly through the existing signal interface:

```python
from btorch.models.connection import ExternalSignal, ModulatedSTDP, ModulatedSTDPOptions

rule = ModulatedSTDP(
    ModulatedSTDPOptions(tau_eligibility=1000.0, eta=0.001),
    modulator=ExternalSignal("scalar", time_first=True),
    trace_scope="neuron",
)
# Attach rule through Synapse(plasticity=rule), then run the network:
with torch.no_grad():
    spikes, records = network(inputs, plasticity_signals={"modulator": rewards})
```

Here `rewards` has shape `[time]`. Constant scalar signals use
`ExternalSignal("scalar")` and a scalar tensor. Edge, post and segment signals
use their declared population dimensions; edge signals use canonical edge
order. All coefficients can be scalar or
input-edge arrays; shared pair traces require scalar pair decay times.
Eligibility always has one slot per canonical edge.

Within each bin, the caller's signal `m` is held constant. Old eligibility `c`
contributes `eta * (m-baseline) * c * tau_eligibility *
(1-exp(-dt/tau_eligibility))` to the weight. Eligibility then decays and receives
the strict-past pair impulse from `PairSTDP` at the bin's end. A reward in the
pairing bin therefore does not act on that new eligibility; a later reward can
change weights on silent steps. Weight bounds apply once after old-weight
propagation. Reset clears eligibility and pair traces, preserving learned
weights; checkpoints retain all three memories. Restore with the same trace
layout, modulator domain and time-axis declaration. Coefficients must remain
finite, and decay times positive, in the selected weight dtype.

This is an explicit binned three-factor rule. Unlike
[NEST's dopamine STDP](https://nest-simulator.readthedocs.io/en/stable/models/stdp_dopamine_synapse.html),
it accepts the modulator directly and does not filter a volume-transmitter
spike pool. No numerical equivalence to that model is implied. The eager
baseline and separate Triton/TileLang execution share the pair equations.
Native execution adds one learning kernel per step, including silent steps;
propagation and neuron/PSC updates keep their existing schedule. Shared pair
traces use compact old/next snapshots while eligibility stays in place.
Signal domains are prepared from their declarations, independently of backend.
Native pointwise learning retains half/bfloat16 arithmetic. Division and math
intrinsics without a low-precision backend interface promote only that
operation, then round its result back immediately; states are not cast wholesale
to FP32. Mixed tensor inputs follow dtype promotion; an FP32 teaching signal
is not downcast to the weight or trace dtype.
Both backends also support `recurrent="helion"` with
`spike_selection="scan"`: one cooperative kernel updates neurons, propagates
old weights and advances eligibility. Shared traces commit after all edge
readers finish. Eligibility can change weights in silent bins, so this rule
still visits every edge; explicit event-only learning discovery raises.
Sparse push propagation remains available in the separate baseline with
`spike_selection="auto"`. Online weight updates require `torch.no_grad()`;
CUDA Graph replay and compiled chunk continuation are supported.

## Inhibitory homeostasis

`InhibitorySTDP(eta=0.001, tau=20.0, alpha=0.2)` uses symmetric pairing with
constant weakening on each presynaptic arrival. For decayed old traces, its
signed increment is
`polarity * eta * (arrival * (post_old - alpha) + post_spike * pre_old)`.
The rule follows the [Vogels--Sprekeler law documented by NEST](https://nest-simulator.readthedocs.io/en/stable/models/vogels_sprekeler_synapse.html),
with btorch's explicit strict-past, combined-clipping timestep convention.
Coincident spikes do not pair with each other; arrival weakening still applies.

The default `polarity=-1, w_min=-1, w_max=0` represents negative current weights.
Positive inhibitory conductances use `polarity=1, w_min=0, w_max=1`. Polarity
and bounds must agree, including at zero. Receptor identity is not inferred;
set `eta=0` on edges that should not learn. Coefficients can be edge arrays.
`trace_scope="neuron"` reuses segment/post trace ownership with scalar `tau`.
Reset and serialization follow the same memory contract as pair learning.

The continuous-time uncorrelated target rate is `alpha / (2 * tau)` in inverse time units:
`tau=20` milliseconds and `alpha=0.2` correspond to 5 Hz. This equilibrium
does not guarantee a network firing rate under correlated activity. With independent
binary spikes at finite `dt`, the unclipped drift changes sign at
`alpha * expm1(dt / tau) / (2 * dt)`; alpha is not silently corrected.
The eager implementation is the reference. Independent Triton and TileLang
learning stages reuse the same declared rule template, including shared trace
ownership, mixed storage precision and compiled/CUDA Graph continuation.
HELION and the two-kernel event schedule reuse this template too. Delayed
connections use HELION scanning; spike-list schedules require zero delay.

## Voltage-dependent learning

`VoltageSTDP` implements an unshifted voltage-gated rule:

```python
from btorch.models.connection import VoltageSTDP, VoltageSTDPOptions

plasticity = VoltageSTDP(VoltageSTDPOptions(
    tau_pre=15.0, tau_plus=7.0, tau_minus=10.0,
    v_rest=-70.0, theta_minus=-70.0, theta_plus=-45.0,
))
# Attach with Synapse(..., plasticity=plasticity).
```

The default `PostState("v", stage="charged")` reads live pre-reset voltage.
Enable the neuron's charged recording (`pre_spike_v=True` for LIF/GLIF3,
`pre_spike=True` for Izhikevich). No neuron equation is embedded in the rule.
Alternatively, bind `voltage=ExternalSignal("post", time_first=True)` and pass
`plasticity_signals={"voltage": voltage_sequence}` to `RecurrentNN`.
Standalone connections supply `signals={"voltage": post_voltage}` explicitly.

The normalized arrival trace decays and then gains `arrival / tau_pre`.
Its strict-past value gates continuous LTP, multiplied by `dt` and two
rectified voltage factors. LTD occurs on arrivals without a `dt` multiplier.
Both voltage filters advance exponentially under the held current sample;
LTP uses right-endpoint quadrature and the combined increment is clipped once.
`theta_plus` must exceed `theta_minus` at every post neuron. This variant adds
no voltage-filter delay or homeostatic amplitude.

Arrival parameters and amplitudes are scalar/edge-aligned. Voltage time
constants, thresholds and reset values are scalar/post-aligned. Both voltage
filters are updated once per post neuron. Reset clears arrival eligibility and
restores voltage filters to `v_rest`, preserving learned weights. LTP can
continue on silent steps; event-only pair traversal is insufficient.
The separate Triton and TileLang backends use two plasticity launches per
timestep: one post-owned filter stage, then one canonical-edge weight/arrival
stage after old-weight propagation. Both stages share the scalar template
renderer. Live charged/state voltage uses the current neuron workspace;
external voltage uses the existing constant or time-first signal interface.
No full voltage history is needed unless requested as a record.
CUDA Graph replay refreshes external signals through the existing chunk inputs;
the native recurrent operator also supports `torch.compile`.
With `recurrent="helion"`, the same stages run within one cooperative timestep
kernel: neuron and post filters publish their values, a grid barrier precedes
one canonical-edge traversal, and another barrier precedes PSC consumption.
Each edge propagates its old weight and then updates continuous learning once.
Delayed propagation uses the same arrival-time convention as the separate
baseline. Silent edges still evolve; this path uses neither event skipping nor
lazy trace clocks. Continuous fusion requires one canonical edge stage, with supported independent
post-owned stages before it and post/segment stages after it. Unsupported
cross-domain stage dependencies raise explicitly.

## Custom learning rules

`Synapse.plasticity` accepts a `PlasticityRule`, which is a `MemoryModule`.
The connection owns weights; the rule owns its registered memories and fixed
coefficient buffers. A transition returns canonical edge increments, and the
connection applies the increment and clips once after old-weight propagation.
For example, a user-defined gated potentiation rule can use one arrival trace:

```python
from btorch.models.connection import ExternalSignal, PlasticityRule

class GatedLTP(PlasticityRule):
    state_domains = {"trace": "edge"}

    def __init__(self):
        super().__init__(
            w_min=0.0, w_max=1.0, signals={"gain": ExternalSignal()},
        )
        self.register_buffer("tau", torch.tensor(20.0))
        self.register_memory("trace", 0.0, (), persistent=True)

    def single_step_forward(self, weight, events, dt):
        old = self.trace.reshape(-1) * torch.exp(-dt / self.tau)
        delta = events.signals["gain"] * events.post_spike * old
        self.trace = (old + events.arrival).reshape_as(self.trace)
        return delta
```

The transition returns `delta`; it must not update weights itself. Read all
strict-past memories before committing new trace values. `register_memory`
provides reset and persistence, while `register_buffer` stores fixed coefficients.
The example uses raw arrival events, so it also works with heterogeneous delays.

Attach an instance with `Synapse(plasticity=GatedLTP(), ...)`. For standalone
updates, pass the required live signal explicitly:

```python
with torch.no_grad():
    current = connection(arrivals)  # Old weights.
    connection.update_plasticity(
        arrivals, post_spikes, dt=1.0, signals={"gain": gain},
    )
```

Every memory has a declared `state_domains` entry: `edge`, `segment` (nonempty
source/delay group), `post`, or `scalar`. Register memories in `__init__` with
`register_memory(..., sizes=(), persistent=True)` for one scalar per domain
member. Connection binding supplies the leading edge/segment/neuron dimension;
explicit `sizes` describe trailing local dimensions. A `scalar` domain adds no
leading dimension. There is no additional initialization hook. Memories retain
normal `init_state`, reset, device/dtype and checkpoint behavior. Fixed coefficients
default to scalar/input-edge buffers and follow canonical edge sorting;
`parameter_domains` declares population or segment arrays when needed.
`events.post_index` and optional `events.segment_index` map canonical edges to
those domains. Segment mappings reuse the sparse representation cache and
are only allocated for rules requesting segment data.

Custom tensor rules also work in eager `RecurrentNN`. Declare a live neuron
signal with `PostState("v", stage="charged")`; it resolves through the neuron
template's record metadata and reads fresh pre-reset voltage each step. Enable
the corresponding record when constructing the neuron (for example,
`LIF(..., pre_spike_v=True)`). `stage="state"` instead reads post-step state.
Signals retain population layout, so gather with `events.post_index` when
forming an edge update. Missing inputs fail before recurrent state advances.
Caller-supplied signals are explicit tensors:

```python
with torch.no_grad():
    spikes, records = network(inputs, plasticity_signals={"gain": gain})
```

`ExternalSignal()` is constant over that call. Declare
`ExternalSignal(time_first=True)` for a signal of shape `(T, *domain_shape)`;
for example, a changing scalar gain has shape `(T,)`. `RecurrentNN` slices
these tensors through its existing chunk/unroll schedule alongside the
network input. Constant and time-first signals can be supplied together.
The CUDA Graph runner owns and refreshes nested tensor inputs on replay, so a
new signal Tensor with the same shape can reuse a capture. Tensor-only custom
transitions can be compiled; arbitrary Python callbacks do not automatically
become GPU kernel templates. Explicit native selection rejects an unsupported
rule before recurrent state advances.

### Out-of-tree native learning templates

Write native Triton or TileLang functions and register their tensor bindings.
Btorch does not parse equations, symbolically trace functions or translate a
custom math DSL. The rule remains a `PlasticityRule` with `register_memory`;
its concrete class opts in with `kernel_model = "lab_burst_stdp"`.

```python
from btorch.models.kernels import (
    PlasticityStage, PlasticityTemplate, PlasticityTrace,
    register_plasticity_template,
)
from my_package.kernels import burst_triton, burst_tilelang

register_plasticity_template(
    "lab_burst_stdp",  # BurstSTDP.kernel_model
    PlasticityTemplate(
        states=("pre_trace", "post_trace"),
        parameters=("a_plus", "a_minus", "tau", "w_min", "w_max"),
        stages=(PlasticityStage(
            domain="edge",
            reads=("weight", "pre_trace", "post_trace", "a_plus", "a_minus",
                   "tau", "w_min", "w_max", "arrival", "post_event", "dt"),
            outputs=("weight", "pre_trace", "post_trace"),
            triton=burst_triton,
            tilelang=burst_tilelang,
        ),),
        traces=(PlasticityTrace("pre_trace", "tau", "arrival"),
                PlasticityTrace("post_trace", "tau", "post_event")),
        event_driven=True,
    ),
)
```

`burst_triton` is an ordinary `@triton.jit` helper. `burst_tilelang` is a
TileLang helper returning scalar expressions during kernel construction. Both
accept the inputs in `reads` order and return a tuple in `outputs` order.
They run **inside** the existing kernel, not as Python callbacks each timestep.
A backend function may be omitted; explicitly selecting it then raises before
simulation state advances. The native functions own their arithmetic, clipping
and precision. Preserve the supplied tensor dtypes; promote only operations
that require it.

Separate learning stages and HELION use the `neuron` renderer. In standard
recurrence with explicit `spike_selection`, propagation and learning share the
`connection` renderer; `connection="auto"` inherits `neuron`. Supply the native
function for the renderer that executes learning, even when the neuron producer
uses a different backend.

For `event_driven=True`, append two compile-time boolean arguments,
`include_ltp` and `include_ltd`, to that signature. Gate weight terms with them,
but advance traces using both original current events. Current coincident
spikes read old decayed traces; apply the combined increment and clipping once.
The connection propagates the old weight before committing the update. Btorch
uses this contract to retain unique edge writers in outgoing/incoming traversal.

`PlasticityTrace` declares an exact independent exponential decay plus additive
event accumulation. It permits lazy silent-edge evolution, shared trace domains
where valid, and materialization at chunk boundaries. It does not implement or
infer the user's equations. Only claim `event_driven=True` if weights remain
unchanged on silent steps and the native trace updates follow these contracts.
Currently native event hooks require one edge stage, all states declared as
these traces, and no live signals. For other rules, omit `event_driven` and use
ordinary native stages, with explicit state/parameter domains and signals.
The existing continuous HELION schedule supports compatible domain dependencies;
unsupported schedules raise rather than change semantics.

The same functions serve ordinary updates, sparse event propagation and HELION
fusion where supported. Memory, reset, serialization, canonical weight identity,
recording and chunk continuation remain owned by btorch. Online STDP requires
`no_grad` or `inference_mode`; it does not register a BPTT backward callback.

Run `python examples/oot_stdp.py` for the complete independent memory-backed
Torch rule and native functions. Use `--backend triton --device cuda --capture`
or `--backend tilelang --device cuda --recurrent helion --capture` to compare
spikes, neuron/PSC states, traces and weights with eager Torch. `--compile`
checks the native recurrent operator; `--dtype` selects the tensor precision.

Register before compilation/capture. Duplicates raise; `replace=True` replaces
external registrations and increments their revision. Built-ins cannot be
replaced. RecurrentNN invalidates its captures automatically. Recreate compiled
callables and standalone graphs; a stale online operator rejects a changed
revision before mutation. A subclass must declare its own `kernel_model`.
Use `plasticity_template_names` and `plasticity_template_revision` for lookups.

`TripletSTDP(TripletSTDPOptions(...))` supplies an all-to-all four-trace
rule with pair and genuine triplet terms. It uses strictly earlier trace
snapshots and the same delayed-arrival convention as `PairSTDP`. Its optimized
separate update and HELION lazy-trace fusion are implemented in both Triton
and TileLang. Both use the same declared state/coefficient bindings and scalar
equations; triplet maintains four independent exponential trace snapshots.
Custom rules require their own native template for optimized execution.
Built-in voltage and modulated rules already declare supported live signals;
an arbitrary Python transition does not acquire GPU support by subclassing them.

## Optimized recurrent execution

Select the neuron/PSC renderer independently of sparse propagation:

```python
from btorch.models import KernelOptions

network.backend = KernelOptions(neuron="tilelang", connection="tilelang")
# Or neuron="triton". Both renderers execute the same STDP scalar statements.
with torch.no_grad():
    spikes, records = network(inputs)
```

The native online trajectory reuses the existing sparse route and paired
PSC/neuron updates. Rule validation, layout resolution and native launch
bindings are prepared once per chunk. Edge-local event rules add one
independent STDP kernel per step, combining decay, LTD, LTP, clipping and trace
increments. Shared event traces also use one learning kernel: edge writers read
old traces while segment/post owners write compact next snapshots. Buffers swap
between steps; only an odd-length chunk needs a boundary copy into persistent
trace storage. Voltage rules retain two ordered launches, updating post filters
before edge learning.
All stages use the selected rule's mathematical template and state domains.
Pair uses two traces and triplet four. Event rules read old trace snapshots
before the combined weight update and current-event increments.
It is a baseline, **not a HELION single-kernel timestep**. The sparse backend
can also be selected independently: `connection="triton"` uses Triton
propagation with TileLang neuron, PSC and plasticity kernels. Static
connections allocate no plasticity traces or update kernels.

With automatic sparse method/backend selection, expanded connection dimensions
strictly below 4096 use dense matmul internally. The learning stages still
operate only on canonical edges and declared trace domains. Their existing
weight writer also updates the prepared dense operand, using the canonical
storage precision; there is no extra per-step matrix rebuild or copy. This
applies across native rules, including delayed/receptor connections. Explicit
push/pull, backend and recurrent discovery choices retain their requested paths.

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
chunk runner (`chunk_size=200` by default). `chunk_mode="split"` uses fixed
pieces; `"chunk"` uses approximately equal pieces. Repeated full pieces share
a capture; a shorter final piece runs eagerly. Warmup/capture restores both
trace state and canonical weights before real execution. Online STDP still
requires `no_grad` and cannot be combined with gradient checkpointing or BPTT.
Without STDP, existing gradient/VJP and checkpoint paths remain available.

## Experimental HELION execution

Select a cooperative fused timestep with `PairSTDP`, `PowerPairSTDP`,
`NearestPairSTDP`, `TripletSTDP`, or no plasticity:

```python
network.backend = KernelOptions(
    neuron="triton",  # Or "tilelang"
    connection_method="push",
    recurrent="helion",
    spike_selection="scan",
)
with torch.no_grad():
    spikes, records = network(inputs)
```

Use `spike_selection="spike_list"` to consume the existing neuron-produced
queue instead of scanning rows. This currently supports zero-delay static
connections and event-only STDP on both backends. Delayed list execution
and continuous rules raise explicit errors. Both discovery methods reuse
HELION's edge traversal; selecting a list does not select the standard packed
fanout kernel.

For zero-delay event-only STDP, use `recurrent="standard"` with either explicit
discovery method to keep PSC/neuron execution separate from a combined
propagation/plasticity kernel:

```python
network.backend = KernelOptions(
    neuron="triton",
    connection="tilelang",  # May differ from neuron in standard recurrence
    connection_method="push",
    recurrent="standard",
    spike_selection="spike_list",  # Or "scan"
)
```

This schedule uses two launches per interior timestep, plus chunk-boundary
initialization and final PSC. Edge traces need a chunk-exit materialization;
shared traces use compact double buffers and copy back only after odd-length
chunks. There are no per-timestep trace copies or extra trace launches.
This schedule currently excludes delays, `x_syn` and merged receptor states. The default
`spike_selection="auto"` retains the separate STDP baseline. Explicit discovery
requires native neuron templates and push propagation; static scan currently
requires `recurrent="helion"`.

With `recurrent="helion"`, each timestep uses one genuine GPU kernel for neuron
updates, sparse outgoing propagation and PSC updates. It reuses the model templates and canonical
connection weights. The fused kernel uses the neuron renderer throughout;
`connection` must be `"auto"` or match `neuron`. Standard recurrence supports
independent neuron and sparse backends. An actual cooperative launch and a device-wide barrier
separate propagation from PSC consumption; configurations exceeding cooperative
launch resources raise rather than using unsafe synchronization.

Static per-edge integer delays use an invocation-local FutureCurrent ring.
Pending arrivals are rebuilt from the existing `Delay.history` at chunk entry,
so continuation also respects weight edits between calls. Requested history
records retain the public memory layout. No plasticity traces are allocated.
CUDA Graph replay and compiled inference retain the existing recurrent memory
interface. This experimental path currently rejects BPTT, `x_syn`,
`merge_synapse`, connection bias and uniform `DelayedPSC` wrappers. Use
`Synapse(delay=...)` for delays; circular delay history is unsupported by the
native recurrent tensor path.

Fewer launches do not guarantee faster execution. Compare with the default
`recurrent="standard"` path for your workload.

With any of these STDP rules, the same option selects D²CSR + PastSpike execution. Segments
are grouped by `(source, delay)` and retain canonical edge IDs; incoming LTP
traversal shares the existing incoming index. A barrier publishes current
spikes before delayed-arrival propagation. Arrival-active edges own their
combined LTD/LTP update; incoming traversal handles only the other edges,
preventing conflicting weight writes. Propagation consumes the old weight.
A second barrier completes propagation and learning before PSC consumption.
These phases remain inside one timestep kernel on both renderers.

With edge-owned traces, each trace decays lazily with its own time constant
when an edge participates.
Triplet updates read all four old snapshots before incrementing fast and slow
traces; they retain the same single-writer ownership as pair updates.
Nearest pairing replaces or consumes eligibility only after evaluating the old
snapshots; the same edge owner commits both traces and their timestamp.
One additional
kernel materializes final traces at chunk exit; initialization and requested
history recording also have costs outside the timestep launch count. Trace
histories explicitly requested at each step require extra edge traversal.
Static execution performs none of this plasticity work.

Lazy exponential decay and repeated per-step decay are mathematically
equivalent but can round differently. Atomic propagation can also change
accumulation order, so fused and standard execution need not produce
bitwise-identical states or spike rasters.

Shared pair traces instead advance every segment/post once per timestep,
after all edge readers finish. They require neither per-edge trace clocks nor
a chunk-exit materialization kernel. Activity discovery uses the selected scan
or, without delays, the neuron-produced spike lists. Static execution uses
CSR + FutureCurrent and allocates no plasticity state.

## Short-term release

`Synapse.transmission` attaches a `TransmissionRule` independently of learned
weights. `TsodyksMarkram` supplies short-term depression and facilitation:

```python
from btorch.models.connection import Synapse, TsodyksMarkram

synapse = Synapse(
    weight=0.5,
    transmission=TsodyksMarkram(U=0.2, tau_rec=800.0, tau_fac=100.0),
)
```

The owning `SparseConnection` gathers each edge's actual delayed binary
arrival. The rule first recovers available resources toward one and decays
utilization toward zero, then increases utilization on an arrival, releases
`utilization * resources`, and subtracts that release from resources.
`tau_fac=0` disables facilitation. Memories start at resources one and
utilization zero; silent steps still recover/decay them. Time constants use
the same units as `environ.context(dt=...)`.

Transmission scales execution weights without changing canonical weights.
It can coexist with `Synapse.plasticity`: propagation uses the current learned
weight and release efficacy; the later STDP update reads raw arrivals and
postsynaptic spikes. STDP does not receive attenuated release as its spike
input. Reset restores release memories, while `state_dict` preserves their
continuation state.

The current release baseline uses eager execution under `torch.no_grad()`
or `torch.inference_mode()`. Optimized recurrent lowering, BPTT, minibatches,
external sparse-event input and structural rewiring with bound release state
raise explicitly. FP16/BF16 memories and pointwise arithmetic retain their
dtype; mixed inputs follow PyTorch promotion. Custom transmission rules own
memories through the same binding lifecycle and implement `step(arrivals, dt)`
to return one efficacy per canonical edge, without mutating weights.

## Standalone Izhikevich example

Run `python -m examples.rsnn_izhikevich_stdp` for the 60K-neuron circuit with
online excitatory pair learning, fixed inhibition and chunk CUDA Graph replay.
Choose `--recurrent standard|helion` and `--spike-selection scan|spike_list`
independently. The example records neuron/receptor histories while keeping
learning histories disabled. See the [example guide](../examples.md#online-stdp-with-izhikevich)
for timing boundaries, the Torch reference and static controls.

## Reproduce cold and warm comparisons

Run `python -m benchmarks.rsnn_izhikevich.run --modes inference stdp` from the
repository root. It reuses the Izhikevich example at 4096 and 60,000 neurons
and measures Torch native with and without CUDA Graph alongside Triton/TileLang
push and HELION. Each case starts in a fresh process; disk caches are allowed.
Cold time includes bounded chunk JIT/warmup/capture. Construction, reset,
validation and writes are separate. Individual results and matched geometric
means retain slowdowns and failures. See
[the benchmark instructions](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/stdp/benchmarks/rsnn_izhikevich/README.md)
for training, timing scope and unsupported combinations.


## Positive conductance weights and pair learning

`ConductanceRecurrentNN` supports online learning with `ConductancePSC` or
`HeterConductancePSC` under eager Torch cell scheduling. Current is evaluated
from preceding conductance and post-reset voltage; neuron emission and
propagation with old weights precede the learning update. Receptor channels
keep their own conductance, and current is summed only after `g * (E - v)`.
The outer heterogeneous `psc` is current; `base_psc.psc` is conductance.

Conductance E and I weights can both be positive. Select excitatory edges by
source/receptor metadata, not by `weight > 0` as in a signed current example.
Use positive E bounds, and freeze I with equal bounds at its initial positive
weight (the walkthrough also sets I amplitudes to zero). Pass these arrays in
input-edge order; connection binding remaps them to canonical order without
changing receptors. PairSTDP does not infer inhibitory sign from a receptor.

Run `python -m examples.rsnn_izhikevich_conductance_stdp` for the small
CPU walkthrough; `--no-plasticity` gives its matched static control. Child
neuron and synapse backends remain literal `"torch"`; the RNN selects
`KernelOptions(neuron="torch", connection=..., connection_method=...)`.
Independent sparse push, including Triton on CUDA, does not select fused
cell lowering, native plasticity kernels or HELION. This example installs no
STP. See [the conductance example](../examples.md#ei-conductance-with-optional-stdp)
for the exact synthetic circuit, state checks and bounded recording policy.
