# Tutorial: A Heterogeneous Multi-Receptor E/I RSNN

This tutorial builds a recurrent network with excitatory and inhibitory
neurons, four receptor channels, receptor-specific synaptic kinetics, Dale's
law, and a trainable sparse connection. The model uses semantic edge metadata:
each edge stores its `pre`, `post`, and `receptor` attributes. The flattened
`n_post * n_receptor` output layout is derived by `SparseConnection` for the
PSC implementation; it is not the model's connectivity representation.

This is the recommended starting point for a new heterogeneous model. The
legacy `from_hetersynapse` constructor remains useful when importing a matrix
produced by older connectome helpers, but new code should normally construct
semantic edges directly.

## Model Design

We use one neuron population with the excitatory cells first and inhibitory
cells second. The receptor channel is the ordered pair of source and target
cell types:

| Channel | Source | Target | Sign |
|---|---|---|---|
| `EE` | E | E | positive |
| `EI` | E | I | positive |
| `IE` | I | E | negative |
| `II` | I | I | negative |

The same semantic connection therefore expresses both cell-type routing and
Dale's law. `HeterSynapsePSC` maintains a PSC state per receptor channel and
sums the channels back to one current per target neuron before `GLIF3` updates
its voltage.

## Create Semantic E/I Edges

The helper below generates a reproducible random graph without SciPy or a
physically expanded matrix. It returns the edge list, receptor ids, signed
initial weights, and the table used by `HeterSynapsePSC.get_psc` for inspection.

```python
import pandas as pd
import torch


def make_ei_edges(
    n_exc: int,
    n_inh: int,
    density: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    n_neuron = n_exc + n_inh
    generator = torch.Generator().manual_seed(11)
    connected = torch.rand(n_neuron, n_neuron, generator=generator) < density
    connected.fill_diagonal_(False)
    pre, post = connected.nonzero(as_tuple=True)

    pre_is_exc = pre < n_exc
    post_is_exc = post < n_exc
    # EE=0, EI=1, IE=2, II=3; output is post-major, then receptor.
    receptor = (~pre_is_exc).to(torch.long) * 2 + (~post_is_exc).to(torch.long)
    magnitude = 0.08 + 0.04 * torch.rand(pre.numel(), generator=generator)
    values = torch.where(pre_is_exc, magnitude, -magnitude)

    receptor_index = pd.DataFrame(
        {
            "receptor_index": [0, 1, 2, 3],
            "pre_receptor_type": ["E", "E", "I", "I"],
            "post_receptor_type": ["E", "I", "E", "I"],
        }
    )
    return pre, post, receptor, values, receptor_index


n_exc, n_inh = 64, 16
n_neuron = n_exc + n_inh
pre, post, receptor, values, receptor_index = make_ei_edges(n_exc, n_inh)
```

The receptor ids are aligned with the edge arrays. The connection validates
that each id is in `[0, n_receptor)` and derives the expanded output index as
`post * n_receptor + receptor`.

## Build the Connection and Heterogeneous PSC

`Synapse(dale=True)` records the sign at construction and keeps each learned
edge on that side of zero after `constrain_net`. The per-receptor time
constants below are repeated once per post-neuron because the PSC's internal
state is flattened in post-major order:
`[post0/receptor0, post0/receptor1, ..., post1/receptor0, ...]`.

The connection and PSC are built together in the model below, which keeps the
edge metadata and receptor index table aligned from construction through
training:

```python
from torch import nn

from btorch.models import environ, functional
from btorch.models.rnn import RecurrentNN


class HeterogeneousEIRSNN(nn.Module):
    def __init__(self, n_input: int, n_class: int):
        super().__init__()
        pre, post, receptor, values, receptor_index = make_ei_edges(n_exc, n_inh)
        connection = SparseConnection.from_edges(
            pre,
            post,
            n_pre=n_neuron,
            n_post=n_neuron,
            synapse=Synapse(
                receptor=receptor,
                n_receptor=len(receptor_index),
                dale=True,
            ),
            values=values,
            hints=Hints(expected_density=0.02),
        )
        tau_syn = torch.tensor([5.0, 8.0, 6.0, 10.0]).repeat(n_neuron)
        psc = HeterSynapsePSC(
            n_neuron=n_neuron,
            n_receptor=len(receptor_index),
            receptor_type_index=receptor_index,
            linear=connection,
            base_psc=AlphaPSC,
            tau_syn=tau_syn,
            step_mode="s",
        )
        self.input = nn.Linear(n_input, n_neuron)
        self.brain = RecurrentNN(
            neuron=GLIF3(
                n_neuron=n_neuron,
                v_threshold=1.0,
                v_reset=0.0,
                v_rest=0.0,
                c_m=1.0,
                tau=20.0,
                tau_ref=2.0,
                step_mode="s",
            ),
            synapse=psc,
            step_mode="m",
            update_state_names=("neuron.v", "synapse.psc"),
        )
        self.readout = nn.Linear(n_neuron, n_class)

        self.receptor_index = receptor_index

    @property
    def recurrent_connection(self) -> SparseConnection:
        return self.brain.synapse.linear

    def forward(self, x: torch.Tensor):
        drive = self.input(x)
        spikes, states = self.brain(drive)
        logits = self.readout(spikes.mean(dim=0))
        return logits, spikes, states


model = HeterogeneousEIRSNN(n_input=24, n_class=4)
```

## Train with Dale's Law

The training loop is ordinary PyTorch. The additional step is
`constrain_net(model)` after every optimizer update, which projects Dale
constrained weights back onto their recorded signs.

```python
from btorch.models.constrain import constrain_net
from btorch.models.init import uniform_v_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch_size, n_step = 16, 40
functional.init_net_state(model, batch_size=batch_size, device=device)
uniform_v_(model.brain.neuron, set_reset_value=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()

for _ in range(10):
    inputs = torch.randn(n_step, batch_size, 24, device=device)
    targets = torch.randint(0, 4, (batch_size,), device=device)
    functional.reset_net(model, batch_size=batch_size)
    optimizer.zero_grad(set_to_none=True)

    with environ.context(dt=1.0):
        logits, spikes, states = model(inputs)
        loss = criterion(logits, targets)

    loss.backward()
    optimizer.step()
    constrain_net(model)
```

If the network is trained with truncated BPTT, call `functional.detach_net`
between chunks rather than resetting the neuron and PSC memories. Reset only
between independent sequences or batches.

## Inspect Receptor Channels and Live Weights

The PSC table is kept next to the model because the connection stores integer
receptor ids, not the human-readable DataFrame. `get_psc` accepts a
`(pre_type, post_type)` tuple in neuron mode.

```python
psc = model.brain.synapse
ee_current = psc.get_psc(("E", "E"))
ie_current = psc.get_psc(("I", "E"))
all_channels = psc.get_psc()  # [..., n_neuron * n_receptor]

conn = model.recurrent_connection
live_weight = conn.weight()       # current autograd-aware edge values
edge_snapshot = conn.edge_table() # detached pre/post/receptor/weight snapshot

print(ee_current.shape, ie_current.shape)
print(edge_snapshot["receptor"].unique())
print(conn.n_post, conn.n_receptor, conn.out_features)
```

The summed `psc.psc` state has shape `[..., n_neuron]`; the inner
`psc.base_psc.psc` and `get_psc()` expose the flattened receptor channels.
After an optimizer step, call `conn.weight()` or `get_psc()` again instead of
holding an old tensor snapshot.

## Backend and CUDA Graph Guidance

The semantic model is independent of the execution backend. Inspect or compare
routes through the connection:

```python
from btorch.sparse import runtime

print(conn.explain())
with runtime.use_backend("aten"):
    reference = conn(torch.randn(2, n_neuron, device=device))
```

On CUDA, an `expected_density` hint can select a Triton push route for sparse
spike activity. Push accumulation is not bitwise reproducible. Use the pull
route or deterministic algorithms when exact reproducibility is required.

For inference with fixed batch and sequence shapes, construct
`RecurrentNN(..., cudagraph=True)` and initialize it before capture. This mode
does not support gradient recording. For training, prefer
`torch.compile(model)` and keep autograd enabled. A weight update refreshes
prepared values; changing edge ids with `set_edges_` or loading a different
topology invalidates the captured graph and triggers recapture on the next
`cudagraph=True` call.

## Common Pitfalls

- Do not build a new model by multiplying `n_neuron` by `n_receptor`. That is
  the lowered tensor layout, not the semantic population size.
- `Synapse.receptor` is one integer per stored edge. `n_receptor` is the number
  of possible channels, not the number of edges.
- For neuron-mode inspection, use `get_psc(("E", "I"))`; string names are for
  connection-mode receptor tables.
- Per-receptor `tau_syn` values must be repeated in post-major channel order
  when passed to `AlphaPSC` through `HeterSynapsePSC`.
- `HeterSynapsePSC` needs `functional.init_net_state` before its first call,
  including when it owns a delay history.
- `from_hetersynapse` is an import adapter for physically expanded matrices.
  Prefer `from_edges(..., Synapse(receptor=..., delay=...))` for new code.
- Dale constraints are applied after `optimizer.step()`, not before the
  backward pass.

## Related API

- [Sparse connectivity guide](../guides/sparse_connectivity.md)
- [Heterogeneous synapses reference](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/skills/btorch-snn-modelling/references/heter_synapses.md)
- [`SparseConnection`](../api/models.md)
- [`HeterSynapsePSC`](../api/models.md)
