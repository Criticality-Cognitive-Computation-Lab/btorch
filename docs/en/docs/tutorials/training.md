# Tutorial 2: Training an SNN

**Author:** btorch contributors  
**Based on:** [`examples/fmnist.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/examples/fmnist.py), [`skills/btorch-snn-modelling/references/training_example.md`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/skills/btorch-snn-modelling/references/training_example.md)

This tutorial explains how to train a spiking neural network with btorch, including state initialization, the `dt` environment, checkpointing, and truncated BPTT.

## Network Setup

We reuse the RSNN pattern from [Tutorial 1](building_rsnn.md), but with a sparse recurrent connection and the Billeh alpha-PSC synapse used in the Fashion-MNIST example:

```python
import torch
from btorch.models import environ, functional
from btorch.models.neurons import GLIF3
from btorch.models.synapse import AlphaPSCBilleh
from btorch.models.linear import SparseConn
from btorch.models.rnn import RecurrentNN
from btorch.models.init import uniform_v_
from btorch.models.regularizer import VoltageRegularizer

# Build sparse connectivity
from tests.utils.conn import build_sparse_mat  # helper from test suite
weights, _, _ = build_sparse_mat(n_e=80, n_i=20, i_e_ratio=1.0)
conn = SparseConn(conn=weights)

neuron = GLIF3(
    n_neuron=100,
    v_threshold=-45.0,
    v_reset=-60.0,
    c_m=2.0,
    tau=20.0,
    k=[1.0 / 80],
    asc_amps=[-0.2],
    tau_ref=2.0,
    detach_reset=False,
    step_mode="s",
    backend="torch",
)

# AlphaPSCBilleh requires dt at init time
environ.set(dt=1.0)
psc = AlphaPSCBilleh(
    n_neuron=100,
    tau_syn=torch.cat([torch.ones(80) * 5.8, torch.ones(20) * 6.5]),
    linear=conn,
    step_mode="s",
)

model = RecurrentNN(
    neuron=neuron,
    synapse=psc,
    step_mode="m",
    update_state_names=("neuron.v", "synapse.psc"),
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

## Initialize and Randomize State

```python
# 1. Register memory buffers
functional.init_net_state(model, batch_size=32, device=device)

# 2. Randomize membrane voltage and store as reset values
uniform_v_(model.neuron, set_reset_value=True)
```

`set_reset_value=True` is important: it tells `reset_net` to restore voltages to these randomized values at the start of each batch.

## Training Loop

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
voltage_reg = VoltageRegularizer(-45.0, -60.0, voltage_cost=1.0)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        x, target = batch  # x: (T, Batch, input_dim)
        x = x.to(device)
        target = target.to(device)

        # Reset state before each batch
        functional.reset_net(model, batch_size=x.shape[1])

        optimizer.zero_grad()

        with environ.context(dt=1.0):
            spikes, states = model(x)

            # spikes: (T, Batch, N) -> rate code
            rate = spikes.mean(dim=0)  # (Batch, N)
            task_loss = criterion(rate, target)

            # Voltage regularization
            v_loss = voltage_reg(states["neuron.v"])
            loss = task_loss + 0.1 * v_loss

        loss.backward()
        optimizer.step()
```

## Checkpointing

Dynamic buffers are excluded from `state_dict()`. To fully restore a model, save memory reset values alongside weights:

```python
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "memories_rv": functional.named_memory_reset_values(model),
    }, path)

def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Load weights (dynamic keys are already excluded)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Restore memory reset values
    if "memories_rv" in ckpt:
        functional.set_memory_reset_values(model, ckpt["memories_rv"])
    if "hidden_states" in ckpt:
        functional.set_hidden_states(model, ckpt["hidden_states"])

    return ckpt["epoch"]
```

## Truncated BPTT

For long sequences, you can break BPTT into chunks with `detach_net`:

```python
chunk_size = 50
for t in range(0, T, chunk_size):
    functional.detach_net(model)
    # Note: do NOT call reset_net here; state should persist across chunks
    spikes, states = model(x[t:t+chunk_size])
    loss = criterion(spikes.mean(0), target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

`detach_net` breaks the computation graph at the current state values, preventing gradients from flowing back to earlier chunks.

## Key Takeaways

1. **Always reset state** before a new batch with `functional.reset_net`.
2. **Always wrap forward** in `environ.context(dt=...)`.
3. **Save `memories_rv`** when checkpointing; `state_dict()` does not include dynamic states.
4. **Use `detach_net`** for truncated BPTT on long sequences.

See the [FAQ](../faq.md) for common errors and troubleshooting.
