# Full Training Example

Complete RSNN training loop with state management and checkpointing.

## Setup

```python
import torch
from btorch.models import environ, functional, init
from btorch.models.neurons import GLIF3
from btorch.models.synapse import AlphaPSCBilleh
from btorch.models.linear import SparseConn
from btorch.models.rnn import RecurrentNN
from btorch.models.init import build_sparse_mat

# Build model
weights, _, _ = build_sparse_mat(n_e=80, n_i=20, i_e_ratio=1.0)
conn = SparseConn(conn=weights)

neuron = GLIF3(n_neuron=100, v_threshold=-50.0, v_reset=-70.0, tau=20.0)
psc = AlphaPSCBilleh(n_neuron=100, tau_syn=5.0, linear=conn)

model = RecurrentNN(
    neuron=neuron,
    synapse=psc,
    step_mode="m",
    update_state_names=("neuron.v", "synapse.psc"),
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
device = "cuda"
model = model.to(device)
```

## State Initialization

`init_net_state` registers memory buffers (v, psc, etc.) and initializes them. `uniform_v_` optionally stores random voltages as reset values.

```python
# Initialize all memory buffers for batch_size=32
functional.init_net_state(model, batch_size=32, device=device, dtype=torch.float32)

# Set random voltages as reset values (deterministic reset each batch)
init.uniform_v_(model.neuron, set_reset_value=True)
```

## Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    
    for batch in train_loader:
        x, target = batch  # x: (T, batch, input_dim)
        x = x.to(device)
        target = target.to(device)
        
        # Reset state to reset values before each forward
        functional.reset_net(model, batch_size=x.shape[1])
        
        # Optional: random init per batch (regularization)
        # init.uniform_state_(model.neuron, ("v",), rand_batch=True)
        
        optimizer.zero_grad()
        
        with environ.context(dt=1.0):
            spikes, states = model(x)
            
            # Compute loss
            task_loss = criterion(spikes, target)
            v_loss = voltage_reg(states["neuron.v"])
            loss = task_loss + 0.1 * v_loss
        
        loss.backward()
        optimizer.step()
```

## Checkpointing

Save/restore memory reset values separately - dynamic buffers are excluded from `state_dict()`.

```python
def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),  # Excludes dynamic buffers
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "memories_rv": functional.named_memory_reset_values(model),
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, weights_only=False)
    
    # Filter out dynamic state keys (already excluded, but be safe)
    state_dict = checkpoint["model_state_dict"]
    dynamic_keys = functional.named_memory_values(model).keys()
    for key in dynamic_keys:
        state_dict.pop(key, None)
    
    # Load weights and optimizer
    model.load_state_dict(state_dict, strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Restore memory reset values
    if "memories_rv" in checkpoint:
        functional.set_memory_reset_values(model, checkpoint["memories_rv"])
    
    return checkpoint["epoch"]
```

## Key Points

1. **init_net_state**: Registers memory buffers (v, psc) and initializes them
2. **reset_net**: Resets buffers to stored reset values (not re-initializes)
3. **uniform_v_**: `set_reset_value=True` stores values for deterministic reset
4. **state_dict**: Excludes dynamic buffers; save/restore `memories_rv` separately
5. **strict=False**: Required when loading since dynamic keys may differ
