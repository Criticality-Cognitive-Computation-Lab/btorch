# FAQ / Common Pitfalls

This page collects the most common mistakes when working with btorch and how to fix them. Content is drawn from the [`btorch-snn-modelling` skill](skills.md) and the test suite.

## 1. Forgetting the `dt` Context

**Symptom:** `KeyError: 'dt is not found in the context.'`

**Fix:** Wrap every forward pass in `environ.context(dt=...)`:

```python
from btorch.models import environ

with environ.context(dt=1.0):
    spikes, states = model(x)
```

See [The `dt` Environment](concepts/dt_environment.md) for details.

## 2. Not Resetting State Between Batches

**Symptom:** State from the previous batch leaks into the current one, causing unstable training or validation results.

**Fix:** Call `reset_net` before each new batch:

```python
from btorch.models import functional

functional.reset_net(model, batch_size=x.shape[1])
```

For deterministic reset, initialize random voltages first:

```python
from btorch.models.init import uniform_v_
uniform_v_(model.neuron, set_reset_value=True)
```

## 3. Wrong State Names (Dot Notation)

**Symptom:** `KeyError` when accessing `states`, or `update_state_names` not recording the variable you expected.

**Fix:** Use dotted names that match the module hierarchy:

```python
model = RecurrentNN(
    neuron=neuron,
    synapse=psc,
    update_state_names=("neuron.v", "synapse.psc"),
)
```

You can inspect valid names with:

```python
print(functional.named_hidden_states(model).keys())
```

## 4. Missing Memory Reset Values in Checkpoints

**Symptom:** After loading a checkpoint, neurons reset to factory defaults instead of the trained initialization values.

**Fix:** Save and restore `_memories_rv` explicitly:

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "memories_rv": functional.named_memory_reset_values(model),
}

# Load
model.load_state_dict(ckpt["model_state_dict"], strict=False)
functional.set_memory_reset_values(model, ckpt["memories_rv"])
```

See [Tutorial 2: Training an SNN](tutorials/training.md) for a complete example.

## 5. OmegaConf `_type_` Usage and CLI Syntax

**Symptom:** `TypeError` or `ValidationError` when passing variant configs on the CLI.

**Fix:** Use the `_type_` key to switch union variants:

```bash
python train.py optimizer="{_type_: SGDConf, lr: 0.01, momentum: 0.95}"
```

Or with nested keys:

```bash
python train.py optimizer._type_=SGDConf optimizer.lr=0.01
```

See the [Configuration Guide](guides/configuration.md) for the full pattern.

## 6. `torch.compile` and Dynamic Buffers

**Symptom:** `torch.compile` fails on models with circular-buffer-based history.

**Fix:** Set `use_circular_buffer=False` in `SpikeHistory` or `DelayedSynapse` when compiling for training:

```python
from btorch.models.history import SpikeHistory
history = SpikeHistory(n_neuron=100, max_delay_steps=5, use_circular_buffer=False)
```

This trades memory efficiency for full `torch.compile` compatibility.
