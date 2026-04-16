# OmegaConf Configuration Guide

btorch provides dataclass-first configuration utilities in [`btorch.utils.conf`](../api/utils.md). This guide explains how to structure configs, load them from files and CLI arguments, and forward options from launcher scripts to worker processes.

## Why Dataclass-First Config

- **Type safety**: Config fields are typed and checked at load time.
- **Single source of truth**: Defaults live in Python dataclasses, not YAML files.
- **Composable**: Nested dataclasses cleanly separate common settings from task-specific settings.
- **CLI-friendly**: OmegaConf parses `key=value` overrides automatically.

## Quick Start

The simplest pattern loads a dataclass and merges it with CLI overrides:

```python
from dataclasses import dataclass
from btorch.utils.conf import load_config

@dataclass
class Config:
    lr: float = 1e-3
    epochs: int = 100

cfg = load_config(Config)
print(cfg.lr)  # overridden by `lr=0.01` on the command line
```

Run with:

```bash
python train.py lr=0.01 epochs=200
```

## Composition Pattern

Real projects usually have nested configs. Here is the pattern used in btorch's own examples:

```python
from dataclasses import dataclass, field

@dataclass
class CommonConf:
    output_path: str = "outputs"
    seed: int = 42

@dataclass
class SolverConf:
    lr: float = 1e-3
    max_iter: int = 1000

@dataclass
class ArgConf:
    common: CommonConf = field(default_factory=CommonConf)
    solver: SolverConf = field(default_factory=SolverConf)
```

CLI overrides use dot notation:

```bash
python train.py common.seed=123 solver.lr=0.005
```

## Loading from File

If you pass `config_path=path/to/config.yaml`, `load_config` merges the YAML on top of defaults before applying CLI overrides:

```bash
python train.py config_path=base.yaml solver.lr=0.005
```

Priority order: **dataclass defaults → config file → CLI arguments**.

## Variant Selection with `_type_`

OmegaConf supports dataclass unions. You can switch between variants without adding a manual `mode: str` field:

```python
from dataclasses import dataclass, field

@dataclass
class AdamConf:
    lr: float = 1e-3

@dataclass
class SGDConf:
    lr: float = 1e-2
    momentum: float = 0.9

@dataclass
class TrainConf:
    optimizer: AdamConf | SGDConf = field(default_factory=AdamConf)
```

Switch at runtime:

```bash
python train.py optimizer="{_type_: SGDConf, lr: 0.01, momentum: 0.95}"
```

## Launcher → Worker Option Forwarding

When running sweeps or distributed jobs, a launcher script often needs to forward CLI overrides to individual workers. `to_dotlist` converts an OmegaConf container into a list of `key=value` strings:

```python
from btorch.utils.conf import load_config, to_dotlist

@dataclass
class BatchConf:
    single: ArgConf = field(default_factory=ArgConf)
    max_workers: int = 4

cfg, cli_cfg = load_config(BatchConf, return_cli=True)

# Forward everything except the per-worker ID
dotlist = to_dotlist(
    cli_cfg.single,
    use_equal=True,
    exclude={"common.id"},
)

# Build worker command
cmd = ["python", "worker.py"] + dotlist + [f"common.id={worker_id}"]
```

## Integration with Training Scripts

The Fashion-MNIST example (`examples/fmnist_lightning.py`) uses a nested `NetworkConfig` dataclass for neuron and synapse hyperparameters, loaded via `load_config`. You can use the same pattern for your own models:

```python
@dataclass
class NetworkConfig:
    n_neuron: int = 256
    dt: float = 1.0

cfg = load_config(NetworkConfig)
with environ.context(dt=cfg.dt):
    # ... training loop
```

## Utility Reference

| Function | Purpose |
|----------|---------|
| [`load_config`](../api/utils.md) | Load dataclass + file + CLI |
| [`to_dotlist`](../api/utils.md) | Flatten config to CLI strings |
| [`diff_conf`](../api/utils.md) | Compute structured diff between two configs |
| [`get_dotkey`](../api/utils.md) | Read nested field by dot path |
| [`set_dotkey`](../api/utils.md) | Write nested field by dot path |
