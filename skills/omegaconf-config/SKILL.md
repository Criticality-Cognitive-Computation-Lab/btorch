---
name: omegaconf-config
description: Dataclass-first configuration using OmegaConf. Use when defining config defaults in Python dataclasses, loading CLI overrides, parameter sweeps, case/preset selection, or option forwarding between launcher and worker processes.
---

# OmegaConf Configuration

## Core Principle

**Dataclasses = source of truth.** All defaults in Python, not YAML.

```python
from omegaconf import OmegaConf

@dataclass
class Config:
    lr: float = 1e-3
    epochs: int = 100

defaults = OmegaConf.structured(Config())
cli_cfg = OmegaConf.from_cli()  # Parse "lr=0.01"
cfg = OmegaConf.unsafe_merge(defaults, cli_cfg)
```

## Key Patterns

| Pattern | When to Use | Key Point |
|---------|-------------|-----------|
| **Composition** | Split config across domains | `CommonConf + TaskConf` via `field(default_factory=...)` |
| **Case Selection** | Named presets selectable via CLI | `case_name` field + `default_from_case()` + `load_config_with_case()` |
| **Option Forwarding** | Launcher → Worker | `return_cli=True` + `to_dotlist(exclude={...})` |
| **Parameter Sweeps** | Grid/random search | `deepcopy(base_cfg)` before modifying |
| **Dataclass Unions** | Polymorphic variants (subcommands, subcases) | OmegaConf auto-injects `_type_`; no manual discriminator |

**Dataclass Unions** enable polymorphic configuration — like subcommands or subcases where
different variants share structural similarity but differ in specific fields. OmegaConf
automatically injects `_type_` to track and select which variant is active; no need to manually add
a discriminator field.

```python
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

# CLI: optimizer=SGDConf optimizer.lr=0.01 optimizer.momentum=0.95
```

## Quick Config Loader

```python
from btorch.utils.conf import load_config

cfg = load_config(MyConfig)  # dataclass defaults → CLI merge
```

## CLI Examples

```bash
python train.py lr=0.01 epochs=50                    # Override defaults
python train.py case_name=fast                       # Select case preset
python train.py case_name=fast lr=0.02              # Case + override
```

## Hydra? Stop.

If the codebase uses Hydra schema/config-file patterns, **stop and confirm with user** before using this skill.

## References

Detailed patterns: `references/examples.md`
