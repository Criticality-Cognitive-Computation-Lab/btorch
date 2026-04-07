---
name: omegaconf-config
description: Dataclass-first configuration using OmegaConf with OOP domain models. Always define config/domain models, defaults, and named cases in Python dataclasses; attach reusable methods on those dataclasses when behavior belongs with the config. Avoid YAML as config source. YAML is only for serialization/deserialization and reproducibility snapshots. Use for hierarchical configs with CLI overrides, parameter sweeps with base configs + per-trial variations, and option forwarding from launcher to worker processes. Explicitly avoid Hydra schema + config-file patterns.
---

# OmegaConf Configuration (Dataclass-First)

## When to Use (Compatibility Gate)

Use this skill only for dataclass-first OmegaConf workflows.

- If the codebase already uses Hydra schema/config-file patterns, stop and
  confirm with the user before proceeding.
- Treat Hydra-based setups as incompatible with this skill's patterns unless
  the user explicitly asks to migrate away from Hydra.
- If no Hydra dependency exists, proceed with dataclass-first OmegaConf.

**Hard rule:** treat dataclasses as OOP domain models for configuration.

**Strong preference:** do not use Hydra schema/config-file patterns in this skill.
Use plain dataclasses + OmegaConf merge utilities only.

- Define config/domain models in Python dataclasses.
- Define default values in Python dataclasses.
- Define named case presets in Python code (for example, `CASES` maps).
- Attach reusable methods on dataclasses when behavior belongs to the model
  (for example, `default_from_case()`, `use_debug()`, `get_ranges()`).
- Use CLI dot overrides to modify dataclass defaults at runtime.

**Do not treat YAML as config source.** Use YAML only for:

- serialization/deserialization (`OmegaConf.save` / `OmegaConf.load`), and
- reproducibility snapshots of already-defined dataclass configs.

**Out of scope:** Hydra usage, Hydra schema patterns, and Hydra config-file composition.

## Quick Start

```python
from dataclasses import dataclass, field
from typing import TypeVar, cast
from omegaconf import OmegaConf

T = TypeVar("T")

@dataclass
class Config:
    lr: float = 1e-3
    epochs: int = 100

def load_config(Param: type[T]) -> T:
    defaults = OmegaConf.structured(Param())
    cli_cfg = OmegaConf.from_cli()  # Parse "lr=0.01"
    cfg = OmegaConf.unsafe_merge(defaults, cli_cfg)
    return cast(T, OmegaConf.to_object(cfg))
```

## Core Patterns

### 1. Composition (Common + Task Config)

```python
@dataclass
class CommonConf:
    """Shared across experiments."""
    output_path: Path = Path("./outputs")
    seed: int = 42

@dataclass
class SolverConf:
    """Task-specific settings."""
    lr: float = 1e-3
    max_iter: int = 1000

@dataclass
class ArgConf:
    """Top-level composition."""
    common: CommonConf = field(default_factory=CommonConf)
    solver: SolverConf = field(default_factory=SolverConf)
```

### 2. Option Forwarding (Launcher → Worker)

```python
from typing import Any, cast

from btorch.utils.conf import load_config, to_dotlist

@dataclass
class BatchConf:
    single: ArgConf = field(default_factory=ArgConf)
    max_workers: int = 4

# Launcher loads config with CLI
cfg, cli_cfg = cast(
    tuple[BatchConf, Any],
    load_config(BatchConf, return_cli=True),
)

# Forward single-task options to workers (exclude launcher-only fields)
dotlist = to_dotlist(
    cli_cfg.single,
    use_equal=True,
    exclude={"common.id"},  # Will be set per-worker
)

# Build worker command
cmd = ["python", "worker.py"] + dotlist + ["common.id=" + str(worker_id)]
```

### 3. Dataclass Unions (`_type_`) for Variant Models

Prefer union type matching over manual string switches. Do not add a
`mode: str` field just to distinguish variants when a dataclass union can encode
the choice directly.

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

Prefer this over manual discriminator fields like:

```python
# Avoid this pattern for variant selection
@dataclass
class TrainConfBad:
    mode: str = "adam"
    lr: float = 1e-3
    momentum: float = 0.0
```

CLI switch with `_type_`:

```bash
# prefer
python train.py optimizer=SGDConf optimizer.lr=0.01 optimizer.momentum=0.95
# alternatively
python train.py optimizer="{_type_: SGDConf,lr: 0.01,momentum: 0.95}"
```

### 4. Type Support and Ignored Runtime Fields

OmegaConf structured configs work well with common typed fields:

- `Literal[...]` for constrained options
- `Sequence[T]`, `list[T]`, `dict[str, T]` for collections
- nested dataclasses and unions of dataclasses

Avoid placing complex runtime objects directly in config (for example
`numpy.ndarray`, tensors, open handles). Mark non-config runtime fields with
`omegaconf_ignore`:

```python
from dataclasses import dataclass, field
from typing import Literal, Sequence
import numpy as np

@dataclass
class FeatureConf:
    mode: Literal["train", "eval"] = "train"
    names: Sequence[str] = ("a", "b")
    weights: list[float] = field(default_factory=lambda: [1.0, 0.5])

    # Runtime-only field: ignored by OmegaConf structured config
    np_cache: np.ndarray | None = field(
        default=None,
        metadata={"omegaconf_ignore": True},
    )

    # Scalar runtime field can also be ignored the same way
    runtime_flag: int = field(default=2, metadata={"omegaconf_ignore": True})
```

### 5. Base + Trial Config (Parameter Sweeps)

```python
from copy import deepcopy
from dataclasses import dataclass
import itertools

@dataclass
class TrialConfig:
    """Base config with sweep candidates."""
    param_a: float = 1.0
    param_b: float = 2.0
    
    # Candidates defined in CODE
    candidates_a: list[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0, 2.0]
    )
    candidates_b: list[float] = field(
        default_factory=lambda: [1.0, 2.0, 4.0]
    )


def run_sweep(base_cfg: TrialConfig):
    """Grid search over candidates."""
    results = []
    
    for a, b in itertools.product(
        base_cfg.candidates_a,
        base_cfg.candidates_b,
    ):
        trial_cfg = deepcopy(base_cfg)
        trial_cfg.param_a = a
        trial_cfg.param_b = b
        
        result = run_trial(trial_cfg)
        results.append({
            "params": {"a": a, "b": b},
            "result": result,
        })
    
    return results
```

### 6. Worker/Launcher Split

**worker.py** (processes one item):
```python
@dataclass
class ArgConf:
    common: CommonConf = field(default_factory=CommonConf)
    solver: SolverConf = field(default_factory=SolverConf)

def main():
    cfg = load_config(ArgConf)
    result = process(cfg.common.id, cfg.solver)
    save_result(result, cfg.common.output_path)

if __name__ == "__main__":
    main()
```

**launcher.py** (distributes work):
```python
@dataclass
class LauncherConf:
    single: ArgConf = field(default_factory=ArgConf)
    ids: list[int] | None = None
    max_workers: int = 4

def main():
    cfg, cli_cfg = load_config(LauncherConf, return_cli=True)
    
    ids = cfg.ids or range(100)
    base_dotlist = to_dotlist(
        cli_cfg.single,
        use_equal=True,
        exclude={"common.id"},
    )
    
    # Dispatch worker commands using your project's existing parallel runtime.
    for id in ids:
        cmd = base_dotlist + [f"common.id={id}"]
        run_worker(cmd)

if __name__ == "__main__":
    main()
```

## btorch Utilities

From `btorch/utils/conf.py`:

- `load_config(Param, return_cli=False)`: Load dataclass + CLI overrides
- `to_dotlist(conf, exclude={})`: Convert to CLI-style overrides for forwarding
- `get_dotkey(obj, "a.b.c")`: Access nested field by dot path
- `set_dotkey(obj, "a.b.c", value)`: Set nested field by dot path

## Best Practices

1. **Dataclass = source of truth**: All defaults in Python, not YAML
2. **Composition over monoliths**: Split into CommonConf + TaskConf
3. **Use `return_cli=True`**: Capture CLI for forwarding to workers
4. **Exclude per-worker fields**: Use `exclude={"common.id"}` in to_dotlist
5. **Deepcopy for sweeps**: Always `deepcopy(base_cfg)` before modifying
6. **Candidates in code**: Define sweep ranges as dataclass fields
7. **Keep behavior with model**: Put reusable config logic on dataclasses
   (`default_from_case`, normalization helpers, debug presets)
8. **Type load results for IDE**: Prefer typed return signatures for wrappers,
   or use `cast(...)` at call sites when using generic helpers

See `references/examples.md` for complete working implementations.
