# OmegaConf Complete Examples

Dataclass-first examples only: domain models/defaults/cases in Python dataclasses, YAML only for ser/deser and reproducibility snapshots.

Avoid Hydra schema + config-file patterns in all examples below.

## Pattern 1: Common + Task Config Composition

```python
# conf.py
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CommonConf:
    """Shared configuration across all experiments.
    
    These fields are used by both the launcher and workers.
    """
    id: int | None = None  # Per-worker identifier
    output_path: Path = Path("./outputs")
    seed: int = 42
    overwrite: bool = False
    
    def __post_init__(self):
        # Auto-set id from environment if not provided
        if self.id is None:
            import os
            self.id = int(os.environ.get("SLURM_PROCID", "0"))
        
        # Ensure output path exists
        self.output_path = Path(self.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)


@dataclass
class SolverConf:
    """Task-specific configuration.
    
    These are the parameters being optimized or varied.
    """
    lr: float = 1e-3
    max_iter: int = 1000
    tolerance: float = 1e-6
    
    # Candidates for grid search - defined in CODE
    lr_candidates: list[float] = field(
        default_factory=lambda: [1e-4, 1e-3, 1e-2]
    )


@dataclass
class ArgConf:
    """Top-level configuration composition."""
    common: CommonConf = field(default_factory=CommonConf)
    solver: SolverConf = field(default_factory=SolverConf)
```

## Pattern 2: Single Worker (Processes One Item)

```python
# worker.py
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import TypeVar, cast
from omegaconf import OmegaConf

from conf import CommonConf, SolverConf


@dataclass
class ArgConf:
    common: CommonConf = field(default_factory=CommonConf)
    solver: SolverConf = field(default_factory=SolverConf)


T = TypeVar("T")


def load_config(Param: type[T], use_config_file: bool = True) -> T:
    """Load config from dataclass defaults + CLI overrides."""
    defaults = OmegaConf.structured(Param())
    cli_cfg = OmegaConf.from_cli()
    
    if use_config_file and "config_path" in cli_cfg:
        from pathlib import Path
        cfg_file = OmegaConf.load(cli_cfg.pop("config_path"))
    else:
        cfg_file = OmegaConf.create()
    
    cfg = OmegaConf.unsafe_merge(defaults, cfg_file, cli_cfg)
    return cast(T, OmegaConf.to_object(cfg))


def process_item(item_id: int, solver_cfg: SolverConf) -> dict:
    """Process a single item."""
    import numpy as np
    
    # Simulate processing
    result = {
        "item_id": item_id,
        "lr": solver_cfg.lr,
        "loss": np.random.random(),
    }
    return result


def main():
    cfg = load_config(ArgConf)
    
    print(f"Processing item {cfg.common.id}")
    print(f"LR: {cfg.solver.lr}, Max iter: {cfg.solver.max_iter}")
    
    result = process_item(cfg.common.id, cfg.solver)
    
    # Save result
    output_file = cfg.common.output_path / f"result_{cfg.common.id}.json"
    output_file.write_text(json.dumps(result, indent=2))
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
```

## Pattern 3: Launcher with Option Forwarding

```python
# launcher.py
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Sequence, cast

from omegaconf import OmegaConf

from btorch.utils.conf import load_config, to_dotlist
from worker import ArgConf as SingleArgConf


@dataclass
class LauncherConf:
    """Launcher configuration.
    
    Contains single-task config plus launcher-specific settings.
    """
    single: SingleArgConf = field(default_factory=SingleArgConf)
    
    # Launcher-specific (NOT forwarded to workers)
    ids: list[int] | None = None
    id_select: str | None = None  # "all" to auto-detect
    max_workers: int = 4
    timeout: int = 600


def get_item_ids(cfg: LauncherConf) -> list[int]:
    """Determine which items to process."""
    if cfg.ids is not None:
        return cfg.ids
    
    if cfg.id_select == "all":
        # Query worker for total count
        result = subprocess.run(
            [sys.executable, "-m", "worker", "common.get_size=true"],
            capture_output=True,
            text=True,
        )
        total = int(result.stdout.strip())
        return list(range(total))
    
    return [0]  # Default to single item


def build_worker_cmd(
    base_dotlist: list[str],
    item_id: int,
) -> list[str]:
    """Build command for a single worker."""
    return (
        [sys.executable, "-m", "worker"]
        + base_dotlist
        + [f"common.id={item_id}"]
    )


def run_worker(cmd: list[str]) -> None:
    """Run a single worker process."""
    subprocess.run(cmd, check=True)


def main():
    # Load config with CLI capture for forwarding
    cfg, cli_cfg = cast(
        tuple[LauncherConf, Any],
        load_config(LauncherConf, return_cli=True),
    )
    
    # Get items to process
    ids = get_item_ids(cfg)
    print(f"Processing {len(ids)} items with {cfg.max_workers} workers")
    
    # Extract CLI overrides for single-task config
    # Exclude 'common.id' since it varies per worker
    base_dotlist = to_dotlist(
        cli_cfg.single,
        use_equal=True,
        exclude={"common.id", "common.get_size"},
    )
    
    print(f"Base options: {' '.join(base_dotlist)}")
    
    # Example: print first worker command
    print(f"Example: {' '.join(build_worker_cmd(base_dotlist, ids[0]))}")
    
    # Run workers in parallel
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
        for item_id in ids:
            cmd = build_worker_cmd(base_dotlist, item_id)
            executor.submit(run_worker, cmd)


if __name__ == "__main__":
    main()
```

## Pattern 4: Parameter Sweep (Base + Trial Configs)

```python
# sweep.py
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import itertools
import json
from typing import Sequence

from omegaconf import OmegaConf


@dataclass
class SweepConfig:
    """Base configuration with sweep candidates."""
    # Base values
    param_a: float = 1.0
    param_b: float = 2.0
    param_c: str = "default"
    
    # Sweep candidates - defined in CODE
    candidates_a: list[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0, 2.0]
    )
    candidates_b: list[float] = field(
        default_factory=lambda: [1.0, 2.0, 4.0]
    )
    candidates_c: list[str] = field(
        default_factory=lambda: ["small", "medium", "large"]
    )
    
    # Which parameters to sweep
    sweep_params: list[str] = field(
        default_factory=lambda: ["param_a", "param_b"]
    )


def generate_trials(base_cfg: SweepConfig) -> Sequence[SweepConfig]:
    """Generate all trial configs from base + candidates."""
    trials = []
    
    # Get candidate lists for swept parameters
    candidate_lists = []
    for param in base_cfg.sweep_params:
        candidates = getattr(base_cfg, f"candidates_{param.split('_')[1]}")
        candidate_lists.append(candidates)
    
    # Cartesian product of all candidates
    for values in itertools.product(*candidate_lists):
        trial = deepcopy(base_cfg)
        for param, value in zip(base_cfg.sweep_params, values):
            setattr(trial, param, value)
        trials.append(trial)
    
    return trials


def run_trial(trial_cfg: SweepConfig) -> dict:
    """Run a single trial."""
    import numpy as np
    
    # Simulate experiment
    score = (
        trial_cfg.param_a * 0.5 +
        trial_cfg.param_b * 0.3 +
        np.random.random() * 0.1
    )
    
    return {
        "params": {
            "a": trial_cfg.param_a,
            "b": trial_cfg.param_b,
            "c": trial_cfg.param_c,
        },
        "score": score,
    }


def save_trial_result(result: dict, output_path: Path, trial_idx: int) -> None:
    """Save trial result to disk."""
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"trial_{trial_idx:04d}.json"
    result_file.write_text(json.dumps(result, indent=2))


def main():
    # Load base config
    defaults = OmegaConf.structured(SweepConfig())
    cli_cfg = OmegaConf.from_cli()
    base_cfg = OmegaConf.to_object(
        OmegaConf.unsafe_merge(defaults, cli_cfg)
    )
    
    print(f"Sweeping: {base_cfg.sweep_params}")
    
    # Generate all trials
    trials = generate_trials(base_cfg)
    print(f"Generated {len(trials)} trials")
    
    # Run trials
    results = []
    for idx, trial_cfg in enumerate(trials):
        print(f"Trial {idx}: a={trial_cfg.param_a}, b={trial_cfg.param_b}")
        result = run_trial(trial_cfg)
        results.append(result)
        
        save_trial_result(
            result,
            base_cfg.output_path,
            idx,
        )
    
    # Save summary
    summary_file = Path(base_cfg.output_path) / "summary.json"
    summary_file.write_text(json.dumps(results, indent=2))
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
```

## Pattern 5: Complete Launcher + Worker Example

```python
# experiments/image_classifier/worker.py
"""Single-image classifier worker."""
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import TypeVar, cast

from omegaconf import OmegaConf


@dataclass
class DataConf:
    image_dir: Path = Path("./images")
    image_id: int = 0


@dataclass
class ModelConf:
    model_name: str = "resnet18"
    checkpoint: Path | None = None


@dataclass
class WorkerConf:
    data: DataConf = field(default_factory=DataConf)
    model: ModelConf = field(default_factory=ModelConf)
    output_path: Path = Path("./outputs")


T = TypeVar("T")


def load_config(Param: type[T]) -> T:
    defaults = OmegaConf.structured(Param())
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.unsafe_merge(defaults, cli_cfg)
    return cast(T, OmegaConf.to_object(cfg))


def classify_image(cfg: WorkerConf) -> dict:
    """Classify a single image."""
    # Simulated classification
    return {
        "image_id": cfg.data.image_id,
        "model": cfg.model.model_name,
        "predicted_class": "cat",
        "confidence": 0.95,
    }


def main():
    cfg = load_config(WorkerConf)
    
    result = classify_image(cfg)
    
    output_file = cfg.output_path / f"pred_{cfg.data.image_id:05d}.json"
    output_file.write_text(json.dumps(result, indent=2))
    print(f"Processed image {cfg.data.image_id}")


if __name__ == "__main__":
    main()
```

```python
# experiments/image_classifier/launcher.py
"""Launcher for distributed image classification."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from btorch.utils.conf import load_config, to_dotlist
from worker import WorkerConf


@dataclass
class LauncherConf:
    """Launcher configuration."""
    # Single-worker config to forward
    worker: WorkerConf = field(default_factory=WorkerConf)
    
    # Launcher-specific (not forwarded)
    image_ids: list[int] | None = None
    image_range: str | None = None  # "0:100" syntax
    max_workers: int = 4
    
    def get_image_ids(self) -> list[int]:
        if self.image_ids is not None:
            return self.image_ids
        if self.image_range is not None:
            start, end = map(int, self.image_range.split(":"))
            return list(range(start, end))
        return [0]


def run_single(cmd: list[str]) -> tuple[int, str]:
    """Run worker and return (image_id, status)."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract image_id from cmd
    id_arg = [a for a in cmd if "data.image_id=" in a][0]
    image_id = int(id_arg.split("=")[1])
    
    status = "success" if result.returncode == 0 else "failed"
    return image_id, status


def main():
    cfg, cli_cfg = cast(
        tuple[LauncherConf, Any],
        load_config(LauncherConf, return_cli=True),
    )
    
    image_ids = cfg.get_image_ids()
    print(f"Classifying {len(image_ids)} images")
    
    # Extract worker config overrides (excluding per-image fields)
    worker_dotlist = to_dotlist(
        cli_cfg.worker,
        use_equal=True,
        exclude={"data.image_id"},
    )
    
    # Build base command
    base_cmd = [sys.executable, "-m", "worker"] + worker_dotlist
    
    # Launch workers
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {
            executor.submit(
                run_single,
                base_cmd + [f"data.image_id={img_id}"]
            ): img_id
            for img_id in image_ids
        }
        
        for future in as_completed(futures):
            img_id = futures[future]
            try:
                _, status = future.result()
                if status == "success":
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Image {img_id} failed: {e}")
                failed += 1
            
            if (completed + failed) % 10 == 0:
                print(f"Progress: {completed} done, {failed} failed")
    
    print(f"Complete: {completed} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
```

## CLI Usage Examples

```bash
# Run single worker
python -m worker data.image_id=42 model.model_name=resnet50

# Run launcher with option forwarding
python -m launcher \
    worker.model.model_name=resnet50 \
    worker.model.checkpoint=/path/to/checkpoint.pth \
    image_range=0:1000 \
    max_workers=8

# Parameter sweep
python sweep.py \
    sweep_params=[param_a,param_c] \
    candidates_a=[0.1,0.5,1.0] \
    candidates_c=[small,large] \
    output_path=./sweep_results
```

## Key Takeaways

1. **Composition**: Split into CommonConf (shared) + TaskConf (specific)
2. **Forwarding**: Use `to_dotlist(cfg, exclude={"field"})` to pass options to workers
3. **Exclude per-item fields**: Always exclude ID fields from forwarded options
4. **Deepcopy for sweeps**: `deepcopy(base)` before modifying for each trial
5. **Candidates in code**: Define sweep ranges as dataclass fields with defaults
