import inspect
from dataclasses import dataclass
from pathlib import Path

from btorch.utils import conf


@dataclass(frozen=True)
class FigPathConfig:
    root_dir: str = "fig"
    benchmark_dir: str = "benchmark"
    tests_dir: str = "tests"
    other_dir: str = "misc"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def caller_file(skip: int = 2) -> Path:
    frame = inspect.stack()[skip]
    return Path(frame.filename).resolve()


def _resolve_cfg(cfg: FigPathConfig | dict | conf.DictConfig | None):
    defaults = conf.OmegaConf.structured(FigPathConfig)
    if cfg is None:
        return defaults
    if isinstance(cfg, FigPathConfig):
        cfg = conf.OmegaConf.structured(cfg)
    elif isinstance(cfg, dict):
        cfg = conf.OmegaConf.create(cfg)
    return conf.OmegaConf.merge(defaults, cfg)


def fig_path(file: str | Path | None = None, cfg: FigPathConfig | dict | None = None):
    file_path = Path(file) if file is not None else caller_file()
    file_path = file_path.resolve()
    root = _repo_root()
    cfg = _resolve_cfg(cfg)

    benchmark_roots = [root / "benchmark", root / "tests" / "benchmark"]
    for bench_root in benchmark_roots:
        if _is_relative_to(file_path, bench_root):
            rel = file_path.relative_to(bench_root)
            path = root / cfg.root_dir / cfg.benchmark_dir / rel.with_suffix("")
            path.mkdir(parents=True, exist_ok=True)
            return path

    tests_root = root / "tests"
    if _is_relative_to(file_path, tests_root):
        rel = file_path.relative_to(tests_root)
        path = root / cfg.root_dir / cfg.tests_dir / rel.with_suffix("")
        path.mkdir(parents=True, exist_ok=True)
        return path

    if _is_relative_to(file_path, root):
        rel = file_path.relative_to(root)
    else:
        rel = Path(file_path.name)
    path = root / cfg.root_dir / cfg.other_dir / rel.with_suffix("")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(
    fig,
    name: str | None = None,
    path: Path | None = None,
    *,
    file: str | Path | None = None,
    cfg: FigPathConfig | dict | None = None,
    suffix: str = "pdf",
    transparent: bool = False,
) -> Path:
    file_path = Path(file) if file is not None else caller_file()
    if path is None:
        path = fig_path(file_path, cfg=cfg)
    if name is None:
        name = file_path.stem
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / f"{name}.{suffix}"
    fig.savefig(output_path.as_posix(), transparent=transparent)
    return output_path
