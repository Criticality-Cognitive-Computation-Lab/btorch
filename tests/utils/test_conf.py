import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from btorch.utils.conf import (
    diff_conf,
    diff_conf_records,
    load_config,
    to_dotlist,
)


omegaconf = pytest.importorskip("omegaconf")
from omegaconf import OmegaConf  # noqa: E402


@dataclass
class CommonConf:
    # Shared arguments used by every single run (mirrors single.py CommonConf).
    # Think: one specimen, one output folder.
    ephys_dataset_path: str = "dataset.nc"
    id: int = 0
    get_size: bool = False
    overwrite: bool = False
    output_path: Path = Path("outputs")

    def ensure_path(self):
        self.output_path.mkdir(parents=True, exist_ok=True)


@dataclass
class SolverConf:
    # Solver-specific knobs that ride along with the single-task config.
    k_candidates: list[int] = field(default_factory=lambda: [1, 2, 3])
    no_excess_zero_fr: bool = False
    plot_fi: bool = False
    plot_filename: str = "fi.png"


@dataclass
class SingleArgConf:
    # Mimics the per-specimen program in single.py.
    # This is what each worker process receives.
    common: CommonConf = field(default_factory=CommonConf)
    solver: SolverConf = field(default_factory=SolverConf)


@dataclass
class BatchArgConf:
    # Manages the whole sweep; batch.single is forwarded to each worker process.
    # batch.py wraps this, decides which specimen IDs to run, and spawns workers.
    single: SingleArgConf = field(default_factory=SingleArgConf)
    id: list[int] | None = None
    id_select: str | None = None
    timeout: int = 60 * 4 * 10
    ncpu_per_task: int = 1
    max_concurrent: int = 64


@dataclass
class UnionOptA:
    value: int = 1
    legacy: str = "old"


@dataclass
class UnionOptB:
    value: int = 1
    tag: str = "b"


@dataclass
class StructuredUnionConf:
    item: UnionOptA | UnionOptB = field(default_factory=UnionOptA)


@dataclass
class PrimitiveUnionConf:
    item: int | str = 1


def test_load_config_merges_file_and_cli(tmp_path, monkeypatch):
    """Mimics the single.py usage: config file + CLI overrides."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "\n".join(
            [
                "common:",
                "  id: 5",
                "  overwrite: true",
                "  output_path: results",
                "solver:",
                "  plot_fi: true",
                "  plot_filename: from_file.png",
            ]
        )
    )

    # Equivalent CLI call the user would make:
    # python single.py config_path=cfg_path common.id=7 \
    #   solver.k_candidates=[9,10] solver.plot_filename=cli.png

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            f"config_path={cfg_file}",
            "common.id=7",
            "solver.k_candidates=[9,10]",
            "solver.plot_filename=cli.png",
        ],
    )

    cfg = load_config(SingleArgConf)

    assert isinstance(cfg, SingleArgConf)
    assert cfg.common.id == 7  # CLI wins over file/defaults
    assert cfg.common.overwrite is True  # from file
    assert cfg.common.output_path == Path("results")
    assert cfg.solver.k_candidates == [9, 10]  # CLI list parsing
    assert cfg.solver.plot_fi is True  # from file
    assert cfg.solver.plot_filename == "cli.png"


def test_to_dotlist_respects_exclusions(monkeypatch):
    """Mimics the batch.py usage where CLI is forwarded to worker tasks."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "single.common.id=3",
            "single.common.get_size=True",
            "single.common.output_path=/tmp/run",
            "single.solver.plot_fi=True",
            "single.solver.plot_filename=cli.png",
        ],
    )

    args, args_cli = load_config(BatchArgConf, return_cli=True)
    args: BatchArgConf

    assert args.single.common.id == 3
    assert args.single.common.output_path == Path("/tmp/run")
    assert args.single.solver.plot_fi is True

    # batch.py strips off id/get_size (set per worker) before forwarding to children
    dotlist = to_dotlist(
        args_cli.single, use_equal=True, exclude={"common.get_size", "common.id"}
    )

    assert "common.id=3" not in dotlist
    assert "common.get_size=True" not in dotlist
    assert set(dotlist) == {
        "common.output_path=/tmp/run",
        "solver.plot_fi=True",
        "solver.plot_filename=cli.png",
    }

    # Subfield traversal should shift path roots, then include/exclude still works.
    solver_only = to_dotlist(
        args_cli.single,
        use_equal=True,
        subfield="solver",
        include={"plot_fi", "plot_filename"},
        exclude={"plot_filename"},
    )
    assert solver_only == ["plot_fi=True"]

    # Nested list traversal with include/exclude at the selected subfield root.
    nested = OmegaConf.create({"a": {"b": [{"c": 1}, {"c": 2}]}})
    assert to_dotlist(nested, subfield="a.b.1", include={"c"}) == ["c=2"]
    assert to_dotlist(nested, subfield="a.b.1", exclude={"c"}) == []


def test_batch_worker_command_build(monkeypatch, tmp_path):
    """End-to-end shape of batch: config file -> batch.single -> worker arg list."""
    cfg_file = tmp_path / "batch.yaml"
    cfg_file.write_text(
        "\n".join(
            [
                "single:",
                "  common:",
                "    output_path: /data/exp",
                "  solver:",
                "    k_candidates: [4,5]",
            ]
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            f"config_path={cfg_file}",
            "single.common.overwrite=True",
            "single.common.id=12",
            "single.solver.plot_fi=True",
        ],
    )

    # BatchArgConf comes from CLI+file; args_cli is the raw CLI OmegaConf
    args, args_cli = load_config(BatchArgConf, return_cli=True)

    # Pretend batch.py builds the per-worker CLI, filtering id/get_size then
    # appending new id
    base_worker_cli = to_dotlist(
        args_cli.single, use_equal=True, exclude={"common.get_size", "common.id"}
    )
    worker_cmd = base_worker_cli + [f"common.id={99}"]

    assert args.single.common.output_path == Path("/data/exp")
    assert args.single.common.overwrite is True
    assert args.single.solver.k_candidates == [4, 5]
    assert args.single.solver.plot_fi is True

    # to_dotlist should carry everything except filtered keys; caller appends target id
    assert "common.id=12" not in worker_cmd
    assert "common.get_size=True" not in worker_cmd
    # Only CLI-provided keys are forwarded; config-file/default values stay in
    # the worker defaults
    assert set(worker_cmd) == {
        "common.overwrite=True",
        "solver.plot_fi=True",
        "common.id=99",
    }


def test_to_dotlist_subfield_missing_policy():
    cfg = OmegaConf.create({"a": {"b": 1}})

    with pytest.raises(KeyError):
        to_dotlist(cfg, subfield="x.y")

    assert to_dotlist(cfg, subfield="x.y", missing_subfield_policy="empty") == []


def test_diff_conf_all_detects_nested_changed_added_removed():
    """Diff config should keep only changed/added/removed leaf paths.

    This test models realistic config edits where a user:
    - changes existing scalar values,
    - adds new fields,
    - removes previous fields,
    - updates list members and extends lists,
    - replaces a full subtree with a scalar.
    """

    conf_a = OmegaConf.create(
        {
            "model": {"hidden": 128, "dropout": 0.1},
            "data": {"path": "/dataset", "batch_size": 16},
            "tags": ["baseline", "v1"],
            "scheduler": {"type": "cosine"},
        }
    )

    conf_b = OmegaConf.create(
        {
            "model": {"hidden": 256, "activation": "gelu"},
            "tags": ["baseline", "v2", "extra"],
            "scheduler": "disabled",
        }
    )

    # Compare B to A and include changed + added + removed keys.
    diff_cfg = diff_conf(conf_a, conf_b)

    # model.hidden changed, model.activation added, model.dropout removed
    # data subtree removed entirely -> each removed leaf path should be listed
    # tags.1 changed and tags.2 added
    # scheduler switched dict -> scalar, reported at the subtree root path
    assert OmegaConf.to_container(diff_cfg) == {
        "model": {
            "hidden": 256,
            "activation": "gelu",
            "dropout": None,
        },
        "data": {"path": None, "batch_size": None},
        "tags": [None, "v2", "extra"],
        "scheduler": "disabled",
    }


def test_diff_conf_mode_filters_are_intentional_and_directional():
    """Each mode should expose only requested categories with concrete values.

    A is the baseline, B is the edited configuration. The assertions
    ensure mode selection is meaningful for downstream workflows that
    may only care about additions/overrides vs removals.
    """

    conf_a = OmegaConf.create(
        {
            "seed": 1,
            "optim": {"lr": 1e-3, "weight_decay": 0.0},
            "log_interval": 10,
        }
    )
    conf_b = OmegaConf.create(
        {
            "seed": 2,  # changed
            "optim": {"lr": 5e-4},  # changed + removed weight_decay
            "project": "exp-42",  # added
        }
    )

    assert OmegaConf.to_container(diff_conf(conf_a, conf_b, mode={"changed"})) == {
        "seed": 2,
        "optim": {"lr": 5e-4},
    }

    assert OmegaConf.to_container(
        diff_conf(conf_a, conf_b, mode={"added", "changed"})
    ) == {
        "seed": 2,
        "optim": {"lr": 5e-4},
        "project": "exp-42",
    }

    assert OmegaConf.to_container(
        diff_conf(conf_a, conf_b, mode={"removed", "changed"})
    ) == {
        "seed": 2,
        "optim": {"lr": 5e-4, "weight_decay": None},
        "log_interval": None,
    }

    assert OmegaConf.to_container(diff_conf(conf_a, conf_b)) == {
        "seed": 2,
        "optim": {"lr": 5e-4, "weight_decay": None},
        "project": "exp-42",
        "log_interval": None,
    }


def test_diff_conf_accepts_plain_inputs_via_structured_conversion():
    """diff_conf should accept plain objects by structuring them internally."""

    conf = OmegaConf.create({"a": 1})

    assert OmegaConf.to_container(diff_conf({"a": 1}, conf)) == {}
    assert OmegaConf.to_container(diff_conf(conf, {"a": 1})) == {}


def test_diff_conf_records_captures_status_and_old_new_values_for_spawning():
    """Value-level diff should be directly usable by parent/child launch logic.

    The parent owns baseline A. It prepares child B by overriding a
    subset of keys. The diff output must preserve both the operation
    type and concrete values so the launcher can decide whether to pass
    key=value or remove keys.
    """

    conf_a = OmegaConf.create(
        {
            "single": {
                "common": {"id": 0, "overwrite": False, "tag": "baseline"},
                "solver": {"plot_fi": False, "plot_filename": "fi.png"},
            }
        }
    )
    conf_b = OmegaConf.create(
        {
            "single": {
                "common": {"id": 42, "overwrite": True},
                "solver": {"plot_fi": True, "plot_filename": "run42.png"},
            }
        }
    )

    records = diff_conf_records(conf_a, conf_b)

    assert records["single.common.id"] == {
        "status": "changed",
        "old": 0,
        "new": 42,
    }
    assert records["single.common.overwrite"] == {
        "status": "changed",
        "old": False,
        "new": True,
    }
    assert records["single.solver.plot_fi"] == {
        "status": "changed",
        "old": False,
        "new": True,
    }
    assert records["single.solver.plot_filename"] == {
        "status": "changed",
        "old": "fi.png",
        "new": "run42.png",
    }
    assert records["single.common.tag"] == {
        "status": "removed",
        "old": "baseline",
        "new": None,
    }


def test_diff_conf_integrates_with_to_dotlist_for_worker_overrides():
    """Master can call diff_conf then to_dotlist to build child CLI arguments.

    This mirrors the practical workflow where a master script computes a
    minimal child argument list and launches workers with only changed
    keys.
    """

    conf_a = OmegaConf.create(
        {
            "single": {
                "common": {"id": 0, "overwrite": False, "tag": "baseline"},
                "solver": {"plot_fi": False, "plot_filename": "fi.png"},
            }
        }
    )
    conf_b = OmegaConf.create(
        {
            "single": {
                "common": {"id": 42, "overwrite": True},
                "solver": {"plot_fi": True, "plot_filename": "run42.png"},
            }
        }
    )

    diff_cfg = diff_conf(conf_a, conf_b)
    child_cli = to_dotlist(diff_cfg, use_equal=True)

    assert set(child_cli) == {
        "single.common.id=42",
        "single.common.overwrite=True",
        "single.solver.plot_fi=True",
        "single.solver.plot_filename=run42.png",
        "single.common.tag=null",
    }


def test_diff_conf_records_supports_structured_union_type_switch():
    """Structured union switch should be a full subtree replacement."""

    conf_a = OmegaConf.structured(StructuredUnionConf(item=UnionOptA(value=7)))
    conf_b = OmegaConf.structured(
        StructuredUnionConf(item=UnionOptB(value=7, tag="new"))
    )

    records = diff_conf_records(conf_a, conf_b)

    assert records["item"]["status"] == "changed"
    assert "UnionOptA" in str(records["item"]["old"])
    assert "UnionOptB" in str(records["item"]["new"])
    assert "item.legacy" not in records


def test_diff_conf_supports_union_values_for_structured_and_primitives():
    """diff_conf should replace a switched union subtree and drop old keys."""

    conf_a = OmegaConf.structured(StructuredUnionConf(item=UnionOptA(value=7)))
    conf_b = OmegaConf.structured(
        StructuredUnionConf(item=UnionOptB(value=7, tag="new"))
    )

    diff_structured = OmegaConf.to_container(diff_conf(conf_a, conf_b))
    assert str(diff_structured["item"]["_type_"]).endswith("UnionOptB")
    assert diff_structured["item"]["tag"] == "new"
    assert "legacy" not in diff_structured["item"]

    prim_a = OmegaConf.structured(PrimitiveUnionConf(item=1))
    prim_b = OmegaConf.structured(PrimitiveUnionConf(item="1"))
    assert OmegaConf.to_container(diff_conf(prim_a, prim_b)) == {"item": "1"}
