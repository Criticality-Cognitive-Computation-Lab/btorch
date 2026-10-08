"""Consistency checks for the extras declared in ``pyproject.toml``."""

import re
from pathlib import Path

import pytest


tomllib = pytest.importorskip("tomllib", reason="tomllib needs Python 3.11+")

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
EXTRAS_REQUIRING_EXTERNAL_BINARY_COMPATIBILITY = {"sparse"}


def _extras() -> dict[str, list[str]]:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["optional-dependencies"]


def test_all_extra_references_every_other_extra():
    """``all`` must be a self-reference to the other extras, never a copy.

    A hand-maintained duplicate list can silently drift; a single
    ``btorch[a,b,...]`` requirement cannot.
    """
    extras = _extras()
    (spec,) = extras["all"]
    match = re.fullmatch(r"btorch\[([\w,\- ]+)\]", spec)
    assert match, f"'all' should be one self-extra requirement, got {spec!r}"
    referenced = {name.strip() for name in match.group(1).split(",")}
    assert referenced == (
        set(extras) - {"all"} - EXTRAS_REQUIRING_EXTERNAL_BINARY_COMPATIBILITY
    )


def test_examples_extra_covers_example_imports():
    """Every third-party import used by ``examples/`` is declared in an
    extra."""
    extras = {re.split(r"[<>=!;\[ ]", r)[0] for r in _extras()["examples"]}
    assert {"torchvision", "seaborn", "tqdm"} <= extras
