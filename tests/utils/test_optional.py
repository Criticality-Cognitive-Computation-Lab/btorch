"""Tests for :mod:`btorch.utils._optional` (the optional-dependency policy)."""

import re
from pathlib import Path

import pytest

from btorch.utils import _optional
from btorch.utils._optional import require


tomllib = pytest.importorskip("tomllib", reason="tomllib needs Python 3.11+")

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_require_returns_the_imported_module():
    import json

    assert require("json", "config") is json


def test_require_imports_dotted_submodules():
    import os.path

    assert require("os.path", "io") is os.path


def test_require_missing_module_raises_install_hint():
    """The message names the top-level package and the extra to install."""
    with pytest.raises(ImportError) as exc:
        require("btorch_no_such_pkg.sub", "viz")
    msg = str(exc.value)
    assert "'btorch_no_such_pkg'" in msg
    assert 'pip install "btorch[viz]"' in msg
    # Without ``purpose`` there is no "for ..." clause.
    assert " for " not in msg.split("but is not")[0]
    # The original ImportError stays attached for debugging.
    assert isinstance(exc.value.__cause__, ImportError)


def test_require_message_includes_purpose():
    with pytest.raises(ImportError, match="required for plotting widgets"):
        require("btorch_no_such_pkg", "viz", "plotting widgets")


def _pyproject_extras() -> dict[str, list[str]]:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["optional-dependencies"]


def test_policy_docstring_lists_exactly_the_real_extras():
    """The extras documented in ``_optional`` must match ``pyproject.toml``."""
    doc = _optional.__doc__
    # Only the "Mapping of extras" paragraph (up to the AllenSDK remark) lists
    # extras as ``name`` tokens; package names there are in parentheses.
    mapping = doc[doc.index("Mapping of extras") : doc.index("AllenSDK")]
    documented = set(re.findall(r"``(\w+)``", mapping))
    assert documented == set(_pyproject_extras())


def test_policy_docstring_mentions_the_install_hint_format():
    assert 'pip install "btorch[<extra>]"' in _optional.__doc__
