"""``btorch.__version__`` comes from the installed metadata, with a safe
fallback."""

import importlib.metadata
import subprocess
import sys
import textwrap

import pytest

import btorch


def test_version_matches_installed_metadata():
    """In an installed environment ``__version__`` is the distribution
    version."""
    try:
        installed = importlib.metadata.version("btorch")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("btorch is not installed in this environment")
    assert btorch.__version__ == installed


def test_import_survives_missing_package_metadata():
    """``import btorch`` must not fail when no installed metadata exists.

    The version is dynamic (derived from git tags at build time), so a source
    checkout that was never installed has no metadata. A fresh interpreter makes
    ``importlib.metadata.version`` raise ``PackageNotFoundError`` before btorch is
    imported; the import must still work and report xarray's "9999" sentinel.
    """
    code = textwrap.dedent(
        """
        import importlib.metadata as metadata

        def missing(name):
            raise metadata.PackageNotFoundError(name)

        metadata.version = missing  # simulate an uninstalled source tree
        import btorch

        print(btorch.__version__)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "9999"
