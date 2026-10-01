"""Check that heavy optional dependencies are truly optional.

The dev environment has every extra installed, so the only reliable way to verify
the lazy-import behaviour is to simulate a missing library in a fresh
interpreter: setting ``sys.modules[name] = None`` makes ``import name`` raise
``ImportError`` exactly as if the package were not installed.

Two properties are verified for each optional library:

1. Importing the public btorch packages still works without it.
2. Calling a function that needs it raises an ``ImportError`` carrying an
   actionable ``pip install "btorch[<extra>]"`` hint.
"""

import subprocess
import sys
import textwrap

import pytest


# library -> (extra that provides it, snippet that must raise ImportError)
OPTIONAL_LIBS = {
    "plotly": (
        "viz",
        "from btorch.visualisation.hex.interactive import grid; grid(2)",
    ),
    "xarray": (
        "io",
        "import torch\n"
        "from btorch.io.serialization import DimLayout, memories_to_xarray\n"
        "memories_to_xarray({'v': torch.zeros(3, 2, 4)},\n"
        "                   DimLayout(dim_counts=(1, 1, 1)))",
    ),
    "zarr": (
        "io",
        "import torch\n"
        "from btorch.io.serialization import DimLayout, save_memories_to_xarray\n"
        "save_memories_to_xarray({'v': torch.zeros(3, 2, 4)}, 'x.zarr',\n"
        "                        DimLayout(dim_counts=(1, 1, 1)))",
    ),
    # numcodecs is only a Zarr v2 codec provider; with Zarr v3 installed its
    # absence is harmless, so only the import-time property is checked.
    "numcodecs": ("io", None),
    "powerlaw": (
        "analysis",
        "import numpy as np\n"
        "from btorch.analysis.dynamic_tools.criticality import _fit_distribution\n"
        "_fit_distribution(np.arange(1, 50))",
    ),
    "nolds": (
        "analysis",
        "import numpy as np\n"
        "from btorch.analysis.dynamic_tools.criticality import compute_dfa\n"
        "compute_dfa(np.random.rand(500))",
    ),
    "fastdtw": (
        "analysis",
        "import numpy as np\n"
        "from btorch.analysis.clustering import cluster_traces\n"
        "cluster_traces([np.zeros(5), np.ones(5)])",
    ),
}

# Modules whose whole purpose is one optional library: importing them (not just
# calling a function) must raise the install hint.
OPTIONAL_LIBS.update(
    {
        "h5py": ("io", "import btorch.utils.hdf5_utils"),
        "hdf5plugin": ("io", "import btorch.utils.hdf5_utils"),
        "omegaconf": ("config", "import btorch.utils.conf"),
        "yaml": ("config", "import btorch.utils.yaml_utils"),
    }
)

# Accelerators with a correct fallback: only the import-time property applies.
OPTIONAL_LIBS.update(
    {"numba": ("fast", None), "polars": ("fast", None), "triton": ("gpu", None)}
)

PUBLIC_MODULES = [
    "btorch.utils",
    "btorch.utils.file",
    "btorch.models",
    "btorch",
    "btorch.io",
    "btorch.analysis",
    "btorch.analysis.dynamic_tools",
    "btorch.visualisation",
    "btorch.visualisation.hex",
]

_BLOCK = "import sys\nsys.modules[{lib!r}] = None\n"


def _run(lib: str, body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter where ``lib`` cannot be imported."""
    code = _BLOCK.format(lib=lib) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )


@pytest.mark.parametrize("lib", sorted(OPTIONAL_LIBS))
def test_public_imports_without_optional_lib(lib):
    """Public btorch modules import fine when ``lib`` is missing."""
    body = "\n".join(f"import {m}" for m in PUBLIC_MODULES)
    res = _run(lib, body)
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize(
    "lib", [k for k, (_, snippet) in sorted(OPTIONAL_LIBS.items()) if snippet]
)
def test_missing_lib_gives_install_hint(lib):
    """Using a feature that needs ``lib`` raises ImportError with a pip
    hint."""
    extra, snippet = OPTIONAL_LIBS[lib]
    body = (
        "try:\n"
        + textwrap.indent(textwrap.dedent(snippet), "    ")
        + "\nexcept ImportError as e:\n"
        "    print('IMPORT_ERROR:', e)\n"
        "else:\n"
        "    raise SystemExit('no ImportError raised')\n"
    )
    res = _run(lib, body)
    assert res.returncode == 0, res.stderr + res.stdout
    msg = res.stdout.split("IMPORT_ERROR:", 1)[1]
    assert "pip install" in msg
    assert f"btorch[{extra}]" in msg
    assert lib in msg


def test_fig_path_works_without_omegaconf():
    """``fig_path``/``save_fig`` config handling needs no OmegaConf."""
    body = """
    from btorch.utils.file import FigPathConfig, _resolve_cfg
    cfg = _resolve_cfg({"root_dir": "figx"})
    assert cfg == FigPathConfig(root_dir="figx")
    assert _resolve_cfg(None) == FigPathConfig()
    """
    res = _run("omegaconf", body)
    assert res.returncode == 0, res.stderr


def test_public_imports_without_any_optional_lib():
    """Public btorch modules import fine with ALL optional libs missing."""
    block = "".join(f"sys.modules[{lib!r}] = None\n" for lib in OPTIONAL_LIBS)
    body = "\n".join(f"import {m}" for m in PUBLIC_MODULES)
    res = subprocess.run(
        [sys.executable, "-c", "import sys\n" + block + body],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert res.returncode == 0, res.stderr
