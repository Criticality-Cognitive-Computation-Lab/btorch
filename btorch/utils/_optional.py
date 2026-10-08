"""Policy and helper for optional dependencies.

Every third-party library that is not a core dependency is declared in a
``btorch[<extra>]`` extra in ``pyproject.toml`` and follows exactly one of two
patterns:

(a) **Hard requirement** -- the feature cannot work without the library
    (xarray/zarr serialization, HDF5/YAML/OmegaConf helpers,
    powerlaw/nolds/fastdtw analysis, plotly plots, Triton GPU timing).
    Import it lazily with :func:`require`, which raises ``ImportError`` with a
    ``pip install "btorch[<extra>]"`` hint. Importing btorch itself never
    needs the library; a module that is *entirely* about the library (e.g.
    :mod:`btorch.utils.hdf5_utils`) may call :func:`require` at module scope.

(b) **Accelerator with a correct fallback** -- results are identical (up to
    numerical precision) without the library, only slower
    (``numba`` in :mod:`btorch.config`, ``polars`` in
    :func:`btorch.analysis.aggregation.aggregate_by_neuropil`,
    ``torch_sparse`` in :mod:`btorch.sparse.runtime`). Use a try-import
    (module scope or lazy), document the fallback where it happens and never
    produce a silently different result. These libraries are still declared
    in an extra (``fast``, ``sparse``) so users can opt in.

Mapping of extras: ``io`` (xarray, zarr, numcodecs, h5py, hdf5plugin),
``config`` (omegaconf, pyyaml), ``analysis`` (powerlaw, nolds, fastdtw),
``viz`` (networkx, plotly), ``sparse`` (torch_scatter, torch_sparse),
``fast`` (numba, polars), ``gpu`` (triton), ``examples`` (torchvision,
seaborn, tqdm) and ``all`` (every extra above). AllenSDK is
deliberately in no extra: its release pins an ancient numpy/pandas stack that
the resolver cannot satisfy next to btorch, so install it separately
(``pip install allensdk``) and keep its guard a plain ``ImportError`` hint.
"""

import importlib
from types import ModuleType


def require(module_name: str, extra: str, purpose: str | None = None) -> ModuleType:
    """Import an optional dependency, or raise ImportError with an install
    hint.

    Args:
        module_name: Dotted module path to import (e.g. ``"plotly.graph_objects"``).
        extra: Name of the btorch extra that provides the module (e.g. ``"viz"``).
        purpose: Optional short description of the feature needing the module.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module is not installed.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        what = f" for {purpose}" if purpose else ""
        raise ImportError(
            f"'{module_name.split('.')[0]}' is required{what} but is not "
            f'installed. Install it with: pip install "btorch[{extra}]"'
        ) from e
