"""Sparse execution runtime: planner, caches, backends and registered ops.

Nothing in this package is needed to define a model. It is the layer below
:mod:`btorch.models.connection` that turns canonical tensor buffers into
executed kernels, and the place where additional backends are registered.
"""

from . import kernels_aten, ops
from .backend import BackendRegistry, KernelCache, registry, use_backend
from .cache import RepresentationCache
from .ops import csr_propagate, pack_spikes, spike_propagate
from .planner import Planner, planner


__all__ = [
    "BackendRegistry",
    "KernelCache",
    "Planner",
    "RepresentationCache",
    "csr_propagate",
    "kernels_aten",
    "ops",
    "pack_spikes",
    "planner",
    "registry",
    "spike_propagate",
    "use_backend",
]
