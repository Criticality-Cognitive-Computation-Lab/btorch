"""Prepare source-major fanout topology for push kernels."""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import torch
from torch import Tensor

from . import csr_layout, kernels_torch
from .backend import KernelCache


_HUB_DEGREE = 2048
_HUB_CHUNK = 1024
_INT32_MAX = 2**31 - 1
_MAX_LAYOUTS = 1024
EDGE_BLOCK = 256
BLOCK_WARPS = 4


@dataclass
class _Layout:
    """Kernel-side copies of one source-major CSR and how to validate them.

    Every tensor is allocated once and rewritten in place by :meth:`fill`,
    and the launch arguments derived from the layout (``n_hub``,
    ``hub_limit``) do not follow the current pattern, so a CUDA graph
    captured around the dense kernel stays valid across rewiring:

    - A layout that never saw a hub (a source with more than ``hub_degree``
      out-edges) launches no hub tasks and passes ``hub_limit = int32 max``:
      the tiles serve every source whatever its degree. A graph captured in
      this state stays correct when a later rewiring creates a hub (the hub
      is then served by one program until the graph is captured again).
    - From the first hub on, and for the rest of the layout's life, the task
      list has a fixed capacity derived from ``E`` alone: a hub of degree
      ``d`` has ``ceil(d / hub_chunk) <= d // hub_chunk + 1`` tasks and at
      most ``E // (hub_degree + 1)`` hubs exist, hence at most
      ``E // hub_chunk + E // (hub_degree + 1)`` tasks. The grid always
      covers the capacity; unused slots are empty edge ranges.
    """

    sources: tuple[weakref.ref, ...]
    versions: tuple[int, ...]
    hub_degree: int
    hub_chunk: int
    crow: Tensor  # int32 [N + 1]
    col: Tensor  # int32 [E]
    perm: Tensor  # int32 [E]
    degree: Tensor  # int64 [N] (indexed by int64 ``active_idx``)
    hub: Tensor  # int32 [3 * max(n_hub, 1)]: (source, first edge, end edge)
    n_hub: int  # hub task slots in the launch grid (0 or the capacity)
    hub_limit: int  # degree above which a tile leaves a source to hub tasks
    mean_degree: float
    epoch: int = 0  # stamp of the last allocation (see ``layout_epoch``)
    n_fill: int = 0  # times the buffers were (re)derived
    source_values: Tensor | None = None  # optional spike-list read layout
    source_value_epoch: int = 0

    def fill(self, t_crow: Tensor, t_col: Tensor, t_perm: Tensor) -> None:
        """(Re)derive every buffer from the canonical tensors, in place.

        Synchronises with the host (once per topology, never per call).
        """
        self.crow.copy_(t_crow)
        self.col.copy_(t_col)
        self.perm.copy_(t_perm)
        torch.sub(t_crow[1:], t_crow[:-1], out=self.degree)
        tasks = _hub_tasks(t_crow, self.degree, self.hub_degree, self.hub_chunk)
        if tasks is not None and not self.n_hub:
            # First hub: switch to the fixed-capacity task list for good.
            n_edge = self.col.shape[0]
            self.n_hub = n_edge // self.hub_chunk + n_edge // (self.hub_degree + 1)
            self.hub = torch.empty(
                3 * self.n_hub, dtype=torch.int32, device=self.col.device
            )
            self.hub_limit = self.hub_degree
            self.epoch = csr_layout._next_epoch()
        if self.n_hub:
            used = 0 if tasks is None else tasks.shape[0]
            if used:
                self.hub[:used].copy_(tasks)
            self.hub[used:] = 0
        self.n_fill += 1
        self.versions = (
            t_crow._version,
            t_col._version,
            t_perm._version,
            t_crow.data_ptr(),
            t_col.data_ptr(),
            t_perm.data_ptr(),
        )


def _hub_tasks(
    t_crow: Tensor, degree: Tensor, hub_degree: int, hub_chunk: int
) -> Tensor | None:
    """Split the out-edges of every hub into ``hub_chunk``-sized tasks.

    Returns the flat ``int32 [3 * H]`` list of ``(source, first edge, end
    edge)`` per task, or ``None`` without hubs. Built once per topology with
    vectorised operations.
    """
    hubs = (degree > hub_degree).nonzero(as_tuple=True)[0]
    if hubs.shape[0] == 0:
        return None
    n_chunks = (degree[hubs] + hub_chunk - 1) // hub_chunk
    owner = torch.repeat_interleave(
        torch.arange(hubs.shape[0], device=hubs.device), n_chunks
    )
    first_task = torch.cumsum(n_chunks, 0) - n_chunks
    local = torch.arange(owner.shape[0], device=hubs.device) - first_task[owner]
    source = hubs[owner]
    lo = t_crow[source] + local * hub_chunk
    hi = torch.minimum(lo + hub_chunk, t_crow[source + 1])
    return torch.stack([source, lo, hi], dim=1).to(torch.int32).reshape(-1)


def _layout(
    cache: KernelCache, t_crow: Tensor, t_col: Tensor, t_perm: Tensor
) -> _Layout | None:
    """Return the memoised kernel layout of a source-major CSR.

    The canonical buffers of a
    :class:`~btorch.sparse.runtime.cache.RepresentationCache` are rewritten
    *in place* on rewiring: same tensor object and address, new contents.
    Identity or address alone therefore cannot key the derived copies. The
    rule used here:

    - A holder is looked up by the ``id`` of the three tensor objects (one
      dictionary lookup; two views of one buffer are different objects and
      never share a holder).
    - A hit is valid only if the holder's weak references still resolve to
      the very tensor objects passed in *and* their ``_version`` counters are
      unchanged. Every in-place write (``copy_`` on rewiring) bumps
      ``_version``, and a live weak reference proves the object was never
      released, so its ``id`` cannot belong to a different tensor.
    - On a stale hit the derived buffers are refreshed in place (mirroring
      the cache contract: tensors captured by a compiled or CUDA graph stay
      valid). The refresh synchronises with the host; :func:`prepare` does
      it ahead of time.
    - A holder is deleted when one of its source tensors dies, which
      releases its device memory, and at most ``_MAX_LAYOUTS`` are kept, so
      the cache cannot grow without bound.

    Returns ``None`` when the layout cannot be tracked or represented:
    inference tensors carry no version counter, and int32 indices require
    ``E`` and ``N`` below ``2**31``.
    """
    n_edge, n_src = t_col.shape[0], t_crow.shape[0] - 1
    device = t_col.device
    if (
        n_edge > _INT32_MAX
        or n_src >= _INT32_MAX
        or t_perm.shape[0] != n_edge
        or t_crow.device != device
        or t_perm.device != device
    ):
        return None
    holders: dict = cache.get(("source_csr", "layouts"), dict)
    key = (id(t_crow), id(t_col), id(t_perm))
    holder: _Layout | None = holders.get(key)
    try:
        # The data pointers catch ``tensor.data = other``, which replaces the
        # contents without touching the object or its version counter.
        versions = (
            t_crow._version,
            t_col._version,
            t_perm._version,
            t_crow.data_ptr(),
            t_col.data_ptr(),
            t_perm.data_ptr(),
        )
    except RuntimeError:  # inference tensor: no version counter
        return None
    if holder is not None:
        refs = holder.sources
        if (
            refs[0]() is t_crow
            and refs[1]() is t_col
            and refs[2]() is t_perm
            and holder.hub_degree == _HUB_DEGREE
            and holder.hub_chunk == _HUB_CHUNK
            # ``resize_`` keeps the object but not the shape.
            and holder.col.shape[0] == n_edge
            and holder.crow.shape[0] == n_src + 1
        ):
            if holder.versions != versions:
                holder.fill(t_crow, t_col, t_perm)
            return holder
        del holders[key]
    while len(holders) >= _MAX_LAYOUTS:
        try:
            del holders[next(iter(holders))]
        except (KeyError, StopIteration, RuntimeError):  # raced with ``_drop``
            break

    def _drop(ref, holders=holders, key=key):
        # Only drop the holder this reference vouched for.
        entry = holders.get(key)
        if entry is not None and any(r is ref for r in entry.sources):
            del holders[key]

    holder = _Layout(
        sources=tuple(weakref.ref(t, _drop) for t in (t_crow, t_col, t_perm)),
        versions=(),
        hub_degree=_HUB_DEGREE,
        hub_chunk=_HUB_CHUNK,
        crow=torch.empty(n_src + 1, dtype=torch.int32, device=device),
        col=torch.empty(n_edge, dtype=torch.int32, device=device),
        perm=torch.empty(n_edge, dtype=torch.int32, device=device),
        degree=torch.empty(n_src, dtype=torch.long, device=device),
        # A valid pointer is needed by the launch even without hub tasks.
        hub=torch.zeros(3, dtype=torch.int32, device=device),
        n_hub=0,
        hub_limit=_INT32_MAX,
        mean_degree=max(1.0, n_edge / max(n_src, 1)),
        epoch=csr_layout._next_epoch(),
    )
    holder.fill(t_crow, t_col, t_perm)
    holders[key] = holder
    return holder


_DENSE_MAX_SCAN = 1 << 21
_DENSE_MIN_DENSITY = 0.02
_DENSE_MAX_WORK = 1 << 20
_TILE_BLOCK = 64


def _supported(values: Tensor, x: Tensor, t_crow: Tensor, t_col: Tensor) -> bool:
    """Return whether the sparse CUDA kernels support these operands.

    Besides dtype and device, the sizes must match the pattern: the kernels
    index ``values`` and the pointers without bounds checks.
    """
    device = x.device
    return (
        x.is_cuda
        and x.ndim == 2
        and values.ndim == 1
        and x.dtype in kernels_torch._FLOAT_DTYPES
        and values.dtype in kernels_torch._FLOAT_DTYPES
        and values.device == device
        and t_col.device == device
        and t_col.shape[0] > 0
        and values.shape[0] == t_col.shape[0]
        and x.shape[1] == t_crow.shape[0] - 1
        and x.shape[0] > 0
        and x.shape[1] > 0
        # Sparse kernels launch on the current device.
        and device.index == torch.cuda.current_device()
    )


# ---------------------------------------------------------------- launchers


def _prefer_dense(layout: _Layout, n_active: int, numel: int) -> bool:
    """Whether ``spike_push`` should run the dense kernel on packed input.

    Measured on an RTX 5090 (the two kernels are within ~20 % of each other
    except for the fixed cost):

    - The dense kernel has the lower fixed cost (no prefix sum, no search:
      ~25 us against ~45 us per call) as long as scanning ``x`` is free, i.e.
      up to ``_DENSE_MAX_SCAN`` elements.
    - Its lanes are filled per source, so it is the faster one per edge only
      when a typical out-edge list fills a block (mean out-degree at least
      ``_TILE_BLOCK``); then it wins from ``_DENSE_MIN_DENSITY`` upwards
      whatever the size. The packed kernel fills every lane regardless of
      the degrees and wins on low-degree graphs once the delivered edges
      (beyond ``_DENSE_MAX_WORK``) outweigh the fixed cost.
    """
    scan_is_free = numel <= _DENSE_MAX_SCAN
    if layout.mean_degree >= _TILE_BLOCK:
        return scan_is_free or n_active >= _DENSE_MIN_DENSITY * numel
    return scan_is_free and n_active * layout.mean_degree <= _DENSE_MAX_WORK


def _reference(
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    values: Tensor,
    x: Tensor,
    active_idx: Tensor,
    ptr: Tensor,
    n_out: int,
    source_major_values: bool,
) -> Tensor:
    """Use Torch for inputs the sparse CUDA kernels do not handle."""
    if source_major_values:
        # Values already follow the source-major order: identity permutation.
        t_perm = torch.arange(t_col.shape[0], device=t_col.device)
    return kernels_torch.spike_push(
        t_crow, t_col, t_perm, values, x, active_idx, ptr, n_out
    )


def layout_epoch(cache, t_crow, t_col, t_perm):
    """Return the existing source-layout epoch without allocating storage."""
    holders: dict = cache.get(("source_csr", "layouts"), dict)
    holder: _Layout | None = holders.get((id(t_crow), id(t_col), id(t_perm)))
    if holder is None:
        return 0
    refs = holder.sources
    if refs[0]() is t_crow and refs[1]() is t_col and refs[2]() is t_perm:
        return holder.epoch
    return 0
