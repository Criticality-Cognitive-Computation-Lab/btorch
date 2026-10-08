"""Triton CUDA kernels for source-driven (push) spike propagation.

Two kernels reproduce :func:`btorch.sparse.runtime.kernels_aten.spike_push`
for float32 CUDA tensors; everything else falls back to the ATen reference.

``spike_push`` (packed input, the registered ``"spike_push"`` backend)
    *Thread per gathered non-zero.* The out-edges of all active ``(sample,
    source)`` pairs of the whole batch form one flat work range ``[0, W)``,
    where ``W`` is the last element of the prefix sum of the active
    out-degrees. A fixed number of persistent programs stride over blocks of
    that range; every lane finds the active source owning its work position
    with a vectorised binary search over the prefix sum, then performs one
    ``atomic_add``. Work is therefore balanced per *edge*: a hub is split
    across as many programs as it has blocks, and sources without out-edges
    cost nothing. One launch covers every sample of the batch, and ``W`` is
    read on the device, so no host synchronisation is added to the one
    ``pack_spikes`` already paid.

``spike_push_dense`` (dense input, no packing)
    *Compaction inside the kernel.* A fixed grid of programs covers every
    ``(sample, tile of sources)``; a program loads its tile of ``x``, returns
    if it is silent and otherwise pushes the out-edges of its active sources.
    Sources with more than ``_HUB_DEGREE`` out-edges are skipped there and
    served by a static list of *hub tasks* (one program per ``(sample, chunk
    of a hub's out-edges)``) appended to the same grid, so long edge lists are
    still split across programs. The grid depends on shapes only: there is no
    ``nonzero``, no host round trip and no data-dependent launch, which makes
    the kernel CUDA-graph friendly.

Both kernels accumulate with float atomics: the summation order of the
contributions to one destination is not deterministic, so results agree with
the reference (and between two runs) only up to float32 rounding of the
partial sums.

Derived layouts (int32 copies of the source-major CSR, the hub task list) are
memoised in the registry's :class:`~btorch.sparse.runtime.backend.KernelCache`;
see :func:`_layout` for the invalidation rule.

CUDA graphs: only the dense kernel can be captured (packing uses ``nonzero``,
which synchronises). :func:`prepare` builds or refreshes the layout ahead of a
capture; an in-place rewrite of the index buffers is absorbed in place, with
unchanged addresses and launch arguments (see :class:`_Layout`), and
:func:`layout_epoch` changes whenever that was not possible.
"""

from __future__ import annotations

import importlib.util
import weakref
from dataclasses import dataclass

import torch
from torch import Tensor

from . import kernels_aten, kernels_triton, kernels_triton_tasks
from .backend import BackendRegistry, KernelCache, registry as _default_registry


# Lanes (gathered edges) per block of the packed kernel and its warps.
_BLOCK = 256
_BLOCK_WARPS = 4
# Upper bound on the persistent programs of the packed kernel (~6 per SM of an
# RTX 5090); fewer are launched when the expected work is smaller.
_MAX_PROGRAMS = 1024

# Dense kernel: sources per program, lanes per out-edge block and its warps.
_TILE = 64
_TILE_BLOCK = 64
_TILE_WARPS = 2
# Sources with more out-edges than this are served by hub tasks of
# ``_HUB_CHUNK`` edges each instead of by their tile.
_HUB_DEGREE = 2048
_HUB_CHUNK = 1024

# Choice between the two kernels inside ``spike_push`` (see ``_prefer_dense``).
_DENSE_MAX_SCAN = 1 << 21
_DENSE_MIN_DENSITY = 0.02
_DENSE_MAX_WORK = 1 << 20

_INT32_MAX = 2**31 - 1
# Layout holders kept per cache. A holder is dropped as soon as one of its
# source tensors dies; the limit only bounds what long-lived tensors can pin
# (the oldest holder goes first).
_MAX_LAYOUTS = 1024

# Runtime arguments of the kernels, in signature order. None of them is
# specialised on its value or alignment, so a compiled kernel depends on the
# constexprs only and can be launched through a cached handle (see
# :func:`btorch.sparse.runtime.kernels_triton._launch`).
_PACKED_ARGS = [
    "crow",
    "col",
    "perm",
    "values",
    "x",
    "active",
    "cum",
    "ptr",
    "out",
    "n_active",
    "n_in",
    "n_out",
    "n_batch",
    "n_iter",
    "b_iter",
    "n_programs",
]
_PACKED_CONSTS = ("BLOCK", "PERMUTE")
_DENSE_ARGS = [
    "crow",
    "col",
    "perm",
    "values",
    "x",
    "hub",
    "out",
    "n_in",
    "n_out",
    "n_tiles",
    "n_hub",
    "hub_degree",
]
_DENSE_CONSTS = ("TILE", "BLOCK", "PERMUTE")


def is_available() -> bool:
    """Whether the Triton push kernels can run in this process."""
    return importlib.util.find_spec("triton") is not None and torch.cuda.is_available()


# ------------------------------------------------------------------ kernels


def _build_kernels():
    """Define the Triton kernels (compiled lazily on their first launch)."""
    import triton
    import triton.language as tl

    @triton.jit(
        do_not_specialize=_PACKED_ARGS, do_not_specialize_on_alignment=_PACKED_ARGS
    )
    def push_packed(
        crow,  # int32 [N + 1] source-major pointers
        col,  # int32 [E] destination of each source-major entry
        perm,  # int32 [E] position in ``values`` of each source-major entry
        values,  # float32 [E]
        x,  # float32 [B * N]
        active,  # int64 [A] packed source indices, sample-major
        cum,  # int64 [A] inclusive prefix sum of the active out-degrees
        ptr,  # int64 [B + 1] sample offsets into ``active``
        out,  # float32 [B * M]
        n_active,
        n_in,
        n_out,
        n_batch,
        n_iter,  # binary-search steps over ``cum``: bit_length(A)
        b_iter,  # binary-search steps over ``ptr``: bit_length(B - 1)
        n_programs,
        BLOCK: tl.constexpr,
        PERMUTE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        total = tl.load(cum + n_active - 1)
        lane = tl.arange(0, BLOCK).to(tl.int64)
        # Persistent program: blocks pid, pid + n_programs, ... of the work.
        for base in range(pid.to(tl.int64) * BLOCK, total, n_programs * BLOCK):
            g = base + lane
            live = g < total
            # owner = first k with cum[k] > g  (skips zero-degree sources).
            lo = tl.zeros([BLOCK], dtype=tl.int32)
            hi = tl.zeros([BLOCK], dtype=tl.int32) + n_active
            for _ in range(n_iter):
                open_ = lo < hi
                mid = (lo + hi) >> 1
                c = tl.load(cum + mid, mask=open_, other=0)
                hi = tl.where(open_ & (c > g), mid, hi)
                lo = tl.where(open_ & (c <= g), mid + 1, lo)
            owner = tl.where(live, lo, 0)
            c_end = tl.load(cum + owner, mask=live, other=0)
            src = tl.load(active + owner, mask=live, other=0)
            row_end = tl.load(crow + src + 1, mask=live, other=0)
            # ``c_end - g`` edges of the owner remain, counted from its end.
            edge = row_end.to(tl.int64) - (c_end - g)
            # sample = first s with ptr[s + 1] > owner, searched in [0, B - 1].
            s_lo = tl.zeros([BLOCK], dtype=tl.int32)
            s_hi = tl.zeros([BLOCK], dtype=tl.int32) + (n_batch - 1)
            for _ in range(b_iter):
                s_open = s_lo < s_hi
                s_mid = (s_lo + s_hi) >> 1
                p = tl.load(ptr + s_mid + 1, mask=s_open, other=0)
                s_hi = tl.where(s_open & (p > owner), s_mid, s_hi)
                s_lo = tl.where(s_open & (p <= owner), s_mid + 1, s_lo)
            # 64-bit offsets: ``sample * n`` overflows int32 on large batches.
            sample = s_lo.to(tl.int64)
            amp = tl.load(x + sample * n_in + src, mask=live, other=0.0)
            dst = tl.load(col + edge, mask=live, other=0)
            if PERMUTE:
                slot = tl.load(perm + edge, mask=live, other=0)
                w = tl.load(values + slot, mask=live, other=0.0)
            else:
                w = tl.load(values + edge, mask=live, other=0.0)
            tl.atomic_add(out + sample * n_out + dst, w * amp, mask=live)

    @triton.jit(
        do_not_specialize=_DENSE_ARGS, do_not_specialize_on_alignment=_DENSE_ARGS
    )
    def push_dense(
        crow,
        col,
        perm,
        values,
        x,
        hub,  # int32 [3 * H]: (source, first edge, end edge) per hub task
        out,
        n_in,
        n_out,
        n_tiles,
        n_hub,
        hub_degree,
        TILE: tl.constexpr,
        BLOCK: tl.constexpr,
        PERMUTE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        per_sample = n_tiles + n_hub
        sample = (pid // per_sample).to(tl.int64)
        task = pid % per_sample
        lane = tl.arange(0, BLOCK)
        x_row = x + sample * n_in
        out_row = out + sample * n_out
        if task < n_tiles:
            first = task * TILE
            sources = first + tl.arange(0, TILE)
            xs = tl.load(x_row + sources, mask=sources < n_in, other=0.0)
            if tl.sum((xs != 0.0).to(tl.int32), axis=0) > 0:
                for src in range(first, tl.minimum(first + TILE, n_in)):
                    amp = tl.load(x_row + src)
                    if amp != 0.0:
                        e0 = tl.load(crow + src)
                        e1 = tl.load(crow + src + 1)
                        if e1 - e0 <= hub_degree:
                            for off in range(e0, e1, BLOCK):
                                edge = off + lane
                                m = edge < e1
                                dst = tl.load(col + edge, mask=m, other=0)
                                if PERMUTE:
                                    slot = tl.load(perm + edge, mask=m, other=0)
                                    w = tl.load(values + slot, mask=m, other=0.0)
                                else:
                                    w = tl.load(values + edge, mask=m, other=0.0)
                                tl.atomic_add(out_row + dst, w * amp, mask=m)
        else:
            h = (task - n_tiles) * 3
            e0 = tl.load(hub + h + 1)
            e1 = tl.load(hub + h + 2)
            # Unused slots of the fixed-capacity task list are empty ranges.
            if e1 > e0:
                amp = tl.load(x_row + tl.load(hub + h))
                if amp != 0.0:
                    for off in range(e0, e1, BLOCK):
                        edge = off + lane
                        m = edge < e1
                        dst = tl.load(col + edge, mask=m, other=0)
                        if PERMUTE:
                            slot = tl.load(perm + edge, mask=m, other=0)
                            w = tl.load(values + slot, mask=m, other=0.0)
                        else:
                            w = tl.load(values + edge, mask=m, other=0.0)
                        tl.atomic_add(out_row + dst, w * amp, mask=m)

    return push_packed, push_dense


def _kernels(cache: KernelCache):
    return cache.get(("triton_push", "kernels"), _build_kernels)


# ------------------------------------------------------------------ layouts


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
            self.epoch = kernels_triton._next_epoch()
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
    holders: dict = cache.get(("triton_push", "layouts"), dict)
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
        epoch=kernels_triton._next_epoch(),
    )
    holder.fill(t_crow, t_col, t_perm)
    holders[key] = holder
    return holder


def _supported(values: Tensor, x: Tensor, t_crow: Tensor, t_col: Tensor) -> bool:
    """Whether the Triton kernels can run on these operands.

    Besides dtype and device, the sizes must match the pattern: the kernels
    index ``values`` and the pointers without bounds checks.
    """
    device = x.device
    return (
        x.is_cuda
        and x.ndim == 2
        and values.ndim == 1
        and x.dtype == torch.float32
        and values.dtype == torch.float32
        and values.device == device
        and t_col.device == device
        and t_col.shape[0] > 0
        and values.shape[0] == t_col.shape[0]
        and x.shape[1] == t_crow.shape[0] - 1
        and x.shape[0] > 0
        and x.shape[1] > 0
        # Triton launches on the current device.
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
    """ATen fallback for inputs the Triton kernels do not handle."""
    if source_major_values:
        # Values already follow the source-major order: identity permutation.
        t_perm = torch.arange(t_col.shape[0], device=t_col.device)
    return kernels_aten.spike_push(
        t_crow, t_col, t_perm, values, x, active_idx, ptr, n_out
    )


def _launch_dense(
    cache: KernelCache,
    layout: _Layout,
    values: Tensor,
    x: Tensor,
    n_out: int,
    permute: bool,
) -> Tensor:
    n_batch, n_in = x.shape
    out = torch.zeros(n_batch, n_out, dtype=torch.float32, device=x.device)
    n_tiles = (n_in + _TILE - 1) // _TILE
    grid = n_batch * (n_tiles + layout.n_hub)
    if grid > _INT32_MAX:
        raise RuntimeError("spike_push_dense: batch too large for one launch.")
    kernels_triton._launch(
        cache,
        "push_dense",
        _kernels(cache)[1],
        x.device.index,
        (grid, 1),
        (
            layout.crow,
            layout.col,
            layout.perm,
            values,
            x,
            layout.hub,
            out,
            n_in,
            n_out,
            n_tiles,
            layout.n_hub,
            layout.hub_limit,
        ),
        _DENSE_CONSTS,
        (_TILE, _TILE_BLOCK, permute),
        _TILE_WARPS,
    )
    return out


def spike_push_dense(
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    values: Tensor,
    x: Tensor,
    n_out: int,
    *,
    source_major_values: bool = False,
    task_values: Tensor | None = None,
    require_task: bool = False,
    cache: KernelCache | None = None,
) -> Tensor:
    """Source-driven propagation straight from the dense input.

    Same result as :func:`spike_push` without ``pack_spikes``: the kernel
    skips silent sources itself. One launch with a shape-only grid and no
    host synchronisation.

    Args:
        t_crow: ``[N + 1]`` pointers of the source-major (transposed) CSR.
        t_col: ``[E]`` destination of each entry in source-major order.
        t_perm: ``[E]`` position in ``values`` of each source-major entry.
        values: ``[E]`` edge values in destination-major order.
        x: ``[B, N]`` dense input.
        n_out: Number of destinations ``M``.
        source_major_values: ``values`` is already in source-major order
            (``values_dst_major[t_perm]``), so ``t_perm`` is not consulted.
            Edge values are then read sequentially instead of through a
            random gather, which roughly halves the kernel time on large
            graphs. Not part of the ATen contract; for callers that can keep
            such a copy.
        cache: Kernel cache holding the derived layouts (default: the global
            registry's).

    Returns:
        ``[B, M]``.
    """
    cache = _default_registry.kernels if cache is None else cache
    if task_values is not None and task_values.numel() != 0:
        task_binding = kernels_triton_tasks.state_for_values(
            cache,
            task_values,
            t_crow,
            t_col,
            t_perm,
        )
        if task_binding is not None:
            return kernels_triton_tasks.run(
                task_binding.task_state,
                task_values,
                x.contiguous(),
                n_out,
                cache,
                workspace=task_binding.workspace,
            )
    if require_task:
        raise RuntimeError(
            "the selected Triton task route has no prepared task state; "
            "replan or prepare the connection before execution"
        )
    layout = (
        _layout(cache, t_crow, t_col, t_perm)
        if _supported(values, x, t_crow, t_col)
        else None
    )
    if layout is None or n_out > _INT32_MAX:
        active_idx, ptr = kernels_aten.pack_spikes(x)
        return _reference(
            t_crow,
            t_col,
            t_perm,
            values,
            x,
            active_idx,
            ptr,
            n_out,
            source_major_values,
        )
    return _launch_dense(
        cache,
        layout,
        values.contiguous(),
        x.contiguous(),
        n_out,
        not source_major_values,
    )


def spike_push(
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    values: Tensor,
    x: Tensor,
    active_idx: Tensor,
    ptr: Tensor,
    n_out: int,
    *,
    source_major_values: bool = False,
    cache: KernelCache | None = None,
) -> Tensor:
    """Source-driven propagation of the packed non-zero inputs (Triton).

    Drop-in replacement of
    :func:`btorch.sparse.runtime.kernels_aten.spike_push` for float32 CUDA
    tensors; other inputs are forwarded to it. The whole batch is served by
    one launch. Accumulation uses float atomics, so the result matches the
    reference up to float32 rounding and is not bit-reproducible.

    Args:
        t_crow: ``[N + 1]`` pointers of the source-major (transposed) CSR.
        t_col: ``[E]`` destination of each entry in source-major order.
        t_perm: ``[E]`` position in ``values`` of each source-major entry.
        values: ``[E]`` edge values in destination-major order.
        x: ``[B, N]`` dense input (supplies the amplitudes).
        active_idx: Packed source indices from ``pack_spikes``.
        ptr: ``[B + 1]`` sample offsets from ``pack_spikes``.
        n_out: Number of destinations ``M``.
        source_major_values: ``values`` is already in source-major order
            (``values_dst_major[t_perm]``), so ``t_perm`` is not consulted.
            Edge values are then read sequentially instead of through a
            random gather, which roughly halves the kernel time on large
            graphs. Not part of the ATen contract; for callers that can keep
            such a copy.
        cache: Kernel cache holding the derived layouts (default: the global
            registry's).

    Returns:
        ``[B, M]``.
    """
    cache = _default_registry.kernels if cache is None else cache
    layout = (
        _layout(cache, t_crow, t_col, t_perm)
        if _supported(values, x, t_crow, t_col)
        else None
    )
    n_active = active_idx.shape[0]
    if (
        layout is None
        or max(n_active, n_out) > _INT32_MAX
        # The packed kernel reads both index lists as int64 on the device.
        or active_idx.dtype != torch.long
        or ptr.dtype != torch.long
        or active_idx.device != x.device
        or ptr.device != x.device
        or ptr.shape[0] != x.shape[0] + 1
    ):
        return _reference(
            t_crow,
            t_col,
            t_perm,
            values,
            x,
            active_idx,
            ptr,
            n_out,
            source_major_values,
        )
    n_batch, n_in = x.shape
    if n_active == 0:
        return torch.zeros(n_batch, n_out, dtype=torch.float32, device=x.device)
    values, x = values.contiguous(), x.contiguous()
    if _prefer_dense(layout, n_active, x.numel()):
        return _launch_dense(cache, layout, values, x, n_out, not source_major_values)
    out = torch.zeros(n_batch, n_out, dtype=torch.float32, device=x.device)
    # Inclusive prefix sum of the active out-degrees; its last element is the
    # total work, which only the kernel reads (no host synchronisation).
    active_idx, ptr = active_idx.contiguous(), ptr.contiguous()
    cum = torch.cumsum(layout.degree[active_idx], 0)
    # ``n_active`` is known on the host for free, so the number of persistent
    # programs follows the *expected* work; the actual work is split evenly
    # among them whatever the degrees of the active sources are.
    expected_blocks = int(n_active * layout.mean_degree / _BLOCK) + 1
    n_programs = min(_MAX_PROGRAMS, expected_blocks)
    kernels_triton._launch(
        cache,
        "push_packed",
        _kernels(cache)[0],
        x.device.index,
        (n_programs, 1),
        (
            layout.crow,
            layout.col,
            layout.perm,
            values,
            x,
            active_idx,
            cum,
            ptr,
            out,
            n_active,
            n_in,
            n_out,
            n_batch,
            n_active.bit_length(),
            (n_batch - 1).bit_length(),
            n_programs,
        ),
        _PACKED_CONSTS,
        (_BLOCK, not source_major_values),
        _BLOCK_WARPS,
    )
    return out


def prepare(
    crow: Tensor,
    col: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    *,
    cache: KernelCache | None = None,
) -> None:
    """Build or refresh the layout the push kernels derive from the index
    buffers of one connection.

    All host-synchronising work happens here: afterwards, and until a buffer
    is written again, :func:`spike_push_dense` neither synchronises nor
    allocates derived tensors, so it can be captured in a CUDA graph. An
    in-place rewrite of the buffers is absorbed in place (graphs captured
    earlier stay valid); otherwise :func:`layout_epoch` changes. Cheap when
    nothing changed; a no-op for buffers the kernels do not serve.

    Args:
        crow: Unused (destination-major pointers; the signature is shared
            with :func:`btorch.sparse.runtime.kernels_triton.prepare`).
        col: Unused.
        t_crow: ``[N + 1]`` pointers of the source-major CSR.
        t_col: ``[E]`` destination of each entry in source-major order.
        t_perm: ``[E]`` position in ``values`` of each source-major entry.
        cache: Kernel cache holding the derived layouts (default: the global
            registry's).
    """
    if not t_col.is_cuda or t_col.shape[0] == 0 or not is_available():
        return
    _layout(
        _default_registry.kernels if cache is None else cache, t_crow, t_col, t_perm
    )


def prepare_task_values(
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    destination_major_values: Tensor,
    batch_size: int,
    *,
    cache: KernelCache | None = None,
    route_key: object | None = None,
) -> tuple[Tensor | None, int]:
    """Prepare queued-task state and pack weights for one recurrent trajectory.

    The returned values are used only by the task forward kernels. Gradients
    continue to be computed from ``destination_major_values`` by the shared
    sparse propagation backward.

    Returns:
        ``(task_values, epoch)``. ``task_values`` is ``None`` when the task
        strategy cannot serve the operands.
    """
    cache = _default_registry.kernels if cache is None else cache
    state = kernels_triton_tasks.prepare(
        cache,
        t_crow,
        t_col,
        t_perm,
        batch_size,
        destination_major_values.dtype,
        route_key=route_key,
    )
    if state is None:
        return None, 0
    packed = kernels_triton_tasks.bind_values(
        cache, state, destination_major_values.detach()
    )
    return packed, state.layout_version


def invalidate_task_state(
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    *,
    cache: KernelCache | None = None,
) -> None:
    """Release queued-task allocations derived from topology buffers."""
    kernels_triton_tasks.invalidate_sources(
        _default_registry.kernels if cache is None else cache,
        t_crow,
        t_col,
        t_perm,
    )


def layout_epoch(
    crow: Tensor,
    col: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    *,
    cache: KernelCache | None = None,
) -> int:
    """Stamp of the derived tensors the push kernels use for these buffers.

    Same contract as
    :func:`btorch.sparse.runtime.kernels_triton.layout_epoch` (the stamps
    of both modules come from one counter): read it after :func:`prepare`;
    it increases whenever a derived tensor was allocated instead of
    rewritten in place (first build, new buffer objects or sizes, eviction,
    the first hub of a layout). ``0`` means nothing is cached. Reading it
    builds nothing and does not synchronise.

    Args:
        crow, col, t_crow, t_col, t_perm: As for :func:`prepare`.
        cache: Kernel cache holding the derived layouts.

    Returns:
        The epoch.
    """
    cache = _default_registry.kernels if cache is None else cache
    holders: dict = cache.get(("triton_push", "layouts"), dict)
    holder: _Layout | None = holders.get((id(t_crow), id(t_col), id(t_perm)))
    if holder is None:
        return 0
    refs = holder.sources
    if refs[0]() is t_crow and refs[1]() is t_col and refs[2]() is t_perm:
        return holder.epoch
    return 0


def register(registry: BackendRegistry, priority: int = 10) -> None:
    """Register the Triton push kernels as backend ``"triton"`` on CUDA.

    Two kernels are registered: ``"spike_push"`` (the packed contract of the
    ATen kernel) and ``"spike_push_dense"`` (``(t_crow, t_col, t_perm,
    values, x, n_out)``, no packing; nothing resolves it until an operator
    asks for it). The derived layouts live in ``registry.kernels``. Not
    called at import. The ``"prepare"`` and ``"layout_epoch"`` entries of the
    backend are registered by
    :func:`btorch.sparse.runtime.kernels_triton.register` and cover this
    module's layouts too.

    Args:
        registry: Registry to add the backend to.
        priority: Priority of the backend (the ATen reference has 0).
    """

    def push(t_crow, t_col, t_perm, values, x, active_idx, ptr, n_out):
        return spike_push(
            t_crow,
            t_col,
            t_perm,
            values,
            x,
            active_idx,
            ptr,
            n_out,
            cache=registry.kernels,
        )

    def push_dense(
        t_crow,
        t_col,
        t_perm,
        values,
        x,
        n_out,
        *,
        task_values=None,
        require_task=False,
    ):
        return spike_push_dense(
            t_crow,
            t_col,
            t_perm,
            values,
            x,
            n_out,
            task_values=task_values,
            require_task=require_task,
            cache=registry.kernels,
        )

    def prepare_values(t_crow, t_col, t_perm, values, batch_size, *, route_key=None):
        return prepare_task_values(
            t_crow,
            t_col,
            t_perm,
            values,
            batch_size,
            cache=registry.kernels,
            route_key=route_key,
        )

    for kernel, fn in (
        ("spike_push", push),
        ("spike_push_dense", push_dense),
        ("prepare_task_values", prepare_values),
    ):
        registry.register(
            kernel,
            "triton",
            fn,
            device="cuda",
            priority=priority,
            available=is_available,
        )
