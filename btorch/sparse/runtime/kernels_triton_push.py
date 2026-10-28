"""Reuse source-major Triton kernels for sparse push routes.

``spike_push`` consumes already-packed source IDs. Its persistent programs
balance work per delivered edge using a prefix sum and binary source lookup.
The device spike-list route in ``kernels_triton_spikelist`` reuses this same
``push_packed`` kernel after device compaction; it does not implement another
sparse product. All samples share one work range, whose length stays on device.

``spike_push_dense`` consumes dense activity. With prepared task values it
executes the existing direct/hash task queues in ``kernels_triton_tasks``.
Complete task routes require that preparation and reject missing state. The
standalone low-level API also retains the tiled dense/hub kernel and its
unsupported-input Torch fallback. Host packing in that low-level fallback is
not used by the complete device spike-list route.

All native push variants accumulate with floating-point atomics, so reduction
order and results can vary within accumulation rounding. Sparse autograd is shared with
pull: its input derivative remains a dense transposed product.

Derived source-major layouts are owned by ``KernelCache``. ``prepare`` refreshes
versioned topology outside execution; shape/address changes advance the layout
epoch. Prepared task queues and device spike lists can be captured in CUDA
graphs. The public host-packing helper itself cannot be captured. Persistent
source-major values retain stable addresses for graph replay and are refreshed
from canonical weights at trajectory preparation.
"""

from __future__ import annotations

import importlib.util

import torch
from torch import Tensor

from . import (
    csr_execution,
    kernels_torch,
    kernels_triton,
    kernels_triton_tasks,
    source_layout,
    task_layout,
)
from .backend import BackendRegistry, KernelCache, registry as _default_registry


# Runtime arguments of the kernels, in signature order. None of them is
# specialised on its value or alignment. Compiled handles include constexprs
# and pointer dtypes (see
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
    "value_stride",
    "x_batch_stride",
    "x_source_stride",
]
_PACKED_CONSTS = ("BLOCK", "PERMUTE", "SPIKE_BLOCK")
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
    "value_stride",
    "x_batch_stride",
    "x_source_stride",
]


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
        values,  # floating storage [E]
        x,  # floating storage [B * N]; values determine product promotion
        active,  # packed source indices; flat sample/source IDs for local lists
        cum,  # int64 [A] inclusive prefix sum of the active out-degrees
        ptr,  # int64 [B + 1] sample offsets into ``active``
        out,  # FP32 or FP64 accumulation [B * M]
        n_active,
        n_in,
        n_out,
        n_batch,
        n_iter,  # binary-search steps over ``cum``: bit_length(A)
        b_iter,  # binary-search steps over ``ptr``: bit_length(B - 1)
        n_programs,
        value_stride,
        x_batch_stride,
        x_source_stride,
        BLOCK: tl.constexpr,
        PERMUTE: tl.constexpr,
        SPIKE_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if SPIKE_BLOCK:
            # Flatten the grid: a large batch/population can have more list
            # blocks than CUDA's grid-y limit, even with a valid source layout.
            group = pid // n_programs
            pid = pid % n_programs
            active += group * SPIKE_BLOCK
            cum += group * SPIKE_BLOCK
            n_active = tl.load(ptr + group)
        total = tl.load(cum + tl.maximum(n_active - 1, 0), mask=n_active > 0, other=0)
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
            if SPIKE_BLOCK:
                sample = (src // n_in).to(tl.int64)
                src = src % n_in
            row_end = tl.load(crow + src + 1, mask=live, other=0)
            # ``c_end - g`` edges of the owner remain, counted from its end.
            edge = row_end.to(tl.int64) - (c_end - g)
            if not SPIKE_BLOCK:
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
            amp = tl.load(
                x + sample * x_batch_stride + src * x_source_stride,
                mask=live,
                other=0.0,
            )
            dst = tl.load(col + edge, mask=live, other=0)
            if PERMUTE:
                slot = tl.load(perm + edge, mask=live, other=0)
                w = tl.load(values + slot * value_stride, mask=live, other=0.0)
            else:
                w = tl.load(values + edge * value_stride, mask=live, other=0.0)
            tl.atomic_add(
                out + sample * n_out + dst, kernels_triton._multiply(w, amp), mask=live
            )

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
        value_stride,
        x_batch_stride,
        x_source_stride,
        TILE: tl.constexpr,
        BLOCK: tl.constexpr,
        PERMUTE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        per_sample = n_tiles + n_hub
        sample = (pid // per_sample).to(tl.int64)
        task = pid % per_sample
        lane = tl.arange(0, BLOCK)
        x_row = x + sample * x_batch_stride
        out_row = out + sample * n_out
        if task < n_tiles:
            first = task * TILE
            sources = first + tl.arange(0, TILE)
            xs = tl.load(
                x_row + sources * x_source_stride, mask=sources < n_in, other=0.0
            )
            if tl.sum((xs != 0.0).to(tl.int32), axis=0) > 0:
                for src in range(first, tl.minimum(first + TILE, n_in)):
                    amp = tl.load(x_row + src * x_source_stride)
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
                                    w = tl.load(
                                        values + slot * value_stride, mask=m, other=0.0
                                    )
                                else:
                                    w = tl.load(
                                        values + edge * value_stride, mask=m, other=0.0
                                    )
                                tl.atomic_add(
                                    out_row + dst,
                                    kernels_triton._multiply(w, amp),
                                    mask=m,
                                )
        else:
            h = (task - n_tiles) * 3
            e0 = tl.load(hub + h + 1)
            e1 = tl.load(hub + h + 2)
            # Unused slots of the fixed-capacity task list are empty ranges.
            if e1 > e0:
                amp = tl.load(x_row + tl.load(hub + h) * x_source_stride)
                if amp != 0.0:
                    for off in range(e0, e1, BLOCK):
                        edge = off + lane
                        m = edge < e1
                        dst = tl.load(col + edge, mask=m, other=0)
                        if PERMUTE:
                            slot = tl.load(perm + edge, mask=m, other=0)
                            w = tl.load(values + slot * value_stride, mask=m, other=0.0)
                        else:
                            w = tl.load(values + edge * value_stride, mask=m, other=0.0)
                        tl.atomic_add(
                            out_row + dst, kernels_triton._multiply(w, amp), mask=m
                        )

    return push_packed, push_dense


def _kernels(cache: KernelCache):
    return cache.get(("triton_push", "kernels"), _build_kernels)


# ------------------------------------------------------------------ layouts


class _Execution:
    def __init__(self, cache):
        self.cache = cache

    def launch(self, name, grid, args, const_names, consts, warps):
        kernel = _kernels(self.cache)[0 if name == "push_packed" else 1]
        kernels_triton._launch(
            self.cache,
            name,
            kernel,
            args[0].device.index,
            grid,
            args,
            const_names,
            consts,
            warps,
        )


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
        task_binding = task_layout.state_for_values(
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
                x,
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
        source_layout._layout(cache, t_crow, t_col, t_perm)
        if source_layout._supported(values, x, t_crow, t_col)
        else None
    )
    if layout is None or n_out > source_layout._INT32_MAX:
        active_idx, ptr = kernels_torch.pack_spikes(x)
        return source_layout._reference(
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
    return csr_execution._launch_dense(
        _Execution(cache),
        cache,
        layout,
        values,
        x,
        n_out,
        not source_major_values,
    )


def spike_push(*args, cache=None, **kwargs):
    cache = _default_registry.kernels if cache is None else cache
    return csr_execution.packed_push(_Execution(cache), *args, cache=cache, **kwargs)


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
    source_layout._layout(
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
    state = task_layout.prepare(
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
    packed = task_layout.bind_values(cache, state, destination_major_values.detach())
    return packed, state.layout_version


def invalidate_task_state(
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    *,
    cache: KernelCache | None = None,
) -> None:
    """Release queued-task allocations derived from topology buffers."""
    task_layout.invalidate_sources(
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
    return source_layout.layout_epoch(
        _default_registry.kernels if cache is None else cache, t_crow, t_col, t_perm
    )


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

    def prepare_dense(*args, **kwargs):
        return csr_execution.prepare_task_push(
            kernels_triton_tasks._Execution(), registry.kernels, *args, **kwargs
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
            prepare_execution=prepare_dense if kernel == "spike_push_dense" else None,
        )
