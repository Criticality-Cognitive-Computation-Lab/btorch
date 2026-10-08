"""Triton (CUDA) backend of the destination-driven sparse kernels.

Implements ``csr_matvec`` and ``edge_grad`` with the semantics of
:mod:`btorch.sparse.runtime.kernels_aten` for ``float32`` data on CUDA and
defers to the ATen kernels for everything else (other dtypes, CPU tensors,
index spaces that do not fit ``int32``, partially broadcast value batches,
grids too large for one launch).

Design, shared by both kernels:

- One launch covers all samples and value-batch members: the launch grid is
  ``(blocks of rows or entries, batch)``. ``csr_matvec`` adds one small
  launch when the pattern has rows longer than ``_SEGMENT`` entries.
- Dense operands are read sample-minor (``[member, neuron, sample]``), so the
  samples of one entry are adjacent in memory and the pattern is read once
  for a whole block of samples. With one sample this is the caller's layout;
  otherwise the operand is transposed once per call (``samples x neurons``
  elements, small next to the ``samples x entries`` products).
- Every output element is written by exactly one program and reduced in a
  fixed order, so results are bitwise reproducible. No atomics are used.
- Indices are read from cached ``int32`` copies (the kernels are
  bandwidth-bound). See :func:`_derived` for the cache and its invalidation.

The module does not register itself; call :func:`register`.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable

import torch
from torch import Tensor

from . import kernels_aten
from .backend import BackendRegistry, KernelCache, registry as _default_registry


try:  # optional dependency
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without triton
    triton = None
    tl = None


_INT32_MAX = 2**31 - 1
# CUDA limits the second grid dimension to 65535 blocks.
_MAX_GRID_Y = 65535

# Rows longer than this many entries are split into segments of this length
# that are reduced by independent programs and combined afterwards (see
# :class:`_RowLayout`). It bounds the work of one program, so the latency of a
# launch does not grow with the longest row.
_SEGMENT = 512

# ``csr_matvec`` tile ``(rows, entries, samples, warps)`` of one program: it
# owns ``rows`` consecutive rows (or row segments) for ``samples`` consecutive
# samples and walks them ``entries`` at a time. The tile is chosen from the
# number of samples only, never from the data, so the reduction order (and
# with it the result) is reproducible. Entries are ``(max samples, tile)``;
# the last one also serves every larger sample count. Tuned on an RTX 5090.
_MATVEC_TILES = (
    (1, (4, 32, 1, 4)),
    (4, (4, 32, 4, 4)),
    (8, (4, 16, 8, 4)),
    (31, (2, 16, 16, 4)),
    (0, (2, 16, 32, 8)),
)

# ``edge_grad`` tile ``(entries, samples, warps)`` of one program, selected
# like the ``csr_matvec`` tile.
_EDGE_TILES = (
    (1, (128, 1, 4)),
    (8, (64, 8, 4)),
    (0, (32, 16, 4)),
)

# Regime rule of ``csr_matvec``: hand calls with at least this many samples
# per member to the ATen kernel (cuSPARSE SpMM). ``None`` never hands over:
# on an RTX 5090 the Triton kernel was at least as fast as cuSPARSE for every
# measured size and batch (``benchmarks/sparse_conn/bench_kernels_pull.py``).
_SPMM_MIN_SAMPLES: int | None = None

# Triton's public launch path re-derives the specialisation of every argument
# on each call, which costs about as much as a small kernel. With the versions
# listed here the compiled kernel handle is cached and launched directly; any
# other version uses the public path.
_DIRECT_LAUNCH_VERSIONS = ("3.6.",)
_DIRECT_LAUNCH = triton is not None and triton.__version__.startswith(
    _DIRECT_LAUNCH_VERSIONS
)


def is_available() -> bool:
    """Whether the Triton kernels can run (Triton importable, CUDA present)."""
    return triton is not None and torch.cuda.is_available()


# ------------------------------------------------------------------ kernels
# Arguments are never specialised on their value or alignment: the compiled
# kernel then depends only on the constexpr tile sizes and on the integer
# width of the scalars, which is what makes the cached direct launch sound.

if triton is not None:
    _MATVEC_ARGS = [
        "ptr_ptr",
        "target_ptr",
        "col_ptr",
        "val_ptr",
        "x_ptr",
        "out_ptr",
        "tmp_ptr",
        "n_seg",
        "n_rows",
        "n_sample",
        "n_sample_block",
        "n_tmp",
        "val_stride",
        "x_stride",
    ]

    @triton.jit(
        do_not_specialize=_MATVEC_ARGS, do_not_specialize_on_alignment=_MATVEC_ARGS
    )
    def _csr_matvec_kernel(
        ptr_ptr,
        target_ptr,
        col_ptr,
        val_ptr,
        x_ptr,
        out_ptr,
        tmp_ptr,
        n_seg,
        n_rows,
        n_sample,
        n_sample_block,
        n_tmp,
        val_stride,
        x_stride,
        BR: tl.constexpr,
        BK: tl.constexpr,
        BS: tl.constexpr,
        HUB: tl.constexpr,
    ):
        """One program per (``BR`` row segments, member, ``BS`` samples).

        The input is sample-minor, ``x[member, column, sample]``, so the
        ``BS`` samples of one entry are adjacent in memory, and the pattern
        tile ``[BR, BK]`` (columns, values) is read once for all of them.
        """
        pid = tl.program_id(0)
        bid = tl.program_id(1)
        member = (bid // n_sample_block).to(tl.int64)
        sample = (bid % n_sample_block) * BS + tl.arange(0, BS)
        sample_mask = sample < n_sample

        seg = pid * BR + tl.arange(0, BR)
        seg_mask = seg < n_seg
        start = tl.load(ptr_ptr + seg, mask=seg_mask, other=0)
        end = tl.load(ptr_ptr + seg + 1, mask=seg_mask, other=0)
        length = end - start
        longest = tl.max(length, axis=0)

        v_base = val_ptr + member * val_stride
        x_base = x_ptr + member * x_stride
        lane = tl.arange(0, BK)
        acc = tl.zeros([BS, BR, BK], dtype=tl.float32)
        for k in range(0, longest, BK):
            pos = k + lane
            mask = pos[None, :] < length[:, None]
            idx = start[:, None] + pos[None, :]
            c = tl.load(col_ptr + idx, mask=mask, other=0)
            w = tl.load(v_base + idx, mask=mask, other=0.0)
            full = mask[None, :, :] & sample_mask[:, None, None]
            if BS == 1:  # a single sample: x[member, column]
                offset = c[None, :, :]
            else:
                offset = c[None, :, :] * n_sample + sample[:, None, None]
            xv = tl.load(x_base + offset, mask=full, other=0.0)
            acc += w[None, :, :] * xv
        total = tl.sum(acc, axis=2)

        slot = member * n_sample + sample.to(tl.int64)
        if HUB:
            # Segments of split rows go to the partial-sum buffer.
            target = tl.load(target_ptr + seg, mask=seg_mask, other=0)
            direct = seg_mask & (target < n_rows)
            partial = seg_mask & (target >= n_rows)
            tl.store(
                out_ptr + (slot * n_rows)[:, None] + target[None, :],
                total,
                mask=sample_mask[:, None] & direct[None, :],
            )
            tl.store(
                tmp_ptr + (slot * n_tmp)[:, None] + (target - n_rows)[None, :],
                total,
                mask=sample_mask[:, None] & partial[None, :],
            )
        else:
            tl.store(
                out_ptr + (slot * n_rows)[:, None] + seg[None, :],
                total,
                mask=sample_mask[:, None] & seg_mask[None, :],
            )

    _COMBINE_ARGS = ["hub_row_ptr", "hub_ptr", "tmp_ptr", "out_ptr", "n_rows", "n_tmp"]

    @triton.jit(
        do_not_specialize=_COMBINE_ARGS, do_not_specialize_on_alignment=_COMBINE_ARGS
    )
    def _hub_combine_kernel(
        hub_row_ptr,
        hub_ptr,
        tmp_ptr,
        out_ptr,
        n_rows,
        n_tmp,
        BK: tl.constexpr,
    ):
        """One program per (split row, sample): add its partial sums in
        order."""
        hub = tl.program_id(0)
        slot = tl.program_id(1).to(tl.int64)
        start = tl.load(hub_ptr + hub)
        end = tl.load(hub_ptr + hub + 1)
        base = tmp_ptr + slot * n_tmp
        acc = tl.zeros([BK], dtype=tl.float32)
        for k in range(start, end, BK):
            pos = k + tl.arange(0, BK)
            acc += tl.load(base + pos, mask=pos < end, other=0.0)
        row = tl.load(hub_row_ptr + hub)
        tl.store(out_ptr + slot * n_rows + row, tl.sum(acc, axis=0))

    _EDGE_ARGS = [
        "row_ptr",
        "col_ptr",
        "grad_ptr",
        "x_ptr",
        "out_ptr",
        "n_edge",
        "n_sample",
        "grad_stride",
        "x_stride",
    ]

    @triton.jit(do_not_specialize=_EDGE_ARGS, do_not_specialize_on_alignment=_EDGE_ARGS)
    def _edge_grad_kernel(
        row_ptr,
        col_ptr,
        grad_ptr,
        x_ptr,
        out_ptr,
        n_edge,
        n_sample,
        grad_stride,
        x_stride,
        BE: tl.constexpr,
        BS: tl.constexpr,
    ):
        """One program per (``BE`` entries, member).

        ``grad`` and ``x`` are sample-minor (``[member, neuron, sample]``):
        the samples of one entry are two contiguous runs, reduced ``BS`` at a
        time in order. No ``[samples, E]`` temporary exists.
        """
        pid = tl.program_id(0)
        member = tl.program_id(1).to(tl.int64)
        k = pid * BE + tl.arange(0, BE)
        mask = k < n_edge
        r = tl.load(row_ptr + k, mask=mask, other=0)
        c = tl.load(col_ptr + k, mask=mask, other=0)
        g_base = grad_ptr + member * grad_stride
        x_base = x_ptr + member * x_stride
        if BS == 1:  # a single sample
            g = tl.load(g_base + r, mask=mask, other=0.0)
            total = g * tl.load(x_base + c, mask=mask, other=0.0)
        else:
            r_off = r * n_sample
            c_off = c * n_sample
            lane = tl.arange(0, BS)
            acc = tl.zeros([BE, BS], dtype=tl.float32)
            for s in range(0, n_sample, BS):
                pos = s + lane
                full = mask[:, None] & (pos < n_sample)[None, :]
                at = pos[None, :]
                g = tl.load(g_base + r_off[:, None] + at, mask=full, other=0.0)
                xv = tl.load(x_base + c_off[:, None] + at, mask=full, other=0.0)
                acc += g * xv
            total = tl.sum(acc, axis=1)
        tl.store(out_ptr + member * n_edge + k, total, mask=mask)


def _no_launch_hooks() -> bool:
    """Whether no Triton launch hook (profiler, tracer) is installed; the
    direct launch would bypass them."""
    runtime = triton.knobs.runtime
    return not (
        getattr(runtime.launch_enter_hook, "calls", True)
        or getattr(runtime.launch_exit_hook, "calls", True)
    )


def _launch(
    cache: KernelCache,
    name: str,
    kernel,
    grid: tuple[int, int],
    args: tuple,
    consts: tuple,
    warps: int,
) -> None:
    """Launch ``kernel`` over ``grid``.

    Args:
        cache: Holds the compiled kernel handles.
        name: Kernel name (cache key).
        kernel: The ``triton.jit`` function.
        grid: ``(programs, batch)``.
        args: Runtime arguments in signature order.
        consts: ``(name, value)`` of every constexpr, in signature order.
        warps: Warps per program.
    """
    if _DIRECT_LAUNCH and _no_launch_hooks():
        # The kernel binary depends on whether a scalar needs 64 bits.
        wide = tuple(type(a) is int and a > _INT32_MAX for a in args)
        handles = cache.get(("triton", "compiled"), dict)
        # A compiled module belongs to the CUDA context it was loaded in.
        key = (name, consts, warps, wide, torch.cuda.current_device())
        compiled = handles.get(key)
        if compiled is not None:
            compiled.run(
                grid[0],
                grid[1],
                1,
                torch.cuda.current_stream().cuda_stream,
                compiled.function,
                compiled.packed_metadata,
                None,
                None,
                None,
                *args,
                *[value for _, value in consts],
            )
            return
        compiled = kernel[grid](*args, **dict(consts), num_warps=warps)
        if hasattr(compiled, "result"):  # asynchronous compilation
            compiled = compiled.result()
        handles[key] = compiled
        return
    kernel[grid](*args, **dict(consts), num_warps=warps)


# ------------------------------------------------------------ derived layouts


class _Slot:
    """Derived layout of one index buffer plus the state it was built from."""

    __slots__ = ("owner", "payload", "version")

    def __init__(self) -> None:
        self.owner: weakref.ref | None = None
        self.version = -1
        self.payload = None


def _derived(cache: KernelCache, kind: str, source: Tensor, build: Callable):
    """Return ``build(source)``, cached in ``cache`` for as long as it is
    valid.

    The index buffers of a connection are rewritten *in place* when it is
    rewired (same tensor object, new contents) and replaced by new tensors
    when the number of edges changes, so neither the object identity nor the
    address alone identifies the contents. The cache therefore uses:

    - a slot key ``(kind, data_ptr, shape, dtype, device)``: one slot per
      buffer, so rewiring overwrites the slot instead of growing the cache.
      The shape is part of the key because every zero-size tensor has
      ``data_ptr() == 0``;
    - the autograd version counter ``source._version``, stored in the slot:
      every in-place write (``copy_``, index assignment, ...) bumps it, also
      under ``no_grad``, which invalidates the slot;
    - a weak reference to the tensor the slot was built from. While that
      tensor is alive its memory cannot be handed to another allocation, so a
      matching address really is the same storage. Once it dies the slot is
      emptied (releasing the derived copy) and rebuilt on the next use, which
      closes the address-reuse hole of a pure ``(data_ptr, _version)`` key.

    Tensors without a version counter (inference tensors) are not cached.
    Writes that bypass the version counter (through DLPack or
    ``__cuda_array_interface__`` aliases) are not detected.
    """
    try:
        version = source._version
    except RuntimeError:  # inference tensor: no version counter
        return build(source)
    key = ("triton", kind, source.data_ptr(), source.shape, source.dtype, source.device)
    slot: _Slot = cache.get(key, _Slot)
    if (
        slot.version == version
        and slot.owner is not None
        and slot.owner() is not None
        and slot.payload is not None
    ):
        return slot.payload
    payload = build(source)

    def _release(ref, slot=slot):
        # Only drop the payload this reference vouched for.
        if slot.owner is ref:
            slot.payload = None
            slot.owner = None

    slot.owner = weakref.ref(source, _release)
    slot.version = version
    slot.payload = payload
    return payload


def _to_int32(index: Tensor) -> Tensor:
    return index.to(torch.int32).contiguous()


def _row_index_int32(crow: Tensor) -> Tensor:
    """Row of every entry, ``[E]`` ``int32`` (expanded row pointers)."""
    counts = crow[1:] - crow[:-1]
    rows = torch.arange(counts.shape[0], dtype=torch.int32, device=crow.device)
    return torch.repeat_interleave(rows, counts)


class _RowLayout:
    """Row pointers as the ``csr_matvec`` kernel reads them.

    Without long rows this is the ``int32`` copy of ``crow``. Otherwise every
    row longer than ``_SEGMENT`` is cut into segments of ``_SEGMENT``
    entries. Segments are consecutive in entry order, so they are described
    by a refined pointer array, plus where each segment's sum goes:

    Attributes:
        ptr: ``[R + 1]`` ``int32`` segment pointers (``R == M`` if unsplit).
        target: ``[R]`` ``int32`` destination of each segment: the row for an
            unsplit row, ``M + j`` for partial sum ``j`` of a split row;
            ``None`` if no row is split.
        hub_row: ``[H]`` ``int32`` the split rows.
        hub_ptr: ``[H + 1]`` ``int32`` partial sums owned by each split row.
        n_tmp: Number of partial sums.
    """

    __slots__ = ("hub_ptr", "hub_row", "n_tmp", "ptr", "target")

    def __init__(self, crow: Tensor) -> None:
        self.ptr = _to_int32(crow)
        self.target = self.hub_row = self.hub_ptr = None
        self.n_tmp = 0
        n_rows = crow.shape[0] - 1
        counts = crow[1:] - crow[:-1]
        long = counts > _SEGMENT
        # One host synchronisation per topology, not per call.
        if n_rows == 0 or not bool(long.any()):
            return
        # Segments per row; empty rows keep one (it writes their zero).
        n_seg = ((counts + (_SEGMENT - 1)) // _SEGMENT).clamp_(min=1)
        first = torch.cumsum(n_seg, 0) - n_seg
        seg_row = torch.repeat_interleave(
            torch.arange(n_rows, device=crow.device), n_seg
        )
        within = torch.arange(seg_row.shape[0], device=crow.device) - first[seg_row]
        ptr = crow[:-1][seg_row] + within * _SEGMENT
        seg_long = long[seg_row]
        partial = torch.cumsum(seg_long, 0) - 1
        self.ptr = _to_int32(torch.cat([ptr, crow[-1:]]))
        self.target = _to_int32(torch.where(seg_long, n_rows + partial, seg_row))
        self.hub_row = _to_int32(long.nonzero().squeeze(1))
        hub_seg = n_seg[long]
        self.hub_ptr = _to_int32(
            torch.cat([hub_seg.new_zeros(1), torch.cumsum(hub_seg, 0)])
        )
        self.n_tmp = int(hub_seg.sum())


def _tile(table: tuple, n_sample: int) -> tuple:
    """Tile of the first table entry that covers ``n_sample`` samples."""
    for limit, tile in table:
        if n_sample <= limit:
            return tile
    return table[-1][1]


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _sample_minor(t: Tensor, n_sample: int) -> Tensor:
    """``[*vb, *sample, n]`` -> contiguous float32 ``[members, n, samples]``.

    With a single sample the two layouts coincide and nothing is copied.
    """
    if t.dtype != torch.float32:
        t = t.to(torch.float32)
    if n_sample > 1:
        t = t.reshape(-1, n_sample, t.shape[-1]).transpose(1, 2)
    return t.contiguous()


def _unsupported(*tensors: Tensor) -> bool:
    """Whether the tensors are not all on the current CUDA device."""
    if triton is None or not all(t.is_cuda for t in tensors):
        return True
    return tensors[-1].device.index != torch.cuda.current_device()


# ---------------------------------------------------------------- csr_matvec


def _csr_matvec(
    cache: KernelCache, crow: Tensor, col: Tensor, values: Tensor, x: Tensor
) -> Tensor:
    reference = kernels_aten.csr_matvec
    n_out = crow.shape[0] - 1
    n_in = x.shape[-1]
    n_edge = col.shape[0]
    n_vb = values.ndim - 1
    if n_vb:
        v_batch, x_batch = values.shape[:-1], x.shape[:n_vb]
        batch = torch.broadcast_shapes(v_batch, x_batch)
        v_member, x_member = _numel(v_batch), _numel(x_batch)
        n_member = _numel(batch)
    else:
        batch = ()
        v_member = x_member = n_member = 1
    sample = x.shape[n_vb:-1]
    n_sample = _numel(sample)
    if (
        _unsupported(crow, col, values, x)
        or torch.promote_types(values.dtype, x.dtype) != torch.float32
        # 32-bit entry, row and sample-minor input offsets.
        or max(n_edge, n_out + 1, n_in * max(n_sample, 1)) >= _INT32_MAX
        # A side that is constant across members is passed once (stride 0);
        # partial broadcasts ([G, 1] against [1, H]) are left to ATen.
        or v_member not in (1, n_member)
        or x_member not in (1, n_member)
        or (_SPMM_MIN_SAMPLES is not None and n_sample >= _SPMM_MIN_SAMPLES)
    ):
        return reference(crow, col, values, x)

    out_shape = (*batch, *sample, n_out)
    if n_out == 0 or n_member * n_sample == 0 or n_edge == 0:
        return torch.zeros(out_shape, dtype=torch.float32, device=x.device)

    rows, entries, samples, warps = _tile(_MATVEC_TILES, n_sample)
    n_sample_block = -(-n_sample // samples)
    layout: _RowLayout = _derived(cache, "rows", crow, _RowLayout)
    hub = layout.target is not None
    if n_member * (n_sample if hub else n_sample_block) > _MAX_GRID_Y:
        return reference(crow, col, values, x)

    col32 = _derived(cache, "int32", col, _to_int32)
    if values.dtype != torch.float32:
        values = values.to(torch.float32)
    values = values.contiguous()
    x = _sample_minor(x, n_sample)
    out = torch.empty(out_shape, dtype=torch.float32, device=x.device)
    tmp = out
    if hub:
        tmp = torch.empty(
            (n_member * n_sample, layout.n_tmp), dtype=torch.float32, device=x.device
        )
    n_seg = layout.ptr.shape[0] - 1
    _launch(
        cache,
        "csr_matvec",
        _csr_matvec_kernel,
        (-(-n_seg // rows), n_member * n_sample_block),
        (
            layout.ptr,
            layout.target if hub else layout.ptr,
            col32,
            values,
            x,
            out,
            tmp,
            n_seg,
            n_out,
            n_sample,
            n_sample_block,
            layout.n_tmp,
            0 if v_member == 1 else n_edge,
            0 if x_member == 1 else n_sample * n_in,
        ),
        (("BR", rows), ("BK", entries), ("BS", samples), ("HUB", hub)),
        warps,
    )
    if hub:
        _launch(
            cache,
            "hub_combine",
            _hub_combine_kernel,
            (layout.hub_row.shape[0], n_member * n_sample),
            (layout.hub_row, layout.hub_ptr, tmp, out, n_out, layout.n_tmp),
            (("BK", 32),),
            1,
        )
    return out


def csr_matvec(crow: Tensor, col: Tensor, values: Tensor, x: Tensor) -> Tensor:
    """``y[..., m] = sum_k values[..., k] * x[..., col[k]]`` over row ``m``.

    Same contract as :func:`btorch.sparse.runtime.kernels_aten.csr_matvec`.
    All samples and value-batch members are computed by one launch whose
    programs own a few rows for a block of samples and walk those rows a
    fixed number of entries at a time. There is no maximum row length: rows
    longer than ``_SEGMENT`` entries are reduced as independent segments and
    a second, small launch adds the partial sums of each such row. Every
    output has one writer and a fixed reduction order, so repeated calls are
    bitwise identical.

    Args:
        crow: ``[M + 1]`` row pointers.
        col: ``[E]`` input index of each entry.
        values: ``[*vb, E]`` entry values.
        x: ``[*vb (broadcastable), *sample, N]`` dense input.

    Returns:
        ``[*vb, *sample, M]``.
    """
    return _csr_matvec(_default_registry.kernels, crow, col, values, x)


# ----------------------------------------------------------------- edge_grad


def _edge_grad(
    cache: KernelCache, crow: Tensor, col: Tensor, grad: Tensor, x: Tensor, n_vb: int
) -> Tensor:
    reference = kernels_aten.edge_grad
    n_out = crow.shape[0] - 1
    n_in = x.shape[-1]
    n_edge = col.shape[0]
    batch = grad.shape[:n_vb]
    n_member = _numel(batch)
    x_member = _numel(x.shape[:n_vb])
    n_sample = _numel(grad.shape[n_vb:-1])
    if (
        _unsupported(crow, col, grad, x)
        or grad.dtype != torch.float32
        or x.is_complex()
        or max(n_edge, max(n_out, n_in) * max(n_sample, 1)) >= _INT32_MAX
        or x_member not in (1, n_member)
        or _numel(x.shape[n_vb:-1]) != n_sample
        or n_member > _MAX_GRID_Y
    ):
        return reference(crow, col, grad, x, n_vb)

    out_shape = (*batch, n_edge)
    if n_edge == 0 or n_member == 0 or n_sample == 0:
        return torch.zeros(out_shape, dtype=torch.float32, device=grad.device)

    entries, samples, warps = _tile(_EDGE_TILES, n_sample)
    row32 = _derived(cache, "row_index", crow, _row_index_int32)
    col32 = _derived(cache, "int32", col, _to_int32)
    grad = _sample_minor(grad, n_sample)
    x = _sample_minor(x, n_sample)
    out = torch.empty(out_shape, dtype=torch.float32, device=grad.device)
    _launch(
        cache,
        "edge_grad",
        _edge_grad_kernel,
        (-(-n_edge // entries), n_member),
        (
            row32,
            col32,
            grad,
            x,
            out,
            n_edge,
            n_sample,
            n_sample * n_out,
            0 if x_member == 1 else n_sample * n_in,
        ),
        (("BE", entries), ("BS", samples)),
        warps,
    )
    return out


def edge_grad(crow: Tensor, col: Tensor, grad: Tensor, x: Tensor, n_vb: int) -> Tensor:
    """Gradient of ``csr_matvec`` w.r.t. ``values``.

    ``g[..., k] = sum_samples grad[..., row[k]] * x[..., col[k]]``. One
    program per block of entries reduces all samples of its entries, so the
    ``[samples, E]`` temporaries of the gather-based reference never exist.
    The row of each entry comes from a cached expanded row index. Samples
    are reduced in a fixed order: repeated calls are bitwise identical.

    Args:
        crow: ``[M + 1]`` row pointers.
        col: ``[E]`` input index of each entry.
        grad: ``[*vb, *sample, M]`` output gradient.
        x: ``[*vb (broadcastable), *sample, N]`` forward input.
        n_vb: Number of value-batch dimensions.

    Returns:
        ``[*vb, E]``.
    """
    return _edge_grad(_default_registry.kernels, crow, col, grad, x, n_vb)


# -------------------------------------------------------------- registration


def register(registry: BackendRegistry, priority: int = 10) -> None:
    """Register the Triton kernels as backend ``"triton"`` for CUDA.

    The derived layouts are cached in ``registry.kernels``.

    Args:
        registry: Registry to add the backend to.
        priority: Priority of the backend; the ATen kernels use ``0``.
    """
    cache = registry.kernels

    def _matvec(crow: Tensor, col: Tensor, values: Tensor, x: Tensor) -> Tensor:
        return _csr_matvec(cache, crow, col, values, x)

    def _grad(crow: Tensor, col: Tensor, grad: Tensor, x: Tensor, n_vb: int) -> Tensor:
        return _edge_grad(cache, crow, col, grad, x, n_vb)

    for kernel, fn in (("csr_matvec", _matvec), ("edge_grad", _grad)):
        registry.register(
            kernel,
            "triton",
            fn,
            device="cuda",
            priority=priority,
            available=is_available,
        )
