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
  elements, small next to the ``samples x entries`` products). Reading the
  caller's sample-major layout instead was measured 10x slower from 32
  samples on, even for operands that fit the device cache: the loads of one
  entry's samples no longer coalesce.
- Every output element is written by exactly one program and reduced in a
  fixed order, so results are bitwise reproducible. No atomics are used.
  The tile is chosen from the number of samples only, never from the data;
  kernels compiled for different tiles agree up to float32 rounding only.
- Indices are read from cached ``int32`` copies (the kernels are
  bandwidth-bound). See :func:`_pattern` for the cache and its invalidation.

CUDA graphs. A captured graph (``torch.cuda.graph``, ``torch.compile(mode=
"reduce-overhead")``) records the device addresses of the derived copies and
the scalar launch arguments, and a replay runs no host code. The backend
therefore guarantees:

- :func:`prepare` builds or refreshes everything the kernels derive from one
  set of index buffers and does all host-synchronising work. A kernel call on
  prepared, unchanged buffers neither synchronises nor allocates derived
  tensors, so it can be captured.
- When the index buffers are rewritten in place (same tensor objects and
  shapes), the derived tensors are rewritten *in place* too and every launch
  argument keeps its value: a graph captured before the rewrite computes with
  the new pattern once :func:`prepare` ran. See :class:`_RowLayout` for how
  the split of long rows is kept independent of the pattern.
- :func:`layout_epoch` changes whenever a derived tensor had to be allocated
  (first build, different buffer objects or sizes, eviction from the cache,
  the first long row of a pattern). A graph captured at another epoch must
  be captured again.

The module does not register itself; call :func:`register`.
"""

from __future__ import annotations

import itertools
import weakref

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

# Tile ``(split rows, partial sums)`` of one program of the launch that adds up
# the partial sums of split rows.
_COMBINE_TILE = (16, 16)

# Regime rule of ``csr_matvec``: hand calls with at least this many samples
# per member to the ATen kernel (cuSPARSE SpMM). ``None`` never hands over:
# on an RTX 5090 the Triton kernel was at least as fast as cuSPARSE for every
# measured size and batch (``benchmarks/sparse_conn/bench_kernels_pull.py``).
_SPMM_MIN_SAMPLES: int | None = None

# Patterns (derived layouts of one ``(crow, col)`` pair) kept per cache. A
# pattern is dropped as soon as one of its index buffers dies; the limit only
# bounds what long-lived buffers can pin (the oldest pattern goes first).
_MAX_PATTERNS = 1024

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
        "perm_ptr",
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
        perm_ptr,
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
        PERM: tl.constexpr,
    ):
        """One program per (``BR`` row segments, member, ``BS`` samples).

        The input is sample-minor, ``x[member, column, sample]``, so the
        ``BS`` samples of one entry are adjacent in memory, and the pattern
        tile ``[BR, BK]`` (columns, values) is read once for all of them.
        ``PERM``: entry ``k`` reads ``values[perm[k]]`` instead of
        ``values[k]``.
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
            if PERM:
                slot = tl.load(perm_ptr + idx, mask=mask, other=0)
                w = tl.load(v_base + slot, mask=mask, other=0.0)
            else:
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
            # Unused segments of the fixed-capacity layout have target -1.
            direct = seg_mask & (target >= 0) & (target < n_rows)
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

    _COMBINE_ARGS = [
        "hub_row_ptr",
        "hub_ptr",
        "tmp_ptr",
        "out_ptr",
        "n_hub",
        "n_rows",
        "n_tmp",
    ]

    @triton.jit(
        do_not_specialize=_COMBINE_ARGS, do_not_specialize_on_alignment=_COMBINE_ARGS
    )
    def _hub_combine_kernel(
        hub_row_ptr,
        hub_ptr,
        tmp_ptr,
        out_ptr,
        n_hub,
        n_rows,
        n_tmp,
        BH: tl.constexpr,
        BK: tl.constexpr,
    ):
        """One program per (``BH`` split-row slots, sample): add the partial
        sums of each row in order.

        The grid covers the capacity of the layout. Unused slots have row
        ``-1`` and follow the used ones, so most programs of a sparse table
        return after one load.
        """
        hub = tl.program_id(0) * BH + tl.arange(0, BH)
        row = tl.load(hub_row_ptr + hub, mask=hub < n_hub, other=-1)
        if tl.max(row, axis=0) >= 0:
            live = row >= 0
            slot = tl.program_id(1).to(tl.int64)
            start = tl.load(hub_ptr + hub, mask=live, other=0)
            end = tl.load(hub_ptr + hub + 1, mask=live, other=0)
            length = end - start
            base = tmp_ptr + slot * n_tmp
            lane = tl.arange(0, BK)
            acc = tl.zeros([BH, BK], dtype=tl.float32)
            for k in range(0, tl.max(length, axis=0), BK):
                pos = k + lane
                mask = pos[None, :] < length[:, None]
                at = start[:, None] + pos[None, :]
                acc += tl.load(base + at, mask=mask, other=0.0)
            tl.store(out_ptr + slot * n_rows + row, tl.sum(acc, axis=1), mask=live)

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


_MATVEC_CONSTS = ("BR", "BK", "BS", "HUB", "PERM")
_COMBINE_CONSTS = ("BH", "BK")
_EDGE_CONSTS = ("BE", "BS")

_HANDLES_KEY = ("triton", "compiled")
_PATTERNS_KEY = ("triton", "patterns")


def _no_launch_hooks() -> bool:
    """Whether no Triton launch hook (profiler, tracer) is installed; the
    direct launch would bypass them."""
    runtime = triton.knobs.runtime
    return not (
        getattr(runtime.launch_enter_hook, "calls", True)
        or getattr(runtime.launch_exit_hook, "calls", True)
    )


def _raw_stream(device_index: int) -> int:
    """Handle of the current CUDA stream of ``device_index``."""
    return torch.cuda.current_stream(device_index).cuda_stream


# ``torch.cuda.current_stream()`` builds a ``Stream`` object on every call,
# which costs about as much as the launch itself; the raw getter does not.
_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", _raw_stream)


def _launch(
    cache: KernelCache,
    name: str,
    kernel,
    device_index: int,
    grid: tuple[int, int],
    args: tuple,
    const_names: tuple[str, ...],
    consts: tuple,
    warps: int,
) -> None:
    """Launch ``kernel`` over ``grid`` on the current stream.

    Every integer in ``args`` must fit ``int32``: the kernel binary depends
    on the width of its scalar arguments, and the cached handle is keyed on
    the constexprs only. The callers guarantee it by leaving larger problems
    to the ATen kernels.

    Args:
        cache: Holds the compiled kernel handles.
        name: Kernel name (cache key).
        kernel: The ``triton.jit`` function.
        device_index: CUDA device of the arguments (the current device).
        grid: ``(programs, batch)``.
        args: Runtime arguments in signature order.
        const_names: Names of the constexprs, in signature order.
        consts: Their values.
        warps: Warps per program.
    """
    if _DIRECT_LAUNCH and _no_launch_hooks():
        handles = cache.get(_HANDLES_KEY, dict)
        # A compiled module belongs to the CUDA context it was loaded in.
        key = (name, consts, warps, device_index)
        compiled = handles.get(key)
        if compiled is not None:
            compiled.run(
                grid[0],
                grid[1],
                1,
                _raw_stream(device_index),
                compiled.function,
                compiled.packed_metadata,
                None,
                None,
                None,
                *args,
                *consts,
            )
            return
        compiled = kernel[grid](
            *args, **dict(zip(const_names, consts, strict=True)), num_warps=warps
        )
        if hasattr(compiled, "result"):  # asynchronous compilation
            compiled = compiled.result()
        handles[key] = compiled
        return
    kernel[grid](*args, **dict(zip(const_names, consts, strict=True)), num_warps=warps)


# ------------------------------------------------------------ derived layouts


# Stamps of :func:`layout_epoch`: process-wide, so they stay monotonic when
# patterns are evicted and rebuilt or a kernel cache is cleared.
_EPOCH = itertools.count(1)


def _next_epoch() -> int:
    """A stamp larger than every stamp handed out before."""
    return next(_EPOCH)


def _int32_copy(index: Tensor) -> Tensor:
    """A contiguous ``int32`` copy of ``index`` that never aliases it (it is
    rewritten in place when ``index`` changes)."""
    return torch.empty(index.shape, dtype=torch.int32, device=index.device).copy_(index)


def _row_index(crow: Tensor) -> Tensor:
    """Row of every entry, ``[E]`` ``int32`` (expanded row pointers)."""
    counts = crow[1:] - crow[:-1]
    rows = torch.arange(counts.shape[0], dtype=torch.int32, device=crow.device)
    return torch.repeat_interleave(rows, counts)


class _RowLayout:
    """Row pointers as the ``csr_matvec`` kernel reads them.

    Two formulations exist; which one a layout uses never depends on the
    *current* contents of ``crow`` in a way a captured graph could notice:

    - *Plain* (a layout that never saw a row longer than ``_SEGMENT``): the
      kernel reads the ``int32`` copy of ``crow`` and one program reduces
      whole rows. This is correct for rows of any length, so a graph
      captured in this formulation stays correct when a later rewiring
      creates a long row; only the latency bound of the split is lost until
      the graph is captured again.
    - *Segmented* (from the first long row on, for the rest of the layout's
      life): every row longer than ``_SEGMENT`` is cut into segments of
      ``_SEGMENT`` entries whose partial sums a second launch adds up.
      Segments are consecutive in entry order, so they are described by a
      refined pointer array plus the destination of each segment's sum.

    The segmented tables have a fixed *capacity* derived from ``M`` and ``E``
    alone. A row of ``c`` entries has ``max(1, ceil(c / S)) <= 1 + c // S``
    segments, hence at most ``M + E // S`` segments in total; at most
    ``E // (S + 1)`` rows are split, into at most ``E // S + E // (S + 1)``
    partial sums. Launch grids and scalar arguments use the capacities, never
    the actual counts; unused slots are marked (target / row ``-1``, empty
    entry range) and their programs store nothing. Refilling the tables for
    another pattern of the same size therefore changes neither an address
    nor a launch argument.

    Attributes:
        crow32: ``[M + 1]`` ``int32`` copy of ``crow`` (always maintained).
        segmented: Whether the segmented formulation is in use.
        ptr: Segment pointers: ``crow32`` if plain, else ``[n_seg + 1]``.
        target: ``[n_seg]`` destination of each segment: the row for an
            unsplit row, ``M + j`` for partial sum ``j`` of a split row,
            ``-1`` for an unused slot; ``None`` if plain.
        hub_row: ``[n_hub]`` the split rows (``-1``: unused slot).
        hub_ptr: ``[n_hub + 1]`` partial sums owned by each split row.
        n_seg: Segments the kernel is launched over (``M`` if plain, else
            the capacity).
        n_tmp: Width of the partial-sum buffer (capacity; 0 if plain).
        n_hub: Split-row slots the combine kernel is launched over
            (capacity; 0 if plain).
        n_split: Rows actually split by the current pattern (informative).
    """

    __slots__ = (
        "crow32",
        "hub_ptr",
        "hub_row",
        "n_hub",
        "n_seg",
        "n_split",
        "n_tmp",
        "ptr",
        "segmented",
        "target",
    )

    def __init__(self, crow: Tensor, n_edge: int) -> None:
        self.crow32 = torch.empty(crow.shape, dtype=torch.int32, device=crow.device)
        self.ptr = self.crow32
        self.target = self.hub_row = self.hub_ptr = None
        self.segmented = False
        self.n_seg = crow.shape[0] - 1
        self.n_tmp = self.n_hub = self.n_split = 0
        self.fill(crow, n_edge)

    def fill(self, crow: Tensor, n_edge: int) -> bool:
        """(Re)derive the tables from ``crow``, in place.

        Synchronises with the host (once per topology, never per call).

        Returns:
            Whether tensors were allocated (the layout became segmented).
        """
        self.crow32.copy_(crow)
        n_rows = crow.shape[0] - 1
        counts = crow[1:] - crow[:-1]
        long = counts > _SEGMENT
        allocated = False
        if not self.segmented:
            if n_rows == 0 or not bool(long.any()):
                return False
            device = crow.device
            n_seg = n_rows + n_edge // _SEGMENT
            n_hub = min(n_rows, n_edge // (_SEGMENT + 1))
            self.ptr = torch.empty(n_seg + 1, dtype=torch.int32, device=device)
            self.target = torch.empty(n_seg, dtype=torch.int32, device=device)
            self.hub_row = torch.empty(n_hub, dtype=torch.int32, device=device)
            self.hub_ptr = torch.empty(n_hub + 1, dtype=torch.int32, device=device)
            self.n_seg, self.n_hub = n_seg, n_hub
            self.n_tmp = n_edge // _SEGMENT + n_hub
            self.segmented = allocated = True
        # Segments per row; empty rows keep one (it writes their zero).
        n_seg = ((counts + (_SEGMENT - 1)) // _SEGMENT).clamp_(min=1)
        first = torch.cumsum(n_seg, 0) - n_seg
        seg_row = torch.repeat_interleave(
            torch.arange(n_rows, device=crow.device), n_seg
        )
        used = seg_row.shape[0]
        within = torch.arange(used, device=crow.device) - first[seg_row]
        seg_long = long[seg_row]
        partial = torch.cumsum(seg_long, 0) - 1
        self.ptr[:used].copy_(crow[:-1][seg_row] + within * _SEGMENT)
        self.ptr[used:] = n_edge  # unused slots own no entries
        self.target[:used].copy_(torch.where(seg_long, n_rows + partial, seg_row))
        self.target[used:] = -1
        hub_row = long.nonzero().squeeze(1)
        n_split = self.n_split = hub_row.shape[0]
        self.hub_row[:n_split].copy_(hub_row)
        self.hub_row[n_split:] = -1
        self.hub_ptr.zero_()
        self.hub_ptr[1 : n_split + 1].copy_(torch.cumsum(n_seg[long], 0))
        return allocated


class _Pattern:
    """Everything the kernels derive from one ``(crow, col)`` pair.

    The derived tensors are allocated once and rewritten in place when the
    index buffers change (:meth:`refresh`), so their addresses are stable
    for as long as the pattern lives.

    Attributes:
        crow_ref, col_ref: Weak references to the index buffers the pattern
            was built from.
        versions: Their ``_version`` counters and addresses at build time.
        supported: Whether the Triton kernels can read the pattern (both
            buffers on one CUDA device, index space within ``int32``).
        device: Device of the buffers.
        n_out, n_edge: ``M`` and ``E``.
        col32: ``[E]`` ``int32`` copy of ``col``.
        rows: :class:`_RowLayout` of ``crow`` (built on first use).
        row32: ``[E]`` ``int32`` row of every entry (built on first use).
        perm32: ``int32`` copy of the last value permutation used with the
            pattern, with the buffer and version it was built from.
        epoch: Stamp of the last allocation of a derived tensor.
        n_fill: Times the derived tensors were (re)derived from the buffers.
    """

    __slots__ = (
        "col32",
        "col_ref",
        "crow_ref",
        "device",
        "epoch",
        "n_edge",
        "n_fill",
        "n_out",
        "perm32",
        "perm_ref",
        "perm_version",
        "row32",
        "rows",
        "supported",
        "versions",
    )

    def __init__(self, crow: Tensor, col: Tensor) -> None:
        self.crow_ref = self.col_ref = self.perm_ref = None
        self.versions = None
        self.perm_version = None
        self.device = crow.device
        self.n_out = crow.shape[0] - 1
        self.n_edge = col.shape[0]
        self.rows = self.row32 = self.perm32 = self.col32 = None
        self.epoch = 0
        self.n_fill = 1
        self.supported = (
            crow.is_cuda
            and col.device == crow.device
            # 32-bit entry and row offsets.
            and max(self.n_edge, self.n_out + 1) < _INT32_MAX
        )
        if self.supported:
            self.col32 = _int32_copy(col)
            self.epoch = _next_epoch()

    def refresh(self, crow: Tensor, col: Tensor) -> None:
        """Rewrite every derived tensor built so far from the (same-sized)
        buffers, in place."""
        self.col32.copy_(col)
        if self.rows is not None and self.rows.fill(crow, self.n_edge):
            self.epoch = _next_epoch()
        if self.row32 is not None:
            self.row32.copy_(_row_index(crow))
        self.n_fill += 1

    def row_layout(self, crow: Tensor) -> _RowLayout:
        """The :class:`_RowLayout` of the pattern (built on first use)."""
        layout = self.rows
        if layout is None:
            layout = self.rows = _RowLayout(crow, self.n_edge)
            self.epoch = _next_epoch()
        return layout

    def row_index(self, crow: Tensor) -> Tensor:
        """``[E]`` ``int32`` row of every entry (built on first use)."""
        row32 = self.row32
        if row32 is None:
            row32 = self.row32 = _row_index(crow).contiguous()
            self.epoch = _next_epoch()
        return row32

    def permutation(self, perm: Tensor) -> Tensor | None:
        """``int32`` copy of ``perm`` (``None`` if it cannot be used).

        The copy of the last permutation tensor is kept and rewritten in
        place when that tensor changes.
        """
        if perm.device != self.device or perm.shape != (self.n_edge,):
            return None
        try:
            version = (perm._version, perm.data_ptr())
        except RuntimeError:  # inference tensor: no version counter
            return _int32_copy(perm)
        ref = self.perm_ref
        if ref is not None and ref() is perm:
            if self.perm_version != version:
                self.perm32.copy_(perm)
                self.perm_version = version
            return self.perm32
        self.perm32 = _int32_copy(perm)
        self.perm_ref = weakref.ref(perm)
        self.perm_version = version
        self.epoch = _next_epoch()
        return self.perm32


def _check_pointers(crow: Tensor, col: Tensor) -> None:
    """Checked once per pattern contents: the kernels trust the pointers."""
    if crow.shape[0] and int(crow[-1]) != col.shape[0]:
        raise ValueError(
            f"crow ends at {int(crow[-1])} but col has {col.shape[0]} entries."
        )


def _pattern(cache: KernelCache, crow: Tensor, col: Tensor) -> _Pattern:
    """Return the :class:`_Pattern` of ``(crow, col)``, cached in ``cache`` for
    as long as it is valid.

    The index buffers of a connection are rewritten *in place* when it is
    rewired (same tensor objects, new contents) and replaced by new tensors
    when the number of edges changes. Neither the object identity nor the
    address alone identifies the contents, and two views of one buffer
    (``base[:n]`` and ``base[::2]``) share address, shape and version. The
    cache therefore uses:

    - the key ``(id(crow), id(col))``: one pattern per pair of tensor
      *objects*, found with a single dictionary lookup. Different views are
      different objects, so they can never be confused;
    - weak references to both tensors, compared by identity on every hit: an
      ``id`` is only reused after its object died, and then the reference no
      longer resolves to the tensor passed in;
    - the autograd version counters ``_version``: every in-place write
      (``copy_``, index assignment, ...) bumps them, also under ``no_grad``.
      The derived tensors of the pattern are then rewritten *in place*
      (:meth:`_Pattern.refresh`): rewiring does not grow the cache, and
      tensors referenced by a captured CUDA graph keep their addresses.

    Refreshing synchronises with the host. :func:`prepare` does it ahead of
    time; a call on prepared, unchanged buffers only compares versions.

    The entry is deleted when either tensor dies (releasing the derived
    copies), and at most ``_MAX_PATTERNS`` entries are kept.

    Tensors without a version counter (inference tensors) are not cached.
    Writes that bypass the version counter (through DLPack or
    ``__cuda_array_interface__`` aliases) are not detected.
    """
    patterns: dict = cache.get(_PATTERNS_KEY, dict)
    key = (id(crow), id(col))
    hit: _Pattern | None = patterns.get(key)
    try:
        # The data pointers catch ``tensor.data = other``, which replaces the
        # contents without touching the object or its version counter.
        versions = (crow._version, col._version, crow.data_ptr(), col.data_ptr())
    except RuntimeError:  # inference tensor: no version counter
        return _Pattern(crow, col)
    if hit is not None and hit.crow_ref() is crow and hit.col_ref() is col:
        if hit.versions == versions:
            return hit
        if (
            # ``resize_`` keeps the object but not the shape.
            hit.n_out == crow.shape[0] - 1
            and hit.n_edge == col.shape[0]
            and hit.device == crow.device == col.device
        ):
            _check_pointers(crow, col)
            hit.refresh(crow, col)
            hit.versions = versions
            return hit
    _check_pointers(crow, col)
    pattern = _Pattern(crow, col)
    if not pattern.supported:
        return pattern  # cheap to rebuild, nothing worth keeping

    def _drop(ref, patterns=patterns, key=key):
        # Only drop the entry this reference vouched for.
        entry = patterns.get(key)
        if entry is not None and (entry.crow_ref is ref or entry.col_ref is ref):
            del patterns[key]

    pattern.crow_ref = weakref.ref(crow, _drop)
    pattern.col_ref = weakref.ref(col, _drop)
    pattern.versions = versions
    patterns.pop(key, None)  # re-insert as the newest entry
    while len(patterns) >= _MAX_PATTERNS:
        try:
            del patterns[next(iter(patterns))]
        except (KeyError, StopIteration, RuntimeError):  # raced with ``_drop``
            break
    patterns[key] = pattern
    return pattern


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


def _on_current_device(pattern: _Pattern, a: Tensor, b: Tensor) -> bool:
    """Whether the pattern and both operands live on the current CUDA
    device."""
    device = pattern.device
    return (
        a.device == device
        and b.device == device
        and device.index == torch.cuda.current_device()
    )


# ---------------------------------------------------------------- csr_matvec


def _csr_matvec(
    cache: KernelCache,
    crow: Tensor,
    col: Tensor,
    values: Tensor,
    x: Tensor,
    perm: Tensor | None = None,
    x_minor: Tensor | None = None,
) -> Tensor:
    """``csr_matvec`` (``perm is None``) or ``csr_matvec_gather``.

    ``x_minor`` is ``_sample_minor(x, n_sample)`` if the caller already has
    it (the backward shares one transposed gradient between its kernels).
    """
    n_vb = values.ndim - 1
    pattern = None
    # ``x`` needs the value-batch dimensions and the neuron axis; the
    # reference kernel raises the error for anything shorter.
    if triton is not None and x.is_cuda and x.ndim > n_vb:
        pattern = _pattern(cache, crow, col)
    if pattern is not None and pattern.supported:
        n_out, n_edge = pattern.n_out, pattern.n_edge
        n_in = x.shape[-1]
        if n_vb:
            v_batch, x_batch = values.shape[:-1], x.shape[:n_vb]
            sample = x.shape[n_vb:-1]
            if v_batch == x_batch:
                batch = v_batch
                v_member = x_member = n_member = _numel(batch)
            else:
                batch = torch.broadcast_shapes(v_batch, x_batch)
                v_member, x_member = _numel(v_batch), _numel(x_batch)
                n_member = _numel(batch)
        else:
            batch = ()
            sample = x.shape[:-1]
            v_member = x_member = n_member = 1
        n_sample = _numel(sample)
        rows, entries, samples, warps = _tile(_MATVEC_TILES, n_sample)
        perm32 = None if perm is None else pattern.permutation(perm)
        supported = (
            _on_current_device(pattern, values, x)
            and (
                (values.dtype == torch.float32 and x.dtype == torch.float32)
                or torch.promote_types(values.dtype, x.dtype) == torch.float32
            )
            # Entry values are read at ``k`` (or ``perm[k]``) without a bound.
            and (values.shape[-1] == n_edge if perm is None else perm32 is not None)
            and values.shape[-1] < _INT32_MAX
            # 32-bit sample-minor input offsets (the lanes past the last
            # sample of a block are masked but still computed).
            and n_in * max(n_sample, 1) + samples < _INT32_MAX
            # A side that is constant across members is passed once (stride
            # 0); partial broadcasts ([G, 1] against [1, H]) are left to ATen.
            and (v_member == n_member or v_member == 1)
            and (x_member == n_member or x_member == 1)
            and (_SPMM_MIN_SAMPLES is None or n_sample < _SPMM_MIN_SAMPLES)
        )
    else:
        supported = False
    if not supported:
        if perm is not None:
            values = values.index_select(-1, perm)
        return kernels_aten.csr_matvec(crow, col, values, x)

    out_shape = (*batch, *sample, n_out)
    if n_out == 0 or n_member * n_sample == 0 or n_edge == 0:
        return torch.zeros(out_shape, dtype=torch.float32, device=x.device)

    layout = pattern.rows
    if layout is None:
        layout = pattern.row_layout(crow)
    hub = layout.segmented
    n_sample_block = -(-n_sample // samples)
    if n_member * (n_sample if hub else n_sample_block) > _MAX_GRID_Y:
        if perm is not None:
            values = values.index_select(-1, perm)
        return kernels_aten.csr_matvec(crow, col, values, x)

    device_index = pattern.device.index
    if values.dtype != torch.float32:
        values = values.to(torch.float32)
    values = values.contiguous()
    x = _sample_minor(x, n_sample) if x_minor is None else x_minor
    out = torch.empty(out_shape, dtype=torch.float32, device=x.device)
    tmp = out
    if hub:
        tmp = torch.empty(
            (n_member * n_sample, layout.n_tmp), dtype=torch.float32, device=x.device
        )
    n_seg = layout.n_seg
    _launch(
        cache,
        "csr_matvec",
        _csr_matvec_kernel,
        device_index,
        (-(-n_seg // rows), n_member * n_sample_block),
        (
            layout.ptr,
            layout.target if hub else layout.ptr,
            pattern.col32,
            pattern.col32 if perm32 is None else perm32,
            values,
            x,
            out,
            tmp,
            n_seg,
            n_out,
            n_sample,
            n_sample_block,
            layout.n_tmp,
            0 if v_member == 1 else values.shape[-1],
            0 if x_member == 1 else n_sample * n_in,
        ),
        _MATVEC_CONSTS,
        (rows, entries, samples, hub, perm32 is not None),
        warps,
    )
    if hub:
        _launch(
            cache,
            "hub_combine",
            _hub_combine_kernel,
            device_index,
            (-(-layout.n_hub // _COMBINE_TILE[0]), n_member * n_sample),
            (
                layout.hub_row,
                layout.hub_ptr,
                tmp,
                out,
                layout.n_hub,
                n_out,
                layout.n_tmp,
            ),
            _COMBINE_CONSTS,
            _COMBINE_TILE,
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


def csr_matvec_gather(
    crow: Tensor, col: Tensor, perm: Tensor, values: Tensor, x: Tensor
) -> Tensor:
    """``csr_matvec(crow, col, values[..., perm], x)`` without the gathered
    copy.

    The kernel reads ``values[..., perm[k]]`` for entry ``k``. This is the
    transposed product of the backward: ``(crow, col)`` is then the
    source-major CSR and ``perm`` the position of each of its entries in the
    destination-major ``values``. Reduction order and result are those of
    :func:`csr_matvec` on the gathered values.

    Args:
        crow: ``[M + 1]`` row pointers.
        col: ``[E]`` input index of each entry.
        perm: ``[E]`` position in ``values`` of each entry.
        values: ``[*vb, E']`` values, indexed through ``perm``.
        x: ``[*vb (broadcastable), *sample, N]`` dense input.

    Returns:
        ``[*vb, *sample, M]``.
    """
    return _csr_matvec(_default_registry.kernels, crow, col, values, x, perm)


# ----------------------------------------------------------------- edge_grad


def _edge_grad(
    cache: KernelCache,
    crow: Tensor,
    col: Tensor,
    grad: Tensor,
    x: Tensor,
    n_vb: int,
    grad_minor: Tensor | None = None,
) -> Tensor:
    """``edge_grad``; ``grad_minor`` is ``_sample_minor(grad, n_sample)`` if
    the caller already has it."""
    supported = False
    # Both operands need the value-batch dimensions and the neuron axis.
    if triton is not None and grad.is_cuda and grad.ndim > n_vb and x.ndim > n_vb:
        pattern = _pattern(cache, crow, col)
        if pattern.supported:
            n_out, n_edge = pattern.n_out, pattern.n_edge
            n_in = x.shape[-1]
            batch = grad.shape[:n_vb]
            n_member = _numel(batch)
            x_member = _numel(x.shape[:n_vb])
            n_sample = _numel(grad.shape[n_vb:-1])
            entries, samples, warps = _tile(_EDGE_TILES, n_sample)
            supported = (
                _on_current_device(pattern, grad, x)
                and grad.dtype == torch.float32
                and not x.is_complex()
                and grad.shape[-1] == n_out
                # 32-bit sample-minor offsets (the lanes past the last
                # sample of a block are masked but still computed).
                and max(n_out, n_in) * (n_sample + samples) < _INT32_MAX
                and (x_member == n_member or x_member == 1)
                and _numel(x.shape[n_vb:-1]) == n_sample
                and n_member <= _MAX_GRID_Y
            )
    if not supported:
        return kernels_aten.edge_grad(crow, col, grad, x, n_vb)

    out_shape = (*batch, n_edge)
    if n_edge == 0 or n_member == 0 or n_sample == 0:
        return torch.zeros(out_shape, dtype=torch.float32, device=grad.device)

    row32 = pattern.row32
    if row32 is None:
        row32 = pattern.row_index(crow)
    grad = _sample_minor(grad, n_sample) if grad_minor is None else grad_minor
    x = _sample_minor(x, n_sample)
    out = torch.empty(out_shape, dtype=torch.float32, device=grad.device)
    _launch(
        cache,
        "edge_grad",
        _edge_grad_kernel,
        pattern.device.index,
        (-(-n_edge // entries), n_member),
        (
            row32,
            pattern.col32,
            grad,
            x,
            out,
            n_edge,
            n_sample,
            n_sample * n_out,
            0 if x_member == 1 else n_sample * n_in,
        ),
        _EDGE_CONSTS,
        (entries, samples),
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


# ------------------------------------------------------------------ backward


def _backward(
    cache: KernelCache,
    crow: Tensor,
    col: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    values: Tensor,
    x: Tensor,
    grad: Tensor,
    need_values: bool,
    need_x: bool,
) -> tuple[Tensor | None, Tensor | None]:
    n_vb = values.ndim - 1
    grad_minor = None
    if (
        need_values
        and need_x
        and grad.is_cuda
        and grad.dtype == torch.float32
        and grad.ndim > n_vb + 1
        and grad.numel() > 0
    ):
        # Both kernels read the output gradient sample-minor: transpose once.
        n_sample = _numel(grad.shape[n_vb:-1])
        if n_sample > 1:
            grad_minor = _sample_minor(grad, n_sample)
    grad_values = grad_x = None
    if need_values:
        grad_values = _edge_grad(cache, crow, col, grad, x, n_vb, grad_minor)
    if need_x:
        grad_x = _csr_matvec(cache, t_crow, t_col, values, grad, t_perm, grad_minor)
    return grad_values, grad_x


def csr_backward(
    crow: Tensor,
    col: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    values: Tensor,
    x: Tensor,
    grad: Tensor,
    need_values: bool = True,
    need_x: bool = True,
) -> tuple[Tensor | None, Tensor | None]:
    """Both gradients of ``y = csr_matvec(crow, col, values, x)``.

    ``grad_values = edge_grad(crow, col, grad, x)`` and the transposed
    product ``grad_x = csr_matvec_gather(t_crow, t_col, t_perm, values,
    grad)`` over the source-major CSR of the same entries. Compared with two
    separate kernel calls, the gradient is brought into the kernels' layout
    once and no source-major copy of the values is made. Results are those
    of the separate kernels, bit for bit.

    Args:
        crow: ``[M + 1]`` row pointers of the destination-major CSR.
        col: ``[E]`` input index of each entry.
        t_crow: ``[N + 1]`` pointers of the source-major CSR.
        t_col: ``[E]`` output index of each source-major entry.
        t_perm: ``[E]`` position in ``values`` of each source-major entry.
        values: ``[*vb, E]`` entry values.
        x: ``[*vb (broadcastable), *sample, N]`` forward input.
        grad: ``[*vb, *sample, M]`` gradient w.r.t. the output.
        need_values: Compute the gradient w.r.t. ``values``.
        need_x: Compute the gradient w.r.t. ``x``.

    Returns:
        ``(grad_values [*vb, E], grad_x [*vb, *sample, N])``; ``None`` for a
        gradient that was not requested. Like the separate kernels, the
        results are not reduced to the shapes of broadcast operands.
    """
    return _backward(
        _default_registry.kernels,
        crow,
        col,
        t_crow,
        t_col,
        t_perm,
        values,
        x,
        grad,
        need_values,
        need_x,
    )


# ------------------------------------------------------- prepare and epoch


def _cached_pattern(cache: KernelCache, crow: Tensor, col: Tensor) -> _Pattern | None:
    """The cached pattern of ``(crow, col)`` if there is one (not validated
    against the buffer versions, nothing is built)."""
    hit = cache.get(_PATTERNS_KEY, dict).get((id(crow), id(col)))
    if hit is not None and hit.crow_ref() is crow and hit.col_ref() is col:
        return hit
    return None


def _prepare(
    cache: KernelCache,
    crow: Tensor,
    col: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
) -> None:
    if triton is None or not (crow.is_cuda and t_crow.is_cuda):
        return
    # Forward (``csr_matvec``) and value gradient (``edge_grad``).
    pattern = _pattern(cache, crow, col)
    if pattern.supported and pattern.versions is not None:
        pattern.row_layout(crow)
        pattern.row_index(crow)
    # Transposed product of the backward (``csr_matvec_gather``).
    pattern = _pattern(cache, t_crow, t_col)
    if pattern.supported and pattern.versions is not None:
        pattern.row_layout(t_crow)
        pattern.permutation(t_perm)


def _layout_epoch(
    cache: KernelCache,
    crow: Tensor,
    col: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
) -> int:
    epoch = 0
    for pattern in (
        _cached_pattern(cache, crow, col),
        _cached_pattern(cache, t_crow, t_col),
    ):
        if pattern is not None:
            epoch = max(epoch, pattern.epoch)
    return epoch


def prepare(
    crow: Tensor, col: Tensor, t_crow: Tensor, t_col: Tensor, t_perm: Tensor
) -> None:
    """Build or refresh every layout the pull kernels derive from the index
    buffers of one connection.

    Covers :func:`csr_matvec` and :func:`edge_grad` on ``(crow, col)`` and
    the transposed product of :func:`csr_backward` on ``(t_crow, t_col,
    t_perm)``. All host-synchronising work happens here: afterwards, and
    until a buffer is written again, the kernels neither synchronise nor
    allocate derived tensors, so they can be captured in a CUDA graph. If
    the buffers were rewritten in place since the last call, the derived
    tensors are rewritten in place as well and graphs captured earlier stay
    valid; whenever that was not possible :func:`layout_epoch` changes.

    Cheap when nothing changed (a few version comparisons); a no-op for
    buffers the Triton kernels do not serve (CPU, inference tensors).
    Call it outside captured regions, after every change of the buffers.

    Args:
        crow: ``[M + 1]`` row pointers of the destination-major CSR.
        col: ``[E]`` input index of each entry.
        t_crow: ``[N + 1]`` pointers of the source-major CSR.
        t_col: ``[E]`` output index of each source-major entry.
        t_perm: ``[E]`` position in ``values`` of each source-major entry.

    Raises:
        ValueError: If a pointer array does not match its index array.
    """
    _prepare(_default_registry.kernels, crow, col, t_crow, t_col, t_perm)


def layout_epoch(
    crow: Tensor, col: Tensor, t_crow: Tensor, t_col: Tensor, t_perm: Tensor
) -> int:
    """Stamp of the derived tensors the pull kernels use for these buffers.

    Read it after :func:`prepare`. It increases whenever a derived tensor
    was allocated instead of rewritten in place (first build, new buffer
    objects or sizes, a pattern evicted from the cache, the first long row
    of a pattern); a CUDA graph is valid only for the epoch it was captured
    at. Stamps are unique across buffer sets and never decrease. ``0``
    means nothing is cached for the buffers. Reading it builds nothing and
    does not synchronise.

    Args:
        crow, col, t_crow, t_col, t_perm: As for :func:`prepare`.

    Returns:
        The epoch.
    """
    return _layout_epoch(_default_registry.kernels, crow, col, t_crow, t_col, t_perm)


# -------------------------------------------------------------- registration


def register(registry: BackendRegistry, priority: int = 10) -> None:
    """Register the Triton kernels as backend ``"triton"`` for CUDA.

    Registers ``"csr_matvec"``, ``"edge_grad"`` and ``"csr_backward"`` (see
    :func:`csr_backward`; an optional kernel that only this backend
    provides). The derived layouts are cached in ``registry.kernels``.

    Also registers the two housekeeping entries of the backend, which cover
    the pull kernels *and* the push kernels of
    :mod:`btorch.sparse.runtime.kernels_triton_push` (a registry holds one
    implementation per name and backend), both with the signature ``(crow,
    col, t_crow, t_col, t_perm)``:

    - ``"prepare"``: :func:`prepare` for both kernel families. Call it after
      every change of the index buffers, outside captured regions.
    - ``"layout_epoch"``: the larger of the two :func:`layout_epoch` values.

    Args:
        registry: Registry to add the backend to.
        priority: Priority of the backend; the ATen kernels use ``0``.
    """
    cache = registry.kernels

    def _matvec(crow: Tensor, col: Tensor, values: Tensor, x: Tensor) -> Tensor:
        return _csr_matvec(cache, crow, col, values, x)

    def _grad(crow: Tensor, col: Tensor, grad: Tensor, x: Tensor, n_vb: int) -> Tensor:
        return _edge_grad(cache, crow, col, grad, x, n_vb)

    def _both(
        crow, col, t_crow, t_col, t_perm, values, x, grad, need_values, need_x
    ) -> tuple[Tensor | None, Tensor | None]:
        return _backward(
            cache,
            crow,
            col,
            t_crow,
            t_col,
            t_perm,
            values,
            x,
            grad,
            need_values,
            need_x,
        )

    def _prepare_all(crow, col, t_crow, t_col, t_perm) -> None:
        # Imported here: the push module imports this one.
        from . import kernels_triton_push as push

        _prepare(cache, crow, col, t_crow, t_col, t_perm)
        push.prepare(crow, col, t_crow, t_col, t_perm, cache=cache)

    def _prepare_pull(crow, col, t_crow, t_col, t_perm) -> None:
        _prepare(cache, crow, col, t_crow, t_col, t_perm)

    def _prepare_push(crow, col, t_crow, t_col, t_perm) -> None:
        from . import kernels_triton_push as push

        push.prepare(crow, col, t_crow, t_col, t_perm, cache=cache)

    def _epoch_all(crow, col, t_crow, t_col, t_perm) -> int:
        from . import kernels_triton_push as push

        return max(
            _layout_epoch(cache, crow, col, t_crow, t_col, t_perm),
            push.layout_epoch(crow, col, t_crow, t_col, t_perm, cache=cache),
        )

    for kernel, fn in (
        ("csr_matvec", _matvec),
        ("edge_grad", _grad),
        ("csr_backward", _both),
        ("prepare", _prepare_all),
        ("prepare_pull", _prepare_pull),
        ("prepare_push", _prepare_push),
        ("layout_epoch", _epoch_all),
    ):
        registry.register(
            kernel,
            "triton",
            fn,
            device="cuda",
            priority=priority,
            available=is_available,
        )
