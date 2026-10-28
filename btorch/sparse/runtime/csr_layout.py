"""Prepare versioned CSR layouts and sample storage independently of
kernels."""

from __future__ import annotations

import itertools
import weakref

import torch
from torch import Tensor

from .backend import KernelCache


_PATTERNS_KEY = ("csr", "patterns")

_INT32_MAX = 2**31 - 1
# CUDA limits the second grid dimension to 65535 blocks.
_MAX_GRID_Y = 65535

# Rows longer than this many entries are split into segments of this length
# that are reduced by independent programs and combined afterwards (see
# :class:`_RowLayout`). It bounds the work of one program, so the latency of a
# launch does not grow with the longest row.
_SEGMENT = 512
# Bound the shape-only list grid; each program strides over further segments.
_LIST_PARTITIONS = 8

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
        owner: Original row of each segment, for output-adjoint support masks.
        hub_row: ``[n_hub]`` the split rows (``-1``: unused slot).
        hub_ptr: ``[n_hub + 1]`` partial sums owned by each split row.
        n_seg: Segments the kernel is launched over (``M`` if plain, else
            the capacity).
        n_tmp: Width of the partial-sum buffer (capacity; 0 if plain).
        n_hub: Split-row slots the combine kernel is launched over
            (capacity; 0 if plain).
        n_split: Rows actually split by the current pattern (informative).
        row_segment_start: Per-row offsets into compact listed-row partials.
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
        "owner",
        "row_segment_start",
    )

    def __init__(self, crow: Tensor, n_edge: int) -> None:
        self.crow32 = torch.empty(crow.shape, dtype=torch.int32, device=crow.device)
        self.ptr = self.crow32
        self.target = self.owner = self.hub_row = self.hub_ptr = None
        self.segmented = False
        self.n_seg = crow.shape[0] - 1
        self.n_tmp = self.n_hub = self.n_split = 0
        self.row_segment_start = torch.arange(
            crow.numel(), dtype=torch.int32, device=crow.device
        )
        self.fill(crow, n_edge)

    def fill(self, crow: Tensor, n_edge: int) -> bool:
        """(Re)derive the tables from ``crow``, in place.

        Synchronises with the host (once per topology, never per call).

        Returns:
            Whether derived tensors were allocated.
        """
        self.crow32.copy_(crow)
        n_rows = crow.shape[0] - 1
        counts = crow[1:] - crow[:-1]
        long = counts > _SEGMENT
        allocated = False
        if not self.segmented:
            if n_rows == 0 or not bool(long.any()):
                return allocated
            device = crow.device
            n_seg = n_rows + n_edge // _SEGMENT
            n_hub = min(n_rows, n_edge // (_SEGMENT + 1))
            self.ptr = torch.empty(n_seg + 1, dtype=torch.int32, device=device)
            self.target = torch.empty(n_seg, dtype=torch.int32, device=device)
            self.owner = torch.empty(n_seg, dtype=torch.int32, device=device)
            self.hub_row = torch.empty(n_hub, dtype=torch.int32, device=device)
            self.hub_ptr = torch.empty(n_hub + 1, dtype=torch.int32, device=device)
            self.n_seg, self.n_hub = n_seg, n_hub
            self.n_tmp = n_edge // _SEGMENT + n_hub
            self.segmented = allocated = True
        # Segments per row; empty rows keep one (it writes their zero).
        n_seg = ((counts + (_SEGMENT - 1)) // _SEGMENT).clamp_(min=1)
        first = torch.cumsum(n_seg, 0) - n_seg
        self.row_segment_start[:-1].copy_(first)
        self.row_segment_start[-1:].copy_(first[-1:] + n_seg[-1:])
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
        self.owner[:used].copy_(seg_row)
        self.owner[used:] = -1
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


def _prepare_sample_minor(t: Tensor, n_sample: int):
    """Prepare reusable sample-minor storage without changing operand dtype."""
    if n_sample <= 1 and t.is_contiguous():
        return lambda input: input
    minor = t.reshape(-1, n_sample, t.shape[-1]).transpose(1, 2) if n_sample > 1 else t
    workspace = torch.empty_like(minor, memory_format=torch.contiguous_format)

    def forward(input):
        minor = (
            input.reshape(-1, n_sample, input.shape[-1]).transpose(1, 2)
            if n_sample > 1
            else input
        )
        workspace.copy_(minor)
        return workspace

    return forward


def _on_current_device(pattern: _Pattern, a: Tensor, b: Tensor) -> bool:
    """Whether the pattern and both operands live on the current CUDA
    device."""
    device = pattern.device
    return (
        a.device == device
        and b.device == device
        and device.index == torch.cuda.current_device()
    )


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
    if not (crow.is_cuda and t_crow.is_cuda):
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
