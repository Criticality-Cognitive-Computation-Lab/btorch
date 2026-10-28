"""Triton (CUDA) backend of the destination-driven sparse kernels.

Implements ``csr_matvec`` and ``edge_grad`` with the semantics of
:mod:`btorch.sparse.runtime.kernels_torch` for floating data on CUDA and
defers to the Torch kernels for unsupported cases (non-floating or CPU tensors,
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
  kernels compiled for different tiles agree up to accumulation rounding only.
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

import torch
from torch import Tensor

from . import csr_execution, csr_layout
from .backend import BackendRegistry, KernelCache, registry as _default_registry


try:  # optional dependency
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without triton
    triton = None
    tl = None


def is_available() -> bool:
    """Whether the Triton kernels can run (Triton importable, CUDA present)."""
    return triton is not None and torch.cuda.is_available()


# ------------------------------------------------------------------ kernels
# Arguments are never specialised on their value or alignment: the compiled
# kernel depends on constexpr tiles, pointer dtypes and scalar integer widths.
# The direct-launch cache includes tiles and dtypes; callers bound scalar widths.

if triton is not None:

    @triton.jit
    def _multiply(a, b):
        """Multiply typed operands in their accumulation precision."""
        dtype: tl.constexpr = (
            tl.float64 if a.dtype == tl.float64 or b.dtype == tl.float64 else tl.float32
        )
        return a.to(dtype) * b.to(dtype)

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
        "mask_ptr",
        "owner_ptr",
        "activity_ptr",
        "grad_values_ptr",
        "list_ptr",
        "list_counts_ptr",
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
        mask_ptr,
        owner_ptr,
        activity_ptr,
        grad_values_ptr,
        list_ptr,
        list_counts_ptr,
        BR: tl.constexpr,
        BK: tl.constexpr,
        BS: tl.constexpr,
        HUB: tl.constexpr,
        PERM: tl.constexpr,
        MASK: tl.constexpr,
        WEIGHT_GRAD: tl.constexpr,
        LIST: tl.constexpr,
        SEGMENT: tl.constexpr,
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
        if LIST:
            # The producer packed rows inside each original 128-neuron block.
            # Empty blocks take no edge traversal; counts never reach the host.
            block = pid // (128 // BR)
            rank = (pid % (128 // BR)) * BR + tl.arange(0, BR)
            count = tl.load(list_counts_ptr + block)
            seg_mask = rank < count
            seg = tl.load(list_ptr + block * 128 + rank, mask=seg_mask, other=0)
            list_sample = seg // n_rows
            seg = seg % n_rows
        row_start = tl.load(ptr_ptr + seg, mask=seg_mask, other=0)
        row_end = tl.load(ptr_ptr + seg + 1, mask=seg_mask, other=0)
        if LIST:
            first_chunk = tl.program_id(2)
            chunk_step = tl.num_programs(2)
            chunks = tl.maximum(
                1, tl.cdiv(tl.max(row_end - row_start, axis=0), SEGMENT)
            )
        else:
            first_chunk, chunks, chunk_step = 0, 1, 1
        for chunk in range(first_chunk, chunks, chunk_step):
            start, end = row_start, row_end
            if LIST:
                start += chunk * SEGMENT
                end = tl.minimum(end, start + SEGMENT)
                start = tl.minimum(start, end)
            length = end - start
            if MASK:
                row = tl.load(owner_ptr + seg, mask=seg_mask, other=-1) if HUB else seg
                if LIST:
                    mask_offset = list_sample[None, :] * n_rows + row[None, :]
                else:
                    mask_offset = (member * n_sample + sample.to(tl.int64))[
                        :, None
                    ] * n_rows + row[None, :]
                active = (
                    tl.load(
                        mask_ptr + mask_offset,
                        mask=sample_mask[:, None]
                        & seg_mask[None, :]
                        & (row[None, :] >= 0),
                        other=0.0,
                    )
                    != 0.0
                )
                if not WEIGHT_GRAD:
                    length = tl.where(
                        tl.sum(active.to(tl.int32), axis=0) > 0, length, 0
                    )
            if WEIGHT_GRAD:
                row = tl.load(owner_ptr + seg, mask=seg_mask, other=-1) if HUB else seg
                amplitude = tl.load(
                    activity_ptr + row[None, :] * n_sample + sample[:, None],
                    mask=sample_mask[:, None] & seg_mask[None, :] & (row[None, :] >= 0),
                    other=0.0,
                )
            longest = tl.max(length, axis=0)

            v_base = val_ptr + member * val_stride
            x_base = x_ptr + member * x_stride
            lane = tl.arange(0, BK)
            compute_dtype: tl.constexpr = (
                tl.float64 if out_ptr.dtype.element_ty == tl.float64 else tl.float32
            )
            acc = tl.zeros([BS, BR, BK], dtype=compute_dtype)
            for k in range(0, longest, BK):
                pos = k + lane
                mask = pos[None, :] < length[:, None]
                idx = start[:, None] + pos[None, :]
                c = tl.load(col_ptr + idx, mask=mask, other=0)
                full = mask[None, :, :] & sample_mask[:, None, None]
                if MASK and not WEIGHT_GRAD:
                    full = full & active[:, :, None]
                if LIST:
                    offset = c[None, :, :] * n_sample + list_sample[None, :, None]
                elif BS == 1:  # a single sample: x[member, column]
                    offset = c[None, :, :]
                else:
                    offset = c[None, :, :] * n_sample + sample[:, None, None]
                xv = tl.load(x_base + offset, mask=full, other=0.0)
                if WEIGHT_GRAD:
                    increment = tl.sum(
                        _multiply(xv, amplitude[:, :, None].to(xv.dtype)), axis=0
                    )
                    # Match the separate edge VJP's storage rounding and the
                    # route's restoration to the original parameter dtype.
                    increment = increment.to(x_ptr.dtype.element_ty).to(
                        grad_values_ptr.dtype.element_ty
                    )
                    changed = mask & (increment != 0.0)
                if PERM:
                    slot_mask = mask
                    if MASK:
                        slot_mask = (
                            mask & (tl.sum(active.to(tl.int32), axis=0) > 0)[:, None]
                        )
                        if WEIGHT_GRAD:
                            slot_mask = slot_mask | changed
                    slot = tl.load(perm_ptr + idx, mask=slot_mask, other=0)
                    weight_mask = mask
                    if MASK:
                        weight_mask = (
                            weight_mask
                            & (tl.sum(active.to(tl.int32), axis=0) > 0)[:, None]
                        )
                    w = tl.load(v_base + slot, mask=weight_mask, other=0.0)
                else:
                    w = tl.load(v_base + idx, mask=mask, other=0.0)
                product_input = xv
                if MASK:
                    product_input = tl.where(active[:, :, None], xv, 0.0)
                acc += _multiply(w[None, :, :], product_input)
                if WEIGHT_GRAD:
                    previous = tl.load(grad_values_ptr + slot, mask=changed, other=0.0)
                    value_dtype: tl.constexpr = (
                        tl.float64
                        if grad_values_ptr.dtype.element_ty == tl.float64
                        else tl.float32
                    )
                    tl.store(
                        grad_values_ptr + slot,
                        previous.to(value_dtype) + increment.to(value_dtype),
                        mask=changed,
                    )
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
                if LIST:
                    first = tl.load(owner_ptr + seg, mask=seg_mask, other=0)
                    last = tl.load(owner_ptr + seg + 1, mask=seg_mask, other=0)
                    seg_mask = seg_mask & (chunk < last - first)
                    output_offset = (
                        list_sample[None, :] * n_tmp + first[None, :] + chunk
                    )
                else:
                    output_offset = (slot * n_rows)[:, None] + seg[None, :]
                tl.store(
                    out_ptr + output_offset,
                    total,
                    mask=sample_mask[:, None] & seg_mask[None, :],
                )

    @triton.jit
    def _sum_row_segments(base, start, length, live, BK: tl.constexpr):
        """Reduce contiguous row partials in the shared split-row order."""
        lane = tl.arange(0, BK)
        compute_dtype: tl.constexpr = (
            tl.float64 if base.dtype.element_ty == tl.float64 else tl.float32
        )
        acc = tl.zeros([start.shape[0], BK], dtype=compute_dtype)
        for k in range(0, tl.max(length, axis=0), BK):
            pos = k + lane
            mask = live[:, None] & (pos[None, :] < length[:, None])
            acc += tl.load(base + start[:, None] + pos[None, :], mask=mask, other=0)
        return tl.sum(acc, axis=1)

    @triton.jit
    def _listed_matvec_output_kernel(
        partial,
        mask,
        output,
        row_start,
        N: tl.constexpr,
        N_ROWS: tl.constexpr,
        N_SEG: tl.constexpr,
        BH: tl.constexpr,
        BK: tl.constexpr,
    ):
        """Reduce listed rows and materialize zero inactive adjoints."""
        index = tl.program_id(0) * BH + tl.arange(0, BH)
        live = index < N
        active = tl.load(mask + index, mask=live, other=0) != 0
        row = index % N_ROWS
        start = tl.load(row_start + row, mask=live, other=0)
        end = tl.load(row_start + row + 1, mask=live, other=0)
        length = tl.where(active, end - start, 0)
        start += (index // N_ROWS) * N_SEG
        value = _sum_row_segments(partial, start, length, live & active, BK)
        tl.store(output + index, value, mask=live)

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
            value = _sum_row_segments(base, start, length, live, BK)
            tl.store(out_ptr + slot * n_rows + row, value, mask=live)

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
        ACCUMULATE: tl.constexpr,
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
            xv = tl.load(x_base + c, mask=mask, other=0.0).to(g.dtype)
            total = _multiply(g, xv)
        else:
            r_off = r * n_sample
            c_off = c * n_sample
            lane = tl.arange(0, BS)
            compute_dtype: tl.constexpr = (
                tl.float64 if grad_ptr.dtype.element_ty == tl.float64 else tl.float32
            )
            acc = tl.zeros([BE, BS], dtype=compute_dtype)
            for s in range(0, n_sample, BS):
                pos = s + lane
                full = mask[:, None] & (pos < n_sample)[None, :]
                at = pos[None, :]
                g = tl.load(g_base + r_off[:, None] + at, mask=full, other=0.0)
                xv = tl.load(x_base + c_off[:, None] + at, mask=full, other=0.0)
                acc += _multiply(g, xv.to(g.dtype))
            total = tl.sum(acc, axis=1)
        destination = out_ptr + member * n_edge + k
        if ACCUMULATE:
            increment = total.to(grad_ptr.dtype.element_ty).to(out_ptr.dtype.element_ty)
            previous = tl.load(destination, mask=mask, other=0.0)
            value_dtype: tl.constexpr = (
                tl.float64 if out_ptr.dtype.element_ty == tl.float64 else tl.float32
            )
            total = previous.to(value_dtype) + increment.to(value_dtype)
        tl.store(destination, total, mask=mask)


def _launch(cache, name, kernel, device_index, grid, args, const_names, consts, warps):
    """Submit a Triton stage through its public launch interface."""
    del cache, name, device_index
    kernel[grid](*args, **dict(zip(const_names, consts, strict=True)), num_warps=warps)


class _Execution:
    def __init__(self, cache):
        self.cache = cache

    def launch(self, name, grid, args, const_names, consts, warps):
        kernel = {
            "csr_matvec": _csr_matvec_kernel,
            "hub_combine": _hub_combine_kernel,
            "edge_grad": _edge_grad_kernel,
            "listed_output": _listed_matvec_output_kernel,
        }[name]
        _launch(
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


# ------------------------------------------------------------ derived layouts


# Stamps of :func:`layout_epoch`: process-wide, so they stay monotonic when
# patterns are evicted and rebuilt or a kernel cache is cleared.
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
    """Execute a product through the shared prepared destination launcher."""
    forward = csr_execution._prepare_matvec(
        _Execution(cache), cache, crow, col, values, x, perm, x_minor
    )
    return forward(x)


def csr_matvec(crow: Tensor, col: Tensor, values: Tensor, x: Tensor) -> Tensor:
    """``y[..., m] = sum_k values[..., k] * x[..., col[k]]`` over row ``m``.

    Same contract as :func:`btorch.sparse.runtime.kernels_torch.csr_matvec`.
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
    """``edge_grad``; ``grad_minor`` is the sample-minor layout of ``grad`` if
    the caller already has it."""
    return csr_execution._prepare_edge_grad(
        _Execution(cache), cache, crow, col, grad, x, n_vb, grad_minor
    )(grad, x)


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
    return csr_execution._prepare_backward(
        _Execution(cache),
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
    )(x, grad)


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
    csr_layout._prepare(_default_registry.kernels, crow, col, t_crow, t_col, t_perm)


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
    return csr_layout._layout_epoch(
        _default_registry.kernels, crow, col, t_crow, t_col, t_perm
    )


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

    def _prepare_forward(crow, col, values, x):
        return csr_execution._prepare_matvec(
            _Execution(cache), cache, crow, col, values, x
        )

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

    def _prepare_reverse(
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
        *,
        input_mask=None,
        grad_values=None,
    ):
        return csr_execution._prepare_backward(
            _Execution(cache),
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
            input_mask,
            grad_values,
        )

    def _prepare_all(crow, col, t_crow, t_col, t_perm) -> None:
        # Imported here: the push module imports this one.
        from . import kernels_triton_push as push

        csr_layout._prepare(cache, crow, col, t_crow, t_col, t_perm)
        push.prepare(crow, col, t_crow, t_col, t_perm, cache=cache)

    def _prepare_pull(crow, col, t_crow, t_col, t_perm) -> None:
        csr_layout._prepare(cache, crow, col, t_crow, t_col, t_perm)

    def _prepare_push(crow, col, t_crow, t_col, t_perm) -> None:
        from . import kernels_triton_push as push

        push.prepare(crow, col, t_crow, t_col, t_perm, cache=cache)

    def _epoch_all(crow, col, t_crow, t_col, t_perm) -> int:
        from . import kernels_triton_push as push

        return max(
            csr_layout._layout_epoch(cache, crow, col, t_crow, t_col, t_perm),
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
            prepare_execution={
                "csr_matvec": _prepare_forward,
                "csr_backward": _prepare_reverse,
            }.get(kernel),
        )
