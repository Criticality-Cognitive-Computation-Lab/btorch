"""Reusable Triton synchronization primitives for generated kernels."""

from __future__ import annotations


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional GPU dependency
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _grid_fence(lane, BLOCK: tl.constexpr):
        """Emit a device-scope fence for every lane in the CTA."""
        return tl.inline_asm_elementwise(
            "membar.gl; mov.u32 $0, $1;",
            "=r,r",
            [lane],
            tl.int32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _grid_sync(
        barrier_ptr,
        epoch_ptr,
        epoch_value,
        PHASE_INDEX: tl.constexpr,
        N_PHASES: tl.constexpr,
        N_CTAS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Synchronize a cooperative Triton grid with a monotonic counter.

        ``epoch_value`` is loaded by the caller before the first phase. Each
        CTA contributes one count, then polls until every CTA has arrived at
        this phase. The final phase advances ``epoch_ptr`` from CTA 0 after
        its target is observed. The counter is never reset, so a later launch
        cannot race with a CTA still polling the previous launch.
        """
        lane = tl.arange(0, BLOCK)
        leader = lane == 0
        zero = tl.full((), 0, barrier_ptr.dtype.element_ty)
        _grid_fence(lane, BLOCK)
        # Every warp must finish publishing its stores before the leader
        # advertises this CTA's arrival to the other CTAs.
        tl.debug_barrier()
        tl.atomic_add(
            barrier_ptr + lane * 0,
            tl.full((BLOCK,), 1, barrier_ptr.dtype.element_ty),
            mask=leader,
            sem="acq_rel",
            scope="gpu",
        )
        target = (epoch_value * N_PHASES + PHASE_INDEX) * N_CTAS
        seen = tl.atomic_cas(
            barrier_ptr,
            zero,
            zero,
            sem="acquire",
            scope="gpu",
        )
        while seen < target:
            seen = tl.atomic_cas(
                barrier_ptr,
                zero,
                zero,
                sem="acquire",
                scope="gpu",
            )
        _grid_fence(lane, BLOCK)
        tl.debug_barrier()
        if PHASE_INDEX == N_PHASES:
            if tl.program_id(0) == 0:
                tl.atomic_add(
                    epoch_ptr + lane * 0,
                    tl.full((BLOCK,), 1, epoch_ptr.dtype.element_ty),
                    mask=leader,
                    sem="release",
                    scope="gpu",
                )

else:

    def _grid_sync(*args, **kwargs):  # pragma: no cover - optional dependency
        raise ImportError("Triton is required for the generated grid barrier")


__all__ = ["_grid_sync"]
