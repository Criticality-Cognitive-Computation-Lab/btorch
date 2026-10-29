"""Device spike-list preparation for the existing packed Triton SpMSpV kernel.

Compact active source IDs and scan their degrees within each producer
block, then reuse the packed fanout kernel. Generated neurons can emit
the same lists directly. Capacity covers every source, so underestimated
activity cannot overflow and CUDA graphs need no host-side count read.
"""

from __future__ import annotations

from torch import Tensor

from . import kernels_triton_push, source_layout, spike_list
from .backend import KernelCache
from .spike_list import SPIKE_BLOCK


try:
    import triton
    import triton.language as tl
except ImportError:  # CPU installations do not need Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _emit_active_list(selected, index, active, counts, BLOCK: tl.constexpr):
        """Share block-local compaction between spike and gradient
        producers."""
        rank = tl.cumsum(selected.to(tl.int32)) - 1
        slot = tl.program_id(0) * BLOCK + rank
        tl.store(active + slot, index, mask=selected)
        tl.store(counts + tl.program_id(0), tl.sum(selected.to(tl.int32)))
        return slot

    @triton.jit
    def _emit_spike_list(
        amplitude,
        index,
        live,
        degree,
        active,
        cumulative,
        counts,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # The neuron epilogue and standalone packer share this exact emission.
        # Local scans replace the global degree scan; fixed block ownership
        # avoids reset/append atomics and cross-CTA ordering assumptions.
        firing = live & (amplitude != 0.0)
        slot = _emit_active_list(firing, index, active, counts, BLOCK)
        fanout = tl.load(degree + index % N, mask=firing, other=0).to(tl.int64)
        prefix = tl.cumsum(fanout)
        tl.store(cumulative + slot, prefix, mask=firing)

    @triton.jit
    def _pack_spike_list_kernel(
        x,
        degree,
        active,
        cumulative,
        counts,
        output,
        N: tl.constexpr,
        N_INPUT: tl.constexpr,
        N_OUTPUT: tl.constexpr,
        BLOCK: tl.constexpr,
        X_BATCH: tl.constexpr,
        X_SOURCE: tl.constexpr,
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(output + index, 0.0, mask=index < N_OUTPUT)
        if tl.program_id(0) < tl.cdiv(N_INPUT, BLOCK):
            amplitude = tl.load(
                x + (index // N) * X_BATCH + (index % N) * X_SOURCE,
                mask=index < N_INPUT,
                other=0.0,
            )
            _emit_spike_list(
                amplitude,
                index,
                index < N_INPUT,
                degree,
                active,
                cumulative,
                counts,
                N,
                BLOCK,
            )


def prepare_spike_list(*args, **kwargs):
    return spike_list.prepare_spike_list(
        *args, **kwargs, bind=_prepare_spike_list_launch
    )


def _prepare_spike_list_launch(execution, x):
    """Bind the existing packed fanout's arguments and executable once."""
    from ...utils.triton import prepare_launch

    queue = execution.spikes
    n_batch, n_in = x.shape
    n_out = execution.output.shape[-1]
    # Several consumers per queue retain parallelism for small populations and
    # long fanouts; all consume the same packed kernel, without a global scan.
    programs = execution.programs
    execution.arguments = [
        execution.layout.crow,
        execution.layout.col,
        execution.layout.perm,
        execution.values if execution.task_values is None else execution.task_values,
        x,
        queue.indices,
        queue.cumulative,
        queue.counts,
        queue.output,
        SPIKE_BLOCK,
        n_in,
        n_out,
        n_batch,
        SPIKE_BLOCK.bit_length(),
        0,
        programs,
        (
            execution.values if execution.task_values is None else execution.task_values
        ).stride(0),
        x.stride(0),
        x.stride(1),
    ]

    pack_arguments = [
        x,
        queue.degree,
        queue.indices,
        queue.cumulative,
        queue.counts,
        queue.output,
    ]
    pack_launch = None

    def pack(input):
        nonlocal pack_launch
        pack_arguments[0] = input
        if pack_launch is None:
            pack_launch = prepare_launch(
                _pack_spike_list_kernel,
                pack_arguments,
                (-(-max(input.numel(), queue.output.numel()) // SPIKE_BLOCK),),
                tuple(
                    zip(
                        ("N", "N_INPUT", "N_OUTPUT", "BLOCK", "X_BATCH", "X_SOURCE"),
                        (
                            queue.n_sources,
                            input.numel(),
                            queue.output.numel(),
                            SPIKE_BLOCK,
                            x.stride(0),
                            x.stride(1),
                        ),
                        strict=True,
                    )
                ),
                changing=(0,),
                writes=(2, 3, 4, 5),
                work=input.numel(),
            )
        pack_launch(pack_arguments)

    execution.pack = pack
    launch = prepare_launch(
        kernels_triton_push._kernels(execution.cache)[0],
        execution.arguments,
        (programs * queue.counts.numel(),),
        tuple(
            zip(
                kernels_triton_push._PACKED_CONSTS,
                (
                    source_layout.EDGE_BLOCK,
                    execution.task_values is None,
                    SPIKE_BLOCK,
                ),
                strict=True,
            )
        ),
        changing=(4,),
        num_warps=source_layout.BLOCK_WARPS,
    )

    def consume(arguments):
        # The shared SpikeList validates the source ABI before this callback.
        # Refresh its owner and pointer once, retaining static launch arguments.
        source = arguments[4]
        launch.owners[4] = source
        launch.arguments[4] = source.data_ptr()
        launch.submit()

    execution.launch = consume


def spike_push_list(
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    values: Tensor,
    x: Tensor,
    n_out: int,
    *,
    task_values: Tensor | None = None,
    require_task: bool = False,
    cache: KernelCache,
) -> Tensor:
    """Propagate once through the shared prepared spike-list launcher."""
    forward = prepare_spike_list(
        t_crow,
        t_col,
        t_perm,
        values,
        x,
        n_out,
        task_values=task_values,
        require_task=require_task,
        cache=cache,
    )
    return forward(x)


def register(registry):
    """Register spike-list preparation and fanout under the Triton backend."""

    def prepare(t_crow, t_col, t_perm, values, batch_size, *, route_key=None):
        return spike_list.prepare_values(
            t_crow,
            t_col,
            t_perm,
            values,
            batch_size,
            cache=registry.kernels,
            route_key=route_key,
        )

    def push(*args, **kwargs):
        return spike_push_list(*args, **kwargs, cache=registry.kernels)

    def prepare_forward(*args, **kwargs):
        return prepare_spike_list(*args, **kwargs, cache=registry.kernels)

    for name, implementation in (
        ("spike_push_list", push),
        ("prepare_spike_list_values", prepare),
    ):
        registry.register(
            name,
            "triton",
            implementation,
            device="cuda",
            priority=20,
            available=kernels_triton_push.is_available,
            prepare_execution=prepare_forward if name == "spike_push_list" else None,
        )
