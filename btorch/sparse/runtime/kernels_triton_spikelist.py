"""Device spike-list preparation for the existing packed Triton SpMSpV kernel.

Compact active source IDs and scan their degrees within each producer
block, then reuse the packed fanout kernel. Generated neurons can emit
the same lists directly. Capacity covers every source, so underestimated
activity cannot overflow and CUDA graphs need no host-side count read.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from . import kernels_torch, kernels_triton, kernels_triton_push
from .activity import BLOCK_SIZE
from .backend import KernelCache


try:
    import triton
    import triton.language as tl
except ImportError:  # CPU installations do not need Triton.
    triton = None
    tl = None


# Match the native cell launch width so one producer owns one queue block.
SPIKE_BLOCK = BLOCK_SIZE


@dataclass(frozen=True)
class SpikeList:
    """Invocation-owned block-local lists and degree prefixes for fanout.

    Every producer writes its own count, including zero for a silent
    block. Capacity covers all sources; neither a hint nor a global
    atomic counter limits activity. Indices are flattened sample/source
    coordinates.
    """

    indices: Tensor
    cumulative: Tensor
    counts: Tensor
    degree: Tensor
    output: Tensor
    n_sources: int

    @property
    def producer_layout(self):
        n_input = self.counts.numel() * SPIKE_BLOCK
        n_output = self.output.numel()
        return self.n_sources, n_output, n_input

    def bindings(self):
        return {
            "spike_indices": self.indices,
            "spike_cumulative": self.cumulative,
            "spike_counts": self.counts,
            "spike_degree": self.degree,
            "spike_accumulation": self.output,
        }


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
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(output + index, 0.0, mask=index < N_OUTPUT)
        if tl.program_id(0) < tl.cdiv(N_INPUT, BLOCK):
            amplitude = tl.load(x + index, mask=index < N_INPUT, other=0.0)
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


def prepare_values(
    t_crow, t_col, t_perm, values, batch_size, *, cache: KernelCache, route_key=None
):
    """Prepare source-major weights once per trajectory, outside its timestep
    loop."""
    if (
        values.dtype not in kernels_torch._FLOAT_DTYPES
        or values.ndim != 1
        or not values.is_cuda
    ):
        return None, 0
    layout = kernels_triton_push._layout(cache, t_crow, t_col, t_perm)
    if (
        layout is None
        or batch_size > kernels_triton._MAX_GRID_Y
        or batch_size * (t_crow.numel() - 1) > 2**31 - 1
    ):
        return None, 0
    # Sparse backward still owns the canonical differentiable values. This is
    # only a read layout; no second autograd path or independent weight exists.
    if layout.source_values is None or layout.source_values.dtype != values.dtype:
        # This read layout outlives the invocation that first requests it.
        # Training may save it for backward after an inference-only run.
        with torch.inference_mode(False):
            layout.source_values = torch.empty_like(values)
        layout.source_value_epoch = kernels_triton._next_epoch()
    # Stable addresses let already-captured graphs observe optimizer updates.
    # The shared layout cache owns lifetime and topology invalidation, just as
    # for the packed fanout's index copies.
    layout.source_values.copy_(values.detach().index_select(0, t_perm.long()))
    return layout.source_values, layout.source_value_epoch


@dataclass
class _PreparedSpikeList:
    spikes: SpikeList | None
    values: Tensor
    layout: object
    task_values: Tensor | None
    output: Tensor
    input_workspace: Tensor | None
    cache: KernelCache
    programs: int = 1

    def _source(self, input):
        if self.input_workspace is not None:
            self.input_workspace.copy_(input)
            return self.input_workspace
        return input

    def __call__(self, input):
        if self.spikes is None:
            return self.output
        input = self._source(input)
        queue = self.spikes
        _pack_spike_list_kernel[
            (triton.cdiv(max(input.numel(), queue.output.numel()), SPIKE_BLOCK),)
        ](
            input,
            queue.degree,
            queue.indices,
            queue.cumulative,
            queue.counts,
            queue.output,
            queue.n_sources,
            input.numel(),
            queue.output.numel(),
            SPIKE_BLOCK,
        )
        return _launch_spike_list(self, input)

    def from_spikes(self, input):
        """Consume an emitted list; its producer already cleared the output."""
        return (
            self.output
            if self.spikes is None
            else _launch_spike_list(self, self._source(input))
        )


def prepare_spike_list(
    t_crow,
    t_col,
    t_perm,
    values,
    x,
    n_out,
    *,
    task_values=None,
    require_task=False,
    cache: KernelCache,
):
    """Prepare block-local list storage for the shared packed SpMSpV kernel."""
    n_batch, n_in = x.shape
    output_dtype = torch.promote_types(values.dtype, x.dtype)
    output = x.new_empty((n_batch, n_out), dtype=output_dtype)
    accumulation = (
        output
        if output_dtype == kernels_torch._accumulation_dtype(output_dtype)
        else torch.empty_like(
            output, dtype=kernels_torch._accumulation_dtype(output_dtype)
        )
    )
    if not x.numel() or not values.numel():
        output.zero_()
        return _PreparedSpikeList(None, values, None, None, output, None, cache)
    layout = kernels_triton_push._layout(cache, t_crow, t_col, t_perm)
    if layout is None or x.numel() > 2**31 - 1:
        raise RuntimeError("Unsupported spike-list layout; replan the connection")
    groups = triton.cdiv(x.numel(), SPIKE_BLOCK)
    queue = SpikeList(
        torch.empty(groups * SPIKE_BLOCK, device=x.device, dtype=torch.int32),
        torch.empty(groups * SPIKE_BLOCK, device=x.device, dtype=torch.int64),
        torch.empty(groups, device=x.device, dtype=torch.int32),
        layout.degree,
        accumulation,
        n_in,
    )
    return _PreparedSpikeList(
        queue,
        values,
        layout,
        task_values,
        output,
        None
        if x.is_contiguous()
        else torch.empty_like(x, memory_format=torch.contiguous_format),
        cache,
        # Size from topology, not measured activity. Sparse blocks need enough
        # consumers to hide dependent prefix/edge loads even when neighboring
        # producer blocks are silent. Persistent consumers cover arbitrary
        # degrees, including hubs; this is not a capacity bound.
        min(
            32,
            max(
                1,
                triton.cdiv(
                    values.numel() * SPIKE_BLOCK, n_in * kernels_triton_push._BLOCK
                ),
            ),
        ),
    )


def _launch_spike_list(execution, x):
    """Consume local prefixes with the existing persistent packed fanout."""
    queue = execution.spikes
    n_batch, n_in = x.shape
    n_out = execution.output.shape[-1]
    # Several consumers per queue retain parallelism for small populations and
    # long fanouts; all consume the same packed kernel, without a global scan.
    programs = execution.programs
    kernels_triton._launch(
        execution.cache,
        "push_packed",
        kernels_triton_push._kernels(execution.cache)[0],
        x.device.index,
        (programs * queue.counts.numel(),),
        (
            execution.layout.crow,
            execution.layout.col,
            execution.layout.perm,
            execution.values
            if execution.task_values is None
            else execution.task_values,
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
        ),
        kernels_triton_push._PACKED_CONSTS,
        (kernels_triton_push._BLOCK, execution.task_values is None, SPIKE_BLOCK),
        kernels_triton_push._BLOCK_WARPS,
    )
    if queue.output is not execution.output:
        execution.output.copy_(queue.output)
    return execution.output


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
        return prepare_values(
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
