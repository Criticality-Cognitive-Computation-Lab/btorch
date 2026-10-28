"""Shared producer/consumer storage contract for sparse CUDA propagation."""

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ...utils.kernel import tensor_layout
from . import csr_layout, kernels_torch, source_layout
from .activity import BLOCK_SIZE
from .backend import KernelCache


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
    layout = source_layout._layout(cache, t_crow, t_col, t_perm)
    if (
        layout is None
        or batch_size > csr_layout._MAX_GRID_Y
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
        layout.source_value_epoch = csr_layout._next_epoch()
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
    cache: KernelCache
    input_layout: tuple
    programs: int = 1
    pack: object = field(default=None, init=False, repr=False)
    launch: object = field(default=None, init=False, repr=False)
    arguments: list = field(default_factory=list, init=False, repr=False)

    def _check_input(self, input):
        if tensor_layout(input) != self.input_layout:
            raise ValueError(
                "Sparse input layout/dtype changed; prepare a new execution"
            )

    def __call__(self, input):
        self._check_input(input)
        if self.spikes is None:
            return self.output
        self.pack(input)
        return self._consume(input)

    def _consume(self, input):
        self.arguments[4] = input
        self.launch(self.arguments)
        if self.spikes.output is not self.output:
            self.output.copy_(self.spikes.output)
        return self.output

    def from_spikes(self, input):
        """Consume an emitted list; its producer already cleared the output."""
        self._check_input(input)
        return self.output if self.spikes is None else self._consume(input)


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
    bind,
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
        return _PreparedSpikeList(
            None, values, None, None, output, cache, tensor_layout(x)
        )
    layout = source_layout._layout(cache, t_crow, t_col, t_perm)
    if layout is None or x.numel() > 2**31 - 1:
        raise RuntimeError("Unsupported spike-list layout; replan the connection")
    groups = -(-x.numel() // SPIKE_BLOCK)
    queue = SpikeList(
        torch.empty(groups * SPIKE_BLOCK, device=x.device, dtype=torch.int32),
        torch.empty(groups * SPIKE_BLOCK, device=x.device, dtype=torch.int64),
        torch.empty(groups, device=x.device, dtype=torch.int32),
        layout.degree,
        accumulation,
        n_in,
    )
    execution = _PreparedSpikeList(
        queue,
        values,
        layout,
        task_values,
        output,
        cache,
        tensor_layout(x),
        # Size from topology, not measured activity. Sparse blocks need enough
        # consumers to hide dependent prefix/edge loads even when neighboring
        # producer blocks are silent. Persistent consumers cover arbitrary
        # degrees, including hubs; this is not a capacity bound.
        min(
            32,
            max(
                1,
                (-(-values.numel() * SPIKE_BLOCK // (n_in * source_layout.EDGE_BLOCK))),
            ),
        ),
    )
    bind(execution, x)
    return execution
