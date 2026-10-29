"""Share segmented CSR and sparse-VJP preparation across GPU renderers.

Execution objects compile and submit named mathematical stages.
Topology, sample layouts, split-row ownership and reusable destinations
remain identical across backends; no renderer selects a different sparse
algorithm.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from ...utils.kernel import tensor_layout
from . import csr_layout as csr, kernels_torch, source_layout, task_layout as tasks
from .activity import GradientList
from .backend import KernelCache


def stage_outputs(name):
    """Declare mutable ABI slots shared by both sparse renderers."""
    return {
        "csr_matvec": (6, 7, 18),
        "hub_combine": (3,),
        "edge_grad": (4,),
        "listed_output": (2,),
        "pack_spikes": (2, 3, 4, 5),
        "push_packed": (8,),
        "pack_tasks": (3, 4, 5),
        "direct_tasks": (7,),
        "hash_tasks": (7, 8, 9),
        "push_dense": (6,),
    }[name]


def stage_inputs(name, constants=()):
    """Declare the invocation slots whose storage may change between steps."""
    if name == "csr_matvec":
        return (
            (5,)
            + ((15,) if constants[5] else ())
            + ((17,) if constants[6] else ())
            + ((19, 20) if constants[7] else ())
        )
    return {
        "hub_combine": (),
        "edge_grad": (2, 3),
        "listed_output": (1,),
        "push_dense": (4, 6),
        "push_packed": (4, 5, 6, 7, 8),
        "pack_spikes": (0,),
        "pack_tasks": (0,),
        "direct_tasks": (0,),
        "hash_tasks": (0,),
    }[name]


def stage_work(name, args, grid):
    """Estimate mathematical work without inspecting device tensor values."""
    match name:
        case "csr_matvec":
            return args[2].numel() * args[10] * (grid[1] // args[11])
        case "edge_grad":
            return args[5] * args[6] * grid[1]
        case "hub_combine":
            return args[4] * grid[1]
        case "listed_output":
            return args[2].numel()
        case _:
            raise KeyError(name)


def _check_input(value, expected):
    if tensor_layout(value) != expected:
        raise ValueError("Sparse input layout/dtype changed; prepare a new execution")


@dataclass(frozen=True)
class _WeightGradient:
    source: Tensor
    destination: Tensor


@dataclass(frozen=True)
class _PreparedMatvec:
    """Bind a product and whether it also accumulates its value VJP."""

    forward: Callable
    accumulates_values: bool = False

    def __call__(self, input, mask=None, activity=None):
        return self.forward(input, mask, activity)


def _prepare_matvec(
    execution,
    cache: KernelCache,
    crow: Tensor,
    col: Tensor,
    values: Tensor,
    x: Tensor,
    perm: Tensor | None = None,
    x_minor: Tensor | None = None,
    input_mask: Tensor | GradientList | None = None,
    weight_gradient: _WeightGradient | None = None,
) -> _PreparedMatvec:
    """Prepare reusable destinations for ``csr_matvec`` or its gather variant.

    ``x_minor`` is the sample-minor layout of ``x`` if the caller already has
    it (the backward shares one transposed gradient between its kernels).
    """
    input_layout = tensor_layout(x)

    def reference(input, mask, activity):
        _check_input(input, input_layout)
        return kernels_torch.csr_matvec(crow, col, values, input)

    n_vb = values.ndim - 1
    pattern = None
    if x.is_cuda and (x.ndim > n_vb):
        pattern = csr._pattern(cache, crow, col)
    if pattern is not None and pattern.supported:
        n_out, n_edge = (pattern.n_out, pattern.n_edge)
        n_in = x.shape[-1]
        if n_vb:
            v_batch, x_batch = (values.shape[:-1], x.shape[:n_vb])
            sample = x.shape[n_vb:-1]
            if v_batch == x_batch:
                batch = v_batch
                v_member = x_member = n_member = csr._numel(batch)
            else:
                batch = torch.broadcast_shapes(v_batch, x_batch)
                v_member, x_member = (csr._numel(v_batch), csr._numel(x_batch))
                n_member = csr._numel(batch)
        else:
            batch = ()
            sample = x.shape[:-1]
            v_member = x_member = n_member = 1
        n_sample = csr._numel(sample)
        rows, entries, samples, warps = csr._tile(csr._MATVEC_TILES, n_sample)
        if (
            isinstance(input_mask, GradientList)
            and n_vb == 0
            and (weight_gradient is None or n_sample == 1)
        ):
            rows, entries, samples, warps = csr._tile(csr._MATVEC_TILES, 1)
        perm32 = None if perm is None else pattern.permutation(perm)
        supported = (
            csr._on_current_device(pattern, values, x)
            and values.dtype in kernels_torch._FLOAT_DTYPES
            and (x.dtype in kernels_torch._FLOAT_DTYPES)
            and (values.shape[-1] == n_edge if perm is None else perm32 is not None)
            and (values.shape[-1] < csr._INT32_MAX)
            and (n_in * max(n_sample, 1) + samples < csr._INT32_MAX)
            and (v_member == n_member or v_member == 1)
            and (x_member == n_member or x_member == 1)
            and (csr._SPMM_MIN_SAMPLES is None or n_sample < csr._SPMM_MIN_SAMPLES)
        )
    else:
        supported = False
    if not supported:
        if perm is not None:
            values = values.index_select(-1, perm)
        return _PreparedMatvec(reference)
    out_shape = (*batch, *sample, n_out)
    output_dtype = torch.promote_types(values.dtype, x.dtype)
    if n_out == 0 or n_member * n_sample == 0 or n_edge == 0:
        empty = torch.zeros(out_shape, dtype=output_dtype, device=x.device)
        layout = tensor_layout(x)

        def empty_product(input, mask, activity):
            _check_input(input, layout)
            return empty

        return _PreparedMatvec(empty_product)
    layout = pattern.rows
    if layout is None:
        layout = pattern.row_layout(crow)
    gradient_list = input_mask if isinstance(input_mask, GradientList) else None
    if gradient_list is not None:
        input_mask = gradient_list.mask
    listed = (
        gradient_list is not None
        and n_vb == 0
        and (weight_gradient is None or n_sample == 1)
    )
    hub = layout.segmented and (not listed)
    n_sample_block = 1 if listed else -(-n_sample // samples)
    if n_member * (n_sample if hub else n_sample_block) > csr._MAX_GRID_Y:
        if perm is not None:
            values = values.index_select(-1, perm)
        return _PreparedMatvec(reference)
    values = values.contiguous()
    prepare_input = csr._prepare_sample_minor(x, n_sample) if x_minor is None else None
    out = torch.empty(out_shape, dtype=output_dtype, device=x.device)
    accumulate_values = (
        weight_gradient is not None
        and perm32 is not None
        and (n_vb == 0)
        and (n_sample_block == 1)
        and (not listed or n_sample == 1)
        and weight_gradient.destination.is_contiguous()
        and (weight_gradient.source.dtype in kernels_torch._FLOAT_DTYPES)
    )
    prepare_activity = (
        csr._prepare_sample_minor(weight_gradient.source, n_sample)
        if accumulate_values
        else None
    )
    tmp = out
    if hub:
        tmp = torch.empty(
            (n_member * n_sample, layout.n_tmp),
            dtype=kernels_torch._accumulation_dtype(output_dtype),
            device=x.device,
        )
    if listed:
        tmp = torch.empty(
            (n_sample, layout.n_seg),
            dtype=kernels_torch._accumulation_dtype(out.dtype),
            device=out.device,
        )

    def forward(input, mask, activity):
        if tensor_layout(input) != input_layout:
            raise ValueError(
                "Sparse input layout/dtype changed; prepare a new execution"
            )
        queue = mask if isinstance(mask, GradientList) else None
        mask = queue.mask if queue is not None else mask
        indices, counts = queue.work(accumulate_values) if listed else (out, out)
        source = x_minor if x_minor is not None else prepare_input(input)
        activity_minor = (
            source if prepare_activity is None else prepare_activity(activity)
        )
        n_seg = gradient_list.capacity if listed else layout.n_seg
        execution.launch(
            "csr_matvec",
            (
                -(-n_seg // rows),
                n_member * n_sample_block,
                min(csr._LIST_PARTITIONS, max(1, -(-n_edge // csr._SEGMENT)))
                if listed
                else 1,
            ),
            (
                layout.crow32 if listed else layout.ptr,
                layout.target if hub else layout.ptr,
                pattern.col32,
                pattern.col32 if perm32 is None else perm32,
                values,
                source,
                tmp if listed else out,
                tmp,
                n_seg,
                n_out,
                n_sample,
                n_sample_block,
                layout.n_seg if listed else layout.n_tmp,
                0 if v_member == 1 else values.shape[-1],
                0 if x_member == 1 else n_sample * n_in,
                source if input_mask is None else mask,
                layout.row_segment_start
                if listed
                else layout.ptr
                if layout.owner is None
                else layout.owner,
                activity_minor,
                out if not accumulate_values else weight_gradient.destination,
                indices,
                counts,
            ),
            ("BR", "BK", "BS", "HUB", "PERM", "MASK", "WEIGHT_GRAD", "LIST", "SEGMENT"),
            (
                rows,
                entries,
                samples,
                hub,
                perm32 is not None,
                input_mask is not None,
                accumulate_values,
                listed,
                csr._SEGMENT,
            ),
            warps,
        )
        if listed:
            execution.launch(
                "listed_output",
                (-(-out.numel() // csr._COMBINE_TILE[0]),),
                (tmp, mask, out, layout.row_segment_start),
                ("N", "N_ROWS", "N_SEG", "BH", "BK"),
                (out.numel(), n_out, layout.n_seg, *csr._COMBINE_TILE),
                1,
            )
        if hub:
            execution.launch(
                "hub_combine",
                (-(-layout.n_hub // csr._COMBINE_TILE[0]), n_member * n_sample),
                (
                    layout.hub_row,
                    layout.hub_ptr,
                    tmp,
                    out,
                    layout.n_hub,
                    n_out,
                    layout.n_tmp,
                ),
                ("BH", "BK"),
                csr._COMBINE_TILE,
                1,
            )
        return out

    return _PreparedMatvec(forward, accumulate_values)


def _prepare_edge_grad(
    execution,
    cache: KernelCache,
    crow: Tensor,
    col: Tensor,
    grad: Tensor,
    x: Tensor,
    n_vb: int,
    grad_minor: Tensor | None = None,
    destination: Tensor | None = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    """``edge_grad``; ``grad_minor`` is the sample-minor layout of ``grad`` if
    the caller already has it."""
    gradient_layout, input_layout = tensor_layout(grad), tensor_layout(x)
    supported = False
    if grad.is_cuda and (grad.ndim > n_vb) and (x.ndim > n_vb):
        pattern = csr._pattern(cache, crow, col)
        if pattern.supported:
            n_out, n_edge = (pattern.n_out, pattern.n_edge)
            n_in = x.shape[-1]
            batch = grad.shape[:n_vb]
            n_member = csr._numel(batch)
            x_member = csr._numel(x.shape[:n_vb])
            n_sample = csr._numel(grad.shape[n_vb:-1])
            entries, samples, warps = csr._tile(csr._EDGE_TILES, n_sample)
            supported = (
                csr._on_current_device(pattern, grad, x)
                and grad.dtype in kernels_torch._FLOAT_DTYPES
                and (x.dtype in kernels_torch._FLOAT_DTYPES)
                and (grad.shape[-1] == n_out)
                and (max(n_out, n_in) * (n_sample + samples) < csr._INT32_MAX)
                and (x_member == n_member or x_member == 1)
                and (csr._numel(x.shape[n_vb:-1]) == n_sample)
                and (n_member <= csr._MAX_GRID_Y)
                and (
                    destination is None
                    or (
                        destination.shape == (*batch, n_edge)
                        and destination.is_contiguous()
                    )
                )
            )
    if not supported:

        def reference(gradient, input):
            _check_input(gradient, gradient_layout)
            _check_input(input, input_layout)
            result = kernels_torch.edge_grad(crow, col, gradient, input, n_vb)
            if destination is None:
                return result
            destination.add_(
                result.sum_to_size(destination.shape).to(destination.dtype)
            )
            return None

        return reference
    out_shape = (*batch, n_edge)
    if n_edge == 0 or n_member == 0 or n_sample == 0:
        out = torch.zeros(out_shape, dtype=grad.dtype, device=grad.device)

        def empty_gradient(gradient, input):
            _check_input(gradient, gradient_layout)
            _check_input(input, input_layout)
            return out if destination is None else None

        return empty_gradient
    row32 = pattern.row32
    if row32 is None:
        row32 = pattern.row_index(crow)
    prepare_grad = (
        csr._prepare_sample_minor(grad, n_sample) if grad_minor is None else None
    )
    prepare_input = csr._prepare_sample_minor(x, n_sample)
    out = (
        torch.empty(out_shape, dtype=grad.dtype, device=grad.device)
        if destination is None
        else destination
    )

    def forward(gradient, input):
        _check_input(gradient, gradient_layout)
        _check_input(input, input_layout)
        gradient = grad_minor if grad_minor is not None else prepare_grad(gradient)
        input = prepare_input(input)
        execution.launch(
            "edge_grad",
            (-(-n_edge // entries), n_member),
            (
                row32,
                pattern.col32,
                gradient,
                input,
                out,
                n_edge,
                n_sample,
                n_sample * n_out,
                0 if x_member == 1 else n_sample * n_in,
            ),
            ("BE", "BS", "ACCUMULATE"),
            (entries, samples, destination is not None),
            warps,
        )
        return out if destination is None else None

    return forward


def _prepare_backward(
    execution,
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
    input_mask: Tensor | None = None,
    grad_values: Tensor | None = None,
):
    n_vb = values.ndim - 1
    grad_minor = None
    prepare_grad = None
    if (
        need_values
        and need_x
        and grad.is_cuda
        and (grad.dtype in kernels_torch._FLOAT_DTYPES)
        and (grad.ndim > n_vb + 1)
        and (grad.numel() > 0)
    ):
        n_sample = csr._numel(grad.shape[n_vb:-1])
        if n_sample > 1:
            prepare_grad = csr._prepare_sample_minor(grad, n_sample)
            grad_minor = prepare_grad(grad)
    input_backward = (
        _prepare_matvec(
            execution,
            cache,
            t_crow,
            t_col,
            values,
            grad,
            t_perm,
            grad_minor,
            input_mask,
            _WeightGradient(x, grad_values)
            if need_values and grad_values is not None
            else None,
        )
        if need_x
        else None
    )
    fused_values = input_backward is not None and input_backward.accumulates_values
    values_backward = (
        _prepare_edge_grad(
            execution, cache, crow, col, grad, x, n_vb, grad_minor, grad_values
        )
        if need_values and (not fused_values)
        else None
    )

    def backward(input, gradient, mask=None):
        if prepare_grad is not None:
            prepare_grad(gradient)
        value_gradient = (
            values_backward(gradient, input) if values_backward is not None else None
        )
        grad_x = (
            input_backward(gradient, mask, input)
            if input_backward is not None
            else None
        )
        return (value_gradient, grad_x)

    return backward


def prepare_tasks(
    execution,
    state: tasks.PreparedTaskState,
    packed_values: Tensor,
    x: Tensor,
    n_out: int,
    cache: KernelCache,
    *,
    workspace: tasks.TaskQueueWorkspace | None = None,
):
    """Prepare reusable output storage and direct/hash task queues.

    Args:
        state: Prepared task state for the sparse pattern and batch capacity.
        packed_values: Edge values in ``state`` task order.
        x: Floating CUDA input with shape ``[batch, n_source]``.
        n_out: Number of destination neurons.
        cache: Runtime cache holding derived layouts and workspaces.

    Returns:
        Callback accepting the prepared input layout and returning the reusable
        promoted floating CUDA output with shape ``[batch, n_out]``.

    Raises:
        TypeError: If the input contract is not supported by this kernel.
        ValueError: If the input shape or prepared capacity is incompatible.
    """
    if (
        x.ndim != 2
        or not x.is_cuda
        or x.dtype not in kernels_torch._FLOAT_DTYPES
        or x.device != state.task_pre_indices.device
    ):
        raise TypeError("task kernels require a floating CUDA x with shape [B, N].")
    if x.shape[1] != state.n_pre:
        raise ValueError(f"expected x.shape[1] == {state.n_pre}, got {x.shape[1]}.")
    if x.shape[0] > state.batch_capacity:
        raise ValueError(
            f"prepared batch capacity is {state.batch_capacity}, got {x.shape[0]}."
        )
    if (
        packed_values.ndim != 1
        or packed_values.numel() != state.edge_count
        or packed_values.dtype not in kernels_torch._FLOAT_DTYPES
        or packed_values.device != x.device
    ):
        raise TypeError("packed_values must be a floating CUDA tensor in task order.")
    if not packed_values.is_contiguous():
        raise ValueError("task values must be contiguous in their prepared edge order.")
    if n_out <= 0 or n_out > tasks._INT32_MAX:
        raise ValueError(f"n_out must be in [1, {tasks._INT32_MAX}], got {n_out}.")

    n_buckets = x.shape[0]
    n_tasks = state.task_count
    output_dtype = torch.promote_types(x.dtype, packed_values.dtype)
    compute_dtype = kernels_torch._accumulation_dtype(output_dtype)
    output = torch.empty((n_buckets, n_out), device=x.device, dtype=compute_dtype)
    if n_buckets == 0 or n_tasks == 0:
        output.zero_()
        result = output.to(output_dtype)
        layout = tensor_layout(x)

        def empty_tasks(input):
            _check_input(input, layout)
            return result

        return empty_tasks

    workspace = state.workspace if workspace is None else workspace
    # A live producer can have greater precision than the packed weights.
    # Retain each precision's hash storage instead of reallocating on steps
    # or when a caller returns to a previously captured input dtype.
    hash_values = workspace.hash_values.get(compute_dtype)
    if hash_values is None:
        hash_values = torch.empty_like(workspace.hash_keys, dtype=compute_dtype)
        workspace.hash_values[compute_dtype] = hash_values
    counts = workspace.counts
    hash_count = counts[1:]
    queue_records = n_buckets * n_tasks
    worker_count = workspace.worker_count
    batch_stride, source_stride = x.stride()
    result = (
        output
        if output_dtype == compute_dtype
        else torch.empty_like(output, dtype=output_dtype)
    )

    input_layout = tensor_layout(x)

    pack_arguments = list(
        (
            x,
            state.task_source_ids,
            state.hash_aggregation_mask,
            workspace.direct_queue,
            workspace.hash_queue,
            counts,
        )
    )
    pack_stage = execution.prepare(
        "pack_tasks",
        (queue_records,),
        pack_arguments,
        (
            "n_tasks",
            "x_batch_stride",
            "x_source_stride",
            "block_sources",
            "enable_hash",
        ),
        (n_tasks, batch_stride, source_stride, tasks._BLOCK_SOURCES, True),
        1,
        changing=(0,),
    )
    direct_arguments = list(
        (
            x,
            packed_values,
            state.task_pre_indices,
            state.task_post_indices,
            state.task_indptr,
            workspace.direct_queue,
            counts,
            output,
        )
    )
    direct_stage = execution.prepare(
        "direct_tasks",
        (worker_count,),
        direct_arguments,
        (
            "x_batch_stride",
            "x_source_stride",
            "n_destinations",
            "block_edges",
            "num_programs",
        ),
        (batch_stride, source_stride, n_out, tasks._EDGE_BLOCK, worker_count),
        4,
        changing=(0,),
    )
    hash_arguments = list(
        (
            x,
            packed_values,
            state.task_pre_indices,
            state.task_post_indices,
            state.task_indptr,
            workspace.hash_queue,
            hash_count,
            workspace.hash_keys,
            hash_values,
            output,
        )
    )
    hash_stage = execution.prepare(
        "hash_tasks",
        (worker_count,),
        hash_arguments,
        (
            "x_batch_stride",
            "x_source_stride",
            "n_destinations",
            "block_edges",
            "hash_capacity",
            "max_probe",
            "num_programs",
        ),
        (
            batch_stride,
            source_stride,
            n_out,
            tasks._EDGE_BLOCK,
            tasks._HASH_CAPACITY,
            tasks._HASH_MAX_PROBE,
            worker_count,
        ),
        4,
        changing=(0,),
    )

    def forward(input):
        if tensor_layout(input) != input_layout:
            raise ValueError(
                "Sparse input layout/dtype changed; prepare a new execution"
            )
        output.zero_()
        counts.zero_()
        pack_arguments[0] = input
        pack_stage(pack_arguments)
        direct_arguments[0] = input
        direct_stage(direct_arguments)
        hash_arguments[0] = input
        hash_stage(hash_arguments)
        if result is not output:
            result.copy_(output)
        return result

    return forward


def run(execution, state, packed_values, x, n_out, cache, *, workspace=None):
    """Execute once; recurrent routes use the reusable prepared callback."""
    return prepare_tasks(
        execution, state, packed_values, x, n_out, cache, workspace=workspace
    )(x)


def prepare_task_push(
    execution,
    cache,
    t_crow,
    t_col,
    t_perm,
    values,
    x,
    n_out,
    *,
    task_values=None,
    require_task=True,
):
    """Bind source-task values and workspace once for a recurrent
    invocation."""
    binding = (
        None
        if task_values is None
        else tasks.state_for_values(cache, task_values, t_crow, t_col, t_perm)
    )
    if binding is None:
        raise RuntimeError("The task route requires prepared task-order values")
    return prepare_tasks(
        execution,
        binding.task_state,
        task_values,
        x,
        n_out,
        cache,
        workspace=binding.workspace,
    )


def _launch_dense(
    execution,
    cache: KernelCache,
    layout: source_layout._Layout,
    values: Tensor,
    x: Tensor,
    n_out: int,
    permute: bool,
) -> Tensor:
    n_batch, n_in = x.shape
    output_dtype = torch.promote_types(values.dtype, x.dtype)
    out = torch.zeros(
        n_batch,
        n_out,
        dtype=kernels_torch._accumulation_dtype(output_dtype),
        device=x.device,
    )
    n_tiles = (n_in + 64 - 1) // 64
    grid = n_batch * (n_tiles + layout.n_hub)
    if grid > csr._INT32_MAX:
        raise RuntimeError("spike_push_dense: batch too large for one launch.")
    execution.launch(
        "push_dense",
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
            values.stride(0),
            x.stride(0),
            x.stride(1),
        ),
        ("TILE", "BLOCK", "PERMUTE"),
        (64, 64, permute),
        2,
    )
    return out.to(output_dtype)


def packed_push(
    execution,
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
    """Propagate packed non-zero inputs with persistent CUDA consumers.

    Drop-in replacement of
    :func:`btorch.sparse.runtime.kernels_torch.spike_push` for floating CUDA
    tensors; other inputs are forwarded to it. The whole batch is served by
    one launch. Accumulation uses float atomics, so the result matches the
    reference up to accumulation rounding and is not bit-reproducible.

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
    layout = (
        source_layout._layout(cache, t_crow, t_col, t_perm)
        if source_layout._supported(values, x, t_crow, t_col)
        else None
    )
    n_active = active_idx.shape[0]
    if (
        layout is None
        or max(n_active, n_out) > csr._INT32_MAX
        # The packed kernel reads both index lists as int64 on the device.
        or active_idx.dtype != torch.long
        or ptr.dtype != torch.long
        or active_idx.device != x.device
        or ptr.device != x.device
        or ptr.shape[0] != x.shape[0] + 1
    ):
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
    n_batch, n_in = x.shape
    if n_active == 0:
        return torch.zeros(
            n_batch,
            n_out,
            dtype=torch.promote_types(values.dtype, x.dtype),
            device=x.device,
        )
    if source_layout._prefer_dense(layout, n_active, x.numel()):
        return _launch_dense(
            execution, cache, layout, values, x, n_out, not source_major_values
        )
    output_dtype = torch.promote_types(values.dtype, x.dtype)
    out = torch.zeros(
        n_batch,
        n_out,
        dtype=kernels_torch._accumulation_dtype(output_dtype),
        device=x.device,
    )
    # Inclusive prefix sum of the active out-degrees; its last element is the
    # total work, which only the kernel reads (no host synchronisation).
    active_idx, ptr = active_idx.contiguous(), ptr.contiguous()
    cum = torch.cumsum(layout.degree[active_idx], 0)
    # ``n_active`` is known on the host for free, so the number of persistent
    # programs follows the *expected* work; the actual work is split evenly
    # among them whatever the degrees of the active sources are.
    expected_blocks = int(n_active * layout.mean_degree / source_layout.EDGE_BLOCK) + 1
    n_programs = min(1024, expected_blocks)
    execution.launch(
        "push_packed",
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
            values.stride(0),
            x.stride(0),
            x.stride(1),
        ),
        ("BLOCK", "PERMUTE", "SPIKE_BLOCK"),
        (source_layout.EDGE_BLOCK, not source_major_values, 0),
        4,
    )
    return out.to(output_dtype)
