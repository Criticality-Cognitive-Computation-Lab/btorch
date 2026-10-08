"""Task-queued Triton kernels for source-major sparse propagation.

The implementation is intentionally private to the Triton push backend.  A
fixed source-major sparsity pattern is partitioned into small edge tasks.  A
device-side discovery kernel classifies active ``(batch, task)`` pairs into a
direct queue or a hash queue; consumer kernels then execute those queues.  The
hash consumer aggregates destinations in a worker-local table before merging
the table into the output.

Only float32 CUDA tensors with a two-dimensional dense input are supported.
All topology and workspace construction belongs in :func:`prepare`, so a
prepared state can be used from a CUDA-graph-captured region without allocating
or synchronising on the forward path.
"""

from __future__ import annotations

import importlib.util
import itertools
import weakref
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .backend import KernelCache


try:  # Optional dependency: importing the sparse runtime must work on CPU.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised without Triton installed
    triton = None
    tl = None


_TRITON_AVAILABLE = (
    importlib.util.find_spec("triton") is not None and triton is not None
)

# These are backend-private defaults.  They mirror PR-70's complete task
# algorithm without adding a second public sparse configuration surface.
_BLOCK_SOURCES = 32
_EDGE_BLOCK = 256
_LONG_FANOUT_THRESHOLD = 256
_HASH_CAPACITY = 512
_HASH_MAX_PROBE = 8
_HASH_MIN_EDGES = 64
_MAX_WORKERS = 128
_INT32_MAX = 2**31 - 1

_STATE_KEY = ("triton_tasks", "states")
_EPOCH = itertools.count(1)


def _next_epoch() -> int:
    return next(_EPOCH)


@dataclass
class TaskQueueWorkspace:
    """Reusable device storage for active queues and local hash tables."""

    queue_capacity: int
    worker_count: int
    direct_queue: Tensor
    hash_queue: Tensor
    counts: Tensor
    hash_keys: Tensor
    hash_values: Tensor


@dataclass
class PreparedTaskState:
    """Static task metadata and a reusable workspace for one sparse pattern."""

    sources: tuple[weakref.ReferenceType, ...]
    versions: tuple[int, ...]
    n_pre: int
    edge_count: int
    batch_capacity: int
    dtype: torch.dtype
    task_pre_indices: Tensor
    task_post_indices: Tensor
    packed_values: Tensor
    task_destination_perm: Tensor
    task_indptr: Tensor
    task_source_ids: Tensor
    hash_aggregation_mask: Tensor
    workspace: TaskQueueWorkspace
    layout_version: int
    layout_cache_key: tuple

    @property
    def task_count(self) -> int:
        """Return the number of static edge tasks."""
        return max(0, self.task_indptr.numel() - 1)


@dataclass
class TaskBinding:
    """Task values and queue workspace owned by one execution context."""

    task_state: PreparedTaskState
    task_values: Tensor
    workspace: TaskQueueWorkspace


@dataclass
class _TaskStateCache:
    """Prepared states and the persistent value buffers bound to them."""

    states: dict[tuple, PreparedTaskState] = field(default_factory=dict)
    bound_values: dict[tuple, tuple[weakref.ReferenceType, TaskBinding]] = field(
        default_factory=dict
    )


if _TRITON_AVAILABLE:

    @triton.jit
    def _pack_tasks_kernel(
        x_ptr,
        task_sources_ptr,
        task_hashable_ptr,
        direct_queue_ptr,
        hash_queue_ptr,
        counts_ptr,
        n_tasks: tl.constexpr,
        n_sources: tl.constexpr,
        block_sources: tl.constexpr,
        enable_hash: tl.constexpr,
    ):
        record = tl.program_id(0)
        task = record % n_tasks
        bucket = record // n_tasks
        source_offsets = tl.arange(0, block_sources)
        sources = tl.load(
            task_sources_ptr + task * block_sources + source_offsets,
        )
        source_mask = sources >= 0
        safe_sources = tl.where(source_mask, sources, 0)
        spikes = tl.load(x_ptr + bucket * n_sources + safe_sources)
        active = tl.sum(tl.where(source_mask & (spikes != 0.0), 1, 0), axis=0) > 0
        use_hash = enable_hash & tl.load(task_hashable_ptr + task)

        if active:
            queue_id = tl.where(use_hash, 1, 0)
            slot = tl.atomic_add(counts_ptr + queue_id, 1)
            queue_ptr = tl.where(
                use_hash,
                hash_queue_ptr + slot * 2,
                direct_queue_ptr + slot * 2,
            )
            tl.store(queue_ptr, bucket)
            tl.store(queue_ptr + 1, task)

    @triton.jit
    def _direct_tasks_kernel(
        x_ptr,
        weight_ptr,
        source_ptr,
        destination_ptr,
        task_indptr_ptr,
        queue_ptr,
        count_ptr,
        output_ptr,
        n_sources: tl.constexpr,
        n_destinations: tl.constexpr,
        block_edges: tl.constexpr,
        num_programs: tl.constexpr,
    ):
        worker = tl.program_id(0)
        task_slot = worker
        count = tl.load(count_ptr)
        edge_offsets = tl.arange(0, block_edges)
        while task_slot < count:
            bucket = tl.load(queue_ptr + task_slot * 2)
            task = tl.load(queue_ptr + task_slot * 2 + 1)
            begin = tl.load(task_indptr_ptr + task)
            end = tl.load(task_indptr_ptr + task + 1)
            edges = begin + edge_offsets
            edge_mask = edges < end
            sources = tl.load(source_ptr + edges, mask=edge_mask, other=0)
            destinations = tl.load(destination_ptr + edges, mask=edge_mask, other=0)
            weights = tl.load(weight_ptr + edges, mask=edge_mask, other=0.0)
            spikes = tl.load(x_ptr + bucket * n_sources + sources)
            tl.atomic_add(
                output_ptr + bucket * n_destinations + destinations,
                spikes * weights,
                mask=edge_mask & (spikes != 0.0),
            )
            task_slot += num_programs

    @triton.jit
    def _hash_tasks_kernel(
        x_ptr,
        weight_ptr,
        source_ptr,
        destination_ptr,
        task_indptr_ptr,
        queue_ptr,
        count_ptr,
        hash_keys_ptr,
        hash_values_ptr,
        output_ptr,
        n_sources: tl.constexpr,
        n_destinations: tl.constexpr,
        block_edges: tl.constexpr,
        hash_capacity: tl.constexpr,
        max_probe: tl.constexpr,
        num_programs: tl.constexpr,
    ):
        worker = tl.program_id(0)
        task_slot = worker
        count = tl.load(count_ptr)
        edge_offsets = tl.arange(0, block_edges)
        hash_offsets = tl.arange(0, hash_capacity)
        hash_base = worker * hash_capacity

        while task_slot < count:
            tl.store(hash_keys_ptr + hash_base + hash_offsets, -1)
            tl.store(hash_values_ptr + hash_base + hash_offsets, 0.0)
            tl.debug_barrier()

            bucket = tl.load(queue_ptr + task_slot * 2)
            task = tl.load(queue_ptr + task_slot * 2 + 1)
            begin = tl.load(task_indptr_ptr + task)
            end = tl.load(task_indptr_ptr + task + 1)
            edges = begin + edge_offsets
            edge_mask = edges < end
            sources = tl.load(source_ptr + edges, mask=edge_mask, other=0)
            destinations = tl.load(destination_ptr + edges, mask=edge_mask, other=-2)
            weights = tl.load(weight_ptr + edges, mask=edge_mask, other=0.0)
            spikes = tl.load(x_ptr + bucket * n_sources + sources)
            contributions = spikes * weights
            valid = edge_mask & (spikes != 0.0)

            unsigned_destinations = destinations.to(tl.uint32)
            slot = ((unsigned_destinations * 2654435761) & (hash_capacity - 1)).to(
                tl.int32
            )
            destination_slot = tl.full((block_edges,), -1, tl.int32)
            found = ~valid
            for _ in tl.static_range(max_probe):
                probe_slot = tl.where(found, tl.maximum(destination_slot, 0), slot)
                compare = tl.where(found, destinations, -1)
                old = tl.atomic_cas(
                    hash_keys_ptr + hash_base + probe_slot,
                    compare,
                    destinations,
                    sem="relaxed",
                    scope="cta",
                )
                success = (old == -1) | (old == destinations)
                newly_found = (~found) & success
                destination_slot = tl.where(newly_found, probe_slot, destination_slot)
                found |= success
                slot = (slot + 1) & (hash_capacity - 1)

            tl.atomic_add(
                hash_values_ptr + hash_base + destination_slot,
                contributions,
                mask=valid & (destination_slot >= 0),
                sem="relaxed",
                scope="cta",
            )
            # A bounded probe must never lose an edge: failed insertions bypass
            # the local table and use the global result directly.
            tl.atomic_add(
                output_ptr + bucket * n_destinations + destinations,
                contributions,
                mask=valid & (destination_slot < 0),
            )
            tl.debug_barrier()

            keys = tl.load(hash_keys_ptr + hash_base + hash_offsets)
            values = tl.load(hash_values_ptr + hash_base + hash_offsets)
            occupied = keys >= 0
            tl.atomic_add(
                output_ptr + bucket * n_destinations + keys,
                values,
                mask=occupied,
            )
            tl.debug_barrier()
            task_slot += num_programs


def _available() -> bool:
    """Return whether this module can launch Triton CUDA kernels."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


def _versions(*tensors: Tensor) -> tuple[int, ...] | None:
    try:
        return tuple(
            value
            for tensor in tensors
            for value in (tensor._version, tensor.data_ptr())
        )
    except RuntimeError:
        return None


def _state_matches(
    state: PreparedTaskState,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    batch_size: int,
    dtype: torch.dtype,
) -> bool:
    return (
        state.sources[0]() is t_crow
        and state.sources[1]() is t_col
        and state.sources[2]() is t_perm
        and state.batch_capacity >= batch_size
        and state.dtype == dtype
        and state.versions == _versions(t_crow, t_col, t_perm)
    )


def _compatible_workspace(
    workspace: TaskQueueWorkspace | None,
    *,
    queue_capacity: int,
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    return (
        workspace is not None
        and workspace.queue_capacity >= queue_capacity
        and workspace.direct_queue.device == device
        and workspace.hash_values.dtype == dtype
        and workspace.hash_keys.shape[1] == _HASH_CAPACITY
        and workspace.worker_count >= min(_MAX_WORKERS, max(1, queue_capacity))
    )


def _ensure_workspace(
    workspace: TaskQueueWorkspace | None,
    *,
    queue_capacity: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TaskQueueWorkspace:
    worker_count = min(_MAX_WORKERS, max(1, queue_capacity))
    if _compatible_workspace(
        workspace,
        queue_capacity=queue_capacity,
        device=device,
        dtype=dtype,
    ):
        return workspace
    queue_capacity = max(1, queue_capacity)
    direct_queue = torch.empty((queue_capacity, 2), device=device, dtype=torch.int32)
    return TaskQueueWorkspace(
        queue_capacity=queue_capacity,
        worker_count=worker_count,
        direct_queue=direct_queue,
        hash_queue=torch.empty_like(direct_queue),
        counts=torch.empty(2, device=device, dtype=torch.int32),
        hash_keys=torch.empty(
            (worker_count, _HASH_CAPACITY), device=device, dtype=torch.int32
        ),
        hash_values=torch.empty(
            (worker_count, _HASH_CAPACITY), device=device, dtype=dtype
        ),
    )


def _build_task_lists(
    t_crow: Tensor, t_col: Tensor
) -> tuple[list[int], list[int], list[list[int]], list[bool]]:
    """Build task edge ids and padded source tables on the host."""
    crow = t_crow.detach().to(device="cpu", dtype=torch.long).tolist()
    destination = t_col.detach().to(device="cpu", dtype=torch.long).tolist()
    n_source = len(crow) - 1
    source_for_edge = [
        source
        for source in range(n_source)
        for _ in range(crow[source + 1] - crow[source])
    ]
    raw_tasks: list[list[int]] = []

    for block_start in range(0, n_source, _BLOCK_SOURCES):
        block_end = min(n_source, block_start + _BLOCK_SOURCES)
        short_edges: list[int] = []
        for source in range(block_start, block_end):
            start, end = crow[source], crow[source + 1]
            source_edges = list(range(start, end))
            if len(source_edges) >= _LONG_FANOUT_THRESHOLD:
                raw_tasks.extend(
                    source_edges[offset : offset + _EDGE_BLOCK]
                    for offset in range(0, len(source_edges), _EDGE_BLOCK)
                )
            else:
                short_edges.extend(source_edges)
        if short_edges:
            raw_tasks.extend(
                short_edges[offset : offset + _EDGE_BLOCK]
                for offset in range(0, len(short_edges), _EDGE_BLOCK)
            )

    packed_edges: list[int] = []
    task_indptr = [0]
    task_sources: list[list[int]] = []
    task_hashable: list[bool] = []

    for task_edges in raw_tasks:
        task_edges = sorted(task_edges, key=destination.__getitem__)
        packed_edges.extend(task_edges)
        task_indptr.append(len(packed_edges))
        unique_sources = sorted({source_for_edge[edge] for edge in task_edges})
        if len(unique_sources) > _BLOCK_SOURCES:
            raise RuntimeError("A sparse task exceeds block_sources capacity.")
        task_sources.append(unique_sources)
        task_hashable.append(len(task_edges) >= _HASH_MIN_EDGES)

    return packed_edges, task_indptr, task_sources, task_hashable


def _task_cache(cache: KernelCache) -> _TaskStateCache:
    task_cache = cache.get(_STATE_KEY, _TaskStateCache)
    for key, state in list(task_cache.states.items()):
        if any(source() is None for source in state.sources):
            _unbind_state(task_cache, state)
            task_cache.states.pop(key, None)
    return task_cache


def invalidate_sources(cache: KernelCache, *sources: Tensor) -> None:
    """Release task states derived from any of the given topology buffers."""
    task_cache = cache.get(_STATE_KEY, _TaskStateCache)
    source_ids = {id(source) for source in sources}
    for key, state in list(task_cache.states.items()):
        if any(
            source() is not None and id(source()) in source_ids
            for source in state.sources
        ):
            _unbind_state(task_cache, state)
            task_cache.states.pop(key, None)


def _unbind_state(task_cache: _TaskStateCache, state: PreparedTaskState) -> None:
    for key, (_, task_binding) in list(task_cache.bound_values.items()):
        if task_binding.task_state is state:
            task_cache.bound_values.pop(key, None)


def _prune_bound_values(task_cache: _TaskStateCache) -> None:
    for key, (value_reference, task_binding) in list(task_cache.bound_values.items()):
        task_state = task_binding.task_state
        if (
            value_reference() is None
            or task_cache.states.get(task_state.layout_cache_key) is not task_state
        ):
            task_cache.bound_values.pop(key, None)


def _remember_state(
    task_cache: _TaskStateCache,
    key: tuple,
    state: PreparedTaskState,
) -> None:
    previous = task_cache.states.get(key)
    if previous is not None and previous is not state:
        _unbind_state(task_cache, previous)
    task_cache.states[key] = state


def _task_state(
    cache: KernelCache,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    batch_size: int,
    dtype: torch.dtype,
    route_key: object | None,
) -> PreparedTaskState | None:
    if (
        not _available()
        or not t_crow.is_cuda
        or t_crow.device != t_col.device
        or t_crow.device != t_perm.device
        or dtype != torch.float32
        or batch_size <= 0
        or t_crow.ndim != 1
        or t_col.ndim != 1
        or t_perm.ndim != 1
        or t_perm.numel() != t_col.numel()
        or t_crow.numel() == 0
        or t_col.numel() > _INT32_MAX
        or t_crow.numel() - 1 > _INT32_MAX
    ):
        return None

    task_cache = _task_cache(cache)
    states = task_cache.states
    key = (id(t_crow), id(t_col), id(t_perm), dtype, route_key)
    old = states.get(key)
    if old is not None and _state_matches(
        old, t_crow, t_col, t_perm, batch_size, dtype
    ):
        return old

    packed_edges, task_indptr, task_sources, task_hashable = _build_task_lists(
        t_crow, t_col
    )
    device = t_crow.device
    edge_perm = torch.tensor(packed_edges, device=device, dtype=torch.long)
    task_destination_perm = t_perm.index_select(0, edge_perm).to(torch.int32)
    task_pre_indices = torch.empty(edge_perm.shape, device=device, dtype=torch.int32)
    task_post_indices = t_col.index_select(0, edge_perm).to(torch.int32)
    packed_values = torch.empty(edge_perm.shape, device=device, dtype=dtype)
    source_ids = torch.full(
        (len(task_sources), _BLOCK_SOURCES),
        -1,
        device=device,
        dtype=torch.int32,
    )
    if task_sources:
        source_ids_cpu = torch.full(
            (len(task_sources), _BLOCK_SOURCES), -1, dtype=torch.int32
        )
        for task, sources in enumerate(task_sources):
            source_ids_cpu[task, : len(sources)] = torch.tensor(
                sources, dtype=torch.int32
            )
        source_ids.copy_(source_ids_cpu)

    n_pre = t_crow.numel() - 1
    source_ids_flat = torch.arange(n_pre, device="cpu", dtype=torch.long)
    crow_cpu = t_crow.detach().to(device="cpu", dtype=torch.long)
    counts = crow_cpu[1:] - crow_cpu[:-1]
    source_for_edge = torch.repeat_interleave(source_ids_flat, counts)
    if edge_perm.numel():
        task_pre_indices.copy_(
            source_for_edge.index_select(0, edge_perm.cpu()).to(device)
        )

    task_indptr_tensor = torch.tensor(task_indptr, device=device, dtype=torch.int32)
    task_hashable_tensor = torch.tensor(task_hashable, device=device, dtype=torch.bool)
    task_count = len(task_indptr) - 1
    queue_capacity = batch_size * max(1, task_count)
    if queue_capacity > _INT32_MAX:
        return None
    workspace = _ensure_workspace(
        None,
        queue_capacity=max(1, queue_capacity),
        device=device,
        dtype=dtype,
    )
    versions = _versions(t_crow, t_col, t_perm)
    if versions is None:
        return None
    state = PreparedTaskState(
        sources=(weakref.ref(t_crow), weakref.ref(t_col), weakref.ref(t_perm)),
        versions=versions,
        n_pre=n_pre,
        edge_count=t_col.numel(),
        batch_capacity=batch_size,
        dtype=dtype,
        task_pre_indices=task_pre_indices,
        task_post_indices=task_post_indices,
        packed_values=packed_values,
        task_destination_perm=task_destination_perm,
        task_indptr=task_indptr_tensor,
        task_source_ids=source_ids,
        hash_aggregation_mask=task_hashable_tensor,
        workspace=workspace,
        layout_version=_next_epoch(),
        layout_cache_key=key,
    )
    _remember_state(task_cache, key, state)
    return state


def prepare(
    cache: KernelCache,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    batch_size: int,
    dtype: torch.dtype,
    *,
    route_key: object | None = None,
) -> PreparedTaskState | None:
    """Prepare static tasks and reusable device workspace.

    Args:
        cache: Runtime cache that owns the non-serialised task state.
        t_crow: Source-major CSR row pointers, ``[n_source + 1]``.
        t_col: Source-major destination ids, ``[nnz]``.
        t_perm: Destination-CSR position of each source-major edge.
        batch_size: Maximum batch size for the reusable queue workspace.
        dtype: Dense value dtype; only ``torch.float32`` is supported.

    Returns:
        A prepared :class:`PreparedTaskState`, or ``None`` when the task backend cannot
        serve the tensors.
    """
    return _task_state(
        cache,
        t_crow,
        t_col,
        t_perm,
        batch_size,
        dtype,
        route_key,
    )


def layout_epoch(
    cache: KernelCache,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    batch_size: int,
    dtype: torch.dtype,
    *,
    route_key: object | None = None,
) -> int:
    """Return the prepared layout epoch without building missing state."""
    states = _task_cache(cache).states
    state = states.get((id(t_crow), id(t_col), id(t_perm), dtype, route_key))
    if state is None or not _state_matches(
        state, t_crow, t_col, t_perm, batch_size, dtype
    ):
        return 0
    return state.layout_version


def pack_values(state: PreparedTaskState, destination_major_values: Tensor) -> Tensor:
    """Pack destination-CSR values into the state task-edge order.

    Args:
        state: Prepared task state.
        destination_major_values: Differentiable ``[nnz]`` values in the current
            destination-CSR order.

    Returns:
        Contiguous task-order values. Its gradient maps back through
        ``t_perm`` when the caller uses it in a differentiable operation.
    """
    if (
        destination_major_values.ndim != 1
        or destination_major_values.numel() != state.edge_count
    ):
        raise ValueError(
            "destination_major_values must be a one-dimensional tensor with "
            f"{state.edge_count} entries."
        )
    if destination_major_values.dtype != state.dtype:
        raise TypeError(
            f"task values must have dtype {state.dtype}, "
            f"got {destination_major_values.dtype}."
        )
    return destination_major_values.index_select(
        0, state.task_destination_perm.to(torch.long)
    ).contiguous()


def bind_values(
    cache: KernelCache,
    state: PreparedTaskState,
    destination_major_values: Tensor,
) -> Tensor:
    """Refresh the persistent task-order value buffer once per trajectory."""
    task_cache = _task_cache(cache)
    if task_cache.states.get(state.layout_cache_key) is not state:
        _remember_state(task_cache, state.layout_cache_key, state)
    packed_values = state.packed_values
    packed_values.copy_(pack_values(state, destination_major_values))
    binding_key = (
        packed_values.device,
        packed_values.data_ptr(),
        packed_values.numel(),
        packed_values.dtype,
    )
    _prune_bound_values(task_cache)
    previous_binding = task_cache.bound_values.pop(binding_key, None)
    if previous_binding is not None and previous_binding[1] is not state:
        _prune_bound_values(task_cache)
    task_binding = TaskBinding(state, packed_values, state.workspace)
    task_cache.bound_values[binding_key] = (
        weakref.ref(packed_values),
        task_binding,
    )
    return packed_values


def create_task_values(
    cache: KernelCache,
    task_state: PreparedTaskState,
    destination_major_values: Tensor,
) -> Tensor:
    """Create independently owned values and workspace for concurrent use."""
    task_values = pack_values(task_state, destination_major_values)
    workspace = _ensure_workspace(
        None,
        queue_capacity=task_state.workspace.queue_capacity,
        device=task_values.device,
        dtype=task_values.dtype,
    )
    task_binding = TaskBinding(task_state, task_values, workspace)
    task_cache = _task_cache(cache)
    _remember_state(task_cache, task_state.layout_cache_key, task_state)
    binding_key = (
        task_values.device,
        task_values.data_ptr(),
        task_values.numel(),
        task_values.dtype,
    )
    task_cache.bound_values[binding_key] = (
        weakref.ref(task_values),
        task_binding,
    )
    return task_values


def state_for_values(
    cache: KernelCache,
    packed_values: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
) -> TaskBinding | None:
    """Return the prepared state only when values and topology both match."""
    task_cache = _task_cache(cache)
    binding_key = (
        packed_values.device,
        packed_values.data_ptr(),
        packed_values.numel(),
        packed_values.dtype,
    )
    _prune_bound_values(task_cache)
    binding = task_cache.bound_values.get(binding_key)
    if binding is None:
        return None
    value_reference, task_binding = binding
    task_state = task_binding.task_state
    if (
        value_reference() is not packed_values
        or task_cache.states.get(task_state.layout_cache_key) is not task_state
        or not _state_matches(
            task_state,
            t_crow,
            t_col,
            t_perm,
            batch_size=0,
            dtype=packed_values.dtype,
        )
    ):
        task_cache.bound_values.pop(binding_key, None)
        _prune_bound_values(task_cache)
        return None
    return task_binding


def restore_binding(
    cache: KernelCache,
    task_state: PreparedTaskState,
    packed_values: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
) -> None:
    """Restore an exact prepared binding after process-local cache eviction."""
    if packed_values is not task_state.packed_values:
        raise ValueError("packed_values does not belong to the supplied task state.")
    if not _state_matches(
        task_state,
        t_crow,
        t_col,
        t_perm,
        batch_size=0,
        dtype=packed_values.dtype,
    ):
        raise ValueError("the saved task state no longer matches the topology.")
    task_cache = _task_cache(cache)
    _remember_state(task_cache, task_state.layout_cache_key, task_state)
    binding_key = (
        packed_values.device,
        packed_values.data_ptr(),
        packed_values.numel(),
        packed_values.dtype,
    )
    task_binding = TaskBinding(
        task_state,
        packed_values,
        task_state.workspace,
    )
    task_cache.bound_values[binding_key] = (
        weakref.ref(packed_values),
        task_binding,
    )


def run(
    state: PreparedTaskState,
    packed_values: Tensor,
    x: Tensor,
    n_out: int,
    cache: KernelCache,
    *,
    workspace: TaskQueueWorkspace | None = None,
) -> Tensor:
    """Run queued direct and hash-aggregated source-major tasks.

    Args:
        state: Prepared task state for the sparse pattern and batch capacity.
        packed_values: Edge values in ``state`` task order.
        x: Float32 CUDA input with shape ``[batch, n_source]``.
        n_out: Number of destination neurons.
        cache: Runtime cache holding compiled Triton launch artefacts.

    Returns:
        Float32 CUDA output with shape ``[batch, n_out]``.

    Raises:
        TypeError: If the input contract is not supported by this kernel.
        ValueError: If the input shape or prepared capacity is incompatible.
    """
    if not _available():
        raise RuntimeError("Triton task kernels require Triton and CUDA.")
    if (
        x.ndim != 2
        or not x.is_cuda
        or x.dtype != torch.float32
        or x.device != state.task_pre_indices.device
    ):
        raise TypeError("task kernels require a float32 CUDA x with shape [B, N].")
    if x.shape[1] != state.n_pre:
        raise ValueError(f"expected x.shape[1] == {state.n_pre}, got {x.shape[1]}.")
    if x.shape[0] > state.batch_capacity:
        raise ValueError(
            f"prepared batch capacity is {state.batch_capacity}, got {x.shape[0]}."
        )
    if (
        packed_values.ndim != 1
        or packed_values.numel() != state.edge_count
        or packed_values.dtype != torch.float32
        or packed_values.device != x.device
    ):
        raise TypeError("packed_values must be a float32 CUDA tensor in task order.")
    if not x.is_contiguous() or not packed_values.is_contiguous():
        raise ValueError("task kernels require contiguous x and packed_values.")
    if n_out <= 0 or n_out > _INT32_MAX:
        raise ValueError(f"n_out must be in [1, {_INT32_MAX}], got {n_out}.")

    n_buckets = x.shape[0]
    n_tasks = state.task_count
    output = torch.zeros((n_buckets, n_out), device=x.device, dtype=x.dtype)
    if n_buckets == 0 or n_tasks == 0:
        return output

    workspace = state.workspace if workspace is None else workspace
    counts = workspace.counts
    counts.zero_()
    queue_records = n_buckets * n_tasks
    _pack_tasks_kernel[(queue_records,)](
        x,
        state.task_source_ids,
        state.hash_aggregation_mask,
        workspace.direct_queue,
        workspace.hash_queue,
        counts,
        n_tasks=n_tasks,
        n_sources=state.n_pre,
        block_sources=_BLOCK_SOURCES,
        enable_hash=True,
        num_warps=1,
    )

    worker_count = workspace.worker_count
    _direct_tasks_kernel[(worker_count,)](
        x,
        packed_values,
        state.task_pre_indices,
        state.task_post_indices,
        state.task_indptr,
        workspace.direct_queue,
        counts,
        output,
        n_sources=state.n_pre,
        n_destinations=n_out,
        block_edges=_EDGE_BLOCK,
        num_programs=worker_count,
        num_warps=4,
    )
    _hash_tasks_kernel[(worker_count,)](
        x,
        packed_values,
        state.task_pre_indices,
        state.task_post_indices,
        state.task_indptr,
        workspace.hash_queue,
        counts[1:],
        workspace.hash_keys,
        workspace.hash_values,
        output,
        n_sources=state.n_pre,
        n_destinations=n_out,
        block_edges=_EDGE_BLOCK,
        hash_capacity=_HASH_CAPACITY,
        max_probe=_HASH_MAX_PROBE,
        num_programs=worker_count,
        num_warps=4,
    )
    return output


__all__ = [
    "bind_values",
    "create_task_values",
    "invalidate_sources",
    "layout_epoch",
    "pack_values",
    "prepare",
    "restore_binding",
    "run",
    "state_for_values",
]
