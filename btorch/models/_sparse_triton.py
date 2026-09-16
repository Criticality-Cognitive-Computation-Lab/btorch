"""Event-driven sparse matrix multiplication implemented with Triton."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import torch

from .._sparse_config import TritonSparseConfig


_TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None

if _TRITON_AVAILABLE:
    import triton
    import triton.language as tl

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
        BLOCK_SOURCES: tl.constexpr,
        ENABLE_HASH: tl.constexpr,
    ):
        record = tl.program_id(0)
        task = record % n_tasks
        bucket = record // n_tasks
        source_offsets = tl.arange(0, BLOCK_SOURCES)
        sources = tl.load(
            task_sources_ptr + task * BLOCK_SOURCES + source_offsets
        )
        source_mask = sources >= 0
        safe_sources = tl.where(source_mask, sources, 0)
        spikes = tl.load(x_ptr + bucket * n_sources + safe_sources)
        active = tl.sum(
            tl.where(source_mask & (spikes != 0.0), 1, 0), axis=0
        ) > 0
        use_hash = ENABLE_HASH & tl.load(task_hashable_ptr + task)

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
        BLOCK_EDGES: tl.constexpr,
        NUM_PROGRAMS: tl.constexpr,
    ):
        worker = tl.program_id(0)
        task_slot = worker
        count = tl.load(count_ptr)
        edge_offsets = tl.arange(0, BLOCK_EDGES)
        while task_slot < count:
            bucket = tl.load(queue_ptr + task_slot * 2)
            task = tl.load(queue_ptr + task_slot * 2 + 1)
            begin = tl.load(task_indptr_ptr + task)
            end = tl.load(task_indptr_ptr + task + 1)
            edges = begin + edge_offsets
            edge_mask = edges < end
            sources = tl.load(source_ptr + edges, mask=edge_mask, other=0)
            destinations = tl.load(
                destination_ptr + edges, mask=edge_mask, other=0
            )
            weights = tl.load(weight_ptr + edges, mask=edge_mask, other=0.0)
            spikes = tl.load(x_ptr + bucket * n_sources + sources)
            tl.atomic_add(
                output_ptr + bucket * n_destinations + destinations,
                spikes * weights,
                mask=edge_mask & (spikes != 0.0),
            )
            task_slot += NUM_PROGRAMS

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
        BLOCK_EDGES: tl.constexpr,
        HASH_CAPACITY: tl.constexpr,
        MAX_PROBE: tl.constexpr,
        NUM_PROGRAMS: tl.constexpr,
    ):
        worker = tl.program_id(0)
        task_slot = worker
        count = tl.load(count_ptr)
        edge_offsets = tl.arange(0, BLOCK_EDGES)
        hash_offsets = tl.arange(0, HASH_CAPACITY)
        hash_base = worker * HASH_CAPACITY

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
            destinations = tl.load(
                destination_ptr + edges, mask=edge_mask, other=-2
            )
            weights = tl.load(weight_ptr + edges, mask=edge_mask, other=0.0)
            spikes = tl.load(x_ptr + bucket * n_sources + sources)
            contributions = spikes * weights
            valid = edge_mask & (spikes != 0.0)

            unsigned_destinations = destinations.to(tl.uint32)
            slot = (
                (unsigned_destinations * 2654435761) & (HASH_CAPACITY - 1)
            ).to(tl.int32)
            destination_slot = tl.full((BLOCK_EDGES,), -1, tl.int32)
            found = ~valid
            for _ in tl.static_range(MAX_PROBE):
                # Lanes that already found a key keep probing that same slot.
                # Invalid lanes use slot zero with an impossible comparison and
                # therefore cannot claim a real table entry.
                probe_slot = tl.where(
                    found, tl.maximum(destination_slot, 0), slot
                )
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
                destination_slot = tl.where(
                    newly_found, probe_slot, destination_slot
                )
                found |= success
                slot = (slot + 1) & (HASH_CAPACITY - 1)

            tl.atomic_add(
                hash_values_ptr + hash_base + destination_slot,
                contributions,
                mask=valid & (destination_slot >= 0),
                sem="relaxed",
                scope="cta",
            )
            # A bounded probe must never lose an edge. Failed insertions bypass
            # the local table and join the global result directly.
            tl.atomic_add(
                output_ptr + bucket * n_destinations + destinations,
                contributions,
                mask=valid & (destination_slot < 0),
            )
            tl.debug_barrier()

            keys = tl.load(hash_keys_ptr + hash_base + hash_offsets)
            values = tl.load(hash_values_ptr + hash_base + hash_offsets)
            occupied = keys >= 0
            # This flush is the global-merge phase. Keeping it in the consumer
            # kernel lets persistent workers reuse a small hash table.
            tl.atomic_add(
                output_ptr + bucket * n_destinations + keys,
                values,
                mask=occupied,
            )
            tl.debug_barrier()
            task_slot += NUM_PROGRAMS


def is_triton_available() -> bool:
    """Return whether the Triton Python package can be imported."""

    return _TRITON_AVAILABLE


def _require_triton() -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            "The 'triton' sparse backend requires Triton. Install a CUDA-enabled "
            "PyTorch/Triton environment or select another sparse backend."
        )


@dataclass
class TritonSparseWorkspace:
    """Reusable device storage for task queues and local hash tables."""

    queue_capacity: int
    worker_count: int
    direct_queue: torch.Tensor
    hash_queue: torch.Tensor
    counts: torch.Tensor
    hash_keys: torch.Tensor
    hash_values: torch.Tensor


def ensure_triton_workspace(
    workspace: TritonSparseWorkspace | None,
    *,
    queue_capacity: int,
    device: torch.device,
    dtype: torch.dtype,
    config: TritonSparseConfig,
) -> TritonSparseWorkspace:
    """Return a compatible workspace, growing it when necessary."""

    worker_count = min(128, max(1, queue_capacity))
    hash_capacity = config.hash_capacity if config.hash else 1
    compatible = (
        workspace is not None
        and workspace.queue_capacity >= queue_capacity
        and workspace.direct_queue.device == device
        and workspace.hash_values.dtype == dtype
        and workspace.worker_count >= worker_count
        and workspace.hash_keys.shape[1] == hash_capacity
    )
    if compatible:
        return workspace
    direct_queue = torch.empty(
        (queue_capacity, 2), device=device, dtype=torch.int32
    )
    return TritonSparseWorkspace(
        queue_capacity=queue_capacity,
        worker_count=worker_count,
        direct_queue=direct_queue,
        hash_queue=torch.empty_like(direct_queue),
        counts=torch.empty(2, device=device, dtype=torch.int32),
        hash_keys=torch.empty(
            (worker_count, hash_capacity),
            device=device,
            dtype=torch.int32,
        ),
        hash_values=torch.empty(
            (worker_count, hash_capacity), device=device, dtype=dtype
        ),
    )


def _launch_sparse_mm(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_source: torch.Tensor,
    packed_destination: torch.Tensor,
    task_indptr: torch.Tensor,
    task_source_ids: torch.Tensor,
    task_hashable: torch.Tensor,
    config: TritonSparseConfig,
    n_destinations: int,
    workspace: TritonSparseWorkspace,
) -> torch.Tensor:
    _require_triton()
    if not x.is_cuda:
        raise RuntimeError("The 'triton' sparse backend requires CUDA tensors.")
    if x.dtype != torch.float32 or packed_weight.dtype != torch.float32:
        raise TypeError("The 'triton' sparse backend currently supports float32 only.")
    if not x.is_contiguous() or not packed_weight.is_contiguous():
        raise ValueError("Triton sparse inputs and weights must be contiguous.")

    n_buckets, n_sources = x.shape
    n_tasks = task_indptr.numel() - 1
    output = torch.zeros(
        (n_buckets, n_destinations), device=x.device, dtype=x.dtype
    )
    if n_tasks == 0 or n_buckets == 0:
        return output

    queue_capacity = n_buckets * n_tasks
    direct_queue = workspace.direct_queue
    hash_queue = workspace.hash_queue
    counts = workspace.counts
    counts.zero_()
    _pack_tasks_kernel[(queue_capacity,)](
        x,
        task_source_ids,
        task_hashable,
        direct_queue,
        hash_queue,
        counts,
        n_tasks=n_tasks,
        n_sources=n_sources,
        BLOCK_SOURCES=config.block_sources,
        ENABLE_HASH=config.hash,
        num_warps=1,
    )

    worker_count = workspace.worker_count
    _direct_tasks_kernel[(worker_count,)](
        x,
        packed_weight,
        packed_source,
        packed_destination,
        task_indptr,
        direct_queue,
        counts,
        output,
        n_sources=n_sources,
        n_destinations=n_destinations,
        BLOCK_EDGES=config.edge_block,
        NUM_PROGRAMS=worker_count,
        num_warps=config.num_warps,
    )
    if config.hash:
        _hash_tasks_kernel[(worker_count,)](
            x,
            packed_weight,
            packed_source,
            packed_destination,
            task_indptr,
            hash_queue,
            counts[1:],
            workspace.hash_keys,
            workspace.hash_values,
            output,
            n_sources=n_sources,
            n_destinations=n_destinations,
            BLOCK_EDGES=config.edge_block,
            HASH_CAPACITY=config.hash_capacity,
            MAX_PROBE=config.hash_max_probe,
            NUM_PROGRAMS=worker_count,
            num_warps=config.num_warps,
        )
    return output


class _TritonSparseMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        packed_weight: torch.Tensor,
        packed_source: torch.Tensor,
        packed_destination: torch.Tensor,
        task_indptr: torch.Tensor,
        task_source_ids: torch.Tensor,
        task_hashable: torch.Tensor,
        config: TritonSparseConfig,
        n_destinations: int,
        workspace: TritonSparseWorkspace,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            x, packed_weight, packed_source, packed_destination
        )
        return _launch_sparse_mm(
            x,
            packed_weight,
            packed_source,
            packed_destination,
            task_indptr,
            task_source_ids,
            task_hashable,
            config,
            n_destinations,
            workspace,
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        x, packed_weight, source, destination = ctx.saved_tensors
        source_long = source.to(torch.long)
        destination_long = destination.to(torch.long)
        grad_x = torch.zeros_like(x)
        grad_x.index_add_(
            1,
            source_long,
            grad_output.index_select(1, destination_long) * packed_weight,
        )
        grad_weight = (
            x.index_select(1, source_long)
            * grad_output.index_select(1, destination_long)
        ).sum(dim=0)
        return (
            grad_x,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def triton_sparse_mm(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    *,
    packed_source: torch.Tensor,
    packed_destination: torch.Tensor,
    task_indptr: torch.Tensor,
    task_source_ids: torch.Tensor,
    task_hashable: torch.Tensor,
    config: TritonSparseConfig,
    n_destinations: int,
    workspace: TritonSparseWorkspace | None = None,
) -> torch.Tensor:
    """Apply an event-driven source-major sparse matrix multiplication."""

    if workspace is None:
        queue_capacity = x.shape[0] * max(0, task_indptr.numel() - 1)
        workspace = ensure_triton_workspace(
            None,
            queue_capacity=max(1, queue_capacity),
            device=x.device,
            dtype=x.dtype,
            config=config,
        )
    return _TritonSparseMM.apply(
        x,
        packed_weight,
        packed_source,
        packed_destination,
        task_indptr,
        task_source_ids,
        task_hashable,
        config,
        n_destinations,
        workspace,
    )


__all__ = [
    "TritonSparseWorkspace",
    "ensure_triton_workspace",
    "is_triton_available",
    "triton_sparse_mm",
]
