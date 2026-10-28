"""Task-queued Triton kernels for source-major sparse propagation.

The implementation is intentionally private to the Triton push backend.  A
fixed source-major sparsity pattern is partitioned into small edge tasks.  A
device-side discovery kernel classifies active ``(batch, task)`` pairs into a
direct queue or a hash queue; consumer kernels then execute those queues.  The
hash consumer aggregates destinations in a worker-local table before merging
the table into the output.

Floating CUDA operands retain their storage dtype and use promoted output
precision. FP16/BF16 contributions accumulate in FP32; FP64 stays FP64.
All topology and workspace construction belongs in :func:`prepare`, so a
prepared state can be used from a CUDA-graph-captured region without allocating
or synchronising on the forward path.
"""

from __future__ import annotations

import importlib.util

import torch

from . import csr_execution, kernels_triton


try:  # Optional dependency: importing the sparse runtime must work on CPU.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised without Triton installed
    triton = None
    tl = None


_TRITON_AVAILABLE = (
    importlib.util.find_spec("triton") is not None and triton is not None
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
        x_batch_stride: tl.constexpr,
        x_source_stride: tl.constexpr,
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
        spikes = tl.load(
            x_ptr + bucket * x_batch_stride + safe_sources * x_source_stride
        )
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
        x_batch_stride: tl.constexpr,
        x_source_stride: tl.constexpr,
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
            spikes = tl.load(
                x_ptr + bucket * x_batch_stride + sources * x_source_stride
            )
            tl.atomic_add(
                output_ptr + bucket * n_destinations + destinations,
                kernels_triton._multiply(spikes, weights),
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
        x_batch_stride: tl.constexpr,
        x_source_stride: tl.constexpr,
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
            spikes = tl.load(
                x_ptr + bucket * x_batch_stride + sources * x_source_stride
            )
            contributions = kernels_triton._multiply(spikes, weights)
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


class _Execution:
    def launch(self, name, grid, args, const_names, consts, warps):
        kernel = {
            "pack_tasks": _pack_tasks_kernel,
            "direct_tasks": _direct_tasks_kernel,
            "hash_tasks": _hash_tasks_kernel,
        }[name]
        kernel[grid](
            *args, **dict(zip(const_names, consts, strict=True)), num_warps=warps
        )

    def prepare(self, name, grid, args, const_names, consts, warps, *, changing):
        from ...utils.triton import prepare_launch

        kernel = {
            "pack_tasks": _pack_tasks_kernel,
            "direct_tasks": _direct_tasks_kernel,
            "hash_tasks": _hash_tasks_kernel,
        }[name]
        return prepare_launch(
            kernel,
            args,
            grid,
            tuple(zip(const_names, consts, strict=True)),
            changing=changing,
            num_warps=warps,
        )


def run(*args, **kwargs):
    return csr_execution.run(_Execution(), *args, **kwargs)
