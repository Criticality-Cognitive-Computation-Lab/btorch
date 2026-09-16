"""Static preprocessing for the event-driven sparse backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .._sparse_config import TritonSparseConfig


@dataclass
class SparsePreprocessed:
    """Device-independent tensors derived from a fixed sparse topology."""

    source_indptr: torch.Tensor
    packed_source: torch.Tensor
    packed_destination: torch.Tensor
    edge_permutation: torch.Tensor
    task_indptr: torch.Tensor
    task_source_ids: torch.Tensor
    task_hashable: torch.Tensor

    @property
    def task_count(self) -> int:
        """Return the number of static edge tasks."""

        return max(0, self.task_indptr.numel() - 1)


def _chunks(values: np.ndarray, size: int) -> list[np.ndarray]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def preprocess_sparse(
    indices: torch.Tensor,
    shape: tuple[int, int],
    *,
    config: TritonSparseConfig,
) -> SparsePreprocessed:
    """Build source-major edge tasks for the Triton sparse backend.

    Args:
        indices: COO indices in the internal ``(destination, source)`` order.
        shape: Original connection shape ``(num_source, num_destination)``.
        config: Triton backend configuration snapshot.

    Returns:
        Static layout whose edge permutation maps the module's weight order to
        the packed task order.
    """

    config.validate()
    if indices.ndim != 2 or indices.shape[0] != 2:
        raise ValueError("indices must have shape (2, edge_count).")

    n_source, _ = shape
    indices_np = indices.detach().cpu().numpy()
    destination = indices_np[0].astype(np.int64, copy=False)
    source = indices_np[1].astype(np.int64, copy=False)
    edge_id = np.arange(source.size, dtype=np.int64)

    # Stable source-major order is the unoptimized baseline. Reordering below is
    # deliberately local to a task, so neuron IDs and public state layouts never
    # change.
    source_order = np.lexsort((edge_id, source))
    degree = np.bincount(source, minlength=n_source)
    source_indptr = np.empty(n_source + 1, dtype=np.int64)
    source_indptr[0] = 0
    np.cumsum(degree, out=source_indptr[1:])
    edges_by_source = [
        source_order[source_indptr[src] : source_indptr[src + 1]]
        for src in range(n_source)
    ]

    raw_tasks: list[np.ndarray] = []
    if config.block:
        for block_start in range(0, n_source, config.block_sources):
            block_end = min(n_source, block_start + config.block_sources)
            short_edges = []
            for src in range(block_start, block_end):
                src_edges = edges_by_source[src]
                if len(src_edges) >= config.long_fanout_threshold:
                    raw_tasks.extend(_chunks(src_edges, config.edge_block))
                elif len(src_edges):
                    short_edges.append(src_edges)
            if short_edges:
                raw_tasks.extend(
                    _chunks(np.concatenate(short_edges), config.edge_block)
                )
    else:
        for src_edges in edges_by_source:
            raw_tasks.extend(_chunks(src_edges, config.edge_block))

    packed_edge_ids: list[int] = []
    task_indptr = [0]
    task_sources: list[np.ndarray] = []
    task_hashable: list[bool] = []
    for task_edges in raw_tasks:
        if config.reorder and len(task_edges) > 1:
            order = np.argsort(destination[task_edges], kind="stable")
            task_edges = task_edges[order]
        packed_edge_ids.extend(task_edges.tolist())
        task_indptr.append(len(packed_edge_ids))
        unique_sources = np.unique(source[task_edges])
        if len(unique_sources) > config.block_sources:
            raise RuntimeError("A sparse task exceeds block_sources capacity.")
        padded_sources = np.full(config.block_sources, -1, dtype=np.int64)
        padded_sources[: len(unique_sources)] = unique_sources
        task_sources.append(padded_sources)
        task_hashable.append(len(task_edges) >= config.hash_min_edges)

    packed_ids = np.asarray(packed_edge_ids, dtype=np.int64)
    if task_sources:
        source_table = np.stack(task_sources)
    else:
        source_table = np.empty((0, config.block_sources), dtype=np.int64)

    def tensor(values: np.ndarray | list[int], dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(values, dtype=dtype, device=indices.device)

    return SparsePreprocessed(
        source_indptr=tensor(source_indptr, torch.int32),
        packed_source=tensor(source[packed_ids], torch.int32),
        packed_destination=tensor(destination[packed_ids], torch.int32),
        edge_permutation=tensor(packed_ids, torch.int64),
        task_indptr=tensor(task_indptr, torch.int32),
        task_source_ids=tensor(source_table, torch.int32),
        task_hashable=tensor(np.asarray(task_hashable), torch.bool),
    )


__all__ = ["SparsePreprocessed", "preprocess_sparse"]
