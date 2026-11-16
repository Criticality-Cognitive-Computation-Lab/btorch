"""Derived execution representations of a sparsity pattern."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RepresentationCache(nn.Module):
    """Execution structures derived from canonical edge lists.

    The canonical model state of a connection is its edge list in *slot*
    order. Kernels need compressed layouts, which this cache derives:

    - ``crow``, ``col``: destination-major CSR (pull / batched products).
    - ``perm``: slot of every CSR entry (``values_csr = values[perm]``).
    - ``t_crow``, ``t_col``, ``t_perm``: source-major CSR of the same edges
      (push, and the transposed product of the input gradient); ``t_perm`` is
      the CSR position of every source-major entry.

    All of them are non-persistent buffers: they follow ``.to()`` with the
    owning module, never enter a ``state_dict``, and are rebuilt from the
    canonical buffers whenever the topology changes. Rebuilding writes into
    the existing tensors when the number of edges is unchanged, so compiled
    graphs that captured them stay valid. A backend may keep derived copies of
    its own (the Triton pull kernels rebuild theirs when the pattern
    changes), so a CUDA graph captured before rewiring must be captured
    again. Rebuild at the optimizer boundary, never between a forward pass
    and its backward.
    """

    crow: Tensor
    col: Tensor
    perm: Tensor
    t_crow: Tensor
    t_col: Tensor
    t_perm: Tensor

    _NAMES = ("crow", "col", "perm", "t_crow", "t_col", "t_perm")
    _SEGMENT_NAMES = (
        "segment_src_ptr",
        "segment_pre",
        "segment_delay",
        "segment_ptr",
        "segment_edge_id",
    )

    def __init__(self) -> None:
        super().__init__()
        for name in (*self._NAMES, *self._SEGMENT_NAMES):
            self.register_buffer(name, torch.empty(0, dtype=torch.long), False)
        self.register_buffer("edge_segment", torch.empty(0, dtype=torch.long), False)
        self._segment_signature = None
        self._edge_segment_signature = None
        self.topology_version = -1
        self.identity_perm = True
        self.shape = (0, 0)
        self.max_source_degree = 0

    @torch.no_grad()
    def build(self, row: Tensor, col: Tensor, shape: tuple[int, int], version: int):
        """(Re)build every representation.

        Args:
            row: ``[E]`` destination (output index) of each slot.
            col: ``[E]`` source (input index) of each slot.
            shape: ``(n_out, n_in)``.
            version: Topology version the structures correspond to.
        """
        n_out, n_in = shape
        # int64 so the linearised key cannot overflow for int32 inputs.
        row, col = row.long(), col.long()
        key = row * n_in + col
        if key.shape[0] > 1 and bool((key[1:] < key[:-1]).any()):
            perm = torch.argsort(key, stable=True)
            identity = False
        else:
            perm = torch.arange(key.shape[0], device=key.device)
            identity = True
        row_s, col_s = row[perm], col[perm]
        crow = self._pointer(row_s, n_out)
        t_perm = torch.argsort(col_s * n_out + row_s, stable=True)
        new = {
            "crow": crow,
            "col": col_s,
            "perm": perm,
            "t_crow": self._pointer(col_s[t_perm], n_in),
            "t_col": row_s[t_perm],
            "t_perm": t_perm,
        }
        self.max_source_degree = (
            int((new["t_crow"][1:] - new["t_crow"][:-1]).max().item()) if n_in else 0
        )
        for name, value in new.items():
            current = getattr(self, name)
            if current.shape == value.shape and current.device == value.device:
                current.copy_(value)
            else:
                setattr(self, name, value)
        self.identity_perm = identity
        self.shape = (n_out, n_in)
        self.topology_version = version
        self._segment_signature = None
        self._edge_segment_signature = None

    @torch.no_grad()
    def prepare_delay_segments(self, n_source: int, n_delay: int) -> tuple[Tensor, ...]:
        """Prepare nonempty source/delay segments with canonical edge identity.

        Reuse the outgoing CSR edge order and destination storage. Incoming
        traversal continues to use ``crow``, ``col`` and ``perm``; it does not
        need a second copy of canonical weights or receptor metadata.
        Preparation may synchronize to discover segment count and runs only
        when the topology or delay layout changes, never within a timestep.

        Returns:
            Source-to-segment pointers, segment sources, segment delays,
            segment-to-edge pointers, canonical outgoing edge IDs and the
            existing outgoing destinations.
        """
        if n_source < 0 or n_delay < 1 or n_source * n_delay != self.shape[1]:
            raise ValueError("Source/delay layout disagrees with the cached topology")
        signature = (self.topology_version, n_source, n_delay, self.t_crow.device)
        if self._segment_signature != signature:
            nonempty = torch.nonzero(
                self.t_crow[1:] != self.t_crow[:-1], as_tuple=True
            )[0]
            source = nonempty // n_delay
            values = (
                self._pointer(source, n_source),
                source,
                nonempty % n_delay,
                torch.cat((self.t_crow.index_select(0, nonempty), self.t_crow[-1:])),
                self.perm.index_select(0, self.t_perm),
            )
            for name, value in zip(self._SEGMENT_NAMES, values, strict=True):
                current = getattr(self, name)
                if current.shape == value.shape and current.device == value.device:
                    current.copy_(value)
                else:
                    setattr(self, name, value)
            self._segment_signature = signature
        return (*[getattr(self, name) for name in self._SEGMENT_NAMES], self.t_col)

    @torch.no_grad()
    def prepare_edge_segments(self, n_source: int, n_delay: int) -> Tensor:
        """Map canonical edges to the existing source/delay segments once.

        Only rules requesting segment-domain memories or signals need
        this inverse map. Ordinary edge-local STDP allocates no per-edge
        map.
        """
        _, source, _, pointer, edge_id, _ = self.prepare_delay_segments(
            n_source, n_delay
        )
        if self._edge_segment_signature != self._segment_signature:
            segment = torch.repeat_interleave(
                torch.arange(source.numel(), device=source.device),
                pointer[1:] - pointer[:-1],
            )
            result = torch.empty_like(edge_id)
            result.scatter_(0, edge_id, segment)
            if (
                self.edge_segment.shape == result.shape
                and self.edge_segment.device == result.device
            ):
                self.edge_segment.copy_(result)
            else:
                self.edge_segment = result
            self._edge_segment_signature = self._segment_signature
        return self.edge_segment

    def crow_rows(self) -> Tensor:
        """``[E]`` output index of every CSR entry (expanded ``crow``)."""
        counts = self.crow[1:] - self.crow[:-1]
        return torch.repeat_interleave(
            torch.arange(counts.shape[0], device=counts.device), counts
        )

    @staticmethod
    def _pointer(sorted_index: Tensor, n: int) -> Tensor:
        pointer = torch.zeros(n + 1, dtype=torch.long, device=sorted_index.device)
        pointer[1:] = torch.cumsum(torch.bincount(sorted_index, minlength=n), 0)
        return pointer

    def extra_repr(self) -> str:
        return (
            f"shape={self.shape}, nnz={self.col.shape[0]}, "
            f"topology_version={self.topology_version}"
        )
