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
    graphs and CUDA graphs that captured them stay valid.
    """

    crow: Tensor
    col: Tensor
    perm: Tensor
    t_crow: Tensor
    t_col: Tensor
    t_perm: Tensor

    _NAMES = ("crow", "col", "perm", "t_crow", "t_col", "t_perm")

    def __init__(self) -> None:
        super().__init__()
        for name in self._NAMES:
            self.register_buffer(name, torch.empty(0, dtype=torch.long), False)
        self.topology_version = -1
        self.identity_perm = True
        self.shape = (0, 0)

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
        for name, value in new.items():
            current = getattr(self, name)
            if current.shape == value.shape and current.device == value.device:
                current.copy_(value)
            else:
                setattr(self, name, value)
        self.identity_perm = identity
        self.shape = (n_out, n_in)
        self.topology_version = version

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
