"""Frozen pre-refactor baseline for the sparse connection benchmark.

``LegacySparseConn`` is a minimal, self-contained copy of the forward pass of
``btorch.models.linear.SparseConn`` as it was *before* the sparse refactor
(the class has since been removed in favour of
:class:`btorch.models.connection.SparseConnection`). It exists only so that
``bench_sparse_conn.py`` can keep reporting ``legacy[native]`` and
``legacy[torch_sparse]`` numbers next to the new implementation.

It is a benchmark fixture, not library code: it is not maintained, has no
Dale's law / constraint support, and must not be imported by models.
"""

import scipy.sparse
import torch
from torch import nn


try:  # optional dependency of the legacy ``torch_sparse`` path
    from torch_sparse import spmm
except ImportError:
    spmm = None


def available_legacy_backends() -> list[str]:
    """Legacy execution paths that can run in this environment."""
    return ["native"] + (["torch_sparse"] if spmm is not None else [])


class LegacySparseConn(nn.Module):
    """Pre-refactor ``SparseConn`` forward (``enforce_dale=False``, no bias).

    Args:
        conn: SciPy sparse matrix ``(n_src, n_dst)``; the layer computes
            ``x @ conn``.
        backend: ``"native"`` builds a ``torch.sparse_coo_tensor`` from the
            stored indices on every call and uses ``torch.sparse.mm``;
            ``"torch_sparse"`` calls ``torch_sparse.spmm`` on the same
            indices.
        device: Target device.

    Attributes:
        indices: ``[2, nnz]`` = ``[dst, src]``, sorted (the transposed
            matrix, so that ``x @ A`` is computed as ``(A^T @ x^T)^T``).
        magnitude: ``[nnz]`` trainable weights.
    """

    def __init__(
        self,
        conn: scipy.sparse.sparray,
        backend: str = "native",
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if backend not in ("native", "torch_sparse"):
            raise ValueError(f"backend must be 'native' or 'torch_sparse': {backend}")
        if backend == "torch_sparse" and spmm is None:
            raise ImportError("torch_sparse is not installed.")
        self.backend = backend
        self.in_features, self.out_features = conn.shape
        # Transpose, then sum duplicates (which also sorts by [dst, src]).
        conn = conn.tocoo().T
        conn.sum_duplicates()
        indices = torch.stack(
            [
                torch.tensor(conn.row, dtype=torch.long, device=device),
                torch.tensor(conn.col, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        self.register_buffer("indices", indices)
        self.magnitude = nn.Parameter(
            torch.tensor(conn.data, dtype=torch.float32, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[..., n_src] -> [..., n_dst]``."""
        no_batch = x.ndim == 1
        if no_batch:
            x = x[None, :]
        value = self.magnitude
        if value.device != x.device or value.dtype != x.dtype:
            value = value.to(device=x.device, dtype=x.dtype)
        leading_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        if self.backend == "native":
            # COO tensor rebuilt from the ``indices`` buffer on every call.
            sp = torch.sparse_coo_tensor(
                indices=self.indices,
                values=value,
                size=(self.out_features, self.in_features),
                is_coalesced=True,
            )
            out = torch.sparse.mm(sp, x_2d.T).T
        else:
            out = spmm(
                self.indices, value, self.out_features, self.in_features, x_2d.T
            ).T
        out = out.reshape(*leading_shape, self.out_features)
        if no_batch:
            out = out[0, :]
        return out
