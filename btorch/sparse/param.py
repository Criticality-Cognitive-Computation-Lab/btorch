from __future__ import annotations

import torch
import torch.nn as nn

from .base import SparseTensorBase


class SparseParam(SparseTensorBase, nn.Module):
    """nn.Module bridge for a SparseTensorBase.

    Trainable tensors are registered as nn.Parameter; structural tensors
    and parameterization buffers are registered as non-persistent
    buffers.

    After device movement (`.to()`, `.cuda()` etc.) the inner
    SparseTensorBase is rebuilt from the updated parameters/buffers via
    `_rebuild()`.
    """

    _btorch_constraint: bool = True
    _inner: SparseTensorBase

    def __init__(self, matrix: SparseTensorBase) -> None:
        nn.Module.__init__(self)
        self._inner = matrix

        if matrix.parameterization is not None:
            # Parameterization owns all trainable tensors; matrix data are
            # the fixed initial weight. Expose them via the parameterization's
            # fixed tensors (which include "initial_weight").
            for name, tensor in matrix.parameterization._trainable_tensors():
                self.register_parameter(name, nn.Parameter(tensor))
            for name, tensor in matrix.parameterization._fixed_tensors():
                self.register_buffer(name, tensor, persistent=False)
        else:
            for name, tensor in matrix._trainable_tensors():
                self.register_parameter(name, nn.Parameter(tensor))

        for name, tensor in matrix._structural_tensors():
            self.register_buffer(name, tensor, persistent=False)

    # ── SparseTensorBase forwarding ───────────────────────────────────────────

    @property
    def shape(self) -> tuple[int, int]:
        return self._inner.shape

    @property
    def properties(self):
        return self._inner.properties

    @property
    def attributes(self):
        return self._inner.attributes

    @property
    def constraint(self):
        return self._inner.constraint

    @property
    def parameterization(self):
        return self._inner.parameterization

    @property
    def _cache(self) -> dict:
        return self._inner._cache

    def mm(self, x: torch.Tensor) -> torch.Tensor:
        return self._current().mm(x)

    def event_mm(self, events) -> torch.Tensor:
        return self._current().event_mm(events)

    def to_coo(self):
        return self._current().to_coo()

    def to_csr(self):
        return self._current().to_csr()

    def to_csc(self):
        return self._current().to_csc()

    def to_ell(self):
        return self._current().to_ell()

    def logical_edges(self):
        return self._current().logical_edges()

    def nnz(self) -> int:
        return self._inner.nnz()

    def effective_values(self) -> torch.Tensor:
        return self._current().effective_values()

    def padded_csr_layout(self):
        return self._current().padded_csr_layout()

    def with_constraint(self, c: object) -> SparseTensorBase:
        return self._current().with_constraint(c)

    def with_parameterization(self, p: object) -> SparseTensorBase:
        return self._current().with_parameterization(p)

    def _trainable_tensors(self):
        return self._inner._trainable_tensors()

    def _structural_tensors(self):
        return self._inner._structural_tensors()

    def _rebuild(self, parameters: dict, buffers: dict) -> SparseTensorBase:
        return self._inner._rebuild(parameters, buffers)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _current(self) -> SparseTensorBase:
        """Return a fresh SparseTensorBase reflecting current parameter
        values."""
        params = dict(self.named_parameters())
        bufs = dict(self.named_buffers())
        return self._inner._rebuild(params, bufs)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        # Rebuild _inner after all tensors have been moved.
        params = dict(self.named_parameters())
        bufs = dict(self.named_buffers())
        self._inner = self._inner._rebuild(params, bufs)
        return self

    def constrain(self) -> None:
        """Apply constraint / Dale-law clamping in-place on registered
        parameters."""
        from .parameterization import GroupedMagnitude

        p = self._inner.parameterization
        if isinstance(p, GroupedMagnitude) and p.dale:
            with torch.no_grad():
                self._parameters["magnitude"].clamp_min_(0)
        elif (
            self._inner.constraint is not None and self._inner.parameterization is None
        ):
            from .constraints import constrain as _constrain

            projected = _constrain(self._current())
            with torch.no_grad():
                self._parameters["data"].copy_(projected.effective_values())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mm(x)


class SparseLinear(nn.Module):
    """Thin nn.Module that applies a SparseParam and an optional bias.

    Usage::
        W = CSR.from_edges(src, dst, vals, shape=(n_pre, n_post))
        layer = SparseLinear(W, bias=True)
    out = layer(x)   # x: (..., n_pre) → (..., n_post)
    """

    sparse_weight: SparseParam
    bias: nn.Parameter | None

    def __init__(
        self,
        matrix: SparseTensorBase,
        bias: bool | torch.Tensor | None = False,
    ) -> None:
        super().__init__()
        self.sparse_weight = SparseParam(matrix)
        n_post = matrix.shape[1]
        if isinstance(bias, torch.Tensor):
            if bias.shape != (n_post,):
                raise ValueError("bias tensor must have shape (n_post,).")
            self.bias: nn.Parameter | None = nn.Parameter(bias.clone())
        elif bias:
            self.bias = nn.Parameter(torch.zeros(n_post))
        else:
            self.bias = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.sparse_weight.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.sparse_weight.mm(x)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        n_pre, n_post = self.shape
        return (
            f"n_pre={n_pre}, n_post={n_post}, "
            f"nnz={self.sparse_weight.nnz()}, "
            f"bias={self.bias is not None}"
        )


SparseConn = SparseLinear
