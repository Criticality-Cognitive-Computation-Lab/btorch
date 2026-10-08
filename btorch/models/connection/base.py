"""Connection realisations.

A *connection* is an ``nn.Module`` mapping pre-synaptic activity to
post-synaptic input: ``current = conn(spikes)``. How it is realised is a
separate question from how it was specified (a rule, a matrix, a formula):

- :class:`~btorch.models.connection.SparseConnection`: explicit edges.
- :class:`StructuredConnection`: a closed-form operator (all-to-all,
  one-to-one, low rank) that never stores a matrix.
- :class:`ImplicitConnection`: a procedural operator applied on the fly.
- :class:`HybridConnection`: a sum of connections over the same populations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor, nn


class Connection(nn.Module):
    """Base class of all connection realisations.

    Attributes:
        n_pre: Number of pre-synaptic (source) neurons.
        n_post: Number of post-synaptic (target) neurons.
        in_features: Size of the last input dimension.
        out_features: Size of the last output dimension.
    """

    n_pre: int
    n_post: int

    @property
    def in_features(self) -> int:
        return self.n_pre

    @property
    def out_features(self) -> int:
        return self.n_post

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class OperatorConnection(Connection):
    """Connection that applies a linear operator along the last axis.

    Args:
        operator: Object with ``shape == (n_post, n_pre)`` and a ``matvec``
            method (a :class:`~btorch.sparse.operator.LinearOperator` or a
            :class:`~btorch.sparse.Sparse`). If it is an ``nn.Module`` its
            parameters are registered.
    """

    def __init__(self, operator: Any):
        super().__init__()
        self.n_post, self.n_pre = (int(s) for s in operator.shape[-2:])
        self.operator = operator

    def forward(self, x: Tensor) -> Tensor:
        return self.operator.matvec(x)

    def extra_repr(self) -> str:
        return f"n_pre={self.n_pre}, n_post={self.n_post}"


class StructuredConnection(OperatorConnection):
    """Connection realised by a structured (closed-form) operator."""


class ImplicitConnection(OperatorConnection):
    """Connection realised by a procedural operator that is never stored."""


class HybridConnection(Connection):
    """Sum of several connections between the same two populations.

    Args:
        parts: Connections with identical ``in_features`` / ``out_features``.
    """

    def __init__(self, parts: Sequence[Connection]):
        super().__init__()
        parts = list(parts)
        if not parts:
            raise ValueError("HybridConnection needs at least one part.")
        first = parts[0]
        for p in parts:
            if (p.in_features, p.out_features) != (
                first.in_features,
                first.out_features,
            ):
                raise ValueError("All parts must connect the same populations.")
        self.n_pre, self.n_post = first.n_pre, first.n_post
        self._in, self._out = first.in_features, first.out_features
        self.parts = nn.ModuleList(parts)

    @property
    def in_features(self) -> int:
        return self._in

    @property
    def out_features(self) -> int:
        return self._out

    def forward(self, x: Tensor) -> Tensor:
        out = self.parts[0](x)
        for part in self.parts[1:]:
            out = out + part(x)
        return out
