"""Projection: a rule plus a synapse between two populations."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor, nn

from ...sparse import Hints
from .base import Connection, StructuredConnection
from .rule import ConnectionRule
from .sparse import SparseConnection
from .synapse import Synapse


def _population_size(population: Any, role: str) -> int:
    """Number of neurons of a population given as an int or a module."""
    if isinstance(population, bool):
        raise TypeError(f"{role} must be a population size or module.")
    if isinstance(population, int):
        return population
    for attr in ("size", "n_neuron", "num_neurons", "out_features"):
        value = getattr(population, attr, None)
        if callable(value):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, (tuple, list, torch.Size)) and value:
            n = 1
            for s in value:
                n *= int(s)
            return n
    raise TypeError(
        f"Cannot infer the number of neurons of {role}={population!r}; pass an "
        "int or a module with `size` / `n_neuron`."
    )


class Projection(nn.Module):
    """Connections from one population to another, NEST style.

    A projection states *which* neurons are connected (the rule) and *what*
    the connections do (the synapse). It does not state how they are stored
    or executed: that is the business of the connection it builds and of the
    runtime below it.

    Args:
        pre: Source population: a neuron count or a module exposing ``size``
            or ``n_neuron``.
        post: Target population.
        rule: Which pairs are connected.
        synapse: Weight, delay and receptor of the edges. Per-edge arrays are
            aligned with the rule's edge list. If omitted, rules that carry
            values (``FromSparse``, ``FromEdges``) use them as trainable
            weights and all other rules start from unit weights.
        generator: Random generator for stochastic rules.
        realization: ``"auto"`` uses a structured operator when the rule has
            one and the synapse is a single fixed weight without receptor or
            delay, and explicit sparse edges otherwise. ``"sparse"`` always
            materialises the edges.
        hints: Performance expectations forwarded to the sparse connection.
        device: Target device.
        dtype: Weight dtype.

    Attributes:
        connection: The realised
            :class:`~btorch.models.connection.Connection`.

    Examples:
        >>> proj = Projection(
        ...     pre=800, post=200,
        ...     rule=FixedIndegree(100),
        ...     synapse=Synapse(weight=0.1, delay=2),
        ... )                                              # doctest: +SKIP
        >>> current = proj(spikes)
    """

    connection: Connection

    def __init__(
        self,
        pre: Any,
        post: Any,
        rule: ConnectionRule,
        synapse: Synapse | None = None,
        *,
        generator: torch.Generator | None = None,
        realization: Literal["auto", "sparse"] = "auto",
        hints: Hints | None = None,
        device=None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if realization not in ("auto", "sparse"):
            raise ValueError("realization must be 'auto' or 'sparse'.")
        synapse = synapse or Synapse()
        self.n_pre = _population_size(pre, "pre")
        self.n_post = _population_size(post, "post")
        self.rule = rule

        weight = synapse.weight
        fixed_scalar = isinstance(weight, (int, float)) and not isinstance(weight, bool)
        structured = (
            realization == "auto"
            and fixed_scalar
            and synapse.delay is None
            and synapse.receptor is None
        )
        operator = (
            rule.as_operator(self.n_pre, self.n_post, dtype=dtype, device=device)
            if structured
            else None
        )
        if operator is not None:
            self.connection = StructuredConnection(operator * float(weight))
            return

        pre_idx, post_idx = rule.edges(
            self.n_pre, self.n_post, generator=generator, device=device
        )
        values = rule.values(self.n_pre, self.n_post, device=device)
        self.connection = SparseConnection.from_edges(
            pre_idx,
            post_idx,
            self.n_pre,
            self.n_post,
            synapse,
            values=values,
            hints=hints,
            device=device,
            dtype=dtype,
        )

    @property
    def in_features(self) -> int:
        return self.connection.in_features

    @property
    def out_features(self) -> int:
        return self.connection.out_features

    def forward(self, x: Tensor) -> Tensor:
        """Map ``[..., in_features]`` activity to ``[..., out_features]``
        input."""
        return self.connection(x)

    def extra_repr(self) -> str:
        return (
            f"n_pre={self.n_pre}, n_post={self.n_post}, "
            f"rule={type(self.rule).__name__}"
        )
