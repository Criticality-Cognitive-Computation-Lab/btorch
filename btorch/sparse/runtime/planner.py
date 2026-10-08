"""Execution planning for sparse propagation.

Planning happens when a connection is built or its hints change, never inside
``forward``. The result is an internal object; model code cannot depend on
it. Use :func:`btorch.sparse.explain` to inspect a decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..properties import Hints
from .backend import registry


@dataclass(frozen=True)
class _Plan:
    """Internal execution plan of one connection."""

    representation: str  # canonical layout the kernels read
    algorithm: str  # "pull" or "adaptive-push"
    backend: str
    max_density: float
    reason: str


class Planner:
    """Choose representation, algorithm and backend for a propagation.

    The three choices are independent. The planner only consults cheap,
    static information: device, shape, batching and the user's
    :class:`~btorch.sparse.Hints`.

    Args:
        push_max_density: Per device type, the largest expected input density
            for which source-driven propagation is planned. Above it (or
            without a density hint) the destination-driven product is used,
            which is deterministic and free of data-dependent control flow.
    """

    def __init__(self, push_max_density: dict[str, float] | None = None):
        self.push_max_density = push_max_density or {"cpu": 0.05, "cuda": 0.05}

    def plan(
        self,
        *,
        device: str,
        value_batched: bool,
        hints: Hints,
    ) -> _Plan:
        limit = self.push_max_density.get(device, 0.0)
        density = hints.expected_density
        if value_batched:
            algorithm, reason = "pull", "shared-pattern batch of values"
        elif density is None:
            algorithm, reason = "pull", "no expected_density hint"
        elif density <= limit:
            algorithm = "adaptive-push"
            reason = f"expected_density {density:g} <= {limit:g}"
        else:
            algorithm, reason = "pull", f"expected_density {density:g} > {limit:g}"
        kernel = "spike_push" if algorithm == "adaptive-push" else "csr_matvec"
        return _Plan(
            representation="destination CSR + source CSR",
            algorithm=algorithm,
            backend=registry.name(kernel, device),
            max_density=limit if algorithm == "adaptive-push" else 0.0,
            reason=reason,
        )


planner = Planner()
