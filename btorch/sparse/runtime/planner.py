"""Execution planning for sparse propagation.

Planning happens when a connection is built or its hints change, never inside
``forward``. The result is an internal object; model code cannot depend on
it. Use :func:`btorch.sparse.explain` to inspect a decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..properties import Hints
from .backend import RouteResolutionSnapshot, registry


@dataclass(frozen=True)
class PlanningContext:
    """Static facts used to filter and rank complete route candidates."""

    device: str
    value_batched: bool
    expected_density: float | None
    deterministic: bool
    dtype: torch.dtype
    requires_backward: bool = True
    compile: bool = False
    capture: bool = False


@dataclass(frozen=True)
class _Plan:
    """Internal execution plan of one connection."""

    representation: str  # canonical layout the kernels read
    algorithm: str  # "pull", "push" or "adaptive-push"
    backend: str
    route: RouteResolutionSnapshot
    deterministic: bool
    max_density: float
    reason: str


class Planner:
    """Choose representation, algorithm and backend for a propagation.

    The planner only consults cheap, static information: device, batching and
    the user's :class:`~btorch.sparse.Hints`. Today it decides one thing, the
    algorithm:

    - ``"pull"``: destination-driven product (deterministic).
    - ``"push"``: source-driven with compaction on the device (Triton on
      CUDA). It is taken whenever the density hint is at or below the limit,
      whatever the density of an individual input turns out to be; it
      accumulates with floating-point atomics, so it is not bitwise
      reproducible and is not planned when deterministic algorithms are
      requested.
    - ``"adaptive-push"``: source-driven with host-side packing (reference
      backend); each call falls back to pull above the density limit.

    Args:
        push_max_density: Per device type, the largest expected input density
            for which source-driven propagation is planned. Above it (or
            without a density hint) the destination-driven product is used,
            which is deterministic and free of data-dependent control flow.
    """

    def __init__(self, push_max_density: dict[str, float] | None = None):
        # Measured crossovers (RTX 5090, heavy-tailed graphs of 4k-1M neurons,
        # batch 1-32): the device-compacting push beats the Triton pull below
        # 1.5-5% density once B * N is large, and never at ~4k neurons with
        # one sample. On CPU the ATen push only wins below ~0.2-0.5% density.
        self.push_max_density = push_max_density or {"cpu": 0.002, "cuda": 0.02}

    def plan(
        self,
        *,
        device: str,
        value_batched: bool,
        hints: Hints,
        dtype: torch.dtype,
    ) -> _Plan:
        limit = self.push_max_density.get(device, 0.0)
        density = hints.expected_density
        deterministic = torch.are_deterministic_algorithms_enabled()
        context = PlanningContext(
            device=device,
            value_batched=value_batched,
            expected_density=density,
            deterministic=deterministic,
            dtype=dtype,
        )
        if value_batched:
            algorithm, reason = "pull", "shared-pattern batch of values"
        elif density is None:
            algorithm, reason = "pull", "no expected_density hint"
        elif density > limit:
            algorithm, reason = "pull", f"expected_density {density:g} > {limit:g}"
        else:
            algorithm = "push"
            if not registry.has_route(algorithm, device, context):
                algorithm = (
                    "pull"
                    if device == "cuda" and registry.override is None
                    else "adaptive-push"
                )
                if not registry.has_route(algorithm, device, context):
                    algorithm = "pull"
            if device == "cuda" and algorithm != "pull" and deterministic:
                # CUDA source-driven routes accumulate with floating-point
                # atomics, including ATen's adaptive reference route.
                algorithm = "pull"
                reason = "deterministic algorithms requested"
            else:
                reason = f"expected_density {density:g} <= {limit:g}"
        if algorithm == "pull":
            context = PlanningContext(
                device=device,
                value_batched=value_batched,
                expected_density=density,
                deterministic=deterministic,
                dtype=dtype,
            )
        route = registry.resolve_route(algorithm, device, context)
        return _Plan(
            representation="destination CSR + source CSR",
            algorithm=algorithm,
            backend=route.backend,
            route=route,
            deterministic=deterministic,
            max_density=limit if algorithm != "pull" else 0.0,
            reason=reason,
        )


planner = Planner()
