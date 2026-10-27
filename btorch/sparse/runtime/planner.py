"""Execution planning for sparse propagation.

Plans are bound when static execution choices change, before launching tensor
operations. The result is internal; model code cannot depend on it. Use
:func:`btorch.sparse.explain` to inspect a decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..properties import Hints
from .backend import RouteResolutionSnapshot, registry
from .method import PropagationMethod, validate_method


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
    firing_rate_hz: float | None = None
    method: PropagationMethod = "auto"
    shape: tuple[int, int] | None = None


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
    method: PropagationMethod = "auto"


class Planner:
    """Choose representation, algorithm and backend for a propagation.

    The planner only consults cheap, static information: device, batching and
    the user's :class:`~btorch.sparse.Hints` and explicit method. An explicit
    method constrains candidate routes and overrides hints; unsupported requests
    raise. Automatic planning chooses the algorithm:

    - ``"pull"``: destination-driven product (deterministic).
    - ``"push"``: source-driven with compaction on the device (Triton on
      CUDA). It is taken when a CUDA firing-rate hint is strictly below 5 Hz
      or the density hint is at or below the limit,
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
        method: PropagationMethod = "auto",
        shape: tuple[int, int] | None = None,
    ) -> _Plan:
        validate_method(method)
        limit = self.push_max_density.get(device, 0.0)
        density = hints.expected_density
        low_rate = (
            device == "cuda"
            and hints.firing_rate_hz is not None
            and hints.firing_rate_hz < 5.0
        )
        deterministic = torch.are_deterministic_algorithms_enabled()
        context = PlanningContext(
            device=device,
            value_batched=value_batched,
            expected_density=density,
            deterministic=deterministic,
            dtype=dtype,
            firing_rate_hz=hints.firing_rate_hz,
            method=method,
            shape=shape,
        )
        if method != "auto":
            algorithm, reason = method, f"explicit method={method!r}"
            if method == "push" and device == "cuda" and deterministic:
                raise RuntimeError(
                    "method='push' requires nondeterministic atomic accumulation"
                )
            if not registry.has_route(algorithm, device, context):
                raise RuntimeError(
                    f"Unsupported sparse method={method!r} on {device} with "
                    f"dtype={dtype}, batched_values={value_batched}, "
                    f"backend={registry.override or 'auto'!r}"
                )
        elif (
            shape is not None
            and max(shape) < 4000
            and registry.override is None
            and dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        ):
            algorithm, reason = "pull", "fewer than 4000 neurons: dense matmul"
        elif value_batched:
            algorithm, reason = "pull", "shared-pattern batch of values"
        elif density is None and not low_rate:
            algorithm, reason = "pull", "no expected_density hint"
        elif density is not None and density > limit and not low_rate:
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
            if algorithm == "pull":
                reason = "no supported source-driven route"
            elif device == "cuda" and deterministic:
                # CUDA source-driven routes accumulate with floating-point
                # atomics, including ATen's adaptive reference route.
                algorithm = "pull"
                reason = "deterministic algorithms requested"
            else:
                reason = (
                    f"firing_rate_hz {hints.firing_rate_hz:g} < 5"
                    if low_rate
                    else f"expected_density {density:g} <= {limit:g}"
                )
        route = registry.resolve_route(algorithm, device, context)
        return _Plan(
            representation=(
                "dense matrix (sparse edge parameters)"
                if route.key.representation == "dense"
                else "device spike list + source CSR"
                if route.key.representation == "spike_list_csr"
                else "destination CSR + source CSR"
            ),
            algorithm=algorithm,
            backend=route.backend,
            route=route,
            deterministic=deterministic,
            max_density=limit if algorithm != "pull" else 0.0,
            reason=reason,
            method=method,
        )


planner = Planner()
