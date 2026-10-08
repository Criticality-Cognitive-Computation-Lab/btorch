"""Tests for backend selection state and invalidation identities."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from btorch.sparse import runtime
from btorch.sparse.runtime.backend import (
    BackendRegistry,
    KernelCache,
    RegisteredOperator,
    RouteImplementation,
    RouteKey,
    RouteSpec,
)
from btorch.sparse.runtime.planner import PlanningContext


def test_registration_generation_and_entry_revision_are_monotonic():
    """Replacing one implementation changes both registry and entry
    identity."""
    registry = BackendRegistry()
    assert registry.registry_generation == 0

    first = lambda: "first"
    replacement = lambda: "replacement"
    registry.register("matvec", "demo", first, device="cpu")
    assert registry.registry_generation == 1
    assert registry.entry_revision("matvec", "demo", "cpu") == 1
    assert registry.resolve("matvec", "cpu") is first

    registry.register("matvec", "demo", replacement, device="cpu")
    assert registry.registry_generation == 2
    assert registry.entry_revision("matvec", "demo", "cpu") == 2
    assert registry.resolve("matvec", "cpu") is replacement

    with pytest.raises(KeyError, match="No backend"):
        registry.entry_revision("matvec", "missing", "cpu")


def test_availability_transition_advances_generation_without_revision_change():
    """Dynamic availability invalidates selection but keeps route identity."""
    available = True
    registry = BackendRegistry()
    registry.register(
        "matvec",
        "preferred",
        lambda: "preferred",
        device="cpu",
        priority=10,
        available=lambda: available,
    )
    registry.register(
        "matvec", "fallback", lambda: "fallback", device="cpu", priority=0
    )
    assert registry.name("matvec", "cpu") == "preferred"
    generation = registry.registry_generation
    revision = registry.entry_revision("matvec", "preferred", "cpu")

    available = False
    assert registry.name("matvec", "cpu") == "fallback"
    assert registry.registry_generation == generation + 1
    assert registry.entry_revision("matvec", "preferred", "cpu") == revision


def test_snapshots_are_immutable_and_hashable():
    """Snapshots contain stable scalar identities, never callable objects."""
    registry = BackendRegistry()
    registry.register("matvec", "demo", lambda: "demo", device=("cpu", "cuda"))

    snapshot = registry.snapshot()
    assert snapshot == registry.snapshot()
    assert snapshot.fingerprint == registry.fingerprint()
    assert hash(snapshot.fingerprint)
    assert snapshot.entries[0].revision == 1
    assert snapshot.entries[1].revision == 1

    resolved = registry.resolve_snapshot("matvec", "cpu")
    assert resolved.name == "demo"
    assert resolved.revision == 1
    assert resolved.registry_generation == snapshot.registry_generation
    assert hash(resolved.fingerprint)


def test_override_is_context_local_and_nested(_local_override):
    """Overrides nest and do not leak into another thread."""
    registry = BackendRegistry()
    registry.register("matvec", "a", lambda: "a", device="cpu", priority=10)
    registry.register("matvec", "b", lambda: "b", device="cpu", priority=0)
    base_revision = registry.override_revision

    registry.set_backend("a")
    try:
        assert registry.override == "a"
        with ThreadPoolExecutor(max_workers=1) as executor:
            other = executor.submit(
                lambda: (registry.override, registry.name("matvec", "cpu"))
            )
            assert other.result() == (None, "a")

        with pytest.raises(RuntimeError, match="boom"):
            with _local_override(registry, "b"):
                assert registry.override == "b"
                assert registry.name("matvec", "cpu") == "b"
                raise RuntimeError("boom")
        assert registry.override == "a"
        assert registry.override_revision > base_revision
    finally:
        registry.set_backend(None)


def test_use_backend_is_context_local_and_does_not_change_generation():
    """The public override context restores state without registry mutation."""
    registry = runtime.registry
    generation = registry.registry_generation
    before = registry.override_snapshot()

    with runtime.use_backend("aten"):
        assert registry.override == "aten"
        inside = registry.override_snapshot()
        assert inside.name == "aten"
        assert inside.revision > before.revision
        with runtime.use_backend(None):
            assert registry.override is None
        assert registry.override == "aten"

    assert registry.override is None
    assert registry.registry_generation == generation
    assert registry.override_revision > before.revision


def test_bound_route_keeps_one_exact_kernel_revision():
    """Replacing registry entries cannot mix an already-bound route."""
    registry = BackendRegistry()
    events = []

    def build(kernels):
        forward_kernel = kernels["forward"]
        backward_kernel = kernels["backward"]

        def forward(*args):
            events.append(forward_kernel())

        def backward(*args):
            events.append(backward_kernel())

        return RouteImplementation(forward, backward)

    registry.register("forward", "demo", lambda: "forward-v1", device="cpu")
    registry.register("backward", "demo", lambda: "backward-v1", device="cpu")
    registry.register_route(
        RouteSpec(
            key=RouteKey("propagate", "post_pre_csr", "pull", "demo"),
            kernel_bindings=(("forward", "demo"), ("backward", "demo")),
            build=build,
        ),
        device="cpu",
    )
    snapshot = registry.resolve_route("pull", "cpu")
    bound = registry.bind_route(snapshot)

    registry.register("forward", "demo", lambda: "forward-v2", device="cpu")
    registry.register("backward", "demo", lambda: "backward-v2", device="cpu")

    bound.implementation.forward()
    bound.implementation.backward()
    assert events == ["forward-v1", "backward-v1"]
    with pytest.raises(RuntimeError, match="stale"):
        registry.bind_route(snapshot)


def test_kernel_cache_clear_rebuilds_derived_artifacts_once():
    """Clearing a kernel cache invalidates derived artifacts as one unit.

    A prepared sparse layout may be requested by several forwards.  It should
    be built once until topology invalidation clears the cache, after which the
    next request must rebuild it rather than returning stale derived tensors.
    """
    cache = KernelCache()
    builds = []

    def build():
        artifact = object()
        builds.append(artifact)
        return artifact

    first = cache.get(("layout", 1), build)
    assert cache.get(("layout", 1), build) is first
    assert builds == [first]

    generation = cache.generation
    cache.clear()
    second = cache.get(("layout", 1), build)
    assert second is not first
    assert builds == [first, second]
    assert cache.generation == generation + 1


def test_route_policy_filters_capabilities_then_ranks_cost_and_override():
    """Route selection applies capability filters before cost and override.

    The low-cost route cannot participate in CUDA-graph capture.  Capture must
    therefore choose the safe route; ordinary execution chooses the cheaper
    route, while an explicit backend override selects the requested complete
    route without mixing callbacks from another backend.
    """
    registry = BackendRegistry()
    registry.register("matvec", "fast", lambda: "fast", device="cpu")
    registry.register("matvec", "safe", lambda: "safe", device="cpu")

    def route(name, *, cost, supports_capture):
        return RouteSpec(
            key=RouteKey("propagate", "post_pre_csr", "pull", name),
            kernel_bindings=(("matvec", name),),
            build=lambda kernels: RouteImplementation(
                forward=kernels["matvec"], backward=lambda: None
            ),
            operator=RegisteredOperator(supports_capture=supports_capture),
            estimate_cost=lambda context: cost,
        )

    registry.register_route(
        route("fast", cost=1.0, supports_capture=False), device="cpu"
    )
    registry.register_route(
        route("safe", cost=5.0, supports_capture=True), device="cpu"
    )
    ordinary = PlanningContext("cpu", False, None, False, torch.float32)
    capture = PlanningContext(
        "cpu", False, None, False, torch.float32, capture=True
    )

    assert registry.resolve_route("pull", "cpu", ordinary).backend == "fast"
    captured = registry.resolve_route("pull", "cpu", capture)
    assert captured.backend == "safe"
    assert registry.bind_route(captured).implementation.forward() == "safe"

    registry.set_backend("safe")
    try:
        forced = registry.resolve_route("pull", "cpu", ordinary)
        assert forced.backend == "safe"
        assert registry.bind_route(forced).implementation.forward() == "safe"
    finally:
        registry.set_backend(None)


def test_route_availability_change_rejects_prepared_binding():
    """A route that disappears after planning cannot execute stale callbacks."""
    available = True
    registry = BackendRegistry()
    registry.register("matvec", "demo", lambda: "demo", device="cpu")
    spec = RouteSpec(
        key=RouteKey("propagate", "post_pre_csr", "pull", "demo"),
        kernel_bindings=(("matvec", "demo"),),
        build=lambda kernels: RouteImplementation(
            forward=kernels["matvec"], backward=lambda: None
        ),
    )
    registry.register_route(spec, device="cpu", available=lambda: available)
    snapshot = registry.resolve_route("pull", "cpu")
    assert registry.bind_route(snapshot).implementation.forward() == "demo"

    available = False
    with pytest.raises(RuntimeError, match="unavailable"):
        registry.bind_route(snapshot)
    assert not registry.has_route("pull", "cpu")


@pytest.fixture
def _local_override():
    """Provide a registry-local override context for private registries."""
    from contextlib import contextmanager

    @contextmanager
    def override(registry, name):
        previous = registry._override_state()
        registry.set_backend(name)
        try:
            yield
        finally:
            registry._restore_override(previous)

    return override
