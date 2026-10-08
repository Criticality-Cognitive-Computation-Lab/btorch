"""Tests for backend selection state and invalidation identities."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from btorch.sparse import runtime
from btorch.sparse.runtime.backend import (
    BackendRegistry,
    RouteImplementation,
    RouteKey,
    RouteSpec,
)


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
