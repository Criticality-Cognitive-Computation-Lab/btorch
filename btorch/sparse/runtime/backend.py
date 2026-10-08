"""Kernel backends and their registry.

A *kernel* is a named low-level routine (``"csr_matvec"``, ``"edge_grad"``,
``"spike_push"``); a *backend* is one implementation of it for a device type.
The registered custom operators in :mod:`btorch.sparse.runtime.ops` look their
kernels up here, so model code never names a backend.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .planner import PlanningContext


@dataclass(frozen=True)
class BackendEntrySnapshot:
    """Immutable description of one registered kernel implementation."""

    kernel: str
    device: str
    name: str
    priority: int
    revision: int
    available: bool


@dataclass(frozen=True)
class BackendOverrideSnapshot:
    """Immutable context-local backend override state."""

    name: str | None
    revision: int


@dataclass(frozen=True)
class BackendResolutionSnapshot:
    """Immutable identity of one resolved kernel route."""

    kernel: str
    device: str
    name: str
    revision: int
    registry_generation: int
    override: str | None
    override_revision: int

    @property
    def fingerprint(self) -> tuple:
        """Return a hashable identity for this resolution."""
        return (
            self.kernel,
            self.device,
            self.name,
            self.revision,
            self.registry_generation,
            self.override,
            self.override_revision,
        )


@dataclass(frozen=True)
class RouteImplementation:
    """The immutable execution callbacks bound to one route revision."""

    forward: Callable
    backward: Callable
    prepare: Callable | None = None
    prepare_values: Callable | None = None


@dataclass(frozen=True)
class RouteKey:
    """Representation, algorithm and backend identity of one operation."""

    operation: str
    representation: str
    algorithm: str
    backend: str


@dataclass(frozen=True)
class RegisteredOperator:
    """Static execution capabilities owned by one complete route."""

    supports_backward: bool = True
    supports_compile: bool = True
    supports_capture: bool = True


@dataclass(frozen=True)
class RouteSpec:
    """One complete forward, backward and preparation registration.

    ``kernel_bindings`` names the exact low-level implementations needed to
    build the route. They are resolved once by :meth:`BackendRegistry.bind_route`;
    execution never resolves them again.
    """

    key: RouteKey
    kernel_bindings: tuple[tuple[str, str], ...]
    build: Callable[[dict[str, Callable]], RouteImplementation]
    operator: RegisteredOperator = RegisteredOperator()
    supports: Callable[[object], bool] = lambda context: True
    estimate_cost: Callable[[object], float] = lambda context: 0.0

    @property
    def algorithm(self) -> str:
        """Return the route's algorithm dimension."""
        return self.key.algorithm

    @property
    def backend(self) -> str:
        """Return the route's backend dimension."""
        return self.key.backend


@dataclass(frozen=True)
class RouteResolutionSnapshot:
    """Immutable identity of one selected complete route."""

    key: RouteKey
    algorithm: str
    backend: str
    device: str
    revision: int
    registry_generation: int
    override: str | None
    override_revision: int
    kernel_revisions: tuple[tuple[str, str, int], ...]

    @property
    def fingerprint(self) -> tuple:
        """Return a hashable identity for this route revision."""
        return (
            self.key,
            self.algorithm,
            self.backend,
            self.device,
            self.revision,
            self.registry_generation,
            self.override,
            self.override_revision,
            self.kernel_revisions,
        )


@dataclass(frozen=True)
class RouteBinding:
    """A route snapshot plus callbacks captured at planning time."""

    snapshot: RouteResolutionSnapshot
    implementation: RouteImplementation

    @property
    def algorithm(self) -> str:
        """Return the selected algorithm family."""
        return self.snapshot.algorithm

    @property
    def backend(self) -> str:
        """Return the selected backend name."""
        return self.snapshot.backend


@dataclass(frozen=True)
class BackendRegistrySnapshot:
    """Immutable registry and override state for invalidation checks."""

    registry_generation: int
    override: str | None
    override_revision: int
    entries: tuple[BackendEntrySnapshot, ...]

    @property
    def generation(self) -> int:
        """Return the process-wide registry generation."""
        return self.registry_generation

    @property
    def backend_override(self) -> str | None:
        """Return the active context-local override."""
        return self.override

    @property
    def fingerprint(self) -> tuple:
        """Return a deterministic, hashable snapshot identity."""
        return (
            self.registry_generation,
            self.override,
            self.override_revision,
            tuple(
                (
                    entry.kernel,
                    entry.device,
                    entry.name,
                    entry.priority,
                    entry.revision,
                    entry.available,
                )
                for entry in self.entries
            ),
        )


@dataclass(frozen=True)
class _OverrideState:
    name: str | None
    revision: int


@dataclass
class _Entry:
    name: str
    fn: Callable
    priority: int
    available: Callable[[], bool]
    revision: int


@dataclass
class _RouteEntry:
    spec: RouteSpec
    device: str
    priority: int
    available: Callable[[], bool]
    revision: int


class KernelCache:
    """Memo for expensive backend artefacts (compiled kernels, workspaces).

    Entries are keyed by an arbitrary hashable (kernel name, dtype, block
    size, ...). Nothing in here is model state; it is never serialised.
    """

    def __init__(self) -> None:
        self._store: dict = {}
        self._generation = 0

    @property
    def generation(self) -> int:
        """Return a counter advanced whenever all artefacts are discarded."""
        return self._generation

    def get(self, key, build: Callable):
        """Return the cached artefact for ``key``, building it on a miss."""
        try:
            return self._store[key]
        except KeyError:
            value = self._store[key] = build()
            return value

    def clear(self) -> None:
        self._store.clear()
        self._generation += 1

    def __len__(self) -> int:
        return len(self._store)


@dataclass
class BackendRegistry:
    """Registry mapping ``(kernel, device type)`` to implementations.

    The highest-priority available backend is used unless an override is
    active (see :func:`use_backend`). Backends are advanced runtime options;
    they do not appear in model constructors.
    """

    _entries: dict[tuple[str, str], list[_Entry]] = field(default_factory=dict)
    _route_entries: dict[tuple[RouteKey, str], _RouteEntry] = field(
        default_factory=dict
    )
    _resolved: dict[tuple[str, str, str | None], _Entry] = field(default_factory=dict)
    _has: dict[tuple[str, str, str | None], bool] = field(default_factory=dict)
    _entry_revisions: dict[tuple[str, str, str], int] = field(default_factory=dict)
    _availability: dict[tuple[str, str, str, int], bool] = field(default_factory=dict)
    _route_revisions: dict[tuple[RouteKey, str], int] = field(default_factory=dict)
    _route_availability: dict[tuple[RouteKey, str, int], bool] = field(
        default_factory=dict
    )
    _registry_generation: int = 0
    _override_context: ContextVar[_OverrideState] = field(
        default_factory=lambda: ContextVar(
            "btorch_sparse_backend_override", default=_OverrideState(None, 0)
        ),
        repr=False,
    )
    kernels: KernelCache = field(default_factory=KernelCache)

    @property
    def _override(self) -> str | None:
        """Return the current context-local override.

        This private compatibility property is retained for older callers and
        tests. New code should use :attr:`override` or :meth:`snapshot`.
        """
        return self._override_state().name

    @_override.setter
    def _override(self, name: str | None) -> None:
        self.set_backend(name)

    @property
    def override(self) -> str | None:
        """Return the backend override active in the current context."""
        return self._override_state().name

    @property
    def override_revision(self) -> int:
        """Return the current context-local override revision."""
        return self._override_state().revision

    @property
    def registry_generation(self) -> int:
        """Return the process-wide registry generation."""
        return self._registry_generation

    @property
    def generation(self) -> int:
        """Alias for :attr:`registry_generation`."""
        return self._registry_generation

    def register(
        self,
        kernel: str,
        name: str,
        fn: Callable,
        *,
        device: str | tuple[str, ...] = ("cpu", "cuda"),
        priority: int = 0,
        available: Callable[[], bool] | None = None,
    ) -> None:
        """Register ``fn`` as backend ``name`` of ``kernel``.

        Args:
            kernel: Kernel name.
            name: Backend name (``"aten"``, ``"triton"``, ...).
            fn: Implementation.
            device: Device type(s) the implementation supports.
            priority: Higher wins when several backends are available.
            available: Optional lazy availability check.
        """
        devices = (device,) if isinstance(device, str) else device
        for dev in devices:
            entries = self._entries.setdefault((kernel, dev), [])
            entries[:] = [entry for entry in entries if entry.name != name]
            entry_key = (kernel, dev, name)
            revision = self._entry_revisions.get(entry_key, 0) + 1
            self._entry_revisions[entry_key] = revision
            self._availability = {
                key: value
                for key, value in self._availability.items()
                if key[:3] != entry_key
            }
            entries.append(
                _Entry(
                    name,
                    fn,
                    priority,
                    available or (lambda: True),
                    revision,
                )
            )
            entries.sort(key=lambda entry: -entry.priority)
        self._invalidate_resolution()
        self._registry_generation += 1

    def available(self, kernel: str, device: str) -> list[str]:
        """Names of the usable backends of ``kernel`` on ``device``, best
        first."""
        return [entry.name for entry in self._usable_entries(kernel, device)]

    def register_route(
        self,
        spec: RouteSpec,
        *,
        device: str | tuple[str, ...] = ("cpu", "cuda"),
        priority: int = 0,
        available: Callable[[], bool] | None = None,
        replace: bool = False,
    ) -> None:
        """Register one complete route candidate.

        A route is deliberately separate from legacy kernel registrations:
        the latter remain independently usable by the public low-level ops,
        while planning only sees complete route contracts.
        """
        devices = (device,) if isinstance(device, str) else device
        for dev in devices:
            key = (spec.key, dev)
            if key in self._route_entries and not replace:
                raise ValueError(
                    f"Route {spec.key!r} is already registered for {dev!r}; "
                    "use replace=True to replace it."
                )
            revision = self._route_revisions.get(key, 0) + 1
            self._route_revisions[key] = revision
            self._route_availability = {
                marker: value
                for marker, value in self._route_availability.items()
                if marker[:2] != key
            }
            self._route_entries[key] = _RouteEntry(
                spec=spec,
                device=dev,
                priority=priority,
                available=available or (lambda: True),
                revision=revision,
            )
        self._invalidate_resolution()
        self._registry_generation += 1

    def has_route(
        self,
        algorithm: str,
        device: str,
        context: PlanningContext | None = None,
    ) -> bool:
        """Return whether a complete route supports ``context``."""
        return bool(self._usable_routes(algorithm, device, context))

    def resolve_route(
        self,
        algorithm: str,
        device: str,
        context: PlanningContext | None = None,
    ) -> RouteResolutionSnapshot:
        """Select one complete route and return only its immutable identity."""
        candidates = self._usable_routes(algorithm, device, context)
        if not candidates:
            raise RuntimeError(f"No route for algorithm {algorithm!r} on {device!r}.")
        override = self.override
        if override is not None:
            forced = [entry for entry in candidates if entry.spec.backend == override]
            if forced:
                candidates = forced
        chosen = min(
            candidates,
            key=lambda entry: (
                -entry.priority,
                entry.spec.estimate_cost(context),
                entry.spec.key,
            ),
        )
        state = self._override_state()
        kernel_revisions = tuple(
            (kernel, backend, self.entry_revision(kernel, backend, device))
            for kernel, backend in chosen.spec.kernel_bindings
        )
        return RouteResolutionSnapshot(
            key=chosen.spec.key,
            algorithm=chosen.spec.algorithm,
            backend=chosen.spec.backend,
            device=device,
            revision=chosen.revision,
            registry_generation=self._registry_generation,
            override=state.name,
            override_revision=state.revision,
            kernel_revisions=kernel_revisions,
        )

    def bind_route(self, snapshot: RouteResolutionSnapshot) -> RouteBinding:
        """Bind exact callbacks for a previously selected route snapshot."""
        if snapshot.registry_generation != self._registry_generation:
            raise RuntimeError(
                "The selected sparse route is stale; replan before binding it."
            )
        entry = self._route_entries.get((snapshot.key, snapshot.device))
        if entry is None or entry.revision != snapshot.revision:
            raise RuntimeError(
                f"Sparse route {snapshot.key!r} changed before it was bound."
            )
        if not self._route_is_available(entry):
            raise RuntimeError(
                f"Sparse route {snapshot.key!r} is unavailable on "
                f"{snapshot.device!r}."
            )
        callbacks: dict[str, Callable] = {}
        for kernel, backend in entry.spec.kernel_bindings:
            current = self._resolve_exact(kernel, snapshot.device, backend)
            expected = next(
                revision
                for name, name_backend, revision in snapshot.kernel_revisions
                if name == kernel and name_backend == backend
            )
            if current.revision != expected:
                raise RuntimeError(
                    f"Sparse route {snapshot.key!r} changed its {kernel!r} "
                    "implementation before it was bound."
                )
            callbacks[kernel] = current.fn
        return RouteBinding(snapshot, entry.spec.build(callbacks))

    def route_revision(self, key: RouteKey, device: str) -> int:
        """Return the revision of a registered route."""
        try:
            return self._route_revisions[(key, device)]
        except KeyError as error:
            raise KeyError(f"No route {key!r} for device {device!r}.") from error

    def has(self, kernel: str, device: str) -> bool:
        """Whether a backend of ``kernel`` is usable on ``device``.

        With a forced backend (:func:`use_backend`) only that backend counts.
        """
        override = self.override
        key = (kernel, device, override)
        known = self._has.get(key)
        if known is None:
            entries = self._usable_entries(kernel, device)
            if override is not None:
                # A forced backend that lacks this (optional) kernel means "no".
                known = any(entry.name == override for entry in entries)
            else:
                known = bool(entries)
            self._has[key] = known
        return known

    def resolve(self, kernel: str, device: str) -> Callable:
        """Return the implementation to use for ``kernel`` on ``device``."""
        return self._resolve(kernel, device).fn

    def name(self, kernel: str, device: str) -> str:
        """Name of the backend :meth:`resolve` would pick."""
        return self._resolve(kernel, device).name

    def _resolve(self, kernel: str, device: str) -> _Entry:
        override = self.override
        key = (kernel, device, override)
        entries = self._usable_entries(kernel, device)
        if not entries:
            raise RuntimeError(f"No backend for kernel {kernel!r} on {device!r}.")
        hit = self._resolved.get(key)
        if hit is not None and any(entry is hit for entry in entries):
            return hit
        chosen = entries[0]
        if override is not None:
            # An override that does not implement this kernel falls back to
            # the default choice, so a partial backend can still be forced.
            chosen = next(
                (entry for entry in entries if entry.name == override), chosen
            )
        self._resolved[key] = chosen
        return chosen

    def _resolve_exact(self, kernel: str, device: str, name: str) -> _Entry:
        """Resolve one named implementation without consulting overrides."""
        entries = self._usable_entries(kernel, device)
        for entry in entries:
            if entry.name == name:
                return entry
        raise RuntimeError(
            f"No available backend {name!r} for kernel {kernel!r} on {device!r}."
        )

    def set_backend(self, name: str | None) -> None:
        """Force backend ``name`` in the current context (``None``: auto).

        Raises:
            ValueError: If no kernel has a backend called ``name``.
        """
        known = {entry.name for entries in self._entries.values() for entry in entries}
        if name is not None and name not in known:
            raise ValueError(
                f"Unknown backend {name!r}; registered backends: {sorted(known)}."
            )
        current = self._override_state()
        self._set_override_state(_OverrideState(name, current.revision + 1))
        self._invalidate_resolution()

    def entry_revision(self, kernel: str, name: str, device: str) -> int:
        """Return the monotonic revision of a registered implementation."""
        try:
            return self._entry_revisions[(kernel, device, name)]
        except KeyError as error:
            raise KeyError(
                f"No backend {name!r} for kernel {kernel!r} on {device!r}."
            ) from error

    revision = entry_revision

    def override_snapshot(self) -> BackendOverrideSnapshot:
        """Return the current context-local override identity."""
        state = self._override_state()
        return BackendOverrideSnapshot(state.name, state.revision)

    def snapshot(self) -> BackendRegistrySnapshot:
        """Return immutable registry state suitable for invalidation keys."""
        records: list[BackendEntrySnapshot] = []
        for (kernel, device), entries in sorted(self._entries.items()):
            statuses = self._observe_availability(kernel, device, entries)
            records.extend(
                BackendEntrySnapshot(
                    kernel=kernel,
                    device=device,
                    name=entry.name,
                    priority=entry.priority,
                    revision=entry.revision,
                    available=available,
                )
                for entry, available in zip(entries, statuses, strict=True)
            )
        state = self._override_state()
        return BackendRegistrySnapshot(
            registry_generation=self._registry_generation,
            override=state.name,
            override_revision=state.revision,
            entries=tuple(records),
        )

    def fingerprint(self) -> tuple:
        """Return a deterministic, hashable registry identity."""
        return self.snapshot().fingerprint

    def resolve_snapshot(self, kernel: str, device: str) -> BackendResolutionSnapshot:
        """Return immutable identity for the currently selected route."""
        entry = self._resolve(kernel, device)
        state = self._override_state()
        return BackendResolutionSnapshot(
            kernel=kernel,
            device=device,
            name=entry.name,
            revision=entry.revision,
            registry_generation=self._registry_generation,
            override=state.name,
            override_revision=state.revision,
        )

    def _override_state(self) -> _OverrideState:
        return self._override_context.get()

    def _set_override_state(self, state: _OverrideState) -> None:
        self._override_context.set(state)

    def _restore_override(self, state: _OverrideState) -> None:
        current = self._override_state()
        self._set_override_state(
            _OverrideState(state.name, max(current.revision, state.revision) + 1)
        )

    def _invalidate_resolution(self) -> None:
        self._resolved.clear()
        self._has.clear()

    def _observe_availability(
        self, kernel: str, device: str, entries: list[_Entry]
    ) -> tuple[bool, ...]:
        statuses: list[bool] = []
        changed = False
        for entry in entries:
            available = bool(entry.available())
            marker = (kernel, device, entry.name, entry.revision)
            previous = self._availability.get(marker)
            if previous is not None and previous != available:
                changed = True
            self._availability[marker] = available
            statuses.append(available)
        if changed:
            self._invalidate_resolution()
            self._registry_generation += 1
        return tuple(statuses)

    def _usable_entries(self, kernel: str, device: str) -> list[_Entry]:
        entries = self._entries.get((kernel, device), [])
        statuses = self._observe_availability(kernel, device, entries)
        return [
            entry
            for entry, available in zip(entries, statuses, strict=True)
            if available
        ]

    def _route_is_available(self, entry: _RouteEntry) -> bool:
        marker = (entry.spec.key, entry.device, entry.revision)
        available = bool(entry.available())
        previous = self._route_availability.get(marker)
        if previous is not None and previous != available:
            self._invalidate_resolution()
            self._registry_generation += 1
        self._route_availability[marker] = available
        return available

    def _usable_routes(
        self,
        algorithm: str,
        device: str,
        context: PlanningContext | None,
    ) -> list[_RouteEntry]:
        override = self.override
        candidates = [
            entry
            for entry in self._route_entries.values()
            if entry.device == device
            and entry.spec.algorithm == algorithm
            and (override is None or entry.spec.backend == override)
            and self._route_is_available(entry)
            and entry.spec.supports(context)
            and (
                not getattr(context, "requires_backward", False)
                or entry.spec.operator.supports_backward
            )
            and (
                not getattr(context, "compile", False)
                or entry.spec.operator.supports_compile
            )
            and (
                not getattr(context, "capture", False)
                or entry.spec.operator.supports_capture
            )
        ]
        return candidates


registry = BackendRegistry()


@contextmanager
def use_backend(name: str | None) -> Iterator[None]:
    """Temporarily force a kernel backend in the current context.

    Examples:
        >>> from btorch.sparse import runtime
        >>> with runtime.use_backend("aten"):      # doctest: +SKIP
        ...     y = conn(x)
    """
    previous = registry._override_state()
    registry.set_backend(name)
    try:
        yield
    finally:
        registry._restore_override(previous)
        registry._invalidate_resolution()
