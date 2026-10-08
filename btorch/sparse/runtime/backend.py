"""Kernel backends and their registry.

A *kernel* is a named low-level routine (``"csr_matvec"``, ``"edge_grad"``,
``"spike_push"``); a *backend* is one implementation of it for a device type.
The registered custom operators in :mod:`btorch.sparse.runtime.ops` look their
kernels up here, so model code never names a backend.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class _Entry:
    name: str
    fn: Callable
    priority: int
    available: Callable[[], bool]


class KernelCache:
    """Memo for expensive backend artefacts (compiled kernels, workspaces).

    Entries are keyed by an arbitrary hashable (kernel name, dtype, block
    size, ...). Nothing in here is model state; it is never serialised.
    """

    def __init__(self) -> None:
        self._store: dict = {}

    def get(self, key, build: Callable):
        """Return the cached artefact for ``key``, building it on a miss."""
        try:
            return self._store[key]
        except KeyError:
            value = self._store[key] = build()
            return value

    def clear(self) -> None:
        self._store.clear()

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
    _override: str | None = None
    _resolved: dict[tuple[str, str], _Entry] = field(default_factory=dict)
    kernels: KernelCache = field(default_factory=KernelCache)

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
            entries[:] = [e for e in entries if e.name != name]
            entries.append(_Entry(name, fn, priority, available or (lambda: True)))
            entries.sort(key=lambda e: -e.priority)
        self._resolved.clear()

    def available(self, kernel: str, device: str) -> list[str]:
        """Names of the usable backends of ``kernel`` on ``device``, best
        first."""
        return [
            e.name for e in self._entries.get((kernel, device), []) if e.available()
        ]

    def resolve(self, kernel: str, device: str) -> Callable:
        """Return the implementation to use for ``kernel`` on ``device``."""
        return self._resolve(kernel, device).fn

    def name(self, kernel: str, device: str) -> str:
        """Name of the backend :meth:`resolve` would pick."""
        return self._resolve(kernel, device).name

    def _resolve(self, kernel: str, device: str) -> _Entry:
        key = (kernel, device)
        hit = self._resolved.get(key)
        if hit is not None:
            return hit
        entries = [e for e in self._entries.get(key, []) if e.available()]
        if not entries:
            raise RuntimeError(f"No backend for kernel {kernel!r} on {device!r}.")
        chosen = entries[0]
        if self._override is not None:
            # An override that does not implement this kernel falls back to
            # the default choice, so a partial backend can still be forced.
            chosen = next((e for e in entries if e.name == self._override), chosen)
        self._resolved[key] = chosen
        return chosen

    def set_backend(self, name: str | None) -> None:
        """Force backend ``name`` wherever it is available (``None``: auto).

        Raises:
            ValueError: If no kernel has a backend called ``name``.
        """
        known = {e.name for entries in self._entries.values() for e in entries}
        if name is not None and name not in known:
            raise ValueError(
                f"Unknown backend {name!r}; registered backends: {sorted(known)}."
            )
        self._override = name
        self._resolved.clear()


registry = BackendRegistry()


@contextmanager
def use_backend(name: str | None) -> Iterator[None]:
    """Temporarily force a kernel backend (advanced / debugging).

    Examples:
        >>> from btorch.sparse import runtime
        >>> with runtime.use_backend("aten"):      # doctest: +SKIP
        ...     y = conn(x)
    """
    previous = registry._override
    registry.set_backend(name)
    try:
        yield
    finally:
        registry.set_backend(previous)
