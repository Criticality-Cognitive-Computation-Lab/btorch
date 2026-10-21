"""Explicit propagation methods, independent of storage format and hints."""

from contextlib import contextmanager
from threading import local
from typing import Literal


PropagationMethod = Literal["auto", "pull", "push"]
# Attribute reads are guarded by Dynamo, so changing scopes also changes a
# compiled route. ContextVar.get cannot be traced in a fullgraph model.
_scope = local()


def validate_method(method: PropagationMethod) -> None:
    """Reject unknown execution methods."""
    if method not in ("auto", "pull", "push"):
        raise ValueError(
            f"Unknown sparse method {method!r}; expected 'auto', 'pull', or 'push'"
        )


def resolve_method(method: PropagationMethod) -> PropagationMethod:
    """Apply a scoped request over the connection's own method."""
    scoped = getattr(_scope, "method", "auto")
    return method if scoped == "auto" else scoped


@contextmanager
def use_method(method: PropagationMethod):
    """Request a sparse execution method within this scope.

    This thread-local scope overrides connection settings and activity hints. ``auto``
    inherits the surrounding scope. Unsupported explicit requests raise rather
    than silently selecting another method. Configure scopes outside compiled
    model execution, like sparse backend selection.

    Args:
        method: ``auto``, destination-driven ``pull``, or spike-list ``push``.
    """
    validate_method(method)
    previous = getattr(_scope, "method", "auto")
    _scope.method = resolve_method(method) if method == "auto" else method
    try:
        yield
    finally:
        _scope.method = previous
