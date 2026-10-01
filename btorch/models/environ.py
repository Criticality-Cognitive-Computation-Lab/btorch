import functools
import threading
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Hashable


# Adapted from brainstate.

# Process-global defaults written by :func:`set` / :func:`unset`, visible from every
# thread. The dict is treated as immutable (copy-on-write): writers build a new
# dict under ``_SETTINGS_LOCK`` and rebind the name, so lock-free readers
# (:func:`get`, :func:`all`) always see one consistent snapshot. Reads stay
# lock-free so they remain cheap and ``torch.compile`` friendly.
_DEFAULTS: dict[Hashable, Any] = {}
_SETTINGS_LOCK = threading.Lock()


class _ThreadContexts(threading.local):
    """Thread-local override stacks pushed by :class:`context`."""

    def __init__(self):
        self.contexts: defaultdict[Hashable, list[Any]] = defaultdict(list)


_LOCAL = _ThreadContexts()


class context:
    """Context manager for temporary computation environment variables.

    Values pushed via ``context`` are thread-local (other threads keep
    seeing the global defaults from :func:`set`), take precedence over those
    defaults inside the ``with`` block, and are automatically popped on exit.
    Can be used as a decorator, context manager, or directly around forward
    passes.

    Args:
        **kwargs: Key-value pairs to push onto the context stack.

    Example:
        >>> with environ.context(dt=1.0):
        ...     spikes, states = model(x)

        >>> @environ.context(dt=1.0)
        ... def forward(model, x):
        ...     return model(x)
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __enter__(self):
        for k, v in self.kwargs.items():
            _LOCAL.contexts[k].append(v)
        return all()

    def __exit__(self, exc_type, exc_value, traceback):
        for k, v in self.kwargs.items():
            _LOCAL.contexts[k].pop()

    def __call__(self, func: Callable) -> Callable:
        return context_decorator(self, func)


def context_decorator(context_instance: "context", func: Callable) -> Callable:
    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
        with context_instance:
            return func(*args, **kwargs)

    return decorate_context


def get(key: str, desc: str | None = None) -> Any:
    """Get a value from the current computation environment.

    Checks the calling thread's context stack first, then the process-global
    defaults set by :func:`set`.

    Args:
        key: Environment variable name.
        desc: Optional description to include in the error message
            if the key is not found.

    Returns:
        The current value for ``key``.

    Raises:
        KeyError: If ``key`` is not found in context or defaults.

    Example:
        >>> dt = environ.get("dt")
    """

    stack = _LOCAL.contexts.get(key)
    if stack:
        return stack[-1]
    defaults = _DEFAULTS
    if key in defaults:
        return defaults[key]

    if desc is not None:
        raise KeyError(
            f"'{key}' is not found in the context. \n"
            f"You can set it by `environ.context({key}=value)` "
            f"locally or `environ.set({key}=value)` globally. \n"
            f"Description: {desc}"
        )
    else:
        raise KeyError(
            f"'{key}' is not found in the context. \n"
            f"You can set it by `environ.context({key}=value)` "
            f"locally or `environ.set({key}=value)` globally."
        )


def all() -> dict:
    """Get all current computation environment variables.

    Returns:
        Dictionary of this thread's active context values and the global
        default settings (context values win).
    """
    r = dict()
    for k, v in _LOCAL.contexts.items():
        if v:
            r[k] = v[-1]
    for k, v in _DEFAULTS.items():
        if k not in r:
            r[k] = v
    return r


def set(**kwargs: Any) -> None:
    """Set global default computation environment variables.

    The defaults are process-global: they persist until changed, are visible
    from every thread, and are used as fallbacks when a key is not present in
    the calling thread's active :class:`context` stack. Use :class:`context`
    for thread-local, scoped overrides.

    Args:
        **kwargs: Key-value pairs to set as defaults.

    Example:
        >>> environ.set(dt=1.0)
    """
    global _DEFAULTS
    with _SETTINGS_LOCK:
        _DEFAULTS = {**_DEFAULTS, **kwargs}


def unset(*keys: Hashable) -> None:
    """Remove global defaults previously set with :func:`set`.

    Unknown keys are ignored. Thread-local :class:`context` overrides are not
    affected.

    Args:
        *keys: Names of the defaults to remove.
    """
    global _DEFAULTS
    with _SETTINGS_LOCK:
        _DEFAULTS = {k: v for k, v in _DEFAULTS.items() if k not in keys}
