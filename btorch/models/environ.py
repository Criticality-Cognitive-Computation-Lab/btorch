import dataclasses
import functools
import threading
from collections import defaultdict
from typing import Any, Hashable


# copied from brainstate


@dataclasses.dataclass
class DefaultContext(threading.local):
    # default environment settings
    settings: dict[Hashable, Any] = dataclasses.field(default_factory=dict)
    # current environment settings
    contexts: defaultdict[Hashable, Any] = dataclasses.field(
        default_factory=lambda: defaultdict(list)
    )


DEFAULT = DefaultContext()


class context:
    """Context manager for temporary computation environment variables.

    Values pushed via ``context`` are thread-local and automatically
    popped on exit. Can be used as a decorator, context manager, or
    directly around forward passes.

    Args:
        **kwargs: Key-value pairs to push onto the context stack.

    Example:
        >>> with environ.context(dt=1.0):
        ...     spikes, states = model(x)

        >>> @environ.context(dt=1.0)
        ... def forward(model, x):
        ...     return model(x)
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        for k, v in self.kwargs.items():
            if k not in DEFAULT.contexts:
                DEFAULT.contexts[k] = []
            DEFAULT.contexts[k].append(v)
        return all()

    def __exit__(self, exc_type, exc_value, traceback):
        for k, v in self.kwargs.items():
            DEFAULT.contexts[k].pop()

    def __call__(self, func):
        return context_decorator(self, func)


def context_decorator(context_instance, func):
    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
        with context_instance:
            return func(*args, **kwargs)

    return decorate_context


def get(key: str, desc: str | None = None) -> Any:
    """Get a value from the current computation environment.

    Checks the context stack first, then global defaults.

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

    if key in DEFAULT.contexts:
        if len(DEFAULT.contexts[key]) > 0:
            return DEFAULT.contexts[key][-1]
    if key in DEFAULT.settings:
        return DEFAULT.settings[key]

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
        Dictionary of all active context and default settings.
    """
    r = dict()
    for k, v in DEFAULT.contexts.items():
        if v:
            r[k] = v[-1]
    for k, v in DEFAULT.settings.items():
        if k not in r:
            r[k] = v
    return r


def set(**kwargs):
    """Set global default computation environment variables.

    These values persist until changed and are used as fallbacks when
    a key is not present in the active context stack.

    Args:
        **kwargs: Key-value pairs to set as defaults.

    Example:
        >>> environ.set(dt=1.0)
    """
    DEFAULT.settings.update(kwargs)
