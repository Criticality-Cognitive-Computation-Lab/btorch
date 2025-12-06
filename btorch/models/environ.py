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


DFAULT = DefaultContext()


class context:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        for k, v in self.kwargs.items():
            if k not in DFAULT.contexts:
                DFAULT.contexts[k] = []
            DFAULT.contexts[k].append(v)
        return all()

    def __exit__(self, exc_type, exc_value, traceback):
        for k, v in self.kwargs.items():
            DFAULT.contexts[k].pop()

    def __call__(self, func):
        return context_decorator(self, func)


def context_decorator(context_instance, func):
    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
        with context_instance:
            return func(*args, **kwargs)

    return decorate_context


def get(key: str, desc: str | None = None):
    """Get one of the default computation environment.

    Returns
    -------
    item: Any
      The default computation environment.
    """

    if key in DFAULT.contexts:
        if len(DFAULT.contexts[key]) > 0:
            return DFAULT.contexts[key][-1]
    if key in DFAULT.settings:
        return DFAULT.settings[key]

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
    """Get all the current default computation environment.

    Returns
    -------
    r: dict
      The current default computation environment.
    """
    r = dict()
    for k, v in DFAULT.contexts.items():
        if v:
            r[k] = v[-1]
    for k, v in DFAULT.settings.items():
        if k not in r:
            r[k] = v
    return r


def set(**kwargs):
    # set default environment
    DFAULT.settings.update(kwargs)
