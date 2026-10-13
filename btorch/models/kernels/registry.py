"""Register cell implementations supplied by btorch or external packages."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

import torch

from .contracts import CellSpec


@dataclass(frozen=True)
class CellBackendEntry:
    """Describe a pure tensor implementation of the sequence callback contract.

    Args:
        name: Public backend name.
        forward: Sequence callback returning output, final states and histories.
        backward: Explicit VJP callback, or None for inference-only backends.
        supports: Predicate over scalar model choices and the input device.
        revision: Registration revision, incremented on explicit replacement.
    """

    name: str
    forward: Callable
    backward: Callable | None
    supports: Callable[[CellSpec, torch.device], bool] | None = None
    revision: int = 1


_BUILTINS = ("torch", "triton", "tilelang", "cupy")
_entries: dict[str, CellBackendEntry] = {}


def cell_backend_names() -> tuple[str, ...]:
    """Return known names without importing optional backend dependencies."""
    return (*_BUILTINS, *(name for name in _entries if name not in _BUILTINS))


def cell_backend_revision(name: str) -> int:
    """Return the registration revision without importing a backend.

    Capture caches use this value to invalidate graphs after an external
    implementation is replaced. Built-in lazy resolution keeps revision
    one.
    """
    entry = _entries.get(name)
    if entry is not None:
        return entry.revision
    if name in _BUILTINS:
        return 1
    raise ValueError(f"Unknown cell backend {name!r}")


def register_cell_backend(
    name: str,
    *,
    forward: Callable,
    backward: Callable | None = None,
    supports: Callable[[CellSpec, torch.device], bool] | None = None,
    replace: bool = False,
) -> CellBackendEntry:
    """Register an out-of-tree cell backend without modifying btorch.

    Forward receives ``(x_seq, states, parameters, spec, history_names)`` and
    returns ``(output_seq, final_states, histories)``. Backward receives those
    inputs, histories and output/final-state/history cotangents, plus the optional
    ``needs_input_grad`` keyword, and returns input/state/parameter gradients.
    Tensor ordering is declared by :class:`~.templates.ModelTemplate`.

    Args:
        name: Identifier used by the normal Module ``backend`` argument.
        forward: Pure tensor sequence forward callback.
        backward: First-order sequence VJP; None means inference only.
        supports: Optional predicate checking model choices and input device.
        replace: Explicitly replace a previous external registration. Replace
            after outstanding backward calls finish, then recreate compiled
            callables and standalone captured graphs. Pending backward calls
            reject changed registrations rather than applying a different VJP.

    Returns:
        Immutable registration description.

    Raises:
        ValueError: If the name is invalid, reserved or already registered.
        TypeError: If a supplied callback is not callable.
    """
    if (
        not isinstance(name, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
    ):
        raise ValueError("Cell backend names must be nonempty Python identifiers")
    if name in _BUILTINS:
        raise ValueError(f"The built-in cell backend {name!r} cannot be replaced")
    if not callable(forward) or (backward is not None and not callable(backward)):
        raise TypeError("forward and backward must be callable")
    if supports is not None and not callable(supports):
        raise TypeError("supports must be callable")
    previous = _entries.get(name)
    if previous is not None and not replace:
        raise ValueError(f"Cell backend {name!r} is already registered")
    entry = CellBackendEntry(
        name,
        forward,
        backward,
        supports,
        revision=1 if previous is None else previous.revision + 1,
    )
    _entries[name] = entry
    return entry


def resolve_cell_backend(
    name: str, spec: CellSpec, device: torch.device | str
) -> CellBackendEntry:
    """Resolve a backend and reject unsupported execution without fallback.

    Args:
        name: Built-in or externally registered name.
        spec: Immutable equation choices.
        device: Input tensor device.

    Returns:
        A validated callback registration.

    Raises:
        ValueError: If no implementation is registered under the name.
        NotImplementedError: If the implementation rejects the model or device.
    """
    entry = _entries.get(name)
    if entry is None:
        if name not in _BUILTINS:
            raise ValueError(
                f"Unknown cell backend {name!r}; known: {cell_backend_names()}"
            )
        path = ".torch_backend" if name == "torch" else f".backends.{name}"
        module = import_module(path, package=__package__)
        forward = getattr(module, "forward", None)
        backward = getattr(module, "backward", None)
        if name == "torch":
            forward, backward = module.forward_sequence, module.backward_sequence
        entry = CellBackendEntry(
            name, forward, backward, getattr(module, "supports", None)
        )
        _entries[name] = entry
    device = torch.device(device)
    if entry.supports is not None and not entry.supports(spec, device):
        raise NotImplementedError(
            f"Cell backend {name!r} does not support {spec.model!r} on {device}"
        )
    return entry
