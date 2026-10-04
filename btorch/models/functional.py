import warnings
from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import nn

from . import base


def _call_on_modules(
    net: nn.Module,
    method: Literal["init_state", "reset"],
    batch_size: int | Sequence[int] | None,
    **kwargs: Any,
) -> None:
    """Move ``net`` to ``kwargs`` device/dtype, then call ``method``
    everywhere.

    Warns (instead of failing) for modules that expose ``method`` without being
    a :class:`base.MemoryModule`. The check unwraps ``torch.compile`` wrappers
    via ``_orig_mod``, which plain modules do not have.
    """
    net.to(device=kwargs.get("device"), dtype=kwargs.get("dtype"))
    for m in net.modules():
        fn = getattr(m, method, None)
        if not callable(fn):
            continue
        if not isinstance(getattr(m, "_orig_mod", m), base.MemoryModule):
            # stacklevel=3: skip this helper and the public caller
            warnings.warn(
                f"Trying to call `{method}()` of {m}, which is not base.MemoryModule",
                stacklevel=3,
            )
        fn(batch_size, **kwargs)


def init_net_state(
    net: nn.Module,
    batch_size: int | Sequence[int] | None = None,
    **kwargs: Any,
) -> None:
    """Initialize state for all MemoryModule instances in a network.

    Walks through every ``Module`` in ``net`` and calls ``init_state()``
    if it is a ``base.MemoryModule`` or has an ``init_state`` method.
    Also moves the network to the device/dtype specified in ``kwargs``.

    Args:
        net: Network to initialize.
        batch_size: Batch size(s) for state initialization.
        **kwargs: Passed to ``init_state()`` (e.g., ``device``, ``dtype``).

    Example:
        >>> functional.init_net_state(model, batch_size=4, device="cuda")
    """

    _call_on_modules(net, "init_state", batch_size, **kwargs)


def reset_net(
    net: nn.Module,
    batch_size: int | Sequence[int] | None = None,
    **kwargs: Any,
) -> None:
    """Reset state for all MemoryModule instances in a network.

    Walks through every ``Module`` in ``net`` and calls ``reset()``
    if it is a ``base.MemoryModule`` or has a ``reset`` method.
    Also moves the network to the device/dtype specified in ``kwargs``.

    Args:
        net: Network to reset.
        batch_size: Batch size(s) for state reset. If None, uses existing size.
        **kwargs: Passed to ``reset()`` (e.g., ``device``, ``dtype``).

    Example:
        >>> functional.reset_net(model, batch_size=4)
    """

    _call_on_modules(net, "reset", batch_size, **kwargs)


reset_net_state = reset_net


def _strip_self(d: set[str]) -> set[str]:
    return set(s.removeprefix("self.").removeprefix("self") for s in d)


def _collect_memory_vars(
    mod: nn.Module,
    target_attr: Literal["_memories", "_memories_rv"],
    names: Sequence[str] | None = None,
    allow_buffer: bool = False,
    clone: bool = False,
) -> dict[str, Any]:
    """Return a proper dotted dict flattened up to items of _memories*.

    Single pass over the module tree (hot path): no per-module intermediate
    dicts, and ``clone`` only touches tensors when actually requested.
    """
    # None -> take everything; else keep a module's whole state (its name matched)
    # or the individually named children (dotted key matched).
    names_set = _strip_self(set(names)) if names is not None else None
    ret = {}
    for name, m in mod.named_modules():
        if not (allow_buffer or isinstance(m, base.MemoryModule)):
            continue
        prefix = "" if name == "" else f"{name}."
        for k, v in getattr(m, target_attr).items():
            key = prefix + k
            if names_set is None or name in names_set or key in names:
                ret[key] = v.clone() if clone and torch.is_tensor(v) else v
    return ret


def _walk_memory_targets(
    mod: nn.Module,
    target_attr: Literal["_memories", "_memories_rv"],
    hidden_states: dict[str, Any] | None,
    allow_buffer: bool = False,
):
    """Resolve each dotted entry of ``hidden_states`` to what it addresses.

    The memories* level doesn't have to be flattened to a dotted dict, e.g.
    ``{"mod": {"v": array1, "Iasc": array2}}`` is accepted. This only
    *locates* targets; the caller decides how to write them. Yields
    ``(kind, module, key, value)`` where ``kind`` is one of:

    * ``"whole"``: ``value`` is a dict for all memories of a MemoryModule.
    * ``"attr"``: ``value`` is one memory ``key`` of a MemoryModule.
    * ``"buffer_whole"`` / ``"buffer_attr"``: the same for plain buffers of a
      non-MemoryModule (only with ``allow_buffer``).
    """
    if hidden_states is None:
        return

    # Entries are not checked for overlap: addressing one memory twice, e.g.
    # ``{"a": {"mem": v0}, "a.mem": v1}``, applies both writes in dict order, so
    # the later one wins.

    for name, hidden_state in hidden_states.items():
        if hidden_state is None:
            continue
        path = name.removeprefix("self.").removeprefix("self").split(".")
        m = mod
        if path[0] == "":
            # set self's mem vars via either {"": {"v": tensor}}
            # or {"self": {"v": tensor}}
            if isinstance(m, base.MemoryModule):
                yield "whole", m, None, hidden_state
            elif allow_buffer and isinstance(m, nn.Module):
                yield "buffer_whole", m, None, hidden_state
            continue
        for p in path[:-1]:
            m = getattr(m, p)
        if target_attr == "_memories_rv" and path[-1] in getattr(m, "_memories_rv", {}):
            # reset values are not attributes; look them up in the registry.
            # Anything else (a submodule, for {"m.subm": {...}}) is an attribute.
            m_leaf = m._memories_rv[path[-1]]
        elif target_attr == "_memories_rv" and not hasattr(m, path[-1]):
            raise KeyError(
                f"{name!r} is neither a registered memory nor a submodule; "
                f"registered: {list(getattr(m, '_memories_rv', {}))}"
            )
        else:
            m_leaf = getattr(m, path[-1])
        if isinstance(m_leaf, nn.Module):
            # set the whole module's mem vars via {"m.subm": {"v": tensor}}
            if isinstance(m_leaf, base.MemoryModule):
                yield "whole", m_leaf, None, hidden_state
            elif allow_buffer:
                yield "buffer_whole", m_leaf, None, hidden_state
        elif allow_buffer:
            # set a specific mem var via {"m.subm.v": tensor}
            yield "buffer_attr", m, path[-1], hidden_state
        else:
            if not isinstance(m, base.MemoryModule):
                raise TypeError(
                    f"cannot set memory {path[-1]!r}: {type(m).__name__} is not a "
                    "MemoryModule"
                )
            yield "attr", m, path[-1], hidden_state


def _set_buffers(m: nn.Module, kv: dict[str, Any], inplace: bool):
    # inplace copies into the existing buffer (keeps its address); else rebinds
    for k, v in kv.items():
        if k not in m._buffers:
            raise KeyError(f"{k} not in buffers {list(m._buffers)}")
        getattr(m, k).copy_(v) if inplace else setattr(m, k, v)


def _set_memories(
    mod: nn.Module,
    hidden_states: dict[str, Any] | None,
    allow_buffer: bool = False,
    inplace: bool = False,
):
    """Write ``_memories`` entries (rebinding, or copying when ``inplace``)."""
    for kind, m, key, v in _walk_memory_targets(
        mod, "_memories", hidden_states, allow_buffer
    ):
        if kind == "whole":
            if inplace:
                for k, val in v.items():
                    m._memories[k].copy_(val)
            else:
                m._memories = v
        elif kind == "attr":
            if inplace:
                m._memories[key].copy_(v)
            else:
                m._memories = {key: v}
        elif kind == "buffer_whole":
            _set_buffers(m, v, inplace)
        else:
            _set_buffers(m, {key: v}, inplace)


def _set_reset_values(
    mod: nn.Module, reset_values: dict[str, Any] | None, strict: bool
):
    """Write ``_memories_rv`` entries through the MemoryModule setters."""
    for kind, m, key, v in _walk_memory_targets(
        mod, "_memories_rv", reset_values, allow_buffer=False
    ):
        if kind == "whole":
            m.set_memories_rv(v, strict=strict)
        else:
            m.set_reset_value(key, v, strict=strict)


# for serialisation as well as rnn to collect states
def named_hidden_states(
    mod: nn.Module,
    names: Sequence[str] | None = None,
    allow_buffer: bool = False,
    clone: bool = False,
) -> dict[str, Any]:
    """Collect hidden states (_memories) from a network as a dotted dict.

    Args:
        mod: Network module to collect from.
        names: Optional sequence of dotted state names to filter.
        allow_buffer: If True, also collect from non-MemoryModule buffers.
        clone: If True, clone each tensor so the returned snapshot is decoupled
            from the live state (e.g. a start state to restore under CUDA graph
            capture, which must not alias the buffers the step overwrites).

    Returns:
        Dotted dictionary mapping ``module.state_name`` to tensor values.

    Example:
        >>> states = functional.named_hidden_states(model)
        >>> states.keys()
        dict_keys(['neuron.v', 'synapse.psc'])
    """
    return _collect_memory_vars(
        mod, "_memories", names, allow_buffer=allow_buffer, clone=clone
    )


named_memory_values = filter_hidden_states = named_hidden_states


def set_hidden_states(
    mod: nn.Module,
    hidden_states: dict[str, Any],
    allow_buffer: bool = False,
    inplace: bool = False,
) -> None:
    """Set hidden states (_memories) in a network from a dotted dict.

    Args:
        mod: Network module to update.
        hidden_states: Dotted dictionary of states.
        allow_buffer: If True, also set on non-MemoryModule buffers.
        inplace: If True, copy values into the existing state tensors instead of
            rebinding them, so the buffers keep their identity and memory addresses
            (e.g. restoring state inside a captured inference graph). Targets must
            already exist with matching shapes. Use only outside autograd: copying
            into a buffer that a live graph still needs raises "a variable needed
            for gradient computation was modified by an inplace operation".

    Example:
        >>> functional.set_hidden_states(model, {"neuron.v": v_tensor})
    """
    _set_memories(mod, hidden_states, allow_buffer=allow_buffer, inplace=inplace)


set_memory_values = set_hidden_states


# Reset values are exchanged as a dotted flattened dict (same layout as
# ``named_hidden_states``), mainly for serialisation, e.g. with torch.save.
def named_memory_reset_values(
    mod: nn.Module, names: Sequence[str] | None = None
) -> dict[str, Any]:
    """Collect memory reset values (_memories_rv) from a network.

    Args:
        mod: Network module to collect from.
        names: Optional sequence of dotted state names to filter.

    Returns:
        Dotted dictionary mapping ``module.state_name`` to reset values.

    Example:
        >>> rv = functional.named_memory_reset_values(model)
    """
    return _collect_memory_vars(mod, "_memories_rv", names, allow_buffer=False)


def set_memory_reset_values(
    mod: nn.Module, reset_values: dict[str, Any], strict: bool = True
) -> None:
    """Set memory reset values (_memories_rv) in a network.

    Network-wide counterpart of :meth:`MemoryModule.set_memories_rv`.
    The method takes a flat ``{memory name: value}`` mapping for one module;
    this function takes a dotted dict addressing memories anywhere in the
    tree, in the layout of :func:`named_memory_reset_values`. An entry may also
    address a whole module with a nested dict, e.g.
    ``{"neuron": {"v": 0.0}}``. Both default to ``strict=True``.

    Args:
        mod: Network module to update.
        reset_values: Dotted dictionary ``{"module.memory": reset value}``
            (e.g. from :func:`named_memory_reset_values`).
        strict: If True, a ``ResetValue`` whose ``sizes`` differ from the
            registered ones raises ``ValueError``. Passed to
            :meth:`MemoryModule.set_reset_value`.

    Example:
        >>> rv = functional.named_memory_reset_values(model)
        >>> functional.set_memory_reset_values(model, rv)
    """

    _set_reset_values(mod, reset_values, strict=strict)


# Short aliases matching the ``memories_rv`` naming of :class:`MemoryModule`.
named_memories_rv = named_memory_reset_values
set_memories_rv = set_memory_reset_values


def detach_net(net: nn.Module) -> None:
    """Detach the computation graph of the whole network from previous time
    steps.

    Walks through every ``Module`` in ``net`` and calls ``detach()``
    if it is a ``base.MemoryModule`` or has a ``detach`` method.

    Args:
        net: Network to detach.

    Example:
        >>> functional.detach_net(model)
    """

    for m in net.modules():
        if hasattr(m, "detach") and callable(m.detach):
            if not isinstance(m, base.MemoryModule):
                warnings.warn(
                    f"Trying to call `detach()` of {m}, which is not "
                    "btorch.models.base.MemoryModule",
                    stacklevel=2,
                )
            m.detach()
