"""In-place state ops that preserve buffer identity -- needed under CUDA graph
capture, where every state tensor must keep its address across replays.

``MemoryModule.reset(inplace=True)`` and ``set_hidden_states(..., inplace=True)``
write into the existing buffers instead of rebinding them to fresh tensors;
``named_hidden_states(..., clone=True)`` snapshots state decoupled from the live
buffers (e.g. a start state to restore each step).
"""

import pytest
import torch

from btorch.models.base import MemoryModule
from btorch.models.functional import named_hidden_states, set_hidden_states


class TwoState(MemoryModule):
    """A zero-init memory (host-free ``zero_`` path) and a constant one
    (``copy_`` path)."""

    def __init__(self, n):
        super().__init__()
        self.register_memory("v", torch.zeros(1), n)  # zero-init
        self.register_memory("w", 2.0, n)  # non-zero constant
        self.init_state()


def test_reset_inplace_reuses_buffers():
    m = TwoState(4)
    m.reset(batch_size=2)
    v, w = m.v, m.w
    m.v.add_(5.0)  # dirty the state
    m.w.add_(5.0)

    m.reset(batch_size=2, inplace=True)

    # Same tensor objects (address preserved), values reset to the registered ones.
    assert m.v is v and m.w is w
    assert torch.equal(m.v, torch.zeros(2, 4))
    assert torch.equal(m.w, torch.full((2, 4), 2.0))

    # Contrast: the default reset rebinds to freshly-allocated tensors.
    m.reset(batch_size=2)
    assert m.v is not v


def test_reset_inplace_cannot_resize():
    m = TwoState(4)
    m.reset(batch_size=2)
    with pytest.raises(ValueError, match="cannot resize"):
        m.reset(batch_size=8, inplace=True)  # different batch -> no realloc allowed


def test_named_hidden_states_clone_decouples():
    m = TwoState(4)
    m.reset(batch_size=2)

    snap = named_hidden_states(m, clone=True)  # decoupled snapshot
    live = named_hidden_states(m)  # references into the live buffers
    m.v.add_(5.0)  # mutate the live state

    assert torch.equal(snap["v"], torch.zeros(2, 4))  # snapshot unaffected
    assert live["v"] is m.v  # no-clone aliases the buffer


def test_set_hidden_states_inplace_preserves_buffers():
    m = TwoState(4)
    m.reset(batch_size=2)
    v = m.v

    set_hidden_states(m, {"v": torch.ones(2, 4)}, inplace=True)
    assert m.v is v  # same object -> address preserved (for CUDA graph replay)
    assert torch.equal(m.v, torch.ones(2, 4))

    # Contrast: the default rebinds to a fresh tensor.
    set_hidden_states(m, {"v": torch.full((2, 4), 3.0)})
    assert m.v is not v and torch.equal(m.v, torch.full((2, 4), 3.0))


def test_net_state_helpers_warn_for_plain_module():
    """Plain (non-compiled) modules exposing ``reset``/``init_state`` only
    warn.

    Regression: the warning path read ``m._orig_mod``, which only exists on
    ``torch.compile``d modules, so a plain ``nn.Module`` raised AttributeError.
    """
    import torch.nn as nn

    from btorch.models import functional

    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def init_state(self, batch_size=None, **kwargs):
            self.calls.append("init_state")

        def reset(self, batch_size=None, **kwargs):
            self.calls.append("reset")

    net = nn.Sequential(Plain())
    # ``warnings.warn`` (not root-logger logging) is used, so capture with pytest.warns.
    with pytest.warns(UserWarning) as record:
        functional.init_net_state(net, batch_size=2)
        functional.reset_net(net, batch_size=2)

    assert net[0].calls == ["init_state", "reset"]
    text = " ".join(str(w.message) for w in record)
    assert "init_state()" in text and "reset()" in text
    # stacklevel points at the caller (this test), not btorch internals.
    assert all(w.filename == __file__ for w in record)


def test_register_memory_validates_user_input():
    """Bad arguments to ``register_memory`` raise ``ValueError``, not
    assert."""
    m = TwoState(3)
    # Name clashes with an existing attribute.
    with pytest.raises(ValueError, match="member variable"):
        m.register_memory("v", torch.zeros(1), 3)
    # Empty sequences carry no value to broadcast.
    with pytest.raises(ValueError, match="empty"):
        m.register_memory("z", [], 3)


def test_memories_rv_setters_reject_unknown_keys():
    """Unknown memory names are a ``KeyError`` for every setter entry point."""
    m = TwoState(3)
    with pytest.raises(KeyError, match="nope"):
        m.set_memories_rv({"nope": 1.0})
    with pytest.raises(KeyError, match="nope"):
        m.memories_rv = {"nope": 1.0}
    with pytest.raises(KeyError, match="nope"):
        m._memories = {"nope": torch.zeros(3)}


def test_set_reset_value_non_strict_rejects_raw_values():
    """``strict=False`` may store a ``ResetValue`` but never a raw object.

    ``_memories_rv`` is ``dict[str, ResetValue]``; a stray value must be
    rejected rather than silently stored.
    """
    m = TwoState(3)
    with pytest.raises(TypeError, match="ResetValue"):
        m.set_reset_value("new", "not-a-reset-value", strict=False)
    assert "new" not in m.memories_rv
    # A genuine ResetValue is still accepted.
    m.set_reset_value("new", m.memories_rv["v"], strict=False)
    assert m.memories_rv["new"] is m.memories_rv["v"]
