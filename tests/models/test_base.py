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
