"""``MemoryModule.reset(inplace=True)`` -- reset state without reallocating.

inplace writes into the existing buffers instead of rebinding them to fresh
tensors, so buffer identity/address is preserved (needed to reset state inside a
captured CUDA graph) and zero-init memories stay on-device.
"""

import pytest
import torch

from btorch.models.base import MemoryModule


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
