"""Tests for shape helpers and per-memory override handling in ``base.py``."""

import pytest
import torch

from btorch.models.base import MemoryModule, is_broadcastable


@pytest.mark.parametrize(
    "shape_from, shape_to, expected",
    [
        ((4,), (3, 4), True),  # leading dims are added
        ((1, 4), (3, 4), True),  # size-1 dims expand
        ((3, 4), (3, 4), True),
        ((), (3, 4), True),  # scalars broadcast to anything
        # Mutually broadcastable, but NOT "from -> to": (5,1) would need to
        # become (1,4) which torch.broadcast_to rejects.
        ((5, 1), (1, 4), False),
        ((3, 4), (4,), False),  # source has more dims than target
        ((2, 4), (3, 4), False),  # incompatible
    ],
)
def test_is_broadcastable_is_directional(shape_from, shape_to, expected):
    assert is_broadcastable(shape_from, shape_to) is expected
    # The check must agree with what torch actually does.
    try:
        torch.broadcast_to(torch.empty(shape_from), shape_to)
        torch_ok = True
    except RuntimeError:
        torch_ok = False
    assert torch_ok is expected


class _TwoMem(MemoryModule):
    def __init__(self):
        super().__init__()
        # 'a' forces float64 and persistent; 'b' has no overrides.
        self.register_memory("a", 0.0, 3, dtype=torch.float64, persistent=True)
        self.register_memory("b", 0.0, 3)


def test_init_state_overrides_do_not_leak_between_memories():
    m = _TwoMem()
    m.init_state(dtype=torch.float32)
    assert m.a.dtype == torch.float64
    # 'b' must use the call-level dtype, not 'a's override.
    assert m.b.dtype == torch.float32
    # Only 'a' asked to be persistent.
    sd = m.state_dict()
    assert "a" in sd and "b" not in sd


def test_reset_overrides_do_not_leak_between_memories():
    m = _TwoMem()
    m.init_state(dtype=torch.float32)
    m.reset(dtype=torch.float32)
    assert m.a.dtype == torch.float64
    assert m.b.dtype == torch.float32


def test_reset_detects_batch_per_memory():
    m = _TwoMem()
    m.init_state(batch_size=2)
    m.reset()
    assert m.a.shape == (2, 3) and m.b.shape == (2, 3)
