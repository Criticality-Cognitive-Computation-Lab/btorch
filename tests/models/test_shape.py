"""Tests for ``btorch.models.shape`` dimension-expansion helpers."""

import torch

from btorch.models.shape import (
    expand_dims,
    expand_leading_dims,
    expand_trailing_dims,
)


def test_expand_leading_appends_new_dims_in_front():
    x = torch.arange(3.0)  # shape (3,)
    out = expand_leading_dims(x, (2, 4))
    # New dims are prepended: (2, 4) + (3,)
    assert out.shape == (2, 4, 3)
    assert torch.equal(out[1, 2], x)


def test_expand_trailing_appends_new_dims_at_end():
    x = torch.arange(3.0)
    out = expand_trailing_dims(x, 5)  # int target is treated as (5,)
    assert out.shape == (3, 5)
    assert torch.equal(out[:, 4], x)


def test_match_full_shape_only_adds_missing_dims():
    x = torch.arange(3.0)
    # Full target shape: only the missing leading dim (2) is inserted.
    out = expand_leading_dims(x, (2, 3), match_full_shape=True)
    assert out.shape == (2, 3)
    out = expand_trailing_dims(x, (3, 2), match_full_shape=True)
    assert out.shape == (3, 2)


def test_broadcast_only_returns_singleton_dims():
    x = torch.arange(3.0)
    out = expand_leading_dims(x, (2, 4), broadcast_only=True)
    # Not expanded: only unsqueezed, so it can broadcast lazily.
    assert out.shape == (1, 1, 3)


def test_view_flag_controls_memory_sharing():
    x = torch.arange(3.0)
    shared = expand_dims(x, 2, False, "leading", view=True)
    copied = expand_dims(x, 2, False, "leading", view=False)
    # A view shares storage with the input; a clone does not.
    assert shared.data_ptr() == x.data_ptr()
    assert copied.data_ptr() != x.data_ptr()
    assert torch.equal(shared, copied)
