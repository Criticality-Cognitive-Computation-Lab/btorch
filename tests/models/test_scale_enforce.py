"""Tests for the ``enforce`` modes of :class:`SupportScaleState`.

The scaled/unscaled guards used to be ``assert "<non-empty string>"`` which can
never fail, so ``enforce="assert"`` silently did nothing. These tests pin the
real behavior: ``"assert"`` raises, ``"ignore"`` returns the input unchanged, and
``"repeated"`` goes ahead and applies the transform again.
"""

import pytest
import torch

from btorch.models.scale import SupportScaleState


class _Scalable(SupportScaleState):
    """Minimal module providing what ``SupportScaleState`` reads."""

    def __init__(self):
        super().__init__()
        self.v_threshold = torch.tensor(-50.0)
        self.v_rest = torch.tensor(-70.0)
        self.init_scale_state()


V = torch.tensor([-70.0, -50.0])


def test_scale_func_assert_raises_when_already_scaled():
    m = _Scalable()
    m._scale_state()  # mark as scaled
    with pytest.raises(RuntimeError, match="already scaled"):
        m.scale_func(V, enforce="assert")
    with pytest.raises(RuntimeError, match="already scaled"):
        m.scale_state(enforce="assert")


def test_scale_func_ignore_and_repeated_when_already_scaled():
    m = _Scalable()
    m._scale_state()
    # "ignore" is a no-op that hands back the input so callers can chain it.
    assert m.scale_func(V, enforce="ignore") is V
    # "repeated" applies the transform again: (v - rest) / scale = [0, 1]
    torch.testing.assert_close(
        m.scale_func(V, enforce="repeated"), torch.tensor([0.0, 1.0])
    )


def test_unscale_func_assert_raises_when_not_scaled():
    m = _Scalable()  # starts unscaled
    with pytest.raises(RuntimeError, match="already unscaled"):
        m.unscale_func(V, enforce="assert")
    with pytest.raises(RuntimeError, match="already unscaled"):
        m.unscale_state(enforce="assert")
    assert m.unscale_func(V, enforce="ignore") is V
    # "repeated" unscales regardless of state: v * scale + rest = [-70, -50] -> ...
    torch.testing.assert_close(
        m.unscale_func(V, enforce="repeated"), V * 20.0 + (-70.0)
    )


def test_scale_func_unscaled_module_works_under_assert():
    m = _Scalable()
    torch.testing.assert_close(m.scale_func(V), torch.tensor([0.0, 1.0]))
