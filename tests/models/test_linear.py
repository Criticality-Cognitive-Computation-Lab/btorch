"""Tests for the PyTorch-compatible dense linear operator."""

import pytest
import torch
from torch import nn
from torch.nn.utils import prune

from btorch.models.constrain import constrain_net
from btorch.models.linear import Linear
from btorch.sparse.operator import LinearOperator


def test_linear_applies_mask_constraint():
    """A sampled mask has operator shape and zeros masked weights."""
    torch.manual_seed(42)
    linear = Linear(10, 8, mask=0.5)

    assert linear.mask is not None
    assert linear.mask.shape == (8, 10)

    constrain_net(linear)
    assert torch.all(linear.weight[linear.mask == 0] == 0)


def test_linear_is_torch_linear_and_linear_operator():
    """Linear preserves the standard ``[out, in]`` PyTorch convention."""
    weight = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    linear = Linear(2, 3, weight=weight, bias=False)
    x = torch.tensor([[2.0, -1.0]])

    assert isinstance(linear, nn.Linear)
    assert isinstance(linear, LinearOperator)
    torch.testing.assert_close(linear(x), nn.functional.linear(x, weight))
    torch.testing.assert_close(linear.matvec(x), linear(x))

    linear = linear.to(dtype=torch.float64)
    torch.testing.assert_close(
        linear(x.double()),
        nn.functional.linear(x.double(), weight.double()),
    )


def test_linear_tracks_parameter_replacement_and_input_dtype():
    """Replacing ``weight`` updates execution while retaining dtype."""
    initial = torch.ones(3, 2, dtype=torch.float64)
    linear = Linear(2, 3, weight=initial, bias=False)
    replacement = nn.Parameter(torch.full((3, 2), 2.0, dtype=torch.float64))
    linear.weight = replacement

    torch.testing.assert_close(
        linear(torch.ones(1, 2, dtype=torch.float64)),
        torch.full((1, 3), 4.0, dtype=torch.float64),
    )


def test_linear_accepts_tensor_bias():
    """A user-provided bias is copied into the live module parameter."""
    bias = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    linear = Linear(2, 3, bias=bias)

    assert linear.bias is not None
    torch.testing.assert_close(linear.bias, bias)


def test_linear_validates_and_normalizes_tensor_mask():
    """Tensor masks follow the weight shape, dtype, and device."""
    linear = Linear(2, 3, mask=torch.ones(3, 2, dtype=torch.bool))
    assert linear.mask.dtype == linear.weight.dtype
    assert linear.mask.device == linear.weight.device

    with pytest.raises(ValueError, match="mask must have shape"):
        Linear(2, 3, mask=torch.ones(2, 3))


def test_linear_uses_pruned_weight_from_standard_linear_path():
    """PyTorch pruning parametrizations are resolved on every forward call."""
    linear = Linear(3, 2, bias=False)
    prune.random_unstructured(linear, name="weight", amount=0.5)
    x = torch.randn(4, 3)

    torch.testing.assert_close(
        linear(x),
        nn.functional.linear(x, linear.weight),
    )
