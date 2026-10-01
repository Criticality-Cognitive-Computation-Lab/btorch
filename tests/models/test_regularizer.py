"""Tests for the loss modules in ``btorch.models.regularizer``."""

import pytest
import torch

from btorch.models.regularizer import (
    FiringRateLoss,
    QuantileDistributionLoss,
    VoltageRegularizer,
)


def test_voltage_regularizer_zero_inside_range():
    reg = VoltageRegularizer(v_threshold=1.0, v_reset=0.0)
    # Normalised voltage = (v - 1) / 1 lies in [-1, 1] for v in [0, 2],
    # so nothing is penalised.
    v = torch.linspace(0.0, 2.0, 12).reshape(3, 4)
    assert reg(v).item() == 0.0


def test_voltage_regularizer_penalises_outliers():
    reg = VoltageRegularizer(v_threshold=1.0, v_reset=0.0, voltage_cost=1.0)
    # v = 4 -> normalised 3 -> relu(3 - 1)^2 = 4; one neuron, one sample.
    assert reg(torch.tensor([[4.0]])).item() == pytest.approx(4.0)
    # Symmetric penalty below the range: v = -2 -> normalised -3 -> 4.
    assert reg(torch.tensor([[-2.0]])).item() == pytest.approx(4.0)


@pytest.mark.parametrize("loss_type", ["pinball", "huber_pinball"])
def test_quantile_loss_zero_for_identical_distributions(loss_type):
    loss_fn = QuantileDistributionLoss(loss_type)
    x = torch.rand(2, 16)
    assert loss_fn(x, x).item() == pytest.approx(0.0, abs=1e-7)


def test_quantile_loss_is_order_invariant_and_reductions():
    target = torch.rand(3, 8)
    pred = torch.rand(3, 8)
    # Inputs are sorted internally, so permuting the last dim changes nothing.
    perm = torch.randperm(8)
    base = QuantileDistributionLoss("pinball", reduction="none")(pred, target)
    shuffled = QuantileDistributionLoss("pinball", reduction="none")(
        pred[:, perm], target[:, perm]
    )
    assert base.shape == (3,)
    assert torch.allclose(base, shuffled)
    # 'mean' and 'sum' reduce the per-batch losses.
    mean = QuantileDistributionLoss("pinball", reduction="mean")(pred, target)
    total = QuantileDistributionLoss("pinball", reduction="sum")(pred, target)
    assert torch.allclose(mean, base.mean())
    assert torch.allclose(total, base.sum())


def test_quantile_loss_rejects_bad_arguments():
    # User-facing argument validation raises ValueError (not AssertionError,
    # which would vanish under ``python -O``).
    with pytest.raises(ValueError, match="loss_type"):
        QuantileDistributionLoss("mse")
    with pytest.raises(ValueError, match="reduction"):
        QuantileDistributionLoss(reduction="median")


def test_quantile_loss_rejects_shape_mismatch():
    loss = QuantileDistributionLoss("pinball")
    with pytest.raises(ValueError, match="same shape"):
        loss(torch.zeros(2, 3), torch.zeros(2, 4))


def test_firing_rate_loss_matches_target_distribution():
    target = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    loss_fn = FiringRateLoss(
        target, input_type="firing_rate", loss_type="huber_pinball"
    )
    assert loss_fn.n_neuron == 5
    # Same rates (in any order) -> zero loss; different rates -> positive.
    assert loss_fn(target.flip(0)).item() == pytest.approx(0.0, abs=1e-7)
    assert loss_fn(target + 0.2).item() > 0.0


def test_firing_rate_loss_rejects_wrong_population_size():
    loss_fn = FiringRateLoss(torch.linspace(0.1, 0.9, 5), input_type="firing_rate")
    with pytest.raises(ValueError, match="n_neuron"):
        loss_fn(torch.zeros(3))


def test_firing_rate_loss_default_loss_type_is_valid():
    # Regression: the default ``loss_type`` used to be 'huber', which
    # QuantileDistributionLoss rejects, so default construction crashed.
    target = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    loss_fn = FiringRateLoss(target, input_type="firing_rate")
    assert loss_fn.loss.loss_type == "huber_pinball"
    assert loss_fn(target + 0.2).item() > 0.0
