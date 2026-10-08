"""Tests of the dense connection layer.

The sparse connection tests live in ``tests/models/connection/``.
"""

import torch

from btorch.models.constrain import constrain_net
from btorch.models.linear import DenseConn


def test_dense_conn_constraints():
    torch.manual_seed(42)
    in_features = 10
    out_features = 8

    # 1. Test Masking (float)
    density = 0.5
    model = DenseConn(in_features, out_features, mask=density, enforce_dale=False)
    assert model.mask is not None
    assert model.mask.shape == (out_features, in_features)

    constrain_net(model)
    # Check that mask is applied
    assert torch.all(model.weight.data[model.mask == 0] == 0)

    # 2. Test Dale's Law
    model_dale = DenseConn(in_features, out_features, enforce_dale=True)
    initial_weights = model_dale.weight.data.clone()

    # Flip some signs manually
    model_dale.weight.data *= -1
    constrain_net(model_dale)

    # Weights that were flipped should be 0 now (because
    # relu((W * initial_sign)) where W * initial_sign < 0)
    # Actually, if initial_sign was 1, and we flipped to -1, then (-1 * 1).relu() = 0.
    # If initial_sign was -1, and we flipped to 1, then (1 * -1).relu() = 0.
    assert torch.all(model_dale.weight.data == 0)  # All signs were flipped

    # If we set them to half the initial value (correct sign)
    model_dale.weight.data = initial_weights * 0.5
    constrain_net(model_dale)
    torch.testing.assert_close(model_dale.weight.data, initial_weights * 0.5)
