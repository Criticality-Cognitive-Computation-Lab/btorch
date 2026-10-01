import numpy as np
import pytest

from btorch.analysis.metrics import indices_to_mask, select_on_metric


def test_indices_to_mask_with_shape_and_array():
    mask = indices_to_mask(np.array([0, 2]), shape=(4,))
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask, np.array([True, False, True, False]))

    template = np.zeros((2, 2))
    mask_from_array = indices_to_mask(np.array([1]), array=template)
    np.testing.assert_array_equal(
        mask_from_array, np.array([[False, True], [False, False]])
    )


def test_select_on_metric_topk_and_any_modes():
    metrics = np.array([0.1, 0.9, 0.5])
    top_two = select_on_metric(metrics, num=2, mode="topk")
    assert set(top_two.tolist()) == {1, 2}

    np.random.seed(0)
    any_indices, any_mask = select_on_metric(
        metrics > 0.4, num=1, mode="any", ret_indices=True
    )
    assert any_mask.sum() == 1
    assert any_indices.shape == (1,)


def test_metrics_argument_validation_raises_value_error():
    """Missing required arguments raise ValueError, not AssertionError."""
    with pytest.raises(ValueError, match="shape"):
        indices_to_mask(np.array([0]))
    with pytest.raises(ValueError, match="num"):
        select_on_metric(np.arange(5.0), num=None, mode="topk")
