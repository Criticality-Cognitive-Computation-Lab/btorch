"""Tests for ``btorch.utils.array`` conversion helpers."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from btorch.utils.array import to_numpy, to_numpy_or_sparse


def test_to_numpy_tensor_and_array_like():
    """Tensors are detached to CPU; lists go through ``np.asarray``."""
    t = torch.arange(3.0, requires_grad=True) * 2
    out = to_numpy(t)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(to_numpy([1, 2]), [1, 2])


def test_to_numpy_strict_rejects_array_likes():
    """``strict=True`` accepts only tensors/ndarrays and names the argument."""
    assert to_numpy(np.ones(2), strict=True).shape == (2,)
    assert to_numpy(torch.ones(2), strict=True).shape == (2,)
    with pytest.raises(TypeError, match="`values` must be a numpy array"):
        to_numpy([1, 2], strict=True, name="values")


def test_to_numpy_or_sparse_keeps_sparse_and_scalars():
    """Sparse inputs stay sparse; sparse torch tensors become scipy COO."""
    m = sp.coo_array(np.eye(3))
    assert to_numpy_or_sparse(m) is m

    dense = torch.eye(3)
    sparse_t = dense.to_sparse()
    out = to_numpy_or_sparse(sparse_t)
    assert sp.issparse(out)
    np.testing.assert_array_equal(out.toarray(), dense.numpy())

    assert isinstance(to_numpy_or_sparse(np.float32(1.5)), np.generic)
    assert isinstance(to_numpy_or_sparse([1, 2]), np.ndarray)
    assert isinstance(to_numpy_or_sparse(dense), np.ndarray)
