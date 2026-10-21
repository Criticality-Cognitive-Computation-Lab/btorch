import numpy as np
import pytest

from tests.utils.conn import build_sparse_mat


def _expected_counts(n_e: int, n_i: int, density: float) -> tuple[int, ...]:
    return (
        int(n_e * n_e * density),
        n_e * int(n_i * density),
        int(n_e * n_i * density),
        int(n_i * n_i * density),
    )


def _assert_unique(matrix) -> None:
    coords = np.column_stack((matrix.row, matrix.col))
    assert len({tuple(coord) for coord in coords}) == matrix.nnz


def test_build_sparse_mat_counts_uniqueness_and_signs():
    n_e, n_i, density = 4, 3, 0.5
    np.random.seed(17)
    full, _, _ = build_sparse_mat(n_e, n_i, density=density)
    np.random.seed(17)
    e_matrix, i_matrix, e_idx, i_idx = build_sparse_mat(
        n_e, n_i, split=True, density=density
    )
    n_ee, n_ie, n_ei, n_ii = _expected_counts(n_e, n_i, density)

    # Each block has the historical floor of its requested density. The row
    # sampler writes unique columns, which makes the full COO representation
    # free of duplicate coordinates even before SciPy coalesces it.
    assert full.nnz == n_ee + n_ie + n_ei + n_ii
    assert e_matrix.nnz == n_ee + n_ei
    assert i_matrix.nnz == n_ie + n_ii
    assert np.count_nonzero((full.row < n_e) & (full.col < n_e)) == n_ee
    assert np.count_nonzero((full.row >= n_e) & (full.col < n_e)) == n_ie
    assert np.count_nonzero((full.row < n_e) & (full.col >= n_e)) == n_ei
    assert np.count_nonzero((full.row >= n_e) & (full.col >= n_e)) == n_ii
    _assert_unique(full)
    _assert_unique(e_matrix)
    _assert_unique(i_matrix)

    # Rows encode presynaptic class: E rows are positive in the full matrix,
    # while I rows are negative. Split matrices expose both classes as positive
    # conductances, so the full matrix is exactly E minus I.
    assert np.all(full.data[full.row < n_e] > 0)
    assert np.all(full.data[full.row >= n_e] < 0)
    assert np.all(e_matrix.data > 0)
    assert np.all(i_matrix.data > 0)
    np.testing.assert_array_equal(e_idx, np.arange(n_e))
    np.testing.assert_array_equal(i_idx, np.arange(n_e, n_e + n_i))
    np.testing.assert_allclose(full.toarray(), e_matrix.toarray() - i_matrix.toarray())


def test_build_sparse_mat_i_to_e_keeps_per_target_count_at_low_density():
    n_e, n_i, density = 4, 3, 0.25
    full, _, _ = build_sparse_mat(n_e, n_i, density=density)

    # The I-to-E block historically samples each of the four E targets with
    # floor(3 * 0.25) == 0 I inputs, rather than flooring the global product.
    assert np.count_nonzero((full.row >= n_e) & (full.col < n_e)) == 0


@pytest.mark.parametrize("n_e, n_i", [(3, 2), (0, 3), (3, 0), (0, 0)])
@pytest.mark.parametrize("density", [0.0, 1.0])
def test_build_sparse_mat_density_extremes_and_empty_populations(
    n_e: int, n_i: int, density: float
):
    e_matrix, i_matrix, e_idx, i_idx = build_sparse_mat(
        n_e, n_i, split=True, density=density
    )
    full = (e_matrix - i_matrix).tocoo()
    expected = sum(_expected_counts(n_e, n_i, density))

    assert full.shape == (n_e + n_i, n_e + n_i)
    assert e_matrix.shape == full.shape
    assert i_matrix.shape == full.shape
    assert full.nnz == expected
    assert (
        e_matrix.nnz
        == _expected_counts(n_e, n_i, density)[0]
        + _expected_counts(n_e, n_i, density)[2]
    )
    assert (
        i_matrix.nnz
        == _expected_counts(n_e, n_i, density)[1]
        + _expected_counts(n_e, n_i, density)[3]
    )
    assert e_idx.shape == (n_e,)
    assert i_idx.shape == (n_i,)
    _assert_unique(full)


def test_build_sparse_mat_i_to_e_uses_selected_mean_fallback(monkeypatch):
    def return_normal_mean(loc=0.0, scale=1.0, size=None):
        # The E-E block has no selected edge in this setup, so the I-E scale
        # must use e_to_e_mean. Returning the requested mean makes that choice
        # directly observable without depending on another random draw.
        if size is None:
            return np.asarray(loc)
        loc = np.asarray(loc)
        if loc.ndim:
            return np.broadcast_to(loc, (size,)).copy()
        return np.full(size, loc)

    monkeypatch.setattr(np.random, "normal", return_normal_mean)
    full, _, _ = build_sparse_mat(
        1,
        4,
        density=0.5,
        i_e_ratio=100,
        e_to_e_mean=4.0e-3,
    )

    # int(1 * 1 * 0.5) == 0, so every selected I-to-E edge uses the documented
    # fallback mean rather than a dense E-E column mean.
    i_to_e = full.data[(full.row >= 1) & (full.col == 0)]
    assert i_to_e.size == 2
    np.testing.assert_allclose(i_to_e, -0.4)


def test_build_sparse_mat_large_low_density_has_no_dense_random_draw(monkeypatch):
    original_normal = np.random.normal

    def reject_dense_draw(*args, **kwargs):
        size = kwargs.get("size")
        if size is None and len(args) >= 3:
            size = args[2]
        if size is not None:
            shape = (size,) if np.isscalar(size) else tuple(size)
            if np.prod(shape, dtype=np.int64) > 50_000:
                raise AssertionError("sparse generator attempted a dense draw")
        return original_normal(*args, **kwargs)

    monkeypatch.setattr(np.random, "normal", reject_dense_draw)
    n_e, n_i, density = 10_000, 2_000, 1.0e-4
    full, e_idx, i_idx = build_sparse_mat(n_e, n_i, density=density)

    # This guard is deliberately large enough to catch the old E-by-E
    # lognormal allocation while keeping the sparse result small and CPU-only.
    assert full.nnz == sum(_expected_counts(n_e, n_i, density))
    assert full.shape == (n_e + n_i, n_e + n_i)
    assert e_idx.size == n_e
    assert i_idx.size == n_i
