"""Tests for branching-ratio analysis and simulation utilities.

Paper reference used for test targets:
Wilting, J., & Priesemann, V. (2018). Inferring collective dynamical states
from widely unobserved systems. Nature Communications, 9(1), 2325.
https://doi.org/10.1038/s41467-018-04725-4
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from btorch.analysis.branching import (
    compute_branching_ratio,
    input_handler,
    simulate_binomial_subsampling,
    simulate_branching,
)
from btorch.utils.file import fig_path


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class TestSimulateBranching:
    """Tests for simulate_branching."""

    def test_basic_simulation(self):
        result = simulate_branching(length=100, m=0.9, activity=10, rng=42)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        assert len(result) == 100
        assert np.all(result >= 0)

    def test_reproducibility(self):
        result1 = simulate_branching(length=100, m=0.9, activity=10, rng=42)
        result2 = simulate_branching(length=100, m=0.9, activity=10, rng=42)
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_produce_different_results(self):
        result1 = simulate_branching(length=100, m=0.9, activity=10, rng=42)
        result2 = simulate_branching(length=100, m=0.9, activity=10, rng=43)
        assert not np.array_equal(result1, result2)


class TestSimulateBinomialSubsampling:
    """Tests for simulate_binomial_subsampling."""

    def test_basic_subsampling(self):
        a_t = np.array([10, 20, 30, 40, 50])
        result = simulate_binomial_subsampling(a_t, alpha=0.5, rng=42)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(a_t)
        assert result.dtype == float

    def test_full_sampling(self):
        a_t = np.array([10, 20, 30], dtype=np.int64)
        result = simulate_binomial_subsampling(a_t, alpha=1.0, rng=42)
        np.testing.assert_array_equal(result, a_t.astype(float))

    def test_zero_sampling(self):
        a_t = np.array([10, 20, 30], dtype=np.int64)
        result = simulate_binomial_subsampling(a_t, alpha=0.0, rng=42)
        np.testing.assert_array_equal(result, np.zeros_like(a_t, dtype=float))


class TestInputHandler:
    """Tests for input_handler."""

    def test_1d_numpy_array(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = input_handler(arr)
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], arr)

    def test_list_of_arrays(self):
        arrays = [np.array([1, 2]), np.array([3, 4])]
        result = input_handler(arrays)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_2d_numpy_array(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = input_handler(arr)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            input_handler(123)


class TestBranchingRatioEstimator:
    """Paper-aligned checks for MR branching-ratio estimation."""

    def test_subsampling_bias_naive_vs_mr(self):
        true_m = 0.96
        alpha_grid = np.array([1.0, 0.5, 0.2, 0.05])
        full_counts = simulate_branching(
            length=30000,
            m=true_m,
            activity=100,
            rng=7,
        )

        naive_vals = []
        mr_vals = []
        for idx, alpha in enumerate(alpha_grid):
            observed = simulate_binomial_subsampling(full_counts, alpha, rng=101 + idx)
            fit = compute_branching_ratio(observed, k_max=20)
            naive_vals.append(fit["naive_branching_ratio"])
            mr_vals.append(fit["branching_ratio"])

        naive = np.asarray(naive_vals)
        mr = np.asarray(mr_vals)

        assert naive[-1] < naive[0] - 0.25
        assert np.max(np.abs(mr - true_m)) < 0.12

        figure_dir = fig_path()
        fig, ax = plt.subplots(figsize=(6.0, 3.5))
        ax.plot(alpha_grid, naive, "o-", label="naive $r_1$")
        ax.plot(alpha_grid, mr, "s-", label=r"MR $\hat{m}$")
        ax.axhline(true_m, linestyle="--", color="k", label="true $m$")
        ax.set_xlabel("sampling fraction $\\alpha$")
        ax.set_ylabel("estimated branching ratio")
        ax.set_title("Naive bias vs MR subsampling robustness")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig((figure_dir / "branching_subsampling_bias.png").as_posix())
        plt.close(fig)

    def test_rk_exponential_decay_under_subsampling(self):
        true_m = 0.95
        full_counts = simulate_branching(
            length=30000,
            m=true_m,
            activity=120,
            rng=19,
        )
        observed = simulate_binomial_subsampling(full_counts, alpha=0.1, rng=211)
        fit = compute_branching_ratio(observed, k_max=25)

        k = fit["k"]
        r_k = fit["r_k"]
        valid = np.isfinite(r_k) & (r_k > 0)

        assert np.sum(valid) >= 8

        k_valid = k[valid]
        log_r = np.log(r_k[valid])
        corr = np.corrcoef(k_valid, log_r)[0, 1]
        assert corr < -0.85

        slope, _ = np.polyfit(k_valid, log_r, 1)
        m_from_log_fit = float(np.exp(slope))
        assert abs(m_from_log_fit - fit["branching_ratio"]) < 0.08

        figure_dir = fig_path()
        fig, ax = plt.subplots(figsize=(6.0, 3.5))
        ax.semilogy(k, r_k, "o", label="$r_k$")
        ax.semilogy(k, fit["a_fit"] * fit["branching_ratio"] ** k, "-", label="fit")
        ax.set_xlabel("lag $k$")
        ax.set_ylabel("slope $r_k$")
        ax.set_title("Exponential MR decay under subsampling")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig((figure_dir / "branching_rk_decay.png").as_posix())
        plt.close(fig)

    def test_estimator_api_and_reproducibility(self):
        trials = [
            simulate_branching(length=5000, m=0.92, activity=90, rng=1),
            simulate_branching(length=5000, m=0.92, activity=90, rng=2),
        ]

        direct_1 = compute_branching_ratio(trials, k_max=15)
        direct_2 = compute_branching_ratio(trials, k_max=15)
        assert direct_1["branching_ratio"] == direct_2["branching_ratio"]

        for key in ["branching_ratio", "naive_branching_ratio", "k", "r_k", "stderr"]:
            assert key in direct_1


def test_get_slopes_returns_documented_named_tuple():
    """``compute_lagged_slopes`` is a named tuple (still unpackable as 7
    values).

    A white-noise series has no lagged correlation; lags with too few
    samples are NaN, and the scatter lists only hold the valid lags.
    """
    from btorch.analysis.branching import LaggedSlopes, compute_lagged_slopes

    rng = np.random.default_rng(0)
    counts = rng.poisson(5.0, size=200).astype(float)
    # k = 1..4; lag 4 keeps 196 samples, so force NaN with a large min_points.
    res = compute_lagged_slopes([counts], k_max=5, min_points=198, scatterpoints=True)

    assert isinstance(res, LaggedSlopes)
    k, r_k, stderr, data_length, mean_activity, xs, ys = res
    np.testing.assert_array_equal(k, [1, 2, 3, 4])
    # Lags 1 and 2 have 199/198 samples (valid); 3 and 4 fall below min_points.
    assert np.isfinite(r_k[:2]).all() and np.isnan(r_k[2:]).all()
    assert np.isnan(stderr[2:]).all()
    assert len(xs) == len(ys) == 2  # only valid lags are stored
    assert data_length == 200 - 1  # total samples minus the first valid lag
    assert mean_activity == pytest.approx(counts.mean())

    # No valid lag at all: documented sentinels.
    empty = compute_lagged_slopes([counts], k_max=5, min_points=10_000)
    assert empty.data_length == 0 and np.isnan(empty.mean_activity)
    assert empty.xs == [] and np.isnan(empty.r_k).all()
