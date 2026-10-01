import numpy as np
import pytest

from btorch.analysis.dynamic_tools.attractor_dynamics import (
    compute_kaplan_yorke_dimension,
    compute_structural_eigenvalue_outliers,
)


@pytest.mark.parametrize(
    ("spectrum", "expected"),
    [
        # Case 1: Lorenz System (Standard Chaos)
        # Typical spectrum: [0.906, 0, -14.572]
        # Sums: 0.906 (>0), 0.906 (>0), -13.666 (<0)
        # k (0-based) = 1 (corresponding to 2 exponents)
        # D_KY = 2 + (0.906 + 0) / |-14.572| = 2.062
        (np.array([0.906, 0.0, -14.572]), 2.06216),
        # Case 2: Hyperchaos (Rossler Hyperchaos)
        # Example spectrum: [0.13, 0.02, 0, -14.0]
        # Sums: 0.13, 0.15, 0.15, -13.85
        # k = 2 (3 exponents)
        # D_KY = 3 + 0.15 / 14.0 = 3.01
        (np.array([0.13, 0.02, 0.0, -14.0]), 3.01071),
        # Case 3: Stable Fixed Point
        # Spectrum: [-0.5, -1.0, -2.0]
        # Sums: -0.5 (<0)
        # k doesn't exist (empty) -> 0
        (np.array([-0.5, -1.0, -2.0]), 0.0),
        # Case 4: Limit Cycle
        # Spectrum: [0, -1.0, -2.0]
        # Sums: 0 (>=0), -1.0 (<0)
        # k = 0 (1 exponent)
        # D_KY = 1 + 0 / |-1.0| = 1.0
        (np.array([0.0, -1.0, -2.0]), 1.0),
    ],
)
def test_kaplan_yorke(spectrum, expected):
    d_ky = compute_kaplan_yorke_dimension(spectrum)
    assert d_ky == pytest.approx(expected, rel=1e-3, abs=1e-3)


def test_structural_outliers():
    rng = np.random.default_rng(0)
    n = 200
    g = 1.5  # Spectral radius

    # Case 1: Random Matrix (Circular Law)
    # W_ij ~ N(0, g^2/N)
    # Eigenvalues should be confined within radius g
    w_random = rng.normal(0, g / np.sqrt(n), (n, n))
    results_rand = compute_structural_eigenvalue_outliers(w_random)
    assert results_rand["spectral_radius"] > 0
    assert results_rand["outlier_count"] < n * 0.1

    # Provide theoretical radius to be strict (instead of estimation).
    results_rand_theo = compute_structural_eigenvalue_outliers(
        w_random, spectral_radius=g
    )
    assert results_rand_theo["spectral_radius"] == pytest.approx(g)
    assert results_rand_theo["outlier_count"] < n * 0.1

    # Case 2: Structured Matrix (Random + Outlier)
    # Add a strong structural component (rank-1 perturbation).
    # This should create an outlier at lambda ~ v^T * u.
    # To ensure a large eigenvalue, align u and v (e.g., u = v).
    u = rng.standard_normal((n, 1))
    u = u / np.linalg.norm(u)
    v = u

    strength = 5.0 * g
    w_struct = w_random + strength * (u @ v.T)

    results_struct = compute_structural_eigenvalue_outliers(w_struct, spectral_radius=g)
    assert results_struct["outlier_count"] >= 1
    max_outlier = np.max(np.abs(results_struct["outliers"]))
    assert max_outlier > g * 3.0


def test_structural_eigenvalue_outliers_result_keys_and_threshold():
    """The result documents every key; outliers lie outside the radius.

    ``diag(2, 0.1, 0.1)`` has one eigenvalue (2) outside a given radius of 1.
    """
    result = compute_structural_eigenvalue_outliers(
        np.diag([2.0, 0.1, 0.1]), spectral_radius=1.0
    )
    assert set(result) == {
        "eigenvalues",
        "max_eigenvalue",
        "outliers",
        "outlier_count",
        "spectral_radius",
    }
    assert result["outlier_count"] == 1
    assert result["max_eigenvalue"] == pytest.approx(2.0)
    assert result["spectral_radius"] == 1.0
    np.testing.assert_allclose(np.abs(result["outliers"]), [2.0])
