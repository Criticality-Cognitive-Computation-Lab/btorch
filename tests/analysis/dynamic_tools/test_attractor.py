import numpy as np
from dynamic_tools.attractor_dynamics import (
    calculate_kaplan_yorke_dimension,
    calculate_structural_eigenvalue_outliers,
)


def test_kaplan_yorke():
    print("\nTesting Kaplan-Yorke Dimension (D_KY)...")

    # Case 1: Lorenz System (Standard Chaos)
    # Typical spectrum: [0.906, 0, -14.572]
    # Sums: 0.906 (>0), 0.906 (>0), -13.666 (<0)
    # k (0-based) = 1 (corresponding to 2 exponents)
    # D_KY = 2 + (0.906 + 0) / |-14.572| = 2 + 0.062 = 2.062
    lorenz_spectrum = np.array([0.906, 0, -14.572])
    d_ky_lorenz = calculate_kaplan_yorke_dimension(lorenz_spectrum)
    print(f"Lorenz System (Expected ~2.06): {d_ky_lorenz:.4f}")

    # Case 2: Hyperchaos (Rossler Hyperchaos)
    # Example spectrum: [0.13, 0.02, 0, -14.0]
    # Sums: 0.13, 0.15, 0.15, -13.85
    # k = 2 (3 exponents)
    # D_KY = 3 + 0.15 / 14.0 = 3.01
    hyper_spectrum = np.array([0.13, 0.02, 0, -14.0])
    d_ky_hyper = calculate_kaplan_yorke_dimension(hyper_spectrum)
    print(f"Hyperchaos (Expected ~3.01): {d_ky_hyper:.4f}")

    # Case 3: Stable Fixed Point
    # Spectrum: [-0.5, -1.0, -2.0]
    # Sums: -0.5 (<0)
    # k doesn't exist (empty) -> 0
    stable_spectrum = np.array([-0.5, -1.0, -2.0])
    d_ky_stable = calculate_kaplan_yorke_dimension(stable_spectrum)
    print(f"Stable System (Expected 0.0): {d_ky_stable:.4f}")

    # Case 4: Limit Cycle
    # Spectrum: [0, -1.0, -2.0]
    # Sums: 0 (>=0), -1.0 (<0)
    # k = 0 (1 exponent)
    # D_KY = 1 + 0 / |-1.0| = 1.0
    limit_cycle_spectrum = np.array([0, -1.0, -2.0])
    d_ky_limit = calculate_kaplan_yorke_dimension(limit_cycle_spectrum)
    print(f"Limit Cycle (Expected 1.0): {d_ky_limit:.4f}")


def test_structural_outliers():
    print("\nTesting Structural Eigenvalue Outliers...")

    N = 200
    g = 1.5  # Spectral radius

    # Case 1: Random Matrix (Circular Law)
    # W_ij ~ N(0, g^2/N)
    # Eigenvalues should be confined within radius g
    print("Case 1: Random Matrix (No Outliers expected)")
    W_random = np.random.normal(0, g / np.sqrt(N), (N, N))

    # We provide the theoretical radius to be strict, or let it estimate
    # Let's let it estimate first
    results_rand = calculate_structural_eigenvalue_outliers(W_random)
    print(f"  Estimated Radius: {results_rand['spectral_radius']:.4f}")
    print(f"  Outlier Count (Estimated): {results_rand['outlier_count']}")

    # Provide theoretical radius
    results_rand_theo = calculate_structural_eigenvalue_outliers(
        W_random, spectral_radius=g
    )
    print(f"  Theoretical Radius: {g:.4f}")
    print(f"  Outlier Count (Theoretical): {results_rand_theo['outlier_count']}")

    # Case 2: Structured Matrix (Random + Outlier)
    # Add a strong structural component
    print("\nCase 2: Structured Matrix (Outliers expected)")
    # Create a rank-1 perturbation: u * v^T
    # This should create an outlier at lambda ~ v^T * u
    # To ensure a large eigenvalue, we align u and v (e.g., u = v)
    u = np.random.randn(N, 1)
    u = u / np.linalg.norm(u)
    v = u  # Symmetric perturbation ensures eigenvalue = strength

    strength = 5.0 * g  # Make it clearly outside
    W_struct = W_random + strength * (u @ v.T)

    results_struct = calculate_structural_eigenvalue_outliers(
        W_struct, spectral_radius=g
    )
    print(f"  Outlier Count: {results_struct['outlier_count']}")
    if results_struct["outlier_count"] > 0:
        max_outlier = np.max(np.abs(results_struct["outliers"]))
        print(f"  Max Outlier Magnitude: {max_outlier:.4f} (Expected ~{strength:.4f})")


if __name__ == "__main__":
    test_kaplan_yorke()
    test_structural_outliers()
