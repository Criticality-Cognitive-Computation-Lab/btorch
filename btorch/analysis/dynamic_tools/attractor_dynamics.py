from typing import TypedDict

import numpy as np


class EigenvalueOutliers(TypedDict):
    """Result of :func:`compute_structural_eigenvalue_outliers`."""

    eigenvalues: np.ndarray  # complex, shape (N,)
    max_eigenvalue: float  # largest eigenvalue magnitude |lambda| (0.0 if N == 0)
    outliers: np.ndarray  # complex eigenvalues with |lambda| > spectral_radius
    outlier_count: int
    spectral_radius: float  # bulk radius used as the threshold


def compute_kaplan_yorke_dimension(lyapunov_spectrum: np.ndarray) -> float:
    """Calculate the Kaplan-Yorke Dimension (D_KY), also known as the Lyapunov
    Dimension.

    Formula: D_KY = k + sum(lambda_i for i=1 to k) / |lambda_{k+1}|
    where k is the max index such that the sum of the first k exponents is non-negative.

    Args:
        lyapunov_spectrum (np.ndarray): Array of Lyapunov exponents, sorted in
            descending order.

    Returns:
        float: The Kaplan-Yorke dimension. Returns 0 if the system is stable
            (all lambda < 0). Returns the number of exponents if the sum of all
            is positive (unbounded/hyperchaos). A stable or fully expanding
            spectrum is a valid result, not a failure; no NaN sentinel is
            used (an empty spectrum gives 0.0).
    """
    ls = np.sort(lyapunov_spectrum)[::-1]

    n = len(ls)

    cum_sum = np.cumsum(ls)

    # k: last index whose cumulative sum is still non-negative.
    positive_sums = np.where(cum_sum >= 0)[0]

    if len(positive_sums) == 0:
        # All cumulative sums are negative.
        # This usually means the first exponent is negative (stable fixed point).
        return 0.0

    k = positive_sums[-1]

    if k == n - 1:
        return float(n)

    # 0-based k is the last summed index, so the integer part is k + 1.
    sum_lambda = cum_sum[k]
    lambda_next = ls[k + 1]

    if lambda_next == 0:
        # Guard against division by zero (lambda_{k+1} is normally negative).
        return float(k + 1)

    d_ky = (k + 1) + sum_lambda / abs(lambda_next)

    return d_ky


def compute_structural_eigenvalue_outliers(
    weight_matrix: np.ndarray, spectral_radius: float | None = None
) -> EigenvalueOutliers:
    """Analyze the eigenvalues of the weight matrix to identify structural
    outliers.

    According to the circular law, eigenvalues of a random matrix are distributed
    within a disk of radius R. Outliers outside this radius indicate structural
    enforcement of specific oscillatory modes (stable dynamics) rather than
    random chaos.

    Args:
        weight_matrix (np.ndarray): The connectivity weight matrix (N x N).
        spectral_radius (float, optional): The theoretical spectral radius of the
            random component. If None, it is estimated as std(W) * sqrt(N).

    Returns:
        EigenvalueOutliers: Dictionary containing:
            - 'eigenvalues': All eigenvalues.
            - 'max_eigenvalue': Largest eigenvalue magnitude.
            - 'outliers': Eigenvalues outside the spectral radius.
            - 'outlier_count': Number of outliers.
            - 'spectral_radius': The radius used for thresholding.
            There is no NaN failure return; invalid input raises.

    Raises:
        ValueError: If ``weight_matrix`` is not square.
    """
    W = np.array(weight_matrix)
    N = W.shape[0]

    if W.shape[0] != W.shape[1]:
        raise ValueError("Weight matrix must be square.")

    eigenvalues = np.linalg.eigvals(W)

    if spectral_radius is None:
        # Circular law: W_ij ~ N(0, sigma^2) gives bulk radius sigma * sqrt(N).
        sigma = np.std(W)
        spectral_radius = sigma * np.sqrt(N)

    magnitudes = np.abs(eigenvalues)
    outlier_indices = np.where(magnitudes > spectral_radius)[0]
    outliers = eigenvalues[outlier_indices]

    return {
        "eigenvalues": eigenvalues,
        "max_eigenvalue": np.max(magnitudes) if len(magnitudes) > 0 else 0.0,
        "outliers": outliers,
        "outlier_count": len(outliers),
        "spectral_radius": spectral_radius,  # Bulk Radius (Threshold)
    }
