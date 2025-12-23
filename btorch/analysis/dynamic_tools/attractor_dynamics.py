import numpy as np


def calculate_kaplan_yorke_dimension(lyapunov_spectrum: np.ndarray):
    """Calculate the Kaplan-Yorke Dimension (D_KY), also known as the Lyapunov
    Dimension.

    Formula: D_KY = k + sum(lambda_i for i=1 to k) / |lambda_{k+1}|
    where k is the max index such that the sum of the first k exponents is non-negative.

    Args:
        lyapunov_spectrum (np.ndarray): Array of Lyapunov exponents, sorted in descending order.

    Returns:
        float: The Kaplan-Yorke dimension. Returns 0 if the system is stable (all lambda < 0).
               Returns the number of exponents if the sum of all is positive (unbounded/hyperchaos).
    """
    # Ensure sorted descending
    ls = np.sort(lyapunov_spectrum)[::-1]

    n = len(ls)

    # Calculate cumulative sums
    cum_sum = np.cumsum(ls)

    # Find k: max index such that sum >= 0
    # We look for the last index where cum_sum >= 0
    positive_sums = np.where(cum_sum >= 0)[0]

    if len(positive_sums) == 0:
        # All cumulative sums are negative.
        # This usually means the first exponent is negative (stable fixed point).
        return 0.0

    k = positive_sums[-1]

    # Check if k is the last element (sum of all is positive)
    if k == n - 1:
        return float(n)

    # Apply formula
    # Note: indices are 0-based in Python, so k corresponds to the (k+1)-th element in 1-based math notation.
    # The formula uses 1-based k.
    # Let's map carefully:
    # Python index k is the index of the last element included in the sum.
    # So we have summed ls[0]...ls[k].
    # The next element is ls[k+1].
    # The integer part of dimension is (k + 1).

    sum_lambda = cum_sum[k]
    lambda_next = ls[k + 1]

    if lambda_next == 0:
        # Avoid division by zero, though theoretically lambda_{k+1} should be negative here
        return float(k + 1)

    d_ky = (k + 1) + sum_lambda / abs(lambda_next)

    return d_ky


def calculate_structural_eigenvalue_outliers(
    weight_matrix: np.ndarray, spectral_radius: float = None
):
    """Analyze the eigenvalues of the weight matrix to identify structural
    outliers.

    According to the circular law, eigenvalues of a random matrix are distributed within a disk
    of radius R. Outliers outside this radius indicate structural enforcement of specific
    oscillatory modes (stable dynamics) rather than random chaos.

    Args:
        weight_matrix (np.ndarray): The connectivity weight matrix (N x N).
        spectral_radius (float, optional): The theoretical spectral radius of the random component.
                                           If None, it is estimated as std(W) * sqrt(N).

    Returns:
        dict: Dictionary containing:
            - 'eigenvalues': All eigenvalues.
            - 'outliers': Eigenvalues outside the spectral radius.
            - 'outlier_count': Number of outliers.
            - 'spectral_radius': The radius used for thresholding.
    """
    # Ensure numpy array
    W = np.array(weight_matrix)
    N = W.shape[0]

    if W.shape[0] != W.shape[1]:
        raise ValueError("Weight matrix must be square.")

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(W)

    # Determine spectral radius if not provided
    if spectral_radius is None:
        # Estimate radius based on random matrix theory
        # For entries with variance sigma^2/N, radius is sigma.
        # Here we have entries with variance var(W).
        # If W_ij ~ N(0, sigma^2), then radius R = sigma * sqrt(N).
        # std(W) corresponds to sigma.
        sigma = np.std(W)
        spectral_radius = sigma * np.sqrt(N)

    # Identify outliers
    magnitudes = np.abs(eigenvalues)
    outlier_indices = np.where(magnitudes > spectral_radius)[0]
    outliers = eigenvalues[outlier_indices]

    return {
        "eigenvalues": eigenvalues,
        # True Spectral Radius
        "max_eigenvalue": np.max(magnitudes) if len(magnitudes) > 0 else 0.0,
        "outliers": outliers,
        "outlier_count": len(outliers),
        "spectral_radius": spectral_radius,  # Bulk Radius (Threshold)
    }
