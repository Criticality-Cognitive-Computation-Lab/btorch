import numpy as np
import pytest

from btorch.analysis.dynamic_tools.criticality import (
    compute_avalanche_statistics,
    compute_dfa,
)


# Optional extras: skip this module when the library is not installed.
pytest.importorskip("powerlaw")
pytest.importorskip("nolds")


def test_criticality():
    # 1. Generate random spike train (Poisson).
    # This should NOT exhibit power law (exponential distribution expected).
    n_neurons = 100
    n_steps = 10000
    p_spike = 0.01

    rng = np.random.default_rng(0)
    spike_train = rng.random((n_steps, n_neurons)) < p_spike

    results = compute_avalanche_statistics(spike_train, bin_size=1)

    # Basic shape and positivity checks for avalanche extraction.
    assert results["sizes"].shape == results["durations"].shape
    assert results["sizes"].size > 0
    assert np.all(results["sizes"] > 0)
    assert np.all(results["durations"] > 0)

    # Verify expected keys and finite fit outputs when available.
    assert "tau" in results
    assert "alpha" in results
    assert "gamma" in results
    assert "CCC" in results
    if results["fit_S"] is not None:
        assert results["tau"] > 0
        assert np.isfinite(results["tau"])
    if results["fit_T"] is not None:
        assert results["alpha"] > 0
        assert np.isfinite(results["alpha"])

    if not np.isnan(results.get("gamma", np.nan)):
        assert results["gamma"] > 0
    if not np.isnan(results.get("CCC", np.nan)):
        assert results["CCC"] <= 1.0


@pytest.mark.parametrize(
    ("series_fn", "alpha_min", "alpha_max"),
    [
        # 1. White Noise (Random) -> Expected alpha ~ 0.5
        (lambda rng, n: rng.standard_normal(n), 0.35, 0.65),
        # 2. Brownian Motion (Random Walk) -> Expected alpha ~ 1.5
        # Cumulative sum of white noise.
        (lambda rng, n: np.cumsum(rng.standard_normal(n)), 1.3, 1.7),
    ],
)
def test_dfa(series_fn, alpha_min, alpha_max):
    n_steps = 10000

    rng = np.random.default_rng(0)
    series = series_fn(rng, n_steps)
    # 3. Pink Noise (1/f) -> Expected alpha ~ 1.0
    # Harder to generate simply, but we can verify the other two.
    alpha = compute_dfa(series, bin_size=1)

    assert alpha_min < alpha < alpha_max
