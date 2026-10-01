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


def test_failure_return_contract_has_stable_keys():
    """Too few avalanches: same keys as a successful fit, NaN/None payloads.

    Before, the early-failure result lacked ``avg_size_by_duration`` and
    ``gamma_stats`` so callers had to special-case it.
    """
    full = compute_avalanche_statistics(
        np.random.default_rng(0).random((5000, 50)) < 0.01
    )
    spikes = np.zeros((50, 3))
    spikes[5, 0] = 1  # a single avalanche
    with pytest.warns(UserWarning, match="Not enough avalanches"):
        failed = compute_avalanche_statistics(spikes)
    assert set(failed) == set(full)
    assert np.isnan(failed["tau"]) and failed["fit_S"] is None
    assert failed["gamma_stats"] is None
    durations, mean_sizes = failed["avg_size_by_duration"]
    assert list(durations) == [1] and list(mean_sizes) == [1.0]

    with pytest.warns(UserWarning, match="Not enough avalanches"):
        empty = compute_avalanche_statistics(np.zeros((50, 3)))
    assert empty["avg_size_by_duration"][0].size == 0
    assert set(empty) == set(full)


def test_fit_helper_failure_shape_is_tuple():
    """Both fit helpers return ``(np.nan, None)`` on failure."""
    from btorch.analysis.dynamic_tools.criticality import (
        _fit_distribution,
        _fit_scaling,
    )

    for out in (_fit_distribution(np.arange(3)), _fit_scaling([1, 2], [1, 2])):
        assert isinstance(out, tuple) and len(out) == 2
        assert np.isnan(out[0]) and out[1] is None
