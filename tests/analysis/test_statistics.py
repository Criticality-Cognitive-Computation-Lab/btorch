"""Usage examples for statistical utilities in btorch.analysis.statistics.

These tests demonstrate practical use cases for analyzing neural data,
including descriptive statistics, log-scale histograms, correlation
analysis, and the stat/percentile decorators for metric aggregation.
"""

import numpy as np
import pytest
import torch

from btorch.analysis.statistics import (
    compute_log_hist,
    describe_array,
    use_percentiles,
    use_stats,
)


# =============================================================================
# Example 1: Exploring firing rate distributions with describe_array
# =============================================================================
def test_describe_array_firing_rate_analysis(capsys):
    """Analyze firing rate statistics across a neural population.

    This demonstrates using describe_array to get a quick statistical
    summary of neuron firing rates, useful for initial data exploration.
    """
    rng = np.random.default_rng(42)
    # Simulate firing rates from a log-normal distribution (common in neural data)
    firing_rates = rng.lognormal(mean=-2.0, sigma=1.5, size=100)

    describe_array(firing_rates)

    captured = capsys.readouterr()
    assert "Mean:" in captured.out
    assert "Median:" in captured.out
    assert "Standard Deviation:" in captured.out
    # Verify median is printed (log-normal is skewed, median != mean)
    assert "50th Percentile (Q2/Median):" in captured.out


# =============================================================================
# Example 2: Log-scale histogram for connection weight distributions
# =============================================================================
def test_compute_log_hist_connection_weights():
    """Compute log-spaced histogram for synaptic weight distribution.

    Synaptic weights often span several orders of magnitude. Log-spaced bins
    provide better visualization of the full distribution, especially for
    heavy-tailed data common in neural networks.

    Note: np.histogram with N bin edges returns N-1 bins, so bins=50 creates
    49 histogram bins plus 50 edge points.
    """
    rng = np.random.default_rng(42)
    # Simulate synaptic weights with a power-law-like distribution
    weights = np.exp(rng.normal(loc=-5, scale=2, size=10000))
    # Ensure all positive for log histogram
    weights = np.abs(weights) + 1e-10

    # Midpoint positioning (default) - useful for plotting line graphs
    # bins=50 creates 50 log-spaced edge points, resulting in 49 histogram bins
    hist_mid, edges_mid = compute_log_hist(weights, bins=50, edge_pos="mid")
    assert len(hist_mid) == 49  # N-1 bins for N edges
    assert len(edges_mid) == 49  # Midpoints of each bin
    # Edges should be at bin centers for plotting
    assert np.all(edges_mid > 0)

    # Sep positioning - useful for bar plots showing bin edges
    hist_sep, edges_sep = compute_log_hist(weights, bins=50, edge_pos="sep")
    assert len(hist_sep) == 49  # N-1 bins for N edges
    assert len(edges_sep) == 50  # N edge points

    # Verify histogram captures most of the distribution (may miss values
    # at exact bin boundaries due to right-edge exclusivity in np.histogram)
    assert hist_mid.sum() > 0.99 * len(weights)


# =============================================================================
# Example 4: Using @use_stat decorator for flexible metric aggregation
# =============================================================================
def test_use_stat_decorator_numpy_and_torch():
    """Demonstrate using @use_stat for computing neuron metrics with optional
    aggregation.

    The use_stat decorator adds stat/stat_info parameters to any
    function that returns per-neuron values. This allows flexible
    aggregation without rewriting the core computation logic.
    """

    @use_stats
    def compute_firing_rates(
        spike_trains,
        *,
        stat=None,
        stat_info=None,
        nan_policy="skip",
        inf_policy="propagate",
    ):
        """Compute mean firing rate per neuron across trials.

        Args:
            spike_trains: Array of shape (n_neurons, n_trials, n_time)
            stat: If specified, return aggregated statistic instead of per-neuron values
            stat_info: Additional statistics to compute and store in info dict

        Returns:
            Tuple of (values or aggregated_stat, info_dict)
        """
        # Sum spikes across time, average across trials
        total_spikes = spike_trains.sum(axis=-1)  # (n_neurons, n_trials)
        rates = total_spikes.mean(axis=-1)  # (n_neurons,)
        return rates, {"raw_spike_counts": total_spikes}

    rng = np.random.default_rng(42)
    n_neurons = 50
    n_trials = 20
    n_time = 100

    # Generate synthetic spike data
    spike_data = rng.poisson(0.1, size=(n_neurons, n_trials, n_time))

    # Use case 1: Get per-neuron firing rates
    rates, info = compute_firing_rates(spike_data)
    assert rates.shape == (n_neurons,)
    assert "raw_spike_counts" in info

    # Use case 2: Get population mean firing rate directly
    mean_rate, info = compute_firing_rates(spike_data, stat="mean")
    assert isinstance(mean_rate, float)
    assert np.isclose(mean_rate, rates.mean())
    assert info["values_mean"] == mean_rate

    # Use case 3: Get per-neuron rates but also compute population statistics
    rates, info = compute_firing_rates(spike_data, stat_info=["mean", "std", "max"])
    assert rates.shape == (n_neurons,)
    assert "values_mean" in info
    assert "values_std" in info
    assert "values_max" in info
    assert np.isclose(info["values_mean"], rates.mean())

    # Use case 4: Test with torch tensors (should work seamlessly)
    spike_data_torch = torch.from_numpy(spike_data).float()

    @use_stats
    def compute_rates_torch(spike_trains, *, stat=None, stat_info=None):
        total_spikes = spike_trains.sum(dim=-1)
        rates = total_spikes.mean(dim=-1)
        return rates, {}

    rates_torch, _ = compute_rates_torch(spike_data_torch)
    assert isinstance(rates_torch, torch.Tensor)
    np.testing.assert_allclose(rates_torch.numpy(), rates, rtol=1e-5)

    # Aggregate torch result
    mean_rate_torch, info = compute_rates_torch(spike_data_torch, stat="median")
    assert isinstance(mean_rate_torch, float)


# =============================================================================
# Example 5: NaN handling policies in use_stat decorator
# =============================================================================
def test_use_stat_nan_handling_policies():
    """Demonstrate different NaN handling policies when aggregating metrics.

    Neural data often contains missing values (NaN) due to recording
    artifacts or excluded time windows. The nan_policy parameter
    controls how these are handled.
    """

    @use_stats
    def get_neuron_metrics(_, *, stat=None, nan_policy="skip"):
        # Simulate metrics where some neurons have NaN values
        values = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        return values, {"count": len(values)}

    # Policy: "skip" (default) - ignore NaN values
    result_skip, info = get_neuron_metrics(None, stat="mean", nan_policy="skip")
    expected_mean = np.nanmean([1.0, 2.0, np.nan, 4.0, 5.0])
    assert np.isclose(result_skip, expected_mean)

    # Policy: "warn" - warn but continue (same result as skip)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result_warn, _ = get_neuron_metrics(None, stat="mean", nan_policy="warn")
        assert len(w) == 1
        assert "NaN values found" in str(w[0].message)
    assert np.isclose(result_warn, expected_mean)

    # Policy: "assert" - raise error if NaN present
    with pytest.raises(ValueError, match="NaN values found"):
        get_neuron_metrics(None, stat="mean", nan_policy="assert")


# =============================================================================
# Example 6: Using @use_percentiles for distribution analysis
# =============================================================================
def test_use_percentiles_decorator():
    """Demonstrate using @use_percentiles for computing distribution
    percentiles.

    The use_percentiles decorator adds a percentiles parameter to
    functions returning per-neuron values, allowing on-demand percentile
    computation.
    """

    @use_percentiles
    def compute_response_modulation_index(responses, *, percentiles=None):
        """Compute response modulation index for each neuron.

        Args:
            responses: Array of shape (n_neurons, n_conditions)
            percentiles: If specified, compute percentiles of the modulation indices

        Returns:
            Tuple of (modulation_indices, info_dict)
        """
        # Modulation index: (max - min) / (max + min) across conditions
        max_resp = responses.max(axis=-1)
        min_resp = responses.min(axis=-1)
        modulation = (max_resp - min_resp) / (max_resp + min_resp + 1e-10)
        return modulation, {"max_response": max_resp, "min_response": min_resp}

    rng = np.random.default_rng(42)
    n_neurons = 100
    n_conditions = 8

    # Generate synthetic tuning curve data
    responses = rng.gamma(shape=2.0, scale=1.0, size=(n_neurons, n_conditions))

    # Use case 1: Get raw modulation indices (no percentiles)
    mod_idx, info = compute_response_modulation_index(responses)
    assert mod_idx.shape == (n_neurons,)
    assert "max_response" in info
    assert "values_percentiles" not in info  # Not computed
    assert "values_levels" not in info  # Not computed

    # Use case 2: Get modulation indices and compute median (50th percentile)
    mod_idx, info = compute_response_modulation_index(responses, percentiles=50)
    assert "values_percentiles" in info
    assert "values_levels" in info
    assert info["values_levels"] == (50.0,)
    median_value = info["values_percentiles"][0]
    assert np.isclose(median_value, np.median(mod_idx))

    # Use case 3: Compute multiple percentiles (quartiles)
    mod_idx, info = compute_response_modulation_index(
        responses, percentiles=(25, 50, 75)
    )
    assert info["values_levels"] == (25.0, 50.0, 75.0)
    p25, p50, p75 = info["values_percentiles"]
    assert p25 < p50 < p75
    assert np.isclose(p25, np.percentile(mod_idx, 25))
    assert np.isclose(p50, np.percentile(mod_idx, 50))
    assert np.isclose(p75, np.percentile(mod_idx, 75))


# =============================================================================
# Example 7: Combining @use_stat and @use_percentiles
# =============================================================================
def test_combined_decorators():
    """Demonstrate combining both decorators for comprehensive analysis.

    Decorators can be chained to provide both aggregation and percentile
    computation capabilities to the same function.
    """

    @use_stats
    @use_percentiles
    def analyze_tuning_selectivity(
        tuning_curves,
        *,
        stat=None,
        stat_info=None,
        percentiles=None,
    ):
        """Analyze tuning selectivity with optional aggregation and
        percentiles.

        Returns selectivity index for each neuron.
        """
        # Selectivity: 1 - (min/max) for normalized curves
        max_tune = tuning_curves.max(axis=-1)
        min_tune = tuning_curves.min(axis=-1)
        selectivity = 1 - min_tune / (max_tune + 1e-10)
        return selectivity, {"peak_response": max_tune}

    rng = np.random.default_rng(42)
    n_neurons = 50
    n_stimuli = 12

    # Generate tuning curves (some neurons more selective than others)
    tuning = rng.gamma(shape=2.0, scale=1.0, size=(n_neurons, n_stimuli))
    # Add selective neurons (strong response to one stimulus)
    for i in range(10):
        tuning[i] = 0.1
        tuning[i, i % n_stimuli] = 5.0

    # Get per-neuron selectivity with population percentiles
    selectivity, info = analyze_tuning_selectivity(
        tuning, stat_info="mean", percentiles=(10, 90)
    )
    assert selectivity.shape == (n_neurons,)
    assert "values_mean" in info  # From use_stat
    assert "values_percentiles" in info  # From use_percentiles (inner decorator)
    assert "values_levels" in info  # From use_percentiles (inner decorator)
    assert info["values_levels"] == (10.0, 90.0)

    # Get aggregated mean with percentiles computed on the aggregated value
    # (though percentiles make most sense on per-neuron values)
    mean_selectivity, info = analyze_tuning_selectivity(tuning, stat="mean")
    assert isinstance(mean_selectivity, float)


# =============================================================================
# Example 8: Using dict stat format for multi-value returns
# =============================================================================
def test_use_stat_dict_format_multiple_returns():
    """Demonstrate using dict stat format for functions returning multiple
    values.

    When a function returns multiple per-neuron arrays (e.g., ECI and
    lag), the dict format {position: label} allows aggregating specific
    positions.
    """

    @use_stats
    def compute_eci_and_lag(data, *, stat=None, stat_info=None):
        """Compute ECI and lag for each neuron.

        Returns:
            Tuple of (eci_values, lag_values, info_dict)
        """
        rng = np.random.default_rng(42)
        n_neurons = data.shape[0]
        eci = rng.normal(0.5, 0.2, size=n_neurons)
        lag = rng.exponential(5.0, size=n_neurons)
        return eci, lag, {"n_neurons": n_neurons}

    rng = np.random.default_rng(42)
    data = rng.random((50, 10))

    # Use case 1: Aggregate both ECI (pos 0) and lag (pos 1) with dict stat
    # The dict maps position to stat choice (e.g., "mean", "std")
    eci_mean, lag_mean, info = compute_eci_and_lag(data, stat={0: "mean", 1: "mean"})
    assert isinstance(eci_mean, float)  # Aggregated to mean
    assert isinstance(lag_mean, float)  # Aggregated to mean
    assert "values0_mean" in info  # Position + stat stored
    assert "values1_mean" in info

    # Use case 2: Aggregate only ECI (only position 0 is returned + info)
    eci_mean, info = compute_eci_and_lag(data, stat={0: "mean"})
    assert isinstance(eci_mean, float)
    assert "values0_mean" in info
    # Note: When using dict stat, only specified positions are returned

    # Use case 3: Different stats for different positions
    eci_max, lag_std, info = compute_eci_and_lag(data, stat={0: "max", 1: "std"})
    assert isinstance(eci_max, float)
    assert isinstance(lag_std, float)
    assert "values0_max" in info
    assert "values1_std" in info

    # Use case 3: Use stat_info with dict format for extra stats
    eci, lag, info = compute_eci_and_lag(data, stat_info={0: ["mean", "std"], 1: "max"})
    assert eci.shape == (50,)
    assert lag.shape == (50,)
    # stat_info stores with position prefix
    assert "values0_mean" in info
    assert "values0_std" in info
    assert "values1_max" in info


# =============================================================================
# Example 9: Using dict percentiles format for multi-value returns
# =============================================================================
def test_use_percentiles_dict_format_multiple_returns():
    """Demonstrate using dict percentiles format for functions returning
    multiple values.

    Similar to stat, percentiles can accept a dict mapping positions to
    labels.
    """

    @use_percentiles
    def compute_eci_and_lag(data, *, percentiles=None):
        """Compute ECI and lag for each neuron."""
        rng = np.random.default_rng(42)
        n_neurons = data.shape[0]
        eci = rng.normal(0.5, 0.2, size=n_neurons)
        lag = rng.exponential(5.0, size=n_neurons)
        return eci, lag, {}

    rng = np.random.default_rng(42)
    data = rng.random((50, 10))

    # Compute percentiles using tuple format (applies to position 0 by default)
    eci, lag, info = compute_eci_and_lag(data, percentiles=(25, 75))
    assert eci.shape == (50,)
    assert lag.shape == (50,)
    # Percentiles stored with separate keys for values and levels
    assert "values_percentiles" in info
    assert "values_levels" in info
    assert info["values_levels"] == (25.0, 75.0)
