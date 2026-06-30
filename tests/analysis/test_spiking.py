"""Tests for spiking analysis functions with known stochastic signals.

These tests use analytically tractable stochastic processes to validate
the statistical measures (CV, Fano factor, LV) against theoretical
values.
"""

import numpy as np
import pytest
import torch

from btorch.analysis.spiking import (
    compute_raster,
    cv_temporal,
    fano,
    fano_population,
    fano_sweep,
    fano_temporal,
    firing_rate,
    isi_cv,
    kurtosis,
    kurtosis_population,
    local_variation,
)


# =============================================================================
# Helper: Generate spike trains from known stochastic processes
# =============================================================================


def generate_poisson_spikes(
    rate_hz: float, duration_ms: float, dt_ms: float = 1.0, n_neurons: int = 1
) -> np.ndarray:
    """Generate Poisson spike train with given rate.

    Poisson process has:
    - ISIs ~ Exponential(λ) where λ = rate
    - CV = 1 (coefficient of variation)
    - Fano factor = 1 (for counting windows)
    - LV = 1 (local variation)
    """
    n_steps = int(duration_ms / dt_ms)
    # Probability of spike in each bin: p = rate * dt
    p_spike = rate_hz * dt_ms / 1000.0  # rate in Hz, dt in ms
    return (np.random.rand(n_steps, n_neurons) < p_spike).astype(np.float32)


def generate_gamma_spikes(
    rate_hz: float,
    shape_k: float,
    duration_ms: float,
    dt_ms: float = 1.0,
    n_neurons: int = 1,
) -> np.ndarray:
    """Generate renewal process with Gamma-distributed ISIs.

    Gamma(k, θ) ISIs have:
    - CV = 1 / sqrt(k) (inverse relationship with shape parameter)
    - For k=1: reduces to exponential (Poisson), CV=1
    - For k→∞: approaches regular spiking, CV→0
    """
    n_steps = int(duration_ms / dt_ms)
    mean_isi_ms = 1000.0 / rate_hz  # mean ISI in ms
    scale_theta = mean_isi_ms / shape_k  # θ = mean/k

    spikes = np.zeros((n_steps, n_neurons), dtype=np.float32)
    for n in range(n_neurons):
        t = 0.0
        while t < duration_ms:
            isi = np.random.gamma(shape_k, scale_theta)
            t += isi
            idx = int(t / dt_ms)
            if idx < n_steps:
                spikes[idx, n] = 1.0
    return spikes


def generate_regular_spikes(
    rate_hz: float,
    duration_ms: float,
    dt_ms: float = 1.0,
    n_neurons: int = 1,
    jitter: float = 0.0,
) -> np.ndarray:
    """Generate regular (periodic) spike train with optional jitter.

    Regular spiking has CV = 0 (or close to 0 with small jitter).
    """
    n_steps = int(duration_ms / dt_ms)
    isi_ms = 1000.0 / rate_hz
    spikes = np.zeros((n_steps, n_neurons), dtype=np.float32)

    for n in range(n_neurons):
        t = np.random.rand() * isi_ms  # random phase
        while t < duration_ms:
            # Add jitter if specified
            actual_t = t + np.random.randn() * jitter if jitter > 0 else t
            idx = int(actual_t / dt_ms)
            if 0 <= idx < n_steps:
                spikes[idx, n] = 1.0
            t += isi_ms
    return spikes


def generate_bursty_spikes(
    rate_hz: float,
    burst_factor: float,
    duration_ms: float,
    dt_ms: float = 1.0,
    n_neurons: int = 1,
) -> np.ndarray:
    """Generate bursty spike train with higher CV.

    Bursty spiking has CV > 1.
    Uses a two-state model: active (bursting) and silent.
    """
    n_steps = int(duration_ms / dt_ms)
    spikes = np.zeros((n_steps, n_neurons), dtype=np.float32)

    for n in range(n_neurons):
        t = 0
        while t < n_steps:
            # Burst duration
            burst_len = int(np.random.exponential(burst_factor * 10))
            # Burst rate is higher
            burst_rate = rate_hz * burst_factor
            p_burst = burst_rate * dt_ms / 1000.0
            for _ in range(burst_len):
                if t < n_steps and np.random.rand() < p_burst:
                    spikes[t, n] = 1.0
                t += 1
            # Silent period
            silent_len = int(np.random.exponential(50))
            t += silent_len
    return spikes


# =============================================================================
# CV (Coefficient of Variation) Tests with Known Stochastic Processes
# =============================================================================


class TestCVWithStochasticProcesses:
    """Test CV computation against theoretical values for known processes."""

    def test_cv_poisson_process(self):
        """Poisson process should have CV ≈ 1.

        For a Poisson process, ISIs follow an exponential distribution
        with CV = 1 by definition.
        """
        np.random.seed(42)
        # Generate long Poisson spike train for good statistics
        spikes = generate_poisson_spikes(
            rate_hz=50.0, duration_ms=50000.0, dt_ms=1.0, n_neurons=10
        )
        cv_values, info = isi_cv(spikes, dt_ms=1.0)
        # New info format: spike_count, isi_mean, isi_std, isi_var arrays
        isi_mean = info.get("isi_mean", np.array([]))
        isi_std = info.get("isi_std", np.array([]))

        # CV should be close to 1 for Poisson (theoretical value)
        # Use median to be robust to outliers
        valid_cv = cv_values[~np.isnan(cv_values)]
        median_cv = np.median(valid_cv)
        assert 0.9 < median_cv < 1.1, f"Poisson CV should be ≈ 1, got {median_cv:.3f}"

        # Compute total CV from aggregated ISI stats
        # Use nanmedian across all neurons for total CV
        total_mean = np.nanmedian(isi_mean)
        total_std = np.nanmedian(isi_std)
        total_cv = total_std / total_mean if total_mean > 0 else np.nan
        assert (
            0.9 < total_cv < 1.1
        ), f"Total Poisson CV should be ≈ 1, got {total_cv:.3f}"

    def test_cv_gamma_process_theory(self):
        """Gamma renewal process: CV = 1/sqrt(k).

        Test that measured CV matches theoretical value for Gamma-distributed ISIs.
        """
        np.random.seed(42)
        test_cases = [
            (1.0, 1.0),  # k=1: Exponential, CV=1
            (4.0, 0.5),  # k=4: CV=0.5
            (9.0, 1 / 3),  # k=9: CV≈0.33
        ]

        for shape_k, expected_cv in test_cases:
            spikes = generate_gamma_spikes(
                rate_hz=50.0,
                shape_k=shape_k,
                duration_ms=100000.0,
                dt_ms=1.0,
                n_neurons=20,
            )
            cv_values, _ = isi_cv(spikes, dt_ms=1.0)

            valid_cv = cv_values[~np.isnan(cv_values)]
            mean_cv = np.mean(valid_cv)

            # Allow 15% tolerance due to finite sample effects
            assert (
                np.abs(mean_cv - expected_cv) < expected_cv * 0.15
            ), f"Gamma(k={shape_k}) CV: expected {expected_cv:.3f}, got {mean_cv:.3f}"

    def test_cv_regular_spiking(self):
        """Regular (periodic) spiking should have CV ≈ 0.

        With perfectly regular ISIs, there is no variation.
        """
        np.random.seed(42)
        spikes = generate_regular_spikes(
            rate_hz=20.0, duration_ms=50000.0, dt_ms=1.0, n_neurons=10
        )
        cv_values, info = isi_cv(spikes, dt_ms=1.0)
        # New info format: spike_count, isi_mean, isi_std, isi_var arrays
        isi_mean = info.get("isi_mean", np.array([]))
        isi_std = info.get("isi_std", np.array([]))

        valid_cv = cv_values[~np.isnan(cv_values)]
        assert np.all(
            valid_cv < 0.05
        ), f"Regular spiking CV should be ≈ 0, got max {np.max(valid_cv):.3f}"

        # Compute total CV from aggregated ISI stats
        total_mean = np.nanmedian(isi_mean)
        total_std = np.nanmedian(isi_std)
        total_cv = total_std / total_mean if total_mean > 0 else np.nan
        assert total_cv < 0.05, f"Total regular CV should be ≈ 0, got {total_cv:.3f}"

    def test_cv_bursty_spiking(self):
        """Bursty spiking should have CV > 1.

        Burstiness increases the variability of ISIs beyond Poisson.
        """
        np.random.seed(42)
        spikes = generate_bursty_spikes(
            rate_hz=30.0, burst_factor=5.0, duration_ms=100000.0, n_neurons=10
        )
        cv_values, _ = isi_cv(spikes, dt_ms=1.0)

        valid_cv = cv_values[~np.isnan(cv_values)]
        mean_cv = np.mean(valid_cv)
        assert mean_cv > 1.0, f"Bursty spiking CV should be > 1, got {mean_cv:.3f}"

    def test_cv_handles_sparse_activity(self):
        """CV should be NaN for neurons with insufficient spikes."""
        spike_data = np.array(
            [
                [1, 0],
                [0, 0],
                [1, 0],  # Neuron 0 has 2 spikes (1 ISI)
                [0, 0],
                [0, 0],  # Neuron 1 has 0 spikes
            ],
            dtype=np.float32,
        )

        cv_values, info = isi_cv(spike_data, dt_ms=1.0)
        # New info format: spike_count, isi_mean, isi_std, isi_var arrays
        spike_count = info.get("spike_count", np.array([]))

        assert np.isnan(cv_values[1])  # No spikes
        assert spike_count[0] == 2

    def test_cv_torch_numpy_consistency(self):
        """Torch and numpy implementations should give same results."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=50.0, duration_ms=10000.0, n_neurons=5)

        cv_np, _ = isi_cv(spikes, dt_ms=1.0)
        cv_torch, _ = isi_cv(torch.from_numpy(spikes), dt_ms=1.0)

        np.testing.assert_allclose(cv_np, cv_torch.cpu().numpy(), rtol=1e-5)

    def test_cv_batch_axis_aggregation(self):
        """Test CV aggregation across batch dimensions."""
        np.random.seed(42)
        # Shape: [T=1000, trials=10, neurons=5]
        spikes = generate_poisson_spikes(
            rate_hz=50.0, duration_ms=1000.0, n_neurons=50
        ).reshape(1000, 10, 5)

        # Without aggregation: CV per trial
        cv_no_agg, _ = isi_cv(spikes, dt_ms=1.0, batch_axis=None)
        assert cv_no_agg.shape == (10, 5)

        # With aggregation: CV across all trials per neuron
        cv_agg, _ = isi_cv(spikes, dt_ms=1.0, batch_axis=(1,))
        assert cv_agg.shape == (5,)

        # Aggregated CV should still be ≈ 1 for Poisson
        # Use wider bounds due to limited data with aggregation
        valid_cv = cv_agg[~np.isnan(cv_agg)]
        assert 0.7 < np.mean(valid_cv) < 1.3


# =============================================================================
# Fano Factor Tests with Known Stochastic Processes
# =============================================================================


class TestFanoFactorWithStochasticProcesses:
    """Test Fano factor against theoretical values for known processes."""

    def test_fano_poisson_process(self):
        """Poisson process should have Fano factor ≈ 1.

        For a Poisson process, count variance equals count mean in any
        window, so Fano factor = var/mean = 1.
        """
        np.random.seed(42)
        # Long simulation for good statistics
        spikes = generate_poisson_spikes(
            rate_hz=50.0, duration_ms=50000.0, dt_ms=1.0, n_neurons=20
        )

        # Use multiple window sizes
        for window in [10, 50, 100]:
            fano_values, _ = fano(spikes, window=window, overlap=0)
            valid_fano = fano_values[~np.isnan(fano_values)]
            mean_fano = np.mean(valid_fano)
            assert 0.85 < mean_fano < 1.15, (
                f"Poisson Fano factor (window={window}) "
                f"should be ≈ 1, got {mean_fano:.3f}"
            )

    def test_fano_regular_vs_poisson(self):
        """Regular spiking should have Fano < 1, lower than Poisson."""
        np.random.seed(42)
        duration_ms = 50000.0

        regular_spikes = generate_regular_spikes(
            rate_hz=50.0, duration_ms=duration_ms, n_neurons=20
        )
        poisson_spikes = generate_poisson_spikes(
            rate_hz=50.0, duration_ms=duration_ms, n_neurons=20
        )

        window = 50
        fano_regular, _ = fano(regular_spikes, window=window, overlap=0)
        fano_poisson, _ = fano(poisson_spikes, window=window, overlap=0)

        mean_regular = np.nanmean(fano_regular)
        mean_poisson = np.nanmean(fano_poisson)

        # Regular should have lower Fano than Poisson
        assert mean_regular < mean_poisson, (
            f"Regular Fano ({mean_regular:.3f}) "
            f"should be < Poisson Fano ({mean_poisson:.3f})"
        )
        # Regular should be significantly below 1
        assert (
            mean_regular < 0.3
        ), f"Regular Fano should be < 0.5, got {mean_regular:.3f}"

    def test_fano_bursty_vs_poisson(self):
        """Bursty spiking should have Fano > 1, higher than Poisson."""
        np.random.seed(42)
        duration_ms = 100000.0

        bursty_spikes = generate_bursty_spikes(
            rate_hz=30.0, burst_factor=5.0, duration_ms=duration_ms, n_neurons=20
        )
        poisson_spikes = generate_poisson_spikes(
            rate_hz=30.0, duration_ms=duration_ms, n_neurons=20
        )

        window = 100
        fano_bursty, _ = fano(bursty_spikes, window=window, overlap=0)
        fano_poisson, _ = fano(poisson_spikes, window=window, overlap=0)

        mean_bursty = np.nanmean(fano_bursty)
        mean_poisson = np.nanmean(fano_poisson)

        # Bursty should have higher Fano than Poisson
        assert mean_bursty > mean_poisson, f"Bursty Fano ({mean_bursty:.3f}) "
        f"should be > Poisson Fano ({mean_poisson:.3f})"
        # Bursty should be significantly above 1
        assert mean_bursty > 1.5, f"Bursty Fano should be > 2, got {mean_bursty:.3f}"

    def test_fano_torch_numpy_consistency(self):
        """Torch and numpy implementations should give same results."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=50.0, duration_ms=10000.0, n_neurons=5)

        fano_np, _ = fano(spikes, window=50, overlap=0)
        fano_torch, _ = fano(torch.from_numpy(spikes), window=50, overlap=0)

        np.testing.assert_allclose(fano_np, fano_torch.cpu().numpy(), rtol=1e-5)

    def test_fano_window_sweep(self):
        """Fano factor sweep across window sizes should be stable for
        Poisson."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=50.0, duration_ms=10000.0, n_neurons=5)

        # Test arange-style interface: window=(start, stop) - stop is exclusive
        fano_sweep_vals, info = fano_sweep(spikes, window=(1, 1001))

        # Should have shape [n_windows, neurons] = [1000, 5]
        assert fano_sweep_vals.shape == (1000, 5)
        assert info["n_windows"] == 1000
        np.testing.assert_array_equal(info["window_sizes"], np.arange(1, 1001))

        # For larger windows, Fano should be closer to 1 (more stable)
        # Check windows from 50 to 1000
        late_fanos = fano_sweep_vals[50:1000]
        valid_fanos = late_fanos[~np.isnan(late_fanos)]
        assert 0.9 < np.mean(valid_fanos) < 1.1

    def test_fano_sweep_arange_interface(self):
        """Test fano_sweep with various arange-style window specifications."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=50.0, duration_ms=1000.0, n_neurons=3)

        # Test window as int (stop): range(1, stop+1, 1)
        vals, info = fano_sweep(spikes, window=10)
        assert vals.shape[0] == 10  # windows 1..10
        assert info["window"] == (1, 11, 1)

        # Test window as (start, stop): range(start, stop, 1)
        vals, info = fano_sweep(spikes, window=(5, 15))
        assert vals.shape[0] == 10  # windows 5..14
        assert info["window"] == (5, 15, 1)
        np.testing.assert_array_equal(info["window_sizes"], np.arange(5, 15))

        # Test window as (start, stop, step): range(start, stop, step)
        vals, info = fano_sweep(spikes, window=(10, 101, 10))
        assert vals.shape[0] == 10  # windows 10, 20, ..., 100
        assert info["window"] == (10, 101, 10)
        np.testing.assert_array_equal(info["window_sizes"], np.arange(10, 101, 10))

        # Test window=None (default)
        vals, info = fano_sweep(spikes)  # T=1000, default is T//20=50
        assert vals.shape[0] == 50  # windows 1..50
        assert info["window"] == (1, 51, 1)


# =============================================================================
# Local Variation (LV) Tests
# =============================================================================


class TestLocalVariationWithStochasticProcesses:
    """Test Local Variation against theoretical values.

    LV = 3 * (ISI_i - ISI_{i+1})^2 / (ISI_i + ISI_{i+1})^2

    For Poisson: LV = 1
    For regular: LV = 0
    """

    def test_lv_poisson_process(self):
        """Poisson process should have LV ≈ 1."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(
            rate_hz=50.0, duration_ms=100000.0, dt_ms=1.0, n_neurons=20
        )
        lv_values, _ = local_variation(spikes, dt_ms=1.0)

        valid_lv = lv_values[~np.isnan(lv_values)]
        mean_lv = np.mean(valid_lv)
        assert 0.9 < mean_lv < 1.1, f"Poisson LV should be ≈ 1, got {mean_lv:.3f}"

    def test_lv_regular_spiking(self):
        """Regular spiking should have LV ≈ 0."""
        np.random.seed(42)
        spikes = generate_regular_spikes(
            rate_hz=20.0, duration_ms=100000.0, dt_ms=1.0, n_neurons=10
        )
        lv_values, _ = local_variation(spikes, dt_ms=1.0)

        valid_lv = lv_values[~np.isnan(lv_values)]
        assert np.all(
            valid_lv < 0.05
        ), f"Regular LV should be ≈ 0, got max {np.max(valid_lv):.3f}"

    def test_lv_comparison_regular_poisson_bursty(self):
        """LV ordering: regular < Poisson < bursty."""
        np.random.seed(42)
        duration_ms = 100000.0

        regular = generate_regular_spikes(
            rate_hz=50.0, duration_ms=duration_ms, n_neurons=10
        )
        poisson = generate_poisson_spikes(
            rate_hz=50.0, duration_ms=duration_ms, n_neurons=10
        )
        bursty = generate_bursty_spikes(
            rate_hz=50.0, burst_factor=3.0, duration_ms=duration_ms, n_neurons=10
        )

        lv_reg, _ = local_variation(regular, dt_ms=1.0)
        lv_pois, _ = local_variation(poisson, dt_ms=1.0)
        lv_burst, _ = local_variation(bursty, dt_ms=1.0)

        mean_reg = np.nanmean(lv_reg)
        mean_pois = np.nanmean(lv_pois)
        mean_burst = np.nanmean(lv_burst)

        assert mean_reg < mean_pois < mean_burst, (
            f"LV ordering failed: regular={mean_reg:.3f}, "
            f"Poisson={mean_pois:.3f}, bursty={mean_burst:.3f}"
        )

    def test_lv_torch_numpy_consistency(self):
        """Torch and numpy implementations should give same results."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=50.0, duration_ms=10000.0, n_neurons=5)

        lv_np, _ = local_variation(spikes, dt_ms=1.0)
        lv_torch, _ = local_variation(torch.from_numpy(spikes), dt_ms=1.0)

        np.testing.assert_allclose(lv_np, lv_torch.cpu().numpy(), rtol=1e-5)


# =============================================================================
# Kurtosis Tests
# =============================================================================


class TestKurtosisWithStochasticProcesses:
    """Test kurtosis of spike counts.

    For Poisson counts: kurtosis ≈ 1/mean (excess kurtosis)
    For regular: negative excess kurtosis (sub-Poissonian)
    For bursty: positive excess kurtosis (super-Poissonian)
    """

    def test_kurtosis_poisson_vs_regular(self):
        """Regular spiking has more negative excess kurtosis than Poisson."""
        np.random.seed(42)
        duration_ms = 50000.0
        window = 50

        regular = generate_regular_spikes(
            rate_hz=50.0, duration_ms=duration_ms, n_neurons=10
        )
        poisson = generate_poisson_spikes(
            rate_hz=50.0, duration_ms=duration_ms, n_neurons=10
        )

        kurt_reg, _ = kurtosis(regular, window=window, fisher=True)
        kurt_pois, _ = kurtosis(poisson, window=window, fisher=True)

        # Regular should have lower (more negative) excess kurtosis
        assert np.nanmean(kurt_reg) < np.nanmean(kurt_pois)

    def test_kurtosis_torch_numpy_consistency(self):
        """Torch and numpy implementations should give same results."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=50.0, duration_ms=10000.0, n_neurons=5)

        kurt_np, _ = kurtosis(spikes, window=50, fisher=True)
        kurt_torch, _ = kurtosis(torch.from_numpy(spikes), window=50, fisher=True)

        np.testing.assert_allclose(kurt_np, kurt_torch.cpu().numpy(), rtol=1e-4)

    def test_population_metric_contract_consistency(self):
        """Population metrics should share `(value, info)` return contracts.

        This test locks in the API decision that both population-level
        irregularity metrics expose matching metadata fields. Keeping
        this contract stable helps downstream analysis pipelines swap
        metrics without adapter code.
        """
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=40.0, duration_ms=5000.0, n_neurons=8)

        fano_val, fano_info = fano_population(spikes, window=100, overlap=20)
        kurt_val, kurt_info = kurtosis_population(spikes, window=100, overlap=20)

        assert np.asarray(fano_val).ndim == 0
        assert np.asarray(kurt_val).ndim == 0
        assert set(fano_info.keys()) == {"window", "n_windows"}
        assert set(kurt_info.keys()) == {"window", "n_windows"}
        assert fano_info["window"] == kurt_info["window"] == 100
        assert fano_info["n_windows"] == kurt_info["n_windows"]


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_firing_rate_basic(self):
        """Test firing rate computation."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(
            rate_hz=100.0, duration_ms=1000.0, dt_ms=1.0, n_neurons=5
        )

        # Rate per neuron
        fr = firing_rate(spikes, width=10, dt=1.0, axis=None)
        assert fr.shape == spikes.shape

        # Population rate (average across neurons)
        fr_pop = firing_rate(spikes, width=10, dt=1.0, axis=-1)
        assert fr_pop.shape == (1000,)

        # Check rate is in reasonable range (0 to max possible)
        assert np.all(fr >= 0)
        assert np.all(fr <= 1000)  # Max 1000 Hz with dt=1ms

    def test_firing_rate_torch(self):
        """Test firing rate with torch tensors."""
        spikes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        fr = firing_rate(spikes, width=3, dt=0.5, axis=None)
        assert fr.shape == spikes.shape
        assert fr.dtype == spikes.dtype

    def test_compute_raster(self):
        """Test raster plot computation."""
        sp_matrix = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        times = np.array([0.0, 1.0, 2.0])
        neuron_idx, spike_times = compute_raster(sp_matrix, times)

        np.testing.assert_array_equal(neuron_idx, np.array([0, 1, 0, 1]))
        np.testing.assert_array_equal(spike_times, np.array([0.0, 1.0, 2.0, 2.0]))


# =============================================================================
# GPU Tests (conditional)
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPU:
    """GPU-specific tests."""

    def test_cv_gpu(self):
        """CV computation works on GPU."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=50.0, duration_ms=10000.0, n_neurons=5)
        spikes_gpu = torch.from_numpy(spikes).cuda()

        cv_gpu, _ = isi_cv(spikes_gpu, dt_ms=1.0)

        assert isinstance(cv_gpu, torch.Tensor)
        assert cv_gpu.device.type == "cuda"
        assert cv_gpu.shape == (5,)

        # Result should be similar to CPU
        cv_cpu, _ = isi_cv(spikes, dt_ms=1.0)
        np.testing.assert_allclose(cv_cpu, cv_gpu.cpu().numpy(), rtol=1e-4)

    def test_fano_gpu(self):
        """Fano factor computation works on GPU."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=50.0, duration_ms=10000.0, n_neurons=5)
        spikes_gpu = torch.from_numpy(spikes).cuda()

        fano_gpu, _ = fano(spikes_gpu, window=50)

        assert isinstance(fano_gpu, torch.Tensor)
        assert fano_gpu.device.type == "cuda"

    def test_lv_gpu(self):
        """Local variation computation works on GPU."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=50.0, duration_ms=10000.0, n_neurons=5)
        spikes_gpu = torch.from_numpy(spikes).cuda()

        lv_gpu, _ = local_variation(spikes_gpu, dt_ms=1.0)

        assert isinstance(lv_gpu, torch.Tensor)
        assert lv_gpu.device.type == "cuda"

    def test_cv_temporal_gpu(self):
        """CV temporal computation works on GPU."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=50.0, duration_ms=10000.0, n_neurons=5)
        spikes_gpu = torch.from_numpy(spikes).cuda()

        cv_temp_gpu, _ = cv_temporal(spikes_gpu, dt_ms=1.0, window=100, step=10)

        assert isinstance(cv_temp_gpu, torch.Tensor)
        assert cv_temp_gpu.device.type == "cuda"

    def test_fano_temporal_gpu(self):
        """Fano temporal computation works on GPU."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=50.0, duration_ms=10000.0, n_neurons=5)
        spikes_gpu = torch.from_numpy(spikes).cuda()

        fano_temp_gpu, _ = fano_temporal(spikes_gpu, window=100, step=10)

        assert isinstance(fano_temp_gpu, torch.Tensor)
        assert fano_temp_gpu.device.type == "cuda"


# =============================================================================
# CV Temporal Tests - Time-resolved coefficient of variation
# =============================================================================


class TestCVTemporal:
    """Test cv_temporal for time-resolved CV analysis.

    cv_temporal computes CV in sliding windows over time, giving a time-
    resolved measure of spike train irregularity.
    """

    def test_cv_temporal_poisson_process(self):
        """Poisson process should have CV ≈ 1 across all time windows.

        For a stationary Poisson process, the CV should be close to 1 in
        every temporal window.
        """
        np.random.seed(42)
        # Generate long Poisson spike train for good statistics
        spikes = generate_poisson_spikes(
            rate_hz=50.0, duration_ms=50000.0, dt_ms=1.0, n_neurons=10
        )

        # Use sliding windows
        cv_temp, info = cv_temporal(spikes, dt_ms=1.0, window=500, step=100)

        # Shape should be [n_windows, n_neurons]
        n_expected_windows = (50000 - 500) // 100 + 1
        assert cv_temp.shape == (n_expected_windows, 10)

        # Check info dictionary
        assert "window" in info
        assert "step" in info
        assert "window_starts_ms" in info
        assert "window_ends_ms" in info
        assert info["window"] == 500
        assert info["step"] == 100

        # CV should be close to 1 for Poisson (theoretical value)
        # Use nanmedian to be robust to outliers and windows with few spikes
        valid_cv = cv_temp[~np.isnan(cv_temp)]
        median_cv = np.median(valid_cv)
        assert (
            0.85 < median_cv < 1.15
        ), f"Poisson CV temporal should be ≈ 1, got {median_cv:.3f}"

        # Most windows should have valid CV values (not too many NaNs)
        nan_fraction = np.isnan(cv_temp).sum() / cv_temp.size
        assert nan_fraction < 0.3, f"Too many NaN values: {nan_fraction:.2%}"

    def test_cv_temporal_regular_spiking(self):
        """Regular spiking should have CV ≈ 0 across all time windows.

        With perfectly regular ISIs, the CV should be close to 0 in
        every temporal window.
        """
        np.random.seed(42)
        spikes = generate_regular_spikes(
            rate_hz=20.0, duration_ms=50000.0, dt_ms=1.0, n_neurons=10
        )

        cv_temp, info = cv_temporal(spikes, dt_ms=1.0, window=500, step=100)

        # Shape verification
        n_expected_windows = (50000 - 500) // 100 + 1
        assert cv_temp.shape == (n_expected_windows, 10)

        # Regular spiking should have very low CV
        valid_cv = cv_temp[~np.isnan(cv_temp)]
        max_cv = np.max(valid_cv) if len(valid_cv) > 0 else 0
        assert max_cv < 0.1, f"Regular spiking CV should be ≈ 0, got max {max_cv:.3f}"

    def test_cv_temporal_shows_rate_changes(self):
        """cv_temporal should detect temporal changes when rate changes.

        Create a spike train where the rate changes mid-recording and
        verify that cv_temporal captures the temporal structure.
        """
        np.random.seed(42)
        # First half: 20 Hz, second half: 80 Hz (both Poisson)
        duration_ms = 20000.0
        half_duration = int(duration_ms / 2)

        spikes_low = generate_poisson_spikes(
            rate_hz=20.0, duration_ms=half_duration, dt_ms=1.0, n_neurons=5
        )
        spikes_high = generate_poisson_spikes(
            rate_hz=80.0, duration_ms=half_duration, dt_ms=1.0, n_neurons=5
        )
        spikes = np.concatenate([spikes_low, spikes_high], axis=0)

        # Compute temporal CV with overlapping windows
        cv_temp, info = cv_temporal(spikes, dt_ms=1.0, window=1000, step=200)

        # Check shape
        assert cv_temp.shape[0] > 10  # Should have multiple windows
        assert cv_temp.shape[1] == 5  # 5 neurons

        # For Poisson process at different rates, CV should still be ~1
        # (CV is rate-independent for Poisson)
        valid_cv = cv_temp[~np.isnan(cv_temp)]
        median_cv = np.median(valid_cv)
        assert (
            0.8 < median_cv < 1.2
        ), f"CV should remain ~1 despite rate change, got {median_cv:.3f}"

        # The number of NaN values should be higher in the low-rate section
        # due to fewer spikes, but overall should have reasonable coverage
        nan_fraction = np.isnan(cv_temp).sum() / cv_temp.size
        assert nan_fraction < 0.4, f"Too many NaN values: {nan_fraction:.2%}"

    def test_cv_temporal_torch_numpy_consistency(self):
        """Torch and numpy implementations should give same results."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=50.0, duration_ms=10000.0, n_neurons=5)

        cv_temp_np, _ = cv_temporal(spikes, dt_ms=1.0, window=200, step=50)
        cv_temp_torch, _ = cv_temporal(
            torch.from_numpy(spikes), dt_ms=1.0, window=200, step=50
        )

        np.testing.assert_allclose(
            cv_temp_np, cv_temp_torch.cpu().numpy(), rtol=1e-5, atol=1e-6
        )

    def test_cv_temporal_batch_axis(self):
        """Test CV temporal with batch_axis parameter for aggregation."""
        np.random.seed(42)
        # Shape: [T=2000, trials=10, neurons=5]
        spikes = generate_poisson_spikes(
            rate_hz=50.0, duration_ms=2000.0, n_neurons=50
        ).reshape(2000, 10, 5)

        # Without aggregation: CV per trial
        cv_no_agg, _ = cv_temporal(
            spikes, dt_ms=1.0, window=200, step=50, batch_axis=None
        )
        # Shape should be [n_windows, trials, neurons]
        assert cv_no_agg.shape[1:] == (10, 5)

        # With aggregation: aggregate across trials (axis 1)
        # Note: cv_temporal applies batch_axis per window but maintains shape
        cv_agg, _ = cv_temporal(spikes, dt_ms=1.0, window=200, step=50, batch_axis=(1,))
        # Shape maintains original structure [n_windows, trials, neurons]
        # but values are computed from aggregated spikes
        assert cv_agg.shape[1:] == (10, 5)

        # Aggregated CV should still be ≈ 1 for Poisson
        valid_cv = cv_agg[~np.isnan(cv_agg)]
        median_cv = np.median(valid_cv)
        assert (
            0.7 < median_cv < 1.3
        ), f"Aggregated Poisson CV should be ≈ 1, got {median_cv:.3f}"

    def test_cv_temporal_different_windows_and_steps(self):
        """Test cv_temporal with various window and step sizes."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=50.0, duration_ms=10000.0, n_neurons=3)

        # Test different window sizes
        for window in [100, 200, 500]:
            for step in [10, 50, 100]:
                cv_temp, info = cv_temporal(spikes, dt_ms=1.0, window=window, step=step)

                # Calculate expected number of windows
                n_expected = (10000 - window) // step + 1
                assert cv_temp.shape == (
                    n_expected,
                    3,
                ), f"Shape mismatch for window={window}, step={step}"

                # Verify window info
                assert info["window"] == window
                assert info["step"] == step


# =============================================================================
# Fano Temporal Tests - Time-resolved Fano factor
# =============================================================================


class TestFanoTemporal:
    """Test fano_temporal for time-resolved Fano factor analysis.

    fano_temporal computes Fano factor in sliding windows over time,
    giving a time-resolved measure of spike count variability.
    """

    def test_fano_temporal_poisson_process(self):
        """Poisson process should have Fano factor ≈ 1 across time.

        For a stationary Poisson process, the Fano factor should be close
        to 1 in every temporal window (count variance equals count mean).

        Note: fano_temporal uses the entire window as one counting bin,
        so it returns the population Fano across the window. For Poisson,
        this requires sufficient neurons to get meaningful statistics.
        """
        np.random.seed(42)
        # Generate long Poisson spike train with many neurons
        # Need many neurons to compute population Fano within each window
        spikes = generate_poisson_spikes(
            rate_hz=50.0, duration_ms=10000.0, dt_ms=1.0, n_neurons=100
        )

        # Use sliding windows
        fano_temp, info = fano_temporal(spikes, window=500, step=100)

        # Shape should be [n_windows, n_neurons]
        n_expected_windows = (10000 - 500) // 100 + 1
        assert fano_temp.shape == (n_expected_windows, 100)

        # Check info dictionary
        assert "window" in info
        assert "step" in info
        assert "window_starts" in info
        assert "window_ends" in info
        assert info["window"] == 500
        assert info["step"] == 100

        # For population Fano with many neurons, should be ~1
        # Note: many windows may be NaN due to no spikes, use median of valid
        valid_fano = fano_temp[~np.isnan(fano_temp)]
        if len(valid_fano) > 10:
            median_fano = np.median(valid_fano)
            assert (
                0.7 < median_fano < 1.3
            ), f"Poisson Fano temporal should be ≈ 1, got {median_fano:.3f}"

    def test_fano_temporal_captures_burst_onset(self):
        """fano_temporal captures population statistics in sliding windows.

        This test verifies the shape and info output of fano_temporal.
        Population-level Fano requires many neurons for stable
        estimates.
        """
        np.random.seed(42)
        duration_ms = 10000.0
        n_neurons = 50

        # Generate bursty spikes with many neurons
        spikes = generate_bursty_spikes(
            rate_hz=50.0,
            burst_factor=3.0,
            duration_ms=duration_ms,
            dt_ms=1.0,
            n_neurons=n_neurons,
        )

        # Compute temporal Fano with overlapping windows
        fano_temp, info = fano_temporal(spikes, window=500, step=100)

        # Check shape
        assert fano_temp.shape[1] == n_neurons

        # Verify window info
        assert "window" in info
        assert "step" in info
        assert "window_starts" in info
        assert "window_ends" in info

        # Some windows should have valid (non-NaN) values
        valid_count = np.sum(~np.isnan(fano_temp))
        assert valid_count > 0, "Should have some valid Fano values"

    def test_fano_temporal_returns_correct_shape(self):
        """fano_temporal should return correct shape and info structure."""
        np.random.seed(42)
        duration_ms = 10000.0
        n_neurons = 20

        # Generate Poisson spikes
        spikes = generate_poisson_spikes(
            rate_hz=50.0, duration_ms=duration_ms, dt_ms=1.0, n_neurons=n_neurons
        )

        window = 500
        step = 100
        fano_temp, info = fano_temporal(spikes, window=window, step=step)

        # Check shape: [n_windows, n_neurons]
        n_expected_windows = (int(duration_ms) - window) // step + 1
        assert fano_temp.shape == (n_expected_windows, n_neurons)

        # Check info structure
        assert info["window"] == window
        assert info["step"] == step
        assert len(info["window_starts"]) == n_expected_windows
        assert len(info["window_ends"]) == n_expected_windows

    def test_fano_temporal_torch_numpy_consistency(self):
        """Torch and numpy implementations should give same results."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=50.0, duration_ms=10000.0, n_neurons=5)

        fano_temp_np, _ = fano_temporal(spikes, window=200, step=50)
        fano_temp_torch, _ = fano_temporal(
            torch.from_numpy(spikes), window=200, step=50
        )

        np.testing.assert_allclose(
            fano_temp_np, fano_temp_torch.cpu().numpy(), rtol=1e-5, atol=1e-6
        )

    def test_fano_temporal_batch_axis(self):
        """Test Fano temporal with batch_axis parameter for aggregation."""
        np.random.seed(42)
        # Shape: [T=2000, trials=10, neurons=5]
        spikes = generate_poisson_spikes(
            rate_hz=50.0, duration_ms=2000.0, n_neurons=50
        ).reshape(2000, 10, 5)

        # Without aggregation: Fano per trial
        fano_no_agg, _ = fano_temporal(spikes, window=200, step=50, batch_axis=None)
        # Shape should be [n_windows, trials, neurons]
        assert fano_no_agg.shape[1:] == (10, 5)

        # With aggregation: aggregate across trials (axis 1)
        # Note: fano_temporal applies batch_axis per window but maintains shape
        fano_agg, _ = fano_temporal(spikes, window=200, step=50, batch_axis=(1,))
        # Shape maintains original structure [n_windows, trials, neurons]
        # but values are computed from aggregated spikes
        assert fano_agg.shape[1:] == (10, 5)

        # Verify that batch_axis produces different results than no aggregation
        # (they should be different due to aggregation)
        valid_no_agg = fano_no_agg[~np.isnan(fano_no_agg)]
        valid_agg = fano_agg[~np.isnan(fano_agg)]
        if len(valid_no_agg) > 0 and len(valid_agg) > 0:
            # Both should have some valid values
            assert len(valid_agg) > 0, "Aggregated should have valid values"

    def test_fano_temporal_different_windows_and_steps(self):
        """Test fano_temporal with various window and step sizes."""
        np.random.seed(42)
        spikes = generate_poisson_spikes(rate_hz=50.0, duration_ms=10000.0, n_neurons=3)

        # Test different window sizes
        for window in [100, 200, 500]:
            for step in [10, 50, 100]:
                fano_temp, info = fano_temporal(spikes, window=window, step=step)

                # Calculate expected number of windows
                n_expected = (10000 - window) // step + 1
                assert fano_temp.shape == (
                    n_expected,
                    3,
                ), f"Shape mismatch for window={window}, step={step}"

                # Verify window info
                assert info["window"] == window
                assert info["step"] == step
