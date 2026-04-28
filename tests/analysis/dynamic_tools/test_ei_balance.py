"""Tests for E/I balance analysis using signals with known properties.

These tests use analytically tractable signals to validate:
- ECI (Excitatory-Inhibitory Cancellation Index)
- Lag correlation between E and I currents
- Full E/I balance metrics
"""

import numpy as np
import pytest
import torch

from btorch.analysis.dynamic_tools.ei_balance import (
    compute_eci,
    compute_ei_balance,
    compute_lag_correlation,
)


# =============================================================================
# Helper: Generate test signals with known properties
# =============================================================================


def generate_zero_signal(
    duration_ms: float = 1000.0, dt_ms: float = 1.0, n_neurons: int = 10
):
    """Generate zero signals for E and I currents."""
    t = np.arange(0, duration_ms, dt_ms)
    T = len(t)
    I_e = np.zeros((T, n_neurons))
    I_i = np.zeros((T, n_neurons))
    return I_e.astype(np.float16), I_i.astype(np.float16)


def generate_almost_perfect_balance_signal(
    duration_ms: float = 1000.0,
    dt_ms: float = 1.0,
    n_neurons: int = 10,
    freq_hz: float = 5.0,
    amp_e: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate nearly balanced E/I currents (I_e + I_i ≈ 0).

    I_i ≈ -I_e, so ECI should be close to 0 (good cancellation).
    """
    t = np.arange(0, duration_ms, dt_ms)
    T = len(t)

    # Generate base excitatory signal (offset sine to ensure positive)
    base_signal = np.sin(2 * np.pi * freq_hz * t / 1000.0)
    # Shift to ensure strictly positive values
    I_e = (
        amp_e
        * (base_signal - base_signal.min() + 0.1).reshape(-1, 1)
        * np.ones((1, n_neurons))
    )
    I_e += 0.05 * np.random.rand(T, n_neurons)  # Add small positive noise only

    # Near-perfect inhibition: I_i ≈ -I_e
    I_i = -I_e
    I_i += 0.05 * np.random.rand(T, n_neurons)

    return I_e.astype(np.float32), I_i.astype(np.float32)


def generate_half_normal_uncorrelated(
    duration_ms: float = 1000.0,
    dt_ms: float = 1.0,
    n_neurons: int = 10,
    sigma_e: float = 1.0,
    sigma_i: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate uncorrelated half-normal E/I currents.

    I_e = |X|, I_i = -|Y| where X, Y ~ N(0, 1).
    Expected ECI ≈ 0.41 for sigma_e = sigma_i (from E[||X|-|Y||]/E[|X|+|Y|]).

    Returns:
        I_e: Positive excitatory currents [T, N]
        I_i: Negative inhibitory currents [T, N]
    """
    T = int(duration_ms / dt_ms)

    I_e = np.abs(sigma_e * np.random.randn(T, n_neurons)).astype(np.float32)
    I_i = -np.abs(sigma_i * np.random.randn(T, n_neurons)).astype(np.float32)

    return I_e, I_i


def generate_signed_gaussian_uncorrelated(
    duration_ms: float = 1000.0,
    dt_ms: float = 1.0,
    n_neurons: int = 10,
    sigma_e: float = 1.0,
    sigma_i: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate uncorrelated signed Gaussian E/I currents.

    I_e = X, I_i = -Y where X, Y ~ N(0, sigma^2).
    Expected ECI = 1/√2 ≈ 0.707 for sigma_e = sigma_i (theoretical).

    Returns:
        I_e: Signed excitatory currents [T, N]
        I_i: Signed inhibitory currents [T, N]
    """
    T = int(duration_ms / dt_ms)

    I_e = (sigma_e * np.random.randn(T, n_neurons)).astype(np.float32)
    I_i = -(sigma_i * np.random.randn(T, n_neurons)).astype(np.float32)

    return I_e, I_i


def generate_phase_shifted_sinusoids(
    duration_ms: float = 1000.0,
    dt_ms: float = 1.0,
    n_neurons: int = 10,
    freq_hz: float = 5.0,
    phase_shift_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate sinusoidal E and I with known phase shift.

    Useful for testing lag correlation. The correlation should peak at
    the known phase shift.
    """
    t = np.arange(0, duration_ms, dt_ms)

    # Excitatory signal
    I_e = np.sin(2 * np.pi * freq_hz * t / 1000.0).reshape(-1, 1) * np.ones(
        (1, n_neurons)
    )

    # Inhibitory with phase shift
    phase_rad = 2 * np.pi * freq_hz * phase_shift_ms / 1000.0
    I_i = -np.sin(2 * np.pi * freq_hz * t / 1000.0 - phase_rad).reshape(
        -1, 1
    ) * np.ones((1, n_neurons))

    return I_e.astype(np.float32), I_i.astype(np.float32)


# =============================================================================
# ECI (Excitatory-Inhibitory Cancellation Index) Tests
# =============================================================================


class TestECIWithKnownSignals:
    """Test ECI against theoretically expected values."""

    def test_eci_perfect_cancellation(self):
        """Perfect E/I cancellation should give ECI ≈ 0.

        When I_i = -I_e, the recurrent current I_rec = I_e + I_i = 0, so
        ECI = |I_rec| / (|I_e| + |I_i|) = 0.
        """
        np.random.seed(42)
        I_e, I_i = generate_almost_perfect_balance_signal(
            duration_ms=5000.0, n_neurons=20
        )

        eci, info = compute_eci(I_e, I_i)

        assert eci.shape == (20,)
        np.testing.assert_allclose(eci, 0.0, atol=1e-1)

    def test_eci_no_inhibition(self):
        """No inhibition should give ECI ≈ 1.

        When I_i = 0, the imbalance |I_e + 0| = |I_e| and denominator =
        |I_e| + 0 = |I_e|, so ECI ≈ 1.
        """
        np.random.seed(42)
        T, N = 1000, 10
        # Ensure positive values for I_e
        I_e = np.abs(np.random.randn(T, N)).astype(np.float32)
        I_i = np.zeros((T, N), dtype=np.float32)

        eci, info = compute_eci(I_e, I_i)

        # ECI should be close to 1
        np.testing.assert_allclose(eci, 1.0, atol=0.1)

    def test_eci_no_excitation(self):
        """No excitation should give ECI ≈ 1.

        When I_e = 0, similar to no inhibition case.
        """
        np.random.seed(42)
        T, N = 1000, 10
        I_e = np.zeros((T, N), dtype=np.float32)
        # Ensure negative values for I_i (inhibitory)
        I_i = -np.abs(np.random.randn(T, N)).astype(np.float32)

        eci, info = compute_eci(I_e, I_i)

        np.testing.assert_allclose(eci, 1.0, atol=0.1)

    def test_eci_all_zeros_with_batch_axis(self):
        """All-zero inputs with batch_axis should preserve non-aggregated
        dims."""
        T, B, N = 100, 5, 10  # time, batch, neurons
        I_e = np.zeros((T, B, N), dtype=np.float32)
        I_i = np.zeros((T, B, N), dtype=np.float32)

        # batch_axis=1 means aggregate over time (0) and batch (1)
        # Output should preserve only neurons: shape (N,)
        eci, info = compute_eci(I_e, I_i, batch_axis=1)
        assert eci.shape == (N,)
        np.testing.assert_allclose(eci, 1.0)

        # Test with tuple batch_axis
        eci, info = compute_eci(I_e, I_i, batch_axis=(1,))
        assert eci.shape == (N,)
        np.testing.assert_allclose(eci, 1.0)

    def test_eci_all_zeros_torch_with_batch_axis(self):
        """All-zero torch inputs with batch_axis should preserve non-aggregated
        dims."""
        T, B, N = 100, 5, 10
        I_e = torch.zeros(T, B, N, dtype=torch.float32)
        I_i = torch.zeros(T, B, N, dtype=torch.float32)

        # batch_axis=1 means aggregate over time (0) and batch (1)
        eci, info = compute_eci(I_e, I_i, batch_axis=1)
        assert eci.shape == (N,)
        assert isinstance(eci, torch.Tensor)
        np.testing.assert_allclose(eci.numpy(), 1.0)

    def test_eci_half_normal_uncorrelated(self):
        """Half-normal uncorrelated E/I gives ECI ≈ 0.41.

        For I_e = |X|, I_i = -|Y| with X, Y ~ N(0, 1):
        ECI = E[||X| - |Y||] / E[|X| + |Y|] ≈ 0.41

        This is < 0.5 because both |X| and |Y| are always positive,
        making their difference typically smaller than their sum.
        """
        np.random.seed(42)
        I_e, I_i = generate_half_normal_uncorrelated(
            duration_ms=10000, n_neurons=50, sigma_e=1.0, sigma_i=1.0
        )

        eci, info = compute_eci(I_e, I_i)

        # Expected ECI ≈ 0.41 for half-normal with equal variance
        expected_eci = 0.414
        np.testing.assert_allclose(
            eci.mean(),
            expected_eci,
            atol=0.05,
            err_msg=(
                f"Half-normal ECI should be ≈ {expected_eci}, got {eci.mean():.3f}"
            ),
        )
        # All neurons should give consistent results
        eci_min, eci_max = eci.min(), eci.max()
        assert np.all((eci > 0.35) & (eci < 0.48)), (
            f"ECI range for half-normal should be (0.35, 0.48), "
            f"got [{eci_min:.3f}, {eci_max:.3f}]"
        )

    def test_eci_signed_gaussian_uncorrelated(self):
        """Signed Gaussian uncorrelated E/I gives ECI = 1/√2 ≈ 0.707.

        For I_e = X, I_i = -Y with X, Y ~ N(0, sigma^2):
        ECI = E[|X - Y|] / E[|X| + |Y|] = 1/√2 ≈ 0.707

        This is > 0.5 because X and Y can have opposite signs,
        making |X - Y| relatively large compared to |X| + |Y|.
        """
        np.random.seed(42)
        I_e, I_i = generate_signed_gaussian_uncorrelated(
            duration_ms=10000, n_neurons=50, sigma_e=1.0, sigma_i=1.0
        )

        eci, info = compute_eci(I_e, I_i)

        # Theoretical ECI = 1/√2 for signed Gaussians
        theoretical_eci = 1.0 / np.sqrt(2)
        eci_mean = eci.mean()
        np.testing.assert_allclose(
            eci_mean,
            theoretical_eci,
            atol=0.02,
            err_msg=(
                f"Signed Gaussian ECI should be ≈ {theoretical_eci:.3f}, "
                f"got {eci_mean:.3f}"
            ),
        )
        # All neurons should give consistent results
        eci_min, eci_max = eci.min(), eci.max()
        assert np.all((eci > 0.65) & (eci < 0.75)), (
            f"ECI range for signed Gaussian should be (0.65, 0.75), "
            f"got [{eci_min:.3f}, {eci_max:.3f}]"
        )

    def test_eci_different_variances(self):
        """ECI varies with variance ratio between E and I.

        For unequal variances, ECI deviates from the equal-variance
        case.
        """
        np.random.seed(42)
        T, N = 5000, 20

        # Test with different sigma ratios using half-normal
        test_cases = [
            (0.5, 1.0, 0.49),  # sigma_e < sigma_i -> ECI < 0.5
            (1.0, 1.0, 0.41),  # sigma_e = sigma_i -> ECI ≈ 0.41
            (2.0, 1.0, 0.49),  # sigma_e > sigma_i -> ECI < 0.5 (symmetric)
            (3.0, 1.0, 0.58),  # sigma_e >> sigma_i -> ECI > 0.5
        ]

        for sigma_e, sigma_i, expected_eci in test_cases:
            I_e = np.abs(sigma_e * np.random.randn(T, N)).astype(np.float32)
            I_i = -np.abs(sigma_i * np.random.randn(T, N)).astype(np.float32)

            eci, _ = compute_eci(I_e, I_i)
            eci_mean = eci.mean()
            np.testing.assert_allclose(
                eci_mean,
                expected_eci,
                atol=0.08,
                err_msg=(
                    f"sigma_e={sigma_e}, sigma_i={sigma_i}: "
                    f"ECI should be ≈ {expected_eci}"
                ),
            )

    def test_eci_with_external_current(self):
        """External current should increase ECI.

        I_ext adds to the imbalance without contributing to the
        denominator.
        """
        np.random.seed(42)
        I_e, I_i = generate_almost_perfect_balance_signal(
            duration_ms=5000.0, n_neurons=10
        )

        # Without external: ECI ≈ 0
        eci_no_ext, info_no_ext = compute_eci(I_e, I_i)
        np.testing.assert_allclose(eci_no_ext, 0.0, atol=1e-1)

        # With external: ECI > 0
        I_ext = np.ones_like(I_e) * 0.5
        eci_with_ext, info_with_ext = compute_eci(I_e, I_i, I_ext=I_ext)

        assert np.all(eci_with_ext > eci_no_ext)
        assert np.all(eci_with_ext > 0.1)  # Should be significantly increased

    def test_eci_external_current_split(self):
        """I_ext positive part adds to I_e, negative part adds to I_i.

        This tests that I_ext is properly split: clamp(I_ext, min=0) is added
        to excitatory current, clamp(I_ext, max=0) is added to inhibitory.
        """
        np.random.seed(42)
        T, N = 1000, 10
        # Start with balanced E/I
        I_e = np.ones((T, N), dtype=np.float32) * 1.0
        I_i = np.ones((T, N), dtype=np.float32) * -1.0

        # Pure positive external current should behave like adding to I_e
        I_ext_pos = np.ones((T, N), dtype=np.float32) * 0.5
        eci_with_pos, _ = compute_eci(I_e, I_i, I_ext=I_ext_pos)
        eci_equiv, _ = compute_eci(I_e + 0.5, I_i)
        np.testing.assert_allclose(eci_with_pos, eci_equiv, rtol=1e-5)

        # Pure negative external current should behave like adding to I_i
        I_ext_neg = np.ones((T, N), dtype=np.float32) * -0.5
        eci_with_neg, _ = compute_eci(I_e, I_i, I_ext=I_ext_neg)
        eci_equiv, _ = compute_eci(I_e, I_i - 0.5)
        np.testing.assert_allclose(eci_with_neg, eci_equiv, rtol=1e-5)

        # Mixed I_ext (both pos and neg values)
        I_ext_mixed = np.concatenate(
            [np.ones((T // 2, N)) * 0.3, np.ones((T // 2, N)) * -0.7], axis=0
        ).astype(np.float32)
        eci_with_mixed, _ = compute_eci(I_e, I_i, I_ext=I_ext_mixed)
        # Equivalent: add positive part to I_e, negative part to I_i
        I_ext_pos_part = np.clip(I_ext_mixed, 0, None)
        I_ext_neg_part = np.clip(I_ext_mixed, None, 0)
        eci_equiv_mixed, _ = compute_eci(I_e + I_ext_pos_part, I_i + I_ext_neg_part)
        np.testing.assert_allclose(eci_with_mixed, eci_equiv_mixed, rtol=1e-5)

    def test_eci_torch_numpy_consistency(self):
        """Torch and numpy implementations should give same results."""
        np.random.seed(42)
        I_e, I_i = generate_almost_perfect_balance_signal(
            duration_ms=1000.0, n_neurons=10
        )

        eci_np, info_np = compute_eci(I_e, I_i)
        eci_torch, info_torch = compute_eci(
            torch.from_numpy(I_e), torch.from_numpy(I_i)
        )

        np.testing.assert_allclose(eci_np, eci_torch, rtol=1e-5)

    def test_eci_batch_axis(self):
        """Test ECI aggregation across batch dimensions."""
        np.random.seed(42)
        # Shape: [T=1000, trials=5, neurons=10]
        I_e = np.abs(np.random.randn(1000, 5, 10)).astype(np.float32)
        # Ensure I_i stays negative (inhibitory)
        I_i = -I_e - 0.1 * np.abs(np.random.randn(1000, 5, 10)).astype(
            np.float32
        )  # Nearly balanced, strictly negative

        # Aggregate over trials
        eci, info = compute_eci(I_e, I_i, batch_axis=(1,))

        assert eci.shape == (10,)

    def test_eci_torch_fp16(self):
        """ECI should handle float16 input correctly."""
        np.random.seed(42)
        I_e = torch.abs(torch.randn(1000, 10, dtype=torch.float16))
        I_i = -torch.abs(torch.randn(1000, 10, dtype=torch.float16))

        eci, info = compute_eci(I_e, I_i, dtype=torch.float32)

        # Should return float32
        assert eci.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestECIGPU:
    """GPU-specific ECI tests."""

    def test_eci_gpu(self):
        """ECI computation works on GPU."""
        np.random.seed(42)
        I_e, I_i = generate_almost_perfect_balance_signal(
            duration_ms=1000.0, n_neurons=10
        )

        I_e_gpu = torch.from_numpy(I_e).cuda()
        I_i_gpu = torch.from_numpy(I_i).cuda()

        eci_gpu, info_gpu = compute_eci(I_e_gpu, I_i_gpu)

        assert isinstance(eci_gpu, torch.Tensor)
        assert eci_gpu.device.type == "cuda"

        # Result should match CPU
        eci_cpu, info_cpu = compute_eci(I_e, I_i)
        np.testing.assert_allclose(eci_cpu, eci_gpu.cpu().numpy(), rtol=1e-4)


# =============================================================================
# Lag Correlation Tests
# =============================================================================


class TestLagCorrelationWithKnownSignals:
    """Test lag correlation with signals having known phase relationships."""

    def test_lag_corr_perfect_correlation_zero_lag(self):
        """Identical signals should have correlation ≈ 1 at lag 0."""
        np.random.seed(42)
        T, N = 500, 10
        t = np.linspace(0, 1, T)
        x = np.sin(2 * np.pi * 5 * t).reshape(-1, 1) * np.ones((1, N))
        y = x.copy()  # Identical signal

        peak_corr, best_lag_ms, info = compute_lag_correlation(
            x, y, dt=1.0, max_lag_ms=20.0
        )

        assert peak_corr.shape == (N,)
        np.testing.assert_allclose(peak_corr, 1.0, atol=1e-5)
        np.testing.assert_allclose(best_lag_ms, 0.0, atol=1.0)

    def test_lag_corr_known_shift(self):
        """Test that lag correlation correctly identifies shifted signals."""
        np.random.seed(42)
        T, N = 1000, 10
        dt_ms = 1.0
        shift_ms = 15.0  # Known lag

        # Create Gaussian pulse train (non-periodic enough for unambiguous lag)
        t = np.arange(T) * dt_ms
        pulse_times = np.arange(50, T * dt_ms, 100)  # Pulses every 100ms

        x = np.zeros((T, N), dtype=np.float32)
        for pt in pulse_times:
            x += np.exp(-0.5 * ((t - pt) / 10.0) ** 2).reshape(-1, 1)

        # Shift y by known amount
        shift_bins = int(shift_ms / dt_ms)
        y = np.roll(x, shift_bins, axis=0)

        peak_corr, best_lag_ms, info = compute_lag_correlation(
            x, y, dt=dt_ms, max_lag_ms=50.0
        )

        # Best lag should be close to the known shift
        best_lag = best_lag_ms
        assert np.all(
            np.abs(np.abs(best_lag) - shift_ms) < 3.0
        ), f"Expected |lag| ≈ {shift_ms}, got {best_lag.mean():.1f}"
        # Correlation should be high
        assert np.all(peak_corr > 0.8), f"Correlation too low: {peak_corr.mean():.3f}"

    def test_lag_corr_anticorrelation(self):
        """Anti-correlated signals (y = -x) should have correlation ≈ -1."""
        np.random.seed(42)
        T, N = 500, 10
        # Use sinusoidal signals for stable anti-correlation
        t = np.linspace(0, 1, T)
        x = np.sin(2 * np.pi * 5 * t).reshape(-1, 1) * np.ones((1, N))
        y = -x  # Perfect anti-correlation

        peak_corr, best_lag_ms, info = compute_lag_correlation(
            x, y, dt=1.0, max_lag_ms=10.0
        )

        # Peak correlation should be high magnitude (negative)
        assert np.all(
            np.abs(peak_corr) > 0.75
        ), f"Anti-correlation too weak: {peak_corr.mean():.3f}"
        assert np.all(
            peak_corr < -0.5
        ), f"Expected negative correlation, got {peak_corr.mean():.3f}"

    def test_lag_corr_uncorrelated(self):
        """Uncorrelated signals should have low correlation."""
        np.random.seed(42)
        T, N = 1000, 20
        x = np.random.randn(T, N).astype(np.float32)
        y = np.random.randn(T, N).astype(np.float32)

        peak_corr, best_lag_ms, info = compute_lag_correlation(
            x, y, dt=1.0, max_lag_ms=20.0
        )

        # Should have low correlation (less than 0.3 for independent signals)
        corr_mean = np.mean(np.abs(peak_corr))
        assert (
            corr_mean < 0.3
        ), f"Uncorrelated signals should have |corr| < 0.3, got {corr_mean:.3f}"

    def test_lag_corr_torch_numpy_consistency(self):
        """Torch and numpy implementations should give same results."""
        np.random.seed(42)
        T, N = 1000, 10
        x = np.random.randn(T, N).astype(np.float32)
        y = np.roll(x, 5, axis=0) + 0.01 * np.random.randn(T, N).astype(np.float32)

        peak_np, lag_np, info_np = compute_lag_correlation(
            x, y, dt=1.0, max_lag_ms=20.0
        )

        peak_torch, lag_torch, info_torch = compute_lag_correlation(
            torch.from_numpy(x), torch.from_numpy(y), dt=1.0, max_lag_ms=20.0
        )

        np.testing.assert_allclose(peak_np, peak_torch.numpy(), rtol=2e-3)
        np.testing.assert_allclose(lag_np, lag_torch.numpy(), atol=1e-1)

    def test_lag_corr_fft_vs_direct(self):
        """FFT and direct correlation methods should give similar results."""
        np.random.seed(42)
        T, N = 500, 10
        x = np.random.randn(T, N).astype(np.float32)
        y = np.roll(x, 5, axis=0) + 0.1 * np.random.randn(T, N).astype(np.float32)

        peak_fft, lag_fft, info_fft = compute_lag_correlation(
            x, y, dt=1.0, max_lag_ms=20.0, use_fft=True
        )
        peak_direct, lag_direct, info_direct = compute_lag_correlation(
            x, y, dt=1.0, max_lag_ms=20.0, use_fft=False
        )

        np.testing.assert_allclose(peak_fft, peak_direct, rtol=5e-2)
        np.testing.assert_allclose(lag_fft, lag_direct, atol=2.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLagCorrelationGPU:
    """GPU-specific lag correlation tests."""

    def test_lag_corr_gpu(self):
        """Lag correlation works on GPU."""
        np.random.seed(42)
        T, N = 1000, 10
        x = np.random.randn(T, N).astype(np.float32)
        y = np.roll(x, 5, axis=0) + 0.01 * np.random.randn(T, N).astype(np.float32)

        x_gpu = torch.from_numpy(x).cuda()
        y_gpu = torch.from_numpy(y).cuda()

        peak_gpu, lag_gpu, info_gpu = compute_lag_correlation(
            x_gpu, y_gpu, dt=1.0, max_lag_ms=20.0
        )

        assert isinstance(peak_gpu, torch.Tensor)
        assert peak_gpu.device.type == "cuda"

        # Should match CPU
        peak_cpu, lag_cpu, info_cpu = compute_lag_correlation(
            x, y, dt=1.0, max_lag_ms=20.0
        )
        np.testing.assert_allclose(peak_cpu, peak_gpu.cpu().numpy(), rtol=2e-3)


# =============================================================================
# Full E/I Balance Tests
# =============================================================================


class TestEIBalanceFull:
    """Test full E/I balance computation with known signals."""

    def test_ei_balance_perfect_balance(self):
        """Perfectly balanced E/I should have low ECI and high tracking
        correlation."""
        np.random.seed(42)
        I_e, I_i = generate_almost_perfect_balance_signal(
            duration_ms=5000.0, n_neurons=20
        )

        # Returns 3 values + info due to dict value_key
        eci, peak_corr, best_lag_ms, info = compute_ei_balance(
            I_e, I_i, dt=1.0, max_lag_ms=20.0
        )

        # Check shapes - per-neuron values returned
        assert eci.shape == (20,)
        assert peak_corr.shape == (20,)
        assert best_lag_ms.shape == (20,)

        # ECI should be near 0 (mean across neurons)
        assert eci.mean() < 0.1, f"ECI should be ≈ 0, got {eci.mean():.3f}"

        # Tracking correlation should be high (I tracks E well)
        assert (
            peak_corr.mean() > 0.9
        ), f"Tracking corr should be high, got {peak_corr.mean():.3f}"

        # Info should contain stat_info results
        assert "eci_mean" in info
        assert "peak_corr_mean" in info
        assert "best_lag_ms_mean" in info

    def test_ei_balance_half_normal_uncorrelated(self):
        """Half-normal uncorrelated E/I should have ECI ≈ 0.41.

        Note: ECI ≈ 0.41 (< 0.5) is the mathematically correct value for
        half-normal signals with equal variance, not a sign of correlation.
        """
        np.random.seed(42)
        I_e, I_i = generate_half_normal_uncorrelated(
            duration_ms=5000.0, n_neurons=20, sigma_e=1.0, sigma_i=1.0
        )

        eci, peak_corr, best_lag_ms, info = compute_ei_balance(
            I_e, I_i, dt=1.0, max_lag_ms=20.0
        )

        # ECI should be ≈ 0.41 for half-normal with equal variance
        expected_eci = 0.414
        np.testing.assert_allclose(
            eci.mean(),
            expected_eci,
            atol=0.05,
            err_msg=(
                f"Half-normal ECI should be ≈ {expected_eci}, got {eci.mean():.3f}"
            ),
        )

        # Tracking correlation should be low (uncorrelated)
        assert (
            peak_corr.mean() < 0.3
        ), f"Uncorrelated tracking corr should be low, got {peak_corr.mean():.3f}"

        # Info should contain stat_info results
        assert "eci_mean" in info
        assert "peak_corr_mean" in info

    def test_ei_balance_signed_gaussian_uncorrelated(self):
        """Signed Gaussian uncorrelated E/I should have ECI = 1/√2 ≈ 0.707."""
        np.random.seed(42)
        I_e, I_i = generate_signed_gaussian_uncorrelated(
            duration_ms=5000.0, n_neurons=20, sigma_e=1.0, sigma_i=1.0
        )

        eci, peak_corr, best_lag_ms, info = compute_ei_balance(
            I_e, I_i, dt=1.0, max_lag_ms=20.0
        )

        # ECI should be ≈ 0.707 for signed Gaussian
        theoretical_eci = 1.0 / np.sqrt(2)
        np.testing.assert_allclose(
            eci.mean(),
            theoretical_eci,
            atol=0.03,
            err_msg=(
                f"Signed Gaussian ECI should be ≈ {theoretical_eci:.3f}, "
                f"got {eci.mean():.3f}"
            ),
        )

        # Tracking correlation should be low (uncorrelated)
        assert (
            peak_corr.mean() < 0.3
        ), f"Uncorrelated tracking corr should be low, got {peak_corr.mean():.3f}"

        # Info should contain stat_info results
        assert "eci_mean" in info
        assert "peak_corr_mean" in info

    def test_ei_balance_with_known_lag(self):
        """E/I with known phase lag should show correct delay magnitude."""
        np.random.seed(42)
        phase_shift_ms = 10.0
        I_e, I_i = generate_phase_shifted_sinusoids(
            duration_ms=5000.0, n_neurons=10, phase_shift_ms=phase_shift_ms
        )

        eci, peak_corr, best_lag_ms, info = compute_ei_balance(
            I_e, I_i, dt=1.0, max_lag_ms=30.0
        )

        # Delay magnitude should be close to the known phase shift
        assert (
            np.abs(np.abs(best_lag_ms.mean()) - phase_shift_ms) < 1.0
        ), f"Expected |delay| ≈ {phase_shift_ms}, got {best_lag_ms.mean():.1f}"

    def test_ei_balance_external_current(self):
        """External current should increase ECI."""
        np.random.seed(42)
        I_e, I_i = generate_almost_perfect_balance_signal(
            duration_ms=5000.0, n_neurons=10
        )

        # Without external
        eci_no_ext, _, _, _ = compute_ei_balance(I_e, I_i)

        # With external
        I_ext = np.ones_like(I_e) * 0.5
        eci_with_ext, _, _, _ = compute_ei_balance(I_e, I_i, I_ext=I_ext)

        assert eci_with_ext.mean() > eci_no_ext.mean(), (
            f"External current should increase ECI: "
            f"{eci_no_ext.mean():.3f} -> {eci_with_ext.mean():.3f}"
        )

    def test_ei_balance_torch_numpy_consistency(self):
        """Full balance metrics should be consistent between torch and
        numpy."""
        np.random.seed(42)
        I_e, I_i = generate_almost_perfect_balance_signal(
            duration_ms=1000.0, n_neurons=10
        )

        eci_np, peak_np, lag_np, _ = compute_ei_balance(I_e, I_i)
        eci_torch, peak_torch, lag_torch, _ = compute_ei_balance(
            torch.from_numpy(I_e), torch.from_numpy(I_i)
        )

        np.testing.assert_allclose(eci_np, eci_torch.numpy(), rtol=1e-2)
        np.testing.assert_allclose(peak_np, peak_torch.numpy(), rtol=1e-2)
        np.testing.assert_allclose(lag_np, lag_torch.numpy(), rtol=1e-2)

        np.random.seed(42)
        I_e, I_i = generate_zero_signal(duration_ms=1000.0, n_neurons=10)

        eci_np, peak_np, lag_np, _ = compute_ei_balance(I_e, I_i)
        eci_torch, peak_torch, lag_torch, _ = compute_ei_balance(
            torch.from_numpy(I_e), torch.from_numpy(I_i)
        )
        np.testing.assert_allclose(eci_np, eci_torch.numpy(), rtol=1e-2)
        np.testing.assert_allclose(peak_np, peak_torch.numpy(), rtol=1e-2)
        np.testing.assert_allclose(lag_np, lag_torch.numpy(), rtol=1e-2)

    def test_ei_balance_metrics_structure(self):
        """Full balance should return all expected metrics."""
        np.random.seed(42)
        I_e, I_i = generate_almost_perfect_balance_signal(
            duration_ms=1000.0, n_neurons=10
        )

        eci, peak_corr, best_lag_ms, info = compute_ei_balance(I_e, I_i)

        # Check return shapes
        assert eci.shape == (10,)
        assert peak_corr.shape == (10,)
        assert best_lag_ms.shape == (10,)

        # Check info contains stat_info results
        assert "eci_mean" in info
        assert "peak_corr_mean" in info
        assert "best_lag_ms_mean" in info

    def test_ei_balance_not_float16_collapse(self):
        """Float16 inputs should respect dtype and not collapse lag
        correlation."""
        T, N = 540, 4
        t = np.arange(T, dtype=np.float32)[:, None]
        base = 250.0 * np.sin(2.0 * np.pi * t / 40.0) + 300.0
        I_e = np.repeat(base, N, axis=1).astype(np.float16)
        I_i = -np.roll(I_e, 4, axis=0)

        eci_np, peak_np, lag_np, info_np = compute_ei_balance(
            I_e,
            I_i,
            dt=1.0,
            max_lag_ms=8.0,
            batch_axis=1,
            dtype=np.float32,
        )

        assert eci_np.dtype == np.float32
        assert peak_np.dtype == np.float32
        assert lag_np.dtype == np.float32
        np.testing.assert_allclose(peak_np.mean(), 0.995, atol=0.02)
        np.testing.assert_allclose(lag_np.mean(), -4.0, atol=0.5)

        if torch.cuda.is_available():
            eci_t, peak_t, lag_t, info_t = compute_ei_balance(
                torch.from_numpy(I_e).cuda(),
                torch.from_numpy(I_i).cuda(),
                dt=1.0,
                max_lag_ms=8.0,
                batch_axis=1,
                dtype=torch.float32,
            )

            assert eci_t.dtype == torch.float32
            assert peak_t.dtype == torch.float32
            assert lag_t.dtype == torch.float32
            np.testing.assert_allclose(peak_t.cpu().numpy(), peak_np, atol=5e-3)
            np.testing.assert_allclose(lag_t.cpu().numpy(), lag_np, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestEIBalanceFullGPU:
    """GPU-specific full balance tests."""

    def test_ei_balance_full_gpu(self):
        """Full E/I balance computation works on GPU."""
        np.random.seed(42)
        I_e, I_i = generate_almost_perfect_balance_signal(
            duration_ms=1000.0, n_neurons=10
        )

        I_e_gpu = torch.from_numpy(I_e).cuda()
        I_i_gpu = torch.from_numpy(I_i).cuda()

        eci_gpu, peak_gpu, lag_gpu, info_gpu = compute_ei_balance(I_e_gpu, I_i_gpu)

        assert isinstance(eci_gpu, torch.Tensor)
        assert eci_gpu.device.type == "cuda"
        assert eci_gpu.mean() < 0.1  # Should still be balanced
