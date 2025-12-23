import numpy as np
from dynamic_tools.criticality import calculate_dfa, compute_avalanche_statistics


def test_criticality():
    # 1. Generate random spike train (Poisson)
    # This should NOT exhibit power law (exponential distribution expected)
    n_neurons = 100
    n_steps = 10000
    p_spike = 0.01

    spike_train = np.random.rand(n_steps, n_neurons) < p_spike

    print("Running avalanche analysis on random Poisson data...")
    results = compute_avalanche_statistics(spike_train, bin_size=1)

    print(f"Found {len(results['sizes'])} avalanches.")
    print(f"Tau (Size exponent): {results['tau']}")
    print(f"Alpha (Duration exponent): {results['alpha']}")
    print(f"Gamma (Size vs Duration exponent): {results['gamma']}")
    if results.get("gamma_stats"):
        print(f"Gamma fit R^2: {results['gamma_stats']['r_squared']:.4f}")

    print(f"Predicted Gamma: {results['gamma_pred']}")
    print(f"CCC (Criticality Consistency Coefficient): {results['CCC']}")

    if results["fit_S"]:
        print(f"Size fit xmin: {results['fit_S'].xmin}")
        R, p = results["fit_S"].distribution_compare("power_law", "exponential")
        print(
            f"Size distribution comparison (Power Law vs Exponential): R={R:.4f}, p={p:.4f}"
        )
        if R < 0:
            print("  -> Data is more likely Exponential (Random/Non-critical)")
        else:
            print("  -> Data is more likely Power Law (Critical)")

    if results["fit_T"]:
        print(f"Duration fit xmin: {results['fit_T'].xmin}")
        R, p = results["fit_T"].distribution_compare("power_law", "exponential")
        print(
            "Duration distribution comparison (Power Law vs Exponential): "
            f"R={R:.4f}, p={p:.4f}"
        )

    # 2. Generate synthetic power-law data (just to test fitting)
    # We can't easily generate spike train that results in power law without a model,
    # but we can check if the function handles the output correctly.

    print("\nTest complete.")


def test_dfa():
    print("\nTesting Detrended Fluctuation Analysis (DFA)...")

    n_steps = 10000

    # 1. White Noise (Random) -> Expected alpha ~ 0.5
    white_noise = np.random.randn(n_steps)
    alpha_white = calculate_dfa(white_noise, bin_size=1)
    print(f"White Noise (Expected ~0.5): {alpha_white:.4f}")

    # 2. Brownian Motion (Random Walk) -> Expected alpha ~ 1.5
    # Cumulative sum of white noise
    brown_noise = np.cumsum(np.random.randn(n_steps))
    alpha_brown = calculate_dfa(brown_noise, bin_size=1)
    print(f"Brownian Motion (Expected ~1.5): {alpha_brown:.4f}")

    # 3. Pink Noise (1/f) -> Expected alpha ~ 1.0
    # Harder to generate simply, but we can verify the other two.

    if 0.4 < alpha_white < 0.6 and 1.4 < alpha_brown < 1.6:
        print("DFA Test Passed!")
    else:
        print("DFA Test Failed (Values outside expected range)")


if __name__ == "__main__":
    test_criticality()
    test_dfa()
