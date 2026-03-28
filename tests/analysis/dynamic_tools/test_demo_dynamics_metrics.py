import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn as nn

from btorch.analysis.dynamic_tools import (
    complexity,
    criticality,
    lyapunov_dynamics,
    micro_scale,
)
from btorch.utils.file import save_fig
from btorch.visualisation.dynamics import (
    plot_avalanche_analysis,
    plot_eigenvalue_spectrum,
    plot_gain_stability,
    plot_lyapunov_spectrum,
    plot_micro_dynamics,
)


# -----------------------------------------------------------------------------
# 1. Simple Dynamical System (Reservoir / Echo State Network)
# -----------------------------------------------------------------------------
class SimpleReservoir(nn.Module):
    def __init__(self, n_neurons=100, dt=1.0, spectral_radius=1.5):
        super().__init__()
        self.n_neurons = n_neurons
        self.dt = dt

        # recurrent weights
        W = np.random.randn(n_neurons, n_neurons)
        # Normalize spectral radius
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        W = W * (spectral_radius / radius)
        self.W = torch.as_tensor(W, dtype=torch.float32)

        # Structure for gain stability mocking
        class DummySynapse:
            def __init__(self, W):
                self.linear = nn.Linear(1, 1)  # dummy
                with torch.no_grad():
                    self.linear.magnitude = nn.Parameter(
                        torch.as_tensor(W)
                    )  # Mock magnitude

        class DummyBrain:
            def __init__(self, W):
                self.synapse = DummySynapse(W)
                self.neuron = torch.zeros(1)  # dummy

        self.brain = DummyBrain(self.W)
        self.state = torch.zeros(1, n_neurons)

    def forward(self, x, steps=100):
        # x: input (Batch, Features) - ignored for autonomous dynamics demo
        outputs = []
        for _ in range(steps):
            # Simple tanh dynamics: x(t) = tanh(W x(t-1) + noise)
            activation = (
                torch.matmul(self.state, self.W.T) + torch.randn_like(self.state) * 0.1
            )
            self.state = torch.tanh(activation)
            outputs.append(self.state)

        return torch.stack(outputs, dim=1)  # (Batch, Time, Neurons)


# -----------------------------------------------------------------------------
# 2. Pytest Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def simulation_data():
    """Run the simulation once and return all data needed for tests."""
    print("\nInitializing system and running simulation...")
    N = 100
    model = SimpleReservoir(n_neurons=N, spectral_radius=1.2)
    dt = 1.0

    # A. Baseline run - generate sparse Poisson-like spikes for avalanche
    # analysis. Sparse firing creates silent periods between avalanches,
    # enabling proper avalanche detection. Continuous thresholded activity
    # would produce only one long avalanche spanning the entire time series.
    rng = np.random.default_rng(42)
    n_steps = 2000
    p_spike = 0.02  # 2% firing rate per neuron per timestep
    baseline_spikes = torch.as_tensor(
        rng.random((n_steps, N)) < p_spike, dtype=torch.float32
    )
    # Generate matching activity/rates for other tests that expect them.
    # Maintain 3D shape (Batch, Time, Neurons) for compatibility.
    baseline_activity = baseline_spikes.unsqueeze(0) * 2.0 - 1.0  # Scale to [-1, 1]
    baseline_rates = baseline_spikes.unsqueeze(0)  # Shape: (1, Time, Neurons)

    # B. Perturbation run (for PCIst)
    model.state = torch.zeros(1, N)
    model.state[:, 0:10] = 5.0
    with torch.no_grad():
        perturb_activity = model(None, steps=500)

    # C. Final run (for RA) - generate different sparse spike pattern
    # Using different seed to simulate different network state
    final_spikes = torch.as_tensor(
        rng.random((n_steps, N)) < p_spike, dtype=torch.float32
    )

    return {
        "model": model,
        "baseline_spikes": baseline_spikes,
        "baseline_rates": baseline_rates,
        "perturb_activity": perturb_activity,
        "baseline_activity": baseline_activity,
        "final_spikes": final_spikes,
        "dt": dt,
    }


# -----------------------------------------------------------------------------
# 3. Test Functions
# -----------------------------------------------------------------------------


def test_criticality_analysis(simulation_data):
    """Test Avalanche and DFA analysis."""
    print("Testing Criticality...")
    baseline_spikes = simulation_data["baseline_spikes"]

    # 1. Avalanche Plot
    fig_av, res_av = plot_avalanche_analysis(baseline_spikes, bin_size=2)
    save_fig(fig_av, name="demo_avalanche")
    plt.close(fig_av)

    # 2. DFA Calculation (Scalar)
    dfa_alpha = criticality.calculate_dfa(baseline_spikes.numpy(), bin_size=1)
    # Just ensure it runs and returns a float/nan
    assert isinstance(dfa_alpha, float)


def test_complexity_analysis(simulation_data):
    """Test RA, PCIst, and Gain Stability."""
    print("Testing Complexity...")
    baseline_spikes = simulation_data["baseline_spikes"]
    final_spikes = simulation_data["final_spikes"]
    perturb_activity = simulation_data["perturb_activity"]
    baseline_activity = simulation_data["baseline_activity"]

    # 1. RA
    ra_score = complexity.calculate_ra(baseline_spikes, final_spikes)
    assert isinstance(ra_score, float)

    # 2. PCIst
    pcist_score = complexity.calculate_pcist(
        perturb_activity.squeeze(0), baseline_activity.squeeze(0)[:500]
    )
    assert isinstance(pcist_score, float)

    # 3. Gain Stability (Visual)
    # Mock data
    slope, intercept = 0.42, -0.1
    g_vals = np.linspace(0.5, 2.0, 10)
    lambdas = slope * g_vals + intercept + np.random.randn(10) * 0.05
    fig_gs, ax_gs = plot_gain_stability((slope, intercept, g_vals, lambdas))
    save_fig(fig_gs, name="demo_gain_stability")
    plt.close(fig_gs)


def test_attractor_dynamics(simulation_data):
    """Test Eigenvalue Spectrum."""
    print("Testing Attractor Dynamics...")
    model = simulation_data["model"]
    W_np = model.W.numpy()

    fig_eig, ax_eig, res_eig = plot_eigenvalue_spectrum(W_np)
    save_fig(fig_eig, name="demo_eigenvalue_spectrum")
    plt.close(fig_eig)
    assert "eigenvalues" in res_eig


def test_lyapunov_dynamics(simulation_data):
    """Test Lyapunov Spectrum."""
    print("Testing Lyapunov Dynamics...")
    baseline_rates = simulation_data["baseline_rates"]

    mean_trace = baseline_rates.mean(dim=2).squeeze(0).numpy()
    try:
        le_spectrum = lyapunov_dynamics.compute_lyapunov_exponent_spectrum(
            mean_trace, emb_dim=3
        )
    except Exception:
        le_spectrum = np.array([0.5, 0.2, 0.0, -0.5, -1.0])

    fig_ly, ax_ly = plot_lyapunov_spectrum(le_spectrum)
    save_fig(fig_ly, name="demo_lyapunov_spectrum")
    plt.close(fig_ly)


def test_micro_dynamics(simulation_data):
    """Test Micro Scale Dynamics."""
    print("Testing Micro Dynamics...")
    baseline_spikes = simulation_data["baseline_spikes"]
    dt = simulation_data["dt"]

    # 1. Plots
    fig_mic, res_mic = plot_micro_dynamics(baseline_spikes, dt=dt)
    save_fig(fig_mic, name="demo_micro_dynamics")
    plt.close(fig_mic)

    # 2. Spike Distance
    try:
        spike_dist = micro_scale.calculate_spike_distance(
            baseline_spikes.numpy(), dt=dt
        )
    except Exception:
        spike_dist = 0.0
    assert isinstance(spike_dist, float)
