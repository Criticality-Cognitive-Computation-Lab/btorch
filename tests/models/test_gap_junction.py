"""Electrical coupling use cases and examples.

This module tests both GapJunction (electrical synapses) and VoltageCoupling
(multicompartment models) using parametrized tests where applicable.

Gap junctions are electrical synapses that allow direct ion flow between neurons,
creating instantaneous, bidirectional coupling. Common use cases include:
1. Synchronizing neuronal populations (e.g., interneuron networks)
2. Modeling specific circuits (retinal ganglion cells, thalamic relay neurons)
3. Rapid signal propagation without synaptic delay
4. Bidirectional coupling between neuron pairs

VoltageCoupling is used for multicompartment neuron models where coupling currents
are computed as W*V rather than W*(V_post - V_pre).
"""

import platform

import matplotlib.pyplot as plt
import pytest
import torch

from btorch.models.synapse import GapJunction, VoltageCoupling
from btorch.utils.file import save_fig


# =============================================================================
# Utility Functions
# =============================================================================


def _identity_linear(n: int) -> torch.nn.Linear:
    """Create a linear layer with identity weights."""
    linear = torch.nn.Linear(n, n, bias=False)
    torch.nn.init.eye_(linear.weight)
    return linear


# =============================================================================
# Parametrized Tests (Shared between GapJunction and VoltageCoupling)
# =============================================================================


CouplingClass = GapJunction | VoltageCoupling


@pytest.mark.parametrize(
    "cls,conductance_attr,kwargs",
    [
        (GapJunction, "g_gap", {"g_gap": 0.5}),
        (VoltageCoupling, "g_couple", {"g_couple": 0.5}),
    ],
)
def test_basic_identity(cls: type[CouplingClass], conductance_attr: str, kwargs: dict):
    """Basic use case: identity weights produce predictable output.

    For GapJunction: I = g * (v_post - v_pre)
    For VoltageCoupling: I = g * v
    """
    # Two neurons with identity weights for predictable output
    instance = cls(n_neuron=2, linear=_identity_linear(2), **kwargs)

    if cls is GapJunction:
        # Neuron 0 at 10mV, neuron 1 at 0mV
        v_pre = torch.tensor([[10.0, 0.0]])
        v_post = torch.tensor([[0.0, 0.0]])
        i_out = instance(v_pre, v_post)
        # delta_v = [0-10, 0-0] = [-10, 0]
        # I = 0.5 * [-10, 0] = [-5, 0]
        expected = torch.tensor([[-5.0, 0.0]])
    else:
        # Compartment 0 at 10mV, compartment 1 at 0mV
        v = torch.tensor([[10.0, 0.0]])
        i_out = instance(v)
        # I = 0.5 * [10, 0] = [5, 0]
        expected = torch.tensor([[5.0, 0.0]])

    torch.testing.assert_close(i_out, expected, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    "cls,conductance_attr,kwargs",
    [
        (GapJunction, "g_gap", {"g_gap": 0.5}),
        (VoltageCoupling, "g_couple", {"g_couple": 0.5}),
    ],
)
def test_learnable_conductance(
    cls: type[CouplingClass], conductance_attr: str, kwargs: dict
):
    """Use case: training conductances as network parameters."""
    instance = cls(n_neuron=3, **kwargs)

    # Make conductance trainable
    g_param = torch.nn.Parameter(getattr(instance, conductance_attr))
    setattr(instance, conductance_attr, g_param)

    if cls is GapJunction:
        v_input = torch.tensor([[10.0, 5.0, 0.0]], requires_grad=True)
        v_post = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
        i_out = instance(v_input, v_post)
    else:
        v_input = torch.tensor([[10.0, 5.0, 0.0]], requires_grad=True)
        i_out = instance(v_input)

    loss = (i_out**2).sum()
    loss.backward()

    g_grad = getattr(instance, conductance_attr).grad
    assert g_grad is not None
    assert g_grad.item() != 0


@pytest.mark.parametrize(
    "cls,conductance_attr,kwargs",
    [
        (GapJunction, "g_gap", {"g_gap": 0.0}),
        (VoltageCoupling, "g_couple", {"g_couple": 0.0}),
    ],
)
def test_zero_conductance(
    cls: type[CouplingClass], conductance_attr: str, kwargs: dict
):
    """Edge case: zero conductance produces no current."""
    instance = cls(n_neuron=3, **kwargs)

    v = torch.randn(2, 3)
    if cls is GapJunction:
        v_post = torch.randn(2, 3)
        i_out = instance(v, v_post)
    else:
        i_out = instance(v)

    assert torch.allclose(i_out, torch.zeros_like(i_out))


@pytest.mark.parametrize(
    "cls,conductance_attr,kwargs",
    [
        (GapJunction, "g_gap", {"g_gap": 0.2}),
        (VoltageCoupling, "g_couple", {"g_couple": 0.2}),
    ],
)
def test_batch_processing(
    cls: type[CouplingClass], conductance_attr: str, kwargs: dict
):
    """Use case: processing multiple neurons across batch dimension."""
    instance = cls(n_neuron=4, linear=_identity_linear(4), **kwargs)

    batch_size = 8
    v = torch.randn(batch_size, 4)

    if cls is GapJunction:
        v_post = torch.randn(batch_size, 4)
        i_out = instance(v, v_post)
        # Each batch element computed independently
        for b in range(batch_size):
            expected = 0.2 * (v_post[b] - v[b])
            torch.testing.assert_close(i_out[b], expected, atol=1e-6, rtol=0.0)
    else:
        i_out = instance(v)
        assert i_out.shape == (batch_size, 4)
        # Each batch element computed independently
        for b in range(batch_size):
            expected = 0.2 * v[b]
            torch.testing.assert_close(i_out[b], expected, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    "cls,conductance_attr,kwargs",
    [
        (GapJunction, "g_gap", {"g_gap": 0.5}),
        (VoltageCoupling, "g_couple", {"g_couple": 0.5}),
    ],
)
def test_time_series(cls: type[CouplingClass], conductance_attr: str, kwargs: dict):
    """Use case: coupling currents in time-series simulations."""
    instance = cls(n_neuron=2, linear=_identity_linear(2), **kwargs)

    T = 100
    t = torch.arange(T, dtype=torch.float32)
    v_seq = torch.stack(
        [
            torch.sin(t * 0.1),
            torch.cos(t * 0.1),
        ],
        dim=1,
    ).unsqueeze(1)  # (T, 1, 2)

    if cls is GapJunction:
        v_post_seq = torch.zeros_like(v_seq)
        i_seq = instance.multi_step_forward(v_seq, v_post_seq)
        # With identity and g=0.5: I = 0.5 * (0 - sin(t)) = -0.5 * sin(t)
        assert i_seq.shape == (T, 1, 2)
        expected = 0.5 * (v_post_seq - v_seq)
        torch.testing.assert_close(i_seq, expected, atol=1e-6, rtol=0.0)
    else:
        i_seq = instance.multi_step_forward(v_seq)
        # With identity and g=0.5: I = 0.5 * V
        assert i_seq.shape == (T, 1, 2)
        expected = 0.5 * v_seq
        torch.testing.assert_close(i_seq, expected, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    "cls,conductance_attr,kwargs",
    [
        (GapJunction, "g_gap", {"g_gap": 0.1}),
        (VoltageCoupling, "g_couple", {"g_couple": 0.1}),
    ],
)
def test_2d_spatial_layout(
    cls: type[CouplingClass], conductance_attr: str, kwargs: dict
):
    """Use case: 2D spatial arrangement of neurons/compartments."""
    instance = cls(n_neuron=(4, 4), **kwargs)

    # 2D voltage map
    v = torch.zeros(1, 4, 4)
    v[0, 1:3, 1:3] = 10.0

    if cls is GapJunction:
        v_post = torch.zeros(1, 4, 4)
        i_out = instance(v, v_post)
        assert i_out.shape == (1, 4, 4)
        assert torch.all(i_out[0, 1:3, 1:3] < 0), "Hotspot should lose current"
    else:
        i_out = instance(v)
        assert i_out.shape == (1, 4, 4)


@pytest.mark.skipif(
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
@pytest.mark.parametrize(
    "cls,conductance_attr,kwargs",
    [
        (GapJunction, "g_gap", {"g_gap": 0.3}),
        (VoltageCoupling, "g_couple", {"g_couple": 0.3}),
    ],
)
def test_compile_compatibility(
    cls: type[CouplingClass], conductance_attr: str, kwargs: dict
):
    """Test that coupling classes work with torch.compile."""
    from tests.utils.compile import compile_or_skip

    instance = cls(n_neuron=4, **kwargs)
    compiled = compile_or_skip(instance)

    v = torch.randn(2, 4)

    if cls is GapJunction:
        v_post = torch.randn(2, 4)
        i_eager = instance(v, v_post)
        i_compiled = compiled(v, v_post)
    else:
        i_eager = instance(v)
        i_compiled = compiled(v)

    torch.testing.assert_close(i_eager, i_compiled, atol=1e-6, rtol=0.0)


# =============================================================================
# GapJunction-Specific Tests
# =============================================================================


def test_gap_junction_bidirectional_symmetry():
    """Gap junctions are bidirectional: I(A->B) = -I(B->A).

    This property ensures energy conservation and reciprocal coupling.
    In a network, each gap junction should be applied twice (once in each
    direction) or use symmetric weight matrices.
    """
    gap = GapJunction(n_neuron=2, g_gap=1.0)

    v_a = torch.tensor([[10.0, 0.0]])
    v_b = torch.tensor([[0.0, 10.0]])

    # Current when A is "pre" and B is "post"
    i_a_to_b = gap(v_a, v_b)
    # Current when B is "pre" and A is "post"
    i_b_to_a = gap(v_b, v_a)

    # Should be equal and opposite
    torch.testing.assert_close(i_a_to_b, -i_b_to_a, atol=1e-6, rtol=0.0)


def test_gap_junction_synchronization():
    """Use case: gap junctions synchronize coupled neurons over time.

    Demonstrates how gap junctions pull coupled neurons toward each other.
    Neuron with higher voltage loses charge (negative current) to neuron
    with lower voltage (positive current when viewed from the other side).
    """
    gap = GapJunction(n_neuron=2, g_gap=0.5)

    # Scenario: neuron 0 at 10mV, neuron 1 at 0mV
    v_pre = torch.tensor([[10.0, 0.0]])
    v_post = torch.tensor([[0.0, 0.0]])

    i_gap = gap(v_pre, v_post)

    # Neuron 0 receives negative current (loses charge to the network)
    assert (
        i_gap[0, 0] < 0
    ), f"Higher voltage neuron should lose current, got {i_gap[0, 0].item()}"

    # Reverse: if we swap pre/post, neuron 1 now receives from neuron 0
    i_gap_reverse = gap(v_post, v_pre)
    # Neuron 0 receives positive current (gains charge from network)
    assert (
        i_gap_reverse[0, 0] > 0
    ), f"Lower voltage neuron should gain current, got {i_gap_reverse[0, 0].item()}"

    # The currents are equal and opposite, demonstrating synchronization
    torch.testing.assert_close(i_gap[0, 0], -i_gap_reverse[0, 0], atol=1e-6, rtol=0.0)


def test_gap_junction_equal_voltage_no_current():
    """Edge case: equal voltages produce no current (Ohm's law).

    When coupled neurons have the same membrane potential, there is no
    driving force for ion flow, regardless of conductance strength.
    """
    gap = GapJunction(n_neuron=4, g_gap=1.0)

    v = torch.randn(2, 4)
    i_gap = gap(v, v.clone())

    assert torch.allclose(i_gap, torch.zeros_like(i_gap), atol=1e-6)


def test_gap_junction_ring_network():
    """Use case: ring network with nearest-neighbor coupling.

    Common in modeling electrically coupled interneuron networks where each
    neuron connects to its neighbors (e.g., cortical fast-spiking interneurons).
    """
    n = 4
    # Create coupling matrix: each neuron connects to neighbors
    linear = torch.nn.Linear(n, n, bias=False)
    with torch.no_grad():
        w = torch.zeros(n, n)
        for i in range(n):
            w[i, (i - 1) % n] = 0.5  # left neighbor
            w[i, (i + 1) % n] = 0.5  # right neighbor
        linear.weight.copy_(w)

    gap = GapJunction(n_neuron=n, g_gap=1.0, linear=linear)

    # One neuron at high voltage, others at rest
    v_pre = torch.zeros(1, n)
    v_pre[0, 0] = 10.0
    v_post = torch.zeros(1, n)

    i_gap = gap(v_pre, v_post)

    # Verify neighbors receive equal current from neuron 0
    assert (
        abs(i_gap[0, 1] - i_gap[0, 3]) < 1e-6
    ), "Neighbors should receive equal current"
    assert (
        abs(i_gap[0, 1]) > 0
    ), f"Neighbor 1 should have current, got {i_gap[0, 1].item()}"
    assert (
        abs(i_gap[0, 3]) > 0
    ), f"Neighbor 3 should have current, got {i_gap[0, 3].item()}"
    # Distant neuron (2) should have no direct coupling
    assert (
        abs(i_gap[0, 2]) < 1e-6
    ), f"Distant neuron should have no current, got {i_gap[0, 2].item()}"


# =============================================================================
# Visualization Tests for Complex Network Dynamics
# =============================================================================


def test_gap_junction_ring_network_visualization():
    """Visualize wave propagation in a ring network.

    Shows how an input pulse at one neuron propagates through
    electrically coupled neurons in a ring topology.
    """
    n = 4
    # Create ring coupling: each neuron connects to neighbors
    linear = torch.nn.Linear(n, n, bias=False)
    with torch.no_grad():
        w = torch.zeros(n, n)
        for i in range(n):
            w[i, (i - 1) % n] = 0.5  # Left neighbor
            w[i, (i + 1) % n] = 0.5  # Right neighbor
        linear.weight.copy_(w)

    gap = GapJunction(n_neuron=n, g_gap=1.0, linear=linear)

    # Simulation parameters
    T = 800
    dt = 0.1  # ms
    time_ms = torch.arange(T, dtype=torch.float32) * dt
    tau = 20.0  # ms
    R = 0.5
    v_rest = -65.0

    # Initialize all neurons at rest
    v = torch.zeros(T, n)
    v[0, :] = v_rest

    # Inject current into neuron 0 at specific times
    i_inject = torch.zeros(T, n)
    for pulse_time in [10.0, 40.0, 70.0]:
        mask = (time_ms >= pulse_time) & (time_ms < pulse_time + 5.0)
        i_inject[mask, 0] = 50.0  # pA injection into neuron 0

    # Simulate with gap junction coupling
    # Gap junction current into neuron i: I_i = g * sum_j W[i,j] * (V_j - V_i)
    # This can be computed as: I = g * (W @ V - diag(sum_j W[i,j]) @ V)
    # For our symmetric ring: sum_j W[i,j] = 1.0 for all i
    for t in range(T - 1):
        # Compute gap junction currents using the weight matrix
        # I_gap[i] = g * sum_j W[i,j] * (V[j] - V[i])
        with torch.no_grad():
            v_flat = v[t : t + 1, :]  # (1, n)
            # neighbor_contrib = W @ V gives sum of weighted neighbor voltages
            neighbor_contrib = torch.nn.functional.linear(v_flat, linear.weight)
            # total_weight[i] = sum_j W[i,j] = degree of node i
            total_weight = linear.weight.sum(dim=1)
            # I = g * (neighbor_contrib - total_weight * V)
            i_gap = gap.g_gap * (neighbor_contrib[0] - total_weight * v_flat[0])

        # Update voltages
        for i in range(n):
            leak = -(v[t, i] - v_rest) / tau
            dv = (leak + R * (i_gap[i] + i_inject[t, i])) * dt
            v[t + 1, i] = v[t, i] + dv

    # Create figure with voltage traces for all neurons
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Plot voltages
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i in range(n):
        axes[0].plot(
            time_ms.numpy(),
            v[:, i].numpy(),
            label=f"Neuron {i}",
            color=colors[i],
            lw=1.5,
        )
    axes[0].axhline(y=v_rest, color="k", linestyle="--", alpha=0.3)
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title("Ring Network: Pulse Propagation via Gap Junctions")
    axes[0].legend(loc="upper right", ncol=n)
    axes[0].grid(True, alpha=0.3)

    # Plot input current
    axes[1].plot(time_ms.numpy(), i_inject[:, 0].numpy(), color="red", lw=2)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Input Current (pA)")
    axes[1].set_title("Current Injection (Neuron 0 only)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, name="gap_junction_ring_network")
    plt.close(fig)


def test_gap_junction_2d_spatial_visualization():
    """Visualize 2D spatial coupling with propagating waves.

    Shows how a localized input spreads through a 2D grid of
    electrically coupled neurons.
    """
    size = 5  # 5x5 grid
    gap = GapJunction(n_neuron=(size, size))

    T = 600
    dt = 0.1
    time_ms = torch.arange(T, dtype=torch.float32) * dt
    tau = 20.0
    v_rest = -70.0

    # Initialize voltage grid
    v = torch.zeros(T, 1, size, size)
    v[0, 0, :, :] = v_rest

    # Input pulse at center neuron
    center = size // 2
    i_inject = torch.zeros(T, 1, size, size)
    pulse_start = 10.0
    pulse_end = 30.0
    mask = (time_ms >= pulse_start) & (time_ms < pulse_end)
    i_inject[mask, 0, center, center] = 100.0  # Strong pulse at center

    # Simulate with proper gap junction coupling
    # Gap junction current: I_i = g * sum_j W[i,j] * (V_j - V_i)
    for t in range(T - 1):
        with torch.no_grad():
            v_flat = v[t].reshape(1, -1)  # (1, size*size)
            # neighbor_contrib = W @ V
            neighbor_contrib = torch.nn.functional.linear(v_flat, gap.linear.weight)
            # total_weight[i] = sum_j W[i,j]
            total_weight = gap.linear.weight.sum(dim=1)
            # I = g * (neighbor_contrib - total_weight * V)
            i_gap_flat = gap.g_gap * (neighbor_contrib[0] - total_weight * v_flat[0])
            i_gap = i_gap_flat.reshape(1, size, size)

        # Update voltages
        leak = -(v[t] - v_rest) / tau
        dv = (leak + i_gap * 0.2 + i_inject[t] * 0.05) * dt
        v[t + 1] = v[t] + dv

    # Select representative neurons: center, neighbors, edge
    center_idx = center * size + center
    neighbor_idx = center * size + (center + 1)
    edge_idx = 0  # Corner

    # Flatten spatial dimensions for plotting
    v_flat = v[:, 0, :, :].reshape(T, size * size)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot selected neuron voltages
    axes[0].plot(
        time_ms.numpy(),
        v_flat[:, center_idx].numpy(force=True),
        label=f"Center ({center},{center})",
        lw=2,
    )
    axes[0].plot(
        time_ms.numpy(),
        v_flat[:, neighbor_idx].numpy(force=True),
        label=f"Neighbor ({center},{center + 1})",
        lw=2,
    )
    axes[0].plot(
        time_ms.numpy(),
        v_flat[:, edge_idx].numpy(force=True),
        label="Corner (0,0)",
        lw=2,
    )
    axes[0].axhline(y=v_rest, color="k", linestyle="--", alpha=0.3)
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title("2D Grid: Wave Propagation from Center")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Plot input current
    axes[1].plot(
        time_ms.numpy(), i_inject[:, 0, center, center].numpy(), color="red", lw=2
    )
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Input Current (pA)")
    axes[1].set_title("Current Injection (Center only)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, name="gap_junction_2d_wave")
    plt.close(fig)


# =============================================================================
# VoltageCoupling-Specific Tests
# =============================================================================


def test_voltage_coupling_multicompartment_dendrite():
    """Use case: dendritic compartments coupled to soma.

    Models a neuron with soma and dendritic compartments where coupling
    currents flow based on absolute voltages, not differences.
    """
    # 3-compartment model: soma + 2 dendrites
    linear = torch.nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        w = torch.zeros(3, 3)
        w[0, 1] = 0.3  # Soma receives from dendrite 1
        w[0, 2] = 0.3  # Soma receives from dendrite 2
        w[1, 0] = 0.5  # Dendrite 1 receives from soma
        w[2, 0] = 0.5  # Dendrite 2 receives from soma
        linear.weight.copy_(w)

    couple = VoltageCoupling(n_neuron=3, g_couple=1.0, linear=linear)

    # Soma at rest, dendrites depolarized
    v = torch.tensor([[0.0, 10.0, 10.0]])

    i_couple = couple(v)

    # Soma receives current from both dendrites
    assert i_couple[0, 0] > 0, "Soma should receive current from dendrites"
    # Dendrites receive from soma (but soma is at 0, so minimal current)
    torch.testing.assert_close(i_couple[0, 1], torch.tensor(0.0), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(i_couple[0, 2], torch.tensor(0.0), atol=1e-6, rtol=0.0)
