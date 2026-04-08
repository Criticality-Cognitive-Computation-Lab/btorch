"""Example: Delayed synaptic connections.

This example demonstrates how to use heterogeneous synaptic delays in btorch.
It shows:
1. Creating a connection matrix with per-connection delays
2. Using SpikeHistory for rolling spike buffer
3. Running a simulation with delayed synaptic transmission
"""

import numpy as np
import scipy.sparse
import torch

from btorch.connectome.connection import expand_conn_for_delays
from btorch.models.history import SpikeHistory
from btorch.models.linear import SparseConn


def simple_delay_demo():
    """Simple 2-neuron network demonstrating synaptic delays."""
    print("=" * 60)
    print("Simple Delay Demo: 2 Neuron Network")
    print("=" * 60)

    # Create network:
    # Neuron 0 -> Neuron 1 with weight 5.0, delay=2 timesteps
    n_neurons = 2

    # Base connection matrix
    conn = scipy.sparse.coo_array(
        ([5.0], ([0], [1])),  # neuron 0 -> neuron 1
        shape=(n_neurons, n_neurons),
    )

    # Per-connection delay (in dt steps)
    delays = np.array([2])  # This connection has 2 timestep delay

    # Expand connection for delays
    # Shape: (2*5, 2) = (10, 2) - 5 delay bins per neuron
    conn_d = expand_conn_for_delays(conn, delays, n_delay_bins=5)
    print(f"Expanded connection shape: {conn_d.shape}")
    print(f"Original: (2, 2), Expanded: (2*5, 2) = {conn_d.shape}")

    # Create linear layer
    linear = SparseConn(conn_d, enforce_dale=False)

    # Create spike history buffer
    history = SpikeHistory(n_neurons, max_delay_steps=5)
    history.init_state(batch_size=1)  # Initialize with batch_size=1

    # Simulate for 10 timesteps
    print("\nSimulation:")
    print(f"{'Time':>4} | {'Input':>10} | {'Spike':>10} | {'PSC to N1':>10}")
    print("-" * 50)

    for t in range(10):
        # Inject strong input to neuron 0 at t=0
        if t == 0:
            x = torch.tensor([[10.0, 0.0]])  # Strong input to neuron 0
        else:
            x = torch.zeros(1, n_neurons)

        # Simple thresholding (spike if input > 1.0)
        spike = (x > 1.0).float()

        # Update history
        history.update(spike)

        # Get delayed spikes
        z_delayed = history.get_flattened(5)

        # Compute postsynaptic current
        psc = linear(z_delayed)

        print(
            f"{t:>4} | {x[0, 0].item():>10.1f} | {spike[0, 0].item():>10.0f} "
            f"| {psc[0, 1].item():>10.1f}"
        )

    print("\nNote: PSC to neuron 1 appears at t=2 (delay=2)")


def multiple_delays_demo():
    """3-neuron network with multiple delay bins."""
    print("\n" + "=" * 60)
    print("Multiple Delays Demo: 3 Neuron Network")
    print("=" * 60)

    n_neurons = 3

    # Connections:
    # 0 -> 1: weight 3.0, delay=0 (fast)
    # 0 -> 2: weight 3.0, delay=2 (medium)
    # 1 -> 2: weight 3.0, delay=4 (slow)
    conn = scipy.sparse.coo_array(
        ([3.0, 3.0, 3.0], ([0, 0, 1], [1, 2, 2])), shape=(n_neurons, n_neurons)
    )

    delays = np.array([0, 2, 4])  # Per-connection delays

    # Expand connection
    conn_d = expand_conn_for_delays(conn, delays, n_delay_bins=5)
    print(f"Connection shape: {conn_d.shape}")

    # Create components
    linear = SparseConn(conn_d, enforce_dale=False)
    history = SpikeHistory(n_neurons, max_delay_steps=5)
    history.init_state(batch_size=1)  # Initialize with batch_size=1

    # Simulate
    print("\nSimulation (spike at neuron 0 at t=0):")
    print(
        f"{'Time':>4} | {'N0':>4} | {'N1':>4} | {'N2':>4} | {'PSC N0':>8} "
        f"| {'PSC N1':>8} | {'PSC N2':>8}"
    )
    print("-" * 60)

    for t in range(8):
        # Input at neuron 0, t=0
        if t == 0:
            x = torch.tensor([[5.0, 0.0, 0.0]])
        else:
            x = torch.zeros(1, n_neurons)

        spike = (x > 1.0).float()
        history.update(spike)
        z_delayed = history.get_flattened(5)
        psc = linear(z_delayed)

        print(
            f"{t:>4} | {spike[0, 0].item():>4.0f} | {spike[0, 1].item():>4.0f} "
            f"| {spike[0, 2].item():>4.0f} | "
            f"{psc[0, 0].item():>8.1f} | {psc[0, 1].item():>8.1f} "
            f"| {psc[0, 2].item():>8.1f}"
        )

    print("\nNote:")
    print("  - N1 receives input at t=0 (delay=0 from N0)")
    print("  - N2 receives input at t=2 (delay=2 from N0)")
    print("  - N2 receives input at t=4 (delay=4 from N1, which spiked at t=0)")


def delay_with_dale_demo():
    """Show delays working with Dale's law."""
    print("\n" + "=" * 60)
    print("Dale's Law + Delays Demo")
    print("=" * 60)

    n_neurons = 3

    # Excitatory connection: positive weight
    # Inhibitory connection: negative weight
    conn = scipy.sparse.coo_array(
        ([2.0, -2.0], ([0, 1], [2, 2])),  # Both target neuron 2
        shape=(n_neurons, n_neurons),
    )

    # Different delays for E and I inputs
    delays = np.array([1, 3])  # E: delay=1, I: delay=3

    conn_d = expand_conn_for_delays(conn, delays, n_delay_bins=5)

    # With enforce_dale=True, signs are preserved
    linear = SparseConn(conn_d, enforce_dale=True)

    history = SpikeHistory(n_neurons, max_delay_steps=5)
    history.init_state(batch_size=1)  # Initialize with batch_size=1

    print("\nSimulation (spikes at N0 and N1 at t=0):")
    print(f"{'Time':>4} | {'N0':>4} | {'N1':>4} | {'PSC N2':>10} | {'Notes':>30}")
    print("-" * 60)

    for t in range(6):
        if t == 0:
            x = torch.tensor([[5.0, 5.0, 0.0]])  # Spikes at N0 and N1
        else:
            x = torch.zeros(1, n_neurons)

        spike = (x > 1.0).float()
        history.update(spike)
        z_delayed = history.get_flattened(5)
        psc = linear(z_delayed)

        note = ""
        if t == 1:
            note = "E input arrives (delay=1)"
        elif t == 3:
            note = "I input arrives (delay=3)"
        elif t == 2:
            note = "E active, waiting for I"

        print(
            f"{t:>4} | {spike[0, 0].item():>4.0f} | {spike[0, 1].item():>4.0f} "
            f"| {psc[0, 2].item():>10.1f} | {note:>30}"
        )

    print("\nNote: Dale's law preserves sign - E is positive, I is negative")


def main():
    """Run all demos."""
    simple_delay_demo()
    multiple_delays_demo()
    delay_with_dale_demo()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
