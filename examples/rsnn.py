import torch
import torch.nn as nn
import torch.nn.functional as F

from btorch.models import environ, functional, glif, rnn, synapse
from btorch.models.linear import DenseConn


class MinimalRSNN(nn.Module):
    """A minimal Recurrent Spiking Neural Network (RSNN) demonstrating the use
    of btorch's core Spiking Neuron models (GLIF) and AlphaPSC synapses wrapped
    in a multi-step RecurrentNN layer."""

    def __init__(
        self,
        num_input: int,
        num_hidden: int,
        num_output: int,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()

        # 1. Input projection
        self.fc_in = nn.Linear(
            num_input, num_hidden, bias=False, device=device, dtype=dtype
        )

        # 2. Recurrent Spiking Layer
        # Define the biological spiking neuron.
        # Note: parameters (c_m, tau, tau_ref, k, asc_amps, etc.) can also be
        # a numpy array or tensor with an N-neuron dimension
        # (e.g., shape `(num_hidden,)`) to support heterogeneous parameters.
        neuron_module = glif.GLIF3(
            n_neuron=num_hidden,
            v_threshold=-45.0,
            v_reset=-60.0,
            c_m=2.0,
            tau=20.0,
            tau_ref=2.0,
            k=[0.1, 0.2],
            asc_amps=[1.0, -2.0],
            step_mode="s",  # single step definition
            backend="torch",
            device=device,
            dtype=dtype,
        )

        # Define the recurrent synaptic connection weights using DenseConn.
        conn = DenseConn(num_hidden, num_hidden, bias=False, device=device, dtype=dtype)

        # Define the synaptic dynamics (Alpha Post-Synaptic Current)
        psc_module = synapse.AlphaPSC(
            n_neuron=num_hidden,
            tau_syn=5.0,
            linear=conn,
            step_mode="s",
        )

        # Wrap into a RecurrentNN multi-step layer
        self.brain = rnn.RecurrentNN(
            neuron=neuron_module,
            synapse=psc_module,
            step_mode="m",  # process multiple time steps (T, B, ...)
            # `update_state_names` takes a tuple of dot-separated strings
            # to select which internal state variables to record and return.
            # E.g., "neuron.v" fetches the membrane potential `v` of `neuron_module`
            # at every timestep. "synapse.psc" fetches the post-synaptic current.
            # Depending on register_memory, some state vars like Iasc in glif
            # can have an extra dimension, yielding a shape like
            # (T, Batch, num_hidden, num_Iasc).
            update_state_names=("neuron.v", "neuron.Iasc", "synapse.psc"),
        )

        # 3. Output readout
        self.fc_out = nn.Linear(
            num_hidden, num_output, bias=False, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor, return_states: bool = False):
        """Forward pass.

        Args:
            x (torch.Tensor): Input sequence of shape (T, Batch, num_input)
            return_states (bool): If True, returns (out, spike, states)
        Returns:
            torch.Tensor or tuple: Output of shape (Batch, num_output),
                                   or (out, spike, states) if requested.
        """
        # Linear projection along the time dimension
        x = self.fc_in(x)  # -> (T, Batch, num_hidden)

        # Process dynamically through recurrent brain layer
        # `spike` will have shape (T, Batch, num_hidden)
        #
        # `states` is a dictionary containing the recorded state variables at each
        # timestep according to `update_state_names` (keys like "neuron.v").
        # You can use `btorch.utils.dict_utils.unflatten_dict(states, dot=True)`
        # to convert this flat dotted-dict into a nested dict:
        # `{"neuron": {"v": Tensor}, "synapse": {"psc": Tensor}}`
        spike, states = self.brain(
            x
        )  # spike: (T, Batch, num_hidden), states values: (T, Batch, num_hidden, ...)

        # Decode using rate-based output (mean spike rate over time)
        rate = spike.mean(dim=0)  # -> (Batch, num_hidden)
        out = self.fc_out(rate)  # -> (Batch, num_output)

        if return_states:
            return out, spike, states
        return out


def sim(
    net: nn.Module,
    num_input: int,
    timesteps: int = 100,
    dt: float = 1.0,
    batch_size: int = 4,
    device="cpu",
):
    """Run a simple forward simulation without training."""
    import matplotlib.pyplot as plt

    from btorch.utils.dict_utils import unflatten_dict
    from btorch.utils.file import save_fig
    from btorch.visualisation.timeseries import plot_neuron_traces, plot_raster

    # Global environment config required for ODE solvers inside neurons
    environ.set(dt=dt)
    print(f"\n--- Running Simulation (dt={dt}, T={timesteps}) ---")

    print("Initializing network internal memory states...")
    functional.init_net_state(
        net, batch_size=batch_size, device=device, dtype=torch.float32
    )

    net.eval()

    # -> (T, Batch, num_input)
    inputs = 10 * torch.rand((timesteps, batch_size, num_input), device=device)

    with torch.no_grad():
        # Reset Network State Before Simulation
        functional.reset_net(
            net, batch_size=batch_size, device=device, dtype=torch.float32
        )
        out, spike, states = net(inputs, return_states=True)
        # out: (Batch, num_output), spike: (T, Batch, num_hidden)

    print(f"Simulation complete. Output shape: {out.shape}")

    # Plotting first batch sample
    print("Generating and saving figures...")
    spike_b0 = spike[:, 0, :]  # -> (T, num_hidden)
    states_nested = unflatten_dict(states, dot=True)
    v_b0 = states_nested["neuron"]["v"][:, 0, :]  # -> (T, num_hidden)
    Iasc_b0 = states_nested["neuron"]["Iasc"][:, 0, ...]  # -> (T, num_hidden, num_Iasc)
    psc_b0 = states_nested["synapse"]["psc"][:, 0, ...]  # -> (T, num_hidden, num_psc)

    # Raster plot
    ax_raster = plot_raster(spike_b0, dt=dt, title="Raster Plot (Batch 0)")
    fig_raster = (
        ax_raster[0].figure if isinstance(ax_raster, tuple) else ax_raster.figure
    )
    save_fig(fig_raster, "rsnn_raster")
    plt.close(fig_raster)

    # Neuron Traces (Plotting first 5 neurons for clarity)
    ax_traces = plot_neuron_traces(
        voltage=v_b0[:, :5],
        dt=dt,
        spikes=spike_b0[:, :5],
        asc=Iasc_b0[:, :5, ...],
        psc=psc_b0[:, :5, ...],
    )
    fig_traces = (
        ax_traces[0].figure if isinstance(ax_traces, tuple) else ax_traces.figure
    )
    save_fig(fig_traces, "rsnn_traces")
    plt.close(fig_traces)
    print("Figures saved successfully.")


def train(
    net: nn.Module,
    num_input: int,
    num_output: int,
    timesteps: int = 15,
    dt: float = 1.0,
    batch_size: int = 4,
    epochs: int = 10,
    device="cpu",
):
    """Run a simple dummy training loop."""
    # Global environment config required for ODE solvers inside neurons
    environ.set(dt=dt)
    print(f"\n--- Running Training (dt={dt}, T={timesteps}, epochs={epochs}) ---")

    print("Initializing network internal memory states...")
    functional.init_net_state(
        net, batch_size=batch_size, device=device, dtype=torch.float32
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    net.train()
    for epoch in range(epochs):
        # Create Dummy Data
        # inputs -> (T, Batch, num_input)
        inputs = torch.rand((timesteps, batch_size, num_input), device=device)
        # targets -> (Batch, num_output)
        targets = torch.rand((batch_size, num_output), device=device)

        # Reset Network State Before Simulation step
        # batch_size can be skipped if it is not changed
        functional.reset_net(
            net, batch_size=batch_size, device=device, dtype=torch.float32
        )

        optimizer.zero_grad()
        out = net(inputs)  # -> (Batch, num_output)
        loss = F.mse_loss(out, targets)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} - Step Loss: {loss.item():.4f}")

    print("Training complete!")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Simple Hyperparameters
    num_input = 20
    num_hidden = 64
    num_output = 5

    # Initialize network
    net = MinimalRSNN(
        num_input=num_input,
        num_hidden=num_hidden,
        num_output=num_output,
        device=device,
    ).to(device)
    print("\nNetwork Architecture:")
    print(net)

    # 1. Run simulation
    sim(net, num_input=num_input, timesteps=100, dt=1.0, batch_size=4, device=device)

    # 2. Run training
    train(
        net,
        num_input=num_input,
        num_output=num_output,
        timesteps=15,
        dt=1.0,
        batch_size=4,
        epochs=10,
        device=device,
    )


if __name__ == "__main__":
    main()
