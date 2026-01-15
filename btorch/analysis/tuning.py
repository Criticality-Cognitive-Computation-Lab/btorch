import torch

from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.rnn import make_rnn


def get_fi_vi_curve(
    neuron_cls,
    neuron_params,
    current_start=0.0,
    current_end=20.0,
    steps=20,
    duration=1000,
    dt=1.0,
    device="cpu",
):
    """Run sweeps to generate f-I and V-I curves.

    Args:
        neuron_cls: The neuron class to instantiate.
        neuron_params: Dictionary of parameters for the neuron.
        current_start: Starting current value for sweep.
        current_end: Ending current value for sweep.
        steps: Number of currents to sweep.
        duration: Duration of simulation in ms.
        dt: Time step in ms.
        device: Device to run on.

    Returns:
        dict: A dictionary containing:
            - currents: Tensor of shape (steps,)
            - frequencies: Tensor of shape (steps, n_neuron)
            - voltages: Tensor of shape (time_steps, steps, n_neuron)
    """
    currents = torch.linspace(current_start, current_end, steps, device=device)

    # Instantiate neuron with original population size
    params = neuron_params.copy()
    n_neuron = params.pop("n_neuron", 1)
    params["device"] = device

    # Instantiate the neuron model
    neuron = neuron_cls(n_neuron=n_neuron, **params)
    # Initialize state with batch_size = steps for parallel sweep
    init_net_state(neuron, batch_size=steps, device=device)

    # Wrap with RecurrentNN for time-stepping
    rnn_model = make_rnn(neuron, update_state_names=["v"])

    # Create input tensor: (Time, Batch, Neurons)
    time_steps = int(duration / dt)
    # expand currents (steps,) -> (time_steps, steps, n_neuron)
    # currents[None, :, None] gives (1, steps, 1)
    input_current = currents[None, :, None].expand(time_steps, steps, n_neuron)

    with torch.no_grad():
        with environ.context(dt=dt):
            spikes, states = rnn_model(input_current)

            # spikes: (Time, steps, n_neuron)
            # states['v']: (Time, steps, n_neuron)

    voltages = states["v"]

    # Calculate firing rates
    # Sum spikes over time, divide by duration (in seconds)
    spike_counts = spikes.sum(dim=0)  # (steps, n_neuron)
    frequencies = spike_counts / (duration / 1000.0)

    return {
        "currents": currents,
        "frequencies": frequencies,
        "voltages": voltages,
        "time": torch.arange(0, duration, dt, device=device),
    }
