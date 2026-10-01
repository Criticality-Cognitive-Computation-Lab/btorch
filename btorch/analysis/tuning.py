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
) -> dict[str, torch.Tensor]:
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

    params = neuron_params.copy()
    n_neuron = params.pop("n_neuron", 1)
    params["device"] = device

    neuron = neuron_cls(n_neuron=n_neuron, **params)
    init_net_state(neuron, batch_size=steps, device=device)

    rnn_model = make_rnn(neuron, update_state_names=["v"])

    time_steps = int(duration / dt)
    input_current = currents[None, :, None].expand(time_steps, steps, n_neuron)

    with torch.no_grad():
        with environ.context(dt=dt):
            spikes, states = rnn_model(input_current)

    voltages = states["v"]

    spike_counts = spikes.sum(dim=0)  # (steps, n_neuron)
    frequencies = spike_counts / (duration / 1000.0)

    return {
        "currents": currents,
        "frequencies": frequencies,
        "voltages": voltages,
        "time": torch.arange(0, duration, dt, device=device),
    }
