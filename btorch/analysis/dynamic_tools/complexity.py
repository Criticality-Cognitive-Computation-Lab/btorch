import numpy as np
import torch

from .lyapunov_dynamics import (
    compute_max_lyapunov_exponent,
    get_continuous_spiking_rate,
)


def calculate_ra(spike_initial: torch.Tensor, spike_final: torch.Tensor) -> float:
    """Calculate Representation Alignment (RA) using spike data.

    RA = Trace(G_final * G_initial) / (||G_final|| * ||G_initial||)
    where G = S * S^T (Gram matrix of spike activity)

    If inputs are 3D tensors, they are assumed to be (batch_size, time_steps,
    num_neurons) and will be averaged over the time dimension (dim=1) to obtain
    firing rates.

    Args:
        spike_initial (torch.Tensor): Initial spike activity. Shape (batch_size,
        num_neurons) or (batch_size, time_steps, num_neurons). spike_final
        (torch.Tensor): Final spike activity. Shape (batch_size, num_neurons) or
        (batch_size, time_steps, num_neurons).

    Returns:
        float: The Representation Alignment (RA) score.
               Low RA -> Rich Regime (Radical restructuring)
               High RA -> Lazy Regime (Little change in internal structure)
    """
    # Ensure inputs are float tensors
    if not isinstance(spike_initial, torch.Tensor):
        spike_initial = torch.tensor(spike_initial, dtype=torch.float32)
    if not isinstance(spike_final, torch.Tensor):
        spike_final = torch.tensor(spike_final, dtype=torch.float32)

    spike_initial = spike_initial.float()
    spike_final = spike_final.float()

    # Handle 3D input: (batch, time, neurons) -> (batch, neurons)
    if spike_initial.ndim == 3:
        spike_initial = spike_initial.mean(dim=1)
    if spike_final.ndim == 3:
        spike_final = spike_final.mean(dim=1)

    # 1. Compute Gram matrix G = S * S^T
    # Shape: (batch_size, batch_size)
    g_initial = torch.matmul(spike_initial, spike_initial.T)
    g_final = torch.matmul(spike_final, spike_final.T)

    # 2. Calculate Trace(G_final * G_initial)
    # Trace(A @ B) = sum(element-wise product of A and B^T)
    # Since G is symmetric, G^T = G, so this is sum(G_final * G_initial)
    product = torch.matmul(g_final, g_initial)
    numerator = torch.trace(product)

    # 3. Calculate norms ||G||
    # Assuming Frobenius norm as is standard for matrix alignment
    norm_initial = torch.norm(g_initial, p="fro")
    norm_final = torch.norm(g_final, p="fro")

    # 4. Calculate RA
    if norm_initial == 0 or norm_final == 0:
        return 0.0  # Avoid division by zero

    ra = numerator / (norm_final * norm_initial)

    return ra.item()


def calculate_pcist(
    response: torch.Tensor, baseline: torch.Tensor, threshold_factor: float = 3.0
) -> float:
    """Calculate the Perturbational Complexity Index based on State Transitions
    (PCIst).

    Definition: A measure of the spatiotemporal complexity of the network's
    response to a specific perturbation.

    Steps:
    1. Perturb (assumed done, input is response).
    2. Measure (input is response matrix).
    3. Decompose: Perform PCA on the response matrix.
    4. Recurrence: Calculate state transitions on the principal components.
    5. Sum significant state transitions weighted by the component's
    Signal-to-Noise ratio.

    Args:
        response (torch.Tensor): The network response to perturbation. Shape
        (time_steps, num_neurons).
        baseline (torch.Tensor): The baseline
        activity before perturbation. Shape (time_steps_base, num_neurons).
        threshold_factor (float): Factor of baseline std dev to define
        significant state excursion. Default 3.0.

    Returns:
        float: The PCIst score.
    """
    # Ensure inputs are float tensors
    if not isinstance(response, torch.Tensor):
        response = torch.tensor(response, dtype=torch.float32)
    if not isinstance(baseline, torch.Tensor):
        baseline = torch.tensor(baseline, dtype=torch.float32)

    response = response.float()
    baseline = baseline.float()

    # Handle batch dimension: if 3D, calculate mean PCIst over batch or raise
    # error? For simplicity, if 3D, we assume (batch, time, neurons) and
    # calculate average PCIst.
    if response.ndim == 3:
        batch_size = response.shape[0]
        pcist_values = []
        for i in range(batch_size):
            # Handle corresponding baseline
            b_sample = baseline[i] if baseline.ndim == 3 else baseline
            pcist_values.append(
                calculate_pcist(response[i], b_sample, threshold_factor)
            )
        return sum(pcist_values) / len(pcist_values)

    # 1. Center data based on baseline mean
    mean_base = baseline.mean(dim=0)
    response_centered = response - mean_base
    baseline_centered = baseline - mean_base

    # 2. PCA on Response
    # We use SVD for PCA: X = U S V^T. Principal components (scores) are X V = U S.
    # response_centered shape: (T, N)
    try:
        # full_matrices=False ensures we get min(T, N) components
        U, S, Vh = torch.linalg.svd(response_centered, full_matrices=False)
    except RuntimeError:
        # Fallback for singular matrices or convergence issues
        return 0.0

    V = Vh.T  # (N, K)

    # Project data onto Principal Components
    # Scores shape: (T, K)
    scores_response = torch.matmul(response_centered, V)
    scores_baseline = torch.matmul(baseline_centered, V)

    # 3. Calculate SNR for each component
    # SNR = Variance(Response) / Variance(Baseline)
    # Add epsilon to avoid division by zero
    var_response = scores_response.var(dim=0)
    var_baseline = scores_baseline.var(dim=0)

    epsilon = 1e-9
    snr = var_response / (var_baseline + epsilon)

    # 4. Calculate State Transitions
    # A state transition is defined as crossing a threshold defined by baseline noise.
    # Threshold for component k: threshold_factor * std(baseline_k)

    std_baseline = scores_baseline.std(dim=0)
    thresholds = threshold_factor * std_baseline  # (K,)

    # Binarize response: 1 if |score| > threshold, else 0
    # We are looking for "significant excursions"
    # Shape: (T, K)
    active_states = (torch.abs(scores_response) > thresholds.unsqueeze(0)).float()

    # Count transitions: change from 0 to 1 or 1 to 0
    # diff along time dimension
    transitions = torch.abs(active_states[1:] - active_states[:-1])

    # Sum transitions for each component
    num_transitions = transitions.sum(dim=0)  # (K,)

    # 5. Weighted Sum "Sum significant state transitions weighted by the
    # component's Signal-to-Noise ratio." We might want to filter components
    # with SNR < 1? The text doesn't strictly say so, but "weighted by SNR"
    # implies low SNR components contribute little. However, if SNR is very
    # high, it dominates. Let's follow the instruction literally.

    pcist_score = (num_transitions * snr).sum()

    return pcist_score.item()


def calculate_lyapunov_exponent(spike_train: torch.Tensor, dt: float = 0.1) -> float:
    """Calculate the maximum Lyapunov exponent for a given spike train.

    Args:
        spike_train (torch.Tensor): The spike train data. Shape (time_steps,
        num_neurons).
        dt (float): Time bin size in milliseconds. Default is 0.1 ms.

    Returns:
        float: The maximum Lyapunov exponent.
    """
    # Ensure spike_train is a 2D tensor
    if spike_train.ndim != 2:
        raise ValueError(
            "spike_train must be a 2D tensor with shape (time_steps, num_neurons)"
        )

    # 1. Calculate the continuous spiking rate using a Gaussian kernel (smooth
    # the spike train) This is effectively a form of kernel density estimation.
    # We use a small bandwidth, as the original dynamics should be captured at a
    # fine timescale.
    bandwidth = 5.0  # in ms, this may need adjustment
    continuous_rate = get_continuous_spiking_rate(spike_train, dt, bandwidth)

    # 2. Calculate the Lyapunov exponent using the continuous rate
    # We use the largest Lyapunov exponent as the measure of chaos/complexity.
    lyapunov_exponent = compute_max_lyapunov_exponent(continuous_rate, dt)

    return lyapunov_exponent


def calculate_gain_stability_sensitivity(
    model, dataloader, g_values=None, dt=1.0, device="cuda"
) -> float:
    """Calculate the Gain-Stability Sensitivity (Susceptibility) slope.

    Definition: The slope of the curve of the Maximum Lyapunov Exponent (lambda_max)
    as a function of global synaptic gain scaling (g).

    Args:
        model: The Brain model.
        dataloader: DataLoader providing input.
        g_values: List of gain scaling factors. Default np.linspace(0.5, 5.0, 10).
        dt: Simulation time step.
        device: Device to run on.

    Returns:
        tuple: (slope, intercept, g_values, lambda_values)
            - slope: The slope of lambda_max vs g.
            - intercept: The intercept of the fit.
            - g_values: The gain values used.
            - lambda_values: The computed max Lyapunov exponents.
    """
    from model import functional, init

    if g_values is None:
        g_values = np.linspace(0.5, 5.0, 10)

    # Access linear layer
    # Assuming model is Brain, model.brain is RecurrentNN, model.brain.synapse
    # is Synapse model.brain.synapse.linear is the layer
    try:
        linear_layer = model.brain.synapse.linear
    except AttributeError:
        print("Could not find linear layer at model.brain.synapse.linear")
        return 0.0

    original_magnitude = linear_layer.magnitude.data.clone()

    lambda_values = []

    model.eval()
    model.to(device)

    # Get one batch of input
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print("Dataloader is empty.")
        return 0.0

    inputs = batch["input"]
    # inputs: (Batch, Time, ...) -> (Time, Batch, ...)
    inputs = inputs.transpose(0, 1).to(device)

    # We can use the first sample in the batch.
    input_sample = inputs[:, 0:1, ...]  # Keep batch dim 1

    for g in g_values:
        # Scale weights
        linear_layer.magnitude.data = original_magnitude * g

        # Reset state
        functional.reset_net(model, device=device)
        init.uniform_v_(model.brain.neuron, set_reset_value=True, batch_size=1)

        # Run
        with torch.no_grad():
            _, brain_out = model(input_sample)
            spikes = brain_out["neuron"]["spike"]  # (Time, Batch, Neurons)

        # Convert to rate
        # spikes: (Time, 1, Neurons) -> (Time, Neurons)
        spikes_sq = spikes.squeeze(1)

        # Continuous rate
        rates = get_continuous_spiking_rate(spikes_sq, dt=dt)

        # Mean population rate for LE calculation
        mean_rate = rates.mean(axis=1)

        # Compute LE
        try:
            le = compute_max_lyapunov_exponent(mean_rate)
        except Exception as e:
            print(f"Error computing LE for g={g}: {e}")
            le = 0.0  # Or NaN?

        lambda_values.append(le)

    # Restore weights
    linear_layer.magnitude.data = original_magnitude

    # Calculate slope
    # Fit line: lambda = slope * g + intercept
    # Handle potential NaNs or Infs
    valid_indices = np.isfinite(lambda_values)
    if np.sum(valid_indices) < 2:
        return 0.0

    g_valid = g_values[valid_indices]
    lambda_valid = np.array(lambda_values)[valid_indices]

    slope, intercept = np.polyfit(g_valid, lambda_valid, 1)

    return slope, intercept, g_values, np.array(lambda_values)
