import warnings

import numpy as np
import torch

from .lyapunov_dynamics import (
    compute_max_lyapunov_exponent,
    get_continuous_spiking_rate,
)


def compute_ra(spike_initial: torch.Tensor, spike_final: torch.Tensor) -> float:
    """Calculate Representation Alignment (RA) using spike data.

    RA = Trace(G_final * G_initial) / (||G_final|| * ||G_initial||)
    where G = S * S^T (Gram matrix of spike activity)

    If inputs are 3D tensors, they are assumed to be (batch_size, time_steps,
    num_neurons) and will be averaged over the time dimension (dim=1) to obtain
    firing rates.

    Args:
        spike_initial (torch.Tensor): Initial spike activity. Shape
            ``(batch_size, num_neurons)`` or
            ``(batch_size, time_steps, num_neurons)``.
        spike_final (torch.Tensor): Final spike activity. Shape
            ``(batch_size, num_neurons)`` or
            ``(batch_size, time_steps, num_neurons)``.

    Returns:
        float: The Representation Alignment (RA) score.
               Low RA -> Rich Regime (Radical restructuring)
               High RA -> Lazy Regime (Little change in internal structure)
            Failure return: ``float("nan")`` (with a warning) when either
            Gram matrix is all zeros (no spikes), so the alignment is
            undefined.
    """
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

    # Assuming Frobenius norm as is standard for matrix alignment
    norm_initial = torch.norm(g_initial, p="fro")
    norm_final = torch.norm(g_final, p="fro")

    if norm_initial == 0 or norm_final == 0:
        # Silent activity: alignment is undefined, report NaN (not a fake 0).
        warnings.warn(
            "compute_ra: all-zero spike activity, alignment is undefined; "
            "returning NaN",
            stacklevel=2,
        )
        return float("nan")

    ra = numerator / (norm_final * norm_initial)

    return ra.item()


def compute_pcist(
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
            ``(time_steps, num_neurons)``.
        baseline (torch.Tensor): The baseline activity before perturbation.
            Shape ``(time_steps_base, num_neurons)``.
        threshold_factor (float): Factor of baseline std dev to define
            significant state excursion. Default 3.0.

    Returns:
        float: The PCIst score, or NaN (with a warning) if the SVD of the
            response fails to converge.
    """
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
            b_sample = baseline[i] if baseline.ndim == 3 else baseline
            pcist_values.append(compute_pcist(response[i], b_sample, threshold_factor))
        return sum(pcist_values) / len(pcist_values)

    mean_base = baseline.mean(dim=0)
    response_centered = response - mean_base
    baseline_centered = baseline - mean_base

    # 2. PCA on Response
    # We use SVD for PCA: X = U S V^T. Principal components (scores) are X V = U S.
    # response_centered shape: (T, N)
    try:
        # full_matrices=False ensures we get min(T, N) components
        U, S, Vh = torch.linalg.svd(response_centered, full_matrices=False)
    except RuntimeError as e:
        # SVD did not converge (e.g. non-finite input). Report NaN rather than
        # a fake score of 0.0, which would be indistinguishable from "no
        # complexity".
        warnings.warn(f"SVD failed in compute_pcist, returning NaN: {e}", stacklevel=2)
        return float("nan")

    V = Vh.T  # (N, K)

    # Project data onto Principal Components
    # Scores shape: (T, K)
    scores_response = torch.matmul(response_centered, V)
    scores_baseline = torch.matmul(baseline_centered, V)

    # SNR = Variance(Response) / Variance(Baseline)
    # Add epsilon to avoid division by zero
    var_response = scores_response.var(dim=0)
    var_baseline = scores_baseline.var(dim=0)

    epsilon = 1e-9
    snr = var_response / (var_baseline + epsilon)

    # A state transition is defined as crossing a threshold defined by baseline noise.
    # Threshold for component k: threshold_factor * std(baseline_k)

    std_baseline = scores_baseline.std(dim=0)
    thresholds = threshold_factor * std_baseline  # (K,)

    # We are looking for "significant excursions"
    # Shape: (T, K)
    active_states = (torch.abs(scores_response) > thresholds.unsqueeze(0)).float()

    # Count transitions: change from 0 to 1 or 1 to 0
    # diff along time dimension
    transitions = torch.abs(active_states[1:] - active_states[:-1])

    num_transitions = transitions.sum(dim=0)  # (K,)

    # 5. Weighted Sum "Sum significant state transitions weighted by the
    # component's Signal-to-Noise ratio." We might want to filter components
    # with SNR < 1? The text doesn't strictly say so, but "weighted by SNR"
    # implies low SNR components contribute little. However, if SNR is very
    # high, it dominates. Let's follow the instruction literally.

    pcist_score = (num_transitions * snr).sum()

    return pcist_score.item()


def compute_lyapunov_exponent(spike_train: torch.Tensor, dt: float = 0.1) -> float:
    """Calculate the maximum Lyapunov exponent for a given spike train.

    Args:
        spike_train (torch.Tensor): The spike train data. Shape
            ``(time_steps, num_neurons)``.
        dt (float): Time bin size in milliseconds. Default is 0.1 ms.

    Returns:
        float: The maximum Lyapunov exponent. Failure return: whatever
            :func:`compute_max_lyapunov_exponent` (``nolds.lyap_r``) yields for
            a degenerate series; this function adds no NaN sentinel of its own.

    Raises:
        ValueError: If ``spike_train`` is not 2D.
    """
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

    # 2. Calculate the Lyapunov exponent of the mean population rate (nolds
    # needs a 1D series). The largest Lyapunov exponent is the measure of
    # chaos/complexity. Embedding parameters keep their defaults.
    mean_rate = continuous_rate.mean(axis=1)
    lyapunov_exponent = compute_max_lyapunov_exponent(mean_rate)

    return lyapunov_exponent


def compute_gain_stability_sensitivity(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    g_values: np.ndarray | None = None,
    dt: float = 1.0,
    device: str = "cuda",
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Calculate the Gain-Stability Sensitivity (Susceptibility) slope.

    Definition: The slope of the curve of the Maximum Lyapunov Exponent (lambda_max)
    as a function of global synaptic gain scaling (g).

    The model must expose ``model.brain.synapse.linear.magnitude`` (the weight
    magnitude being scaled) and ``model.brain.neuron``. The model weights are
    always restored, even if the sweep raises.

    Args:
        model: The Brain model.
        dataloader: DataLoader providing input.
        g_values: Array of gain scaling factors. Default np.linspace(0.5, 5.0, 10).
        dt: Simulation time step.
        device: Device to run on.

    Returns:
        tuple: (slope, intercept, g_values, lambda_values)
            - slope: The slope of lambda_max vs g (NaN if fewer than two
              finite lambda values were obtained).
            - intercept: The intercept of the fit (NaN as above).
            - g_values: The gain values used.
            - lambda_values: The computed max Lyapunov exponents; NaN for gains
              where the estimate failed.

    Raises:
        AttributeError: If the model lacks ``brain.synapse.linear.magnitude``.
        ValueError: If the dataloader is empty.
    """
    # Local import: btorch.models is heavy and only needed for this routine.
    from btorch.models import functional, init

    if g_values is None:
        g_values = np.linspace(0.5, 5.0, 10)
    g_values = np.asarray(g_values, dtype=float)

    # Assuming model is Brain, model.brain is RecurrentNN, model.brain.synapse
    # is Synapse and model.brain.synapse.linear is the layer.
    try:
        linear_layer = model.brain.synapse.linear
        original_magnitude = linear_layer.magnitude.data.clone()
    except AttributeError as e:
        raise AttributeError(
            "model must expose model.brain.synapse.linear.magnitude"
        ) from e

    model.eval()
    model.to(device)

    try:
        batch = next(iter(dataloader))
    except StopIteration:
        raise ValueError("dataloader is empty") from None

    inputs = batch["input"]
    # inputs: (Batch, Time, ...) -> (Time, Batch, ...)
    inputs = inputs.transpose(0, 1).to(device)

    # We can use the first sample in the batch.
    input_sample = inputs[:, 0:1, ...]  # Keep batch dim 1

    lambda_values = []
    try:
        for g in g_values:
            linear_layer.magnitude.data = original_magnitude * g

            functional.reset_net(model, batch_size=1, device=device)
            init.uniform_v_(model.brain.neuron, set_reset_value=True)

            with torch.no_grad():
                _, brain_out = model(input_sample)
                spikes = brain_out["neuron"]["spike"]  # (Time, Batch, Neurons)

            # spikes: (Time, 1, Neurons) -> (Time, Neurons)
            spikes_sq = spikes.squeeze(1)

            rates = get_continuous_spiking_rate(spikes_sq, dt=dt)

            # Mean population rate for LE calculation
            mean_rate = rates.mean(axis=1)

            # Compute LE; nolds raises ValueError/RuntimeError on degenerate
            # series (e.g. too short or constant). Record NaN, not a fake 0.
            try:
                le = compute_max_lyapunov_exponent(mean_rate)
            except (ValueError, RuntimeError, ArithmeticError) as e:
                warnings.warn(f"Lyapunov exponent failed for g={g}: {e}", stacklevel=2)
                le = float("nan")

            lambda_values.append(le)
    finally:
        # Restore weights even if the sweep raised
        linear_layer.magnitude.data = original_magnitude

    lambda_arr = np.asarray(lambda_values, dtype=float)

    # Fit line: lambda = slope * g + intercept, ignoring NaN/Inf entries
    valid = np.isfinite(lambda_arr)
    if np.sum(valid) < 2:
        warnings.warn(
            "Fewer than two finite Lyapunov exponents; slope is NaN", stacklevel=2
        )
        return float("nan"), float("nan"), g_values, lambda_arr

    slope, intercept = np.polyfit(g_values[valid], lambda_arr[valid], 1)

    return float(slope), float(intercept), g_values, lambda_arr
