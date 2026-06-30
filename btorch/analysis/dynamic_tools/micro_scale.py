import numpy as np
import torch
from scipy.stats import kurtosis, skew


def calculate_fr_distribution(
    spikes: np.ndarray | torch.Tensor,
    dt: float = 1.0,
) -> dict:
    """Compute the mean population firing rate at each timestep and
    characterize its distribution.

    Args:
        spikes: Spike matrix of shape ``(Time, Neurons)``.
        dt: Simulation time step in ms.

    Returns:
        Dictionary with keys ``rates``, ``mean``, ``skew``, ``kurt``.
    """
    if isinstance(spikes, torch.Tensor):
        spikes = spikes.detach().cpu().numpy()

    window_size = 5
    kernel = np.ones(window_size) / (window_size * dt / 1000.0)  # convert to Hz
    pop_spikes = spikes.mean(axis=1)  # (T,)
    rates = np.convolve(pop_spikes, kernel, mode="same")

    return {
        "rates": rates,
        "mean": np.mean(rates),
        "skew": skew(rates),
        "kurt": kurtosis(rates),
    }


def calculate_cv_isi(
    spikes: np.ndarray | torch.Tensor,
    dt: float = 1.0,
) -> dict:
    """Compute the CV of the ISI for each neuron in the population and
    characterize its distribution.

    Args:
        spikes: Spike matrix of shape ``(Time, Neurons)``.
        dt: Simulation time step in ms.

    Returns:
        Dictionary with keys ``cv_isi`` and ``mean``.
    """
    if isinstance(spikes, torch.Tensor):
        spikes = spikes.detach().cpu().numpy()

    num_neurons = spikes.shape[1]
    cv_isi_list = []

    for n in range(num_neurons):
        spike_times = np.where(spikes[:, n] > 0)[0] * dt  # convert to ms
        if len(spike_times) < 2:
            cv_isi_list.append(np.nan)  # fewer than two spikes: ISI undefined
            continue

        isis = np.diff(spike_times)
        if np.mean(isis) == 0:
            cv_isi_list.append(np.nan)
            continue

        cv_isi = np.std(isis) / np.mean(isis)
        cv_isi_list.append(cv_isi)

    cv_isi_array = np.array(cv_isi_list)
    mean_cv_isi = np.nanmean(cv_isi_array)  # ignore NaNs when averaging

    return {
        "cv_isi": cv_isi_array,
        "mean": mean_cv_isi,
    }


def calculate_spike_distance(
    spikes: np.ndarray | torch.Tensor,
    dt: float = 1.0,
    subset_size: int = 100,
    seed: "int | None" = None,
) -> float:
    """Compute the SPIKE-distance (Kreuz et al., 2013).

    Measures the degree of asynchrony between spike trains. Zero means fully
    synchronous.

    Args:
        spikes: Spike matrix of shape ``(Time, Neurons)``.
        dt: Simulation time step in ms.
        subset_size: Number of randomly sampled neurons for pairwise distance.
        seed: Random seed for neuron sampling.

    Returns:
        Mean SPIKE-distance across all neuron pairs.
    """
    if isinstance(spikes, torch.Tensor):
        spikes = spikes.detach().cpu().numpy()

    T_steps, N = spikes.shape
    times = np.arange(T_steps) * dt

    # random subsampling to keep pairwise computation tractable
    if N > subset_size:
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.choice(N, subset_size, replace=False)
        selected_spikes = spikes[:, indices]
        N_subset = subset_size
    else:
        selected_spikes = spikes
        N_subset = N

    # precompute t_prev, t_next, and isi for each neuron; shape: (N_subset, T_steps)
    t_prev = np.zeros((N_subset, T_steps))
    t_next = np.zeros((N_subset, T_steps))
    isi = np.zeros((N_subset, T_steps))

    for n in range(N_subset):
        spike_indices = np.where(selected_spikes[:, n] > 0)[0]
        spike_times = spike_indices * dt

        if len(spike_times) == 0:
            # no spikes: span the full interval as a boundary fallback
            t_prev[n, :] = 0
            t_next[n, :] = times[-1]
            isi[n, :] = times[-1]
            continue

        # Use a forward scan for t_prev = max(s | s <= t)
        # and a backward scan for t_next = min(s | s >= t).
        # searchsorted alone does not handle ties correctly at spike times,
        # so we fill with explicit linear scans instead.

        # t_prev
        curr_spike = 0.0
        spike_idx = 0
        for t_idx, t in enumerate(times):
            if spike_idx < len(spike_times) and t >= spike_times[spike_idx]:
                curr_spike = spike_times[spike_idx]
                # advance index if we've reached the next spike
                if spike_idx < len(spike_times) - 1 and t >= spike_times[spike_idx + 1]:
                    spike_idx += 1
                    curr_spike = spike_times[spike_idx]
            t_prev[n, t_idx] = curr_spike

        # t_next
        curr_spike = times[-1]
        spike_idx = len(spike_times) - 1
        for t_idx in range(T_steps - 1, -1, -1):
            t = times[t_idx]
            if spike_idx >= 0 and t <= spike_times[spike_idx]:
                curr_spike = spike_times[spike_idx]
                if spike_idx > 0 and t <= spike_times[spike_idx - 1]:
                    spike_idx -= 1
                    curr_spike = spike_times[spike_idx]
            t_next[n, t_idx] = curr_spike

        isi[n, :] = t_next[n, :] - t_prev[n, :]

        isi[n, isi[n, :] == 0] = dt

    # Pairwise SPIKE-distance:
    # S(t) = (|dt_p1-dt_p2|*isi2 + |dt_f1-dt_f2|*isi1) / (0.5*(isi1+isi2)^2)
    dt_p = times[None, :] - t_prev  # (N, T)
    dt_f = t_next - times[None, :]  # (N, T)

    pairwise_distances = []

    # All pairs: with N_subset=50, this is 1225 pairs — acceptable cost.
    for i in range(N_subset):
        for j in range(i + 1, N_subset):
            isi1 = isi[i]
            isi2 = isi[j]

            avg_isi_sq = 0.5 * (isi1 + isi2) ** 2
            avg_isi_sq[avg_isi_sq == 0] = 1.0  # avoid division by zero

            term1 = np.abs(dt_p[i] - dt_p[j]) * isi2
            term2 = np.abs(dt_f[i] - dt_f[j]) * isi1

            s_t = (term1 + term2) / avg_isi_sq

            dist = np.mean(s_t)  # integrate over time
            pairwise_distances.append(dist)

    if not pairwise_distances:
        return 0.0

    return np.mean(pairwise_distances)
