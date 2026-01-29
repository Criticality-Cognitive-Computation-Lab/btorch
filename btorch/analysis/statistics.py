from typing import Literal

import numpy as np


def describe_array(array: np.ndarray):
    """Print descriptive statistics for a 1D array."""
    mean = np.mean(array)
    median = np.median(array)
    std_dev = np.std(array)
    min_val = np.min(array)
    max_val = np.max(array)
    q25 = np.percentile(array, 25)
    q50 = np.percentile(array, 50)  # This is the same as the median
    q75 = np.percentile(array, 75)

    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"25th Percentile (Q1): {q25}")
    print(f"50th Percentile (Q2/Median): {q50}")
    print(f"75th Percentile (Q3): {q75}")


def compute_log_hist(data, bins=1000, edge_pos: Literal["mid", "sep"] = "mid"):
    bin_edges = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), num=bins)
    hist, edges = np.histogram(data, bins=bin_edges)
    if edge_pos == "mid":
        bin_edges = 0.5 * (edges[:-1] + edges[1:])
    return hist, bin_edges


def get_corr_stats(n, max_dur, nrep, ncell, nframe):
    """
    n: spike activity of shape (nrep*ncell, nframe)
    max_dur: temporal correlation of [-max_dur, max_dur]
    """
    n = n.reshape((nrep, ncell, nframe))

    nframe = n.shape[-1]

    # Mean spike count across repetitions
    lam = np.mean(n, axis=0)
    # Mean spike count for each neuron
    mean_lambda = np.expand_dims(np.mean(lam, axis=1), axis=1)

    # Stimulus (PSTH) covariance
    z_lambda = lam - mean_lambda
    C_stim = np.zeros((ncell, ncell, max_dur * 2 + 1))
    for t in range(-max_dur, max_dur + 1):
        C_stim[:, :, t + max_dur] = (
            np.matmul(z_lambda, np.roll(z_lambda, t, axis=1).T) / nframe
        )

    # Total covariance
    z_n = n - [mean_lambda]
    C_tot = np.zeros((ncell, ncell, max_dur * 2 + 1))
    C_tot_temp = np.zeros((ncell, ncell, nrep))
    for t in range(-max_dur, max_dur + 1):
        for ri in range(nrep):
            C_tot_temp[:, :, ri] = (
                np.matmul(z_n[ri, :, :], np.roll(z_n[ri, :, :], t, axis=1).T) / nframe
            )
        C_tot[:, :, t + max_dur] = C_tot_temp.mean(2)

    # Noise covariance
    n_l = n - lam
    C_noise = np.zeros((ncell, ncell, max_dur * 2 + 1))
    nlnl = np.zeros((ncell, ncell, nrep))

    for t in range(-max_dur, max_dur + 1):
        n_l_shift = np.roll(n_l, t, axis=2)
        for ri in range(nrep):
            nlnl[:, :, ri] = np.matmul(n_l[ri, :, :], n_l_shift[ri, :, :].T) / nframe

        C_noise[:, :, t + max_dur] = nlnl.mean(2)

    stim_corr = []
    noise_corr = []

    for ci in range(ncell):
        for cj in range(ci):
            noise_corr.append(
                C_noise[ci, cj, max_dur]
                / np.sqrt(C_tot[ci, ci, max_dur] * C_tot[cj, cj, max_dur])
            )
            stim_corr.append(
                C_stim[ci, cj, max_dur]
                / np.sqrt(C_tot[ci, ci, max_dur] * C_tot[cj, cj, max_dur])
            )

            if noise_corr[-1] > 2:
                print(ci, cj)

    total_corr = [stim_corr[n] + noise_corr[n] for n in range(len(stim_corr))]

    return C_stim, C_tot, C_noise, stim_corr, total_corr, noise_corr
