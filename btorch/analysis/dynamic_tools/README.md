# Dynamic Analysis Tools

This module provides a comprehensive suite of tools for analyzing the dynamical properties of spiking neural networks. These metrics allow researchers to ascertain whether a network exhibits **criticality**, operates at the **edge of chaos**, or demonstrates high **complexity**—states often associated with optimal information processing in biological brains.

## 1. Criticality Analysis (`criticality.py`)
**"Is the brain poised at a critical point?"**

The Criticality Hypothesis suggests that the brain operates near a phase transition between ordered and disordered states. At this "critical point," information transmission, dynamic range, and computational capacity are maximized.

### Metrics
-   **Avalanche Analysis**: Neural activity often propagates in "avalanches." In critical systems, the size ($S$) and duration ($T$) of these avalanches follow power-law distributions:
    -   $P(S) \sim S^{-\tau}$
    -   $P(T) \sim T^{-\alpha}$
    -   Scaling Relation: $\langle S \rangle (T) \sim T^\gamma$, where $\gamma \approx (\alpha - 1) / (\tau - 1)$.
    -   **Usage**: `compute_avalanche_statistics(spike_train)` returns these exponents and the Correlation Coefficient of Criticality (CCC/DCC) to quantify how close the data is to scaling predictions.
-   **Detrended Fluctuation Analysis (DFA)**: Measures the long-range temporal correlations (LRTC) in the signal.
    -   **Usage**: `calculate_dfa(spike_train)` returns the Hurst exponent ($H$ or $\alpha_{DFA}$). $0.5 < \alpha < 1.0$ indicates scale-free LRTC, a hallmark of criticality.

## 2. Complexity & Information (`complexity.py`)
**"How rich and integrated is the network's state space?"**

Complexity measures quantify the diversity of patterns the network can generate and its ability to integrate information.

### Metrics
-   **Perturbational Complexity Index (PCIst)**: Inspired by measures of consciousness (PCI), this metric assesses the complexity of the network's *deterministic response* to a perturbation.
    -   **Usage**: `calculate_pcist(response, baseline)`. High PCIst implies the network is both integrated (signal spreads) and differentiated (signal is complex/non-trivial).
-   **Representation Alignment (RA)**: Measures how the internal geometry of neural representations changes over time (e.g., due to learning or drift).
    -   **Usage**: `calculate_ra(spike_initial, spike_final)`. High RA indicates stable preservation of relative similarity structures in the population code.
-   **Gain-Stability Sensitivity**: Analyzes how the Maximum Lyapunov Exponent ($\lambda_{max}$) changes with the global gain ($g$).
    -   **Usage**: `calculate_gain_stability_sensitivity`. The slope of this relationship indicates ease of transitioning to chaos.

## 3. Chaos & Attractors (`lyapunov_dynamics.py`, `attractor_dynamics.py`)
**"Is the system stable, chaotic, or at the edge of chaos?"**

These metrics characterize the stability of the network's trajectory in phase space.

### Metrics
-   **Lyapunov Exponents (LE)**: Measure the rate of separation of infinitesimaly close trajectories.
    -   $\lambda_{max} < 0$: Stable (Fixed point or Limit cycle).
    -   $\lambda_{max} > 0$: Chaotic.
    -   $\lambda_{max} \approx 0$: Edge of Chaos (common in biological networks).
    -   **Usage**: `compute_max_lyapunov_exponent` or `compute_lyapunov_exponent_spectrum`.
-   **Kaplan-Yorke Dimension ($D_{KY}$)**: The fractal dimension of the attractor, derived from the full Lyapunov spectrum. Higher $D_{KY}$ implies a higher-dimensional effective state space.
    -   **Usage**: `calculate_kaplan_yorke_dimension(spectrum)`.
-   **Structural Eigenvalues**: Analyzes the eigenspectrum of the connectivity matrix $W$.
    -   **Outliers**: Eigenvalues significantly outside the main spectral circle often correspond to structurally enforced modes (e.g., cell assemblies or memory patterns).
    -   **Usage**: `calculate_structural_eigenvalue_outliers(weight_matrix)`.

## 4. Micro-scale Dynamics (`micro_scale.py`)
**"What are the statistical properties of individual neurons?"**

These provide the foundational statistics of the population activity.

### Metrics
-   **Firing Rate Distribution**: Does the network show a log-normal firing rate distribution (common in cortex)?
    -   **Usage**: `calculate_fr_distribution`.
-   **Coefficient of Variation (CV) of ISI**: Measures spike irregularity.
    -   $CV \approx 1$: Poisson-like (irregular/noisy).
    -   $CV \ll 1$: Regular/Clock-like.
    -   $CV > 1$: Bursting.
    -   **Usage**: `calculate_cv_isi`.
-   **Spike Distance**: Quantifies similarity between spike trains (e.g., VP-distance or van Rossum).
    -   **Usage**: `calculate_spike_distance`.

---

## Ascertaining Brain Dynamics: A Cookbook

1.  **Check for Criticality**:
    -   Run `plot_avalanche_analysis`. Do you see straight lines on the log-log plots? Is the CCC high (> 0.8)?
    -   Run `calculate_dfa`. Is the exponent $\approx 0.75$?
    -   *Interpretation*: If yes, the network is likely optimizing information transmission.

2.  **Check for Edge of Chaos**:
    -   Run `plot_lyapunov_spectrum`. Is the largest exponent close to 0?
    -   Run `calculate_gain_stability_sensitivity`.
    -   *Interpretation*: Edge of Chaos supports both memory (stability) and computation (separability).

3.  **Check for Integration/Complexity**:
    -   Run `calculate_pcist` on perturbation data.
    -   *Interpretation*: High complexity suggests the network can support rich, diverse cognitive states.
