"""Dataset utilities and noise generation for neuromorphic simulations.

This module provides noise generators (functional and layer-based) commonly
used for simulating background activity, synaptic noise, and input currents
in spiking neural networks.

Noise Types
-----------
**Ornstein-Uhlenbeck (OU)**:
    Temporally correlated Gaussian noise with configurable time constant
    (``tau``) and standard deviation (``sigma``). Useful for modeling
    synaptic noise and membrane potential fluctuations.

**Poisson**:
    Discrete event noise with configurable rate. Suitable for spike train
    generation and stochastic synaptic inputs.

**Pink (1/f)**:
    Colored noise with power spectral density proportional to 1/frequency.
    Generated via causal FIR filtering of white noise. Useful for modeling
    naturalistic temporal correlations.

Functional API
--------------
    - [`ou_noise`](btorch/datasets/noise.py:25): Generate OU noise sequence
    - [`ou_noise_like`](btorch/datasets/noise.py:123): OU noise with
      reference tensor
    - [`poisson_noise`](btorch/datasets/noise.py:156): Generate Poisson
      events
    - [`poisson_noise_like`](btorch/datasets/noise.py:192): Poisson with
      reference tensor
    - [`pink_noise`](btorch/datasets/noise.py:284): Generate pink noise
    - [`pink_noise_like`](btorch/datasets/noise.py:344): Pink noise with
      reference tensor

Layer API
---------
    - [`OUNoiseLayer`](btorch/datasets/noise.py:367): Stateful OU noise
      module with single/multi-step modes
    - [`PoissonNoiseLayer`](btorch/datasets/noise.py:510): Stateless Poisson
      encoder/generator module
    - [`PinkNoiseLayer`](btorch/datasets/noise.py:603): Stateful pink noise
      module with FIR history

All noise functions support:
    - Per-neuron or scalar parameters (broadcastable)
    - Deterministic sampling via ``torch.Generator``
    - GPU/CPU device placement
    - Compatible with ``torch.compile``
"""

from btorch.datasets.noise import (
    OUNoiseLayer,
    PinkNoiseLayer,
    PoissonNoiseLayer,
    ou_noise,
    ou_noise_like,
    pink_noise,
    pink_noise_like,
    poisson_noise,
    poisson_noise_like,
)


__all__ = [
    "OUNoiseLayer",
    "PinkNoiseLayer",
    "PoissonNoiseLayer",
    "ou_noise",
    "ou_noise_like",
    "pink_noise",
    "pink_noise_like",
    "poisson_noise",
    "poisson_noise_like",
]
