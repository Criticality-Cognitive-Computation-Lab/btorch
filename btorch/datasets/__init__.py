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
    - :func:`~btorch.datasets.noise.ou_noise`: Generate OU noise sequence
    - :func:`~btorch.datasets.noise.ou_noise_like`: OU noise with
      reference tensor
    - :func:`~btorch.datasets.noise.poisson_noise`: Generate Poisson
      events
    - :func:`~btorch.datasets.noise.poisson_noise_like`: Poisson with
      reference tensor
    - :func:`~btorch.datasets.noise.pink_noise`: Generate pink noise
    - :func:`~btorch.datasets.noise.pink_noise_like`: Pink noise with
      reference tensor

Layer API
---------
    - :class:`~btorch.datasets.noise.OUNoiseLayer`: Stateful OU noise
      module with single/multi-step modes
    - :class:`~btorch.datasets.noise.PoissonNoiseLayer`: Stateless Poisson
      encoder/generator module
    - :class:`~btorch.datasets.noise.PinkNoiseLayer`: Stateful pink noise
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
