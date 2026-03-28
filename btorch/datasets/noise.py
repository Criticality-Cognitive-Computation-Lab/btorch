"""Noise generation utilities for spiking neural network simulations.

This module provides functional and stateful (layer-based) noise generators
for creating temporally structured input noise, background activity, and
synaptic noise in SNNs.

Physical Units
--------------
All time-related parameters use consistent units based on ``dt``:

- ``tau``: Time constant in same units as ``dt`` (typically milliseconds)
- ``sigma``: Standard deviation in arbitrary units matching the output
- ``rate``: Events per unit time (Hz if dt is in seconds, kHz if dt is ms)
- ``dt``: Simulation time step (default usually from ``environ.get("dt")``)

Shape Conventions
-----------------
- Functional API: Output shape is ``(T, *size)`` where ``size`` is the
  per-timestep shape (e.g., ``(B, N)`` for batch x neurons).
- Layer API multi-step: Returns ``(T, *n_neuron)``.
- Layer API single-step: Returns ``(*n_neuron)``.

Stateful Behavior
-----------------
Layer classes (``OUNoiseLayer``, ``PoissonNoiseLayer``, ``PinkNoiseLayer``)
maintain internal state between calls when ``stateful=True``:

- OU: Stores current noise value (carries over between sequences)
- Poisson: Stores last sampled value
- Pink: Stores white noise history for FIR continuity

State is preserved across ``multi_step_forward`` calls and updated at the
end of each sequence (last timestep becomes initial state for next call).
Use ``reset()`` (inherited from ``MemoryModule``) to reinitialize state.

Determinism
-----------
All generators accept an optional ``torch.Generator`` for reproducible
sampling. Note that multi-step OU uses vectorized convolution which may
produce slightly different values than sequential single-step calls due
to floating-point ordering.
"""

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from btorch.models import base, environ


def _unflatten_td(seq: Tensor, rest_shape: Sequence[int]) -> Tensor:
    """Convert [D, T] to [T, *rest_shape] where D = prod(rest_shape)."""
    if seq.ndim != 2:
        raise ValueError(f"Expected 2D [D, T] tensor, got {seq.shape}.")
    t = seq.shape[1]
    return seq.transpose(0, 1).reshape((t,) + rest_shape)


def randn_like(
    like: Tensor, generator: torch.Generator | None = None, **kwargs
) -> Tensor:
    """Generate standard normal noise matching a reference tensor's metadata.

    Args:
        like: Reference tensor providing shape, device, dtype.
        generator: Optional RNG generator for deterministic sampling.
        **kwargs: Additional args passed to ``torch.empty_like``.

    Returns:
        Tensor with same shape/device/dtype as ``like``, filled with N(0,1).
    """
    return torch.empty_like(like).normal_(generator=generator, **kwargs)


def ou_noise(
    *size: int,
    sigma: Tensor,
    tau: Tensor,
    T: int,
    dt: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    noise0: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate Ornstein-Uhlenbeck (OU) noise sequence.

    OU noise follows the stochastic differential equation:
        dx = -x/tau * dt + sigma * sqrt(2/tau) * dW

    The exact discretization used is:
        n_{t+1} = alpha * n_t + beta * eps_t
        alpha = exp(-dt/tau)
        beta = sigma * sqrt(1 - exp(-2*dt/tau))
        eps_t ~ N(0, 1)

    Args:
        *size: Shape of the noise per timestep (e.g., ``B, N`` for batch,
            neurons). Output will be ``(T, *size)``.
        sigma: Standard deviation of the stationary distribution. Can be
            scalar or per-element (broadcastable to ``size``).
        tau: Time constant controlling correlation length. Can be scalar or
            per-element (broadcastable to ``size``). Same units as ``dt``.
        T: Number of timesteps to generate.
        dt: Simulation timestep (same units as ``tau``).
        device: Device for the output tensor (if ``noise0`` not provided).
        dtype: Dtype for the output tensor (if ``noise0`` not provided).
        noise0: Initial noise state with shape ``size``. If provided, ``size``
            must match or be empty. Defaults to N(0,1) sample if None.
        generator: Optional RNG generator for deterministic sampling.

    Returns:
        Tensor of shape ``(T, *size)`` containing the OU noise sequence.

    Raises:
        ValueError: If neither ``size`` nor ``noise0`` is provided.
        RuntimeError: If ``sigma`` or ``tau`` cannot broadcast to ``size``.
    """
    if noise0 is None:
        if len(size) == 0:
            raise ValueError("Provide size or noise0.")
        noise0 = torch.randn(size, device=device, dtype=dtype, generator=generator)
    elif len(size) != 0:
        if tuple(size) != tuple(noise0.shape):
            raise ValueError(f"size={size} does not match noise0.shape={noise0.shape}.")

    alpha = torch.exp(-dt / tau)
    beta = sigma * torch.sqrt(1.0 - torch.exp(-2.0 * dt / tau))

    rest_shape = noise0.shape
    D = noise0.numel()

    device = noise0.device
    dtype = noise0.dtype

    noise0_flat = noise0.reshape(-1)
    eps = torch.randn((D, T), device=device, dtype=dtype, generator=generator)

    beta_flat = beta.reshape(-1)
    if beta_flat.numel() == 1:
        beta_flat = beta_flat.expand(D)
    elif beta_flat.numel() != D:
        msg = (
            f"beta has {beta_flat.numel()} elems but D={D}; "
            "sigma must be broadcastable to noise."
        )
        raise RuntimeError(msg)

    u = beta_flat[:, None] * eps  # [D,T]

    if alpha.numel() == 1:
        a = alpha.reshape(1)

        powers = torch.arange(T - 1, -1, -1, device=device, dtype=dtype)
        w = torch.pow(a.to(dtype=dtype), powers).view(1, 1, T)  # [1,1,T]

        u_pad = F.pad(u.unsqueeze(1), (T - 1, 0))
        n_from_u = F.conv1d(u_pad, w).squeeze(1)

        factors = torch.pow(
            a.to(dtype=dtype),
            torch.arange(1, T + 1, device=device, dtype=dtype),
        )
        n_seq = n_from_u + noise0_flat[:, None] * factors[None, :]

    else:
        alpha_flat = alpha.reshape(-1)
        if alpha_flat.numel() == 1:
            alpha_flat = alpha_flat.expand(D)
        elif alpha_flat.numel() != D:
            msg = (
                f"alpha has {alpha_flat.numel()} elems but D={D}; "
                "tau must be broadcastable to noise."
            )
            raise RuntimeError(msg)

        u_ch = u.unsqueeze(0)
        u_pad = F.pad(u_ch, (T - 1, 0))

        powers = torch.arange(T - 1, -1, -1, device=device, dtype=dtype)
        w = torch.pow(alpha_flat[:, None], powers[None, :])  # [D,T]
        w = w[:, None, :]

        n_from_u = F.conv1d(u_pad, w, groups=D).squeeze(0)

        tpow = torch.arange(1, T + 1, device=device, dtype=dtype)
        factors = torch.pow(alpha_flat[:, None], tpow[None, :])
        n_seq = n_from_u + noise0_flat[:, None] * factors

    return _unflatten_td(n_seq, rest_shape)


def ou_noise_like(
    like: Tensor,
    sigma: Tensor,
    tau: Tensor,
    *,
    T: int,
    dt: float,
    noise0: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate OU noise matching a reference tensor's shape and device.

    Convenience wrapper around ``ou_noise`` that infers ``size``, ``device``,
    and ``dtype`` from a reference tensor.

    Args:
        like: Reference tensor with shape ``(*batch, *neuron)`` that defines
            the per-timestep shape. Output will be ``(T, *like.shape)``.
        sigma: Standard deviation (scalar or broadcastable to ``like.shape``).
        tau: Time constant (scalar or broadcastable to ``like.shape``).
        T: Number of timesteps.
        dt: Simulation timestep.
        noise0: Optional initial state with shape ``like.shape``.
        generator: Optional RNG generator.

    Returns:
        OU noise tensor of shape ``(T, *like.shape)``.
    """
    if noise0 is None:
        noise0 = randn_like(like, generator=generator)
    return ou_noise(
        sigma=sigma,
        tau=tau,
        T=T,
        dt=dt,
        noise0=noise0,
        generator=generator,
    )


def poisson_noise(
    *size: int,
    rate: float | Tensor,
    T: int,
    dt: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate Poisson noise (discrete event counts).

    Samples are drawn from Poisson distribution with lambda = rate * dt
    for each timestep and element. The output represents event counts per
    timestep (0, 1, 2, ...).

    Args:
        *size: Shape per timestep (e.g., ``B, N``). Output is ``(T, *size)``.
        rate: Event rate per unit time. Can be scalar or per-element
            (broadcastable to ``size``).
        T: Number of timesteps.
        dt: Simulation timestep (scales the rate: lambda = rate * dt).
        device: Device for the output tensor.
        dtype: Dtype for the output (must be floating point).
        generator: Optional RNG generator for deterministic sampling.

    Returns:
        Event count tensor of shape ``(T, *size)`` with dtype ``float``.

    Raises:
        ValueError: If ``size`` is empty, ``T < 0``, or dtype is not floating.
        ValueError: If ``rate * dt`` is negative.
    """
    if len(size) == 0:
        raise ValueError("Provide output size for poisson_noise.")
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}.")

    if dtype is None:
        sample_dtype = torch.get_default_dtype()
    else:
        sample_dtype = dtype
    if not torch.empty((), dtype=sample_dtype).is_floating_point():
        raise ValueError("poisson_noise requires a floating dtype.")

    base = torch.zeros(size, device=device, dtype=sample_dtype)
    lam = torch.as_tensor(rate, device=base.device, dtype=sample_dtype) * float(dt)
    if torch.any(lam < 0):
        raise ValueError("Poisson rate * dt must be non-negative.")

    lam_full = base + lam
    lam_seq = lam_full.unsqueeze(0).expand((T,) + tuple(size))
    return torch.poisson(lam_seq, generator=generator)


def poisson_noise_like(
    like: Tensor,
    rate: float | Tensor,
    *,
    T: int,
    dt: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate Poisson noise matching a reference tensor's metadata.

    Args:
        like: Reference tensor defining per-timestep shape ``(*size)``.
            Output will be ``(T, *like.shape)``.
        rate: Event rate per unit time (scalar or broadcastable).
        T: Number of timesteps.
        dt: Simulation timestep.
        generator: Optional RNG generator.

    Returns:
        Poisson event counts of shape ``(T, *like.shape)``.
    """
    dtype = like.dtype if like.is_floating_point() else torch.get_default_dtype()
    return poisson_noise(
        *like.shape,
        rate=rate,
        T=T,
        dt=dt,
        device=like.device,
        dtype=dtype,
        generator=generator,
    )


def _pink_fir_kernel(
    fir_order: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Build a deterministic FIR kernel for 1/f-like PSD.

    Uses truncated fractional-integration coefficients:
        h[0] = 1
        h[k] = h[k-1] * (k - 1 + alpha) / k,  alpha=0.5

    This gives |H(f)| ~ f^-alpha at low frequency, so PSD ~ 1/f for alpha=0.5.

    Args:
        fir_order: Length of the FIR kernel (>= 1).
        device: Device for the kernel tensor.
        dtype: Dtype for the kernel (must be floating point).

    Returns:
        Normalized FIR kernel of shape ``(fir_order,)``.
    """
    if fir_order < 1:
        raise ValueError(f"fir_order must be >= 1, got {fir_order}.")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("Pink-noise FIR kernel requires floating dtype.")

    alpha = 0.5
    kernel = torch.empty(fir_order, device=device, dtype=dtype)
    kernel[0] = 1.0
    for k in range(1, fir_order):
        kernel[k] = kernel[k - 1] * ((k - 1 + alpha) / k)

    kernel = kernel / kernel.norm(p=2).clamp_min(torch.finfo(dtype).eps)
    return kernel


def _white_noise_2d(
    template: Tensor, T: int, generator: torch.Generator | None = None
) -> Tensor:
    """Generate white noise with shape [D, T] matching template metadata."""
    D = template.numel()
    if generator is None:
        return torch.randn((D, T), device=template.device, dtype=template.dtype)

    out = torch.empty((D, T), device=template.device, dtype=template.dtype)
    for t in range(T):
        out[:, t] = randn_like(template, generator=generator).reshape(-1)
    return out


def _apply_fir_2d(
    white: Tensor, kernel: Tensor, history: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Apply a causal FIR filter to [D, T] white noise.

    Args:
        white: White noise of shape ``(D, T)``.
        kernel: FIR kernel of shape ``(fir_order,)``.
        history: Optional previous white noise history of shape
            ``(D, fir_order-1)`` for continuity across calls.

    Returns:
        Tuple of (filtered_noise, new_history) where filtered_noise has
        shape ``(D, T)`` and new_history has shape ``(D, fir_order-1)``.
    """
    D, _ = white.shape
    fir_order = kernel.numel()
    hist_len = fir_order - 1

    if history is None:
        history = torch.zeros((D, hist_len), device=white.device, dtype=white.dtype)
    else:
        if history.shape != (D, hist_len):
            raise ValueError(
                f"Expected history shape {(D, hist_len)}, got {history.shape}."
            )

    if hist_len > 0:
        seq = torch.cat([history, white], dim=1)
    else:
        seq = white

    # conv1d performs cross-correlation, so we flip to obtain causal FIR.
    weight = kernel.flip(0).view(1, 1, fir_order)
    out = F.conv1d(seq.unsqueeze(1), weight).squeeze(1)

    if hist_len > 0:
        new_history = seq[:, -hist_len:]
    else:
        new_history = seq[:, :0]
    return out, new_history


def pink_noise(
    *size: int,
    T: int,
    fir_order: int = 64,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    white_history: Tensor | None = None,
    generator: torch.Generator | None = None,
    return_white_history: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Generate pink (1/f) noise using a causal FIR filter.

    Pink noise has power spectral density proportional to 1/frequency,
    creating naturalistic temporal correlations. Generated by filtering
    white noise through a fractional integration FIR kernel.

    Args:
        *size: Shape per timestep. Output will be ``(T, *size)``.
        T: Number of timesteps to generate.
        fir_order: Length of the FIR filter kernel (default 64). Higher
            values give better low-frequency approximation but more state.
        device: Device for the output tensor (if ``white_history`` not given).
        dtype: Dtype for the output (must be floating point).
        white_history: Optional previous white noise history with shape
            ``(*size, fir_order-1)`` for continuity across calls.
        generator: Optional RNG generator for deterministic sampling.
        return_white_history: If True, also return the updated history tensor
            for stateful usage.

    Returns:
        Pink noise tensor of shape ``(T, *size)``. If ``return_white_history``
        is True, returns ``(noise, history)`` where history has shape
        ``(*size, fir_order-1)``.

    Raises:
        ValueError: If ``T < 0``, ``fir_order < 1``, or dtype not floating.
        ValueError: If ``size`` conflicts with ``white_history`` shape.
    """
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}.")
    if fir_order < 1:
        raise ValueError(f"fir_order must be >= 1, got {fir_order}.")

    hist_len = fir_order - 1
    if white_history is not None:
        if white_history.ndim < 1:
            raise ValueError("white_history must have at least one dimension.")
        if white_history.shape[-1] != hist_len:
            raise ValueError(
                f"white_history last dim must be {hist_len}, "
                f"got {white_history.shape[-1]}."
            )
        rest_shape = tuple(white_history.shape[:-1])
        sample_device = white_history.device
        sample_dtype = white_history.dtype
        history_flat = white_history.reshape(-1, hist_len)
        if len(size) != 0 and tuple(size) != rest_shape:
            raise ValueError(
                f"size={size} does not match white_history shape {rest_shape}."
            )
    else:
        if len(size) == 0:
            raise ValueError("Provide size or white_history for pink_noise.")
        rest_shape = tuple(size)
        sample_device = device or torch.device("cpu")
        sample_dtype = dtype or torch.get_default_dtype()
        history_flat = None

    if not torch.empty((), dtype=sample_dtype).is_floating_point():
        raise ValueError("pink_noise requires a floating dtype.")

    template = torch.empty(rest_shape, device=sample_device, dtype=sample_dtype)
    white = _white_noise_2d(template, T=T, generator=generator)
    kernel = _pink_fir_kernel(fir_order, device=sample_device, dtype=sample_dtype)
    out_flat, new_hist_flat = _apply_fir_2d(white, kernel, history_flat)
    out = _unflatten_td(out_flat, rest_shape)

    if not return_white_history:
        return out
    new_history = new_hist_flat.reshape(rest_shape + (hist_len,))
    return out, new_history


def pink_noise_like(
    like: Tensor,
    *,
    T: int,
    fir_order: int = 64,
    white_history: Tensor | None = None,
    generator: torch.Generator | None = None,
    return_white_history: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Generate pink noise matching a reference tensor's metadata.

    Args:
        like: Reference tensor with shape ``(*size)``. Output will be
            ``(T, *like.shape)``.
        T: Number of timesteps.
        fir_order: FIR filter length.
        white_history: Optional history tensor with shape
            ``(*like.shape, fir_order-1)``.
        generator: Optional RNG generator.
        return_white_history: If True, also return updated history.

    Returns:
        Pink noise of shape ``(T, *like.shape)``, or ``(noise, history)``
        tuple if ``return_white_history=True``.
    """
    dtype = like.dtype if like.is_floating_point() else torch.get_default_dtype()
    return pink_noise(
        *like.shape,
        T=T,
        fir_order=fir_order,
        device=like.device,
        dtype=dtype,
        white_history=white_history,
        generator=generator,
        return_white_history=return_white_history,
    )


class OUNoiseLayer(base.MemoryModule):
    """Ornstein-Uhlenbeck (OU) noise layer for temporally correlated noise.

    Implements exact discretization of the OU process where ``sigma`` is the
    stationary standard deviation:

        n_{t+1} = alpha * n_t + beta * eps_t
        alpha = exp(-dt/tau)
        beta  = sigma * sqrt(1 - exp(-2*dt/tau))
        eps_t ~ N(0, 1)

    The layer supports both single-step (stateful) and multi-step (vectorized)
    modes. In stateful mode, the noise state persists across forward calls.

    Tensor Conventions:
        - Multi-step input: Output shape is ``(T, *batch_dims, *n_neuron)``
          where neuron dims are trailing.
        - Single-step input: Output shape is ``(*batch_dims, *n_neuron)``.
        - Internal state ``self.noise`` stores the state BEFORE the current
          step/sequence (i.e., the initial condition).

    Multi-step Backend:
        - Scalar tau/sigma: Uses single conv1d (fast)
        - Per-neuron tau/sigma: Uses grouped conv1d, O(T^2 * D) complexity

    Determinism Note:
        Multi-step uses vectorized RNG and convolution, so exact equality with
        repeated single-step updates is not guaranteed even with a generator.
        Use single-step loops if you need step-by-step equivalence.

    Args:
        n_neuron: Number of neurons or shape of trailing neuron dims.
        sigma: Stationary standard deviation (scalar or per-neuron).
        tau: Time constant in same units as dt (scalar or per-neuron).
        step_mode: ``'s'`` for single-step, ``'m'`` for multi-step.
        trainable: Whether sigma and tau are learnable parameters.
        stateful: If True, maintain noise state between calls (required for
            single-step mode).
        tau_min: Minimum value for tau (clamped for numerical stability).

    Attributes:
        noise: Current noise state tensor (only if ``stateful=True``).
        sigma: Standard deviation (Parameter if trainable, else buffer).
        tau: Time constant (Parameter if trainable, else buffer).
    """

    noise: Tensor

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        sigma: float | Tensor = 0.5,
        tau: float | Tensor = 10.0,
        step_mode: Literal["s", "m"] = "m",
        trainable: bool = False,
        *,
        stateful: bool = False,
        tau_min: float = 1e-6,
    ):
        super().__init__()
        if not stateful and step_mode == "s":
            raise ValueError("stateful must be True for single-step mode.")
        self.step_mode = step_mode
        self.stateful = stateful
        self.n_neuron, self.size = base.normalize_n_neuron(n_neuron)
        self.tau_min = float(tau_min)

        # Memory state: stored as "n_0" (state BEFORE current step/sequence).
        # Will be expanded lazily to match x[0] if register_memory does not
        # include batch dims.
        if stateful:
            self.register_memory("noise", 0.0, self.n_neuron)

        sigma_t = torch.as_tensor(sigma)
        tau_t = torch.as_tensor(tau)

        if trainable:
            self.sigma = nn.Parameter(sigma_t)
            self.tau = nn.Parameter(tau_t)
        else:
            self.register_buffer("sigma", sigma_t)
            self.register_buffer("tau", tau_t)

    def single_step_forward(
        self, dt: float | None = None, *, generator: torch.Generator | None = None
    ) -> Tensor:
        """Single-step update of OU noise.

        Uses ``self.noise`` for device, dtype, and shape. Returns the updated
        noise tensor (no input ``x`` is required or used).

        Args:
            dt: Timestep (defaults to ``environ.get("dt")``).
            generator: Optional RNG generator.

        Returns:
            Updated noise tensor with same shape as ``self.noise``.
        """
        assert self.stateful, "single_step_forward requires stateful=True"

        sigma, tau = self.sigma, self.tau
        dt: float = dt if dt is not None else environ.get("dt")

        alpha = torch.exp(-dt / tau)
        beta = sigma * torch.sqrt(1.0 - torch.exp(-2.0 * dt / tau))

        self.noise = alpha * self.noise + beta * randn_like(
            self.noise, generator=generator
        )
        return self.noise

    # ---------------- public multi-step ----------------

    def multi_step_forward(
        self,
        T: int,
        dt: float | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate a multi-step OU noise sequence.

        Args:
            T: Number of timesteps.
            dt: Timestep (defaults to ``environ.get("dt")``).
            generator: Optional RNG generator.

        Returns:
            Noise sequence of shape ``(T, *noise_shape)`` where ``noise_shape``
            matches ``self.noise.shape``.

        Raises:
            RuntimeError: If stateful but noise buffer not initialized.
        """
        if T == 0:
            return torch.empty(
                (0,) + tuple(self.noise.shape),
                device=self.noise.device,
                dtype=self.noise.dtype,
            )

        if not hasattr(self, "noise"):
            raise RuntimeError(
                "OUNoiseLayer: noise buffer is not initialized. Call reset(...) first."
            )

        dt = dt if dt is not None else environ.get("dt")
        out = ou_noise(
            sigma=self.sigma,
            tau=self.tau,
            T=T,
            dt=dt,
            noise0=self.noise,
            generator=generator,
        )

        if self.stateful:
            # Update self.noise to the final state after the sequence.
            self.noise = out[-1]
        return out

    def extra_repr(self) -> str:
        return (
            f"n_neuron={self.n_neuron}, sigma={self._format_repr_value(self.sigma)}, "
            f"tau={self._format_repr_value(self.tau)}, step_mode={self.step_mode}, "
            f"tau_min={self.tau_min}"
        )


class PoissonNoiseLayer(base.MemoryModule):
    """Poisson noise layer for discrete event generation.

    Generates Poisson-distributed event counts with rate scaled by ``dt``:
    ``lambda = rate * dt`` per timestep. Supports both single-step stateful
    sampling and vectorized multi-step generation.

    In stateful mode, the last sampled value is preserved as ``self.noise``
    and used as the initial state for subsequent calls.

    Args:
        n_neuron: Number of neurons or shape of trailing neuron dims.
        rate: Events per unit time (scalar or per-neuron, broadcastable).
        step_mode: ``'s'`` for single-step, ``'m'`` for multi-step.
        trainable: Whether rate is a learnable parameter.
        stateful: If True, maintain state between calls (required for
            single-step mode).

    Attributes:
        noise: Last sampled value (only if ``stateful=True``).
        rate: Event rate (Parameter if trainable, else buffer).
    """

    noise: Tensor

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        rate: float | Tensor = 1.0,
        step_mode: Literal["s", "m"] = "m",
        trainable: bool = False,
        *,
        stateful: bool = False,
    ):
        super().__init__()
        if not stateful and step_mode == "s":
            raise ValueError("stateful must be True for single-step mode.")
        self.step_mode = step_mode
        self.stateful = stateful
        self.n_neuron, self.size = base.normalize_n_neuron(n_neuron)

        if stateful:
            self.register_memory("noise", 0.0, self.n_neuron)

        rate_t = torch.as_tensor(rate)
        if trainable:
            self.rate = nn.Parameter(rate_t)
        else:
            self.register_buffer("rate", rate_t)

    def single_step_forward(
        self, dt: float | None = None, *, generator: torch.Generator | None = None
    ) -> Tensor:
        """Single-step Poisson sampling.

        Args:
            dt: Timestep (defaults to ``environ.get("dt")``).
            generator: Optional RNG generator.

        Returns:
            Event counts tensor with same shape as ``self.noise``.
        """
        assert self.stateful, "single_step_forward requires stateful=True"
        dt = dt if dt is not None else environ.get("dt")

        lam = torch.as_tensor(
            self.rate, device=self.noise.device, dtype=self.noise.dtype
        )
        lam = lam * float(dt)
        if torch.any(lam < 0):
            raise ValueError("Poisson rate * dt must be non-negative.")
        lam = torch.zeros_like(self.noise) + lam

        self.noise = torch.poisson(lam, generator=generator)
        return self.noise

    def multi_step_forward(
        self,
        T: int,
        dt: float | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Vectorized multi-step Poisson sampling.

        Args:
            T: Number of timesteps.
            dt: Timestep (defaults to ``environ.get("dt")``).
            generator: Optional RNG generator.

        Returns:
            Event counts of shape ``(T, *noise_shape)``.
        """
        if T == 0:
            return torch.empty(
                (0,) + tuple(self.noise.shape),
                device=self.noise.device,
                dtype=self.noise.dtype,
            )
        if not hasattr(self, "noise"):
            raise RuntimeError(
                "PoissonNoiseLayer: noise buffer is not initialized. "
                "Call init_state(...) first."
            )

        dt = dt if dt is not None else environ.get("dt")
        out = poisson_noise(
            *self.noise.shape,
            rate=self.rate,
            T=T,
            dt=dt,
            device=self.noise.device,
            dtype=self.noise.dtype,
            generator=generator,
        )
        if self.stateful:
            self.noise = out[-1]
        return out

    def extra_repr(self) -> str:
        return (
            f"n_neuron={self.n_neuron}, rate={self._format_repr_value(self.rate)}, "
            f"step_mode={self.step_mode}"
        )


class PinkNoiseLayer(base.MemoryModule):
    """Pink (1/f) noise layer using causal FIR filtering.

    Generates colored noise with PSD ~ 1/frequency by filtering white noise
    through a fractional integration FIR kernel. Supports both single-step
    (stateful, with history preservation) and multi-step (vectorized) modes.

    In stateful mode, the FIR history is preserved across calls for seamless
    continuation of noise sequences.

    Args:
        n_neuron: Number of neurons or shape of trailing neuron dims.
        fir_order: Length of the FIR filter kernel (default 64).
        step_mode: ``'s'`` for single-step, ``'m'`` for multi-step.
        stateful: If True, maintain FIR history between calls (required for
            single-step mode).

    Attributes:
        noise: Current noise value (only if ``stateful=True``).
        white_history: Previous white noise samples for FIR continuity
            (shape ``(*n_neuron, fir_order-1)``).
        fir_order: Length of the FIR kernel.
    """

    noise: Tensor
    white_history: Tensor

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        fir_order: int = 64,
        step_mode: Literal["s", "m"] = "m",
        *,
        stateful: bool = False,
    ):
        super().__init__()
        if not stateful and step_mode == "s":
            raise ValueError("stateful must be True for single-step mode.")
        if fir_order < 1:
            raise ValueError(f"fir_order must be >= 1, got {fir_order}.")

        self.step_mode = step_mode
        self.stateful = stateful
        self.fir_order = int(fir_order)
        self.n_neuron, self.size = base.normalize_n_neuron(n_neuron)

        if stateful:
            self.register_memory("noise", 0.0, self.n_neuron)
            self.register_memory(
                "white_history", 0.0, self.n_neuron + (self.fir_order - 1,)
            )

    def single_step_forward(
        self, *, generator: torch.Generator | None = None
    ) -> Tensor:
        """Single-step pink-noise update using FIR history.

        Args:
            generator: Optional RNG generator.

        Returns:
            Single noise sample with shape ``(*n_neuron)``.
        """
        assert self.stateful, "single_step_forward requires stateful=True"
        out, new_hist = pink_noise(
            T=1,
            fir_order=self.fir_order,
            white_history=self.white_history,
            generator=generator,
            return_white_history=True,
        )
        self.noise = out[0]
        self.white_history = new_hist
        return self.noise

    def multi_step_forward(
        self,
        T: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Vectorized multi-step pink-noise generation.

        Args:
            T: Number of timesteps.
            generator: Optional RNG generator.

        Returns:
            Noise sequence of shape ``(T, *n_neuron)``.
        """
        if T == 0:
            return torch.empty(
                (0,) + tuple(self.noise.shape),
                device=self.noise.device,
                dtype=self.noise.dtype,
            )
        if not hasattr(self, "noise") or not hasattr(self, "white_history"):
            raise RuntimeError(
                "PinkNoiseLayer memories are not initialized. "
                "Call init_state(...) first."
            )

        out, new_hist = pink_noise(
            T=T,
            fir_order=self.fir_order,
            white_history=self.white_history,
            generator=generator,
            return_white_history=True,
        )
        if self.stateful:
            self.noise = out[-1]
            self.white_history = new_hist
        return out

    def extra_repr(self) -> str:
        return (
            f"n_neuron={self.n_neuron}, fir_order={self.fir_order}, "
            f"step_mode={self.step_mode}"
        )
