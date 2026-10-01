from collections.abc import Iterable, Sequence
from typing import Protocol

import pandas as pd
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from ..types import TensorLike
from . import environ
from .base import (
    MemoryModule,
    flatten_neuron,
    normalize_n_neuron,
    unflatten_neuron,
)
from .bilinear import SymmetricBilinear
from .history import SpikeHistory
from .ode import exp_euler_step


class Synapse(Protocol):
    """Minimum Synapse interface."""

    # TODO: rework spikingjelly's synapse abstraction
    n_neuron: tuple[int, ...]
    size: int
    psc: torch.Tensor

    def __call__(self, x): ...


class BasePSC(MemoryModule):
    """Base class for post-synaptic current models.

    Provides infrastructure for synaptic dynamics including weight
    application and PSC state management. Delay handling is managed
    externally (e.g. via :class:`DelayedPSC`).

    Timing convention: a spike delivered to :meth:`single_step_forward` at
    step ``t`` already changes the returned PSC at step ``t`` for every
    subclass (the first response is at the delivery step, delay 0). The
    impulse response ``k[t]`` returned by ``get_kernel`` therefore starts at
    ``t = 0`` with a non-zero value and :meth:`multi_step_forward` equals
    stepping :meth:`single_step_forward` exactly.

    Args:
        n_neuron: Number of post-synaptic neurons.
        linear: Linear layer for weight application.
        step_mode: Step mode. Default: "s".
        backend: Compute backend. Default: "torch".
    """

    n_neuron: tuple[int, ...]
    size: int
    psc: torch.Tensor

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        linear: torch.nn.Module,
        step_mode: str = "s",
        backend: str = "torch",
    ):
        super().__init__()

        self.n_neuron, self.size = normalize_n_neuron(n_neuron)
        self.step_mode = step_mode
        self.backend = backend
        self.linear = linear

        self.register_memory("psc", 0.0, self.n_neuron)

    def extra_repr(self) -> str:
        return f"step_mode={self.step_mode}, backend={self.backend}"

    def conductance_charge(self) -> None:
        raise NotImplementedError()

    def adaptation_charge(self, z: torch.Tensor) -> None:
        # Flatten only when input still carries multi-dimensional neuron dims
        z_flat, leading = flatten_neuron(z, self.n_neuron, self.size)
        wz = self.linear(z_flat)
        wz = unflatten_neuron(wz, leading, self.n_neuron)
        self.psc = self.psc + wz

    def current_charge(self, v: Tensor | None = None) -> Tensor:
        if v is not None:
            raise NotImplementedError(
                "Only current-based PSC is supported."
                "Conductance-based PSC requires voltage from post-syn neurons, "
                "which the current abstraction doesn't support."
            )
        else:
            return self.psc

    def single_step_forward(self, z: torch.Tensor):
        self.conductance_charge()
        self.adaptation_charge(z)
        current = self.current_charge()
        return current

    def multi_step_forward(
        self, z_seq: torch.Tensor, kernel_len: int = 64
    ) -> torch.Tensor:
        """Full-sequence forward via grouped 1D conv.

        Extends :meth:`MemoryModule.multi_step_forward` (whose ``*args, **kwargs``
        are subclass-defined) with the optional ``kernel_len`` truncation. The
        sequence is the first positional argument, as in the base signature.

        Args:
            z_seq: (T, *batch, *n_neuron) spike sequence
            kernel_len: length of the PSC impulse response kernel (truncation).
                Default: 64.

        Returns:
            (T, *batch, *n_neuron) PSC sequence.
        """
        dt = environ.get("dt")

        z_flat, leading = flatten_neuron(z_seq, self.n_neuron, self.size)
        wz_seq = self.linear(z_flat)

        kernel = self.get_kernel(dt, kernel_len)
        kernel = kernel.to(wz_seq.device, wz_seq.dtype)

        wz_channels = wz_seq.reshape(wz_seq.shape[0], -1).transpose(0, 1).unsqueeze(0)
        n_channels = wz_channels.shape[1]

        depthwise_kernel = (
            kernel.flip(0).view(1, 1, kernel_len).repeat(n_channels, 1, 1)
        )
        wz_padded = F.pad(wz_channels, (kernel_len - 1, 0))
        out_flat = F.conv1d(wz_padded, depthwise_kernel, groups=n_channels)
        out_flat = out_flat.squeeze(0).transpose(0, 1)

        out = out_flat.reshape(*leading, self.size)
        return unflatten_neuron(out, leading, self.n_neuron)


class ExponentialPSC(BasePSC):
    """Exponential decay synapse model.

    Simple first-order synapse with single exponential decay:
        d(psc)/dt = -psc / tau_syn

    Args:
        n_neuron: Number of neurons.
        tau_syn: Synaptic time constant (ms).
        linear: Linear layer for weights.
        step_mode: Step mode. Default: "s".
        backend: Compute backend. Default: "torch".
    """

    tau_syn: torch.Tensor | torch.nn.Parameter

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        tau_syn: float | TensorLike,
        linear,
        step_mode: str = "s",
        backend: str = "torch",
    ):
        super().__init__(
            n_neuron,
            linear,
            step_mode=step_mode,
            backend=backend,
        )

        self.register_buffer("tau_syn", torch.as_tensor(tau_syn), persistent=False)

    def dpsc(self, psc: Tensor) -> tuple[Tensor, Tensor]:
        derivative = -psc / self.tau_syn
        linear = -1.0 / self.tau_syn
        return derivative, linear

    def conductance_charge(self) -> Tensor:
        self.psc = exp_euler_step(self.dpsc, self.psc, dt=environ.get("dt"))
        return self.psc

    def get_kernel(self, dt: float | Tensor, kernel_len: int) -> Tensor:
        """Exponential decay kernel.

        k[t] = a^t for t >= 0 where a = exp(-dt/tau_syn).
        """
        a = torch.exp(-dt / self.tau_syn)
        t = torch.arange(kernel_len, dtype=a.dtype, device=a.device)
        return a**t


class _Adaptive2VarPSC(BasePSC):
    h: torch.Tensor

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        linear,
        step_mode: str = "s",
        backend: str = "torch",
    ):
        super().__init__(n_neuron, linear, step_mode=step_mode, backend=backend)

        self.register_memory("h", 0.0, self.n_neuron)


class AlphaPSCBilleh(_Adaptive2VarPSC):
    tau_syn: torch.Tensor | torch.nn.Parameter
    syn_decay: torch.Tensor

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        tau_syn: float | TensorLike,
        linear: torch.nn.Module,
        step_mode: str = "s",
        backend: str = "torch",
    ):
        """The Current-Based Alpha form of PSC, from [1], ensuring a post-
        synaptic current with synapse weight W = 1.0 has an amplitude of 1.0 pA
        at its peak (``tau_syn`` steps after the spike, counting the delivery
        step as the first). A spike acts in its delivery step.

        NOTE: this model assumes environ.get("dt") == 1.0

        [1] Billeh, Y. N. et al. Systematic integration of structural and
        functional data into multi-scale models of mouse primary visual
        cortex. 662189 Preprint at https://doi.org/10.1101/662189 (2019).

        :param tau_syn: the synaptic time constant
        :type tau_syn: float or torch.Tensor
        """

        super().__init__(n_neuron, linear, step_mode, backend)

        dt = environ.get("dt", None)
        if dt is not None and dt != 1.0:
            raise ValueError(f"dt must be 1.0 for this model, got {dt}")

        self.register_buffer("tau_syn", torch.as_tensor(tau_syn), persistent=False)

        self.register_buffer(
            "syn_decay", torch.exp(-1.0 / self.tau_syn), persistent=False
        )

    def conductance_charge(self) -> Tensor:
        # Only decay the rise variable here; the input is injected into ``h``
        # (and ``psc`` advanced from the updated ``h``) in adaptation_charge so
        # that a spike acts in its delivery step.
        self.h = self.syn_decay * self.h
        return self.psc

    def adaptation_charge(self, z: torch.Tensor) -> None:
        # Flatten only when input still carries multi-dimensional neuron dims
        if len(self.n_neuron) > 1 and z.shape[-len(self.n_neuron) :] == self.n_neuron:
            z_flat, leading = flatten_neuron(z, self.n_neuron, self.size)
        else:
            z_flat = z
            leading = z.shape[:-1]
        wz = self.linear(z_flat)
        if len(self.n_neuron) > 1 and z_flat is not z:
            wz = unflatten_neuron(wz, leading, self.n_neuron)
        self.h = self.h + torch.e / self.tau_syn * wz
        self.psc = self.syn_decay * self.psc + self.syn_decay * self.h

    def get_kernel(self, dt: float | Tensor, kernel_len: int) -> Tensor:
        """AlphaPSC Billeh variant kernel.

        Kernel follows the exact single-step recurrence
        ``h_t = a h_{t-1} + c w z_t``, ``psc_t = a psc_{t-1} + a h_t`` with
        ``c = e/tau_syn``, so a spike acts in its delivery step:
        k[t] = (t + 1) * e/tau_syn * a^(t+1) for t >= 0,
        where a = exp(-1/tau_syn). The peak (1.0 for unit weight) is reached
        at kernel index ``tau_syn - 1`` (``tau_syn`` steps after the spike
        counting the delivery step as the first).

        NOTE: dt is assumed to be 1.0 for this model (enforced in __init__).
        """
        a = self.syn_decay
        t = torch.arange(kernel_len, dtype=a.dtype, device=a.device)
        return (t + 1) * (torch.e / self.tau_syn) * (a ** (t + 1))


class AlphaPSC(_Adaptive2VarPSC):
    tau_syn: torch.Tensor | torch.nn.Parameter
    g_max: torch.Tensor | torch.nn.Parameter

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        tau_syn: float | TensorLike,
        linear: torch.nn.Module,
        g_max=1.0,
        step_mode: str = "s",
        backend: str = "torch",
    ):
        """The Alpha form (current-based) of PSC, from Brainpy/BrainState."""

        super().__init__(n_neuron, linear, step_mode, backend)

        self.register_buffer("tau_syn", torch.as_tensor(tau_syn), persistent=False)
        self.register_buffer("g_max", torch.as_tensor(g_max), persistent=False)

    def dg(self, psc: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        derivative = -psc / self.tau_syn + h / self.tau_syn
        linear = -1.0 / self.tau_syn
        return derivative, linear

    def dh(self, h: Tensor) -> tuple[Tensor, Tensor]:
        derivative = -h / self.tau_syn
        linear = -1.0 / self.tau_syn
        return derivative, linear

    def conductance_charge(self) -> None:
        # Only decay ``h`` here; ``psc`` is advanced from the input-updated
        # ``h`` in adaptation_charge so a spike acts in its delivery step.
        self.h = exp_euler_step(self.dh, self.h, dt=environ.get("dt"))

    def adaptation_charge(self, z: torch.Tensor) -> None:
        # Flatten only when input still carries multi-dimensional neuron dims
        if len(self.n_neuron) > 1 and z.shape[-len(self.n_neuron) :] == self.n_neuron:
            z_flat, leading = flatten_neuron(z, self.n_neuron, self.size)
        else:
            z_flat = z
            leading = z.shape[:-1]
        wz = self.g_max * self.linear(z_flat)
        if len(self.n_neuron) > 1 and z_flat is not z:
            wz = unflatten_neuron(wz, leading, self.n_neuron)
        self.h = self.h + wz
        self.psc = exp_euler_step(self.dg, self.psc, self.h, dt=environ.get("dt"))

    def get_kernel(self, dt: float | Tensor, kernel_len: int) -> Tensor:
        """AlphaPSC (Brainpy variant) kernel.

        Kernel follows the exact single-step recurrence
        ``h_t = a h_{t-1} + g_max w z_t``,
        ``psc_t = a psc_{t-1} + (1 - a) h_t`` so a spike acts in its
        delivery step:
        k[t] = g_max * (t + 1) * (1 - a) * a^t for t >= 0,
        where a = exp(-dt/tau_syn).
        """
        a = torch.exp(-dt / self.tau_syn)
        t = torch.arange(kernel_len, dtype=a.dtype, device=a.device)
        kernel = (t + 1) * (1 - a) * (a**t)
        return self.g_max.to(kernel.dtype) * kernel


class DualExponentialPSC(BasePSC):
    tau_rise: torch.Tensor | torch.nn.Parameter
    tau_decay: torch.Tensor | torch.nn.Parameter
    a: torch.Tensor
    g_rise: torch.Tensor
    g_decay: torch.Tensor

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        tau_decay: float | TensorLike,
        tau_rise: float | TensorLike,
        linear: torch.nn.Module,
        A: float | TensorLike | None = None,
        step_mode: str = "s",
        backend: str = "torch",
    ):
        """The Double Exponential form of PSC, from Brainpy/BrainState."""

        super().__init__(
            n_neuron=n_neuron,
            linear=linear,
            step_mode=step_mode,
            backend=backend,
        )

        self.register_buffer("tau_decay", torch.as_tensor(tau_decay), persistent=False)
        self.register_buffer("tau_rise", torch.as_tensor(tau_rise), persistent=False)

        if A is None:
            A = (
                self.tau_decay
                / (self.tau_decay - self.tau_rise)
                * (self.tau_rise / self.tau_decay)
                ** (self.tau_rise / (self.tau_rise - self.tau_decay))
            )
        A = torch.as_tensor(A)
        a = (self.tau_decay - self.tau_rise) / self.tau_rise / self.tau_decay * A
        self.register_buffer("a", a, persistent=False)

        self.register_memory("g_rise", 0.0, self.n_neuron)
        self.register_memory("g_decay", 0.0, self.n_neuron)

    def dg_rise(self, g_rise: Tensor) -> tuple[Tensor, Tensor]:
        derivative = -g_rise / self.tau_rise
        linear = -1.0 / self.tau_rise
        return derivative, linear

    def dg_decay(self, g_decay: Tensor) -> tuple[Tensor, Tensor]:
        derivative = -g_decay / self.tau_decay
        linear = -1.0 / self.tau_decay
        return derivative, linear

    def conductance_charge(self) -> None:
        self.g_rise = exp_euler_step(self.dg_rise, self.g_rise, dt=environ.get("dt"))
        self.g_decay = exp_euler_step(self.dg_decay, self.g_decay, dt=environ.get("dt"))

    def adaptation_charge(self, z: torch.Tensor) -> None:
        # Flatten only when input still carries multi-dimensional neuron dims
        if len(self.n_neuron) > 1 and z.shape[-len(self.n_neuron) :] == self.n_neuron:
            z_flat, leading = flatten_neuron(z, self.n_neuron, self.size)
        else:
            z_flat = z
            leading = z.shape[:-1]
        wz = self.linear(z_flat)
        if len(self.n_neuron) > 1 and z_flat is not z:
            wz = unflatten_neuron(wz, leading, self.n_neuron)
        self.g_rise = self.g_rise + wz
        self.g_decay = self.g_decay + wz
        # The analytic response a*(e^{-t/tau_d} - e^{-t/tau_r}) is zero at the
        # injection instant, so psc is read out one dt after the input
        # (``g`` advanced by one decay step) to make the spike act in its
        # delivery step; the stored ``g`` stay at the injection instant.
        dt = environ.get("dt")
        a_r = torch.exp(-dt / self.tau_rise)
        a_d = torch.exp(-dt / self.tau_decay)
        self.psc = self.a * (a_d * self.g_decay - a_r * self.g_rise)

    def get_kernel(self, dt: float | Tensor, kernel_len: int) -> Tensor:
        """Dual-exponential (alpha-shaped) kernel.

        Kernel: k[t] = a * (a_d^(t+1) - a_r^(t+1)) for t >= 0 so a spike acts
        in its delivery step, where a = self.a, a_d = exp(-dt/tau_decay),
        a_r = exp(-dt/tau_rise).
        """
        a_r = torch.exp(-dt / self.tau_rise)
        a_d = torch.exp(-dt / self.tau_decay)
        t = torch.arange(kernel_len, dtype=self.a.dtype, device=self.a.device)
        return self.a * (a_d ** (t + 1) - a_r ** (t + 1))


class BilinearMixingSynapse(MemoryModule):
    """PSC with bilinear + linear mixing across receptor/input dimensions.

    Used as the dendritic stage for both DLIF (delta/exponential PSC)
    and DBNN (dual-exponential/alpha PSC).

    Dynamics (per timestep):
        PSC per receptor: base_psc.single_step_forward(z[..., n_receptor])
        Mix: out = bilinear(psc_per_receptor) + psc_per_receptor.sum(dim=-1)

    Args:
        n_neuron: Number of output neurons.
        n_receptor: Number of input receptors (input dimension D).
        base_psc: BasePSC subclass instance for synaptic dynamics.
        bilinear_mask: Optional mask passed to SymmetricBilinear.
        kernel_len: Default kernel length for multistep conv. Default: 64.
    """

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        n_receptor: int,
        base_psc: BasePSC,
        bilinear_mask: float | Tensor | None = None,
        kernel_len: int = 64,
    ):
        super().__init__()
        self.n_neuron, self.size = normalize_n_neuron(n_neuron)
        self.n_receptor = n_receptor
        self.base_psc = base_psc
        self.kernel_len = kernel_len

        self.bilinear = SymmetricBilinear(
            in_features=n_receptor,
            out_features=1,
            bias=True,
            mask=bilinear_mask,
        )

    @property
    def psc(self) -> torch.Tensor:
        return self._psc

    @psc.setter
    def psc(self, value: torch.Tensor):
        self._psc = value

    def init_state(
        self,
        batch_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        persistent: bool = True,
        skip_mem_name: Iterable[str] = (),
    ) -> None:
        self.base_psc.init_state(
            batch_size,
            dtype,
            device,
            persistent,
            skip_mem_name=skip_mem_name,
        )
        self._psc = torch.zeros(
            *((batch_size,) if batch_size is not None else ()),
            *self.n_neuron,
            dtype=dtype,
            device=device,
        )

    def reset(
        self,
        batch_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        skip_mem_name: Iterable[str] = (),
    ) -> None:
        self.base_psc.reset(
            batch_size,
            dtype,
            device,
            skip_mem_name=skip_mem_name,
        )
        self._psc = torch.zeros(
            *((batch_size,) if batch_size is not None else ()),
            *self.n_neuron,
            dtype=dtype,
            device=device,
        )

    def single_step_forward(self, z: torch.Tensor):
        z_flat, leading = flatten_neuron(z, self.n_neuron, self.size)
        z_expanded = z_flat.reshape(*leading, self.size * self.n_receptor)
        psc_expanded = self.base_psc.single_step_forward(z_expanded)
        psc_per_receptor = psc_expanded.reshape(
            *leading, *self.n_neuron, self.n_receptor
        )
        bilinear_term = self.bilinear(psc_per_receptor)
        linear_term = psc_per_receptor.sum(dim=-1, keepdim=True)
        self._psc = (bilinear_term + linear_term).squeeze(-1)
        return self._psc

    def multi_step_forward(
        self, z_seq: torch.Tensor, kernel_len: int | None = None
    ) -> torch.Tensor:
        """Multi-receptor forward; ``kernel_len`` defaults to
        ``self.kernel_len``."""
        if kernel_len is None:
            kernel_len = self.kernel_len
        T, *batch_shape, n_neuron, n_receptor = z_seq.shape
        leading = (*batch_shape,)

        z_expanded = z_seq.reshape(T, *leading, self.size * n_receptor)
        psc_flat = self.base_psc.multi_step_forward(z_expanded, kernel_len=kernel_len)

        psc_per_receptor = psc_flat.reshape(T, *leading, n_neuron, n_receptor)
        bilinear_term = self.bilinear(psc_per_receptor)
        linear_term = psc_per_receptor.sum(dim=-1, keepdim=True)
        out = (bilinear_term + linear_term).squeeze(-1)
        return out


class DelayedPSC(MemoryModule):
    """Wrapper that adds delay buffering to any BasePSC subclass.

    Delays are managed orthogonally to synaptic dynamics via SpikeHistory.
    Delay is configured here via ``max_delay_steps``; BasePSC subclasses no
    longer take a delay/latency argument themselves.

    Args:
        psc: BasePSC subclass instance (e.g. ExponentialPSC, AlphaPSC).
        max_delay_steps: Maximum delay steps to buffer. Default: 1.
        use_circular_buffer: If False (default), use torch.cat for
            torch.compile compatibility. If True, use circular buffer
            for memory-efficient simulation.

    Example:
        >>> psc = AlphaPSC(n_neuron=100, tau_syn=5.0, linear=linear)
        >>> delayed = DelayedPSC(psc, max_delay_steps=5)
    """

    def __init__(
        self,
        psc: BasePSC,
        max_delay_steps: int = 1,
        use_circular_buffer: bool = False,
    ):
        super().__init__()
        self.psc_module = psc
        self.max_delay_steps = max_delay_steps
        self.use_circular_buffer = use_circular_buffer

        if max_delay_steps > 1:
            self.history = SpikeHistory(
                n_neuron=psc.n_neuron,
                max_delay_steps=max_delay_steps + 1,
                use_circular_buffer=use_circular_buffer,
            )
        else:
            self.history = None

    @property
    def n_neuron(self) -> tuple[int, ...]:
        return self.psc_module.n_neuron

    @property
    def size(self) -> int:
        return self.psc_module.size

    @property
    def step_mode(self) -> str:
        return self.psc_module.step_mode

    @step_mode.setter
    def step_mode(self, value: str):
        self.psc_module.step_mode = value

    @property
    def backend(self) -> str:
        return self.psc_module.backend

    @backend.setter
    def backend(self, value: str):
        self.psc_module.backend = value

    @property
    def psc(self) -> torch.Tensor:
        return self.psc_module.psc

    def single_step_forward(self, z: torch.Tensor):
        if self.history is not None:
            self.history.update(z)
            z_delayed = self.history.get_delay(self.max_delay_steps)
        else:
            z_delayed = z
        return self.psc_module.single_step_forward(z_delayed)

    def multi_step_forward(
        self, z_seq: torch.Tensor, kernel_len: int = 64
    ) -> torch.Tensor:
        """Step the delayed PSC over time; ``kernel_len`` is ignored here."""
        T = z_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(z_seq[t])
            y_seq.append(y)
        return torch.stack(y_seq)

    def extra_repr(self):
        return (
            f"max_delay_steps={self.max_delay_steps}, "
            f"circular={self.use_circular_buffer}"
        )


class HeterSynapsePSC(BasePSC):
    """Heterogeneous synapse PSC supporting multiple receptor types.

    Manages its own delay buffering when ``max_delay_steps > 1``,
    making it compatible with delay-expanded connection matrices from
    ``make_hetersynapse_conn(..., delay_col=..., n_delay_bins=...)``.

    Args:
        n_neuron: Number of neurons.
        n_receptor: Number of receptor types.
        receptor_type_index: DataFrame mapping receptor types to indices.
        linear: Linear layer for weight application.
        base_psc: BasePSC subclass to use for dynamics. Default: AlphaPSC.
        max_delay_steps: Maximum delay steps to buffer. Default: 1.
        use_circular_buffer: If False (default), use torch.cat for
            torch.compile compatibility. If True, use circular buffer.
        step_mode: Step mode. Default: "s".
        backend: Compute backend. Default: "torch".
        **kwargs: Passed to ``base_psc`` constructor.
    """

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        n_receptor: int,
        receptor_type_index: pd.DataFrame,
        linear: torch.nn.Module,
        base_psc: type[BasePSC] = AlphaPSC,
        max_delay_steps: int = 1,
        use_circular_buffer: bool = False,
        step_mode: str = "s",
        backend: str = "torch",
        **kwargs,
    ):
        super().__init__(n_neuron, linear, step_mode=step_mode, backend=backend)

        self.max_delay_steps = max_delay_steps
        self.use_circular_buffer = use_circular_buffer

        if max_delay_steps > 1:
            self.history = SpikeHistory(
                n_neuron=self.size,
                max_delay_steps=max_delay_steps,
                use_circular_buffer=use_circular_buffer,
            )
        else:
            self.history = None

        self.base_psc = base_psc(
            n_neuron=self.size * n_receptor,
            linear=linear,
            step_mode=step_mode,
            backend=backend,
            **kwargs,
        )
        self.n_receptor = n_receptor
        self.receptor_type_index = receptor_type_index

    def init_state(
        self,
        batch_size: int | tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        persistent: bool = True,
        skip_mem_name: Iterable[str] = (),
    ) -> None:
        super().init_state(
            batch_size,
            dtype,
            device,
            persistent,
            skip_mem_name=skip_mem_name,
        )
        if self.history is not None:
            self.history.init_state(batch_size, dtype, device, persistent)

    def reset(
        self,
        batch_size: int | tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        skip_mem_name: Iterable[str] = (),
    ) -> None:
        super().reset(
            batch_size,
            dtype,
            device,
            skip_mem_name=skip_mem_name,
        )
        if self.history is not None:
            self.history.reset(batch_size, dtype, device)

    def _flatten_input(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        raw_shape = self.n_neuron
        expanded_shape = (*self.n_neuron, self.n_receptor)

        if z.shape[-len(raw_shape) :] == raw_shape:
            leading = z.shape[: -len(raw_shape)]
            return z.reshape(*leading, self.size), leading, False

        if z.shape[-len(expanded_shape) :] == expanded_shape:
            leading = z.shape[: -len(expanded_shape)]
            return z.reshape(*leading, self.size * self.n_receptor), leading, True

        raise RuntimeError(
            "HeterSynapsePSC input shape mismatch. Expected trailing shape "
            f"{raw_shape} or {expanded_shape}, got {z.shape}."
        )

    def _validate_delayed_input(
        self, has_receptor_axis: bool, input_shape: torch.Size
    ) -> None:
        if has_receptor_axis:
            raise RuntimeError(
                "Delayed HeterSynapsePSC expects input without receptor axis. "
                f"Expected trailing shape {self.n_neuron}, got {input_shape}."
            )

    def _reshape_sum_receptor(
        self, psc_flat: torch.Tensor, leading: tuple[int, ...]
    ) -> torch.Tensor:
        return psc_flat.reshape(*leading, *self.n_neuron, self.n_receptor).sum(-1)

    def single_step_forward(self, z: torch.Tensor):
        z_flat, leading, has_receptor_axis = self._flatten_input(z)

        if self.history is not None:
            self._validate_delayed_input(has_receptor_axis, z.shape)
            self.history.update(z_flat)
            z_flat = self.history.get_flattened(self.max_delay_steps)

        psc = self.base_psc.single_step_forward(z_flat)
        self.psc = self._reshape_sum_receptor(psc, leading)
        return self.psc

    def get_psc(
        self,
        receptor_type: int | str | tuple[str, str] | None = None,
        psc: torch.Tensor | None = None,
        validate_nan: bool = True,
    ):
        """Get PSC for specific receptor type(s).

        Mode is automatically detected from receptor_type_index columns:
        - neuron mode: has 'pre_receptor_type' and 'post_receptor_type'
        - connection mode: has only 'receptor_type'

        Args:
            receptor_type:
                - None: return summed PSC across all receptor types
                - int: receptor index
                - str: receptor type name (connection mode only)
                - tuple[str, str]: (pre_type, post_type) pair (neuron mode only)
            psc: Optional PSC tensor to query, defaults to self.base_psc.psc
            validate_nan: If True, raise error if NaN values detected

        Returns:
            PSC tensor for the specified receptor type(s).
        """
        psc = psc if psc is not None else self.base_psc.psc

        if receptor_type is None:
            result = psc
        else:
            # Autodetect mode from receptor_type_index columns
            has_pre_post = (
                "pre_receptor_type" in self.receptor_type_index.columns
                and "post_receptor_type" in self.receptor_type_index.columns
            )

            if isinstance(receptor_type, tuple):
                # Must be neuron mode
                if not has_pre_post:
                    raise ValueError(
                        "Tuple receptor_type requires neuron mode "
                        "(receptor_type_index must have 'pre_receptor_type' and "
                        "'post_receptor_type' columns)"
                    )
                pre_type, post_type = receptor_type
                idx = self.receptor_type_index.set_index(
                    ["pre_receptor_type", "post_receptor_type"]
                )
                receptor_idx = idx.loc[(pre_type, post_type), "receptor_index"]
            elif isinstance(receptor_type, str):
                # String lookup - mode depends on columns
                if has_pre_post:
                    raise ValueError(
                        "String receptor_type not supported in neuron mode. "
                        "Use tuple (pre_receptor_type, post_receptor_type) instead."
                    )
                idx = self.receptor_type_index.set_index("receptor_type")
                receptor_idx = idx.loc[receptor_type, "receptor_index"]
            else:
                # Integer index
                receptor_idx = int(receptor_type)

            result = psc.view(*psc.shape[:-1], *self.n_neuron, self.n_receptor)[
                ..., receptor_idx
            ]

        if validate_nan and torch.isnan(result).any():
            raise ValueError("NaN values detected in PSC tensor")

        return result

    def multi_step_forward(
        self, z_seq: torch.Tensor, kernel_len: int = 64
    ) -> torch.Tensor:
        """Multi-receptor delayed forward.

        ``kernel_len`` only applies on the history-free path (convolution);
        with delays the PSC is stepped sequentially and it is ignored.
        """
        if self.history is not None:
            _, _, has_receptor_axis = self._flatten_input(z_seq)
            self._validate_delayed_input(has_receptor_axis, z_seq.shape)

            T = z_seq.shape[0]
            y_seq = []
            for t in range(T):
                y_seq.append(self.single_step_forward(z_seq[t]))
            return torch.stack(y_seq)

        z_flat, leading_with_time, _ = self._flatten_input(z_seq)

        psc_flat = self.base_psc.multi_step_forward(z_flat, kernel_len=kernel_len)
        return self._reshape_sum_receptor(psc_flat, leading_with_time)


class GapJunction(nn.Module):
    """Electrical synapse (gap junction) for direct coupling between neurons.

    Gap junctions allow ion flow between connected neurons proportional to
    the voltage difference. The current is computed as:

        I_gap = g_gap * linear(v_post - v_pre)

    where `linear` models both the connection topology and conductance weights.
    Unlike chemical synapses, gap junctions are instantaneous (no delay or
    synaptic dynamics) and bidirectional.

    Args:
        n_neuron: Number of neurons.
        g_gap: Global scaling factor for gap junction conductance. Default: 1.0.
        linear: Linear layer for weight application (models connection and
            conductance). If None, an identity weight matrix is used.
            Default: None.
        step_mode: Step mode. Default: "s".
        backend: Compute backend. Default: "torch".

    Attributes:
        g_gap: Gap junction global scaling factor.
        linear: Linear transformation for connection weights.

    Example:
        >>> gap = GapJunction(n_neuron=4, g_gap=0.1)
        >>> v_pre = torch.randn(2, 4)   # pre-synaptic voltage (mV)
        >>> v_post = torch.randn(2, 4)  # post-synaptic voltage (mV)
        >>> i_gap = gap(v_pre, v_post)  # gap junction current (pA)
    """

    n_neuron: tuple[int, ...]
    size: int
    g_gap: torch.Tensor

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        g_gap: float | TensorLike = 1.0,
        linear: torch.nn.Module | None = None,
        step_mode: str = "s",
    ):
        super().__init__()

        self.n_neuron, self.size = normalize_n_neuron(n_neuron)
        self.step_mode = step_mode

        self.register_buffer("g_gap", torch.as_tensor(g_gap))

        if linear is None:
            self.linear = torch.nn.Linear(self.size, self.size, bias=False)
            torch.nn.init.uniform_(self.linear.weight)
        else:
            self.linear = linear

    def extra_repr(self) -> str:
        return (
            f"n_neuron={self.n_neuron}, g_gap={self.g_gap.item():.4g}, "
            f"step_mode={self.step_mode}"
        )

    def forward(
        self,
        v_pre: Float[Tensor, "*batch n_neuron"],
        v_post: Float[Tensor, "*batch n_neuron"],
    ) -> Float[Tensor, "*batch n_neuron"]:
        """Compute gap junction current from voltage difference.

        Args:
            v_pre: Pre-synaptic membrane potential (mV).
            v_post: Post-synaptic membrane potential (mV).

        Returns:
            Gap junction current I_gap = g_gap * linear(v_post - v_pre) (pA).
        """
        v_pre_flat, leading = flatten_neuron(v_pre, self.n_neuron, self.size)
        v_post_flat, _ = flatten_neuron(v_post, self.n_neuron, self.size)

        delta_v_flat = v_post_flat - v_pre_flat
        i_gap_flat = self.g_gap * self.linear(delta_v_flat)

        return unflatten_neuron(i_gap_flat, leading, self.n_neuron)

    single_step_forward = forward

    def multi_step_forward(
        self,
        v_pre_seq: Float[Tensor, "T *batch n_neuron"],
        v_post_seq: Float[Tensor, "T *batch n_neuron"],
    ) -> Float[Tensor, "T *batch n_neuron"]:
        """Multi-step forward over time dimension.

        Args:
            v_pre_seq: Pre-synaptic voltage sequence (T, *batch, n_neuron).
            v_post_seq: Post-synaptic voltage sequence (T, *batch, n_neuron).

        Returns:
            Gap junction current sequence (T, *batch, n_neuron).
        """
        T = v_pre_seq.shape[0]
        i_seq = []
        for t in range(T):
            i_gap = self.forward(v_pre_seq[t], v_post_seq[t])
            i_seq.append(i_gap)
        return torch.stack(i_seq)


class VoltageCoupling(nn.Module):
    """Voltage coupling for multicompartment neuron models.

    Models coupling currents between compartments via linear weighting of
    membrane potentials. Unlike GapJunction which computes `W*(V_post - V_pre)`,
    VoltageCoupling directly computes `W*V` for coupling currents between
    compartments.

    The current is computed as:

        I_couple = g_couple * linear(v)

    where `linear` models the coupling conductance between compartments.

    Args:
        n_neuron: Number of neurons (or compartments).
        g_couple: Global scaling factor for coupling conductance. Default: 1.0.
        linear: Linear layer for weight application (models coupling
            conductance). If None, an identity weight matrix is used.
            Default: None.
        step_mode: Step mode. Default: "s".

    Attributes:
        g_couple: Coupling global scaling factor.
        linear: Linear transformation for coupling weights.

    Example:
        >>> couple = VoltageCoupling(n_neuron=4, g_couple=0.1)
        >>> v = torch.randn(2, 4)  # compartment voltages (mV)
        >>> i_couple = couple(v)   # coupling current (pA)
    """

    n_neuron: tuple[int, ...]
    size: int
    g_couple: torch.Tensor

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        g_couple: float | TensorLike = 1.0,
        linear: torch.nn.Module | None = None,
        step_mode: str = "s",
    ):
        super().__init__()

        self.n_neuron, self.size = normalize_n_neuron(n_neuron)
        self.step_mode = step_mode

        self.register_buffer("g_couple", torch.as_tensor(g_couple))

        if linear is None:
            self.linear = torch.nn.Linear(self.size, self.size, bias=False)
            torch.nn.init.uniform_(self.linear.weight)
        else:
            self.linear = linear

    def extra_repr(self) -> str:
        return (
            f"n_neuron={self.n_neuron}, g_couple={self.g_couple.item():.4g}, "
            f"step_mode={self.step_mode}"
        )

    def forward(
        self,
        v: Float[Tensor, "*batch n_neuron"],
    ) -> Float[Tensor, "*batch n_neuron"]:
        """Compute coupling current from voltage.

        Args:
            v: Membrane potential (mV).

        Returns:
            Coupling current I_couple = g_couple * linear(v) (pA).
        """
        v_flat, leading = flatten_neuron(v, self.n_neuron, self.size)

        i_couple_flat = self.g_couple * self.linear(v_flat)

        return unflatten_neuron(i_couple_flat, leading, self.n_neuron)

    single_step_forward = forward

    def multi_step_forward(
        self,
        v_seq: Float[Tensor, "T *batch n_neuron"],
    ) -> Float[Tensor, "T *batch n_neuron"]:
        """Multi-step forward over time dimension.

        Args:
            v_seq: Voltage sequence (T, *batch, n_neuron).

        Returns:
            Coupling current sequence (T, *batch, n_neuron).
        """
        T = v_seq.shape[0]
        i_seq = []
        for t in range(T):
            i_couple = self.forward(v_seq[t])
            i_seq.append(i_couple)
        return torch.stack(i_seq)
