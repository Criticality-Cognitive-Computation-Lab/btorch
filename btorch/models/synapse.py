from collections.abc import Iterable, Sequence
from typing import Protocol

import pandas as pd
import torch
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
    application, delay buffering, and PSC state management.

    Args:
        n_neuron: Number of post-synaptic neurons.
        linear: Linear layer for weight application.
        latency: Synaptic delay (ms). Default: 0.0.
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
        latency: float = 0.0,
        step_mode="s",
        backend="torch",
    ):
        super().__init__()

        self.n_neuron, self.size = normalize_n_neuron(n_neuron)
        self.step_mode = step_mode
        self.backend = backend
        self.latency = latency
        self.linear = linear

        self.register_memory("psc", 0.0, self.n_neuron)

        if latency > 0:
            self.latency_steps = round(latency / environ.get("dt"))
            self.register_memory(
                "delay_buffer",
                0,
                (self.latency_steps + 1, *self.n_neuron),
            )

    def init_state(
        self,
        batch_size=None,
        dtype=None,
        device=None,
        persistent=True,
        skip_mem_name: Iterable[str] = (),
    ):
        super().init_state(
            batch_size,
            dtype,
            device,
            persistent,
            skip_mem_name=("delay_buffer",) + skip_mem_name,
        )
        if self.latency > 0:
            delay_buffer_sizes = self._memories_rv["delay_buffer"].sizes
            if batch_size is not None:
                if isinstance(batch_size, int):
                    batch_size = (batch_size,)
                delay_buffer_sizes = (
                    delay_buffer_sizes[0],
                    *batch_size,
                    *delay_buffer_sizes[1:],
                )
            self.register_buffer(
                "delay_buffer",
                torch.zeros(delay_buffer_sizes, dtype=dtype, device=device),
            )

    def reset(
        self,
        batch_size=None,
        dtype=None,
        device=None,
        skip_mem_name: Iterable[str] = (),
    ):
        super().reset(
            batch_size,
            dtype,
            device,
            skip_mem_name=("delay_buffer",) + skip_mem_name,
        )
        if self.latency > 0:
            delay_buffer_sizes = self._memories_rv["delay_buffer"].sizes
            if batch_size is None:
                extra_dims = self.delay_buffer.ndim - len(delay_buffer_sizes)
                if extra_dims > 0:
                    batch_size = self.delay_buffer.shape[1 : 1 + extra_dims]
            if batch_size is not None:
                if isinstance(batch_size, int):
                    batch_size = (batch_size,)
                delay_buffer_sizes = (
                    delay_buffer_sizes[0],
                    *batch_size,
                    *delay_buffer_sizes[1:],
                )
            self.delay_buffer = torch.zeros(
                delay_buffer_sizes, dtype=dtype, device=device
            )

    def extra_repr(self):
        return f" step_mode={self.step_mode}, backend={self.backend}"

    def _flatten_neuron(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        return flatten_neuron(x, self.n_neuron, self.size)

    def _unflatten_neuron(
        self, x: torch.Tensor, leading_shape: tuple[int, ...]
    ) -> torch.Tensor:
        return unflatten_neuron(x, leading_shape, self.n_neuron)

    def conductance_charge(self):
        raise NotImplementedError()

    def adaptation_charge(self, z: torch.Tensor):
        raise NotImplementedError()

    def current_charge(self, v=None):
        if v is not None:
            raise NotImplementedError(
                "Only current-based PSC is supported."
                "Conductance-based PSC requires voltage from post-syn neurons, "
                "which the current abstraction doesn't support."
            )
        else:
            return self.psc

    def single_step_forward(self, z: torch.Tensor):
        if self.latency > 0:
            self.delay_buffer = torch.cat(
                (z.unsqueeze(dim=0), self.delay_buffer[:-1]), dim=0
            )
            spike = self.delay_buffer[-1]
        else:
            spike = z

        self.conductance_charge()
        self.adaptation_charge(spike)
        current = self.current_charge()
        return current

    def multi_step_forward(self, z_seq: torch.Tensor):
        T = z_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(z_seq[t])
            y_seq.append(y)

        return torch.stack(y_seq)


class ExponentialPSC(BasePSC):
    """Exponential decay synapse model.

    Simple first-order synapse with single exponential decay:
        d(psc)/dt = -psc / tau_syn

    Args:
        n_neuron: Number of neurons.
        tau_syn: Synaptic time constant (ms).
        linear: Linear layer for weights.
        latency: Synaptic delay (ms). Default: 0.0.
        step_mode: Step mode. Default: "s".
        backend: Compute backend. Default: "torch".
    """

    tau_syn: torch.Tensor | torch.nn.Parameter

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        tau_syn: float | TensorLike,
        linear,
        latency: float = 0.0,
        step_mode="s",
        backend="torch",
    ):
        super().__init__(
            n_neuron,
            linear,
            latency=latency,
            step_mode=step_mode,
            backend=backend,
        )

        self.register_buffer("tau_syn", torch.as_tensor(tau_syn), persistent=False)

    def dpsc(self, psc):
        derivative = -psc / self.tau_syn
        linear = -1.0 / self.tau_syn
        return derivative, linear

    def conductance_charge(self):
        self.psc = exp_euler_step(self.dpsc, self.psc, dt=environ.get("dt"))
        return self.psc

    def adaptation_charge(self, z: torch.Tensor):
        z_flat, leading = self._flatten_neuron(z)
        wz = self.linear(z_flat)
        wz = self._unflatten_neuron(wz, leading)
        self.psc = self.psc + wz


class _Adaptive2VarPSC(BasePSC):
    h: torch.Tensor

    def __init__(
        self, n_neuron: int | Sequence[int], linear, step_mode="s", backend="torch"
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
        step_mode="s",
        backend="torch",
    ):
        """The Current-Based Alpha form of PSC, from [1], ensuring a post-
        synaptic current with synapse weight W = 1.0 has an amplitude of 1.0 pA
        at the peak time point of t = tau_syn.

        NOTE: this model assumes environ.get("dt") == 1.0

        [1] Billeh, Y. N. et al. Systematic integration of structural and
        functional data into multi-scale models of mouse primary visual
        cortex. 662189 Preprint at https://doi.org/10.1101/662189 (2019).

        :param tau_syn: the synaptic time constant
        :type tau_syn: float or torch.Tensor
        """

        super().__init__(n_neuron, linear, step_mode, backend)

        try:
            dt = environ.get("dt")
            assert dt == 1.0, "dt must be 1.0 for this model"
        except KeyError:
            pass

        self.register_buffer("tau_syn", torch.as_tensor(tau_syn), persistent=False)

        self.register_buffer(
            "syn_decay", torch.exp(-1.0 / self.tau_syn), persistent=False
        )

    def conductance_charge(self):
        self.psc = self.syn_decay * self.psc + self.syn_decay * self.h
        return self.psc

    def adaptation_charge(self, z: torch.Tensor):
        z_flat, leading = self._flatten_neuron(z)
        wz = self.linear(z_flat)
        wz = self._unflatten_neuron(wz, leading)
        self.h = self.syn_decay * self.h + torch.e / self.tau_syn * wz


class AlphaPSC(_Adaptive2VarPSC):
    tau_syn: torch.Tensor | torch.nn.Parameter
    g_max: torch.Tensor | torch.nn.Parameter

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        tau_syn: float | TensorLike,
        linear: torch.nn.Module,
        g_max=1.0,
        step_mode="s",
        backend="torch",
    ):
        """The Alpha form (current-based) of PSC, from Brainpy/BrainState."""

        super().__init__(n_neuron, linear, step_mode, backend)

        self.register_buffer("tau_syn", torch.as_tensor(tau_syn), persistent=False)
        self.register_buffer("g_max", torch.as_tensor(g_max), persistent=False)

    def dg(self, psc, h):
        derivative = -psc / self.tau_syn + h / self.tau_syn
        linear = -1.0 / self.tau_syn
        return derivative, linear

    def dh(self, h):
        derivative = -h / self.tau_syn
        linear = -1.0 / self.tau_syn
        return derivative, linear

    def conductance_charge(self):
        self.psc = exp_euler_step(self.dg, self.psc, self.h, dt=environ.get("dt"))

    def adaptation_charge(self, z: torch.Tensor):
        z_flat, leading = self._flatten_neuron(z)
        wz = self.g_max * self.linear(z_flat)
        wz = self._unflatten_neuron(wz, leading)
        self.h = exp_euler_step(self.dh, self.h, dt=environ.get("dt")) + wz


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
        latency: float = 0.0,
        A: float | TensorLike | None = None,
        step_mode="s",
        backend="torch",
    ):
        """The Double Exponential form of PSC, from Brainpy/BrainState."""

        super().__init__(
            n_neuron=n_neuron,
            linear=linear,
            latency=latency,
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

    def dg_rise(self, g_rise):
        derivative = -g_rise / self.tau_rise
        linear = -1.0 / self.tau_rise
        return derivative, linear

    def dg_decay(self, g_decay):
        derivative = -g_decay / self.tau_decay
        linear = -1.0 / self.tau_decay
        return derivative, linear

    def conductance_charge(self):
        self.g_rise = exp_euler_step(self.dg_rise, self.g_rise, dt=environ.get("dt"))
        self.g_decay = exp_euler_step(self.dg_decay, self.g_decay, dt=environ.get("dt"))

    def adaptation_charge(self, z: torch.Tensor):
        z_flat, leading = self._flatten_neuron(z)
        wz = self.linear(z_flat)
        wz = self._unflatten_neuron(wz, leading)
        self.g_rise = self.g_rise + wz
        self.g_decay = self.g_decay + wz
        self.psc = self.a * (self.g_decay - self.g_rise)


class HeterSynapsePSC(BasePSC):
    def __init__(
        self,
        n_neuron: int | Sequence[int],
        n_receptor: int,
        receptor_type_index: pd.DataFrame,
        linear: torch.nn.Module,
        base_psc: type[BasePSC] = AlphaPSC,
        step_mode="s",
        backend="torch",
        **kwargs,
    ):
        super().__init__(
            n_neuron, linear, latency=0, step_mode=step_mode, backend=backend
        )

        self.base_psc = base_psc(
            n_neuron=self.size * n_receptor,
            linear=linear,
            step_mode=step_mode,
            backend=backend,
            **kwargs,
        )
        self.n_receptor = n_receptor
        self.receptor_type_index = (
            receptor_type_index  # Store as-is, get_psc will handle indexing
        )

    def single_step_forward(self, z: torch.Tensor):
        z_flat, leading = self._flatten_neuron(z)
        psc = self.base_psc.single_step_forward(z_flat)
        self.psc = psc.view(*psc.shape[:-1], *self.n_neuron, self.n_receptor).sum(-1)
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
        backend: str = "torch",
    ):
        super().__init__()

        self.n_neuron, self.size = normalize_n_neuron(n_neuron)
        self.step_mode = step_mode
        self.backend = backend

        # Register global scaling factor as buffer (non-trainable by default)
        self.register_buffer("g_gap", torch.as_tensor(g_gap))

        # Linear layer for connection weights (models both connection and conductance)
        if linear is None:
            self.linear = torch.nn.Linear(self.size, self.size, bias=False)
            torch.nn.init.uniform_(self.linear.weight)
        else:
            self.linear = linear

    def extra_repr(self) -> str:
        return (
            f"n_neuron={self.n_neuron}, g_gap={self.g_gap.item():.4g}, "
            f"step_mode={self.step_mode}, backend={self.backend}"
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

        # Current is proportional to weighted voltage difference
        # linear models both connection topology and conductance
        delta_v_flat = v_post_flat - v_pre_flat
        i_gap_flat = self.g_gap * self.linear(delta_v_flat)

        return unflatten_neuron(i_gap_flat, leading, self.n_neuron)

    def single_step_forward(
        self,
        v_pre: Float[Tensor, "*batch n_neuron"],
        v_post: Float[Tensor, "*batch n_neuron"],
    ) -> Float[Tensor, "*batch n_neuron"]:
        """Single step forward (alias for forward)."""
        return self.forward(v_pre, v_post)

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


class VoltageCoupling(GapJunction):
    """Alias for GapJunction, emphasizing voltage coupling role of gap
    junctions in compartment models."""

    pass
