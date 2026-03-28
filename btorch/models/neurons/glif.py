"""Generalized leaky integrate-and-fire (GLIF) neuron models.

This module implements the GLIF3 model from the Allen Institute [1], which
extends standard LIF with after-spike currents (ASC) that capture
spike-frequency adaptation and other slow currents.

The GLIF3 neuron follows:
    dV/dt = -(V - V_rest) / tau + (I_in + sum(I_asc)) / c_m
    dI_asc/dt = -k * I_asc

where I_asc are after-spike currents that increment by asc_amps at each spike.

References:
    [1] Teeter et al., "Generalized leaky integrate-and-fire models
        classify multiple neuron types," Nat. Commun., 2018.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ...types import TensorLike
from .. import environ
from ..base import BaseNode
from ..ode import exp_euler_step
from ..shape import expand_trailing_dims
from ..surrogate import ATan


def get_rheobase(
    v_threshold: float | torch.Tensor,
    v_rest: float | torch.Tensor,
    c_m: float | torch.Tensor,
    tau: float | torch.Tensor,
) -> float | torch.Tensor:
    """Calculate rheobase current.

    The rheobase is the minimum constant input current required to make
    the neuron fire. For GLIF models:
        I_rheobase = (v_threshold - v_rest) * c_m / tau

    Args:
        v_threshold: Firing threshold (mV).
        v_rest: Resting potential (mV).
        c_m: Membrane capacitance (pF).
        tau: Membrane time constant (ms).

    Returns:
        Rheobase current (pA).
    """
    # For GLIF3, rheobase can be calculated as:
    # I_rheobase = (v_threshold - v_rest) * c_m / tau
    I_rheobase = (v_threshold - v_rest) * c_m / tau
    return I_rheobase


class GLIF3(BaseNode):
    """GLIF3 model with after-spike currents and refractory period.

    The GLIF3 model extends standard LIF by adding after-spike currents
    (ASC) that capture spike-frequency adaptation. Each spike adds
    asc_amps to the ASC vector, which then decays exponentially with
    time constants 1/k.

    Dynamics:
        dV/dt = -(V - V_rest) / tau + (I_in + sum(I_asc)) / c_m
        dI_asc/dt = -k * I_asc

        At spike: I_asc += asc_amps

    Args:
        n_neuron: Number of neurons (int or tuple of dimensions).
        v_threshold: Firing threshold (mV). Default: -50.0.
        v_reset: Reset voltage after spike (mV). Default: -70.0.
        v_rest: Resting potential (mV). Defaults to v_reset if None.
        c_m: Membrane capacitance (pF). Default: 0.05.
        tau: Membrane time constant (ms). Default: 20.0.
        k: ASC decay rates (ms^-1), can be list for multiple ASC components.
            Default: [0.2].
        asc_amps: ASC amplitudes (pA) added at each spike.
            Default: [0.0].
        tau_ref: Refractory period (ms). Default: 0.0.
        trainable_param: Set of parameter names to make trainable.
        surrogate_function: Surrogate gradient function. Default: ATan().
        detach_reset: If True, detach reset signal. Default: False.
        hard_reset: If True, use hard reset. Default: False.
        pre_spike_v: If True, store pre-spike voltage. Default: False.
        step_mode: Step mode. Default: "s".
        backend: Backend implementation. Default: "torch".
        device: Device for tensors. Default: None.
        dtype: Data type for tensors. Default: None.

    Attributes:
        v: Membrane potential, shape (*batch, n_neuron).
        Iasc: After-spike currents, shape (*batch, n_neuron, n_Iasc).
        refractory: Refractory counter (if tau_ref > 0).
        c_m, tau, tau_ref: Neuron parameters.
        k: ASC decay rates, shape (n_neuron, n_Iasc) or (n_Iasc,).
        asc_amps: ASC amplitudes, shape (n_neuron, n_Iasc) or (n_Iasc,).
        n_Iasc: Number of ASC components.

    References:
        Teeter et al., "Generalized leaky integrate-and-fire models
        classify multiple neuron types," Nature Communications, 2018.
    """

    # make mypy typing and autocompletion easier
    Iasc: torch.Tensor
    refractory: torch.Tensor | None

    c_m: torch.Tensor | torch.nn.Parameter
    tau: torch.Tensor | torch.nn.Parameter
    tau_ref: torch.Tensor | torch.nn.Parameter | None
    k: torch.Tensor | torch.nn.Parameter
    asc_amps: torch.Tensor | torch.nn.Parameter

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        v_threshold: float | Float[TensorLike, " n_neuron"] = -50.0,  # mV
        v_reset: float | Float[TensorLike, " n_neuron"] = -70.0,  # mV
        v_rest: None | float | Float[TensorLike, " n_neuron"] = None,
        c_m: float | Float[TensorLike, " n_neuron"] = 0.05,  # 1/20 pfarad
        tau: float | Float[TensorLike, " n_neuron"] = 20.0,  # ms
        k: float | Sequence[float] | Float[TensorLike, "n_neuron {self.n_Iasc}"] = [
            0.2
        ],  # ms^-1
        asc_amps: float
        | Sequence[float]
        | Float[TensorLike, "n_neuron {self.n_Iasc}"] = [0.0],  # pA
        tau_ref: float | Float[TensorLike, " n_neuron"] | None = 0.0,  # ms
        trainable_param: set[str] = set(),
        surrogate_function: Callable = ATan(),
        detach_reset: bool = False,
        hard_reset: bool = False,
        pre_spike_v: bool = False,
        step_mode: Literal["s"] = "s",
        backend: Literal["torch"] = "torch",
        device=None,
        dtype=None,
    ):
        super().__init__(
            n_neuron=n_neuron,
            v_threshold=v_threshold,
            v_reset=v_reset,
            trainable_param=trainable_param,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            pre_spike_v=pre_spike_v,
        )
        _factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.hard_reset = hard_reset
        self._def_param("c_m", c_m, **_factory_kwargs)
        self._def_param("tau", tau, **_factory_kwargs)
        self._use_refractory = tau_ref is not None
        if self._use_refractory:
            self._def_param("tau_ref", tau_ref, **_factory_kwargs)
            self.register_memory("refractory", 0.0, self.n_neuron)
        else:
            self.tau_ref = None

        # for compat
        if v_rest is not None:
            self._def_param("_v_rest", v_rest, **_factory_kwargs)
        else:
            self._v_rest = None

        # Handle after-spike currents
        if isinstance(asc_amps, float):
            asc_amps = [asc_amps]

        if isinstance(k, float):
            k = [k]

        self._def_param("k", k, allow_trailing_dims=True, **_factory_kwargs)
        self._def_param(
            "asc_amps", asc_amps, allow_trailing_dims=True, **_factory_kwargs
        )

        self.n_Iasc: int = self.asc_amps.shape[-1]

        self.register_memory(
            "Iasc",
            [
                0.0,
            ]
            * self.n_Iasc,
            self.n_neuron + (self.n_Iasc,),
        )

    @property
    def v_rest(self) -> torch.Tensor:
        """Resting potential (mV).

        For compatibility with GLIF4/GLIF5, falls back to v_reset if
        not explicitly set during initialization.

        Returns:
            Resting potential tensor.
        """
        if self._v_rest is None:
            return self.v_reset
        return self._v_rest

    @v_rest.setter
    def v_rest(self, v_rest: float | torch.Tensor):
        """Set resting potential.

        Args:
            v_rest: New resting potential value (mV).
        """
        if self._v_rest is not None:
            self._v_rest = v_rest

    def dIasc(self, Iasc: Float[Tensor, "*batch n_neuron {self.n_Iasc}"]):
        """Compute ASC derivative for exponential Euler integration.

        Args:
            Iasc: After-spike currents, shape (*batch, n_neuron, n_Iasc).

        Returns:
            Tuple of (derivative, linear_coefficient) for exp_euler_step.
        """
        return -self.k * Iasc, -self.k

    def dV(
        self,
        v: Float[Tensor, "*batch n_neuron"],
        Iasc: Float[Tensor, "*batch n_neuron {self.n_Iasc}"],
        x: Float[Tensor, "*batch n_neuron"],
    ):
        """Compute membrane potential derivative for exp Euler integration.

        Args:
            v: Membrane potential, shape (*batch, n_neuron).
            Iasc: After-spike currents, shape (*batch, n_neuron, n_Iasc).
            x: Input current, shape (*batch, n_neuron).

        Returns:
            Tuple of (derivative, linear_coefficient) for exp_euler_step.
        """
        Isum = x
        # torch.autocast will cast half to float32 for sum op
        # see https://docs.pytorch.org/docs/stable/amp.html#ops-that-can-autocast-to-float32
        # here Iasc generally only have <4 modes, so no overflow guaranteed
        return (
            -(v - self.v_rest) / self.tau
            + (Isum + Iasc.sum(-1, dtype=Iasc.dtype)) / self.c_m,
            -1.0 / self.tau,
        )

    def neuronal_charge(self, x: Float[Tensor, "*batch n_neuron"]):
        v = exp_euler_step(self.dV, self.v, self.Iasc, x, dt=environ.get("dt"))
        self.v = v

    def neuronal_adaptation(self):
        self.Iasc = exp_euler_step(self.dIasc, self.Iasc, dt=environ.get("dt"))

    def neuronal_fire(self):
        # Check if voltage exceeds threshold and not in refractory period
        spike = self.surrogate_function(
            (self.v - self.v_threshold) / (self.v_threshold - self.v_reset)
        )
        if not self._use_refractory:
            return spike
        not_in_refractory = self.refractory == 0
        spike = spike * not_in_refractory.detach().to(self.v.dtype)
        return spike

    def neuronal_reset(self, spike: Float[Tensor, "*batch n"]):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.pre_spike_v:
            self.v_pre_spike = self.v.clone()

        if self.hard_reset:
            # hard reset
            self.v = self.v - (self.v - self.v_reset) * spike_d
        else:
            # soft reset
            self.v = self.v - (self.v_threshold - self.v_reset) * spike_d

        # Add after-spike currents
        self.Iasc = self.Iasc + self.asc_amps * spike_d[..., None]

        if self._use_refractory:
            # Set refractory period
            self.refractory = torch.relu(
                self.refractory + spike_d * self.tau_ref - environ.get("dt")
            )

    def get_rheobase(self):
        """Calculate rheobase current, the minimum constant input current
        required to make the neuron fire."""
        return get_rheobase(self.v_threshold, self.v_rest, self.c_m, self.tau)

    def extra_repr(self):
        parts = [
            f"c_m={self._format_repr_value(self.c_m)}",
            f"tau={self._format_repr_value(self.tau)}",
            f"tau_ref={self._format_repr_value(self.tau_ref)}"
            if self._use_refractory
            else "tau_ref=None",
            f"n_Iasc={self.n_Iasc}",
            f"k={self._format_repr_value(self.k)}",
            f"asc_amps={self._format_repr_value(self.asc_amps)}",
            "v_rest=auto"
            if self._v_rest is None
            else f"v_rest={self._format_repr_value(self._v_rest)}",
        ]
        base = super().extra_repr()
        if base:
            parts.append(base)
        return ", ".join(parts)

    # TODO: headache to define precise input-output shapes
    # TODO: shape handling not torch.compile friendly
    def _normalize_state_shapes(
        self,
        x: TensorLike | float,
        v0: TensorLike | float,
        Iasc0: TensorLike | float,
        dt: TensorLike | float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = device or self.v_reset.device
        dtype = dtype or self.v_reset.dtype
        x, v0, Iasc0 = (
            torch.as_tensor(x, device=device, dtype=dtype),
            torch.as_tensor(v0, device=device, dtype=dtype),
            torch.as_tensor(Iasc0, device=device, dtype=dtype),
        )
        if isinstance(dt, float):
            dt = torch.tensor([dt], device=device, dtype=dtype)
        else:
            dt = torch.as_tensor(dt, device=device, dtype=dtype)

        shapes = (x.shape, v0.shape, Iasc0.shape[:-1])
        longest_shape = max(shapes, key=len)
        if dt.shape[0] != longest_shape[0]:
            dt = expand_trailing_dims(dt, longest_shape, broadcast_only=True)

        return x, v0, Iasc0, dt

    def forward_exact_no_spike(
        self,
        x: Float[Tensor, "*batch #neuron"] | Float[Tensor, "*batch"],
        v0: Float[Tensor, "*batch neuron"] | None = None,
        Iasc0: Float[Tensor, "*batch neuron {self.n_Iasc}"] | None = None,
        dt: float
        | Float[TensorLike, "#time *batch neuron"]
        | Float[TensorLike, "#*batch neuron"]
        | Float[TensorLike, "#time *batch"]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dt is None:
            dt = environ.get("dt")

        update = (v0 is None) and (Iasc0 is None)
        if v0 is None:
            v0 = self.v
        if Iasc0 is None:
            Iasc0 = self.Iasc

        x, v0, Iasc0, dt = self._normalize_state_shapes(x, v0, Iasc0, dt)

        v_inf = self.v_reset + x * self.tau / self.c_m

        exp_m = torch.exp(-dt / self.tau)
        # (time, batch, neuron, n_Iasc)
        exp_asc = torch.exp(-dt[..., None] * self.k)

        Iasc = Iasc0 * exp_asc

        # degenerate case if tau=tau_asc=1/k
        Iasc_contrib = torch.where(
            torch.abs(self.k - 1 / self.tau[..., None]) > 1e-12,
            (Iasc0 / self.c_m[..., None])
            * (exp_asc - exp_m[..., None])
            / (1.0 / self.tau[..., None] - self.k),
            (Iasc0 / self.c_m[..., None]) * (dt * exp_m)[..., None],
        )
        v = v_inf + (v0 - v_inf) * exp_m + Iasc_contrib.sum(dim=-1)

        if update:
            self.v = v
            self.Iasc = Iasc
        return v, Iasc
