"""Adaptive leaky integrate-and-fire (ALIF) neuron models.

This module provides ALIF and ELIF (exponential LIF) neuron implementations
with conductance-based adaptation mechanisms.

The ALIF neuron extends LIF by adding a potassium conductance (g_k) that
increases with each spike, creating spike-frequency adaptation:

    dv/dt = (-g_leak * (v - E_leak) - g_k * (v - E_k) + x) / c_m
    dg_k/dt = -g_k / tau_adapt

where g_k increments by dg_k at each spike and decays exponentially,
creating negative feedback that slows firing rate over time.
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
from ..surrogate import Sigmoid


class ALIF(BaseNode):
    """Adaptive leaky integrate-and-fire neuron with conductance adaptation.

    The ALIF model extends standard LIF by adding a voltage-dependent
    potassium conductance (g_k) that creates spike-frequency adaptation.
    Each spike increases g_k by dg_k, which then decays exponentially.

    Dynamics:
        dv/dt = (-g_leak * (v - E_leak) - g_k * (v - E_k) + x) / c_m
        dg_k/dt = -g_k / tau_adapt

        At spike: g_k += dg_k

    Args:
        n_neuron: Number of neurons (int or tuple of dimensions).
        v_threshold: Firing threshold (mV). Default: 1.0.
        v_reset: Reset voltage after spike (mV). Default: 0.0.
        c_m: Membrane capacitance (pF). Default: 1.0.
        g_leak: Leak conductance (nS). Default: 1.0.
        E_leak: Leak reversal potential (mV). Default: 0.0.
        E_k: Potassium reversal potential (mV). Default: -70.0.
        g_k_init: Initial adaptation conductance (nS). Default: 0.0.
        tau_adapt: Adaptation time constant (ms). Default: 20.0.
        dg_k: Adaptation increment per spike (nS). Default: 0.0.
        tau_ref: Refractory period (ms). None disables refractory.
            Default: None.
        trainable_param: Set of parameter names to make trainable.
        surrogate_function: Surrogate gradient function. Default: Sigmoid().
        detach_reset: If True, detach reset signal. Default: False.
        hard_reset: If True, use hard reset. Default: False.
        pre_spike_v: If True, store pre-spike voltage. Default: False.
        step_mode: Step mode. Default: "s".
        backend: Backend implementation. Default: "torch".
        device: Device for tensors. Default: None.
        dtype: Data type for tensors. Default: None.

    Attributes:
        v: Membrane potential, shape (*batch, n_neuron).
        g_k: Adaptation conductance, shape (*batch, n_neuron).
        refractory: Refractory counter (if tau_ref specified).
        c_m, g_leak, E_leak, E_k: Neuron parameters.
        tau_adapt: Adaptation time constant.
        dg_k: Per-spike adaptation increment.
    """

    g_k: torch.Tensor
    refractory: torch.Tensor | None

    c_m: torch.Tensor | torch.nn.Parameter
    g_leak: torch.Tensor | torch.nn.Parameter
    E_leak: torch.Tensor | torch.nn.Parameter
    E_k: torch.Tensor | torch.nn.Parameter
    tau_adapt: torch.Tensor | torch.nn.Parameter
    dg_k: torch.Tensor | torch.nn.Parameter
    tau_ref: torch.Tensor | torch.nn.Parameter | None

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        v_threshold: float | Float[TensorLike, " n_neuron"] = 1.0,
        v_reset: float | Float[TensorLike, " n_neuron"] = 0.0,
        c_m: float | Float[TensorLike, " n_neuron"] = 1.0,
        g_leak: float | Float[TensorLike, " n_neuron"] = 1.0,
        E_leak: float | Float[TensorLike, " n_neuron"] = 0.0,
        E_k: float | Float[TensorLike, " n_neuron"] = -70.0,
        g_k_init: float | Float[TensorLike, " n_neuron"] = 0.0,
        tau_adapt: float | Float[TensorLike, " n_neuron"] = 20.0,
        dg_k: float | Float[TensorLike, " n_neuron"] = 0.0,
        tau_ref: float | Float[TensorLike, " n_neuron"] | None = None,
        trainable_param: set[str] = set(),
        surrogate_function: Callable = Sigmoid(),
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
            hard_reset=hard_reset,
            pre_spike_v=pre_spike_v,
            step_mode=step_mode,
            backend=backend,
            device=device,
            dtype=dtype,
        )
        _factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self._def_param("c_m", c_m, **_factory_kwargs)
        self._def_param("g_leak", g_leak, **_factory_kwargs)
        self._def_param("E_leak", E_leak, **_factory_kwargs)
        self._def_param("E_k", E_k, **_factory_kwargs)
        self._def_param("tau_adapt", tau_adapt, **_factory_kwargs)
        self._def_param("dg_k", dg_k, **_factory_kwargs)
        self._use_refractory = tau_ref is not None
        if self._use_refractory:
            self._def_param("tau_ref", tau_ref, **_factory_kwargs)
            self.register_memory("refractory", 0.0, self.n_neuron)
        else:
            self.tau_ref = None

        self.register_memory("g_k", g_k_init, self.n_neuron)

    @property
    def v_rest(self):
        if self._v_rest is None:
            return self.v_reset
        return self._v_rest

    @v_rest.setter
    def v_rest(self, v_rest):
        if self._v_rest is not None:
            self._v_rest = v_rest

    @property
    def v_peak(self):
        return self.v_threshold

    @v_peak.setter
    def v_peak(self, value):
        self.v_threshold = value

    def dV(
        self,
        v: Float[Tensor, "*batch n_neuron"],
        g_k: Float[Tensor, "*batch n_neuron"],
        x: Float[Tensor, "*batch n_neuron"],
    ):
        leak_term = -self.g_leak * (v - self.E_leak)
        adapt_term = -g_k * (v - self.E_k)
        derivative = (leak_term + adapt_term + x) / self.c_m
        linear = (-self.g_leak - g_k) / self.c_m
        return derivative, linear

    def dgk(
        self, g_k: Float[Tensor, "*batch n_neuron"]
    ) -> Float[Tensor, "*batch n_neuron"]:
        derivative = -g_k / self.tau_adapt
        linear = -1.0 / self.tau_adapt
        return derivative, linear

    def neuronal_charge(self, x: Float[Tensor, "*batch n_neuron"]):
        dt = environ.get("dt")
        self.v = exp_euler_step(self.dV, self.v, self.g_k, x, dt=dt)

    def neuronal_adaptation(self):
        dt = environ.get("dt")
        self.g_k = exp_euler_step(self.dgk, self.g_k, dt=dt)

    def neuronal_fire(self):
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
            self.v = self.v - (self.v - self.v_reset) * spike_d
        else:
            self.v = self.v - (self.v_threshold - self.v_reset) * spike_d

        self.g_k = self.g_k + self.dg_k * spike_d

        if self._use_refractory:
            self.refractory = torch.relu(
                self.refractory + spike_d * self.tau_ref - environ.get("dt")
            )

    def extra_repr(self):
        g_k_init = self._memories_rv["g_k"].value
        parts = [
            f"c_m={self._format_repr_value(self.c_m)}",
            f"g_leak={self._format_repr_value(self.g_leak)}",
            f"E_leak={self._format_repr_value(self.E_leak)}",
            f"E_k={self._format_repr_value(self.E_k)}",
            f"tau_adapt={self._format_repr_value(self.tau_adapt)}",
            f"dg_k={self._format_repr_value(self.dg_k)}",
            f"g_k_init={self._format_repr_value(g_k_init)}",
            f"tau_ref={self._format_repr_value(self.tau_ref)}"
            if self._use_refractory
            else "tau_ref=None",
        ]
        base = super().extra_repr()
        if base:
            parts.append(base)
        return ", ".join(parts)


class ELIF(ALIF):
    """Exponential integrate-and-fire neuron with adaptation.

    The ELIF model extends ALIF by adding an exponential term to the
    voltage dynamics, creating a sharp upswing when approaching threshold
    (the "initiation zone"). This captures the rapid depolarization seen
    in real neurons.

    Dynamics:
        dv/dt = (g_leak * delta_T * exp((v - v_T) / delta_T)
                 - g_leak * (v - E_leak) - g_k * (v - E_k) + x) / c_m
        dg_k/dt = -g_k / tau_adapt

    The exponential term creates a soft threshold effect where membrane
    potential accelerates as it approaches v_T.

    Args:
        n_neuron: Number of neurons (int or tuple of dimensions).
        v_threshold: Firing threshold (mV). Default: 1.0.
        v_reset: Reset voltage after spike (mV). Default: 0.0.
        c_m: Membrane capacitance (pF). Default: 1.0.
        g_leak: Leak conductance (nS). Default: 1.0.
        E_leak: Leak reversal potential (mV). Default: 0.0.
        E_k: Potassium reversal potential (mV). Default: -70.0.
        g_k_init: Initial adaptation conductance (nS). Default: 0.0.
        tau_adapt: Adaptation time constant (ms). Default: 20.0.
        dg_k: Adaptation increment per spike (nS). Default: 0.0.
        tau_ref: Refractory period (ms). Default: 0.0.
        delta_T: Slope factor for exponential term (mV). Default: 1.0.
        v_T: Soft threshold potential (mV). Default: 0.0.
        trainable_param: Set of parameter names to make trainable.
        surrogate_function: Surrogate gradient function. Default: Sigmoid().
        detach_reset: If True, detach reset signal. Default: False.
        hard_reset: If True, use hard reset. Default: False.
        pre_spike_v: If True, store pre-spike voltage. Default: False.
        step_mode: Step mode. Default: "s".
        backend: Backend implementation. Default: "torch".
        device: Device for tensors. Default: None.
        dtype: Data type for tensors. Default: None.

    Attributes:
        delta_T: Slope factor for exponential term.
        v_T: Soft threshold potential.
    """

    delta_T: torch.Tensor | torch.nn.Parameter
    v_T: torch.Tensor | torch.nn.Parameter

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        v_threshold: float | Float[TensorLike, " n_neuron"] = 1.0,
        v_reset: float | Float[TensorLike, " n_neuron"] = 0.0,
        c_m: float | Float[TensorLike, " n_neuron"] = 1.0,
        g_leak: float | Float[TensorLike, " n_neuron"] = 1.0,
        E_leak: float | Float[TensorLike, " n_neuron"] = 0.0,
        E_k: float | Float[TensorLike, " n_neuron"] = -70.0,
        g_k_init: float | Float[TensorLike, " n_neuron"] = 0.0,
        tau_adapt: float | Float[TensorLike, " n_neuron"] = 20.0,
        dg_k: float | Float[TensorLike, " n_neuron"] = 0.0,
        tau_ref: float | Float[TensorLike, " n_neuron"] | None = 0.0,
        delta_T: float | Float[TensorLike, " n_neuron"] = 1.0,
        v_T: float | Float[TensorLike, " n_neuron"] = 0.0,
        trainable_param: set[str] = set(),
        surrogate_function: Callable = Sigmoid(),
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
            c_m=c_m,
            g_leak=g_leak,
            E_leak=E_leak,
            E_k=E_k,
            g_k_init=g_k_init,
            tau_adapt=tau_adapt,
            dg_k=dg_k,
            tau_ref=tau_ref,
            trainable_param=trainable_param,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            hard_reset=hard_reset,
            pre_spike_v=pre_spike_v,
            step_mode=step_mode,
            backend=backend,
            device=device,
            dtype=dtype,
        )
        _factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self._def_param("delta_T", delta_T, **_factory_kwargs)
        self._def_param("v_T", v_T, **_factory_kwargs)

    def dV(
        self,
        v: Float[Tensor, "*batch n_neuron"],
        g_k: Float[Tensor, "*batch n_neuron"],
        x: Float[Tensor, "*batch n_neuron"],
    ):
        leak_term = -self.g_leak * (v - self.E_leak)
        adapt_term = -g_k * (v - self.E_k)
        exp_term = self.g_leak * self.delta_T * torch.exp((v - self.v_T) / self.delta_T)
        derivative = (leak_term + adapt_term + exp_term + x) / self.c_m
        linear = (-self.g_leak - g_k + exp_term / self.delta_T) / self.c_m
        return derivative, linear

    def neuronal_charge(self, x: Float[Tensor, "*batch n_neuron"]):
        dt = environ.get("dt")
        self.v = exp_euler_step(self.dV, self.v, self.g_k, x, dt=dt)

    def extra_repr(self):
        parts = [
            f"delta_T={self._format_repr_value(self.delta_T)}",
            f"v_T={self._format_repr_value(self.v_T)}",
        ]
        base = super().extra_repr()
        if base:
            parts.append(base)
        return ", ".join(parts)
