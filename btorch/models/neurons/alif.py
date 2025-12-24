from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from .. import environ
from ..base import BaseNode
from ..ode import exp_euler_step_auto
from ..surrogate import Sigmoid
from ..types import TensorLike


class ALIF(BaseNode):
    """Adaptive leaky integrate-and-fire neuron with conductance-based
    adaptation.

    The model follows a simple conductance formulation:

    dv/dt = (-g_leak * (v - E_leak) - g_k * (v - E_k) + x) / c_m
    dg_k/dt = -g_k / tau_adapt
    """

    g_k: torch.Tensor
    refractory: torch.Tensor

    c_m: torch.Tensor | torch.nn.Parameter
    g_leak: torch.Tensor | torch.nn.Parameter
    E_leak: torch.Tensor | torch.nn.Parameter
    E_k: torch.Tensor | torch.nn.Parameter
    tau_adapt: torch.Tensor | torch.nn.Parameter
    dg_k: torch.Tensor | torch.nn.Parameter
    tau_ref: torch.Tensor | torch.nn.Parameter

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
        tau_ref: float | Float[TensorLike, " n_neuron"] = 0.0,
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
        self._def_param("tau_ref", tau_ref, **_factory_kwargs)

        self.register_memory("g_k", g_k_init, self.n_neuron)
        self.register_memory("refractory", 0.0, self.n_neuron)

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
        return (leak_term + adapt_term + x) / self.c_m

    def dgk(
        self, g_k: Float[Tensor, "*batch n_neuron"]
    ) -> Float[Tensor, "*batch n_neuron"]:
        return -g_k / self.tau_adapt

    def neuronal_charge(self, x: Float[Tensor, "*batch n_neuron"]):
        dt = environ.get("dt")
        self.v = exp_euler_step_auto(self.dV, self.v, self.g_k, x, dt=dt)

    def neuronal_adaptation(self):
        dt = environ.get("dt")
        self.g_k = exp_euler_step_auto(self.dgk, self.g_k, dt=dt)

    def neuronal_fire(self):
        not_in_refractory = self.refractory == 0
        spike = self.surrogate_function(
            (self.v - self.v_threshold) / (self.v_threshold - self.v_reset)
        ) * not_in_refractory.detach().to(self.v.dtype)
        return spike

    def neuronal_reset(self, spike: Float[Tensor, "*batch n"]):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.pre_spike_v:
            self.v_pre_spike = self.v.clone()

        if self.hard_reset:
            self.v -= (self.v - self.v_reset) * spike_d
        else:
            self.v -= (self.v_threshold - self.v_reset) * spike_d

        self.g_k += self.dg_k * spike_d

        self.refractory = torch.relu(
            self.refractory + spike_d * self.tau_ref - environ.get("dt")
        )

    def extra_repr(self):
        return super().extra_repr()


class ELIF(ALIF):
    """Exponential integrate-and-fire neuron with conductance-based
    adaptation."""

    delta_T: torch.Tensor | torch.nn.Parameter
    v_T: torch.Tensor | torch.nn.Parameter

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        v_threshold: float | Float[TensorLike, " n_neuron"] = 1.0,
        v_reset: float | Float[TensorLike, " n_neuron"] = 0.0,
        v_rest: None | float | Float[TensorLike, " n_neuron"] = None,
        c_m: float | Float[TensorLike, " n_neuron"] = 1.0,
        g_leak: float | Float[TensorLike, " n_neuron"] = 1.0,
        E_leak: float | Float[TensorLike, " n_neuron"] = 0.0,
        E_k: float | Float[TensorLike, " n_neuron"] = -70.0,
        g_k_init: float | Float[TensorLike, " n_neuron"] = 0.0,
        tau_adapt: float | Float[TensorLike, " n_neuron"] = 20.0,
        dg_k: float | Float[TensorLike, " n_neuron"] = 0.0,
        tau_ref: float | Float[TensorLike, " n_neuron"] = 0.0,
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
        return (leak_term + adapt_term + exp_term + x) / self.c_m

    def neuronal_charge(self, x: Float[Tensor, "*batch n_neuron"]):
        dt = environ.get("dt")
        self.v = exp_euler_step_auto(self.dV, self.v, self.g_k, x, dt=dt)

    def extra_repr(self):
        return super().extra_repr()
