from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from .. import environ
from ..base import BaseNode
from ..ode import euler_step
from ..scale import SupportScaleState
from ..surrogate import Sigmoid
from ..types import TensorLike


class LIF(BaseNode, SupportScaleState):
    """Leaky integrate-and-fire neuron with optional refractory period."""

    refractory: torch.Tensor
    c_m: torch.Tensor | torch.nn.Parameter
    tau: torch.Tensor | torch.nn.Parameter
    tau_ref: torch.Tensor | torch.nn.Parameter

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        v_threshold: float | Float[TensorLike, " n_neuron"] = 1.0,
        v_reset: float | Float[TensorLike, " n_neuron"] = 0.0,
        v_rest: None | float | Float[TensorLike, " n_neuron"] = None,
        c_m: float | Float[TensorLike, " n_neuron"] = 1.0,
        tau: float | Float[TensorLike, " n_neuron"] = 20.0,
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
        self._def_param("tau", tau, **_factory_kwargs)
        self._def_param("tau_ref", tau_ref, **_factory_kwargs)

        if v_rest is not None:
            self._def_param("_v_rest", v_rest, **_factory_kwargs)
        else:
            self._v_rest = None

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

    def dV(
        self,
        v: Float[Tensor, "*batch n_neuron"],
        x: Float[Tensor, "*batch n_neuron"],
    ):
        derivative = -(v - self.v_rest) / self.tau + x / self.c_m
        return derivative

    def neuronal_charge(self, x: Float[Tensor, "*batch n_neuron"]):
        v = euler_step(self.dV, self.v, x, dt=environ.get("dt"))
        self.v = v

    def neuronal_adaptation(self):
        # LIF has no intrinsic adaptation other than the refractory counter.
        return None

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

        self.refractory = torch.relu(
            self.refractory + spike_d * self.tau_ref - environ.get("dt")
        )

    def extra_repr(self):
        return super().extra_repr()


class IF(LIF):
    """Integrate-and-fire neuron without leak."""

    def dV(
        self,
        x: Float[Tensor, "*batch n_neuron"],
    ):
        derivative = x / self.c_m
        return derivative

    def neuronal_charge(self, x: Float[Tensor, "*batch n_neuron"]):
        v = euler_step(self.dV, x, dt=environ.get("dt"))
        self.v = v
