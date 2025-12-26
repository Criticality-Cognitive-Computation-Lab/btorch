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

    refractory: torch.Tensor | None
    c_m: torch.Tensor | torch.nn.Parameter
    tau: torch.Tensor | torch.nn.Parameter
    tau_ref: torch.Tensor | torch.nn.Parameter | None

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        v_threshold: float | Float[TensorLike, " n_neuron"] = 1.0,
        v_reset: float | Float[TensorLike, " n_neuron"] = 0.0,
        c_m: float | Float[TensorLike, " n_neuron"] = 1.0,
        tau: float | Float[TensorLike, " n_neuron"] = 20.0,
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
        self._def_param("tau", tau, **_factory_kwargs)
        self._use_refractory = tau_ref is not None
        if self._use_refractory:
            self._def_param("tau_ref", tau_ref, **_factory_kwargs)
            self.register_memory("refractory", 0.0, self.n_neuron)
        else:
            self.tau_ref = None

    def dV(
        self,
        v: Float[Tensor, "*batch n_neuron"],
        x: Float[Tensor, "*batch n_neuron"],
    ):
        derivative = -(v - self.v_reset) / self.tau + x / self.c_m
        return derivative

    def neuronal_charge(self, x: Float[Tensor, "*batch n_neuron"]):
        v = euler_step(self.dV, self.v, x, dt=environ.get("dt"))
        self.v = v

    def neuronal_adaptation(self):
        # LIF has no intrinsic adaptation other than the refractory counter.
        return None

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
            self.v -= (self.v - self.v_reset) * spike_d
        else:
            self.v -= (self.v_threshold - self.v_reset) * spike_d

        if self._use_refractory:
            self.refractory = torch.relu(
                self.refractory + spike_d * self.tau_ref - environ.get("dt")
            )

    def extra_repr(self):
        parts = [
            f"c_m={self._format_repr_value(self.c_m)}",
            f"tau={self._format_repr_value(self.tau)}",
            f"tau_ref={self._format_repr_value(self.tau_ref)}"
            if self._use_refractory
            else "tau_ref=None",
        ]
        base = super().extra_repr()
        if base:
            parts.append(base)
        return ", ".join(parts)


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
