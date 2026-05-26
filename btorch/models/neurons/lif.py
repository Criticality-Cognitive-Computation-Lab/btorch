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
        backend: Literal["torch", "triton"] = "torch",
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

    @staticmethod
    def _broadcast_param(
        value: torch.Tensor,
        target_shape: tuple[int, ...],
    ) -> torch.Tensor:
        return torch.broadcast_to(value, target_shape).contiguous()

    def _should_use_triton(self, x: torch.Tensor) -> bool:
        return (
            self.backend == "triton"
            and x.is_cuda
            and not self._use_refractory
        )

    def _single_step_forward_torch(self, x: Float[Tensor, "*batch n_neuron"]):
        return BaseNode.single_step_forward(self, x)

    def _single_step_forward_triton(self, x: Float[Tensor, "*batch n_neuron"]):
        from ...backend.triton.lif import triton_lif_single_step

        state_shape = tuple(x.shape)
        if tuple(self.v.shape) != state_shape:
            self.v = self._broadcast_param(self.v, state_shape)

        v_prev = self.v
        x_flat = x.contiguous().reshape(-1)
        v_prev_flat = v_prev.reshape(-1).contiguous()
        v_threshold = self._broadcast_param(self.v_threshold, state_shape).reshape(-1)
        v_reset = self._broadcast_param(self.v_reset, state_shape).reshape(-1)
        c_m = self._broadcast_param(self.c_m, state_shape).reshape(-1)
        tau = self._broadcast_param(self.tau, state_shape).reshape(-1)

        spikes_hard_flat, v_out_flat = triton_lif_single_step(
            x_flat,
            v_prev_flat,
            v_threshold,
            v_reset,
            c_m,
            tau,
            dt=float(environ.get("dt")),
            hard_reset=self.hard_reset,
        )
        spikes_hard = spikes_hard_flat.reshape(state_shape)
        v_out = v_out_flat.reshape(state_shape)

        needs_proxy = torch.is_grad_enabled() and (
            x.requires_grad
            or v_prev.requires_grad
            or any(
                isinstance(param, torch.Tensor) and param.requires_grad
                for param in (self.v_threshold, self.v_reset, self.c_m, self.tau)
            )
        )
        needs_charge_state = self.pre_spike_v or needs_proxy

        if needs_charge_state:
            v_charged = euler_step(self.dV, v_prev, x, dt=environ.get("dt"))
            if self.pre_spike_v:
                self.v_pre_spike = v_charged.clone()
        else:
            v_charged = None

        if not needs_proxy:
            self.v = v_out
            return spikes_hard

        assert v_charged is not None
        v_scale = self.v_threshold - self.v_reset
        spike_soft = self.surrogate_function((v_charged - self.v_threshold) / v_scale)
        spike = spikes_hard.detach() + spike_soft - spike_soft.detach()
        spike_reset = spike.detach() if self.detach_reset else spike

        if self.hard_reset:
            v_proxy = v_charged - (v_charged - self.v_reset) * spike_reset
        else:
            v_proxy = v_charged - v_scale * spike_reset

        self.v = v_out.detach() + v_proxy - v_proxy.detach()
        return spike

    def single_step_forward(self, x: Float[Tensor, "*batch n_neuron"]):
        if self._should_use_triton(x):
            return self._single_step_forward_triton(x)
        return self._single_step_forward_torch(x)

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
