"""Leaky integrate-and-fire (LIF) neuron models.

This module provides LIF and IF (integrate-and-fire) neuron implementations
with optional refractory periods. These are the simplest spiking neuron models,
suitable for basic neuromorphic computing tasks and as building blocks for
more complex networks.

The LIF neuron follows the dynamics:
    dV/dt = -(V - V_reset) / tau + I / c_m

where V is membrane potential, tau is the time constant, I is input current,
and c_m is membrane capacitance.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ...types import TensorLike
from .. import environ
from ..base import BaseNode
from ..ode import euler_step
from ..surrogate import Sigmoid


class LIF(BaseNode):
    """Leaky integrate-and-fire neuron with optional refractory period.

    The LIF neuron integrates input current while leaking towards a resting
    potential. When the membrane potential exceeds a threshold, a spike is
    emitted and the potential is reset.

    Dynamics:
        dV/dt = -(V - V_reset) / tau + I / c_m

        If tau_ref is specified, a refractory period prevents spiking for
        tau_ref milliseconds after each spike.

    Args:
        n_neuron: Number of neurons (int or tuple of dimensions).
        v_threshold: Firing threshold (mV). Default: 1.0.
        v_reset: Reset voltage after spike (mV). Default: 0.0.
        c_m: Membrane capacitance. Default: 1.0.
        tau: Membrane time constant (ms). Default: 20.0.
        tau_ref: Refractory period duration (ms). None disables refractory
            behavior. Default: None.
        trainable_param: Set of parameter names to make trainable.
            Default: empty set.
        surrogate_function: Surrogate gradient function for backpropagation.
            Default: Sigmoid().
        detach_reset: If True, detach reset signal from computation graph.
            Default: False.
        hard_reset: If True, reset to v_reset directly. If False, subtract
            (v_threshold - v_reset) from membrane potential (soft reset).
            Default: False.
        pre_spike_v: If True, store pre-spike voltage in v_pre_spike buffer.
            Default: False.
        step_mode: Step mode, currently only "s" (single step) supported.
            Default: "s".
        backend: Backend implementation. Default: "torch".
        device: Device for tensors. Default: None.
        dtype: Data type for tensors. Default: None.

    Attributes:
        v: Membrane potential tensor, shape (*batch, n_neuron).
        refractory: Refractory counter (if tau_ref specified).
        c_m: Membrane capacitance (parameter or buffer).
        tau: Time constant (parameter or buffer).
        tau_ref: Refractory period (parameter or buffer, or None).

    Shape:
        - Input: (*batch, n_neuron)
        - Output: (*batch, n_neuron) spike tensor (0 or 1)
    """

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
            self.v = self.v - (self.v - self.v_reset) * spike_d
        else:
            self.v = self.v - (self.v_threshold - self.v_reset) * spike_d

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
    """Integrate-and-fire neuron without leak.

    Simplified variant of LIF that lacks the leak term, meaning the membrane
    potential integrates input current linearly without decay:

        dV/dt = I / c_m

    This model is useful for theoretical analysis and as a baseline,
    though it lacks biological realism due to unbounded integration.

    Args:
        n_neuron: Number of neurons (int or tuple of dimensions).
        v_threshold: Firing threshold. Default: 1.0.
        v_reset: Reset voltage after spike. Default: 0.0.
        c_m: Membrane capacitance. Default: 1.0.
        tau: Time constant (inherited from LIF but not used in dynamics).
        tau_ref: Refractory period duration. Default: None.
        trainable_param: Set of parameter names to make trainable.
        surrogate_function: Surrogate gradient function. Default: Sigmoid().
        detach_reset: If True, detach reset signal. Default: False.
        hard_reset: If True, use hard reset. Default: False.
        pre_spike_v: If True, store pre-spike voltage. Default: False.
        step_mode: Step mode. Default: "s".
        backend: Backend implementation. Default: "torch".
        device: Device for tensors. Default: None.
        dtype: Data type for tensors. Default: None.
    """

    def dV(
        self,
        x: Float[Tensor, "*batch n_neuron"],
    ) -> Float[Tensor, "*batch n_neuron"]:
        """Compute membrane potential derivative (no leak term).

        Args:
            x: Input current, shape (*batch, n_neuron).

        Returns:
            dV/dt derivative, shape (*batch, n_neuron).
        """
        derivative = x / self.c_m
        return derivative

    def neuronal_charge(self, x: Float[Tensor, "*batch n_neuron"]):
        """Update membrane potential using Euler integration (no leak).

        Args:
            x: Input current, shape (*batch, n_neuron).
        """
        v = euler_step(self.dV, x, dt=environ.get("dt"))
        self.v = v
