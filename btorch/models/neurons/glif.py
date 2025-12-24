from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from .. import environ
from ..base import BaseNode
from ..ode import exp_euler_step_auto
from ..scale import SupportScaleState
from ..shape import expand_trailing_dims
from ..surrogate import ATan
from ..types import TensorLike


class GLIF3(BaseNode, SupportScaleState):
    """GLIF3 model as described in [1]. Leaky integrate and fire model with
    refractory period and after spike currents.

    TODO: support parameter scatter

    [1] C. Teeter et al., "Generalized leaky integrate-and-fire models
    classify multiple neuron types," Nat. Commun., vol. 9, no. 1, p.
    709, Feb. 2018, doi: 10.1038/s41467-017-02717-4.
    """

    # make mypy typing and autocompletion easier
    Iasc: torch.Tensor
    refractory: torch.Tensor

    c_m: torch.Tensor | torch.nn.Parameter
    tau: torch.Tensor | torch.nn.Parameter
    tau_ref: torch.Tensor | torch.nn.Parameter
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
        tau_ref: float | Float[TensorLike, " n_neuron"] = 0.0,  # ms
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
        self._def_param("tau_ref", tau_ref, **_factory_kwargs)

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
        self.register_memory("refractory", 0.0, self.n_neuron)

    @property
    def v_rest(self):
        """For compat with glif4, glif5, use v_reset instead."""
        if self._v_rest is None:
            return self.v_reset
        return self._v_rest

    @v_rest.setter
    def v_rest(self, v_rest):
        if self._v_rest is not None:
            self._v_rest = v_rest

    def dIasc(self, Iasc: Float[Tensor, "*batch n_neuron {self.n_Iasc}"]):
        return -self.k * Iasc

    def dV(
        self,
        v: Float[Tensor, "*batch n_neuron"],
        Iasc: Float[Tensor, "*batch n_neuron {self.n_Iasc}"],
        x: Float[Tensor, "*batch n_neuron"],
    ):
        Isum = x
        # torch.autocast will cast half to float32 for sum op
        # see https://docs.pytorch.org/docs/stable/amp.html#ops-that-can-autocast-to-float32
        # here Iasc generally only have <4 modes, so no overflow guaranteed
        return (
            -(v - self.v_rest) / self.tau
            + (Isum + Iasc.sum(-1, dtype=Iasc.dtype)) / self.c_m
        )

    def neuronal_charge(self, x: Float[Tensor, "*batch n_neuron"]):
        v = exp_euler_step_auto(self.dV, self.v, self.Iasc, x, dt=environ.get("dt"))
        self.v = v

    def neuronal_adaptation(self):
        self.Iasc = exp_euler_step_auto(self.dIasc, self.Iasc, dt=environ.get("dt"))

    def neuronal_fire(self):
        # Check if voltage exceeds threshold and not in refractory period
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
            # hard reset
            self.v -= (self.v - self.v_reset) * spike_d
        else:
            # soft reset
            self.v -= (self.v_threshold - self.v_reset) * spike_d

        # Add after-spike currents
        self.Iasc += self.asc_amps * spike_d[..., None]

        # Set refractory period
        self.refractory = torch.relu(
            self.refractory + spike_d * self.tau_ref - environ.get("dt")
        )

    def extra_repr(self):
        # TODO
        return super().extra_repr()

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
