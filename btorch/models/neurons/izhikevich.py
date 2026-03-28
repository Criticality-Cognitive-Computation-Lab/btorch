"""Izhikevich neuron model.

Efficient 2D model reproducing diverse cortical spiking patterns.
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


class Izhikevich(BaseNode):
    """Izhikevich neuron with quadratic dynamics and recovery variable.

    Efficient model reproducing diverse spiking patterns (tonic, bursting,
    etc.) via a 2D ODE system with quadratic nonlinearity.

    Dynamics:
        dv/dt = (k*(v-v_rest)*(v-v_threshold) - u + I) / c_m
        du/dt = a * (b*(v-v_rest) - u)

    At spike: v=v_reset, u=u+d

    Args:
        n_neuron: Number of neurons.
        v_threshold: Threshold (mV). Default: 30.0.
        v_reset: Reset voltage (mV). Default: -65.0.
        v_rest: Resting potential (mV). Default: -65.0.
        v_peak: Spike cutoff (mV). Default: -40.0.
        c_m: Capacitance (pF). Default: 100.0.
        k: Scaling factor (nS/mV). Default: 0.7.
        a: Recovery timescale (ms^-1). Default: 0.03.
        b: Recovery coupling (nS). Default: -2.0.
        d: Recovery jump (pA). Default: 100.0.
        trainable_param: Trainable parameters. Default: ().
        surrogate_function: Surrogate for backprop. Default: Sigmoid().
        detach_reset: Detach reset signal. Default: False.
        hard_reset: Hard vs soft reset. Default: False.
        pre_spike: Store pre-spike values. Default: False.
        step_mode: Step mode. Default: "s".
        backend: Backend. Default: "torch".
        device: Device. Default: None.
        dtype: Dtype. Default: None.

    Attributes:
        v: Membrane potential (*batch, n_neuron).
        u: Recovery variable (*batch, n_neuron).

    Reference:
        Izhikevich, IEEE Trans. Neural Networks, 2003.
    """

    HIPPOCAMPOME_TO_ARGS = {
        "k": "k",
        "a": "a",
        "b": "b",
        "d": "d",
        "C": "c_m",
        "vr": "v_rest",
        "vt": "v_threshold",
        "vpeak": "v_peak",
        "vmin": "v_reset",
    }

    u: torch.Tensor
    u_pre_spike: torch.Tensor

    v_reset: torch.Tensor | torch.nn.Parameter
    v_rest: torch.Tensor | torch.nn.Parameter
    v_peak: torch.Tensor | torch.nn.Parameter
    c_m: torch.Tensor | torch.nn.Parameter
    k: torch.Tensor | torch.nn.Parameter
    a: torch.Tensor | torch.nn.Parameter
    b: torch.Tensor | torch.nn.Parameter
    d: torch.Tensor | torch.nn.Parameter

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        v_threshold: float | Float[TensorLike, " n_neuron"] = 30.0,
        v_reset: float | Float[TensorLike, " n_neuron"] = -65.0,
        v_rest: float | Float[TensorLike, " n_neuron"] = -65.0,
        v_peak: float | Float[TensorLike, " n_neuron"] = -40.0,
        c_m: float | Float[TensorLike, " n_neuron"] = 100.0,
        k: float | Float[TensorLike, " n_neuron"] = 0.7,
        a: float | Float[TensorLike, " n_neuron"] = 0.03,
        b: float | Float[TensorLike, " n_neuron"] = -2.0,
        d: float | Float[TensorLike, " n_neuron"] = 100.0,
        trainable_param: set[str] = set(),
        surrogate_function: Callable = Sigmoid(),
        detach_reset: bool = False,
        hard_reset: bool = False,
        pre_spike: bool = False,
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
            pre_spike_v=pre_spike,
            step_mode=step_mode,
            backend=backend,
            device=device,
            dtype=dtype,
        )
        _factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self._def_param("c_m", c_m, **_factory_kwargs)
        self._def_param("v_rest", v_rest, **_factory_kwargs)
        self._def_param("v_peak", v_peak, **_factory_kwargs)
        self._def_param("k", k, **_factory_kwargs)
        self._def_param("a", a, **_factory_kwargs)
        self._def_param("b", b, **_factory_kwargs)
        self._def_param("d", d, **_factory_kwargs)

        self.register_memory("u", 0, self.n_neuron)
        if pre_spike:
            self.register_memory("u_pre_spike", None, self.n_neuron)

    @classmethod
    def from_hippocampome(
        cls,
        n_neuron: int | Sequence[int],
        k,
        a,
        b,
        d,
        C,
        vr,
        vt,
        vpeak,
        vmin,
        **kwargs,
    ):
        """
        Build an :class:`Izhikevich` neuron using parameter names from
        https://hippocampome.org.

        Parameter mapping (HippoCampome -> Izhikevich args):
        - k -> k (scaling factor)
        - a -> a (recovery time constant)
        - b -> b (recovery sensitivity)
        - d -> d (reset current)
        - C -> c_m (capacitance)
        - vr -> v_rest (resting potential)
        - vt -> v_threshold (instantaneous threshold)
        - vpeak -> v_peak (spike cutoff)
        - vmin -> v_reset (post-spike reset voltage)

        All values are expected in the same units as the canonical
        Izhikevich model (mV, pF, pA).
        """
        kwargs.setdefault("pre_spike", True)
        return cls(
            n_neuron,
            v_threshold=vt,
            v_reset=vmin,
            v_rest=vr,
            v_peak=vpeak,
            c_m=C,
            k=k,
            a=a,
            b=b,
            d=d,
            **kwargs,
        )

    @classmethod
    def from_canonical_quadratic(
        cls,
        n_neuron: int | Sequence[int],
        p1: float = 0.04,
        p2: float = 5.0,
        # TODO: p3: float = 0.0, adjust equation
        v_rest: float = -65.0,
        c_m: float = 1.0,
        v_peak: float = 30.0,
        **kwargs,
    ):
        """
        Instantiate using the canonical quadratic form
        ``dV/dt = p1*v^2 + p2*v + p3 - u + I``.

        The mapping assumes ``c_m`` acts as the membrane capacitance and that
        ``k/c_m`` equals ``p1``. The linear term enforces
        ``v_threshold = -p2/p1 - v_rest``. Remaining
        keyword arguments are passed directly to :class:`Izhikevich`.
        """
        k = p1 * c_m
        v_threshold = -p2 / p1 - v_rest
        # i_bias = p3 - p1 * v_rest * v_threshold

        return cls(
            n_neuron,
            v_threshold=v_threshold,
            v_reset=kwargs.pop("v_reset", v_rest),
            v_rest=v_rest,
            v_peak=v_peak,
            c_m=c_m,
            k=k,
            **kwargs,
        )

    def dV(
        self,
        v: Float[Tensor, "*batch n_neuron"],
        u: Float[Tensor, "*batch n_neuron"],
        x: Float[Tensor, "*batch n_neuron"],
    ):
        quadratic = self.k * (v - self.v_rest) * (v - self.v_threshold)
        return (x + quadratic - u) / self.c_m

    def dU(
        self,
        u: Float[Tensor, "*batch n_neuron"],
        v: Float[Tensor, "*batch n_neuron"],
    ):
        return self.a * (self.b * (v - self.v_rest) - u)

    def neuronal_charge(self, x: Float[Tensor, "*batch n_neuron"]):
        dt = environ.get("dt")
        self.v = euler_step(self.dV, self.v, self.u, x, dt=dt)

    def neuronal_adaptation(self):
        dt = environ.get("dt")
        self.u = euler_step(self.dU, self.u, self.v, dt=dt)

    def neuronal_fire(self):
        # TODO: confirm scaling with (self.v_threshold - self.v_reset)
        # or (self.v_peak - self.v_reset)
        spike = self.surrogate_function(
            (self.v - self.v_peak) / (self.v_threshold - self.v_reset)
        )
        return spike

    def neuronal_reset(self, spike: Float[Tensor, "*batch n"]):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.pre_spike_v:
            self.v_pre_spike = self.v.clone()
            self.u_pre_spike = self.u.clone()

        if self.hard_reset:
            self.v = self.v - (self.v - self.v_reset) * spike_d
        else:
            self.v = self.v - (self.v_peak - self.v_reset) * spike_d

        self.u = self.u + self.d * spike_d

    def extra_repr(self):
        parts = [
            f"c_m={self._format_repr_value(self.c_m)}",
            f"k={self._format_repr_value(self.k)}",
            f"a={self._format_repr_value(self.a)}",
            f"b={self._format_repr_value(self.b)}",
            f"d={self._format_repr_value(self.d)}",
            f"v_rest={self._format_repr_value(self.v_rest)}",
            f"v_peak={self._format_repr_value(self.v_peak)}",
        ]
        base = super().extra_repr()
        if base:
            parts.append(base)
        return ", ".join(parts)
