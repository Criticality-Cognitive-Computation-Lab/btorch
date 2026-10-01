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
from numbers import Number
from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ...types import TensorLike
from .. import environ
from ..base import BaseNode
from ..ode import exp_euler_step
from ..surrogate import Erf


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
        tau_ref: Refractory period (ms). None disables refractory.
            Default: None.
        trainable_param: Set of parameter names to make trainable.
        surrogate_function: Surrogate gradient function.
            Default: None, which builds a fresh ``Erf(alpha=4,
            damping_factor=0.5)`` per neuron, matching the
            Gaussian surrogate used in Chen et al. (2022).
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
        refractory: Refractory counter (if tau_ref is not None).
        c_m, tau, tau_ref: Neuron parameters.
        k: ASC decay rates, shape (n_neuron, n_Iasc) or (n_Iasc,).
        asc_amps: ASC amplitudes, shape (n_neuron, n_Iasc) or (n_Iasc,).
        n_Iasc: Number of ASC components.

    References:
        Teeter et al., "Generalized leaky integrate-and-fire models classify
        multiple neuron types," *Nature Communications*, 2018.

        Chen, G., Scherr, F., & Maass, W. (2022). A data-based large-scale
        model for primary visual cortex enables brain-like robust and versatile
        visual processing. *Science Advances*, 8(44), eabq7592.
        https://doi.org/10.1126/sciadv.abq7592
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
        k: float
        | Sequence[float]
        | Float[TensorLike, "n_neuron {self.n_Iasc}"]
        | None = None,  # ms^-1, default [0.2]
        asc_amps: float
        | Sequence[float]
        | Float[TensorLike, "n_neuron {self.n_Iasc}"]
        | None = None,  # pA, default [0.0]
        tau_ref: float | Float[TensorLike, " n_neuron"] | None = None,  # ms
        trainable_param: set[str] | None = None,
        surrogate_function: Callable | None = None,
        detach_reset: bool = False,
        hard_reset: bool = False,
        pre_spike_v: bool = False,
        step_mode: Literal["s"] = "s",
        backend: Literal["torch"] = "torch",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        # Built per call so no mutable / Module default is shared across neurons.
        if k is None:
            k = [0.2]
        if asc_amps is None:
            asc_amps = [0.0]
        if surrogate_function is None:
            surrogate_function = Erf(alpha=4.0, damping_factor=0.5)
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
        self.def_param(
            "c_m",
            c_m,
            trainable_param=self.trainable_param,
            **_factory_kwargs,
        )
        self.def_param(
            "tau",
            tau,
            trainable_param=self.trainable_param,
            **_factory_kwargs,
        )
        self._use_refractory = tau_ref is not None
        if self._use_refractory:
            self.def_param(
                "tau_ref",
                tau_ref,
                trainable_param=self.trainable_param,
                **_factory_kwargs,
            )
            self.register_memory("refractory", 0.0, self.n_neuron)
        else:
            self.tau_ref = None

        # for compat
        if v_rest is not None:
            self.def_param(
                "_v_rest",
                v_rest,
                trainable_param=self.trainable_param,
                **_factory_kwargs,
            )
        else:
            self._v_rest = None

        if isinstance(asc_amps, Number):
            asc_amps = [asc_amps]
        if isinstance(k, Number):
            k = [k]

        resolved_asc_sizes = self.def_param_resolve_sizes(
            k,
            asc_amps,
            sizes=self.n_neuron + (None,),
        )
        self.n_Iasc: int = resolved_asc_sizes[-1]

        self.def_param(
            "k",
            k,
            sizes=resolved_asc_sizes,
            trainable_param=self.trainable_param,
            normalize_to_sizes=True,
            **_factory_kwargs,
        )
        self.def_param(
            "asc_amps",
            asc_amps,
            sizes=resolved_asc_sizes,
            trainable_param=self.trainable_param,
            normalize_to_sizes=True,
            **_factory_kwargs,
        )

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

    def dIasc(self, Iasc: Float[Tensor, "*batch n_neuron {self.n_Iasc}"]) -> tuple:
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
    ) -> tuple:
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

        self.Iasc = self.Iasc + self.asc_amps * spike_d[..., None]

        if self._use_refractory:
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
    def forward_exact_no_spike(
        self,
        x: Float[TensorLike, "*#state"] | float,
        t: float | Float[TensorLike, " n_time"] | None = None,
        v0: Float[Tensor, "*state"] | None = None,
        Iasc0: Float[Tensor, "*state {self.n_Iasc}"] | None = None,
        update_state: bool = False,
    ) -> tuple[
        Float[Tensor, " n_time *state"],
        Float[Tensor, " n_time *state {self.n_Iasc}"],
    ]:
        r"""Evaluate the closed-form no-spike trajectory at the given times.

        Between spikes the GLIF3 ODEs are linear, so for a constant input
        :math:`x` they have an exact solution. This evaluates it directly at
        every requested time (vectorised, not step by step):

        .. math::
            I_{asc}(t) = I_{asc,0} e^{-k t}

        and :math:`v(t)` is the leaky-integrator response to :math:`x` plus the
        contribution of the decaying after-spike currents. Spikes and resets are
        **not** modelled; use it for the sub-threshold regime.

        Shape contract (``state`` is the neuron state shape, i.e.
        ``(*batch, *n_neuron)``; every argument is broadcast against it from the
        right, exactly like ordinary tensor arithmetic):

        * ``v0``: ``(*state)``, defaults to the current ``self.v``.
        * ``Iasc0``: ``(*state, n_Iasc)``, defaults to the current ``self.Iasc``.
        * ``x``: constant input current, broadcastable to ``(*state)`` (a
          python/0-dim scalar, ``(n_neuron,)``, or ``(*batch, n_neuron)``). A
          per-batch scalar needs an explicit trailing axis, e.g. ``x[..., None]``.
        * ``t``: elapsed time(s) since the initial state in the same unit as
          ``dt``: a python float / 0-dim tensor (one time point) or a 1-D
          tensor ``(time,)``. Defaults to a single step ``environ.get("dt")``.

        Args:
            x: Constant input current.
            t: Elapsed time(s) since ``v0`` / ``Iasc0``.
            v0: Initial membrane potential.
            Iasc0: Initial after-spike currents.
            update_state: If True, store the state at the **last** requested
                time in ``self.v`` / ``self.Iasc`` (the state has no time axis).
                The module state is never touched otherwise.

        Returns:
            ``(v, Iasc)`` with shapes ``(time, *state)`` and
            ``(time, *state, n_Iasc)``, where the leading axis indexes ``t``.

        Raises:
            ValueError: If ``t`` is not a scalar or a 1-D tensor.
        """
        dtype, device = self.v_reset.dtype, self.v_reset.device
        if t is None:
            t = environ.get("dt")
        t = torch.as_tensor(t, dtype=dtype, device=device)
        if t.ndim == 0:
            t = t.reshape(1)
        if t.ndim != 1:
            raise ValueError(
                "t must be a scalar or a 1-D tensor of shape (time,), "
                f"got shape {tuple(t.shape)}"
            )

        x = torch.as_tensor(x, dtype=dtype, device=device)
        v0 = torch.as_tensor(self.v if v0 is None else v0, dtype=dtype, device=device)
        Iasc0 = torch.as_tensor(
            self.Iasc if Iasc0 is None else Iasc0, dtype=dtype, device=device
        )
        # Fail early (with a readable message) if the shapes cannot broadcast.
        state_shape = torch.broadcast_shapes(x.shape, v0.shape, Iasc0.shape[:-1])

        # (time, 1, ..., 1): one singleton axis per state axis.
        t_b = t.reshape(-1, *([1] * len(state_shape)))
        exp_m = torch.exp(-t_b / self.tau)  # (time, *state)
        exp_asc = torch.exp(-t_b[..., None] * self.k)  # (time, *state, n_Iasc)

        v_inf = self.v_reset + x * self.tau / self.c_m
        Iasc = Iasc0 * exp_asc

        # Contribution of Iasc to v. When tau == 1 / k the generic formula is
        # 0 / 0, so that branch uses its limit; the denominator is replaced by
        # 1 there so the unused branch stays finite (no NaN gradients).
        inv_tau = 1.0 / self.tau[..., None]
        gap = inv_tau - self.k
        degenerate = torch.abs(gap) <= 1e-12
        safe_gap = torch.where(degenerate, torch.ones_like(gap), gap)
        Iasc_c_m = Iasc0 / self.c_m[..., None]
        exp_m_asc = exp_m[..., None]
        Iasc_contrib = torch.where(
            degenerate,
            Iasc_c_m * (t_b[..., None] * exp_m_asc),
            Iasc_c_m * (exp_asc - exp_m_asc) / safe_gap,
        )
        v = v_inf + (v0 - v_inf) * exp_m + Iasc_contrib.sum(dim=-1)

        if update_state:
            self.v = v[-1]
            self.Iasc = Iasc[-1]
        return v, Iasc
