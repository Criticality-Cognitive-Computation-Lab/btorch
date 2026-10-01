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

    def exact_no_spike_at(
        self,
        x: Float[TensorLike, "*#state"] | float,
        t: Float[TensorLike, "*#state"] | float,
        v0: Float[Tensor, "*state"] | None = None,
        Iasc0: Float[Tensor, "*state {self.n_Iasc}"] | None = None,
    ) -> tuple[Float[Tensor, "*state"], Float[Tensor, "*state {self.n_Iasc}"]]:
        r"""Closed-form no-spike state ``t`` after ``(v0, Iasc0)``, elementwise.

        Between spikes the GLIF3 ODEs are linear, so for a constant input
        :math:`x` they have an exact solution:

        .. math::
            I_{asc}(t) = I_{asc,0} e^{-k t}

        and :math:`v(t)` is the leaky-integrator response to :math:`x` plus the
        contribution of the decaying after-spike currents. Spikes and resets are
        **not** modelled (sub-threshold regime).

        This is the elementwise primitive: there is **no time axis**. Every
        argument is broadcast against the others from the right, like ordinary
        tensor arithmetic, so each batch element and each neuron can have its own
        ``x``, ``t``, ``v0`` and ``Iasc0`` (and its own parameters). ``state`` below
        is the broadcast of ``x``, ``t``, ``v0`` and ``Iasc0[..., 0]``:

        * ``x``: constant input current, ``(*#state)`` (a python/0-dim scalar,
          ``(n_neuron,)``, ``(*batch, n_neuron)``, ...). A per-batch scalar needs
          an explicit trailing axis, e.g. ``x[..., None]``.
        * ``t``: elapsed time since ``v0`` / ``Iasc0``, ``(*#state)``, in the same
          unit as ``dt``.
        * ``v0``: ``(*state)``, defaults to the current ``self.v``.
        * ``Iasc0``: ``(*state, n_Iasc)``, defaults to the current ``self.Iasc``.

        It is a pure, differentiable function (the module state is only read,
        and gradients flow to ``t``, ``x``, ``v0`` and ``Iasc0``), which makes it
        a building block for iterative root finding such as "when does ``v``
        reach the threshold?". For example one Newton step on ``v(t) - v_th``,
        with ``dv/dt`` from the model's own ODE (:meth:`dV`):

        >>> v, Iasc = neuron.exact_no_spike_at(x, t, v0, Iasc0)
        >>> dv_dt, _ = neuron.dV(v, Iasc, x)
        >>> t = t - (v - neuron.v_threshold) / dv_dt

        Args:
            x: Constant input current.
            t: Elapsed time since ``v0`` / ``Iasc0``.
            v0: Initial membrane potential.
            Iasc0: Initial after-spike currents.

        Returns:
            ``(v, Iasc)`` of shapes ``(*state)`` and ``(*state, n_Iasc)``.

        Raises:
            RuntimeError: If ``x``, ``t``, ``v0`` and ``Iasc0`` do not broadcast
                (raised by torch).
        """
        t = torch.as_tensor(t, dtype=self.v_reset.dtype, device=self.v_reset.device)
        v0 = self.v if v0 is None else v0
        Iasc0 = self.Iasc if Iasc0 is None else Iasc0

        exp_m = torch.exp(-t / self.tau)  # (*state)
        exp_asc = torch.exp(-t[..., None] * self.k)  # (*state, n_Iasc)

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
            Iasc_c_m * (t[..., None] * exp_m_asc),
            Iasc_c_m * (exp_asc - exp_m_asc) / safe_gap,
        )
        v = v_inf + (v0 - v_inf) * exp_m + Iasc_contrib.sum(dim=-1)

        # ``v`` already has the full broadcast shape; ``Iasc`` only depends on
        # ``Iasc0`` and ``t``, so expand it to match (a no-op when already full).
        Iasc = Iasc.expand(*v.shape, Iasc.shape[-1]).contiguous()
        return v, Iasc

    def forward_exact_no_spike(
        self,
        x: Float[TensorLike, "*#state"] | float,
        t: float | Float[TensorLike, " n_time *#state"] | None = None,
        v0: Float[Tensor, "*state"] | None = None,
        Iasc0: Float[Tensor, "*state {self.n_Iasc}"] | None = None,
        t_mode: Literal["homo", "heter"] = "homo",
    ) -> tuple[
        Float[Tensor, " n_time *state"],
        Float[Tensor, " n_time *state {self.n_Iasc}"],
    ]:
        r"""Evaluate the closed-form no-spike trajectory at several times.

        The trajectory version of :meth:`exact_no_spike_at` (see there for the
        maths and the broadcasting rules). The first axis of ``t`` is always the
        time axis; ``t_mode`` says how the remaining axes are read, so the layout
        is never guessed from shapes:

        * ``"homo"`` (default): one time grid shared by every batch element and
          neuron. ``t`` is a python float / 0-dim tensor (one time point) or a
          1-D tensor ``(n_time,)``; any other shape raises ``ValueError``.
        * ``"heter"``: the grid may differ per batch element and/or neuron. The
          axes of ``t`` after the time axis are aligned to the right of ``state``
          and broadcast against it:

          ========================  ==========================================
          ``t`` shape               meaning
          ========================  ==========================================
          ``(n_time, n_neuron)``    a different time per neuron
          ``(n_time, batch, 1)``    a different time per batch element
          ``(n_time, batch, n)``    a different time per batch element and neuron
          ========================  ==========================================

        If ``t`` carries batch/neuron axes that ``v0`` / ``Iasc0`` / ``x`` lack,
        the output gains them. ``t`` defaults to a single step
        ``environ.get("dt")``. For heterogeneous times *without* a time axis (e.g.
        inside a root finder) call :meth:`exact_no_spike_at` directly.

        This is a pure function: the module state (``self.v`` /
        ``self.Iasc``) is only read, never written. To continue a simulation
        from the end of the trajectory, assign the last time point yourself:

        >>> v, Iasc = neuron.forward_exact_no_spike(x, t=t)
        >>> neuron.v, neuron.Iasc = v[-1], Iasc[-1]

        Args:
            x: Constant input current.
            t: Elapsed time(s) since ``v0`` / ``Iasc0``, time axis first.
            v0: Initial membrane potential.
            Iasc0: Initial after-spike currents.
            t_mode: ``"homo"`` for a grid shared by everything, ``"heter"`` for
                a grid that varies per batch element and/or neuron.

        Returns:
            ``(v, Iasc)`` with shapes ``(n_time, *state)`` and
            ``(n_time, *state, n_Iasc)``.

        Raises:
            ValueError: If ``t_mode`` is unknown, or ``t_mode="homo"`` and ``t``
                is not a scalar or 1-D.
            RuntimeError: If ``x``, ``v0``, ``Iasc0`` and the trailing axes of
                ``t`` do not broadcast (raised by torch).
        """
        if t_mode not in ("homo", "heter"):
            raise ValueError(f"t_mode must be 'homo' or 'heter', got {t_mode!r}")
        if t is None:
            t = environ.get("dt")
        t = torch.as_tensor(t, dtype=self.v_reset.dtype, device=self.v_reset.device)
        t = t.reshape(1) if t.ndim == 0 else t  # one shared time point
        if t_mode == "homo" and t.ndim != 1:
            raise ValueError(
                f"t_mode='homo' needs a scalar or 1-D t of shape (n_time,), got "
                f"{tuple(t.shape)}; use t_mode='heter' for per-element times"
            )
        v0 = self.v if v0 is None else v0
        Iasc0 = self.Iasc if Iasc0 is None else Iasc0
        # Align the trailing axes of ``t`` to the right of the state.
        state_ndim = max(torch.as_tensor(x).ndim, v0.ndim, Iasc0.ndim - 1, t.ndim - 1)
        t = t.reshape(t.shape[0], *([1] * (state_ndim - t.ndim + 1)), *t.shape[1:])
        return self.exact_no_spike_at(x, t, v0, Iasc0)
