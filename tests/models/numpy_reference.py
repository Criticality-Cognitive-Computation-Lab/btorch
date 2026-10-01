"""Pure-numpy reference implementation of btorch neurons, PSCs and networks.

This module is a *test oracle*. It re-implements the discrete-time dynamics of
the btorch models (``LIF``, ``GLIF3`` and the ``ExponentialPSC``, ``AlphaPSC``,
``AlphaPSCBilleh``, ``DualExponentialPSC`` synapses) with plain numpy and no
torch dependency, deliberately written independently of the torch code. The
parity tests in ``tests/models/test_numpy_parity.py`` step the torch modules
and these references side by side, so a regression in either implementation
(timing, update order, formula) shows up as a mismatch.

Everything runs in float64 (:data:`DTYPE`) so parity can be checked to near
machine precision.

Timing convention (identical to btorch, see ``BasePSC`` and
``tests/models/neurons/test_neuron_timing.py``):

* **PSCs**: a spike delivered to ``step(z, dt)`` at step ``t`` already acts on
  the PSC returned at step ``t``, for *every* PSC type (the impulse response
  starts at the delivery step, delay 0).
* **Neurons**: the input ``x`` given at step ``t`` reaches the membrane
  potential and the emitted spike of step ``t``. The reset and the adaptation
  caused by that spike (``v`` reset, after-spike currents, refractory
  counter) are applied at the end of step ``t`` and first act from step
  ``t + 1``.
* **Network**: ``z_t = neuron(sum_k psc_k[t-1] + x_t)`` followed by
  ``psc_k[t] = synapse_k(z_t)``, i.e. recurrent feedback has a one-step
  latency, exactly like ``btorch.models.rnn.RecurrentNN``.

Weights are stored as ``(n_in, n_out)`` matrices (``y = z @ W``), either dense
``np.ndarray`` or any ``scipy.sparse`` array.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse
from numpy.typing import ArrayLike, NDArray


DTYPE = np.float64

Array = NDArray[np.float64]
Weight = Array | scipy.sparse.sparray


def _arr(x: ArrayLike) -> Array:
    """Convert to a float64 array."""
    return np.asarray(x, dtype=DTYPE)


def _project(weight: Weight, z: Array) -> Array:
    """Apply a ``(n_in, n_out)`` weight (dense or sparse) to spikes ``z``."""
    return np.asarray(z @ weight, dtype=DTYPE)


# --------------------------------------------------------------------------- #
# Neurons
# --------------------------------------------------------------------------- #


class LIF:
    """Leaky integrate-and-fire neuron with optional refractory period.

    Mirrors :class:`btorch.models.neurons.lif.LIF`. Forward Euler for the
    membrane:

    .. math::
        v \\leftarrow v + dt \\left(-\\frac{v - v_{reset}}{\\tau}
        + \\frac{x}{c_m}\\right)

    A spike is emitted when ``v >= v_threshold`` (and the neuron is not
    refractory); the reset then subtracts ``v_threshold - v_reset`` (soft) or
    sets ``v = v_reset`` (hard).

    Args:
        n_neuron: Number of neurons.
        v_threshold: Firing threshold, scalar or ``(n_neuron,)``.
        v_reset: Reset (and resting) potential, scalar or ``(n_neuron,)``.
        c_m: Membrane capacitance, scalar or ``(n_neuron,)``.
        tau: Membrane time constant, scalar or ``(n_neuron,)``.
        tau_ref: Refractory duration; ``None`` disables the refractory period.
        hard_reset: Use hard instead of soft reset.

    Attributes:
        v: Membrane potential, shape ``(n_neuron,)``; starts at ``v_reset``.
        refractory: Remaining refractory time, shape ``(n_neuron,)``.
    """

    def __init__(
        self,
        n_neuron: int,
        v_threshold: ArrayLike = 1.0,
        v_reset: ArrayLike = 0.0,
        c_m: ArrayLike = 1.0,
        tau: ArrayLike = 20.0,
        tau_ref: ArrayLike | None = None,
        hard_reset: bool = False,
    ):
        self.n_neuron = n_neuron
        self.v_threshold = _arr(v_threshold)
        self.v_reset = _arr(v_reset)
        self.c_m = _arr(c_m)
        self.tau = _arr(tau)
        self.tau_ref = None if tau_ref is None else _arr(tau_ref)
        self.hard_reset = hard_reset

        self.v = np.broadcast_to(self.v_reset, (n_neuron,)).copy()
        self.refractory = np.zeros(n_neuron, dtype=DTYPE)

    def step(self, x: ArrayLike, dt: float) -> Array:
        """Advance one step with input current ``x``; return spikes (0/1)."""
        x = _arr(x)
        # charge: forward Euler, input acts on this step's v
        self.v = self.v + dt * (-(self.v - self.v_reset) / self.tau + x / self.c_m)
        # fire
        spike = self.v >= self.v_threshold
        if self.tau_ref is not None:
            spike = spike & (self.refractory == 0)
        spike = spike.astype(DTYPE)
        # reset: acts from the next step
        if self.hard_reset:
            self.v = self.v - (self.v - self.v_reset) * spike
        else:
            self.v = self.v - (self.v_threshold - self.v_reset) * spike
        if self.tau_ref is not None:
            self.refractory = np.maximum(
                self.refractory + spike * self.tau_ref - dt, 0.0
            )
        return spike


class GLIF3:
    """GLIF3 neuron: LIF with after-spike currents (ASC).

    Mirrors :class:`btorch.models.neurons.glif.GLIF3`:

    .. math::
        \\frac{dv}{dt} = -\\frac{v - v_{rest}}{\\tau}
        + \\frac{x + \\sum_j I_{asc,j}}{c_m}, \\qquad
        \\frac{dI_{asc,j}}{dt} = -k_j I_{asc,j}

    Both equations are advanced with exponential Euler, ``v`` first (using the
    ``I_asc`` of the previous step), then ``I_asc``. At a spike ``I_asc`` is
    incremented by ``asc_amps``.

    Args:
        n_neuron: Number of neurons.
        v_threshold: Firing threshold, scalar or ``(n_neuron,)``.
        v_reset: Reset potential, scalar or ``(n_neuron,)``.
        c_m: Membrane capacitance, scalar or ``(n_neuron,)``.
        tau: Membrane time constant, scalar or ``(n_neuron,)``.
        k: ASC decay rates, ``(n_asc,)`` or ``(n_neuron, n_asc)``.
        asc_amps: ASC increments at a spike, same shapes as ``k``.
        v_rest: Resting potential; defaults to ``v_reset``.
        tau_ref: Refractory duration; ``None`` disables the refractory period.
        hard_reset: Use hard instead of soft reset.

    Attributes:
        v: Membrane potential, shape ``(n_neuron,)``; starts at ``v_reset``.
        Iasc: After-spike currents, shape ``(n_neuron, n_asc)``.
        refractory: Remaining refractory time, shape ``(n_neuron,)``.
    """

    def __init__(
        self,
        n_neuron: int,
        v_threshold: ArrayLike,
        v_reset: ArrayLike,
        c_m: ArrayLike,
        tau: ArrayLike,
        k: Sequence[float] | ArrayLike,
        asc_amps: Sequence[float] | ArrayLike,
        v_rest: ArrayLike | None = None,
        tau_ref: ArrayLike | None = None,
        hard_reset: bool = False,
    ):
        self.n_neuron = n_neuron
        self.v_threshold = _arr(v_threshold)
        self.v_reset = _arr(v_reset)
        self.v_rest = self.v_reset if v_rest is None else _arr(v_rest)
        self.c_m = _arr(c_m)
        self.tau = _arr(tau)
        self.k = _arr(k)
        self.asc_amps = _arr(asc_amps)
        self.tau_ref = None if tau_ref is None else _arr(tau_ref)
        self.hard_reset = hard_reset
        self.n_asc = self.k.shape[-1]

        self.v = np.broadcast_to(self.v_reset, (n_neuron,)).copy()
        self.Iasc = np.zeros((n_neuron, self.n_asc), dtype=DTYPE)
        self.refractory = np.zeros(n_neuron, dtype=DTYPE)

    def step(self, x: ArrayLike, dt: float) -> Array:
        """Advance one step with input current ``x``; return spikes (0/1)."""
        x = _arr(x)
        # charge: exponential Euler for v with the previous step's Iasc
        a = -1.0 / self.tau
        deriv = -(self.v - self.v_rest) / self.tau + (x + self.Iasc.sum(-1)) / self.c_m
        self.v = self.v + np.expm1(dt * a) / a * deriv
        # adaptation: Iasc decays exactly
        self.Iasc = self.Iasc * np.exp(-dt * self.k)
        # fire
        spike = self.v >= self.v_threshold
        if self.tau_ref is not None:
            spike = spike & (self.refractory == 0)
        spike = spike.astype(DTYPE)
        # reset: acts from the next step
        if self.hard_reset:
            self.v = self.v - (self.v - self.v_reset) * spike
        else:
            self.v = self.v - (self.v_threshold - self.v_reset) * spike
        self.Iasc = self.Iasc + self.asc_amps * spike[:, None]
        if self.tau_ref is not None:
            self.refractory = np.maximum(
                self.refractory + spike * self.tau_ref - dt, 0.0
            )
        return spike


# --------------------------------------------------------------------------- #
# Post-synaptic currents
# --------------------------------------------------------------------------- #


class ExponentialPSC:
    """Single-exponential PSC: ``psc <- psc * exp(-dt/tau) + z @ W``.

    Args:
        weight: ``(n_in, n_out)`` weight, dense or scipy sparse.
        tau_syn: Synaptic time constant.

    Attributes:
        psc: Post-synaptic current, shape ``(n_out,)``.
    """

    def __init__(self, weight: Weight, tau_syn: ArrayLike):
        self.weight = weight
        self.tau_syn = _arr(tau_syn)
        self.psc = np.zeros(weight.shape[1], dtype=DTYPE)

    def step(self, z: Array, dt: float) -> Array:
        """Decay, inject the spikes ``z`` and return the PSC."""
        self.psc = self.psc * np.exp(-dt / self.tau_syn) + _project(self.weight, z)
        return self.psc


class AlphaPSC:
    """Alpha PSC (brainpy form) with rise variable ``h``.

    .. math::
        h \\leftarrow e^{-dt/\\tau} h + g_{max} W z, \\qquad
        psc \\leftarrow e^{-dt/\\tau}\\, psc + (1 - e^{-dt/\\tau})\\, h

    Args:
        weight: ``(n_in, n_out)`` weight, dense or scipy sparse.
        tau_syn: Synaptic time constant.
        g_max: Peak conductance scale.

    Attributes:
        psc: Post-synaptic current, shape ``(n_out,)``.
        h: Rise variable, shape ``(n_out,)``.
    """

    def __init__(self, weight: Weight, tau_syn: ArrayLike, g_max: float = 1.0):
        self.weight = weight
        self.tau_syn = _arr(tau_syn)
        self.g_max = g_max
        self.psc = np.zeros(weight.shape[1], dtype=DTYPE)
        self.h = np.zeros(weight.shape[1], dtype=DTYPE)

    def step(self, z: Array, dt: float) -> Array:
        """Decay ``h``, inject ``z``, advance ``psc`` from the new ``h``."""
        a = np.exp(-dt / self.tau_syn)
        self.h = a * self.h + self.g_max * _project(self.weight, z)
        self.psc = a * self.psc + (1.0 - a) * self.h
        return self.psc


class AlphaPSCBilleh:
    """Alpha PSC of Billeh et al. 2019 (peak amplitude 1 for unit weight).

    Requires ``dt == 1``. With ``a = exp(-1/tau)`` and ``c = e/tau``:

    .. math::
        h \\leftarrow a h + c\\, W z, \\qquad
        psc \\leftarrow a\\, psc + a\\, h

    Args:
        weight: ``(n_in, n_out)`` weight, dense or scipy sparse.
        tau_syn: Synaptic time constant.

    Attributes:
        psc: Post-synaptic current, shape ``(n_out,)``.
        h: Rise variable, shape ``(n_out,)``.
    """

    def __init__(self, weight: Weight, tau_syn: ArrayLike):
        self.weight = weight
        self.tau_syn = _arr(tau_syn)
        self.psc = np.zeros(weight.shape[1], dtype=DTYPE)
        self.h = np.zeros(weight.shape[1], dtype=DTYPE)

    def step(self, z: Array, dt: float) -> Array:
        """Decay ``h``, inject ``z``, advance ``psc`` from the new ``h``."""
        if dt != 1.0:
            raise ValueError(f"AlphaPSCBilleh requires dt == 1.0, got {dt}")
        a = np.exp(-1.0 / self.tau_syn)
        self.h = a * self.h
        self.h = self.h + np.e / self.tau_syn * _project(self.weight, z)
        self.psc = a * self.psc + a * self.h
        return self.psc


class DualExponentialPSC:
    """Dual-exponential PSC with rise and decay time constants.

    Both ``g_rise`` and ``g_decay`` decay, receive the input, and the PSC is
    read out one decay step after the injection (so the spike acts in its
    delivery step):

    .. math::
        psc = A' \\left(a_d\\, g_{decay} - a_r\\, g_{rise}\\right),
        \\quad A' = \\frac{\\tau_d - \\tau_r}{\\tau_r \\tau_d} A

    where ``a_d = exp(-dt/tau_decay)``, ``a_r = exp(-dt/tau_rise)`` and ``A``
    defaults to the normalisation giving unit peak for unit weight.

    Args:
        weight: ``(n_in, n_out)`` weight, dense or scipy sparse.
        tau_decay: Decay time constant (must exceed ``tau_rise``).
        tau_rise: Rise time constant.
        A: Amplitude scale; ``None`` selects the unit-peak normalisation.

    Attributes:
        psc: Post-synaptic current, shape ``(n_out,)``.
        g_rise: Rising component, shape ``(n_out,)``.
        g_decay: Decaying component, shape ``(n_out,)``.
    """

    def __init__(
        self,
        weight: Weight,
        tau_decay: float,
        tau_rise: float,
        A: float | None = None,
    ):
        self.weight = weight
        self.tau_decay = float(tau_decay)
        self.tau_rise = float(tau_rise)
        if A is None:
            A = (
                tau_decay
                / (tau_decay - tau_rise)
                * (tau_rise / tau_decay) ** (tau_rise / (tau_rise - tau_decay))
            )
        self.a = (tau_decay - tau_rise) / tau_rise / tau_decay * A
        self.psc = np.zeros(weight.shape[1], dtype=DTYPE)
        self.g_rise = np.zeros(weight.shape[1], dtype=DTYPE)
        self.g_decay = np.zeros(weight.shape[1], dtype=DTYPE)

    def step(self, z: Array, dt: float) -> Array:
        """Decay both components, inject ``z`` and read out the PSC."""
        a_r = np.exp(-dt / self.tau_rise)
        a_d = np.exp(-dt / self.tau_decay)
        self.g_rise = a_r * self.g_rise
        self.g_decay = a_d * self.g_decay
        wz = _project(self.weight, z)
        self.g_rise = self.g_rise + wz
        self.g_decay = self.g_decay + wz
        self.psc = self.a * (a_d * self.g_decay - a_r * self.g_rise)
        return self.psc


PSC = ExponentialPSC | AlphaPSC | AlphaPSCBilleh | DualExponentialPSC


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #


@dataclass
class Network:
    """Recurrent network: one neuron population and one or more PSC channels.

    Each step computes ``z = neuron(sum(psc) + x)`` from the PSCs of the
    *previous* step and then feeds ``z`` to every synapse (one-step recurrent
    latency, like ``btorch.models.rnn.RecurrentNN``). Multiple synapses allow
    e.g. separate excitatory and inhibitory channels with different time
    constants.

    Attributes:
        neuron: Neuron population.
        synapses: PSC channels whose currents are summed into the neuron.
    """

    neuron: LIF | GLIF3
    synapses: Sequence[PSC]

    def step(self, x: ArrayLike, dt: float) -> Array:
        """Advance one step with external current ``x``; return spikes."""
        i_syn = np.zeros(self.neuron.n_neuron, dtype=DTYPE)
        for syn in self.synapses:
            i_syn = i_syn + syn.psc
        z = self.neuron.step(i_syn + _arr(x), dt)
        for syn in self.synapses:
            syn.step(z, dt)
        return z
