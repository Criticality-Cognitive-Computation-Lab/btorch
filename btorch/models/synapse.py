from typing import Protocol

import torch

from . import environ
from .base import MemoryModule
from .ode import exp_euler_step_auto
from .types import TensorLike


class Synapse(Protocol):
    """Minimum Synapse interface."""

    # TODO: rework spikingjelly's synapse abstraction

    psc: torch.Tensor

    def __call__(self, x): ...


class BasePSC(MemoryModule):
    psc: torch.Tensor

    def __init__(
        self,
        n_neuron: int,
        linear: torch.nn.Module,
        latency: float = 0.0,
        step_mode="s",
        backend="torch",
    ):
        super().__init__()

        self.n_neuron = n_neuron
        self.step_mode = step_mode
        self.backend = backend
        self.latency = latency
        self.linear = linear

        self.register_memory("psc", 0.0, n_neuron)

        # TODO: delay should have its own class
        if latency > 0:
            self.latency_steps = round(latency / environ.get("dt"))
            self.register_memory(
                "delay_buffer",
                0,
                (self.latency_steps + 1, n_neuron),
            )
            self.register_memory("delay_ptr", 0, 1)

    @torch.no_grad()
    def init_state(
        self,
        batch_size=None,
        dtype=None,
        device=None,
        persistent=True,
        skip_mem_name: tuple[str, ...] = (),
    ):
        super().init_state(
            batch_size,
            dtype,
            device,
            persistent,
            skip_mem_name=("delay_buffer",) + skip_mem_name,
        )
        if self.latency > 0:
            delay_buffer_sizes = self._memories_rv["delay_buffer"].sizes
            if batch_size is not None:
                delay_buffer_sizes = (
                    delay_buffer_sizes[0],
                    batch_size,
                    delay_buffer_sizes[1],
                )
            self.register_buffer(
                "delay_buffer",
                torch.zeros(delay_buffer_sizes, dtype=dtype, device=device),
                persistent=persistent,
            )
            self.register_buffer(
                "delay_ptr",
                torch.tensor([0], dtype=torch.long, device=device),
                persistent=persistent,
            )

    @torch.no_grad()
    def reset(
        self,
        batch_size=None,
        dtype=None,
        device=None,
        skip_mem_name: tuple[str, ...] = (),
    ):
        super().reset(
            batch_size,
            dtype,
            device,
            skip_mem_name=("delay_ptr", "delay_buffer") + skip_mem_name,
        )
        if self.latency > 0:
            if batch_size is None and self.delay_buffer.ndim > 2:
                # TODO: what if batch_size or n_neuron are not one dimensional?
                batch_size = self.delay_buffer.shape[1]
            delay_buffer_sizes = self._memories_rv["delay_buffer"].sizes
            if batch_size is not None:
                delay_buffer_sizes = (
                    delay_buffer_sizes[0],
                    batch_size,
                    delay_buffer_sizes[1],
                )
            self.delay_buffer = torch.zeros(
                delay_buffer_sizes, dtype=dtype, device=device
            )
            self.delay_ptr = torch.tensor(0, dtype=torch.long, device=device)

    def extra_repr(self):
        return f" step_mode={self.step_mode}, backend={self.backend}"

    def conductance_charge(self):
        raise NotImplementedError()

    def adaptation_charge(self, z: torch.Tensor):
        raise NotImplementedError()

    def current_charge(self, v=None):
        if v is not None:
            raise NotImplementedError(
                "Only current-based PSC is supported."
                "Conductance-based PSC requires voltage from post-syn neurons, "
                "which the current abstraction doesn't support."
            )
        else:
            return self.psc

    def single_step_forward(self, z: torch.Tensor):
        if self.latency > 0.0:
            self.delay_ptr += 1
            idx = self.delay_ptr % self.delay_buffer.shape[0]
            self.delay_buffer[idx] = z

            spike = self.delay_buffer[
                (self.delay_ptr - self.latency_steps) % self.delay_buffer.shape[0]
            ]
            print(spike)

        else:
            spike = z

        self.conductance_charge()
        self.adaptation_charge(spike)
        current = self.current_charge()
        return current

    def multi_step_forward(self, z_seq: torch.Tensor):
        T = z_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(z_seq[t])
            y_seq.append(y)

        return torch.stack(y_seq)


class ExponentialPSC(BasePSC):
    tau_syn: torch.Tensor | torch.nn.Parameter

    def __init__(
        self,
        n_neuron,
        tau_syn: float | TensorLike,
        linear,
        step_mode="s",
        backend="torch",
    ):
        super().__init__(n_neuron, linear, step_mode=step_mode, backend=backend)

        self.register_buffer("tau_syn", torch.as_tensor(tau_syn), persistent=False)

    def dpsc(self, psc):
        return -psc / self.tau_syn

    def conductance_charge(self):
        return exp_euler_step_auto(self.dpsc, self.psc, dt=environ.get("dt"))

    def adaptation_charge(self, z: torch.Tensor):
        wz = self.linear(z)
        self.psc = self.psc + wz


class _Adaptive2VarPSC(BasePSC):
    h: torch.Tensor

    def __init__(self, n_neuron, linear, step_mode="s", backend="torch"):
        super().__init__(n_neuron, linear, step_mode=step_mode, backend=backend)

        self.register_memory("h", 0.0, n_neuron)


class AlphaPSCBilleh(_Adaptive2VarPSC):
    tau_syn: torch.Tensor | torch.nn.Parameter
    syn_decay: torch.Tensor

    def __init__(
        self,
        n_neuron: int,
        tau_syn: float | TensorLike,
        linear: torch.nn.Module,
        step_mode="s",
        backend="torch",
    ):
        """The Current-Based Alpha form of PSC, from [1], ensuring a post-
        synaptic current with synapse weight W = 1.0 has an amplitude of 1.0 pA
        at the peak time point of t = tau_syn.

        NOTE: this model assumes environ.get("dt") == 1.0

        [1] Billeh, Y. N. et al. Systematic integration of structural and
        functional data into multi-scale models of mouse primary visual
        cortex. 662189 Preprint at https://doi.org/10.1101/662189 (2019).

        :param tau_syn: the synaptic time constant
        :type tau_syn: float or torch.Tensor
        """

        super().__init__(n_neuron, linear, step_mode, backend)

        try:
            dt = environ.get("dt")
            assert dt == 1.0, "dt must be 1.0 for this model"
        except KeyError:
            pass

        self.register_buffer("tau_syn", torch.as_tensor(tau_syn), persistent=False)

        self.register_buffer(
            "syn_decay", torch.exp(-1.0 / self.tau_syn), persistent=False
        )

    def conductance_charge(self):
        self.psc = self.syn_decay * self.psc + self.syn_decay * self.h
        return self.psc

    def adaptation_charge(self, z: torch.Tensor):
        wz = self.linear(z)
        self.h = self.syn_decay * self.h + torch.e / self.tau_syn * wz


class AlphaPSC(_Adaptive2VarPSC):
    tau_syn: torch.Tensor | torch.nn.Parameter
    g_max: torch.Tensor | torch.nn.Parameter

    def __init__(
        self,
        n_neuron: int,
        tau_syn: float | TensorLike,
        linear: torch.nn.Module,
        g_max=1.0,
        step_mode="s",
        backend="torch",
    ):
        """The Alpha form (current-based) of PSC, from Brainpy/BrainState."""

        super().__init__(n_neuron, linear, step_mode, backend)

        self.register_buffer("tau_syn", torch.as_tensor(tau_syn), persistent=False)
        self.register_buffer("g_max", torch.as_tensor(g_max), persistent=False)

    def dg(self, psc, h):
        return -psc / self.tau_syn + h / self.tau_syn

    def dh(self, h):
        return -h / self.tau_syn

    def conductance_charge(self):
        self.psc = exp_euler_step_auto(self.dg, self.psc, self.h, dt=environ.get("dt"))

    def adaptation_charge(self, z: torch.Tensor):
        wz = self.g_max * self.linear(z)
        self.h = exp_euler_step_auto(self.dh, self.h, dt=environ.get("dt")) + wz


class DualExponentialPSC(BasePSC):
    tau_rise: torch.Tensor | torch.nn.Parameter
    tau_decay: torch.Tensor | torch.nn.Parameter
    a: torch.Tensor
    g_rise: torch.Tensor
    g_decay: torch.Tensor

    def __init__(
        self,
        n_neuron: int,
        tau_decay: float | TensorLike,
        tau_rise: float | TensorLike,
        linear: torch.nn.Module,
        latency: float = 0.0,
        A: float | TensorLike | None = None,
        step_mode="s",
        backend="torch",
    ):
        """The Double Exponential form of PSC, from Brainpy/BrainState."""

        super().__init__(
            n_neuron=n_neuron,
            linear=linear,
            latency=latency,
            step_mode=step_mode,
            backend=backend,
        )

        self.register_buffer("tau_decay", torch.as_tensor(tau_decay), persistent=False)
        self.register_buffer("tau_rise", torch.as_tensor(tau_rise), persistent=False)

        if A is None:
            A = (
                self.tau_decay
                / (self.tau_decay - self.tau_rise)
                * (self.tau_rise / self.tau_decay)
                ** (self.tau_rise / (self.tau_rise - self.tau_decay))
            )
        A = torch.as_tensor(A)
        a = (self.tau_decay - self.tau_rise) / self.tau_rise / self.tau_decay * A
        self.register_buffer("a", a, persistent=False)

        self.register_memory("g_rise", 0.0, n_neuron)
        self.register_memory("g_decay", 0.0, n_neuron)
        self.register_memory("psc", 0.0, n_neuron)

    def dg_rise(self, g_rise):
        return -g_rise / self.tau_rise

    def dg_decay(self, g_decay):
        return -g_decay / self.tau_decay

    def conductance_charge(self):
        self.g_rise = exp_euler_step_auto(
            self.dg_rise, self.g_rise, dt=environ.get("dt")
        )
        self.g_decay = exp_euler_step_auto(
            self.dg_decay, self.g_decay, dt=environ.get("dt")
        )

    def adaptation_charge(self, z: torch.Tensor):
        wz = self.linear(z)
        self.g_rise = self.g_rise + wz
        self.g_decay = self.g_decay + wz
        self.psc = self.a * (self.g_decay - self.g_rise)


class HeterSynapsePSC(BasePSC):
    def __init__(
        self,
        n_neuron: int,
        n_receptor: int,
        linear: torch.nn.Module,
        base_psc: type[BasePSC] = AlphaPSC,
        step_mode="s",
        backend="torch",
        **kwargs,
    ):
        super().__init__(
            n_neuron, linear, latency=0, step_mode=step_mode, backend=backend
        )

        self.base_psc = base_psc(
            n_neuron=n_neuron * n_receptor,
            linear=linear,
            step_mode=step_mode,
            backend=backend,
            **kwargs,
        )
        self.n_receptor = n_receptor
        self.register_memory("psc", 0.0, n_neuron)

    def single_step_forward(self, z: torch.Tensor):
        psc = self.base_psc.single_step_forward(z)
        self.psc = psc.view(*psc.shape[:-1], self.n_neuron, self.n_receptor).sum(-1)
        return self.psc
