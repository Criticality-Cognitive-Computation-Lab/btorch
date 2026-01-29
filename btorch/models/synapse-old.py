from typing import Iterable, Optional

import numpy as np
import torch

from . import environ
from .base import MemoryModule
from .ode import exp_euler_step_auto


class BasePSC(MemoryModule):
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
        self.latency_steps = round(latency / environ.get("dt"))
        self.latency = latency
        self.linear = linear

        if latency > 0:
            self.register_memory(
                "delay_buffer",
                0,
                (self.latency_steps + 1, n_neuron),
            )

    def init_state(
        self,
        batch_size=None,
        dtype=None,
        device=None,
        persistent=True,
        skip_mem_name: Iterable[str] = (),
    ):
        super().init_state(
            batch_size,
            dtype,
            device,
            persistent,
            skip_mem_name=("delay_buffer",) + skip_mem_name,
        )
        if self.latency > 0:
            delay_buffer_sizes = self._memories_rv["delay_buffer"][1]["sizes"]
            if batch_size is not None:
                delay_buffer_sizes = (
                    delay_buffer_sizes[0],
                    batch_size,
                    delay_buffer_sizes[1],
                )
            self.register_buffer(
                "delay_buffer",
                torch.zeros(delay_buffer_sizes, dtype=dtype, device=device),
            )

    def reset(
        self,
        batch_size=None,
        dtype=None,
        device=None,
        skip_mem_name: Iterable[str] = (),
    ):
        super().reset(
            batch_size,
            dtype,
            device,
            skip_mem_name=("delay_buffer", ) + skip_mem_name,
        )
        if self.latency > 0:
            if batch_size is None and self.delay_buffer.ndim > 2:
                # TODO: what if batch_size or n_neuron are not one dimensional?
                batch_size = self.delay_buffer.shape[1]
            delay_buffer_sizes = self._memories_rv["delay_buffer"][1]["sizes"]
            if batch_size is not None:
                delay_buffer_sizes = (
                    delay_buffer_sizes[0],
                    batch_size,
                    delay_buffer_sizes[1],
                )
            self.delay_buffer = torch.zeros(
                delay_buffer_sizes, dtype=dtype, device=device
            )

    def extra_repr(self):
        return f" step_mode={self.step_mode}, backend={self.backend}"

    def conductance_charge(self):
        raise NotImplementedError()

    def adaptation_charge(self):
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
            self.delay_buffer = torch.cat(
                (z.unsqueeze(dim=0), self.delay_buffer[:-1].clone()), dim=0)
            spike = self.delay_buffer[-1]
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
    def __init__(
        self,
        n_neuron,
        tau_syn: float | torch.Tensor,
        linear,
        step_mode="s",
        backend="torch",
    ):
        super().__init__(n_neuron, linear, step_mode, backend)

        self.register_memory("psc", 0.0, n_neuron)
        self.register_buffer(
            "tau_syn", torch.as_tensor(tau_syn), persistent=False)

    def dpsc(self, psc):
        return -psc / self.tau_syn

    def conductance_charge(self):
        return exp_euler_step_auto(self.dpsc, self.psc, dt=environ.get("dt"))

    def adaptation_charge(self, z: torch.Tensor):
        wz = self.linear(z)
        self.psc = self.psc + wz


class _Adaptive2VarPSC(BasePSC):
    def __init__(self, n_neuron, linear, latency: float = 0.0, step_mode="s", backend="torch"):
        super().__init__(n_neuron, linear, latency, step_mode, backend)

        self.register_memory("psc", 0.0, n_neuron)
        self.register_memory("h", 0.0, n_neuron)


class AlphaPSCBilleh(_Adaptive2VarPSC):
    def __init__(
        self,
        n_neuron: int,
        tau_syn: float | torch.Tensor,
        linear: torch.nn.Module,
        latency: float = 0.0,
        step_mode="s",
        backend="torch",
    ):
        """The Current-Based Alpha form of PSC, from [1], ensuring a post-
        synaptic current with synapse weight W = 1.0 has an amplitude of 1.0 pA
        at the peak time point of t = tau_syn.

        NOTE: it does **NOT** respect environ.get("dt")

        [1] Billeh, Y. N. et al. Systematic integration of structural and
        functional data into multi-scale models of mouse primary visual
        cortex. 662189 Preprint at https://doi.org/10.1101/662189 (2019).

        :param tau_syn: the synaptic time constant
        :type tau_syn: float or torch.Tensor
        """

        super().__init__(n_neuron, linear, latency, step_mode, backend)

        self.register_buffer(
            "tau_syn", torch.as_tensor(tau_syn), persistent=False)

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
    def __init__(
        self,
        n_neuron: int,
        tau_syn,
        linear: torch.nn.Module,
        g_max=1.0,
        latency: float = 0.0,
        step_mode="s",
        backend="torch",
    ):
        """The Alpha form (current-based) of PSC, from Brainpy/BrainState."""

        super().__init__(n_neuron, linear, latency, step_mode, backend)

        self.register_buffer(
            "tau_syn", torch.as_tensor(tau_syn), persistent=False)
        self.register_buffer("g_max", torch.as_tensor(g_max), persistent=False)

    def dg(self, psc, h):
        return -psc / self.tau_syn + h / self.tau_syn

    def dh(self, h):
        return -h / self.tau_syn

    def conductance_charge(self):
        self.psc = exp_euler_step_auto(
            self.dg, self.psc, self.h, dt=environ.get("dt"))

    def adaptation_charge(self, z: torch.Tensor):
        wz = self.g_max * self.linear(z)
        self.h = exp_euler_step_auto(
            self.dh, self.h, dt=environ.get("dt")) + wz


class DualExponentialPSC(BasePSC):
    def __init__(
        self,
        n_neuron: int,
        tau_decay: float | torch.Tensor,
        tau_rise: float | torch.Tensor,
        linear: torch.nn.Module,
        latency: float = 0.0,
        A: Optional[float | torch.Tensor] = None,
        step_mode="s",
        backend="torch",
    ):
        """The Double Exponential form of PSC, from Brainpy/BrainState."""

        super().__init__(n_neuron, linear, latency=latency, step_mode=step_mode, backend=backend)

        self.register_buffer("tau_decay", torch.as_tensor(
            tau_decay), persistent=False)
        self.register_buffer("tau_rise", torch.as_tensor(
            tau_rise), persistent=False)

        if A is None:
            A = (
                self.tau_decay
                / (self.tau_decay - self.tau_rise)
                * np.float_power(
                    self.tau_rise / self.tau_decay,
                    self.tau_rise / (self.tau_rise - self.tau_decay),
                )
            )
        # a = (self.tau_decay - self.tau_rise) / \
        #     self.tau_rise / self.tau_decay * A
        a=A
        self.register_buffer("a", torch.as_tensor(a), persistent=False)

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
        base_psc: type = AlphaPSC,
        step_mode="s",
        backend="torch",
        receptor_is_exc: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(n_neuron, linear, step_mode, backend)

        self.base_psc = base_psc(
            n_neuron=n_neuron * n_receptor,
            linear=linear,
            step_mode=step_mode,
            backend=backend,
            **kwargs,
        )
        self.n_receptor = n_receptor
        # receptor_is_exc: shape [R], bool tensor, True表示该receptor_index来自兴奋性突触（E->*）
        if receptor_is_exc is None:
            # 若未提供，则默认前一半为E，其余为I（仅作为兜底，不建议依赖）
            receptor_is_exc = torch.zeros(n_receptor, dtype=torch.bool)
            receptor_is_exc[: (n_receptor // 2)] = True
        self.register_buffer("receptor_is_exc", receptor_is_exc.bool(), persistent=False)

        # 注册 memory：总电流、E电流、I电流（按post neuron）
        self.register_memory("psc", 0.0, n_neuron)
        self.register_memory("psc_e", 0.0, n_neuron)
        self.register_memory("psc_i", 0.0, n_neuron)

        #breakpoint()

    def single_step_forward(self, z: torch.Tensor):
        psc = self.base_psc.single_step_forward(z)
        # Support both no-batch [N*R] and batched [B, N*R] shapes
        #breakpoint()
        if psc.dim() == 2:
            B = psc.shape[0]
            psc_br = psc.view(B, self.n_neuron, self.n_receptor)
            self.psc = psc_br.sum(-1)
            #breakpoint()
            # E/I 拆分
            if self.receptor_is_exc.any():
                #breakpoint()
                self.psc_e = psc_br[..., self.receptor_is_exc].sum(-1)
                self.psc_i = psc_br[..., ~self.receptor_is_exc].sum(-1)
            else:
                # 若全为False，则E分量为0，总计入I
                self.psc_e = torch.zeros_like(self.psc)
                self.psc_i = self.psc.clone()
        elif psc.dim() == 1:
            psc_nr = psc.view(self.n_neuron, self.n_receptor)
            self.psc = psc_nr.sum(-1)
            #breakpoint()
            if self.receptor_is_exc.any():
                #breakpoint()
                self.psc_e = psc_nr[:, self.receptor_is_exc].sum(-1)
                self.psc_i = psc_nr[:, ~self.receptor_is_exc].sum(-1)
            else:
                self.psc_e = torch.zeros_like(self.psc)
                self.psc_i = self.psc.clone()
        else:
            raise RuntimeError(
                f"Unexpected PSC shape {psc.shape}; expected [N*R] or [B, N*R] with N={self.n_neuron}, R={self.n_receptor}"
            )
        return self.psc, self.psc_e, self.psc_i

class HeterSynapseDualPSC(BasePSC):
    def __init__(
        self,
        n_neuron: int,
        n_receptor: int,
        linear: torch.nn.Module,
        tau_decay: float | list | torch.Tensor,
        tau_rise: float | list | torch.Tensor,
        receptor_is_exc: torch.Tensor | None = None,
        latency: float = 0.0,
        step_mode="s",
        backend="torch",
        **kwargs,
    ):
        """
        Heterogeneous Dual Exponential PSC.
        Implements the Dual Exponential kernel logic (via composition) but supports
        multi-receptor structures (e.g. AMPA, NMDA, GABA) with distinct time constants.

        The total number of underlying synaptic units will be n_neuron * n_receptor.
        """
        super().__init__(n_neuron, linear, latency=latency, step_mode=step_mode, backend=backend)

        self.n_receptor = n_receptor
        
        # ----------------------------------------------------------------------
        # 1. Parameter Expansion Logic
        #    DualExponentialPSC expects parameters of shape [Total_Units] (or scalar).
        #    Our memory layout in HeterSynapsePSC (via view) is [Batch, Neuron, Receptor].
        #    So the flattened sequence corresponds to [n0_r0, n0_r1, ..., n1_r0, ...].
        #    Therefore, receptor-specific parameters must be repeated n_neuron times.
        # ----------------------------------------------------------------------
        def _expand_param(param, name):
            t = torch.as_tensor(param)
            if t.ndim == 0:
                # Scalar: applies to all receptors
                return t
            elif t.ndim == 1:
                if t.shape[0] == n_receptor:
                    # Pattern: [r1, r2] -> [r1, r2, r1, r2, ...] to match (N, R) layout
                    return t.repeat(n_neuron)
                elif t.shape[0] == n_neuron * n_receptor:
                    # Already fully specified
                    return t
                else:
                    raise ValueError(
                        f"{name} shape {t.shape} does not match n_receptor ({n_receptor}) "
                        f"or total units ({n_neuron * n_receptor})."
                    )
            else:
                raise ValueError(f"{name} must be 0-D or 1-D.")

        expanded_tau_decay = _expand_param(tau_decay, "tau_decay")
        expanded_tau_rise = _expand_param(tau_rise, "tau_rise")

        # ----------------------------------------------------------------------
        # 2. Composition: Instantiate the inner solver
        # ----------------------------------------------------------------------
        self.base_psc = DualExponentialPSC(
            n_neuron=n_neuron * n_receptor,
            tau_decay=expanded_tau_decay,
            tau_rise=expanded_tau_rise,
            linear=linear,
            latency=latency,  # Latency is shared/homogeneous in this implementation
            step_mode=step_mode,
            backend=backend,
            **kwargs,
        )

        # ----------------------------------------------------------------------
        # 3. Receptor Type Logic (E/I Separation) - Copied from HeterSynapsePSC
        # ----------------------------------------------------------------------
        if receptor_is_exc is None:
            # Default fallback: first half E, second half I
            receptor_is_exc = torch.zeros(n_receptor, dtype=torch.bool)
            receptor_is_exc[: (n_receptor // 2)] = True
        
        self.register_buffer("receptor_is_exc", receptor_is_exc.bool(), persistent=False)

        # Register output memories for the aggregated neuron level
        self.register_memory("psc", 0.0, n_neuron)
        self.register_memory("psc_e", 0.0, n_neuron)
        self.register_memory("psc_i", 0.0, n_neuron)

    def single_step_forward(self, z: torch.Tensor):
        # 1. Compute all synaptic currents (flattened: N * R)
        psc = self.base_psc.single_step_forward(z)
        
        # 2. Reshape and Aggregate (Logic matches HeterSynapsePSC)
        if psc.dim() == 2:
            B = psc.shape[0]
            # View as [Batch, Neuron, Receptor]
            psc_br = psc.view(B, self.n_neuron, self.n_receptor)
            
            # Sum all receptors for total PSC
            self.psc = psc_br.sum(-1)

            # E/I Separation
            if self.receptor_is_exc.any():
                self.psc_e = psc_br[..., self.receptor_is_exc].sum(-1)
                self.psc_i = psc_br[..., ~self.receptor_is_exc].sum(-1)
            else:
                self.psc_e = torch.zeros_like(self.psc)
                self.psc_i = self.psc.clone()

        elif psc.dim() == 1:
            # No batch dimension [N * R]
            psc_nr = psc.view(self.n_neuron, self.n_receptor)
            
            self.psc = psc_nr.sum(-1)

            if self.receptor_is_exc.any():
                self.psc_e = psc_nr[:, self.receptor_is_exc].sum(-1)
                self.psc_i = psc_nr[:, ~self.receptor_is_exc].sum(-1)
            else:
                self.psc_e = torch.zeros_like(self.psc)
                self.psc_i = self.psc.clone()
        else:
            raise RuntimeError(
                f"Unexpected PSC shape {psc.shape}; expected [N*R] or [B, N*R] "
                f"with N={self.n_neuron}, R={self.n_receptor}"
            )

        return self.psc, self.psc_e, self.psc_i