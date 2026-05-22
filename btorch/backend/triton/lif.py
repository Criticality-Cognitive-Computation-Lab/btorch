from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import triton

from btorch.models import base

from .lif_kernels import (
    lif_single_step_fwd_kernel,
    lif_multistep_fwd_kernel,
    lif_multistep_soft_noref_bwd_kernel,
)
from .sparse import coo_spmm


def _normalize_n_neuron(
    n_neuron: int | Sequence[int],
) -> tuple[int, ...]:
    if isinstance(n_neuron, int):
        return (n_neuron,)
    n_neuron = tuple(n_neuron)
    if len(n_neuron) == 0:
        raise ValueError("n_neuron must contain at least one dimension.")
    return n_neuron


def _expand_param(
    value: float | torch.Tensor | Sequence[float],
    target_shape: tuple[int, ...],
    *,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        return torch.full(
            target_shape,
            tensor.item(),
            device=tensor.device,
            dtype=tensor.dtype,
        )
    return torch.broadcast_to(tensor, target_shape).contiguous()


def triton_lif_multistep(
    x_seq: torch.Tensor,
    v: torch.Tensor,
    v_threshold: torch.Tensor,
    v_reset: torch.Tensor,
    c_m: torch.Tensor,
    tau: torch.Tensor,
    *,
    dt: float,
    hard_reset: bool = False,
    refractory: torch.Tensor | None = None,
    tau_ref: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not x_seq.is_cuda:
        raise ValueError("triton_lif_multistep requires CUDA tensors.")
    if x_seq.ndim != 2:
        raise ValueError("x_seq must be flattened to shape (T, numel).")
    if x_seq.shape[1] != v.numel():
        raise ValueError("x_seq.shape[1] must match the number of state elements.")

    if x_seq.shape[0] == 0:
        spikes = torch.empty_like(x_seq)
        next_refractory = refractory.clone() if refractory is not None else None
        return spikes, v.clone(), next_refractory

    if (refractory is None) != (tau_ref is None):
        raise ValueError("refractory and tau_ref must either both be set or both be None.")

    x_seq = x_seq.contiguous()
    v = v.contiguous()
    v_threshold = v_threshold.contiguous()
    v_reset = v_reset.contiguous()
    c_m = c_m.contiguous()
    tau = tau.contiguous()

    use_refractory = refractory is not None
    if use_refractory:
        refractory = refractory.contiguous()
        tau_ref = tau_ref.contiguous()
        refractory_out = torch.empty_like(refractory)
    else:
        refractory = v
        tau_ref = v
        refractory_out = v

    spikes = torch.empty_like(x_seq)
    v_out = torch.empty_like(v)

    numel = v.numel()
    steps = x_seq.shape[0]
    block_size = 256
    grid = (triton.cdiv(numel, block_size),)

    if x_seq.requires_grad and (hard_reset or use_refractory):
        raise NotImplementedError(
            "Triton LIF backward currently supports only soft reset without "
            "refractory for x_seq gradients."
        )

    supports_x_grad = x_seq.requires_grad and not hard_reset and not use_refractory

    if supports_x_grad:
        spikes, v_out = _TritonLIFSoftResetNoRefractory.apply(
            x_seq,
            v,
            v_threshold,
            v_reset,
            c_m,
            tau,
            float(dt),
        )
        return spikes, v_out, None

    lif_multistep_fwd_kernel[grid](
        x_seq,
        spikes,
        v,
        refractory,
        v_threshold,
        v_reset,
        c_m,
        tau,
        tau_ref,
        v_out,
        refractory_out,
        spikes,
        x_seq.stride(0),
        x_seq.stride(1),
        spikes.stride(0),
        spikes.stride(1),
        spikes.stride(0),
        spikes.stride(1),
        numel,
        dt,
        BLOCK_SIZE=block_size,
        T=steps,
        HARD_RESET=hard_reset,
        USE_REFRACTORY=use_refractory,
        SAVE_U_PRE_SPIKE=False,
        num_warps=4,
    )

    return spikes, v_out, refractory_out if use_refractory else None


def triton_lif_single_step(
    x: torch.Tensor,
    v: torch.Tensor,
    v_threshold: torch.Tensor,
    v_reset: torch.Tensor,
    c_m: torch.Tensor,
    tau: torch.Tensor,
    *,
    dt: float,
    hard_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one forward-only LIF update on flattened CUDA tensors."""
    if not x.is_cuda:
        raise ValueError("triton_lif_single_step requires CUDA tensors.")
    if x.ndim != 1:
        raise ValueError("x must be flattened to shape (numel,).")
    if x.shape != v.shape:
        raise ValueError("x and v must have identical flattened shapes.")

    x = x.contiguous()
    v = v.contiguous()
    v_threshold = v_threshold.contiguous()
    v_reset = v_reset.contiguous()
    c_m = c_m.contiguous()
    tau = tau.contiguous()

    spikes = torch.empty_like(x)
    v_out = torch.empty_like(v)

    numel = x.numel()
    block_size = 256
    grid = (triton.cdiv(numel, block_size),)

    lif_single_step_fwd_kernel[grid](
        x,
        spikes,
        v,
        v_threshold,
        v_reset,
        c_m,
        tau,
        v_out,
        numel,
        dt,
        BLOCK_SIZE=block_size,
        HARD_RESET=hard_reset,
        num_warps=4,
    )
    return spikes, v_out


class _TritonLIFSoftResetNoRefractory(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_seq: torch.Tensor,
        v: torch.Tensor,
        v_threshold: torch.Tensor,
        v_reset: torch.Tensor,
        c_m: torch.Tensor,
        tau: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spikes = torch.empty_like(x_seq)
        v_out = torch.empty_like(v)
        u_pre_spike = torch.empty_like(x_seq)

        numel = v.numel()
        steps = x_seq.shape[0]
        block_size = 256
        grid = (triton.cdiv(numel, block_size),)

        lif_multistep_fwd_kernel[grid](
            x_seq,
            spikes,
            v,
            v,
            v_threshold,
            v_reset,
            c_m,
            tau,
            v,
            v_out,
            v,
            u_pre_spike,
            x_seq.stride(0),
            x_seq.stride(1),
            spikes.stride(0),
            spikes.stride(1),
            u_pre_spike.stride(0),
            u_pre_spike.stride(1),
            numel,
            dt,
            BLOCK_SIZE=block_size,
            T=steps,
            HARD_RESET=False,
            USE_REFRACTORY=False,
            SAVE_U_PRE_SPIKE=True,
            num_warps=4,
        )

        ctx.dt = dt
        ctx.save_for_backward(
            u_pre_spike,
            v_threshold,
            v_reset,
            c_m,
            tau,
        )
        return spikes, v_out

    @staticmethod
    def backward(
        ctx,
        grad_spikes: torch.Tensor | None,
        grad_v_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        u_pre_spike, v_threshold, v_reset, c_m, tau = ctx.saved_tensors

        if grad_spikes is None:
            grad_spikes = torch.zeros_like(u_pre_spike)
        else:
            grad_spikes = grad_spikes.contiguous()

        if grad_v_out is None:
            grad_v_out = torch.zeros_like(v_threshold)
        else:
            grad_v_out = grad_v_out.contiguous()

        grad_x = torch.empty_like(u_pre_spike)
        numel = v_threshold.numel()
        steps = u_pre_spike.shape[0]
        block_size = 256
        grid = (triton.cdiv(numel, block_size),)

        lif_multistep_soft_noref_bwd_kernel[grid](
            grad_spikes,
            grad_v_out,
            u_pre_spike,
            v_threshold,
            v_reset,
            c_m,
            tau,
            grad_x,
            grad_spikes.stride(0),
            grad_spikes.stride(1),
            grad_x.stride(0),
            grad_x.stride(1),
            u_pre_spike.stride(0),
            u_pre_spike.stride(1),
            numel,
            ctx.dt,
            BLOCK_SIZE=block_size,
            T=steps,
            num_warps=4,
        )

        return grad_x, None, None, None, None, None, None


class TritonMultiStepLIF(torch.nn.Module):
    """Standalone Triton LIF forward path for multi-step benchmarking.

    This module intentionally does not integrate into the project's existing
    neuron hierarchy. It exists as an isolated forward-only baseline for
    comparing a fused Triton implementation against the reference
    `btorch.models.neurons.lif.LIF` path.
    """

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        *,
        v_threshold: float | torch.Tensor = 1.0,
        v_reset: float | torch.Tensor = 0.0,
        c_m: float | torch.Tensor = 1.0,
        tau: float | torch.Tensor = 20.0,
        tau_ref: float | torch.Tensor | None = None,
        hard_reset: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.n_neuron = _normalize_n_neuron(n_neuron)
        self.hard_reset = hard_reset
        self._use_refractory = tau_ref is not None

        self.register_buffer(
            "v_threshold_base",
            _expand_param(v_threshold, self.n_neuron, device=device, dtype=dtype),
        )
        self.register_buffer(
            "v_reset_base",
            _expand_param(v_reset, self.n_neuron, device=device, dtype=dtype),
        )
        self.register_buffer(
            "c_m_base",
            _expand_param(c_m, self.n_neuron, device=device, dtype=dtype),
        )
        self.register_buffer(
            "tau_base",
            _expand_param(tau, self.n_neuron, device=device, dtype=dtype),
        )

        if self._use_refractory:
            self.register_buffer(
                "tau_ref_base",
                _expand_param(tau_ref, self.n_neuron, device=device, dtype=dtype),
            )
        else:
            self.tau_ref_base = None

        self.register_buffer("v", self.v_reset_base.clone())
        if self._use_refractory:
            self.register_buffer("refractory", torch.zeros_like(self.v_reset_base))
        else:
            self.refractory = None

        self._state_shape = self.v.shape
        self._refresh_flat_views()

    def _expand_base(self, tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.broadcast_to(tensor, shape).contiguous()

    def _refresh_flat_views(self) -> None:
        state_shape = tuple(self.v.shape)
        self._v_flat = self.v.reshape(-1).contiguous()
        self._v_threshold_flat = self._expand_base(
            self.v_threshold_base, state_shape
        ).reshape(-1)
        self._v_reset_flat = self._expand_base(self.v_reset_base, state_shape).reshape(-1)
        self._c_m_flat = self._expand_base(self.c_m_base, state_shape).reshape(-1)
        self._tau_flat = self._expand_base(self.tau_base, state_shape).reshape(-1)
        if self._use_refractory:
            self._refractory_flat = self.refractory.reshape(-1).contiguous()
            self._tau_ref_flat = self._expand_base(
                self.tau_ref_base, state_shape
            ).reshape(-1)

    @torch.no_grad()
    def reset_state(self, batch_size: int | Sequence[int] | None = None) -> None:
        if batch_size is None:
            batch_shape: tuple[int, ...] = ()
        elif isinstance(batch_size, int):
            batch_shape = (batch_size,)
        else:
            batch_shape = tuple(batch_size)

        state_shape = batch_shape + self.n_neuron
        self.v = self._expand_base(self.v_reset_base, state_shape)
        self._state_shape = state_shape
        if self._use_refractory:
            self.refractory = torch.zeros_like(self.v)
        self._refresh_flat_views()

    def forward(self, x_seq: torch.Tensor, *, dt: float) -> torch.Tensor:
        if x_seq.ndim < 2:
            raise ValueError("x_seq must have shape (T, *batch, n_neuron).")
        if tuple(x_seq.shape[-len(self.n_neuron) :]) != self.n_neuron:
            raise ValueError(
                f"Expected trailing neuron shape {self.n_neuron}, got "
                f"{tuple(x_seq.shape[-len(self.n_neuron):])}."
            )
        if x_seq.device != self.v_threshold_base.device:
            raise ValueError("x_seq.device must match the Triton module device.")
        if x_seq.dtype != self.v_threshold_base.dtype:
            raise ValueError("x_seq.dtype must match the Triton module dtype.")

        state_shape = tuple(x_seq.shape[1:])
        if state_shape != self._state_shape or self.v.device != x_seq.device:
            batch_ndim = len(state_shape) - len(self.n_neuron)
            self.reset_state(x_seq.shape[1 : 1 + batch_ndim])

        x_flat = x_seq.contiguous().reshape(x_seq.shape[0], -1)
        spikes, v_out, refractory_out = triton_lif_multistep(
            x_flat,
            self._v_flat,
            self._v_threshold_flat,
            self._v_reset_flat,
            self._c_m_flat,
            self._tau_flat,
            dt=dt,
            hard_reset=self.hard_reset,
            refractory=self._refractory_flat if self._use_refractory else None,
            tau_ref=self._tau_ref_flat if self._use_refractory else None,
        )

        self.v = v_out.reshape(state_shape)
        self._v_flat = v_out
        if self._use_refractory and refractory_out is not None:
            self.refractory = refractory_out.reshape(state_shape)
            self._refractory_flat = refractory_out

        return spikes.reshape_as(x_seq)


class TritonSingleStepLIF(torch.nn.Module):
    """Stateful forward-only Triton LIF module for a single time step."""

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        *,
        v_threshold: float | torch.Tensor = 1.0,
        v_reset: float | torch.Tensor = 0.0,
        c_m: float | torch.Tensor = 1.0,
        tau: float | torch.Tensor = 20.0,
        hard_reset: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.n_neuron = _normalize_n_neuron(n_neuron)
        self.hard_reset = hard_reset

        self.register_buffer(
            "v_threshold_base",
            _expand_param(v_threshold, self.n_neuron, device=device, dtype=dtype),
        )
        self.register_buffer(
            "v_reset_base",
            _expand_param(v_reset, self.n_neuron, device=device, dtype=dtype),
        )
        self.register_buffer(
            "c_m_base",
            _expand_param(c_m, self.n_neuron, device=device, dtype=dtype),
        )
        self.register_buffer(
            "tau_base",
            _expand_param(tau, self.n_neuron, device=device, dtype=dtype),
        )
        self.register_buffer("v", self.v_reset_base.clone())
        self._state_shape = self.v.shape
        self._refresh_flat_views()

    def _expand_base(self, tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.broadcast_to(tensor, shape).contiguous()

    def _refresh_flat_views(self) -> None:
        state_shape = tuple(self.v.shape)
        self._v_flat = self.v.reshape(-1).contiguous()
        self._v_threshold_flat = self._expand_base(
            self.v_threshold_base, state_shape
        ).reshape(-1)
        self._v_reset_flat = self._expand_base(self.v_reset_base, state_shape).reshape(-1)
        self._c_m_flat = self._expand_base(self.c_m_base, state_shape).reshape(-1)
        self._tau_flat = self._expand_base(self.tau_base, state_shape).reshape(-1)

    @torch.no_grad()
    def reset_state(self, batch_size: int | Sequence[int] | None = None) -> None:
        if batch_size is None:
            batch_shape: tuple[int, ...] = ()
        elif isinstance(batch_size, int):
            batch_shape = (batch_size,)
        else:
            batch_shape = tuple(batch_size)

        state_shape = batch_shape + self.n_neuron
        self.v = self._expand_base(self.v_reset_base, state_shape)
        self._state_shape = state_shape
        self._refresh_flat_views()

    def forward(self, x: torch.Tensor, *, dt: float) -> torch.Tensor:
        if x.ndim < 1:
            raise ValueError("x must have shape (*batch, n_neuron).")
        if tuple(x.shape[-len(self.n_neuron) :]) != self.n_neuron:
            raise ValueError(
                f"Expected trailing neuron shape {self.n_neuron}, got "
                f"{tuple(x.shape[-len(self.n_neuron):])}."
            )
        if x.device != self.v_threshold_base.device:
            raise ValueError("x.device must match the Triton module device.")
        if x.dtype != self.v_threshold_base.dtype:
            raise ValueError("x.dtype must match the Triton module dtype.")

        state_shape = tuple(x.shape)
        if state_shape != self._state_shape or self.v.device != x.device:
            batch_ndim = len(state_shape) - len(self.n_neuron)
            self.reset_state(x.shape[:batch_ndim])

        spikes, v_out = triton_lif_single_step(
            x.contiguous().reshape(-1),
            self._v_flat,
            self._v_threshold_flat,
            self._v_reset_flat,
            self._c_m_flat,
            self._tau_flat,
            dt=dt,
            hard_reset=self.hard_reset,
        )
        self.v = v_out.reshape(state_shape)
        self._v_flat = v_out
        return spikes.reshape_as(x)


class TritonSparseLIFRNN(base.MemoryModule):
    """Single-layer sparse recurrent LIF network using per-step Triton kernels.

    Each time step runs exactly two kernels in sequence:
    1. sparse recurrent current via COO SpMM
    2. single-step Triton LIF update
    """

    def __init__(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        *,
        n_neuron: int | Sequence[int],
        v_threshold: float | torch.Tensor = 1.0,
        v_reset: float | torch.Tensor = 0.0,
        c_m: float | torch.Tensor = 1.0,
        tau: float | torch.Tensor = 20.0,
        hard_reset: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.n_neuron = _normalize_n_neuron(n_neuron)
        self.size = math.prod(self.n_neuron)
        self.hard_reset = hard_reset

        if indices.shape[0] != 2:
            raise ValueError("indices must have shape (2, nnz).")
        self.register_buffer(
            "indices",
            indices.to(device=device, dtype=torch.long).contiguous(),
        )
        self.register_buffer(
            "values",
            values.to(device=device, dtype=dtype or values.dtype).contiguous(),
        )
        self.register_buffer(
            "v_threshold_base",
            _expand_param(v_threshold, self.n_neuron, device=device, dtype=dtype),
        )
        self.register_buffer(
            "v_reset_base",
            _expand_param(v_reset, self.n_neuron, device=device, dtype=dtype),
        )
        self.register_buffer(
            "c_m_base",
            _expand_param(c_m, self.n_neuron, device=device, dtype=dtype),
        )
        self.register_buffer(
            "tau_base",
            _expand_param(tau, self.n_neuron, device=device, dtype=dtype),
        )

        self.register_memory("v", self.v_reset_base, self.n_neuron)
        self.register_memory("spike", 0.0, self.n_neuron)
        self.init_state(device=device, dtype=dtype)
        self._state_shape = tuple(self.v.shape)
        self._refresh_flat_views()

    def _expand_base(self, tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.broadcast_to(tensor, shape).contiguous()

    def _refresh_flat_views(self) -> None:
        state_shape = tuple(self.v.shape)
        self._v_flat = self.v.reshape(-1).contiguous()
        self._v_threshold_flat = self._expand_base(
            self.v_threshold_base, state_shape
        ).reshape(-1)
        self._v_reset_flat = self._expand_base(self.v_reset_base, state_shape).reshape(-1)
        self._c_m_flat = self._expand_base(self.c_m_base, state_shape).reshape(-1)
        self._tau_flat = self._expand_base(self.tau_base, state_shape).reshape(-1)

    @torch.no_grad()
    def reset(
        self,
        batch_size=None,
        dtype=None,
        device=None,
        skip_mem_name: tuple[str, ...] = (),
    ):
        super().reset(
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            skip_mem_name=skip_mem_name,
        )
        self._state_shape = tuple(self.v.shape)
        self._refresh_flat_views()

    @torch.no_grad()
    def reset_state(self, batch_size: int | Sequence[int] | None = None) -> None:
        self.reset(
            batch_size=batch_size,
            device=self.values.device,
            dtype=self.values.dtype,
        )

    def _ensure_state_shape(self, state_shape: tuple[int, ...]) -> None:
        if state_shape != self._state_shape or self.v.device != self.values.device:
            batch_shape = state_shape[: len(state_shape) - len(self.n_neuron)]
            self.reset(
                batch_size=batch_shape,
                device=self.values.device,
                dtype=self.values.dtype,
            )
            self._state_shape = tuple(self.v.shape)
            self._refresh_flat_views()

    def _recurrent_current(self) -> torch.Tensor:
        spike_2d = self.spike.reshape(-1, self.size)
        current_t = coo_spmm(
            self.indices,
            self.values,
            spike_2d.T.contiguous(),
            size_m=self.size,
        )
        return current_t.T.reshape_as(self.spike)

    def single_step_forward(self, x_t: torch.Tensor, *, dt: float) -> torch.Tensor:
        if x_t.ndim < 1:
            raise ValueError("x_t must have shape (*batch, n_neuron).")
        if tuple(x_t.shape[-len(self.n_neuron) :]) != self.n_neuron:
            raise ValueError(
                f"Expected trailing neuron shape {self.n_neuron}, got "
                f"{tuple(x_t.shape[-len(self.n_neuron):])}."
            )
        if x_t.device != self.values.device:
            raise ValueError("x_t.device must match the recurrent weight device.")
        if x_t.dtype != self.values.dtype:
            raise ValueError("x_t.dtype must match the recurrent weight dtype.")
        self._ensure_state_shape(tuple(x_t.shape))

        current = x_t + self._recurrent_current()
        spikes, v_out = triton_lif_single_step(
            current.contiguous().reshape(-1),
            self._v_flat,
            self._v_threshold_flat,
            self._v_reset_flat,
            self._c_m_flat,
            self._tau_flat,
            dt=dt,
            hard_reset=self.hard_reset,
        )
        self.v = v_out.reshape_as(current)
        self.spike = spikes.reshape_as(current)
        self._v_flat = v_out
        return self.spike

    def forward(self, x_seq: torch.Tensor, *, dt: float) -> torch.Tensor:
        if x_seq.ndim < 2:
            raise ValueError("x_seq must have shape (T, *batch, n_neuron).")
        spikes = []
        for t in range(x_seq.shape[0]):
            spikes.append(self.single_step_forward(x_seq[t], dt=dt))
        return torch.stack(spikes, dim=0)
