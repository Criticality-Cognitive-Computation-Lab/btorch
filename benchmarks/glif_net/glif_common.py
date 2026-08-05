from __future__ import annotations

import importlib.util
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from btorch.models import environ
from btorch.models.base import MemoryModule
from btorch.models.functional import init_net_state, reset_net_state
from btorch.models.neurons.glif import GLIF3
from btorch.models.rnn import RecurrentNNAbstract
from btorch.models.surrogate import ATan


_DT = 1.0
_ALPHA = 2.0
_M = 2
_HARD_RESET = False


@dataclass(frozen=True)
class GLIF3StepOps:
    """A backend's GLIF3 kernel entry points behind one object.

    Call it for the single (training) step; use ``.multistep_fused`` /
    ``.dense_multistep_fused`` / ``.sparse_multistep_fused`` for the fused
    multistep paths.
    """

    step: Callable
    multistep_fused: Callable
    dense_multistep_fused: Callable
    sparse_multistep_fused: Callable = None

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)


# A sparse recurrent weight, preprocessed once. ``crow``/``col``/``val`` are the
# CSR arrays the fused SpMV kernels read; ``coo_indices`` = stack([row, col]) is
# kept so the training path can wrap fresh ``val`` in a COO tensor with
# ``is_coalesced=True`` (no runtime coalesce). All tensors are contiguous int32/
# fp32 on device; edges are sorted + deduplicated.
SparseWeight = namedtuple("SparseWeight", ["crow", "col", "val", "coo_indices", "N"])


def scale_free_csr(N: int, density: float, device, seed: int = 0,
                   gamma: float = 2.5) -> SparseWeight:
    """Scale-free recurrent connectivity. In-degrees (nonzeros per row) follow a
    power law and columns are drawn by preferential attachment, so a few hub
    neurons dominate — the heavy load imbalance a good SpMV must handle. Coalesced
    once here (the one-time preprocess); nothing is re-sorted at runtime."""
    rng = np.random.default_rng(seed)
    nnz_target = int(density * N * N)
    deg = rng.pareto(gamma - 1.0, size=N) + 1.0
    deg = np.clip(np.round(deg / deg.sum() * nnz_target).astype(np.int64), 1, N)
    pop = rng.pareto(gamma - 1.0, size=N) + 1.0
    col = rng.choice(N, size=int(deg.sum()), p=pop / pop.sum())
    row = np.repeat(np.arange(N), deg)
    val = rng.standard_normal(len(col)) / np.sqrt(deg.mean())

    idx = torch.stack([torch.from_numpy(row), torch.from_numpy(col)])
    coo = torch.sparse_coo_tensor(idx, torch.from_numpy(val).float(), (N, N)).coalesce()
    rows, cols = coo.indices()
    crow = torch.zeros(N + 1, dtype=torch.int32)
    crow[1:] = torch.bincount(rows, minlength=N).cumsum(0)
    return SparseWeight(
        crow.to(device), cols.int().contiguous().to(device),
        coo.values().to(device), coo.indices().to(device), N)


class _SparseSpmv(torch.autograd.Function):
    """lin = W @ s for a COO weight, with gradients to the values and to ``s``.

    Both forward and backward are O(nnz) scatter/gathers, so nothing dense is
    ever materialized. This is why we do NOT use ``torch.sparse.mm`` for training:
    its backward w.r.t. the sparse values builds a dense (N, N) gradient, which
    OOMs at N=2**15 (~27 GB) even though the final gradient is only nnz-sized.
    """

    @staticmethod
    def forward(ctx, rows, cols, N, val, s):
        lin = torch.zeros(N, device=s.device, dtype=s.dtype)
        lin.index_add_(0, rows, val * s[cols])
        ctx.save_for_backward(rows, cols, val, s)
        ctx.N = N
        return lin

    @staticmethod
    def backward(ctx, glin):
        rows, cols, val, s = ctx.saved_tensors
        grad_val = glin[rows] * s[cols]           # nnz-sized, no dense (N, N)
        grad_s = torch.zeros(ctx.N, device=s.device, dtype=s.dtype)
        grad_s.index_add_(0, cols, val * glin[rows])
        return None, None, None, grad_val, grad_s


def sparse_spmv(weight: SparseWeight, s: torch.Tensor) -> torch.Tensor:
    """lin = W @ s with gradients to the weight values and to ``s``."""
    return _SparseSpmv.apply(
        weight.coo_indices[0].long(), weight.coo_indices[1].long(),
        weight.N, weight.val, s)


def sparse_multistep_autograd(
    step: Callable,
    x_seq: torch.Tensor,
    weight: SparseWeight,
    bias: torch.Tensor,
    v: torch.Tensor,
    Iasc: torch.Tensor,
    params: dict,
    not_refrac: torch.Tensor,
    dt: float,
    M: int,
    hard_reset: bool,
    alpha: float,
    spmv: Callable = sparse_spmv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autograd sparse-recurrent multistep for training: single-step op composed
    with a differentiable sparse SpMV each step. Shared by every backend; ``spmv``
    lets a backend inject its own (e.g. Warp's tape-based SpMV) in place of the
    default nnz-sized torch op."""
    T, B = x_seq.shape
    s_prev = torch.zeros(B, device=x_seq.device, dtype=x_seq.dtype)
    v_cur, I_cur = v, Iasc.reshape(-1)
    spikes, voltages = [], []
    for t in range(T):
        x_in = x_seq[t] + bias + spmv(weight, s_prev)
        v_cur, I_cur, s_prev = step(
            v=v_cur, Iasc=I_cur, x=x_in, params=params, not_refrac=not_refrac,
            dt=dt, M=M, hard_reset=hard_reset, alpha=alpha,
        )
        spikes.append(s_prev)
        voltages.append(v_cur)
    return torch.stack(spikes), torch.stack(voltages), v_cur, I_cur.view(B, M)


def dense_multistep_autograd(
    step: Callable,
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    v: torch.Tensor,
    Iasc: torch.Tensor,
    params: dict,
    not_refrac: torch.Tensor,
    dt: float,
    M: int,
    hard_reset: bool,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autograd dense (neuron + recurrent connection) multistep for training.

    Composes the single-step autograd op with a torch matmul each step, so torch
    builds the graph and the gradients for ``weight`` / ``bias`` / ``x_seq`` /
    initial state / ``asc_amps`` all fall out of the single-step backward. Shared
    by every backend since it only depends on the backend's single-step op.
    """
    T, B = x_seq.shape
    s_prev = torch.zeros(B, device=x_seq.device, dtype=x_seq.dtype)
    v_cur, I_cur = v, Iasc.reshape(-1)
    spikes, voltages = [], []
    for t in range(T):
        x_in = x_seq[t] + bias + torch.mv(weight, s_prev)
        v_cur, I_cur, s_prev = step(
            v=v_cur, Iasc=I_cur, x=x_in, params=params, not_refrac=not_refrac,
            dt=dt, M=M, hard_reset=hard_reset, alpha=alpha,
        )
        spikes.append(s_prev)
        voltages.append(v_cur)
    return torch.stack(spikes), torch.stack(voltages), v_cur, I_cur.view(B, M)


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def providers() -> list[str]:
    backend = ["torch_eager"]
    if hasattr(torch, "compile"):
        backend.append("torch_compile")
    if has_module("triton"):
        backend.append("triton")
    if has_module("warp"):
        backend.append("warp")
    if has_module("cupy"):
        backend.append("cupy")
    return backend


class GLIFDenseNet(RecurrentNNAbstract):
    def __init__(self, n_neuron: int, neuron: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.neuron = neuron
        self.linear = nn.Linear(n_neuron, n_neuron)
        self.register_memory("spike", 0.0, n_neuron)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if hasattr(self.neuron, "step_fn") and hasattr(
            self.neuron.step_fn, "dense_multistep_fused"
        ):
            # dense_multistep_fused routes internally: a lean fused/cuBLAS forward
            # under no_grad, an autograd path when gradients are required.
            step_fn = self.neuron.step_fn
            spike_seq, v_seq, v_out, I_out = step_fn.dense_multistep_fused(
                x_seq=x_seq,
                weight=self.linear.weight,
                bias=self.linear.bias,
                v=self.neuron.v,
                Iasc=self.neuron.Iasc,
                params={
                    "v_th": self.neuron.v_th,
                    "v_reset": self.neuron.v_reset,
                    "v_rest": self.neuron.v_rest,
                    "c_m": self.neuron.c_m,
                    "tau": self.neuron.tau,
                    "k": self.neuron.k.reshape(-1),
                    "asc_amps": self.neuron.asc_amps.reshape(-1),
                },
                not_refrac=self.neuron.not_refrac,
                dt=self.neuron.dt,
                M=self.neuron.M,
                hard_reset=self.neuron.hard_reset,
                alpha=self.neuron.alpha,
            )
            self.neuron.v = v_out
            self.neuron.Iasc = I_out
            self.spike = spike_seq[-1]
            return spike_seq, {"v": v_seq}
        return super().multi_step_forward(x_seq)

    def single_step_forward(self, x):
        x = x + self.linear(self.spike)
        z = self.neuron(x)
        self.spike = z
        return z, {"v": self.neuron.v}


class GLIFSparseNet(RecurrentNNAbstract):
    """Eager sparse-recurrent GLIF3 net for the torch.compile baseline: the
    recurrent term is a 5% scale-free SpMV done with a **CSR** sparse tensor, so
    ``torch.sparse.mm`` dispatches to **cuSPARSE** (``csrmv``) rather than
    torch's slower native COO path. The CSR tensor is rebuilt inside the step
    from stored ``crow``/``col`` buffers and a ``val`` parameter so autograd
    stays per-call (a persistent coalesced tensor would be backwarded twice)."""

    def __init__(self, n_neuron: int, neuron: nn.Module, weight, bias, **kwargs):
        super().__init__(**kwargs)
        self.neuron = neuron
        self.n_neuron = int(n_neuron)
        self.register_buffer("crow", weight.crow)
        self.register_buffer("col", weight.col)
        self.val = nn.Parameter(weight.val.detach().clone())
        self.register_buffer("bias", bias)
        self.register_memory("spike", 0.0, n_neuron)

    def single_step_forward(self, x):
        W = torch.sparse_csr_tensor(self.crow, self.col, self.val,
                                    (self.n_neuron, self.n_neuron))
        x = x + torch.sparse.mm(W, self.spike.reshape(-1, 1)).reshape(-1) + self.bias
        z = self.neuron(x)
        self.spike = z
        return z, {"v": self.neuron.v}


class GLIF3Kernel(MemoryModule):
    def __init__(
        self,
        n_neuron: int,
        params: dict,
        step_fn: Callable,
        trainable: bool,
    ):
        super().__init__()
        self.M = int(params["k"].shape[-1])
        self.n_neuron = int(n_neuron)
        self.step_fn = step_fn
        self.dt = float(_DT)
        self.hard_reset = bool(_HARD_RESET)
        self.alpha = float(_ALPHA)

        self.register_memory("v", params["v_reset"], self.n_neuron)
        self.register_memory("Iasc", 0.0, (self.n_neuron, self.M))
        self.register_buffer("v_th", params["v_th"])
        self.register_buffer("v_reset", params["v_reset"])
        self.register_buffer("v_rest", params["v_rest"])
        self.register_buffer("c_m", params["c_m"])
        self.register_buffer("tau", params["tau"])
        self.register_buffer("k", params["k"])
        if trainable:
            self.asc_amps = nn.Parameter(params["asc_amps"])
        else:
            self.register_buffer("asc_amps", params["asc_amps"])
        self.register_buffer(
            "not_refrac",
            torch.ones(self.n_neuron, device=params["v_reset"].device),
        )

    def forward(self, x: torch.Tensor):
        v = self.v
        Iasc = self.Iasc.reshape(-1)
        params = {
            "v_th": self.v_th,
            "v_reset": self.v_reset,
            "v_rest": self.v_rest,
            "c_m": self.c_m,
            "tau": self.tau,
            "k": self.k.reshape(-1),
            "asc_amps": self.asc_amps.reshape(-1),
        }
        v_out, I_out, s_out = self.step_fn(
            v=v,
            Iasc=Iasc,
            x=x,
            params=params,
            not_refrac=self.not_refrac,
            dt=self.dt,
            M=self.M,
            hard_reset=self.hard_reset,
            alpha=self.alpha,
        )
        self.v = v_out
        self.Iasc = I_out.view(self.n_neuron, self.M)
        return s_out


def _make_params(N: int, device: torch.device):
    gen = torch.Generator(device=device).manual_seed(123)
    weight = torch.randn((N, N), generator=gen, device=device, dtype=torch.float32)
    weight = weight / max(1, N) ** 0.5
    bias = torch.randn((N,), generator=gen, device=device, dtype=torch.float32)

    v_th = torch.full((N,), -50.0, device=device, dtype=torch.float32)
    v_reset = torch.full((N,), -70.0, device=device, dtype=torch.float32)
    v_rest = v_reset.clone()
    c_m = torch.full((N,), 0.05, device=device, dtype=torch.float32)
    tau = torch.full((N,), 20.0, device=device, dtype=torch.float32)
    k = 0.1 + 0.2 * torch.rand((N, _M), generator=gen, device=device)
    asc_amps = 0.05 * torch.randn((N, _M), generator=gen, device=device)

    params = {
        "v_th": v_th,
        "v_reset": v_reset,
        "v_rest": v_rest,
        "c_m": c_m,
        "tau": tau,
        "k": k,
        "asc_amps": asc_amps,
    }
    return weight, bias, params


def make_inputs(
    T: int, N: int, device: torch.device, require_grad: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    weight, bias, params = _make_params(N, device)
    gen = torch.Generator(device=device).manual_seed(456)
    x_seq = torch.randn((T, N), generator=gen, device=device, dtype=torch.float32)

    if require_grad:
        weight.requires_grad_(True)
        bias.requires_grad_(True)
        x_seq.requires_grad_(True)
        params["asc_amps"].requires_grad_(True)

    return weight, bias, x_seq, params


def build_neuron(provider: str, N: int, params: dict, require_grad: bool):
    device = params["v_reset"].device
    if provider in ("torch_eager", "torch_compile"):
        trainable = {"asc_amps"} if require_grad else set()
        neuron = GLIF3(
            n_neuron=N,
            v_threshold=params["v_th"],
            v_reset=params["v_reset"],
            v_rest=params["v_rest"],
            c_m=params["c_m"],
            tau=params["tau"],
            k=params["k"],
            asc_amps=params["asc_amps"],
            tau_ref=0.0,
            hard_reset=_HARD_RESET,
            # Exact ATan (1/(1+(alpha·u)²)) — matches what every kernel implements
            # in its backward, so gradient parity holds off-threshold, not just near it.
            surrogate_function=ATan(alpha=_ALPHA, spiking=True),
            trainable_param=trainable,
            step_mode="s",
            backend="torch",
            device=device,
            dtype=torch.float32,
        )
        return neuron

    if provider == "triton":
        from benchmarks.glif_net.glif_triton import glif3_step_triton

        step_fn = glif3_step_triton
    elif provider == "warp":
        from benchmarks.glif_net.glif_warp import glif3_step_warp

        step_fn = glif3_step_warp
    elif provider == "cupy":
        from benchmarks.glif_net.glif_cupy import glif3_step_cupy

        step_fn = glif3_step_cupy
    elif provider == "tilelang":
        from benchmarks.glif_net.glif_tilelang import glif3_step_tilelang

        step_fn = glif3_step_tilelang
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return GLIF3Kernel(
        n_neuron=N,
        params=params,
        step_fn=step_fn,
        trainable=require_grad,
    )


def build_model(
    provider: str, T: int, N: int, require_grad: bool
) -> tuple[nn.Module, torch.Tensor, list[torch.Tensor]]:
    device = torch.device("cuda")
    weight, bias, x_seq, params = make_inputs(T, N, device, require_grad)
    neuron = build_neuron(provider, N, params, require_grad)

    model = GLIFDenseNet(N, neuron, unroll=8)
    init_net_state(model, device=device, dtype=torch.float32)
    model.linear.weight.data.copy_(weight)
    model.linear.bias.data.copy_(bias)

    grads = [model.linear.weight, model.linear.bias, x_seq]
    if require_grad:
        grads.append(model.neuron.asc_amps)
    return model, x_seq, grads


def run_model(model: nn.Module, x_seq: torch.Tensor):
    reset_net_state(model)
    with environ.context(dt=_DT):
        spike_seq, _ = model(x_seq)
    return spike_seq
