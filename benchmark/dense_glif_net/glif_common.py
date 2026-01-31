from __future__ import annotations

import importlib.util
from typing import Callable

import torch
import torch.nn as nn

from btorch.models import environ
from btorch.models.base import MemoryModule
from btorch.models.functional import init_net_state, reset_net_state
from btorch.models.neurons.glif import GLIF3
from btorch.models.rnn import RecurrentNNAbstract
from btorch.models.surrogate import ATanApprox


_DT = 1.0
_ALPHA = 2.0
_M = 2
_HARD_RESET = False


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
        if (
            hasattr(self.neuron, "step_fn")
            and hasattr(self.neuron.step_fn, "dense_multistep_fused")
            and not torch.is_grad_enabled()
        ):
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
            surrogate_function=ATanApprox(alpha=_ALPHA, spiking=True),
            trainable_param=trainable,
            step_mode="s",
            backend="torch",
            device=device,
            dtype=torch.float32,
        )
        return neuron

    if provider == "triton":
        from benchmark.dense_glif_net.glif_triton import glif3_step_triton

        step_fn = glif3_step_triton
    elif provider == "warp":
        from benchmark.dense_glif_net.glif_warp import glif3_step_warp

        step_fn = glif3_step_warp
    elif provider == "cupy":
        from benchmark.dense_glif_net.glif_cupy import glif3_step_cupy

        step_fn = glif3_step_cupy
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
