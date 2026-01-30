from __future__ import annotations

import importlib.util
from typing import Callable

import torch
import torch.nn as nn
from triton.testing import Benchmark, do_bench, perf_report

from btorch.models import environ
from btorch.models.base import MemoryModule
from btorch.models.functional import init_net_state, reset_net_state
from btorch.models.neurons.glif import GLIF3
from btorch.models.rnn import make_rnn
from btorch.models.surrogate import ATanApprox
from btorch.utils.file import fig_path


_DT = 1.0
_ALPHA = 2.0
_M = 2
_HARD_RESET = False
_T_SWEEP = [int(v) for v in torch.logspace(6, 11, 16, base=2)]
_N_SWEEP = [int(v) for v in torch.logspace(3, 5, 16, base=10)]


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _providers() -> list[str]:
    providers = ["torch_eager"]
    if hasattr(torch, "compile"):
        providers.append("torch_compile")
    if _has_module("triton"):
        providers.append("triton")
    if _has_module("warp"):
        providers.append("warp")
    if _has_module("cupy"):
        providers.append("cupy")
    return providers


_PROVIDERS = _providers()
_LINE_NAMES = {
    "torch_eager": "Torch Eager",
    "torch_compile": "Torch Compile",
    "triton": "Triton",
    "warp": "Warp",
    "cupy": "CuPy",
}
_STYLES = [
    ("red", "-"),
    ("blue", "-"),
    ("green", "-"),
    ("orange", "-"),
    ("purple", "-"),
]


class GLIFDenseNet(MemoryModule):
    def __init__(self, n_neuron: int, neuron: nn.Module):
        super().__init__()
        self.neuron = neuron
        self.linear = nn.Linear(n_neuron, n_neuron)
        self.register_memory("spike", 0.0, n_neuron)

    def forward(self, x):
        x = x + self.linear(self.spike)
        z = self.neuron(x)
        self.spike = z
        return z


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


def _make_inputs(
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


def _build_neuron(provider: str, N: int, params: dict, require_grad: bool):
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


def _build_model(
    provider: str, T: int, N: int, require_grad: bool
) -> tuple[nn.Module, torch.Tensor, list[torch.Tensor]]:
    device = torch.device("cuda")
    weight, bias, x_seq, params = _make_inputs(T, N, device, require_grad)
    neuron = _build_neuron(provider, N, params, require_grad)

    model = make_rnn(GLIFDenseNet)(N, neuron)
    init_net_state(model, device=device, dtype=torch.float32)
    model.rnn_cell.linear.weight.data.copy_(weight)
    model.rnn_cell.linear.bias.data.copy_(bias)

    grads = [model.rnn_cell.linear.weight, model.rnn_cell.linear.bias, x_seq]
    if require_grad:
        grads.append(model.rnn_cell.neuron.asc_amps)
    return model, x_seq, grads


def _bench_ms(fn: Callable, grads: list[torch.Tensor] | None):
    ms = do_bench(
        fn,
        quantiles=[0.5, 0.2, 0.8],
        grad_to_none=grads,
    )
    return tuple(ms)


def _run_model(model: nn.Module, x_seq: torch.Tensor):
    reset_net_state(model)
    with environ.context(dt=_DT):
        spike_seq, _ = model(x_seq)
    return spike_seq


@perf_report(
    [
        Benchmark(
            x_names=["T"],
            x_vals=_T_SWEEP,
            line_arg="provider",
            line_vals=_PROVIDERS,
            line_names=[_LINE_NAMES[p] for p in _PROVIDERS],
            styles=_STYLES[: len(_PROVIDERS)],
            ylabel="ms",
            plot_name="dense_glif_forward_vs_T",
            args={"N": 64},
        ),
        Benchmark(
            x_names=["N"],
            x_vals=_N_SWEEP,
            line_arg="provider",
            line_vals=_PROVIDERS,
            line_names=[_LINE_NAMES[p] for p in _PROVIDERS],
            styles=_STYLES[: len(_PROVIDERS)],
            ylabel="ms",
            plot_name="dense_glif_forward_vs_n_neuron",
            args={"T": 100},
        ),
    ]
)
def bench_dense_glif_forward(T: int, N: int, provider: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GLIF benchmarks.")

    model, x_seq, _ = _build_model(provider, T, N, require_grad=False)

    if provider == "torch_compile":
        compiled = torch.compile(model)
        reset_net_state(model)
        with environ.context(dt=_DT):
            compiled(x_seq)
        torch.cuda.synchronize()

        def fn():
            with torch.no_grad():
                reset_net_state(model)
                with environ.context(dt=_DT):
                    compiled(x_seq)
    else:

        def fn():
            with torch.no_grad():
                _run_model(model, x_seq)

    return _bench_ms(fn, grads=None)


@perf_report(
    [
        Benchmark(
            x_names=["T"],
            x_vals=_T_SWEEP,
            line_arg="provider",
            line_vals=_PROVIDERS,
            line_names=[_LINE_NAMES[p] for p in _PROVIDERS],
            styles=_STYLES[: len(_PROVIDERS)],
            ylabel="ms",
            plot_name="dense_glif_forward_backward_vs_T",
            args={"N": 128},
        ),
        Benchmark(
            x_names=["N"],
            x_vals=_N_SWEEP,
            line_arg="provider",
            line_vals=_PROVIDERS,
            line_names=[_LINE_NAMES[p] for p in _PROVIDERS],
            styles=_STYLES[: len(_PROVIDERS)],
            ylabel="ms",
            plot_name="dense_glif_forward_backward_vs_n_neuron",
            args={"T": 100},
        ),
    ]
)
def bench_dense_glif_forward_backward(T: int, N: int, provider: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GLIF benchmarks.")

    model, x_seq, grads = _build_model(provider, T, N, require_grad=True)

    if provider == "torch_compile":
        reset_net_state(model)
        compiled = torch.compile(model)
        with environ.context(dt=_DT):
            spike_seq = compiled(x_seq)[0]
        spike_seq.sum().backward()
        torch.cuda.synchronize()

        def fn():
            reset_net_state(model)
            with environ.context(dt=_DT):
                spike_seq_inner = compiled(x_seq)[0]
            spike_seq_inner.sum().backward()
    else:

        def fn():
            spike_seq_inner = _run_model(model, x_seq)
            spike_seq_inner.sum().backward()

    return _bench_ms(fn, grads=grads)


if __name__ == "__main__":
    bench_dense_glif_forward.run(
        show_plots=False,
        print_data=False,
        return_df=False,
        save_path=fig_path(__file__),
    )
    bench_dense_glif_forward_backward.run(
        show_plots=False,
        print_data=False,
        return_df=False,
        save_path=fig_path(__file__),
    )
