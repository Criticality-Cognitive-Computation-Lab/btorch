from __future__ import annotations

from typing import Callable

import torch
from triton.testing import Benchmark, do_bench, perf_report

from benchmark.dense_glif_net.glif_common import (
    _DT,
    build_model,
    providers,
    run_model,
)
from btorch.models import environ
from btorch.models.functional import reset_net_state
from btorch.utils.file import fig_path


_MAX_N = 4096 * 4
_MAX_T = 2048


def _unique_sorted(values: list[int]) -> list[int]:
    return sorted(set(values))


_T_SWEEP = _unique_sorted(
    [int(v) for v in torch.logspace(6, 11, 16, base=2) if int(v) <= _MAX_T]
)
_N_SWEEP = _unique_sorted(
    [int(v) for v in torch.logspace(3, 5, 16, base=10) if int(v) <= _MAX_N]
)


_PROVIDERS = providers()
_RECOMPILE_LIMIT = 4 * (len(_T_SWEEP) + len(_N_SWEEP))
torch._dynamo.config.recompile_limit = _RECOMPILE_LIMIT
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


def _bench_ms(fn: Callable, grads: list[torch.Tensor] | None):
    ms = do_bench(
        fn,
        quantiles=[0.5, 0.2, 0.8],
        grad_to_none=grads,
    )
    return tuple(ms)


def _warmup_forward(fn: Callable, steps: int = 2):
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()


def _warmup_forward_backward(fn: Callable, grads: list[torch.Tensor], steps: int = 2):
    for _ in range(steps):
        fn()
        for g in grads:
            if g.grad is not None:
                g.grad.zero_()
    torch.cuda.synchronize()


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

    model, x_seq, _ = build_model(provider, T, N, require_grad=False)

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

        def warmup_fn():
            with torch.no_grad():
                reset_net_state(model)
                with environ.context(dt=_DT):
                    compiled(x_seq)
    else:

        def fn():
            with torch.no_grad():
                run_model(model, x_seq)

        def warmup_fn():
            with torch.no_grad():
                run_model(model, x_seq)

    _warmup_forward(warmup_fn)
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

    model, x_seq, grads = build_model(provider, T, N, require_grad=True)

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

        def warmup_fn():
            reset_net_state(model)
            with environ.context(dt=_DT):
                spike_seq_inner = compiled(x_seq)[0]
            spike_seq_inner.sum().backward()
    else:

        def fn():
            spike_seq_inner = run_model(model, x_seq)
            spike_seq_inner.sum().backward()

        def warmup_fn():
            spike_seq_inner = run_model(model, x_seq)
            spike_seq_inner.sum().backward()

    _warmup_forward_backward(warmup_fn, grads)
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
