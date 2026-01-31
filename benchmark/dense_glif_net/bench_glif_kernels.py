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
_PROVIDERS.remove("triton")  # unsure why triton so slow
for _name in list(_PROVIDERS):
    # disable "triton", fused compilation takes really long
    if _name in {"warp", "cupy"}:
        _PROVIDERS.append(f"fused_{_name}")
_RECOMPILE_LIMIT = 4 * (len(_T_SWEEP) + len(_N_SWEEP))
torch._dynamo.config.recompile_limit = _RECOMPILE_LIMIT
_LINE_NAMES = {
    "torch_eager": "Torch Eager",
    "torch_compile": "Torch Compile",
    "triton": "Triton",
    "warp": "Warp",
    "cupy": "CuPy",
    "fused_triton": "Fused Triton",
    "fused_warp": "Fused Warp",
    "fused_cupy": "Fused CuPy",
}
_STYLE_MAP = {
    "torch_eager": ("black", "-"),
    "torch_compile": ("gray", "-"),
    "triton": ("red", "-"),
    "warp": ("blue", "-"),
    "cupy": ("green", "-"),
    "fused_triton": ("red", "--"),
    "fused_warp": ("blue", "--"),
    "fused_cupy": ("green", "--"),
}
_STYLES = [_STYLE_MAP[p] for p in _PROVIDERS]


def _bench_ms(fn: Callable, grads: list[torch.Tensor] | None, *, use_quantiles: bool):
    quantiles = [0.5, 0.2, 0.8] if use_quantiles else None
    ms = do_bench(
        fn,
        warmup=1,
        rep=5,
        quantiles=quantiles,
        grad_to_none=grads,
    )
    if use_quantiles:
        return tuple(ms)
    return (ms, None, None)


def _run_dense_multistep(model: torch.nn.Module, x_seq: torch.Tensor):
    neuron = model.neuron
    step_fn = neuron.step_fn
    if hasattr(step_fn, "dense_multistep_fused") and not torch.is_grad_enabled():
        spike_seq, _, v_out, I_out = step_fn.dense_multistep_fused(
            x_seq=x_seq,
            weight=model.linear.weight,
            bias=model.linear.bias,
            v=neuron.v,
            Iasc=neuron.Iasc,
            params={
                "v_th": neuron.v_th,
                "v_reset": neuron.v_reset,
                "v_rest": neuron.v_rest,
                "c_m": neuron.c_m,
                "tau": neuron.tau,
                "k": neuron.k.reshape(-1),
                "asc_amps": neuron.asc_amps.reshape(-1),
            },
            not_refrac=neuron.not_refrac,
            dt=neuron.dt,
            M=neuron.M,
            hard_reset=neuron.hard_reset,
            alpha=neuron.alpha,
        )
        neuron.v = v_out
        neuron.Iasc = I_out
        model.spike = spike_seq[-1]
        return spike_seq
    raise RuntimeError("Multistep fused kernel not available for this backend.")


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

    use_multistep = provider.startswith("fused_")
    base_provider = provider.removeprefix("fused_") if use_multistep else provider
    mode = "fused" if use_multistep else "default"
    print(f"[bench] start forward provider={provider} T={T} N={N} mode={mode}")
    model, x_seq, _ = build_model(base_provider, T, N, require_grad=False)

    if base_provider == "torch_compile":
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
        if use_multistep:

            def fn():
                with torch.no_grad():
                    reset_net_state(model)
                    with environ.context(dt=_DT):
                        _run_dense_multistep(model, x_seq)
        else:

            def fn():
                with torch.no_grad():
                    run_model(model, x_seq)

    ms = _bench_ms(fn, grads=None, use_quantiles=False)
    print(f"[bench] forward provider={provider} T={T} N={N} ms={ms}")
    return ms


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

    use_multistep = provider.startswith("fused_")
    base_provider = provider.removeprefix("fused_") if use_multistep else provider
    mode = "fused" if use_multistep else "default"
    print(f"[bench] start fwd+bwd provider={provider} T={T} N={N} mode={mode}")
    model, x_seq, grads = build_model(base_provider, T, N, require_grad=True)

    if base_provider == "torch_compile":
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
            spike_seq_inner = run_model(model, x_seq)
            spike_seq_inner.sum().backward()

    ms = _bench_ms(fn, grads=grads, use_quantiles=False)
    print(f"[bench] fwd+bwd provider={provider} T={T} N={N} ms={ms}")
    return ms


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
