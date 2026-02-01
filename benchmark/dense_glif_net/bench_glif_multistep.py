import os
from collections.abc import Callable

import torch
from triton.testing import Benchmark, do_bench, perf_report

from benchmark.dense_glif_net.glif_common import _DT, make_inputs, providers
from btorch.models import environ
from btorch.models.functional import init_net_state, reset_net_state
from btorch.models.neurons.glif import GLIF3
from btorch.models.rnn import make_rnn
from btorch.utils.file import fig_path


_MAX_N = 4096 * 4
_MAX_T = 2048


def _unique_sorted(values: list[int]) -> list[int]:
    return sorted(set(values))


_DEBUG = os.getenv("BTORCH_BENCH_DEBUG", "0") == "1"
if _DEBUG:
    _T_SWEEP = [8, 16]
    _N_SWEEP = [64, 128]
else:
    _T_SWEEP = _unique_sorted(
        [int(v) for v in torch.logspace(6, 11, 16, base=2) if int(v) <= _MAX_T]
    )
    _N_SWEEP = _unique_sorted(
        [int(v) for v in torch.logspace(2, 4, 16, base=10) if int(v) <= _MAX_N]
    )

_PROVIDERS = providers()
# if "torch_compile" in _PROVIDERS:
#     _PROVIDERS.append("torch_compile_unrolled")
_LINE_NAMES = {
    "torch_eager": "Torch Eager",
    "torch_compile": "Torch Compile",
    # "torch_compile_unrolled": "Torch Compile (Unrolled)",
    "triton": "Triton",
    "warp": "Warp",
    "cupy": "CuPy",
}
_STYLE_MAP = {
    "torch_eager": ("black", "-"),
    "torch_compile": ("gray", "-"),
    # "torch_compile_unrolled": ("dimgray", "--"),
    "triton": ("red", "-"),
    "warp": ("blue", "-"),
    "cupy": ("green", "-"),
}
_STYLES = [_STYLE_MAP[p] for p in _PROVIDERS]
_AUTO_SKIP_MS = 300.0
_AUTO_SKIP_CACHE: dict[tuple[str, str], float] = {}
_AUTO_SKIP_SWEEP: str | None = None


def _bench_ms(fn: Callable, grads: list[torch.Tensor] | None, *, use_quantiles: bool):
    quantiles = [0.5, 0.2, 0.8] if use_quantiles else None
    ms = do_bench(
        fn,
        warmup=1,
        rep=2,
        quantiles=quantiles,
        grad_to_none=grads,
    )
    if use_quantiles:
        return tuple(ms)
    return (ms, None, None)


def _run_multistep_torch(
    params: dict, x_seq: torch.Tensor, dt: float, hard_reset: bool
) -> torch.Tensor:
    B = x_seq.shape[1]
    glif = GLIF3(
        n_neuron=B,
        v_threshold=params["v_th"],
        v_reset=params["v_reset"],
        v_rest=params["v_rest"],
        c_m=params["c_m"],
        tau=params["tau"],
        k=params["k"],
        asc_amps=params["asc_amps"],
        tau_ref=0.0,
        hard_reset=hard_reset,
        trainable_param=set(),
        device=x_seq.device,
        dtype=x_seq.dtype,
    )
    glif.v = params["v_reset"].clone()
    glif.Iasc = torch.zeros_like(params["k"])
    glif.refractory = torch.zeros_like(params["v_reset"])
    with environ.context(dt=dt):
        return glif.multi_step_forward(x_seq)


def _run_multistep_kernel(
    provider: str,
    x_seq: torch.Tensor,
    params: dict,
    dt: float,
    hard_reset: bool,
) -> torch.Tensor:
    if provider == "triton":
        from benchmark.dense_glif_net.glif_triton import glif3_step_triton as step_fn
    elif provider == "warp":
        from benchmark.dense_glif_net.glif_warp import glif3_step_warp as step_fn
    elif provider == "cupy":
        from benchmark.dense_glif_net.glif_cupy import glif3_step_cupy as step_fn
    else:
        raise ValueError(f"Unknown provider: {provider}")

    B, M = params["k"].shape
    v = params["v_reset"].clone()
    Iasc = torch.zeros_like(params["k"])
    not_refrac = torch.ones_like(params["v_reset"])
    s_seq, _, _, _ = step_fn.multistep_fused(
        x_seq=x_seq,
        v=v,
        Iasc=Iasc,
        params={
            "v_th": params["v_th"],
            "v_reset": params["v_reset"],
            "v_rest": params["v_rest"],
            "c_m": params["c_m"],
            "tau": params["tau"],
            "k": params["k"].reshape(-1),
            "asc_amps": params["asc_amps"].reshape(-1),
        },
        not_refrac=not_refrac,
        dt=dt,
        M=M,
        hard_reset=hard_reset,
        alpha=2.0,
    )
    return s_seq


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
            plot_name="glif_multistep_forward_vs_T",
            args={"N": 256, "sweep": "T"},
        ),
        Benchmark(
            x_names=["N"],
            x_vals=_N_SWEEP,
            line_arg="provider",
            line_vals=_PROVIDERS,
            line_names=[_LINE_NAMES[p] for p in _PROVIDERS],
            styles=_STYLES[: len(_PROVIDERS)],
            ylabel="ms",
            plot_name="glif_multistep_forward_vs_n_neuron",
            args={"T": 100, "sweep": "N"},
        ),
    ]
)
def bench_glif_multistep_forward(T: int, N: int, provider: str, sweep: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GLIF benchmarks.")

    device = torch.device("cuda")
    _, _, x_seq, params = make_inputs(T, N, device, require_grad=False)

    global _AUTO_SKIP_SWEEP
    if sweep not in {"T", "N"}:
        raise ValueError("sweep must be 'T' or 'N'.")
    if _AUTO_SKIP_SWEEP != sweep:
        _AUTO_SKIP_CACHE.clear()
        _AUTO_SKIP_SWEEP = sweep
    skip_key = (provider, f"forward_{sweep}")
    last_ms = _AUTO_SKIP_CACHE.get(skip_key)
    if last_ms is not None and last_ms > _AUTO_SKIP_MS:
        print(
            f"[bench] auto-skip provider={provider} T={T} N={N} "
            f"(last {last_ms:.1f} ms)"
        )
        return (float("nan"), None, None)

    if provider in {"torch_eager", "torch_compile", "torch_compile_unrolled"}:
        glif = GLIF3(
            n_neuron=N,
            v_threshold=params["v_th"],
            v_reset=params["v_reset"],
            v_rest=params["v_rest"],
            c_m=params["c_m"],
            tau=params["tau"],
            k=params["k"],
            asc_amps=params["asc_amps"],
            tau_ref=0.0,
            hard_reset=False,
            trainable_param=set(),
            device=device,
            dtype=x_seq.dtype,
            step_mode="m",
        )
        init_net_state(glif, device=device, dtype=x_seq.dtype)
        if provider == "torch_compile":
            glif = torch.compile(glif)

            def fn():
                with torch.no_grad():
                    reset_net_state(glif)
                    with environ.context(dt=_DT):
                        glif(x_seq)

        elif provider == "torch_compile_unrolled":
            wrapper = make_rnn(glif)
            init_net_state(wrapper, device=device, dtype=x_seq.dtype)
            wrapper = torch.compile(wrapper)

            def fn():
                with torch.no_grad():
                    reset_net_state(wrapper)
                    with environ.context(dt=_DT):
                        wrapper.multi_step_forward(x_seq)

        else:

            def fn():
                with torch.no_grad():
                    reset_net_state(glif)
                    with environ.context(dt=_DT):
                        glif.multi_step_forward(x_seq)

    else:

        def fn():
            with torch.no_grad():
                _run_multistep_kernel(provider, x_seq, params, dt=_DT, hard_reset=False)

    ms = _bench_ms(fn, grads=None, use_quantiles=False)
    _AUTO_SKIP_CACHE[skip_key] = float(ms[0])
    print(f"[bench] multistep provider={provider} T={T} N={N} ms={ms}")
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
            plot_name="glif_multistep_forward_backward_vs_T",
            args={"N": 256, "sweep": "T"},
        ),
        Benchmark(
            x_names=["N"],
            x_vals=_N_SWEEP,
            line_arg="provider",
            line_vals=_PROVIDERS,
            line_names=[_LINE_NAMES[p] for p in _PROVIDERS],
            styles=_STYLES[: len(_PROVIDERS)],
            ylabel="ms",
            plot_name="glif_multistep_forward_backward_vs_n_neuron",
            args={"T": 100, "sweep": "N"},
        ),
    ]
)
def bench_glif_multistep_forward_backward(T: int, N: int, provider: str, sweep: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GLIF benchmarks.")

    global _AUTO_SKIP_SWEEP
    if sweep not in {"T", "N"}:
        raise ValueError("sweep must be 'T' or 'N'.")
    if _AUTO_SKIP_SWEEP != sweep:
        _AUTO_SKIP_CACHE.clear()
        _AUTO_SKIP_SWEEP = sweep
    skip_key = (provider, f"fwd_bwd_{sweep}")
    last_ms = _AUTO_SKIP_CACHE.get(skip_key)
    if last_ms is not None and last_ms > _AUTO_SKIP_MS:
        print(
            f"[bench] auto-skip fwd+bwd provider={provider} T={T} N={N} "
            f"(last {last_ms:.1f} ms)"
        )
        return (float("nan"), None, None)

    device = torch.device("cuda")
    _, _, x_seq, params = make_inputs(T, N, device, require_grad=True)
    params["asc_amps"].requires_grad_(True)

    if provider in {"torch_eager", "torch_compile", "torch_compile_unrolled"}:
        glif = GLIF3(
            n_neuron=N,
            v_threshold=params["v_th"],
            v_reset=params["v_reset"],
            v_rest=params["v_rest"],
            c_m=params["c_m"],
            tau=params["tau"],
            k=params["k"],
            asc_amps=params["asc_amps"],
            tau_ref=0.0,
            hard_reset=False,
            trainable_param={"asc_amps"},
            device=device,
            dtype=x_seq.dtype,
            step_mode="m",
        )
        init_net_state(glif, device=device, dtype=x_seq.dtype)
        if provider == "torch_compile":
            glif = torch.compile(glif)

            def fn():
                reset_net_state(glif)
                with environ.context(dt=_DT):
                    spike_seq = glif(x_seq)
                spike_seq.sum().backward()

        elif provider == "torch_compile_unrolled":
            wrapper = make_rnn(glif)
            init_net_state(wrapper, device=device, dtype=x_seq.dtype)
            wrapper = torch.compile(wrapper)

            def fn():
                reset_net_state(wrapper)
                with environ.context(dt=_DT):
                    spike_seq = wrapper.multi_step_forward(x_seq)
                spike_seq.sum().backward()

        else:

            def fn():
                reset_net_state(glif)
                with environ.context(dt=_DT):
                    spike_seq = glif.multi_step_forward(x_seq)
                spike_seq.sum().backward()

    else:

        def fn():
            s_seq = _run_multistep_kernel(
                provider, x_seq, params, dt=_DT, hard_reset=False
            )
            s_seq.sum().backward()

    ms = _bench_ms(fn, grads=[x_seq, params["asc_amps"]], use_quantiles=False)
    _AUTO_SKIP_CACHE[skip_key] = float(ms[0])
    print(f"[bench] multistep fwd+bwd provider={provider} T={T} N={N} ms={ms}")
    return ms


if __name__ == "__main__":
    bench_glif_multistep_forward.run(
        show_plots=False,
        print_data=False,
        return_df=False,
        save_path=fig_path(__file__),
    )
    bench_glif_multistep_forward_backward.run(
        show_plots=False,
        print_data=False,
        return_df=False,
        save_path=fig_path(__file__),
    )
