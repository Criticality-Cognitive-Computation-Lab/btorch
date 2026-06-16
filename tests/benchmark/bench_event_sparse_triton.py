import argparse
from dataclasses import asdict

import matplotlib.pyplot as plt
import torch

from btorch.utils.file import fig_path
from tests.benchmark.event_sparse_bench_utils import (
    EventSparseBenchConfig,
    build_event_sparse_case,
    build_provider_modules,
)
from tests.utils.bench import do_bench


def _provider_order(include_compiled: bool) -> list[str]:
    providers = [
        "torch_dense",
        "torch_sparse",
        "spike_list_build",
        "triton_pre_span_list",
        "triton_post_span_list",
    ]
    if include_compiled:
        providers.extend(
            [
                "torch_dense_compile",
                "torch_sparse_compile",
                "spike_list_build_compile",
                "triton_pre_span_list_compile",
                "triton_post_span_list_compile",
            ]
        )
    return providers


def _provider_args(provider: str, case):
    if "span_list" in provider:
        return case.spike_count, case.spike_ind
    return (case.spike,)


def _benchmark_provider(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    *,
    warmup_ms: int,
    rep_ms: int,
    timing_method: str,
) -> float:
    module.eval()
    with torch.no_grad():
        _ = module(*args)
        if args[0].is_cuda:
            torch.cuda.synchronize()

        def fn():
            module(*args)

        return do_bench(
            fn,
            warmup=warmup_ms,
            rep=rep_ms,
            timing_method=timing_method,
            return_mode="mean",
        )


def _benchmark_config(
    cfg: EventSparseBenchConfig,
    *,
    include_compiled: bool,
    warmup_ms: int,
    rep_ms: int,
    timing_method: str,
) -> dict[str, float]:
    case = build_event_sparse_case(cfg)
    modules = build_provider_modules(
        case,
        include_compiled=include_compiled,
    )
    results = {}
    for provider in _provider_order(include_compiled):
        results[provider] = _benchmark_provider(
            modules[provider],
            _provider_args(provider, case),
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
            timing_method=timing_method,
        )
    return results


def _plot_results(
    x_vals: list[float],
    results: dict[str, list[float]],
    *,
    x_label: str,
    title: str,
    output_name: str,
) -> None:
    styles = {
        "torch_dense": ("black", "-"),
        "torch_sparse": ("tab:red", "-"),
        "spike_list_build": ("tab:green", "-"),
        "triton_pre_span_list": ("tab:blue", ":"),
        "triton_post_span_list": ("tab:orange", ":"),
        "torch_dense_compile": ("black", "--"),
        "torch_sparse_compile": ("tab:red", "--"),
        "spike_list_build_compile": ("tab:green", "--"),
        "triton_pre_span_list_compile": ("tab:blue", "-."),
        "triton_post_span_list_compile": ("tab:orange", "-."),
    }

    baseline = results.get("torch_dense")
    if baseline is None:
        raise ValueError("results must contain 'torch_dense' to compute speedup.")

    fig, (ax_latency, ax_speedup) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    for provider, times in results.items():
        color, linestyle = styles.get(provider, ("gray", "-"))
        ax_latency.plot(
            x_vals,
            times,
            label=provider,
            color=color,
            linestyle=linestyle,
            marker="o",
        )
        speedup = [
            base_time / provider_time if provider_time > 0 else float("nan")
            for base_time, provider_time in zip(baseline, times, strict=True)
        ]
        ax_speedup.plot(
            x_vals,
            speedup,
            label=provider,
            color=color,
            linestyle=linestyle,
            marker="o",
        )

    ax_latency.set_ylabel("Latency (ms)")
    ax_latency.set_title(title)
    ax_latency.grid(True, alpha=0.25)
    ax_latency.legend(loc="best")

    ax_speedup.axhline(1.0, color="black", linewidth=1.0, alpha=0.35)
    ax_speedup.set_xlabel(x_label)
    ax_speedup.set_ylabel("Speedup vs torch_dense (x)")
    ax_speedup.grid(True, alpha=0.25)

    fig.tight_layout()
    out_dir = fig_path(__file__)
    fig.savefig((out_dir / f"{output_name}.png").as_posix(), dpi=160)
    plt.close(fig)


def benchmark_vs_active_ratio(
    *,
    batch_size: int,
    n_pre: int,
    n_post: int,
    fanout: int,
    active_ratios: list[float],
    include_compiled: bool,
    warmup_ms: int,
    rep_ms: int,
    timing_method: str,
    seed: int,
) -> dict[str, list[float]]:
    results = {provider: [] for provider in _provider_order(include_compiled)}
    for idx, active_ratio in enumerate(active_ratios):
        cfg = EventSparseBenchConfig(
            batch_size=batch_size,
            n_pre=n_pre,
            n_post=n_post,
            fanout=fanout,
            active_ratio=active_ratio,
            seed=seed + idx,
        )
        config_results = _benchmark_config(
            cfg,
            include_compiled=include_compiled,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
            timing_method=timing_method,
        )
        for provider, value in config_results.items():
            results[provider].append(value)
        print(
            "active_ratio="
            f"{active_ratio:.3f} "
            + " ".join(
                f"{provider}={config_results[provider]:.4f}ms"
                for provider in _provider_order(include_compiled)
            )
        )
    return results


def benchmark_vs_fanout(
    *,
    batch_size: int,
    n_pre: int,
    n_post: int,
    fanouts: list[int],
    active_ratio: float,
    include_compiled: bool,
    warmup_ms: int,
    rep_ms: int,
    timing_method: str,
    seed: int,
) -> dict[str, list[float]]:
    results = {provider: [] for provider in _provider_order(include_compiled)}
    for idx, fanout in enumerate(fanouts):
        cfg = EventSparseBenchConfig(
            batch_size=batch_size,
            n_pre=n_pre,
            n_post=n_post,
            fanout=fanout,
            active_ratio=active_ratio,
            seed=seed + idx,
        )
        config_results = _benchmark_config(
            cfg,
            include_compiled=include_compiled,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
            timing_method=timing_method,
        )
        for provider, value in config_results.items():
            results[provider].append(value)
        print(
            f"fanout={fanout} "
            + " ".join(
                f"{provider}={config_results[provider]:.4f}ms"
                for provider in _provider_order(include_compiled)
            )
        )
    return results


def _parse_float_list(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def _parse_int_list(values: str) -> list[int]:
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Triton event sparse kernels against naive dense PyTorch "
            "and torch.compile wrappers."
        )
    )
    parser.add_argument(
        "--sweep",
        choices=["active_ratio", "fanout"],
        default="active_ratio",
        help="Which parameter to sweep on the x-axis.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-pre", type=int, default=4096)
    parser.add_argument("--n-post", type=int, default=4096)
    parser.add_argument("--fanout", type=int, default=32)
    parser.add_argument("--active-ratio", type=float, default=0.05)
    parser.add_argument(
        "--active-ratios",
        type=str,
        default="0.01,0.02,0.05,0.1,0.2,0.5,1.0",
        help="Comma-separated active ratio sweep values.",
    )
    parser.add_argument(
        "--fanouts",
        type=str,
        default="4,8,16,32,64,128",
        help="Comma-separated fanout sweep values.",
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    parser.add_argument(
        "--timing-method",
        choices=["gpu", "total"],
        default="gpu",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Do not include torch.compile providers.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    include_compiled = not args.skip_compile
    print("Benchmark config:")
    print(
        asdict(
            EventSparseBenchConfig(
                batch_size=args.batch_size,
                n_pre=args.n_pre,
                n_post=args.n_post,
                fanout=args.fanout,
                active_ratio=args.active_ratio,
                seed=args.seed,
            )
        )
    )

    if args.sweep == "active_ratio":
        x_vals = _parse_float_list(args.active_ratios)
        results = benchmark_vs_active_ratio(
            batch_size=args.batch_size,
            n_pre=args.n_pre,
            n_post=args.n_post,
            fanout=args.fanout,
            active_ratios=x_vals,
            include_compiled=include_compiled,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
            timing_method=args.timing_method,
            seed=args.seed,
        )
        _plot_results(
            x_vals,
            results,
            x_label="Active Ratio",
            title=(
                "Event Sparse Kernel Benchmark vs Active Ratio "
                f"(batch={args.batch_size}, n_pre={args.n_pre}, "
                f"n_post={args.n_post}, fanout={args.fanout})"
            ),
            output_name="event_sparse_vs_active_ratio",
        )
    else:
        x_vals = _parse_int_list(args.fanouts)
        results = benchmark_vs_fanout(
            batch_size=args.batch_size,
            n_pre=args.n_pre,
            n_post=args.n_post,
            fanouts=x_vals,
            active_ratio=args.active_ratio,
            include_compiled=include_compiled,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
            timing_method=args.timing_method,
            seed=args.seed,
        )
        _plot_results(
            x_vals,
            results,
            x_label="Fanout / Row Stride",
            title=(
                "Event Sparse Kernel Benchmark vs Fanout "
                f"(batch={args.batch_size}, n_pre={args.n_pre}, "
                f"n_post={args.n_post}, active_ratio={args.active_ratio})"
            ),
            output_name="event_sparse_vs_fanout",
        )


if __name__ == "__main__":
    main()
