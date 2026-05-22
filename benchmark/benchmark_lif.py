import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from btorch.backend.triton.lif import TritonMultiStepLIF
from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.neurons.lif import LIF
from btorch.utils.file import fig_path


def _benchmark_fn(factory, *, warmup: int = 10, repeat: int = 30) -> float:
    for _ in range(warmup):
        fn = factory()
        fn()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn = factory()
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat


def _benchmark_reuse_fn(fn, *, warmup: int = 10, repeat: int = 30) -> float:
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat


def _build_input(
    steps: int,
    batch_size: int,
    n_neuron: int,
    *,
    device: str,
    dtype: torch.dtype,
    seed: int = 0,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x_seq = torch.randn(
        (steps, batch_size, n_neuron),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return 0.2 * x_seq + 0.35


def _build_torch_lif(
    n_neuron: int,
    *,
    tau_ref: float | None,
    device: str,
    dtype: torch.dtype,
) -> LIF:
    return LIF(
        n_neuron=n_neuron,
        v_threshold=1.0,
        v_reset=0.0,
        c_m=1.0,
        tau=20.0,
        tau_ref=tau_ref,
        hard_reset=False,
        step_mode="m",
        device=device,
        dtype=dtype,
    )


def _build_triton_lif(
    n_neuron: int,
    *,
    tau_ref: float | None,
    device: str,
    dtype: torch.dtype,
) -> TritonMultiStepLIF:
    return TritonMultiStepLIF(
        n_neuron=n_neuron,
        v_threshold=1.0,
        v_reset=0.0,
        c_m=1.0,
        tau=20.0,
        tau_ref=tau_ref,
        hard_reset=False,
        device=torch.device(device),
        dtype=dtype,
    )


class _CompiledLIFWrapper(nn.Module):
    def __init__(self, lif: LIF, dt: float):
        super().__init__()
        self.lif = lif
        self.dt = dt

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        with environ.context(dt=self.dt):
            return self.lif(x_seq)


def _make_torch_runner(
    x_seq: torch.Tensor,
    *,
    tau_ref: float | None,
    device: str,
    dtype: torch.dtype,
):
    n_neuron = x_seq.shape[-1]
    batch_size = x_seq.shape[1]
    neuron = _build_torch_lif(
        int(n_neuron),
        tau_ref=tau_ref,
        device=device,
        dtype=dtype,
    )
    init_net_state(
        neuron,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )

    def run():
        with torch.no_grad(), environ.context(dt=1.0):
            neuron(x_seq)

    return run


def _make_compile_runner(
    x_seq: torch.Tensor,
    *,
    tau_ref: float | None,
    device: str,
    dtype: torch.dtype,
):
    if not hasattr(torch, "compile"):
        return None

    n_neuron = x_seq.shape[-1]
    batch_size = x_seq.shape[1]
    lif = _build_torch_lif(
        int(n_neuron),
        tau_ref=tau_ref,
        device=device,
        dtype=dtype,
    )
    init_net_state(
        lif,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    compiled = torch.compile(_CompiledLIFWrapper(lif, dt=1.0))

    with torch.no_grad():
        compiled(x_seq)

    def run():
        init_net_state(
            compiled,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            compiled(x_seq)

    return run


def _make_triton_runner(
    x_seq: torch.Tensor,
    *,
    tau_ref: float | None,
    device: str,
    dtype: torch.dtype,
):
    n_neuron = x_seq.shape[-1]
    batch_size = x_seq.shape[1]
    neuron = _build_triton_lif(
        int(n_neuron),
        tau_ref=tau_ref,
        device=device,
        dtype=dtype,
    )
    neuron.reset_state(batch_size=batch_size)

    def run():
        with torch.no_grad():
            neuron(x_seq, dt=1.0)

    return run


def benchmark_lif(
    *,
    steps: int = 128,
    batch_size: int = 64,
    tau_ref: float | None = 2.0,
    tolerance_ratio: float = 1e-6,
    dtype: torch.dtype = torch.float32,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available, skipping benchmark")
        return

    sizes = np.logspace(3, 5, 20, dtype=int)
    results = {
        "torch_eager_ms": [],
        "torch_compile_ms": [],
        "triton_ms": [],
        "speedup_eager": [],
        "speedup_compile": [],
        "triton_error_rate": [],
        "compile_error_rate": [],
    }
    valid_sizes = []

    for n_neuron in sizes:
        print(
            f"Benchmarking LIF with T={steps}, B={batch_size}, N={n_neuron}, "
            f"tau_ref={tau_ref}..."
        )
        try:
            x_seq = _build_input(
                steps,
                batch_size,
                int(n_neuron),
                device=device,
                dtype=dtype,
            )

            torch_check = _build_torch_lif(
                int(n_neuron),
                tau_ref=tau_ref,
                device=device,
                dtype=dtype,
            )
            init_net_state(
                torch_check,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad(), environ.context(dt=1.0):
                y_torch = torch_check(x_seq)
            compile_error_rate = float("nan")
            compile_check_runner = _make_compile_runner(
                x_seq,
                tau_ref=tau_ref,
                device=device,
                dtype=dtype,
            )
            if compile_check_runner is not None:
                compile_check_lif = _build_torch_lif(
                    int(n_neuron),
                    tau_ref=tau_ref,
                    device=device,
                    dtype=dtype,
                )
                init_net_state(
                    compile_check_lif,
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                )
                compiled_check = torch.compile(
                    _CompiledLIFWrapper(compile_check_lif, dt=1.0)
                )
                with torch.no_grad():
                    y_compile = compiled_check(x_seq)
                compile_mismatch = y_compile != y_torch
                compile_error_rate = (
                    int(compile_mismatch.sum().item()) / compile_mismatch.numel()
                )
            triton_check = _build_triton_lif(
                int(n_neuron),
                tau_ref=tau_ref,
                device=device,
                dtype=dtype,
            )
            triton_check.reset_state(batch_size=batch_size)
            with torch.no_grad():
                y_triton = triton_check(x_seq, dt=1.0)

            mismatch = y_triton != y_torch
            mismatch_count = int(mismatch.sum().item())
            total_count = mismatch.numel()
            error_rate = mismatch_count / total_count
            if error_rate > tolerance_ratio:
                print(
                    f"Warning: error rate {error_rate:.3e} exceeds tolerance "
                    f"{tolerance_ratio:.3e} at N={int(n_neuron)}"
                )
            if not np.isnan(compile_error_rate) and compile_error_rate > tolerance_ratio:
                print(
                    f"Warning: compiled error rate {compile_error_rate:.3e} exceeds "
                    f"tolerance {tolerance_ratio:.3e} at N={int(n_neuron)}"
                )

            torch_eager_time = _benchmark_fn(
                lambda: _make_torch_runner(
                    x_seq, tau_ref=tau_ref, device=device, dtype=dtype
                )
            )
            compile_runner = _make_compile_runner(
                x_seq,
                tau_ref=tau_ref,
                device=device,
                dtype=dtype,
            )
            if compile_runner is not None:
                torch_compile_time = _benchmark_reuse_fn(compile_runner)
            else:
                torch_compile_time = float("nan")
            triton_time = _benchmark_fn(
                lambda: _make_triton_runner(
                    x_seq, tau_ref=tau_ref, device=device, dtype=dtype
                )
            )

            results["torch_eager_ms"].append(torch_eager_time * 1000)
            results["torch_compile_ms"].append(torch_compile_time * 1000)
            results["triton_ms"].append(triton_time * 1000)
            results["speedup_eager"].append(torch_eager_time / triton_time)
            results["speedup_compile"].append(
                torch_compile_time / triton_time
                if not np.isnan(torch_compile_time)
                else float("nan")
            )
            results["triton_error_rate"].append(error_rate)
            results["compile_error_rate"].append(compile_error_rate)
            valid_sizes.append(int(n_neuron))

        except Exception as exc:
            print(f"OOM or error at N={n_neuron}: {exc}")
            break

    if not valid_sizes:
        print("No valid benchmark results collected.")
        return

    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.plot(
        valid_sizes,
        results["torch_eager_ms"],
        color="#2ca02c",
        marker="^",
        linewidth=2,
        label="Torch LIF Eager",
    )
    ax.plot(
        valid_sizes,
        results["torch_compile_ms"],
        color="#9467bd",
        marker="x",
        linewidth=2,
        label="Torch LIF Compile",
    )
    ax.plot(
        valid_sizes,
        results["triton_ms"],
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label="Triton LIF",
    )
    ax.set_title("LIF Forward Latency", fontsize=12, fontweight="bold")
    ax.set_xlabel("Neuron Count N", fontsize=10)
    ax.set_ylabel("Time (ms)", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)

    ax = axes[1]
    ax.plot(
        valid_sizes,
        results["speedup_eager"],
        color="#ff7f0e",
        marker="D",
        linewidth=2,
        label="Eager / Triton",
    )
    ax.plot(
        valid_sizes,
        results["speedup_compile"],
        color="#8c564b",
        marker="s",
        linewidth=2,
        label="Compile / Triton",
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title("LIF Triton Speedup", fontsize=12, fontweight="bold")
    ax.set_xlabel("Neuron Count N", fontsize=10)
    ax.set_ylabel("Speedup (x)", fontsize=10)
    ax.set_xscale("log")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)

    ax = axes[2]
    ax.plot(
        valid_sizes,
        results["triton_error_rate"],
        color="#d62728",
        marker="s",
        linewidth=2,
        label="Triton vs Eager",
    )
    ax.plot(
        valid_sizes,
        results["compile_error_rate"],
        color="#7f7f7f",
        marker="x",
        linewidth=2,
        label="Compile vs Eager",
    )
    ax.axhline(
        tolerance_ratio,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"Tolerance = {tolerance_ratio:.1e}",
    )
    ax.set_title("LIF Error Rate", fontsize=12, fontweight="bold")
    ax.set_xlabel("Neuron Count N", fontsize=10)
    ax.set_ylabel("Error Rate", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)

    fig.suptitle(
        (
            f"Multi-step LIF Benchmark (T={steps}, B={batch_size}, "
            f"tau_ref={tau_ref}, tol={tolerance_ratio:.1e})"
        ),
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.98,
        f"Simulation steps: {steps}",
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#fff3cd",
            "edgecolor": "#b58900",
            "alpha": 0.95,
        },
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    output_dir = fig_path(__file__)
    output_path = output_dir / "lif_benchmark.png"
    plt.savefig(output_path, dpi=300)
    print(f"Benchmark plot saved to {output_path}")

    print("\nBenchmark summary:")
    for (
        n_neuron,
        torch_eager_ms,
        torch_compile_ms,
        triton_ms,
        speedup_eager,
        speedup_compile,
        triton_error_rate,
        compile_error_rate,
    ) in zip(
        valid_sizes,
        results["torch_eager_ms"],
        results["torch_compile_ms"],
        results["triton_ms"],
        results["speedup_eager"],
        results["speedup_compile"],
        results["triton_error_rate"],
        results["compile_error_rate"],
    ):
        print(
            f"N={n_neuron:>6d} | "
            f"eager={torch_eager_ms:>8.3f} ms | "
            f"compile={torch_compile_ms:>8.3f} ms | "
            f"triton={triton_ms:>8.3f} ms | "
            f"eager/triton={speedup_eager:>6.2f}x | "
            f"compile/triton={speedup_compile:>6.2f}x | "
            f"triton_err={triton_error_rate:.3e} | "
            f"compile_err={compile_error_rate:.3e}"
        )

def test_precision_scale():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available, skipping precision scale test")
        return []

    steps = 128
    batch_size = 64
    tau_ref = 2.0
    dtype = torch.float32
    errors = []
    #sizes = np.unique(np.logspace(3, 5, 20, dtype=int))
    sizes = np.unique(np.logspace(3, 4.3, 15, dtype=int))

    for n_neuron in sizes:
        print(
            f"Testing precision scale with T={steps}, B={batch_size}, "
            f"N={n_neuron}, tau_ref={tau_ref}..."
        )
        try:
            x_seq = _build_input(
                steps,
                batch_size,
                int(n_neuron),
                device=device,
                dtype=dtype,
            )

            torch_lif = _build_torch_lif(
                int(n_neuron),
                tau_ref=tau_ref,
                device=device,
                dtype=dtype,
            )
            init_net_state(
                torch_lif,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

            triton_lif = _build_triton_lif(
                int(n_neuron),
                tau_ref=tau_ref,
                device=device,
                dtype=dtype,
            )
            triton_lif.reset_state(batch_size=batch_size)

            with torch.no_grad(), environ.context(dt=1.0):
                y_torch = torch_lif(x_seq)
            with torch.no_grad():
                y_triton = triton_lif(x_seq, dt=1.0)

            diff = (y_triton - y_torch).abs()
            mismatch = y_triton != y_torch
            mismatch_count = int(mismatch.sum().item())
            total_count = mismatch.numel()
            mismatch_ratio = mismatch_count / total_count
            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())

            v_diff = (triton_lif.v - torch_lif.v).abs()
            max_v_diff = float(v_diff.max().item())
            mean_v_diff = float(v_diff.mean().item())

            errors.append(
                {
                    "n_neuron": int(n_neuron),
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                    "mismatch_count": mismatch_count,
                    "mismatch_ratio": mismatch_ratio,
                    "max_v_diff": max_v_diff,
                    "mean_v_diff": mean_v_diff,
                }
            )

            print(
                f"N={int(n_neuron):>6d} | "
                f"spike_mismatch={mismatch_count:>8d}/{total_count} "
                f"({mismatch_ratio:.3e}) | "
                f"max_diff={max_diff:.2e} | "
                f"mean_diff={mean_diff:.2e} | "
                f"max_v_diff={max_v_diff:.2e}"
            )

        except Exception as exc:
            print(f"Precision scale error at N={n_neuron}: {exc}")
            errors.append(
                {
                    "n_neuron": int(n_neuron),
                    "error": str(exc),
                }
            )

    valid_errors = [item for item in errors if "error" not in item]
    if not valid_errors:
        print("No valid precision-scale results collected.")
        return errors

    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = [item["n_neuron"] for item in valid_errors]

    ax = axes[0]
    ax.plot(
        x,
        [item["mismatch_ratio"] for item in valid_errors],
        color="#d62728",
        marker="o",
        linewidth=2,
        label="Spike mismatch ratio",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Neuron Count N", fontsize=10)
    ax.set_ylabel("Mismatch Ratio", fontsize=10)
    ax.set_title("Spike Mismatch vs N", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(frameon=True, fontsize=9)

    ax = axes[1]
    ax.plot(
        x,
        [item["max_diff"] for item in valid_errors],
        color="#ff7f0e",
        marker="s",
        linewidth=2,
        label="Max spike abs diff",
    )
    ax.plot(
        x,
        [item["mean_diff"] for item in valid_errors],
        color="#1f77b4",
        marker="^",
        linewidth=2,
        label="Mean spike abs diff",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Neuron Count N", fontsize=10)
    ax.set_ylabel("Spike Error", fontsize=10)
    ax.set_title("Spike Error Scale", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(frameon=True, fontsize=9)

    ax = axes[2]
    ax.plot(
        x,
        [item["max_v_diff"] for item in valid_errors],
        color="#2ca02c",
        marker="D",
        linewidth=2,
        label="Max voltage abs diff",
    )
    ax.plot(
        x,
        [item["mean_v_diff"] for item in valid_errors],
        color="#9467bd",
        marker="x",
        linewidth=2,
        label="Mean voltage abs diff",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Neuron Count N", fontsize=10)
    ax.set_ylabel("Voltage Error", fontsize=10)
    ax.set_title("Voltage Error Scale", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(frameon=True, fontsize=9)

    fig.suptitle(
        f"LIF Precision Scale Test (T={steps}, B={batch_size}, tau_ref={tau_ref})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    output_dir = fig_path(__file__)
    output_path = output_dir / "lif_precision_scale.png"
    plt.savefig(output_path, dpi=300)
    print(f"Precision scale plot saved to {output_path}")

    return errors
if __name__ == "__main__":
    benchmark_lif()
   #test_precision_scale()
