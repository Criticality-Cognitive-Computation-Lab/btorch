import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _build_target(
    batch_size: int,
    head_dim: int,
    *,
    device: str,
    dtype: torch.dtype,
    seed: int = 1,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        (batch_size, head_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def _build_head(
    n_neuron: int,
    head_dim: int,
    *,
    device: str,
    dtype: torch.dtype,
    seed: int = 2,
) -> torch.nn.Linear:
    torch.manual_seed(seed)
    return torch.nn.Linear(
        n_neuron,
        head_dim,
        bias=False,
        device=device,
        dtype=dtype,
    )


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
    target: torch.Tensor,
    *,
    tau_ref: float | None,
    head_dim: int,
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
    head = _build_head(
        int(n_neuron),
        head_dim,
        device=device,
        dtype=dtype,
    )
    optimizer = torch.optim.SGD(head.parameters(), lr=1e-3)

    def run():
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), environ.context(dt=1.0):
            spikes = neuron(x_seq)
        features = spikes.detach().mean(dim=0).requires_grad_(True)
        logits = head(features)
        loss = F.mse_loss(logits, target)
        loss.backward()
        optimizer.step()

    return run


def _make_compile_runner(
    x_seq: torch.Tensor,
    target: torch.Tensor,
    *,
    tau_ref: float | None,
    head_dim: int,
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
    head = _build_head(
        int(n_neuron),
        head_dim,
        device=device,
        dtype=dtype,
    )
    optimizer = torch.optim.SGD(head.parameters(), lr=1e-3)

    with torch.no_grad():
        compiled(x_seq)

    def run():
        init_net_state(
            compiled,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            spikes = compiled(x_seq)
        features = spikes.detach().mean(dim=0).requires_grad_(True)
        logits = head(features)
        loss = F.mse_loss(logits, target)
        loss.backward()
        optimizer.step()

    return run


def _make_triton_runner(
    x_seq: torch.Tensor,
    target: torch.Tensor,
    *,
    tau_ref: float | None,
    head_dim: int,
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
    head = _build_head(
        int(n_neuron),
        head_dim,
        device=device,
        dtype=dtype,
    )
    optimizer = torch.optim.SGD(head.parameters(), lr=1e-3)

    def run():
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            spikes = neuron(x_seq, dt=1.0)
        features = spikes.detach().mean(dim=0).requires_grad_(True)
        logits = head(features)
        loss = F.mse_loss(logits, target)
        loss.backward()
        optimizer.step()

    return run


def benchmark_lif_back(
    *,
    steps: int = 128,
    batch_size: int = 64,
    head_dim: int = 1024,
    tau_ref: float | None = 2.0,
    tolerance_ratio: float = 1e-6,
    dtype: torch.dtype = torch.float32,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available, skipping backward benchmark")
        return

    sizes = np.unique(np.logspace(3, 5, 20, dtype=int))
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
            f"Benchmarking train-like LIF step with T={steps}, B={batch_size}, "
            f"N={n_neuron}, head_dim={head_dim}, tau_ref={tau_ref}..."
        )
        try:
            x_seq = _build_input(
                steps,
                batch_size,
                int(n_neuron),
                device=device,
                dtype=dtype,
            )
            target = _build_target(
                batch_size,
                head_dim,
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
                target,
                tau_ref=tau_ref,
                head_dim=head_dim,
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
                    x_seq,
                    target,
                    tau_ref=tau_ref,
                    head_dim=head_dim,
                    device=device,
                    dtype=dtype,
                )
            )
            compile_runner = _make_compile_runner(
                x_seq,
                target,
                tau_ref=tau_ref,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
            )
            if compile_runner is not None:
                torch_compile_time = _benchmark_reuse_fn(compile_runner)
            else:
                torch_compile_time = float("nan")
            triton_time = _benchmark_fn(
                lambda: _make_triton_runner(
                    x_seq,
                    target,
                    tau_ref=tau_ref,
                    head_dim=head_dim,
                    device=device,
                    dtype=dtype,
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
        print("No valid backward benchmark results collected.")
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
        label="Torch Eager + Torch Head Backward",
    )
    ax.plot(
        valid_sizes,
        results["torch_compile_ms"],
        color="#9467bd",
        marker="x",
        linewidth=2,
        label="Torch Compile + Torch Head Backward",
    )
    ax.plot(
        valid_sizes,
        results["triton_ms"],
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label="Triton LIF + Torch Head Backward",
    )
    ax.set_title("Train-like Step Latency", fontsize=12, fontweight="bold")
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
    ax.set_title("Train-like Step Speedup", fontsize=12, fontweight="bold")
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
    ax.set_title("Forward Error Rate", fontsize=12, fontweight="bold")
    ax.set_xlabel("Neuron Count N", fontsize=10)
    ax.set_ylabel("Error Rate", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)

    fig.suptitle(
        (
            f"Train-like LIF Benchmark (T={steps}, B={batch_size}, "
            f"head_dim={head_dim}, tau_ref={tau_ref}, tol={tolerance_ratio:.1e})"
        ),
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    output_dir = fig_path(__file__)
    output_path = output_dir / "lif_backward_benchmark.png"
    plt.savefig(output_path, dpi=300)
    print(f"Backward benchmark plot saved to {output_path}")

    print("\nBackward benchmark summary:")
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
            f"eager_step={torch_eager_ms:>8.3f} ms | "
            f"compile_step={torch_compile_ms:>8.3f} ms | "
            f"triton_step={triton_ms:>8.3f} ms | "
            f"eager/triton={speedup_eager:>6.2f}x | "
            f"compile/triton={speedup_compile:>6.2f}x | "
            f"triton_err={triton_error_rate:.3e} | "
            f"compile_err={compile_error_rate:.3e}"
        )


if __name__ == "__main__":
    benchmark_lif_back()
