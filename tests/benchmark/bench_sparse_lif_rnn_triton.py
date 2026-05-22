import argparse
import time
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from btorch.backend.triton.lif import TritonSparseLIFRNN, triton_lif_single_step
from btorch.backend.triton.sparse import coo_spmm
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
    hidden_size: int,
    *,
    device: str,
    dtype: torch.dtype,
    seed: int = 0,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x_seq = torch.randn(
        (steps, batch_size, hidden_size),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return 0.15 * x_seq + 0.2


def _build_random_coo(
    hidden_size: int,
    density: float,
    *,
    device: str,
    dtype: torch.dtype,
    seed: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    nnz = max(1, int(hidden_size * hidden_size * density))
    rows = torch.randint(0, hidden_size, (nnz,), generator=generator, device=device)
    cols = torch.randint(0, hidden_size, (nnz,), generator=generator, device=device)
    indices = torch.stack([rows, cols], dim=0)
    values = torch.randn(nnz, generator=generator, device=device, dtype=dtype)
    values = 0.1 * values / max(hidden_size * density, 1.0)
    return indices, values


def _torch_sparse_mm(
    indices: torch.Tensor,
    values: torch.Tensor,
    spike: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor:
    sparse = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(hidden_size, hidden_size),
        device=spike.device,
        dtype=spike.dtype,
    ).coalesce()
    spike_2d = spike.reshape(-1, hidden_size)
    recurrent = torch.sparse.mm(sparse, spike_2d.T).T
    return recurrent.reshape_as(spike)


class TorchSparseLIFRNN(torch.nn.Module):
    def __init__(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        *,
        hidden_size: int,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        c_m: float = 1.0,
        tau: float = 20.0,
        hard_reset: bool = False,
        device: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hard_reset = hard_reset
        self.register_buffer("indices", indices.to(device=device, dtype=torch.long))
        self.register_buffer("values", values.to(device=device, dtype=dtype))
        self.register_buffer(
            "v_threshold",
            torch.full((hidden_size,), v_threshold, device=device, dtype=dtype),
        )
        self.register_buffer(
            "v_reset",
            torch.full((hidden_size,), v_reset, device=device, dtype=dtype),
        )
        self.register_buffer(
            "c_m",
            torch.full((hidden_size,), c_m, device=device, dtype=dtype),
        )
        self.register_buffer(
            "tau",
            torch.full((hidden_size,), tau, device=device, dtype=dtype),
        )
        self.register_buffer("v", self.v_reset.clone())
        self.register_buffer("spike", torch.zeros_like(self.v_reset))
        self._state_shape = tuple(self.v.shape)

    @torch.no_grad()
    def reset_state(self, batch_size: int) -> None:
        state_shape = (batch_size, self.hidden_size)
        self.v = torch.broadcast_to(self.v_reset, state_shape).clone()
        self.spike = torch.zeros_like(self.v)
        self._state_shape = state_shape

    def _lif_step(self, current: torch.Tensor, dt: float) -> torch.Tensor:
        dv = -(self.v - self.v_reset) / self.tau + current / self.c_m
        v = self.v + dt * dv
        spike = (v >= self.v_threshold).to(v.dtype)
        if self.hard_reset:
            v = v - (v - self.v_reset) * spike
        else:
            v = v - (self.v_threshold - self.v_reset) * spike
        self.v = v
        self.spike = spike
        return spike

    def forward(self, x_seq: torch.Tensor, *, dt: float) -> torch.Tensor:
        if tuple(x_seq.shape[1:]) != self._state_shape:
            self.reset_state(x_seq.shape[1])

        spikes = []
        for t in range(x_seq.shape[0]):
            recurrent = _torch_sparse_mm(
                self.indices,
                self.values,
                self.spike,
                self.hidden_size,
            )
            spikes.append(self._lif_step(x_seq[t] + recurrent, dt))
        return torch.stack(spikes, dim=0)


class _CompiledSparseLIFWrapper(torch.nn.Module):
    def __init__(self, model: TorchSparseLIFRNN, dt: float):
        super().__init__()
        self.model = model
        self.dt = dt

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return self.model(x_seq, dt=self.dt)


class TritonSparseLIFRNNCUDAGraph:
    """CUDA-graph runner for the fixed-shape Triton sparse LIF RNN forward."""

    def __init__(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        *,
        steps: int,
        batch_size: int,
        hidden_size: int,
        device: str,
        dtype: torch.dtype,
        dt: float,
    ):
        if not hasattr(torch.cuda, "CUDAGraph"):
            raise RuntimeError("CUDA Graph is not available in this PyTorch build.")

        self.indices = indices
        self.values = values
        self.steps = steps
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.dt = dt

        self.x_static = torch.empty(
            (steps, batch_size, hidden_size),
            device=device,
            dtype=dtype,
        )
        self.out_static = torch.empty_like(self.x_static)
        self.v = torch.zeros((batch_size, hidden_size), device=device, dtype=dtype)
        self.spike = torch.zeros_like(self.v)
        self.v_threshold = torch.ones_like(self.v)
        self.v_reset = torch.zeros_like(self.v)
        self.c_m = torch.ones_like(self.v)
        self.tau = torch.full_like(self.v, 20.0)
        self.v_threshold_flat = self.v_threshold.reshape(-1)
        self.v_reset_flat = self.v_reset.reshape(-1)
        self.c_m_flat = self.c_m.reshape(-1)
        self.tau_flat = self.tau.reshape(-1)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self._run_impl()
        torch.cuda.current_stream().wait_stream(warmup_stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._run_impl()

    def _run_impl(self) -> None:
        self.v.zero_()
        self.spike.zero_()
        for t in range(self.steps):
            spike_2d = self.spike.reshape(-1, self.hidden_size)
            recurrent = coo_spmm(
                self.indices,
                self.values,
                spike_2d.T.contiguous(),
                size_m=self.hidden_size,
            )
            current = self.x_static[t] + recurrent.T.reshape_as(self.spike)
            spike_flat, v_next_flat = triton_lif_single_step(
                current.reshape(-1),
                self.v.reshape(-1),
                self.v_threshold_flat,
                self.v_reset_flat,
                self.c_m_flat,
                self.tau_flat,
                dt=self.dt,
                hard_reset=False,
            )
            self.v.copy_(v_next_flat.reshape_as(self.v))
            self.spike.copy_(spike_flat.reshape_as(self.spike))
            self.out_static[t].copy_(self.spike)

    def replay(self, x_seq: torch.Tensor) -> torch.Tensor:
        self.x_static.copy_(x_seq)
        self.graph.replay()
        return self.out_static


def _build_torch_model(
    indices: torch.Tensor,
    values: torch.Tensor,
    *,
    hidden_size: int,
    device: str,
    dtype: torch.dtype,
) -> TorchSparseLIFRNN:
    return TorchSparseLIFRNN(
        indices,
        values,
        hidden_size=hidden_size,
        device=device,
        dtype=dtype,
    )


def _build_triton_model(
    indices: torch.Tensor,
    values: torch.Tensor,
    *,
    hidden_size: int,
    device: str,
    dtype: torch.dtype,
) -> TritonSparseLIFRNN:
    return TritonSparseLIFRNN(
        indices,
        values,
        n_neuron=hidden_size,
        device=torch.device(device),
        dtype=dtype,
    )


def _make_torch_runner(
    x_seq: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
    dt: float,
):
    hidden_size = x_seq.shape[-1]
    batch_size = x_seq.shape[1]
    model = _build_torch_model(
        indices,
        values,
        hidden_size=hidden_size,
        device=device,
        dtype=dtype,
    )

    def run() -> None:
        model.reset_state(batch_size)
        with torch.no_grad():
            model(x_seq, dt=dt)

    return run


def _make_compile_runner(
    x_seq: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
    dt: float,
):
    if not hasattr(torch, "compile"):
        return None

    hidden_size = x_seq.shape[-1]
    batch_size = x_seq.shape[1]
    model = _build_torch_model(
        indices,
        values,
        hidden_size=hidden_size,
        device=device,
        dtype=dtype,
    )
    compiled = torch.compile(_CompiledSparseLIFWrapper(model, dt))

    with torch.no_grad():
        model.reset_state(batch_size)
        compiled(x_seq)

    def run() -> None:
        model.reset_state(batch_size)
        with torch.no_grad():
            compiled(x_seq)

    return run


def _make_triton_runner(
    x_seq: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
    dt: float,
):
    hidden_size = x_seq.shape[-1]
    batch_size = x_seq.shape[1]
    model = _build_triton_model(
        indices,
        values,
        hidden_size=hidden_size,
        device=device,
        dtype=dtype,
    )

    def run() -> None:
        model.reset_state(batch_size)
        with torch.no_grad():
            model(x_seq, dt=dt)

    return run


def _make_triton_cudagraph_runner(
    x_seq: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
    dt: float,
):
    if not hasattr(torch.cuda, "CUDAGraph"):
        return None

    graph_runner = TritonSparseLIFRNNCUDAGraph(
        indices,
        values,
        steps=x_seq.shape[0],
        batch_size=x_seq.shape[1],
        hidden_size=x_seq.shape[2],
        device=device,
        dtype=dtype,
        dt=dt,
    )

    def run() -> None:
        graph_runner.replay(x_seq)

    return run


def _compute_error_rate(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    mismatch = candidate != reference
    return int(mismatch.sum().item()) / mismatch.numel()


def _benchmark_sparse_lif_rnn_sweep(
    *,
    x_label: str,
    x_values: np.ndarray,
    build_case: Callable[[int], dict[str, int | float]],
    title_prefix: str,
    output_name: str,
    tolerance_ratio: float,
    dtype: torch.dtype,
    dt: float,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available, skipping benchmark")
        return

    results = {
        "torch_eager_ms": [],
        "torch_compile_ms": [],
        "triton_ms": [],
        "triton_cudagraph_ms": [],
        "speedup_eager": [],
        "speedup_compile": [],
        "speedup_triton": [],
        "triton_error_rate": [],
        "compile_error_rate": [],
        "triton_cudagraph_error_rate": [],
    }
    valid_x = []
    case_labels = []

    for x_value in x_values:
        case = build_case(int(x_value))
        print(
            f"Benchmarking sparse LIF RNN with "
            f"T={case['steps']}, B={case['batch_size']}, "
            f"N={case['hidden_size']}, density={case['density']}..."
        )
        try:
            x_seq = _build_input(
                int(case["steps"]),
                int(case["batch_size"]),
                int(case["hidden_size"]),
                device=device,
                dtype=dtype,
            )
            indices, values = _build_random_coo(
                int(case["hidden_size"]),
                float(case["density"]),
                device=device,
                dtype=dtype,
            )

            torch_check = _build_torch_model(
                indices,
                values,
                hidden_size=int(case["hidden_size"]),
                device=device,
                dtype=dtype,
            )
            torch_check.reset_state(int(case["batch_size"]))
            with torch.no_grad():
                y_torch = torch_check(x_seq, dt=dt)

            compile_error_rate = float("nan")
            compile_check_runner = _make_compile_runner(
                x_seq,
                indices,
                values,
                device=device,
                dtype=dtype,
                dt=dt,
            )
            if compile_check_runner is not None:
                compile_check = _build_torch_model(
                    indices,
                    values,
                    hidden_size=int(case["hidden_size"]),
                    device=device,
                    dtype=dtype,
                )
                compiled_check = torch.compile(
                    _CompiledSparseLIFWrapper(compile_check, dt)
                )
                with torch.no_grad():
                    compile_check.reset_state(int(case["batch_size"]))
                    y_compile = compiled_check(x_seq)
                compile_error_rate = _compute_error_rate(y_compile, y_torch)

            triton_check = _build_triton_model(
                indices,
                values,
                hidden_size=int(case["hidden_size"]),
                device=device,
                dtype=dtype,
            )
            triton_check.reset_state(int(case["batch_size"]))
            with torch.no_grad():
                y_triton = triton_check(x_seq, dt=dt)

            triton_cudagraph_error_rate = float("nan")
            triton_cudagraph_runner = _make_triton_cudagraph_runner(
                x_seq,
                indices,
                values,
                device=device,
                dtype=dtype,
                dt=dt,
            )
            if triton_cudagraph_runner is not None:
                graph_check = TritonSparseLIFRNNCUDAGraph(
                    indices,
                    values,
                    steps=int(case["steps"]),
                    batch_size=int(case["batch_size"]),
                    hidden_size=int(case["hidden_size"]),
                    device=device,
                    dtype=dtype,
                    dt=dt,
                )
                with torch.no_grad():
                    y_triton_cudagraph = graph_check.replay(x_seq).clone()
                triton_cudagraph_error_rate = _compute_error_rate(
                    y_triton_cudagraph,
                    y_torch,
                )

            triton_error_rate = _compute_error_rate(y_triton, y_torch)
            if triton_error_rate > tolerance_ratio:
                print(
                    f"Warning: Triton error rate {triton_error_rate:.3e} exceeds "
                    f"tolerance {tolerance_ratio:.3e} at {x_label}={int(x_value)}"
                )
            if (
                not np.isnan(compile_error_rate)
                and compile_error_rate > tolerance_ratio
            ):
                print(
                    f"Warning: compile error rate {compile_error_rate:.3e} exceeds "
                    f"tolerance {tolerance_ratio:.3e} at {x_label}={int(x_value)}"
                )
            if (
                not np.isnan(triton_cudagraph_error_rate)
                and triton_cudagraph_error_rate > tolerance_ratio
            ):
                print(
                    "Warning: Triton CUDA Graph error rate "
                    f"{triton_cudagraph_error_rate:.3e} exceeds tolerance "
                    f"{tolerance_ratio:.3e} at {x_label}={int(x_value)}"
                )

            torch_eager_time = _benchmark_fn(
                lambda: _make_torch_runner(
                    x_seq,
                    indices,
                    values,
                    device=device,
                    dtype=dtype,
                    dt=dt,
                )
            )
            compile_runner = _make_compile_runner(
                x_seq,
                indices,
                values,
                device=device,
                dtype=dtype,
                dt=dt,
            )
            if compile_runner is not None:
                torch_compile_time = _benchmark_reuse_fn(compile_runner)
            else:
                torch_compile_time = float("nan")
            triton_time = _benchmark_fn(
                lambda: _make_triton_runner(
                    x_seq,
                    indices,
                    values,
                    device=device,
                    dtype=dtype,
                    dt=dt,
                )
            )
            triton_cudagraph_runner = _make_triton_cudagraph_runner(
                x_seq,
                indices,
                values,
                device=device,
                dtype=dtype,
                dt=dt,
            )
            if triton_cudagraph_runner is not None:
                triton_cudagraph_time = _benchmark_reuse_fn(triton_cudagraph_runner)
            else:
                triton_cudagraph_time = float("nan")

            results["torch_eager_ms"].append(torch_eager_time * 1000)
            results["torch_compile_ms"].append(torch_compile_time * 1000)
            results["triton_ms"].append(triton_time * 1000)
            results["triton_cudagraph_ms"].append(triton_cudagraph_time * 1000)
            speedup_base = (
                triton_cudagraph_time
                if not np.isnan(triton_cudagraph_time)
                else triton_time
            )
            results["speedup_eager"].append(torch_eager_time / speedup_base)
            results["speedup_compile"].append(
                torch_compile_time / speedup_base
                if not np.isnan(torch_compile_time)
                else float("nan")
            )
            results["speedup_triton"].append(triton_time / speedup_base)
            results["triton_error_rate"].append(triton_error_rate)
            results["compile_error_rate"].append(compile_error_rate)
            results["triton_cudagraph_error_rate"].append(
                triton_cudagraph_error_rate
            )
            valid_x.append(int(x_value))
            case_labels.append(case)
        except Exception as exc:
            print(f"OOM or error at {x_label}={x_value}: {exc}")
            break

    if not valid_x:
        print("No valid benchmark results collected.")
        return

    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.plot(
        valid_x,
        results["torch_eager_ms"],
        color="#2ca02c",
        marker="^",
        linewidth=2,
        label="Torch Eager",
    )
    ax.plot(
        valid_x,
        results["torch_compile_ms"],
        color="#9467bd",
        marker="x",
        linewidth=2,
        label="Torch Compile",
    )
    ax.plot(
        valid_x,
        results["triton_ms"],
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label="Triton",
    )
    ax.plot(
        valid_x,
        results["triton_cudagraph_ms"],
        color="#17becf",
        marker="P",
        linewidth=2,
        label="Triton + CUDA Graph",
    )
    ax.set_title(f"{title_prefix} Forward Latency", fontsize=12, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Time (ms)", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)

    ax = axes[1]
    ax.plot(
        valid_x,
        results["speedup_eager"],
        color="#ff7f0e",
        marker="D",
        linewidth=2,
        label="Eager / Triton + Graph",
    )
    ax.plot(
        valid_x,
        results["speedup_compile"],
        color="#8c564b",
        marker="s",
        linewidth=2,
        label="Compile / Triton + Graph",
    )
    ax.plot(
        valid_x,
        results["speedup_triton"],
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label="Triton / Triton + Graph",
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title(
        f"{title_prefix} CUDA Graph Speedup",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Speedup (x)", fontsize=10)
    ax.set_xscale("log")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)

    ax = axes[2]
    ax.plot(
        valid_x,
        results["triton_error_rate"],
        color="#d62728",
        marker="s",
        linewidth=2,
        label="Triton vs Eager",
    )
    ax.plot(
        valid_x,
        results["compile_error_rate"],
        color="#7f7f7f",
        marker="x",
        linewidth=2,
        label="Compile vs Eager",
    )
    ax.plot(
        valid_x,
        results["triton_cudagraph_error_rate"],
        color="#17becf",
        marker="P",
        linewidth=2,
        label="Triton + Graph vs Eager",
    )
    ax.axhline(
        tolerance_ratio,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"Tolerance = {tolerance_ratio:.1e}",
    )
    ax.set_title(f"{title_prefix} Error Rate", fontsize=12, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Error Rate", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)

    first_case = case_labels[0]
    fig.suptitle(
        (
            f"{title_prefix} Benchmark "
            f"(T={first_case['steps']}, B={first_case['batch_size']}, "
            f"N={first_case['hidden_size']}, density={first_case['density']}, "
            f"tol={tolerance_ratio:.1e})"
        ),
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.98,
        "Execution per step: spmm + lif",
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
    output_path = output_dir / output_name
    plt.savefig(output_path, dpi=300)
    print(f"Benchmark plot saved to {output_path}")

    print(f"\n{title_prefix} summary:")
    for (
        x_value,
        torch_eager_ms,
        torch_compile_ms,
        triton_ms,
        triton_cudagraph_ms,
        speedup_eager,
        speedup_compile,
        speedup_triton,
        triton_error_rate,
        compile_error_rate,
        triton_cudagraph_error_rate,
    ) in zip(
        valid_x,
        results["torch_eager_ms"],
        results["torch_compile_ms"],
        results["triton_ms"],
        results["triton_cudagraph_ms"],
        results["speedup_eager"],
        results["speedup_compile"],
        results["speedup_triton"],
        results["triton_error_rate"],
        results["compile_error_rate"],
        results["triton_cudagraph_error_rate"],
    ):
        print(
            f"{x_label}={x_value:>6d} | "
            f"eager={torch_eager_ms:>8.3f} ms | "
            f"compile={torch_compile_ms:>8.3f} ms | "
            f"triton={triton_ms:>8.3f} ms | "
            f"triton_graph={triton_cudagraph_ms:>8.3f} ms | "
            f"eager/triton_graph={speedup_eager:>6.2f}x | "
            f"compile/triton_graph={speedup_compile:>6.2f}x | "
            f"triton/triton_graph={speedup_triton:>6.2f}x | "
            f"triton_err={triton_error_rate:.3e} | "
            f"compile_err={compile_error_rate:.3e} | "
            f"triton_graph_err={triton_cudagraph_error_rate:.3e}"
        )


def benchmark_sparse_lif_rnn_vs_n(
    *,
    steps: int = 128,
    batch_size: int = 64,
    density: float = 0.01,
    tolerance_ratio: float = 1e-6,
    dtype: torch.dtype = torch.float32,
    dt: float = 1.0,
):
    sizes_n = np.unique(np.logspace(2, 4, 16, dtype=int))

    def build_case(hidden_size: int) -> dict[str, int | float]:
        return {
            "steps": steps,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "density": density,
        }

    _benchmark_sparse_lif_rnn_sweep(
        x_label="Neuron Count N",
        x_values=sizes_n,
        build_case=build_case,
        title_prefix="Sparse LIF RNN vs N",
        output_name="sparse_lif_rnn_benchmark_vs_n.png",
        tolerance_ratio=tolerance_ratio,
        dtype=dtype,
        dt=dt,
    )


def benchmark_sparse_lif_rnn_vs_t(
    *,
    hidden_size: int = 4096,
    batch_size: int = 64,
    density: float = 0.01,
    tolerance_ratio: float = 1e-6,
    dtype: torch.dtype = torch.float32,
    dt: float = 1.0,
):
    sizes_t = np.unique(np.logspace(1, 3, 12, dtype=int))

    def build_case(steps: int) -> dict[str, int | float]:
        return {
            "steps": steps,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "density": density,
        }

    _benchmark_sparse_lif_rnn_sweep(
        x_label="Time Steps T",
        x_values=sizes_t,
        build_case=build_case,
        title_prefix="Sparse LIF RNN vs T",
        output_name="sparse_lif_rnn_benchmark_vs_t.png",
        tolerance_ratio=tolerance_ratio,
        dtype=dtype,
        dt=dt,
    )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a sparse recurrent single-layer LIF RNN with "
            "per-step spmm + lif execution."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["bench-n", "bench-t", "bench-all"],
        default="bench-all",
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-neuron", type=int, default=4096)
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--tolerance-ratio", type=float, default=1e-6)
    return parser


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    if args.mode in {"bench-n", "bench-all"}:
        benchmark_sparse_lif_rnn_vs_n(
            steps=args.steps,
            batch_size=args.batch_size,
            density=args.density,
            dt=args.dt,
            tolerance_ratio=args.tolerance_ratio,
        )
    if args.mode in {"bench-t", "bench-all"}:
        benchmark_sparse_lif_rnn_vs_t(
            hidden_size=args.n_neuron,
            batch_size=args.batch_size,
            density=args.density,
            dt=args.dt,
            tolerance_ratio=args.tolerance_ratio,
        )
