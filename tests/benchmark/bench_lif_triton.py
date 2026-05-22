import argparse
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from triton.testing import Benchmark, perf_report

from btorch.backend.triton.lif import TritonMultiStepLIF
from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.neurons.lif import LIF
from btorch.utils.file import fig_path

from ..utils.bench import do_bench


providers = ["torch_lif", "triton_fused"]
line_names = ["Torch LIF", "Triton Fused"]
styles = [("red", "-"), ("blue", "-")]


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@contextmanager
def _nvtx_range(name: str, enabled: bool):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


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


def _build_reference_lif(
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


def _assert_correctness(
    *,
    steps: int = 16,
    batch_size: int = 8,
    n_neuron: int = 1024,
    tau_ref: float | None = None,
    dt: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> None:
    x_seq = _build_input(
        steps,
        batch_size,
        n_neuron,
        device=device,
        dtype=dtype,
    )

    ref = _build_reference_lif(
        n_neuron,
        tau_ref=tau_ref,
        device=device,
        dtype=dtype,
    )
    fused = _build_triton_lif(
        n_neuron,
        tau_ref=tau_ref,
        device=device,
        dtype=dtype,
    )

    init_net_state(ref, batch_size=batch_size, device=device, dtype=dtype)
    fused.reset_state(batch_size=batch_size)

    with torch.no_grad(), environ.context(dt=dt):
        spike_ref = ref(x_seq)
        spike_fused = fused(x_seq, dt=dt)

    torch.testing.assert_close(spike_fused, spike_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(fused.v, ref.v, atol=1e-6, rtol=0.0)
    if tau_ref is not None:
        torch.testing.assert_close(
            fused.refractory,
            ref.refractory,
            atol=1e-6,
            rtol=0.0,
        )


def _make_forward_runner(
    provider: str,
    *,
    steps: int,
    batch_size: int,
    n_neuron: int,
    tau_ref: float | None,
    dt: float,
    device: str,
    dtype: torch.dtype,
):
    x_seq = _build_input(
        steps,
        batch_size,
        n_neuron,
        device=device,
        dtype=dtype,
    )

    if provider == "torch_lif":
        neuron = _build_reference_lif(
            n_neuron,
            tau_ref=tau_ref,
            device=device,
            dtype=dtype,
        )
        init_net_state(neuron, batch_size=batch_size, device=device, dtype=dtype)

        def fn():
            with torch.no_grad(), environ.context(dt=dt):
                neuron(x_seq)

        return fn, x_seq

    if provider == "triton_fused":
        neuron = _build_triton_lif(
            n_neuron,
            tau_ref=tau_ref,
            device=device,
            dtype=dtype,
        )
        neuron.reset_state(batch_size=batch_size)

        def fn():
            with torch.no_grad():
                neuron(x_seq, dt=dt)

        return fn, x_seq

    raise ValueError(f"Unknown provider: {provider}")


def _make_trainish_runner(
    provider: str,
    *,
    steps: int,
    batch_size: int,
    n_neuron: int,
    head_dim: int,
    tau_ref: float | None,
    dt: float,
    device: str,
    dtype: torch.dtype,
) -> callable:
    x_seq = _build_input(
        steps,
        batch_size,
        n_neuron,
        device=device,
        dtype=dtype,
    )
    target = _build_target(
        batch_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    head = _build_head(
        n_neuron,
        head_dim,
        device=device,
        dtype=dtype,
    )
    optimizer = torch.optim.SGD(head.parameters(), lr=1e-3)

    if provider == "torch_lif":
        neuron = _build_reference_lif(
            n_neuron,
            tau_ref=tau_ref,
            device=device,
            dtype=dtype,
        )
        init_net_state(neuron, batch_size=batch_size, device=device, dtype=dtype)

        def forward_lif():
            with environ.context(dt=dt):
                return neuron(x_seq)

    elif provider == "triton_fused":
        neuron = _build_triton_lif(
            n_neuron,
            tau_ref=tau_ref,
            device=device,
            dtype=dtype,
        )
        neuron.reset_state(batch_size=batch_size)

        def forward_lif():
            return neuron(x_seq, dt=dt)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    def run_step(*, use_nvtx: bool = False) -> float:
        _sync(device)
        start = time.perf_counter()

        with _nvtx_range("lif_forward", use_nvtx):
            spikes = forward_lif()

        with _nvtx_range("torch_head_backward", use_nvtx):
            optimizer.zero_grad(set_to_none=True)
            features = spikes.detach().mean(dim=0).requires_grad_(True)
            logits = head(features)
            loss = F.mse_loss(logits, target)
            loss.backward()
            optimizer.step()

        _sync(device)
        end = time.perf_counter()
        return (end - start) * 1000

    return run_step


@perf_report(
    [
        Benchmark(
            x_names=["steps"],
            x_vals=[32, 64, 128, 256],
            line_arg="provider",
            line_vals=providers,
            line_names=line_names,
            styles=styles,
            ylabel="GElts/s",
            plot_name="lif_forward_training_scale_vs_steps",
            args={"N": 8192, "batch_size": 64, "tau_ref": 2.0},
            x_log=True,
        ),
        Benchmark(
            x_names=["N"],
            x_vals=[2048, 4096, 8192, 16384],
            line_arg="provider",
            line_vals=providers,
            line_names=line_names,
            styles=styles,
            ylabel="GElts/s",
            plot_name="lif_forward_training_scale_vs_width",
            args={"steps": 128, "batch_size": 64, "tau_ref": 2.0},
            x_log=True,
        ),
    ]
)
def bench_lif_forward_large(N, provider, steps, batch_size, tau_ref):
    device = "cuda"
    dtype = torch.float32
    dt = 1.0

    fn, x_seq = _make_forward_runner(
        provider,
        steps=steps,
        batch_size=batch_size,
        n_neuron=N,
        tau_ref=tau_ref,
        dt=dt,
        device=device,
        dtype=dtype,
    )

    ms = do_bench(fn, timing_method="gpu", quantiles=[0.5, 0.2, 0.8])
    total_elements = x_seq.numel()
    gelts = lambda elapsed_ms: total_elements / (elapsed_ms * 1e-3) / 1e9
    return tuple(gelts(t) for t in ms)


@perf_report(
    [
        Benchmark(
            x_names=["steps"],
            x_vals=[32, 64, 128, 256],
            line_arg="provider",
            line_vals=providers,
            line_names=line_names,
            styles=styles,
            ylabel="ms/step",
            plot_name="lif_trainish_step_vs_steps",
            args={
                "N": 8192,
                "batch_size": 64,
                "head_dim": 1024,
                "tau_ref": 2.0,
            },
            x_log=True,
        ),
        Benchmark(
            x_names=["N"],
            x_vals=[2048, 4096, 8192, 16384],
            line_arg="provider",
            line_vals=providers,
            line_names=line_names,
            styles=styles,
            ylabel="ms/step",
            plot_name="lif_trainish_step_vs_width",
            args={
                "steps": 128,
                "batch_size": 64,
                "head_dim": 1024,
                "tau_ref": 2.0,
            },
            x_log=True,
        ),
    ]
)
def bench_lif_trainish_step(N, provider, steps, batch_size, head_dim, tau_ref):
    device = "cuda"
    dtype = torch.float32
    dt = 1.0

    run_step = _make_trainish_runner(
        provider,
        steps=steps,
        batch_size=batch_size,
        n_neuron=N,
        head_dim=head_dim,
        tau_ref=tau_ref,
        dt=dt,
        device=device,
        dtype=dtype,
    )

    def fn():
        run_step(use_nvtx=False)

    ms = do_bench(fn, timing_method="total", quantiles=[0.5, 0.2, 0.8])
    return tuple(ms)


def _run_nsys_profile(args: argparse.Namespace) -> None:
    run_step = _make_trainish_runner(
        args.provider,
        steps=args.steps,
        batch_size=args.batch_size,
        n_neuron=args.n_neuron,
        head_dim=args.head_dim,
        tau_ref=args.tau_ref,
        dt=args.dt,
        device=args.device,
        dtype=torch.float32,
    )

    for _ in range(args.warmup):
        run_step(use_nvtx=False)

    times = []
    for _ in range(args.iters):
        ms = run_step(use_nvtx=True)
        times.append(ms)

    times_t = torch.tensor(times)
    print(f"provider={args.provider}")
    print(
        f"shape=(T={args.steps}, B={args.batch_size}, N={args.n_neuron}), "
        f"head_dim={args.head_dim}, tau_ref={args.tau_ref}"
    )
    print(
        "step_ms: "
        f"mean={times_t.mean().item():.3f}, "
        f"median={times_t.median().item():.3f}, "
        f"min={times_t.min().item():.3f}, "
        f"max={times_t.max().item():.3f}"
    )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark and nsys-profile the reference multi-step LIF against "
            "a fused Triton forward path."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["correctness", "bench-forward", "bench-trainish", "nsys"],
        default="bench-forward",
    )
    parser.add_argument(
        "--provider",
        choices=providers,
        default="torch_lif",
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-neuron", type=int, default=8192)
    parser.add_argument("--head-dim", type=int, default=1024)
    parser.add_argument("--tau-ref", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    return parser


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    if args.mode == "correctness":
        _assert_correctness(tau_ref=None)
        _assert_correctness(tau_ref=args.tau_ref)
    elif args.mode == "bench-forward":
        bench_lif_forward_large.run(
            show_plots=False,
            print_data=True,
            return_df=False,
            save_path=fig_path(__file__),
        )
    elif args.mode == "bench-trainish":
        bench_lif_trainish_step.run(
            show_plots=False,
            print_data=True,
            return_df=False,
            save_path=fig_path(__file__),
        )
    elif args.mode == "nsys":
        _run_nsys_profile(args)
