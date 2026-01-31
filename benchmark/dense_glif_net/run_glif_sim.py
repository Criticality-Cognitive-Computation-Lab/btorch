from __future__ import annotations

import argparse
import time

import torch

from benchmark.dense_glif_net.glif_common import (
    _DT,
    build_model,
    providers,
)
from btorch.models import environ
from btorch.models.functional import reset_net_state


def _providers_with_multistep() -> list[str]:
    base = providers()
    extra = []
    for name in base:
        if name in {"triton", "warp", "cupy"}:
            extra.append(f"multistep_{name}")
    return base + extra


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a GLIF dense simulation with configurable backend."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="torch_eager",
        choices=_providers_with_multistep(),
        help="Backend provider to use.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=100,
        help="Simulation duration (T).",
    )
    parser.add_argument(
        "--n-neuron",
        type=int,
        default=128,
        help="Number of neurons (N).",
    )
    parser.add_argument(
        "--compile-backend",
        type=str,
        default=None,
        help="torch.compile backend (only for provider torch_compile).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations before timing.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Run backward pass and include it in timing.",
    )
    return parser.parse_args()


def _run_once(
    model: torch.nn.Module,
    x_seq: torch.Tensor,
    compiled: torch.nn.Module | None,
    use_multistep: bool,
    include_backward: bool,
) -> None:
    target = compiled if compiled is not None else model
    reset_net_state(model)
    grad_ctx = torch.enable_grad() if include_backward else torch.no_grad()
    with grad_ctx:
        with environ.context(dt=_DT):
            if use_multistep:
                neuron = model.neuron
                step_fn = neuron.step_fn
                if hasattr(step_fn, "dense_multistep_fused") and not include_backward:
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
                else:
                    spike_seq = target(x_seq)[0]
                neuron.v = v_out
                neuron.Iasc = I_out
                model.spike = spike_seq[-1]
            else:
                spike_seq = target(x_seq)[0]
    if include_backward:
        spike_seq.sum().backward()


def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GLIF simulations.")

    provider = args.provider
    use_multistep = provider.startswith("multistep_")
    if use_multistep:
        provider = provider.removeprefix("multistep_")

    model, x_seq, grads = build_model(
        provider, args.duration, args.n_neuron, require_grad=args.backward
    )

    compiled = None
    if provider == "torch_compile":
        compile_kwargs = {}
        if args.compile_backend is not None:
            compile_kwargs["backend"] = args.compile_backend
        compiled = torch.compile(model, **compile_kwargs)

    warmup_ms = None
    for i in range(args.warmup):
        if i == 0:
            warmup_start = time.perf_counter()
        _run_once(model, x_seq, compiled, use_multistep, args.backward)
        if args.backward:
            for grad in grads:
                if grad.grad is not None:
                    grad.grad.zero_()
        if i == 0:
            torch.cuda.synchronize()
            warmup_ms = (time.perf_counter() - warmup_start) * 1e3
    if warmup_ms is not None:
        print(f"warmup_ms={warmup_ms:.4f}")
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.runs):
        _run_once(model, x_seq, compiled, use_multistep, args.backward)
        if args.backward:
            for grad in grads:
                if grad.grad is not None:
                    grad.grad.zero_()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1e3 / max(1, args.runs)

    print(
        f"provider={args.provider} duration={args.duration} "
        f"n_neuron={args.n_neuron} backward={args.backward} "
        f"compile_backend={args.compile_backend} "
        f"avg_ms={elapsed:.4f}"
    )


if __name__ == "__main__":
    main()
