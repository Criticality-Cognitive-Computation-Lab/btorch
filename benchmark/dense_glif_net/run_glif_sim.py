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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a GLIF dense simulation with configurable backend."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="torch_eager",
        choices=providers(),
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
        default=2,
        help="Warmup iterations before timing.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
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
    include_backward: bool,
) -> None:
    target = compiled if compiled is not None else model
    reset_net_state(model)
    with environ.context(dt=_DT):
        spike_seq = target(x_seq)[0]
    if include_backward:
        spike_seq.sum().backward()


def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GLIF simulations.")

    model, x_seq, grads = build_model(
        args.provider, args.duration, args.n_neuron, require_grad=args.backward
    )

    compiled = None
    if args.provider == "torch_compile":
        compile_kwargs = {}
        if args.compile_backend is not None:
            compile_kwargs["backend"] = args.compile_backend
        compiled = torch.compile(model, **compile_kwargs)
        reset_net_state(model)
        with environ.context(dt=_DT):
            compiled(x_seq)
        torch.cuda.synchronize()

    for _ in range(args.warmup):
        _run_once(model, x_seq, compiled, args.backward)
        if args.backward:
            for grad in grads:
                if grad.grad is not None:
                    grad.grad.zero_()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.runs):
        _run_once(model, x_seq, compiled, args.backward)
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
