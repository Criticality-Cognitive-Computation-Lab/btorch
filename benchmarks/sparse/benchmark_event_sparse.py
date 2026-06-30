"""Benchmark numeric and event execution for fixed-degree sparse matrices."""

from __future__ import annotations

import argparse
import time

import torch

from btorch.models import SparseLinear
from btorch.sparse import ELL, BinaryEvents, Normal


def measure(fn, *, warmup: int = 10, repeat: int = 100) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1_000 / repeat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--active-ratio", type=float, default=0.01)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The event benchmark requires CUDA.")
    device = torch.device("cuda")
    matrix = ELL.random(
        shape=(args.n, args.n),
        fan_out=args.degree,
        value=Normal(0.0, 0.02),
        seed=1234,
        allow_self=False,
        device=device,
    )
    linear = SparseLinear(matrix).eval()
    spikes = (
        torch.rand(args.batch_size, args.n, device=device) < args.active_ratio
    ).float()

    with torch.no_grad():
        numeric = linear(spikes)
        event = linear(BinaryEvents(spikes))
        torch.testing.assert_close(event, numeric, atol=1e-5, rtol=1e-5)
        numeric_ms = measure(lambda: linear(spikes))
        event_ms = measure(lambda: linear(BinaryEvents(spikes)))

    print(f"numeric: {numeric_ms:.4f} ms")
    print(f"event:   {event_ms:.4f} ms")
    print(f"speedup: {numeric_ms / event_ms:.2f}x")


if __name__ == "__main__":
    main()
