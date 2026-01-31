from __future__ import annotations

import time

import torch


try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("cupy is required to run this script.") from exc


def _to_cupy(tensor: torch.Tensor) -> "cp.ndarray":
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))


def _bench_loop(iters: int, size: int, use_external_stream: bool) -> float:
    if use_external_stream:
        stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    else:
        stream = None

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        x = torch.randn((size,), device="cuda", dtype=torch.float32)
        if stream is not None:
            with stream:
                y = _to_cupy(x)
                y += 1.0
        else:
            y = _to_cupy(x)
            y += 1.0
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1e6 / iters


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this script.")

    iters = 2000
    size = 1_000_000

    no_stream_us = _bench_loop(iters, size, use_external_stream=False)
    ext_stream_us = _bench_loop(iters, size, use_external_stream=True)

    print("CuPy/Torch stream sync check (microseconds per iteration)")
    print(f"default stream usage: {no_stream_us:.3f} us")
    print(f"external stream usage: {ext_stream_us:.3f} us")


if __name__ == "__main__":
    main()
