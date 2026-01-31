from __future__ import annotations

import time

import torch


try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("cupy is required to run this script.") from exc


def _to_cupy(tensor: torch.Tensor) -> "cp.ndarray":
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))


def _bench_dlpack(tensor: torch.Tensor, iters: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = _to_cupy(tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1e6 / iters


def _bench_cached(tensor: torch.Tensor, iters: int) -> float:
    arr = _to_cupy(tensor)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = arr
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1e6 / iters


def _bench_pointer_only(tensor: torch.Tensor, iters: int) -> float:
    ptr = tensor.data_ptr()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = ptr
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1e6 / iters


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this script.")

    device = torch.device("cuda")
    B, M = 4096, 4
    iters = 20000

    v = torch.randn((B,), device=device, dtype=torch.float32)
    Iasc = torch.randn((B * M,), device=device, dtype=torch.float32)

    dlpack_us = _bench_dlpack(v, iters)
    cached_us = _bench_cached(v, iters)
    ptr_us = _bench_pointer_only(v, iters)

    dlpack_I_us = _bench_dlpack(Iasc, iters)
    cached_I_us = _bench_cached(Iasc, iters)
    ptr_I_us = _bench_pointer_only(Iasc, iters)

    print("DLPack wrapper cost (microseconds per call)")
    print(f"v (B={B}): dlpack={dlpack_us:.3f} us, cached={cached_us:.3f} us")
    print(f"v (B={B}): data_ptr={ptr_us:.3f} us")
    print(
        f"Iasc (B*M={B*M}): dlpack={dlpack_I_us:.3f} us,"
        f" cached={cached_I_us:.3f} us"
    )
    print(f"Iasc (B*M={B*M}): data_ptr={ptr_I_us:.3f} us")


if __name__ == "__main__":
    main()
