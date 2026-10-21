"""Compare Triton pull kernels with cuSPARSE products and Torch edge gradients.

Times ``csr_matvec`` (forward) and ``edge_grad`` (gradient w.r.t. the values)
of ``btorch.sparse.runtime.kernels_triton`` and ``kernels_torch`` on raw CSR
buffers, without modules, custom operators or autograd in the way.

Run::

    python -m benchmarks.sparse_conn.bench_kernels_pull \
        --out benchmarks/sparse_conn/results/kernels_pull_rtx5090.json

Workloads are ``name: (n_neuron, in-degree, n_hub, hub_degree)``. Rows have a
uniform length except for ``n_hub`` rows of ``hub_degree`` entries.

The forward baseline reuses the runtime's native cuSPARSE product, including
its contiguous transposed operand. Edge gradients use the Torch reference.

Timing: calls are queued back to back and the device is synchronised once
per repeat, so the figure is the steady-state cost per call including the
Python launch overhead. Implementations are timed round-robin and the
median and the minimum over the repeats are printed as ``median (min)`` in
milliseconds (the GPU may be shared); the speed-up is the best baseline's
median over the Triton median.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from btorch.sparse.runtime import kernels_torch, kernels_triton


_BASELINES = ("torch_ms", "cusparse_ms")

WORKLOADS = {
    "4k": (4096, 100, 0, 0),
    "100k": (100_000, 100, 0, 0),
    "1M": (1_000_000, 50, 0, 0),
    # Heavy tail: 20 rows that are 200x longer than the rest.
    "100k-hub": (100_000, 100, 20, 20_000),
    "4k-hub": (4096, 100, 4, 20_000),
}


def make_csr(n: int, degree: int, n_hub: int, hub_degree: int, device: str):
    """Random square CSR ``(crow, col)`` with optional hub rows."""
    gen = torch.Generator(device=device).manual_seed(0)
    counts = torch.full((n,), degree, dtype=torch.long, device=device)
    if n_hub:
        hubs = torch.randperm(n, device=device, generator=gen)[:n_hub]
        counts[hubs] = hub_degree
    crow = torch.zeros(n + 1, dtype=torch.long, device=device)
    crow[1:] = torch.cumsum(counts, 0)
    col = torch.randint(0, n, (int(crow[-1]),), device=device, generator=gen)
    return crow, col


def time_calls(
    fns: dict, *, target_s: float = 0.25, repeats: int = 9
) -> dict[str, tuple[float, float]]:
    """``(median, min)`` seconds per call of every function in ``fns``.

    The implementations are timed round-robin, so a burst of foreign
    load on a shared GPU hits all of them instead of biasing one.
    """
    iters = {}
    for name, fn in fns.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        once = time.perf_counter() - start
        iters[name] = max(3, min(2000, int(target_s / max(once, 1e-6))))
    samples = {name: [] for name in fns}
    for _ in range(repeats):
        for name, fn in fns.items():
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iters[name]):
                fn()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) / iters[name])
    return {k: (statistics.median(v), min(v)) for k, v in samples.items()}


def bench_workload(name: str, batches: list[int], density: float, device: str):
    n, degree, n_hub, hub_degree = WORKLOADS[name]
    crow, col = make_csr(n, degree, n_hub, hub_degree, device)
    gen = torch.Generator(device=device).manual_seed(1)
    values = torch.randn(col.shape[0], device=device, generator=gen)
    rows = []
    for batch in batches:
        # Spike-like input: a fraction ``density`` of ones.
        x = (torch.rand(batch, n, device=device, generator=gen) < density).float()
        grad = torch.randn(batch, n, device=device, generator=gen)
        # name -> callable per implementation; "cusparse" only for forward.
        cases = {
            "forward": {
                "cusparse": lambda x=x: kernels_torch.csr_matvec_cusparse(
                    crow, col, values, x
                ),
                "triton": lambda x=x: kernels_triton.csr_matvec(crow, col, values, x),
            },
            "edge_grad": {
                "torch": lambda x=x, grad=grad: kernels_torch.edge_grad(
                    crow, col, grad, x, 0
                ),
                "triton": lambda x=x, grad=grad: kernels_triton.edge_grad(
                    crow, col, grad, x, 0
                ),
            },
        }
        for case, impls in cases.items():
            # Agreement of the two backends on this exact input.
            baseline = "cusparse" if case == "forward" else "torch"
            ref, out = impls[baseline](), impls["triton"]()
            err = float((ref - out).abs().max() / ref.abs().max().clamp_min(1e-30))
            del ref, out
            row = {
                "workload": name,
                "n": n,
                "nnz": int(col.shape[0]),
                "batch": batch,
                "kernel": case,
                "max_rel_err": err,
            }
            for impl, (median, best) in time_calls(impls).items():
                row[f"{impl}_ms"] = median * 1e3
                row[f"{impl}_min_ms"] = best * 1e3
            best_other = min(row[k] for k in _BASELINES if k in row)
            row["speedup_vs_best"] = best_other / row["triton_ms"]
            rows.append(row)
            cells = "  ".join(
                f"{impl} {row[f'{impl}_ms']:8.4f} ({row[f'{impl}_min_ms']:8.4f})"
                for impl in impls
            )
            print(
                f"{name:9s} B={batch:<4d} {case:9s} {cells}"
                f"  x{row['speedup_vs_best']:5.2f}  err {err:.1e}",
                flush=True,
            )
        del x, grad
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workloads", nargs="+", default=["4k", "100k", "1M"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = "cuda"
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__)
    results = []
    for name in args.workloads:
        results += bench_workload(name, args.batches, args.density, device)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        meta = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
        args.out.write_text(json.dumps({"meta": meta, "results": results}, indent=1))


if __name__ == "__main__":
    main()
