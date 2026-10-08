"""Latency benchmark for sparse connection layers.

Times ``y = conn(x)`` (forward) and forward+backward for every available
implementation on synthetic recurrent graphs, so that changes to the sparse
runtime can be compared against the legacy ``torch.sparse`` / ``torch_sparse``
paths and a dense reference.

Run::

    python benchmarks/sparse_conn/bench_sparse_conn.py --device cuda \
        --workloads debug v1 --out benchmarks/sparse_conn/results/local.json

Workloads are ``name: (n_neuron, mean in-degree)``; every workload is run for
each batch size and spike density.  Conversion/preprocessing (module
construction) is timed separately from steady-state execution.
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy.sparse
import torch


WORKLOADS = {
    "debug": (256, 16),
    "v1": (4096, 100),  # ~4k-neuron recurrent column
    "large": (100_000, 100),
    "xlarge": (1_000_000, 50),
}


def make_graph(n: int, indegree: int, seed: int = 0) -> scipy.sparse.coo_array:
    """Random recurrent graph (rows = source, cols = destination)."""
    rng = np.random.default_rng(seed)
    nnz = n * indegree
    src = rng.integers(0, n, nnz)
    dst = np.repeat(np.arange(n), indegree)
    w = rng.standard_normal(nnz).astype(np.float32)
    mat = scipy.sparse.coo_array((w, (src, dst)), shape=(n, n))
    mat.sum_duplicates()
    return mat


def _builders(n: int, device: str, compile_modes: bool) -> dict[str, Callable]:
    """Map implementation name -> ``f(scipy matrix) -> module``."""
    from btorch.models import linear

    out: dict[str, Callable] = {}
    for backend in linear.available_sparse_backends():
        out[f"legacy[{backend}]"] = lambda m, b=backend: linear.SparseConn(
            m, enforce_dale=False, sparse_backend=b, device=device
        )
    if n <= 8192:
        out["dense"] = lambda m: linear.DenseConn(
            n,
            n,
            weight=torch.tensor(m.toarray(), device=device),
            device=device,
        )
    try:  # new runtime (absent before the sparse refactor)
        from btorch.models.connection import SparseConnection
    except ImportError:
        SparseConnection = None
    if SparseConnection is not None:
        out["new[eager]"] = lambda m: SparseConnection.from_adjacency(m).to(device)
        if compile_modes:
            out["new[compile]"] = lambda m: torch.compile(
                SparseConnection.from_adjacency(m).to(device), fullgraph=True
            )
    return out


def _time(fn: Callable, device: str, repeats: int) -> float:
    """Median latency of ``fn`` in milliseconds."""
    for _ in range(3):
        fn()
    samples = []
    inner = 5  # calls per sample, so one synchronisation is amortised
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3 / inner)
    return float(np.median(samples))


def _bench_module(mod, name, wl, n, nnz, build_ms, args) -> list[dict]:
    """Time one module over every batch size and spike density."""
    device = args.device
    rows = []
    params = [p for p in mod.parameters() if p.requires_grad]
    for batch in args.batch:
        for density in args.density:
            g = torch.Generator(device="cpu").manual_seed(0)
            x = (torch.rand(batch, n, generator=g) < density).float()
            x = x.to(device).requires_grad_(True)

            def fwd():
                with torch.no_grad():
                    mod(x)

            def fwd_bwd():
                torch.autograd.grad(mod(x).square().sum(), [x, *params])

            row = {
                "workload": wl,
                "n": n,
                "nnz": nnz,
                "impl": name,
                "batch": batch,
                "density": density,
                "device": device,
                "build_ms": build_ms,
            }
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            # Forward and backward are timed separately so that a backward
            # failure (e.g. out of memory) still reports the forward latency.
            for key, fn in (("fwd_ms", fwd), ("fwd_bwd_ms", fwd_bwd)):
                try:
                    row[key] = _time(fn, device, args.repeats)
                except Exception as e:
                    row[key] = float("nan")
                    row[f"{key}_error"] = f"{type(e).__name__}: {str(e)[:70]}"
            if device == "cuda":
                row["peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
            rows.append(row)
            print(
                f"{wl:7s} {name:22s} B={batch:<3d} p={density:<5g} "
                f"fwd {row['fwd_ms']:9.3f} ms  fwd+bwd {row['fwd_bwd_ms']:9.3f} ms"
                f"  peak {row.get('peak_mb', float('nan')):8.1f} MB"
                f"  (build {build_ms:.0f} ms)"
                + "".join(f"  [{v}]" for k, v in row.items() if k.endswith("_error")),
                flush=True,
            )
    return rows


def run(args: argparse.Namespace) -> list[dict]:
    rows = []
    device = args.device
    for wl in args.workloads:
        n, indegree = WORKLOADS[wl]
        mat = make_graph(n, indegree)
        for name, build in _builders(n, device, args.compile).items():
            if args.impl and not any(k in name for k in args.impl):
                continue
            try:
                t0 = time.perf_counter()
                mod = build(mat)
                build_ms = (time.perf_counter() - t0) * 1e3
            except Exception as e:  # report, keep going
                print(f"{wl:7s} {name:22s} build FAILED: {type(e).__name__}: {e}")
                continue
            rows += _bench_module(mod, name, wl, n, int(mat.nnz), build_ms, args)
            mod = None
            if device == "cuda":
                torch.cuda.empty_cache()
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workloads", nargs="+", default=["debug", "v1"])
    p.add_argument("--batch", nargs="+", type=int, default=[1, 32])
    p.add_argument("--density", nargs="+", type=float, default=[0.01, 0.5])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--compile", action="store_true", help="also time torch.compile")
    p.add_argument("--impl", nargs="*", default=None, help="substring filter")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    rows = run(args)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
