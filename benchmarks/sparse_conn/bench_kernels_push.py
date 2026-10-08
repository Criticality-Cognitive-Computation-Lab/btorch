"""Kernel-level benchmark of spike propagation: push (Triton, ATen) against
pull.

Times, on raw buffers (no modules, custom operators or autograd):

- ``pack``: ``kernels_aten.pack_spikes`` (host ``nonzero``; synchronises).
- ``aten``: ``pack`` + ``kernels_aten.spike_push``.
- ``packed``: ``pack`` + the packed Triton kernel (thread per gathered edge).
- ``packed_k``: the packed Triton kernel alone, on indices packed beforehand.
- ``auto``: ``pack`` + ``kernels_triton_push.spike_push`` as registered (it
  uses the dense kernel for small or dense inputs).
- ``dense``: ``kernels_triton_push.spike_push_dense`` (no pack at all).
- ``dense_sm``: the same with source-major values (no ``t_perm`` gather).
- ``pull``: ``kernels_aten.csr_matvec`` (``torch.sparse`` CSR product).
- ``pull_tri``: ``kernels_triton.csr_matvec`` when that module is importable.

Run::

    python benchmarks/sparse_conn/bench_kernels_push.py \
        --out benchmarks/sparse_conn/results/kernels_push_rtx5090.json

Workloads are ``name: (n_neuron, mean out-degree)``. Out-degrees are heavy
tailed (sources drawn with log-normal weights, sigma 1), targets uniform.

Timing: calls are queued back to back and the device is synchronised once per
repeat, so a figure is the steady-state cost per call including the Python
launch overhead (and, for the packed paths, the synchronisation inside
``nonzero``). On a shared GPU another process's time slices inflate
everything for seconds at a time, so the whole table is measured in several
passes and the *minimum* per figure is reported (many short passes, e.g.
``--passes 20 --target-s 0.01 --repeats 3``, sample more quiet windows).
Every measurement is followed by a trivial calibration call; a cell whose
calibration never came close to the best one seen is flagged ``*``.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from btorch.sparse.runtime import kernels_aten, kernels_triton_push as ktp
from btorch.sparse.runtime.backend import KernelCache
from btorch.sparse.runtime.cache import RepresentationCache


try:  # Optional: the Triton pull kernel, as a second pull baseline.
    from btorch.sparse.runtime import kernels_triton
except ImportError:  # pragma: no cover
    kernels_triton = None


WORKLOADS = {
    "4k": (4096, 100),
    "100k": (100_000, 100),
    "1M": (1_000_000, 50),
}

# Skip the ATen push when its ``W``-sized int64 temporaries (about six of
# them) would not fit the memory share of this benchmark.
_ATEN_MAX_WORK = 100_000_000


def make_graph(n: int, degree: int, device: str):
    """Random graph with heavy-tailed out-degree, as kernel buffers."""
    gen = torch.Generator(device=device).manual_seed(0)
    n_edge = n * degree
    weight = torch.exp(torch.randn(n, device=device, generator=gen))
    src = torch.multinomial(weight, n_edge, replacement=True, generator=gen)
    dst = torch.randint(0, n, (n_edge,), device=device, generator=gen)
    cache = RepresentationCache().to(device)
    cache.build(dst, src, (n, n), 0)
    del src, dst, weight
    values = torch.randn(n_edge, device=device, generator=gen)
    torch.cuda.empty_cache()
    return cache, values


def time_call(fn, *, target_s: float = 0.05, repeats: int = 9) -> float:
    """Best seconds per call of ``fn`` over ``repeats`` batches of calls."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    once = time.perf_counter() - start
    iters = max(2, min(2000, int(target_s / max(once, 1e-6))))
    best = math.inf
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - start) / iters)
    return best


def calibrate(device: str) -> float:
    """Seconds per trivial GPU call; grows when the device is time sliced."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(200):
        torch.zeros(1, 4096, device=device)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / 200


def measure_cell(cache, values, values_sm, x, kernel_cache, args) -> dict:
    """Microseconds per call of every path on one input."""
    n = x.shape[-1]
    push_args = (cache.t_crow, cache.t_col, cache.t_perm, values, x)
    active_idx, ptr = kernels_aten.pack_spikes(x)
    layout = ktp._layout(kernel_cache, cache.t_crow, cache.t_col, cache.t_perm)
    work = int(layout.degree[active_idx].sum())

    def aten():
        idx, ptr = kernels_aten.pack_spikes(x)
        return kernels_aten.spike_push(*push_args, idx, ptr, n)

    def never_dense(*_):
        return False

    def triton(force_packed: bool):
        def call():
            ktp._prefer_dense = never_dense if force_packed else default
            idx, ptr = kernels_aten.pack_spikes(x)
            return ktp.spike_push(*push_args, idx, ptr, n, cache=kernel_cache)

        return call

    def packed_kernel():
        ktp._prefer_dense = never_dense
        return ktp.spike_push(*push_args, active_idx, ptr, n, cache=kernel_cache)

    default = ktp._prefer_dense
    paths = {
        "pack": lambda: kernels_aten.pack_spikes(x),
        "packed": triton(True),
        "packed_k": packed_kernel,
        "auto": triton(False),
        "dense": lambda: ktp.spike_push_dense(*push_args, n, cache=kernel_cache),
        "dense_sm": lambda: ktp.spike_push_dense(
            cache.t_crow,
            cache.t_col,
            cache.t_perm,
            values_sm,
            x,
            n,
            source_major_values=True,
            cache=kernel_cache,
        ),
        "pull": lambda: kernels_aten.csr_matvec(cache.crow, cache.col, values, x),
    }
    if work <= _ATEN_MAX_WORK:
        paths["aten"] = aten
    if kernels_triton is not None:
        paths["pull_tri"] = lambda: kernels_triton.csr_matvec(
            cache.crow, cache.col, values, x
        )
    row = {}
    try:
        for name, fn in paths.items():
            if name in args.skip:
                continue
            try:
                seconds = time_call(fn, target_s=args.target_s, repeats=args.repeats)
                row[name] = seconds * 1e6
            except torch.OutOfMemoryError:
                # Shared device: another process may hold most of the memory.
                torch.cuda.empty_cache()
    finally:
        ktp._prefer_dense = default
    row["calibration"] = calibrate(x.device.type) * 1e6
    row["work"] = work

    # Accuracy against a float64 pull product, relative to the largest output,
    # and the run-to-run spread caused by the order of the atomics.
    exact = kernels_aten.csr_matvec(cache.crow, cache.col, values.double(), x.double())
    scale = float(exact.abs().max().clamp_min(1e-30))
    ktp._prefer_dense = never_dense
    try:
        idx, ptr = kernels_aten.pack_spikes(x)
        packed = [
            ktp.spike_push(*push_args, idx, ptr, n, cache=kernel_cache)
            for _ in range(2)
        ]
    finally:
        ktp._prefer_dense = default
    dense = ktp.spike_push_dense(*push_args, n, cache=kernel_cache)
    pull = kernels_aten.csr_matvec(cache.crow, cache.col, values, x)
    row["err_packed"] = float((packed[0] - exact).abs().max()) / scale
    row["err_dense"] = float((dense - exact).abs().max()) / scale
    row["err_pull"] = float((pull - exact).abs().max()) / scale
    row["rerun_packed"] = float((packed[0] - packed[1]).abs().max()) / scale
    return row


def crossover(densities: list[float], push: list[float], pull: list[float]):
    """Density at which ``push`` stops beating ``pull`` (log interpolation).

    Returns ``None`` when push wins at every measured density, ``0.0`` when it
    never wins.
    """
    ratio = [math.log(a / b) for a, b in zip(push, pull, strict=True)]
    if ratio[0] >= 0:
        return 0.0
    for i in range(1, len(ratio)):
        if ratio[i] >= 0:
            t = -ratio[i - 1] / (ratio[i] - ratio[i - 1])
            lo, hi = math.log(densities[i - 1]), math.log(densities[i])
            return math.exp(lo + t * (hi - lo))
    return None


TIMED = ("pack", "aten", "packed", "packed_k", "auto", "dense", "dense_sm")
PULLS = ("pull", "pull_tri")


def bench_workload(name: str, args, device: str) -> list[dict]:
    n, degree = WORKLOADS[name]
    cache, values = make_graph(n, degree, device)
    values_sm = values[cache.t_perm]
    kernel_cache = KernelCache()
    max_degree = int((cache.t_crow[1:] - cache.t_crow[:-1]).max())
    cells: dict[tuple[int, float], dict] = {}
    for _ in range(args.passes):
        for batch in args.batches:
            for density in args.densities:
                # The same input in every pass.
                gen = torch.Generator(device=device)
                gen.manual_seed(batch * 1000 + round(density * 1e4))
                noise = torch.rand(batch, n, device=device, generator=gen)
                x = (noise < density).float()
                del noise
                row = measure_cell(cache, values, values_sm, x, kernel_cache, args)
                del x
                best = cells.setdefault((batch, density), row)
                for key in (*TIMED, *PULLS, "calibration"):
                    if key in row:
                        best[key] = min(best.get(key, math.inf), row[key])
    floor = min(c["calibration"] for c in cells.values())
    columns = [c for c in (*TIMED, *PULLS) if any(c in r for r in cells.values())]
    print(f"\n{name}: N={n} E={n * degree} max out-degree {max_degree}")
    print(
        f"{'B':>3s} {'density':>8s} "
        + " ".join(f"{c:>9s}" for c in columns)
        + "   err packed/dense/pull, rerun   [us per call]"
    )
    results = []
    for batch in args.batches:
        rows = [cells[batch, density] for density in args.densities]
        for density, row in zip(args.densities, rows, strict=True):
            row.update(workload=name, n=n, batch=batch, density=density)
            row["contended"] = row["calibration"] > 1.5 * floor
            print(
                f"{batch:3d} {density:8.3f} "
                + " ".join(
                    f"{row[c]:9.1f}" if c in row else f"{'-':>9s}" for c in columns
                )
                + f"   {row['err_packed']:.0e}/{row['err_dense']:.0e}/"
                f"{row['err_pull']:.0e}, {row['rerun_packed']:.0e}"
                + (" *" if row["contended"] else ""),
                flush=True,
            )
        results += rows
        for pull in PULLS:
            for path in ("auto", "dense", "dense_sm"):
                if any(pull not in r or path not in r for r in rows):
                    continue
                cross = crossover(
                    args.densities, [r[path] for r in rows], [r[pull] for r in rows]
                )
                text = "above the range" if cross is None else f"{cross:.3f}"
                print(f"    crossover density {path:>8s} vs {pull:>8s}: {text}")
                results.append(
                    {
                        "workload": name,
                        "batch": batch,
                        "path": path,
                        "pull": pull,
                        "crossover": cross,
                    }
                )
    del cache, values, values_sm
    torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workloads", nargs="+", default=["4k", "100k", "1M"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument(
        "--densities",
        nargs="+",
        type=float,
        default=[0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    )
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--target-s", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--skip", nargs="*", default=[], help="paths not to time")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = "cuda"
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__)
    results = []
    for name in args.workloads:
        results += bench_workload(name, args, device)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        meta = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
        args.out.write_text(json.dumps({"meta": meta, "results": results}, indent=1))


if __name__ == "__main__":
    main()
