"""Latency benchmark for sparse connection layers (PLAN section 51).

Times ``y = conn(x)`` (forward, ``torch.no_grad``) and forward+backward
(gradients w.r.t. the input and the weights of ``conn(x).square().sum()``)
for every available implementation on synthetic recurrent graphs:

- ``legacy[native]`` / ``legacy[torch_sparse]``: the pre-refactor
  ``SparseConn`` forward (a frozen copy kept in ``_legacy_baseline.py``);
- ``dense``: :class:`btorch.models.linear.DenseConn` on the same matrix (only
  up to 8,192 neurons);
- ``new[eager]``: :class:`btorch.models.connection.SparseConnection` with the
  default plan (destination-driven "pull" product);
- ``new[compile]`` (``--compile``): the same module under
  ``torch.compile(fullgraph=True)``;
- ``new[push-hint]``: built with ``Hints(expected_density=<density>)`` for
  the spike density being timed, so the planner may pick source-driven
  propagation for sparse activity. The plan that was actually chosen is
  recorded in the row (``plan``).

Run::

    python benchmarks/sparse_conn/bench_sparse_conn.py --device cuda --compile \
        --workloads debug v1 large xlarge \
        --out benchmarks/sparse_conn/results/conn_rtx5090.json

Workloads are ``name: (n_neuron, mean in-degree)``; every workload is run for
each batch size and spike density.

What is reported, per row:

- ``build_ms``: construction of the module from the SciPy matrix on the host
  plus the move to the device (conversion / preprocessing), measured once;
- ``compile_*_ms`` (``new[compile]`` only): duration of the first call of the
  compiled forward and of the first compiled forward+backward at this batch
  size, i.e. compilation, reported separately from the steady state;
- ``fwd_ms`` / ``fwd_bwd_ms``: median steady-state latency; ``*_min_ms`` is
  the fastest sample (see :func:`_common.time_calls`). The first calls,
  which build the kernel-side layouts and compile the Triton kernels, are
  warm-up and not included;
- ``fwd_peak_mb`` / ``fwd_bwd_peak_mb``: peak allocated GPU memory during the
  phase (module included); ``base_mb`` is what was allocated before it;
- ``gpu_util`` / ``gpu_other_mb``: load of the (possibly shared) GPU by other
  jobs when the module was timed;
- ``fwd_probe_us`` / ``fwd_bwd_probe_us``: latency of a trivial kernel around
  the samples (:func:`_common.contention_probe`). Only samples taken while
  the GPU was not being time-sliced with other jobs are used; a phase that
  got no such sample within ``--max-wait`` seconds is flagged
  ``*_throttled`` and its numbers are those of the scheduler.

Configurations that do not fit the memory budget (``--max-gpu-gb``) or the memory
that is currently free are skipped with a recorded reason instead of
crashing; so is everything that raises.
"""

import argparse
import gc
import json
import time
from collections.abc import Callable
from pathlib import Path

import scipy.sparse
import torch
from _common import (
    Skipped,
    add_gpu_arguments,
    best_probe,
    env_info,
    gpu_state,
    is_oom,
    make_graph,
    require_memory,
    setup_device,
    sync,
    time_calls,
)
from _legacy_baseline import LegacySparseConn, available_legacy_backends

from btorch.models.connection import SparseConnection
from btorch.models.linear import DenseConn
from btorch.sparse import Hints


WORKLOADS = {
    "debug": (256, 16),
    "v1": (4096, 100),  # ~4k-neuron recurrent column
    "large": (100_000, 100),
    "xlarge": (1_000_000, 50),
}

# Largest network for which the dense reference is built.
DENSE_MAX_NEURONS = 8192

# Implementations whose execution plan depends on the spike density: their
# hints are refreshed for every density that is timed (see ``_bench_module``).
PUSH_HINT = "new[push-hint]"
COMPILED = "new[compile]"


OOM = "skipped: exceeds memory cap (out of memory inside this process)"


def estimate_bytes(name: str, n: int, nnz: int, batch: int) -> float:
    """Rough upper estimate of the device memory one configuration needs.

    Used only to decide whether a configuration is attempted on a shared
    GPU; deliberately pessimistic.
    """
    io = 40.0 * batch * n  # input, output, their gradients, loss temporaries
    if name == "dense":
        return 16.0 * n * n + io  # weight, its gradient, build temporaries
    if name == "legacy[native]":
        # COO indices and values, the per-call COO/CSR temporaries, and the
        # [nnz, batch] products of the sparse-matrix gradient.
        return 60.0 * nnz + 12.0 * nnz * batch + io
    if name == "legacy[torch_sparse]":
        # ``spmm`` gathers [nnz, batch] products; autograd keeps them.
        return 24.0 * nnz + 24.0 * nnz * batch + io
    # SparseConnection: canonical edges, two CSR layouts with permutations
    # (int64), the kernels' int32 copies, values and their gradient.
    return 90.0 * nnz + io


def _builders(n: int, device: str, compile_modes: bool) -> dict[str, Callable]:
    """Map implementation name -> ``f(scipy matrix) -> module`` (on
    ``device``)."""
    out: dict[str, Callable] = {}
    # Pre-refactor baseline (frozen copy of the removed ``SparseConn``).
    for backend in available_legacy_backends():
        out[f"legacy[{backend}]"] = lambda m, b=backend: LegacySparseConn(
            m, backend=b, device=device
        )
    if n <= DENSE_MAX_NEURONS:
        out["dense"] = lambda m: DenseConn(
            n,
            n,
            weight=torch.tensor(m.toarray(), device=device),
            device=device,
        )
    out["new[eager]"] = lambda m: SparseConnection.from_adjacency(m).to(device)
    if compile_modes:
        out[COMPILED] = lambda m: SparseConnection.from_adjacency(m).to(device)
    out[PUSH_HINT] = lambda m: SparseConnection.from_adjacency(
        m, hints=Hints(expected_density=0.01)
    ).to(device)
    return out


def _timed_first_call(fn: Callable, device: str) -> float:
    sync(device)
    t0 = time.perf_counter()
    fn()
    sync(device)
    return (time.perf_counter() - t0) * 1e3


def _phase(fn: Callable, device: str, args) -> dict:
    """Time one phase; returns ``{median, min, peak_mb, base_mb}`` or an
    ``error``."""
    out: dict = {}
    try:
        for _ in range(3):  # build layouts, compile kernels
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            out["base_mb"] = torch.cuda.memory_allocated() / 2**20
        out.update(time_calls(fn, device, repeats=args.repeats, max_wait=args.max_wait))
        if device == "cuda":
            out["peak_mb"] = torch.cuda.max_memory_allocated() / 2**20

    except Exception as e:  # report, keep going
        out["error"] = OOM if is_oom(e) else f"{type(e).__name__}: {str(e)[:80]}"
        if device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
    return out


def _bench_module(mod, name, wl, n, nnz, build_ms, args) -> list[dict]:
    """Time one module over every batch size and spike density."""
    device = args.device
    rows = []
    params = [p for p in mod.parameters() if p.requires_grad]
    run = mod
    if name == COMPILED:
        # All ``SparseConnection`` modules share one code object, so the
        # compiled-graph cache of earlier workloads (other shapes, other
        # modules) must not count against this module's recompile limit.
        torch._dynamo.reset()
        # Compile from scratch: the on-disk graph caches would hide the
        # compile time, and they are not keyed on the backward formula of a
        # custom operator (a cache written by an older btorch replays the
        # old backward graph).
        torch._inductor.config.force_disable_caches = True
        run = torch.compile(mod, fullgraph=True)
    for batch in args.batch:
        compiled_this_batch = False
        for density in args.density:
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
            rows.append(row)
            state = gpu_state()
            row["gpu_util"] = state.get("util")
            row["gpu_other_mb"] = state.get("other_mb")
            try:
                require_memory(
                    estimate_bytes(name, n, nnz, batch), device, args.max_gb, name
                )
            except Skipped as e:
                row["skipped"] = str(e)
                print(f"{wl:7s} {name:22s} B={batch:<3d} p={density:<5g} SKIPPED: {e}")
                continue

            g = torch.Generator(device="cpu").manual_seed(0)
            x = (torch.rand(batch, n, generator=g) < density).float()
            x = x.to(device).requires_grad_(True)
            if name == PUSH_HINT:
                # One module serves every density: tell the planner which
                # density is about to be timed, exactly as if the module had
                # been built with ``hints=Hints(expected_density=density)``.
                mod.set_hints(Hints(expected_density=density))
            if isinstance(mod, SparseConnection):
                row["plan"] = f"{mod._plan.algorithm}/{mod._plan.backend}"

            def fwd(run=run, x=x):
                with torch.no_grad():
                    run(x)

            def fwd_bwd(run=run, x=x):
                torch.autograd.grad(run(x).square().sum(), [x, *params])

            # Forward and backward are timed separately so that a backward
            # failure (e.g. out of memory) still reports the forward latency.
            for key, fn in (("fwd", fwd), ("fwd_bwd", fwd_bwd)):
                if (
                    key == "fwd_bwd"
                    and name == "legacy[native]"
                    and device == "cuda"
                    and 4.0 * n * n > args.max_gb * 2**30
                ):
                    # Known footprint: the gradient of ``torch.sparse.mm``
                    # w.r.t. the sparse operand asks for N x N floats.
                    row["fwd_bwd_error"] = (
                        f"skipped: backward needs N x N floats "
                        f"({4.0 * n * n / 2**30:.0f} GB), exceeds memory cap"
                    )
                    continue
                if name == COMPILED and not compiled_this_batch:
                    try:
                        row[f"compile_{key}_ms"] = _timed_first_call(fn, device)
                    except Exception as e:
                        row[f"{key}_error"] = f"{type(e).__name__}: {str(e)[:80]}"
                        continue
                res = _phase(fn, device, args)
                if "error" in res:
                    row[f"{key}_error"] = res["error"]
                    continue
                row[f"{key}_ms"] = res["median"]
                row[f"{key}_min_ms"] = res["min"]
                row[f"{key}_peak_mb"] = res["peak_mb"] if device == "cuda" else None
                row[f"{key}_inner"] = res["inner"]
                row[f"{key}_probe_us"] = res["probe_us"]
                row[f"{key}_throttled"] = res["throttled"]
                row[f"{key}_samples"] = res["repeats"]
                row.setdefault("base_mb", res.get("base_mb"))
            compiled_this_batch = True
            nan = float("nan")
            print(
                f"{wl:7s} {name:22s} B={batch:<3d} p={density:<5g} "
                f"fwd {row.get('fwd_ms', nan):9.3f} "
                f"(min {row.get('fwd_min_ms', nan):8.3f}) ms  "
                f"fwd+bwd {row.get('fwd_bwd_ms', nan):9.3f} "
                f"(min {row.get('fwd_bwd_min_ms', nan):8.3f}) ms  "
                f"peak {row.get('fwd_bwd_peak_mb') or nan:8.1f} MB  "
                f"build {build_ms:.0f} ms  util {row['gpu_util']}"
                + (f"  [{row['plan']}]" if "plan" in row else "")
                + "".join(f"  [{k}: {v}]" for k, v in row.items() if "error" in k),
                flush=True,
            )
            del x, fwd, fwd_bwd
    return rows


def check_parity(mods: dict, n: int, device: str) -> dict[str, float]:
    """Largest relative deviation of every module from the first one.

    All implementations must compute the same ``x @ W``; a benchmark of
    modules that disagree would be meaningless.
    """
    g = torch.Generator(device="cpu").manual_seed(1)
    x = torch.randn(3, n, generator=g).to(device)
    ref, out = None, {}
    with torch.no_grad():
        for name, mod in mods.items():
            y = mod(x)
            if ref is None:
                ref = y
            out[name] = float((y - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
    return out


def run(args: argparse.Namespace) -> list[dict]:
    rows = []
    device = args.device
    for wl in args.workloads:
        n, indegree = WORKLOADS[wl]
        t0 = time.perf_counter()
        mat: scipy.sparse.coo_array = make_graph(n, indegree)
        nnz = int(mat.nnz)
        print(
            f"--- {wl}: n={n} nnz={nnz} "
            f"(graph generated in {time.perf_counter() - t0:.1f} s)",
            flush=True,
        )
        builders = _builders(n, device, args.compile)
        parity_ref = None
        for name, build in builders.items():
            if args.impl and not any(k in name for k in args.impl):
                continue
            mod = None
            try:
                require_memory(
                    estimate_bytes(name, n, nnz, min(args.batch)),
                    device,
                    args.max_gb,
                    name,
                )
                sync(device)
                t0 = time.perf_counter()
                mod = build(mat)
                sync(device)
                build_ms = (time.perf_counter() - t0) * 1e3
                if device == "cuda":
                    # Return the temporaries of the build to the device: on
                    # a shared GPU they may be all the memory there is.
                    gc.collect()
                    torch.cuda.empty_cache()
            except Skipped as e:
                reason = str(e)
            except Exception as e:  # report, keep going
                reason = OOM if is_oom(e) else f"{type(e).__name__}: {str(e)[:80]}"
                reason = f"build failed: {reason}"
            if mod is None:
                print(f"{wl:7s} {name:22s} SKIPPED: {reason}", flush=True)
                rows += [
                    {
                        "workload": wl,
                        "n": n,
                        "nnz": nnz,
                        "impl": name,
                        "batch": b,
                        "density": d,
                        "device": device,
                        "skipped": reason,
                    }
                    for b in args.batch
                    for d in args.density
                ]
            else:
                # Every implementation against the first one that was built.
                try:
                    pair = {"ref": parity_ref, name: mod} if parity_ref else {name: mod}
                    parity = check_parity(pair, n, device)[name]
                    if parity_ref is None and n <= 100_000:
                        parity_ref = mod
                except Exception as e:
                    parity = float("nan")
                    print(f"{wl:7s} {name:22s} parity check failed: {e}")
                if parity > 1e-3:
                    raise RuntimeError(
                        f"{name} disagrees with the reference by {parity:.2e}"
                    )
                new = _bench_module(mod, name, wl, n, nnz, build_ms, args)
                for r in new:
                    r["parity_rel_err"] = parity
                rows += new
                if mod is not parity_ref:
                    del mod
            if device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
        parity_ref = None
        del mat
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        if args.out is not None:  # keep partial results of long runs
            _write(args, rows)
    return rows


def _write(args: argparse.Namespace, rows: list[dict]) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"env": args.env, "args": args.cli, "rows": rows}
    payload["best_probe_us"] = best_probe()
    args.out.write_text(json.dumps(payload, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_gpu_arguments(p)
    p.add_argument("--workloads", nargs="+", default=["debug", "v1"])
    p.add_argument("--batch", nargs="+", type=int, default=[1, 32])
    p.add_argument("--density", nargs="+", type=float, default=[0.01, 0.5])
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--compile", action="store_true", help="also time torch.compile")
    p.add_argument("--impl", nargs="*", default=None, help="substring filter")
    p.add_argument(
        "--max-wait",
        type=float,
        default=20.0,
        help="extra seconds spent per measurement waiting for clean samples",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    setup_device(args)
    args.cli = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    args.env = env_info(args.device)
    args.env["gpu_selection"] = args.gpu
    rows = run(args)
    if args.out is not None:
        _write(args, rows)


if __name__ == "__main__":
    main()
