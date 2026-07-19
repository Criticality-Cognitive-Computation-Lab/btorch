"""Timestep sweep: FlexSN vs inductor compile strategies, ``T`` up to 1024.

Compilation strategy, not kernel arithmetic, decides how a spiking neuron
scales with sequence length. This sweep pits FlexSN's Triton template against
several ``torch.compile`` lowerings and CUDA-graph capture:

Variants:
    flexsn              spikingjelly FlexSN (Triton). ``tl.static_range``
                        unrolls the time loop, so Triton compile grows ~quad-
                        ratically; capped at ``--max-flexsn-T`` (default 64).
    unrolled            ``torch.compile(loop, fullgraph, dynamic=False)``:
                        dynamo unrolls ``T``, inductor fuses. Capped at
                        ``--max-unrolled-T`` (compile cost / fragmentation).
    unrolled_cg         same, ``mode="reduce-overhead"`` (auto CUDA graphs).
    unrolled_manual_cg  default-mode compiled fn captured into a manual
                        ``torch.cuda.CUDAGraph`` and timed as ``replay()``.
    scan                ``torch.compile`` of the ``scan`` HOP: host while-loop,
                        one fused launch per step. Compiles once (dims go
                        dynamic), runs at any ``T``.
    scan_manual_cg      the scan host loop captured into a CUDA graph -- tests
                        whether scan's cost is launch overhead or memory
                        traffic.
    cupy                hand-written CuPy float4 kernel (:class:`kernels.
                        ComplicatedLIFNode`); compiles once, runs at any ``T``.

Timing uses :func:`btorch.utils.bench.do_bench` (median, CUDA-event timed, L2
flushed). Training is forward + ``torch.autograd.grad`` w.r.t. ``(x, y)`` of the
summed spike trains. Every entry (including infeasible ones, which carry a
status string) is appended to the JSON results file so partial sweeps remain
plottable.

Usage::

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        python benchmark/flexsn_vs_compile/bench_sweep.py --N 32768
"""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from kernels import ALPHA, atan_sg, cupy_impl, make_core
from torch._higher_order_ops.scan import scan

from btorch.utils.bench import do_bench
from btorch.utils.file import fig_path


_core = make_core(atan_sg)  # pure-torch straight-through ATan core


def loop(x_seq, y_seq, v0, rho0):
    """Unrolled Python time loop (dynamo unrolls it under torch.compile)."""
    v, rho = v0, rho0
    s1_l, s2_l = [], []
    for t in range(x_seq.shape[0]):
        s1, s2, v, rho = _core(x_seq[t], y_seq[t], v, rho)
        s1_l.append(s1)
        s2_l.append(s2)
    return torch.stack(s1_l), torch.stack(s2_l), v, rho


def _scan_combine(carry, xs):
    v, rho = carry
    x, y = xs
    s1, s2, v, rho = _core(x, y, v, rho)
    return (v, rho), (s1.clone(), s2.clone())


def scan_loop(x_seq, y_seq, v0, rho0):
    (v, rho), (s1, s2) = scan(_scan_combine, (v0, rho0), (x_seq, y_seq))
    return s1, s2, v, rho


def scan_wrapper(compiled):
    """Force ``requires_grad`` on the init carry outside the compiled region.

    torch 2.11 ``scan`` autograd silently drops cross-step carry gradients
    when the init carry has ``requires_grad=False``; ``requires_grad_()`` is
    not dynamo-traceable, so it must be applied before entering the graph.
    """
    def f(x_seq, y_seq, v0, rho0):
        if torch.is_grad_enabled() and x_seq.requires_grad:
            v0 = v0.detach().requires_grad_(True)
            rho0 = rho0.detach().requires_grad_(True)
        return compiled(x_seq, y_seq, v0, rho0)
    return f


def bench(fn):
    """Mean, std device time (ms) for a plain callable, L2 flushed per rep."""
    times = do_bench(fn, warmup=25, rep=100, return_mode="all",
                     timing_method="gpu")
    return statistics.mean(times), statistics.stdev(times)


def bench_cudagraph(fn):
    """Mean, std device time (ms) for ``fn`` replayed from a CUDA graph.

    ``btorch.utils.bench.do_bench`` has no CUDA-graph path, so we use triton's
    ``do_bench_cudagraph``: it captures ``fn`` into a graph (unrolling many
    calls to amortize host launch overhead) and times replays -- the correct
    way to measure graph-captured work.
    """
    from triton.testing import do_bench_cudagraph

    times = do_bench_cudagraph(fn, rep=100, return_mode="all")
    return statistics.mean(times), statistics.stdev(times)


def save(results_file: Path, entry: dict) -> None:
    try:
        results = json.loads(results_file.read_text())
    except FileNotFoundError:
        results = []
    results.append(entry)
    results_file.write_text(json.dumps(results, indent=1))


def run_variant(results_file, name, T, N, make_fn, do_manual_cg=False):
    """Bench one variant. ``make_fn(train)`` -> zero-arg fwd(+bwd) callable."""
    entry = dict(variant=name, T=T, N=N)
    for mode in ("inf", "train"):
        key_ms, key_std, key_s = f"{mode}_ms", f"{mode}_ms_std", f"first_{mode}_s"
        if do_manual_cg and mode == "train":
            # torch 2.11: capturing a compiled fwd+bwd invalidates the capture
            # and poisons CUDA RNG state for the rest of the process. Verified
            # empirically; reduce-overhead is the supported cudagraph-training
            # route.
            entry["train_status"] = ("unsupported: fwd+bwd graph capture "
                                     "invalidated (torch 2.11)")
            print(f"  {name:22s} train T={T:5d}: skipped (capture unsupported)",
                  flush=True)
            continue
        fn = None
        try:
            fn = make_fn(mode == "train")
            t0 = time.time()
            fn()
            torch.cuda.synchronize()
            entry[key_s] = time.time() - t0
            mean, std = bench_cudagraph(fn) if do_manual_cg else bench(fn)
            entry[key_ms] = mean
            entry[key_std] = std
            print(f"  {name:22s} {mode:5s} T={T:5d}: {mean:8.3f} +/- {std:6.3f} "
                  f"ms (first call {entry[key_s]:.1f}s)", flush=True)
        except torch.cuda.OutOfMemoryError:
            entry[f"{mode}_status"] = "OOM"
            print(f"  {name:22s} {mode:5s} T={T:5d}: OOM", flush=True)
        except Exception as e:  # noqa: BLE001 - record any backend failure
            entry[f"{mode}_status"] = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"  {name:22s} {mode:5s} T={T:5d}: FAILED "
                  f"{entry[f'{mode}_status']}", flush=True)
        del fn
        gc.collect()
        torch.cuda.empty_cache()
    save(results_file, entry)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1 << 15, help="neurons per step")
    ap.add_argument("--T", type=int, nargs="+", default=[4, 16, 64, 256, 1024])
    ap.add_argument("--max-flexsn-T", type=int, default=64)
    ap.add_argument("--max-unrolled-T", type=int, default=256)
    ap.add_argument("--append", action="store_true",
                    help="keep existing results file (resume a partial sweep)")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    # Many static-shape compiles of the same `loop` code object across the T
    # sweep; do NOT dynamo-reset (it would wipe scan's dynamic cache too).
    torch._dynamo.config.cache_size_limit = 64

    results_file = fig_path() / "bench_sweep.json"
    if not args.append or not results_file.exists():
        results_file.write_text("[]")

    from spikingjelly.activation_based import surrogate
    from spikingjelly.activation_based.neuron.flexsn import FlexSN

    flexsn = FlexSN(core=make_core(surrogate.ATan(alpha=ALPHA)),
                    num_inputs=2, num_states=2, num_outputs=2,
                    example_inputs=tuple(torch.zeros(args.N, device=dev)
                                         for _ in range(4)),
                    backend="triton")
    scan_c = torch.compile(scan_loop, fullgraph=True)  # shared: goes dynamic

    for T in args.T:
        print(f"== T={T} N={args.N} ==", flush=True)
        try:
            x = torch.randn(T, args.N, device=dev) * 2.0
        except torch.cuda.OutOfMemoryError:
            print(f"  block T={T} skipped: OOM allocating inputs", flush=True)
            for v in ("flexsn", "unrolled", "unrolled_cg",
                      "unrolled_manual_cg", "scan", "scan_manual_cg"):
                save(results_file, dict(variant=v, T=T, N=args.N,
                                        inf_status="OOM allocating inputs",
                                        train_status="OOM allocating inputs"))
            continue
        y = torch.randn(T, args.N, device=dev)
        v0 = torch.zeros(args.N, device=dev)
        rho0 = torch.zeros(args.N, device=dev)
        xg = x.clone().requires_grad_(True)
        yg = y.clone().requires_grad_(True)

        # bind the per-iteration tensors as defaults so each `maker` closure
        # captures this block's tensors explicitly (not by late reference)
        def maker(callee, reset=False, *, x=x, y=y, xg=xg, yg=yg,
                  v0=v0, rho0=rho0):
            def make_fn(train):
                if train:
                    def fn():
                        if reset:
                            flexsn.reset()
                        out = (callee(xg, yg) if reset
                               else callee(xg, yg, v0, rho0))
                        s1, s2 = out[0], out[1]
                        torch.autograd.grad(s1.sum() + s2.sum(), [xg, yg])
                else:
                    def fn():
                        with torch.no_grad():
                            if reset:
                                flexsn.reset()
                                callee(x, y)
                            else:
                                callee(x, y, v0, rho0)
                return fn
            return make_fn

        # -- flexsn --
        if T <= args.max_flexsn_T:
            run_variant(results_file, "flexsn", T, args.N,
                        maker(flexsn, reset=True))
        else:
            save(results_file, dict(
                variant="flexsn", T=T, N=args.N,
                inf_status="skipped: static_range compile infeasible",
                train_status="skipped: static_range compile infeasible"))
            print(f"  flexsn skipped at T={T} (compile infeasible)", flush=True)

        # -- unrolled inductor variants --
        if T <= args.max_unrolled_T:
            unrolled = torch.compile(loop, fullgraph=True, dynamic=False)
            run_variant(results_file, "unrolled", T, args.N, maker(unrolled))
            unrolled_cg = torch.compile(loop, fullgraph=True, dynamic=False,
                                        mode="reduce-overhead")
            run_variant(results_file, "unrolled_cg", T, args.N,
                        maker(unrolled_cg))
            # manual capture last: a failed capture can poison allocator state
            run_variant(results_file, "unrolled_manual_cg", T, args.N,
                        maker(unrolled), do_manual_cg=True)
            del unrolled, unrolled_cg
        else:
            for v in ("unrolled", "unrolled_manual_cg", "unrolled_cg"):
                save(results_file, dict(
                    variant=v, T=T, N=args.N,
                    inf_status="skipped: unroll compile infeasible",
                    train_status="skipped: unroll compile infeasible"))
            print(f"  unrolled variants skipped at T={T}", flush=True)

        # -- scan variants (never reset dynamo here; scan_c stays cached) --
        scan_fn = scan_wrapper(scan_c)
        run_variant(results_file, "scan", T, args.N, maker(scan_fn))
        run_variant(results_file, "scan_manual_cg", T, args.N,
                    maker(scan_fn), do_manual_cg=True)

        # -- hand-written CuPy kernel (compiles once, runs at any T) --
        run_variant(results_file, "cupy", T, args.N, maker(cupy_impl))

        del x, y, xg, yg
        gc.collect()
        torch.cuda.empty_cache()

    print("done ->", results_file, flush=True)


if __name__ == "__main__":
    main()
