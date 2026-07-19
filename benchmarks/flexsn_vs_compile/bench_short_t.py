"""Short-T benchmark: torch.compile (unrolled) vs FlexSN vs hand-written CuPy.

An SNN neuron is a chain of elementwise ops, so for small fixed ``T`` dynamo
can unroll the time loop and inductor fuses the whole sequence into a single
kernel (state stays in registers) -- structurally the same code FlexSN emits
from its Triton template and the hand-written CuPy kernel, minus the per-
``(T, N)`` mega-compile and the fixed custom-op dispatch overhead. This script
shows that at small ``T`` the generic compiler path matches the purpose-built
kernels at inference and beats FlexSN at training.

Timing uses :func:`btorch.utils.bench.do_bench` with CUDA-event timing (L2
flushed between reps). Results are written next to the resolved figure
directory as JSON so ``plot.py`` can render Figure 1.

Usage::

    python benchmark/flexsn_vs_compile/bench_short_t.py --N 65536 --T 2 4 8 16
"""
import argparse
import json
import statistics

import torch
from kernels import cupy_impl, eager_loop, make_flexsn

from btorch.utils.bench import do_bench
from btorch.utils.file import fig_path


def stats_ms(fn) -> tuple[float, float]:
    """Mean and std of device time in ms (CUDA events, L2 flushed per rep)."""
    times = do_bench(fn, warmup=25, rep=100, return_mode="all",
                     timing_method="gpu")
    return statistics.mean(times), statistics.stdev(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1 << 20, help="neurons per step")
    ap.add_argument("--T", type=int, nargs="+", default=[2, 4, 8, 16])
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)

    flexsn = make_flexsn((args.N,), dev)
    compiled = torch.compile(eager_loop, fullgraph=True, dynamic=False)

    # (name, callable, resets_state) for the three accelerated backends
    def flexsn_call(xi, yi, v0, rho0):
        flexsn.reset()
        s1, s2 = flexsn(xi, yi)
        return s1, s2, None, None

    backends = {
        "inductor": compiled,
        "flexsn": flexsn_call,
        "cupy": cupy_impl,
    }

    print(f"N={args.N}  ({torch.cuda.get_device_name(0)}, "
          f"torch {torch.__version__})")
    print("cells are mean +/- std (ms) over 100 reps")
    hdr = (f"{'T':>4} | " + " ".join(f"{b + ' inf':>17}" for b in backends)
           + " | " + " ".join(f"{b + ' tr':>17}" for b in backends))
    print(hdr)
    print("-" * len(hdr))

    results = []
    for T in args.T:
        x = torch.randn(T, args.N, device=dev) * 2.0
        y = torch.randn(T, args.N, device=dev)
        v0 = torch.zeros(args.N, device=dev)
        rho0 = torch.zeros(args.N, device=dev)

        # correctness gate: outputs and input grads must match the reference
        def run(fn):
            xi = x.clone().requires_grad_(True)
            yi = y.clone().requires_grad_(True)
            s1, s2, *_ = fn(xi, yi, v0, rho0)
            (s1.sum() + 0.7 * s2.sum()).backward()
            return s1, s2, xi.grad, yi.grad

        ref = run(eager_loop)
        for name, fn in backends.items():
            for a, b in zip(ref, run(fn)):
                assert torch.allclose(a, b, atol=1e-4), \
                    f"{name} mismatch at T={T}"

        entry = dict(T=T, N=args.N)
        row_inf, row_tr = [], []
        for name, fn in backends.items():
            def infer(fn=fn):
                with torch.no_grad():
                    fn(x, y, v0, rho0)

            xg = x.clone().requires_grad_(True)
            yg = y.clone().requires_grad_(True)

            def train(fn=fn, xg=xg, yg=yg):
                xg.grad = None
                yg.grad = None
                s1, s2, *_ = fn(xg, yg, v0, rho0)
                (s1.sum() + s2.sum()).backward()

            inf_mean, inf_std = stats_ms(infer)
            tr_mean, tr_std = stats_ms(train)
            entry[f"{name}_inf"] = inf_mean
            entry[f"{name}_inf_std"] = inf_std
            entry[f"{name}_train"] = tr_mean
            entry[f"{name}_train_std"] = tr_std
            row_inf.append((inf_mean, inf_std))
            row_tr.append((tr_mean, tr_std))

        cells = " ".join(f"{m:>8.3f}±{s:<6.3f}" for m, s in row_inf)
        cells += " | " + " ".join(f"{m:>8.3f}±{s:<6.3f}" for m, s in row_tr)
        print(f"{T:>4} | {cells}")
        results.append(entry)

    out_file = fig_path() / "bench_short_t.json"
    out_file.write_text(json.dumps(results, indent=1))
    print("saved", out_file)


if __name__ == "__main__":
    main()
