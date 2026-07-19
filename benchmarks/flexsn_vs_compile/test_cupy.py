"""Rigorous correctness + benchmark for the CuPy ComplicatedLIFNode.

Correctness: compares the CuPy backend against the eager reference
(`eager_loop` / autograd) on forward outputs, final states, and *all* input
gradients, across several shapes -- including N not divisible by 4 (exercises
the float4 edge path), nonzero initial states, and gradients seeded on the
returned final states.

Benchmark: triton.testing.do_bench median for inference and training vs the
unrolled torch.compile baseline.

Run: python benchmark/flexsn_vs_compile/test_cupy.py
"""
import torch
import triton.testing
from kernels import ComplicatedLIFNode, core_step, cupy_impl, eager_loop


def _eager_with_states(x, y, v0, rho0):
    """Reference forward returning stacked spikes AND final states."""
    T = x.shape[0]
    v, rho = v0, rho0
    s1_l, s2_l = [], []
    for t in range(T):
        s1, s2, v, rho = core_step(x[t], y[t], v, rho)
        s1_l.append(s1)
        s2_l.append(s2)
    return torch.stack(s1_l), torch.stack(s2_l), v, rho


def check(T, N, seed=0, nonzero_init=False, seed_state_grads=False):
    torch.manual_seed(seed)
    dev = "cuda"
    x = torch.randn(T, N, device=dev) * 2.0
    y = torch.randn(T, N, device=dev)
    v0 = (torch.randn(N, device=dev) if nonzero_init
          else torch.zeros(N, device=dev))
    rho0 = (torch.randn(N, device=dev).abs() if nonzero_init
            else torch.zeros(N, device=dev))

    node = ComplicatedLIFNode(store_state_seqs=True)

    def run(fn):
        xi = x.clone().requires_grad_(True)
        yi = y.clone().requires_grad_(True)
        vi = v0.clone().requires_grad_(True)
        ri = rho0.clone().requires_grad_(True)
        s1, s2, vf, rhof = fn(xi, yi, vi, ri)
        # weight the loss over time + seed grads on final states so every
        # gradient path is exercised
        w = torch.linspace(0.5, 1.5, T, device=dev).unsqueeze(1)
        loss = (s1 * w).sum() + (s2 * 0.7 * w).sum()
        if seed_state_grads:
            loss = loss + vf.sum() * 0.3 + rhof.sum() * 0.2
        loss.backward()
        return (s1, s2, vf, rhof, xi.grad, yi.grad, vi.grad, ri.grad)

    ref = run(_eager_with_states)
    got = run(node)

    names = ["s1", "s2", "v_final", "rho_final",
             "grad_x", "grad_y", "grad_v0", "grad_rho0"]
    worst = 0.0
    fails = []
    for nm, a, b in zip(names, ref, got):
        # a None grad means "no gradient path" (e.g. y is unused at T=1); treat
        # it as an explicit zero so it compares against the kernel's 0 output
        if a is None and b is None:
            continue
        if a is None:
            a = torch.zeros_like(b)
        if b is None:
            b = torch.zeros_like(a)
        denom = a.abs().max().item() + 1e-9
        rel = (a - b).abs().max().item() / denom
        worst = max(worst, rel)
        if rel > 2e-4:
            fails.append(f"{nm} rel={rel:.2e}")

    # inference path (no_grad) takes the save_residuals=0 branch; verify its
    # outputs directly -- the grad path above never exercises it
    with torch.no_grad():
        s1_inf, s2_inf, _, _ = node(x, y, v0, rho0)
    for nm, a, b in [("s1_inf", ref[0], s1_inf), ("s2_inf", ref[1], s2_inf)]:
        rel = (a - b).abs().max().item() / (a.abs().max().item() + 1e-9)
        worst = max(worst, rel)
        if rel > 2e-4:
            fails.append(f"{nm} rel={rel:.2e}")
    tag = (f"T={T:<4} N={N:<7} "
           f"init={'rand' if nonzero_init else 'zero'} "
           f"state_grads={int(seed_state_grads)}")
    print(f"  [{'OK ' if not fails else 'FAIL'}] {tag}  worst_rel={worst:.2e}"
          + ("  " + "; ".join(fails) if fails else ""))
    return not fails


def bench():
    dev = "cuda"
    N = 32768
    print("\n== benchmark (median us, do_bench) ==")
    print(f"{'T':>4} | {'cupy inf':>9} {'compile inf':>11} | "
          f"{'cupy tr':>9} {'compile tr':>11}")
    compiled = torch.compile(eager_loop, fullgraph=True, dynamic=False)
    for T in (4, 8, 16, 32):
        x = torch.randn(T, N, device=dev) * 2.0
        y = torch.randn(T, N, device=dev)
        v0 = torch.zeros(N, device=dev)
        rho0 = torch.zeros(N, device=dev)

        def cupy_inf():
            with torch.no_grad():
                cupy_impl(x, y, v0, rho0)

        def compile_inf():
            with torch.no_grad():
                compiled(x, y, v0, rho0)

        xg = x.clone().requires_grad_(True)
        yg = y.clone().requires_grad_(True)

        def cupy_tr():
            xg.grad = yg.grad = None
            s1, s2, _, _ = cupy_impl(xg, yg, v0, rho0)
            (s1.sum() + s2.sum()).backward()

        def compile_tr():
            xg.grad = yg.grad = None
            s1, s2, _, _ = compiled(xg, yg, v0, rho0)
            (s1.sum() + s2.sum()).backward()

        r = [triton.testing.do_bench(f, warmup=25, rep=100,
                                     return_mode="median") * 1e3
             for f in (cupy_inf, compile_inf, cupy_tr, compile_tr)]
        print(f"{T:>4} | {r[0]:>8.2f}u {r[1]:>10.2f}u | "
              f"{r[2]:>8.2f}u {r[3]:>10.2f}u")


if __name__ == "__main__":
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    print("== correctness ==")
    ok = True
    for T in (1, 4, 16, 64):
        ok &= check(T, 32768)
    # N not divisible by 4 -> exercises the float4 edge/tail path
    for N in (1023, 4095, 65537):
        ok &= check(16, N)
    ok &= check(16, 32768, nonzero_init=True)
    ok &= check(16, 32768, nonzero_init=True, seed_state_grads=True)
    ok &= check(16, 1023, nonzero_init=True, seed_state_grads=True)
    print("ALL CORRECT" if ok else "SOME FAILED")
    bench()
