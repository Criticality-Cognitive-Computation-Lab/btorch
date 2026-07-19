"""FlexSN gradient-correctness test vs the eager reference.

Exercises the full backward, including the gradients w.r.t. the *initial*
states (``grad_v0`` / ``grad_rho0``) -- the path affected by the
``grad_init_state_store`` fix in spikingjelly's FlexSN template. Uses nonzero,
differentiable initial states so every gradient output is checked, across
several N and T.

Run: python benchmarks/flexsn_vs_compile/test_flexsn.py
"""
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from kernels import core_step, make_flexsn  # noqa: E402


dev = "cuda"


def eager_with_states(x, y, v0, rho0):
    """Reference forward returning stacked spikes and final states."""
    v, rho = v0, rho0
    s1l, s2l = [], []
    for t in range(x.shape[0]):
        s1, s2, v, rho = core_step(x[t], y[t], v, rho)
        s1l.append(s1)
        s2l.append(s2)
    return torch.stack(s1l), torch.stack(s2l), v, rho


def run(fn, x, y, v0, rho0, w, flex=False):
    """Forward + weighted-loss backward; returns outputs and all input grads."""
    xi = x.clone().requires_grad_(True)
    yi = y.clone().requires_grad_(True)
    vi = v0.clone().requires_grad_(True)
    ri = rho0.clone().requires_grad_(True)
    if flex:
        fn.reset()
        fn.states = [vi, ri]      # inject differentiable initial states
        s1, s2 = fn(xi, yi)
        vf, rf = fn.states        # final states after the sequence
    else:
        s1, s2, vf, rf = fn(xi, yi, vi, ri)
    loss = (s1 * w).sum() + (s2 * 0.7 * w).sum() + vf.sum() * 0.3 + rf.sum() * 0.2
    loss.backward()
    return dict(s1=s1, s2=s2, v_final=vf, rho_final=rf,
                grad_x=xi.grad, grad_y=yi.grad,
                grad_v0=vi.grad, grad_rho0=ri.grad)


def main():
    ok_all = True
    for N in (8192, 32768):
        flexsn = make_flexsn((N,), dev)
        for T in (16, 64, 256):
            torch.manual_seed(T + N)
            x = torch.randn(T, N, device=dev) * 2.0
            y = torch.randn(T, N, device=dev)
            v0 = torch.randn(N, device=dev)             # nonzero init states
            rho0 = torch.randn(N, device=dev).abs()
            w = torch.linspace(0.5, 1.5, T, device=dev).unsqueeze(1)
            ref = run(eager_with_states, x, y, v0, rho0, w)
            got = run(flexsn, x, y, v0, rho0, w, flex=True)
            worst, worst_key = 0.0, ""
            for k, a in ref.items():
                b = got[k]
                if b is None:
                    print(f"  N={N} T={T}: {k} is None in FlexSN!")
                    ok_all = False
                    continue
                rel = (a - b).abs().max().item() / (a.abs().max().item() + 1e-9)
                if rel > worst:
                    worst, worst_key = rel, k
            ok = worst < 2e-3
            ok_all &= ok
            print(f"N={N:6d} T={T:5d}: worst_rel={worst:.2e} ({worst_key})  "
                  f"{'OK' if ok else 'MISMATCH'}", flush=True)
    print("ALL CORRECT" if ok_all else "SOME MISMATCH")


if __name__ == "__main__":
    main()
