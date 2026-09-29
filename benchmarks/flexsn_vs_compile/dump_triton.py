"""Dump, side by side, the Triton that *inductor* generates for the unrolled
short-T neuron and the Triton that *FlexSN* generates for the same neuron.

Two captures land in the resolved figure directory:

- ``inf_T{T}_N{N}.py`` / ``train_T{T}_N{N}.py`` — inductor. The script re-execs
  itself with ``TORCH_LOGS=output_code`` to capture the exact Python + Triton
  module inductor compiles for the unrolled time loop (inference and training).
- ``flexsn_inf_T{T}_N{N}.py`` / ``flexsn_train_T{T}_N{N}.py`` — FlexSN
  (``backend="triton"``). Its templates hand the generated kernel source to
  ``compile_triton_code_str``; we wrap that call to capture the four kernels
  (inference, inference-final-state, forward, backward) as they are built.

The annotated reading and comparison lives in ``TRITON_CODEGEN.md`` next to this
file (inductor vs FlexSN, plus spikingjelly's hand-written neuron kernels and a
PTX check of Triton's automatic half2 packing).

Run: python benchmarks/flexsn_vs_compile/dump_triton.py --T 4 --N 32768
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _child() -> None:
    """Capture one backend's generated Triton source (dispatch on env)."""
    if os.environ.get("DUMP_BACKEND") == "flexsn":
        _child_flexsn()
        return

    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kernels import eager_loop, eager_loop_save

    T = int(os.environ["DUMP_T"])
    N = int(os.environ["DUMP_N"])
    mode = os.environ["DUMP_MODE"]
    dev = "cuda"
    x = torch.randn(T, N, device=dev)
    y = torch.randn(T, N, device=dev)
    v0 = torch.zeros(N, device=dev)
    rho0 = torch.zeros(N, device=dev)

    if mode == "inf":
        f = torch.compile(eager_loop, fullgraph=True, dynamic=False)
        with torch.no_grad():
            f(x, y, v0, rho0)
    elif mode == "save":
        # Return the per-step history (like FlexSN) and let AOTAutograd decide
        # whether to keep it for backward or rematerialize the step.
        f = torch.compile(eager_loop_save, fullgraph=True, dynamic=False)
        xg = x.requires_grad_(True)
        yg = y.requires_grad_(True)
        s1, s2, _, _, *_ = f(xg, yg, v0, rho0)
        (s1.sum() + s2.sum()).backward()
    else:
        f = torch.compile(eager_loop, fullgraph=True, dynamic=False)
        xg = x.requires_grad_(True)
        yg = y.requires_grad_(True)
        s1, s2, _, _ = f(xg, yg, v0, rho0)
        (s1.sum() + s2.sum()).backward()


def _extract_output_code(log: str) -> str:
    """Strip the ``[__output_code]`` prefix, preserving source indentation.

    The logger emits ``[__output_code] <line>`` with a single separator space;
    ``lstrip`` would eat the code's own indentation and produce snapshots that
    are not parseable Python.
    """
    out = []
    for line in log.splitlines():
        if "[__output_code]" not in line:
            continue
        code = line.split("[__output_code]", 1)[1].removeprefix(" ")
        if code.startswith("Output code"):
            continue
        out.append(code)
    return "\n".join(out)


def _child_flexsn() -> None:
    """Build FlexSN and dump the Triton source its templates generate.

    FlexSN's builders (``build_inference_kernel`` / ``build_training_kernels``)
    hand their generated kernel source to
    ``spikingjelly...flexsn.template.compile_triton_code_str``. Wrapping that
    name in the template module records every kernel (inference, final-state,
    forward, backward) as it is compiled. ``T`` only enters as a ``tl.constexpr``
    so this source is ``T``-independent; the filename keeps ``T`` for symmetry
    with the inductor dump.
    """
    import importlib

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kernels import make_flexsn

    # Dynamic import: spikingjelly may live in a separate checkout, so a static
    # import is not resolvable in every environment.
    try:
        tmpl = importlib.import_module(
            "spikingjelly.activation_based.triton_kernel.flexsn.template"
        )
    except ImportError as e:  # pragma: no cover - environment dependent
        print(f"[flexsn] cannot import the FlexSN template module: {e}")
        return

    captured: dict[str, str] = {}
    orig = tmpl.compile_triton_code_str

    def capture(code: str, name: str, *a, **k):
        captured[name] = code
        return orig(code, name, *a, **k)

    setattr(tmpl, "compile_triton_code_str", capture)

    T = int(os.environ["DUMP_T"])
    N = int(os.environ["DUMP_N"])
    outdir = Path(os.environ["DUMP_OUTDIR"])

    # Construction builds all four kernels, which is enough to capture them.
    make_flexsn((N,), "cuda")

    inf = [c for n, c in captured.items() if n.startswith("flexsn_inference_kernel")]
    fwd = [c for n, c in captured.items() if n.startswith("flexsn_forward_kernel")]
    bwd = [c for n, c in captured.items() if n.startswith("flexsn_backward_kernel")]
    (outdir / f"flexsn_inf_T{T}_N{N}.py").write_text("\n\n".join(inf))
    (outdir / f"flexsn_train_T{T}_N{N}.py").write_text("\n\n".join(fwd + bwd))
    print(
        f"[flexsn] wrote {outdir}  "
        f"({len(inf)} inference + {len(fwd)} forward + {len(bwd)} backward)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--N", type=int, default=32768)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    if os.environ.get("DUMP_CHILD") == "1":
        _child()
        return

    from btorch.utils.file import fig_path

    outdir = args.outdir or (fig_path() / "generated_triton")
    outdir.mkdir(parents=True, exist_ok=True)

    for mode in ("inf", "train", "save"):
        env = dict(
            os.environ,
            DUMP_CHILD="1",
            DUMP_MODE=mode,
            DUMP_T=str(args.T),
            DUMP_N=str(args.N),
            TORCH_LOGS="output_code",
        )
        proc = subprocess.run(
            [sys.executable, __file__], env=env, capture_output=True, text=True
        )
        code = _extract_output_code(proc.stderr + proc.stdout)
        if not code.strip():
            print(
                f"[{mode}] no output_code captured; stderr tail:\n{proc.stderr[-800:]}"
            )
            continue
        dst = outdir / f"{mode}_T{args.T}_N{args.N}.py"
        dst.write_text(code)
        n_kernels = code.count("async_compile.triton(")
        print(f"[{mode}] wrote {dst}  ({n_kernels} triton kernel(s))")

    # FlexSN is captured in its own child (it does not use torch.compile).
    env = dict(
        os.environ,
        DUMP_CHILD="1",
        DUMP_BACKEND="flexsn",
        DUMP_T=str(args.T),
        DUMP_N=str(args.N),
        DUMP_OUTDIR=str(outdir),
    )
    proc = subprocess.run(
        [sys.executable, __file__], env=env, capture_output=True, text=True
    )
    print(proc.stdout.strip())
    if proc.returncode != 0 or not proc.stdout.strip():
        print(
            f"[flexsn] dump failed (rc={proc.returncode}); stderr tail:\n"
            f"{proc.stderr[-800:]}"
        )


if __name__ == "__main__":
    main()
