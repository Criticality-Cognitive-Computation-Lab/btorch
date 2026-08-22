"""Speedup-ratio figures for the GLIF3 kernels, relative to pure eager PyTorch.

Reads the same sweep JSON as ``plot_glif_kernels.py`` and draws, for each
``kind`` (``neuron``, ``dense``, ``sparse``), a 2x2 panel of **speedup vs the
eager baseline** (``eager_ms / backend_ms``, log-scale, dashed line at 1x) over
N and T, for inference and training. Every backend (kernel DSLs plus
``torch.compile`` where measured) is shown on the same axes so the relative
cost of each optimization layer -- fused kernel vs ``torch.compile`` -- is
directly comparable. Run::

    python -m benchmarks.glif_net.bench_glif_full --out results.json
    python -m benchmarks.glif_net.plot_glif_speedup --results results.json
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from benchmarks.glif_net.plot_glif_kernels import (
    BACKEND_COLOR, BACKENDS, COMPILE_COLOR, _series, _style,
)
from btorch.utils.file import fig_path

KINDS = ["neuron", "dense", "sparse"]


def _speedup(eager, backend_vals):
    """eager_ms / backend_ms per point; None (OOM / missing) drops the point."""
    out = []
    for e, b in zip(eager, backend_vals):
        if e is None or b is None or b == 0:
            out.append(None)
        else:
            out.append(e / b)
    return out


def _panel(ax, results, kind, mode, axis, xvals, label):
    eager = results.get(f"{kind}/{mode}/eager", {}).get(axis)
    if eager is None:
        ax.text(0.5, 0.5, "no eager baseline", transform=ax.transAxes, ha="center")
        return
    for backend in BACKENDS:
        row = results.get(f"{kind}/{mode}/{backend}")
        if row is None:
            continue
        color = BACKEND_COLOR[backend]
        sp = _speedup(eager, row[axis])
        ax.plot(*_series(xvals, sp), color=color, ls="-", marker="o", zorder=2)
    compiled = results.get(f"{kind}/{mode}/torch_compile")
    if compiled is not None:
        sp = _speedup(eager, compiled[axis])
        ax.plot(*_series(xvals, sp), color=COMPILE_COLOR, ls="-", marker="s", zorder=3)
    ax.axhline(1.0, color="0.5", ls=(0, (3, 2)), lw=0.8, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (neurons)" if axis == "N" else "T (steps)")
    ax.set_ylabel("speedup vs eager (x)")
    ax.grid(True, which="major", ls=":", lw=0.4, alpha=0.5)
    ax.set_title(mode, fontsize=8)
    ax.text(-0.24, 1.03, label, transform=ax.transAxes, fontweight="bold", fontsize=9)


def plot_kind(results, meta, kind, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.4))
    n_x, t_x = meta["n_sweep"]["N"], meta["t_sweep"]["T"]
    for ax, mode, axis, xvals, label in [
        (axes[0, 0], "inference", "N", n_x, "a"),
        (axes[0, 1], "inference", "T", t_x, "b"),
        (axes[1, 0], "training", "N", n_x, "c"),
        (axes[1, 1], "training", "T", t_x, "d"),
    ]:
        _panel(ax, results, kind, mode, axis, xvals, label)

    handles = [Line2D([], [], color=BACKEND_COLOR[b], marker="o", label=b) for b in BACKENDS]
    if any(f"{kind}/{m}/torch_compile" in results for m in ("inference", "training")):
        handles.append(Line2D([], [], color=COMPILE_COLOR, marker="s",
                              label="torch.compile (reduce-overhead)"))
    handles.append(Line2D([], [], color="0.5", ls=(0, (3, 2)), label="eager (1x)"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"GLIF3 {kind} multistep — speedup vs eager PyTorch, RTX 5090  "
                 f"(N sweep at T={meta['n_sweep']['T']}, T sweep at N={meta['t_sweep']['N']})",
                 y=-0.02, fontsize=7.5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    for ext in ("png", "pdf"):
        path = out_dir / f"glif_{kind}_speedup.{ext}"
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench_glif_full_results.json")
    args = ap.parse_args()
    with open(args.results) as f:
        data = json.load(f)
    _style()
    out_dir = fig_path()
    for kind in KINDS:
        plot_kind(data["results"], data["meta"], kind, out_dir)


if __name__ == "__main__":
    main()
