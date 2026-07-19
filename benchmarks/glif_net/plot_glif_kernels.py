"""Nature-style performance figures for the GLIF3 kernels.

Reads the sweep JSON from ``bench_glif_full.py`` and draws, for each recurrent
kind (dense, sparse) in its own figure, a 2x2 panel of runtime vs N and vs T for
inference and training. Each panel overlays the three backends (colour) for the
recurrent kind (solid) against the neuron-only multistep baseline (dashed), plus
a ``torch.compile(mode="reduce-overhead")`` baseline (pink squares) on both the
dense and sparse figures. Run::

    python -m benchmarks.glif_net.bench_glif_full --out results.json
    python -m benchmarks.glif_net.plot_glif_kernels --results results.json
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from btorch.utils.file import fig_path

# Wong colour-blind-safe palette, one hue per backend.
BACKEND_COLOR = {"triton": "#0072B2", "warp": "#D55E00", "cupy": "#009E73"}
BACKENDS = ["triton", "warp", "cupy"]
COMPILE_COLOR = "#CC79A7"        # torch.compile(reduce-overhead) baseline


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.3, "lines.markersize": 3.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    })


def _series(x, y):
    """Drop out-of-memory (None) points, keeping x/y aligned."""
    xy = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
    return [p[0] for p in xy], [p[1] for p in xy]


def _panel(ax, results, kind, mode, axis, xvals, label):
    for backend in BACKENDS:
        color = BACKEND_COLOR[backend]
        rec = results[f"{kind}/{mode}/{backend}"][axis]
        neu = results[f"neuron/{mode}/{backend}"][axis]
        ax.plot(*_series(xvals, neu), color=color, lw=1.0, ls=(0, (3, 2)),
                marker="o", mfc="white", mew=0.8, alpha=0.55, zorder=1)
        ax.plot(*_series(xvals, rec), color=color, ls="-", marker="o", zorder=2)
    compiled = results.get(f"{kind}/{mode}/torch_compile")
    if compiled is not None:
        ax.plot(*_series(xvals, compiled[axis]), color=COMPILE_COLOR, ls="-",
                marker="s", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N (neurons)" if axis == "N" else "T (steps)")
    ax.set_ylabel("time per pass (ms)")
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
    handles += [Line2D([], [], color="0.35", ls="-", label=f"{kind} recurrent"),
                Line2D([], [], color="0.35", ls=(0, (3, 2)), label="neuron only")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"GLIF3 {kind} recurrent multistep — RTX 5090  "
                 f"(N sweep at T={meta['n_sweep']['T']}, T sweep at N={meta['t_sweep']['N']})",
                 y=-0.02, fontsize=7.5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    for ext in ("png", "pdf"):
        path = out_dir / f"glif_{kind}_sweep.{ext}"
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
    for kind in ("dense", "sparse"):
        plot_kind(data["results"], data["meta"], kind, out_dir)


if __name__ == "__main__":
    main()
