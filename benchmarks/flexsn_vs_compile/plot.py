"""Render the FlexSN-vs-compile benchmark figures.

Reads the JSON produced by ``bench_short_t.py`` and ``bench_sweep.py`` (both
next to the resolved ``fig/benchmark/flexsn_vs_compile/`` directory) and writes
publication-grade SVG/PDF/TIFF/PNG:

- ``fig1_short_t``: bar pair (inference | training) showing inductor matches
  FlexSN at inference and beats it at training for small ``T``.
- ``fig2_sweep``: three log-log panels (inference, training, compile cost)
  showing how each compile strategy scales with sequence length.

Usage::

    python benchmark/flexsn_vs_compile/plot.py
"""
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from btorch.utils.file import fig_path


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
})

# NMI pastel family + directional accents (see btorch nature-figure palette)
P = {
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "ours_large": "#F0C0CC",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "neutral_dark": "#606060",
    "delta_up": "#2E9E44",
}
MM = 1 / 25.4


def save_pub(fig, out_dir, name, dpi=600):
    for ext in ("svg", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.tiff", dpi=dpi, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    print("saved", out_dir / f"{name}.[svg/pdf/tiff/png]")


_SHORT_T_BACKENDS = [  # json key prefix, label, color
    ("inductor", "torch.compile (unrolled)", P["red_strong"]),
    ("flexsn", "FlexSN (Triton)", P["baseline_dark"]),
    ("cupy", "CuPy (float4)", P["teal"]),
]


def fig_short_t(json_file, out_dir):
    data = json.loads(json_file.read_text())
    Ts = [d["T"] for d in data]
    x = np.arange(len(Ts))
    n = len(_SHORT_T_BACKENDS)
    w = 0.8 / n

    fig, axes = plt.subplots(1, 2, figsize=(120 * MM, 46 * MM))
    for suffix, title, ax in [("inf", "Inference", axes[0]),
                              ("train", "Training (fwd + bwd)", axes[1])]:
        series = {k: np.array([d[f"{k}_{suffix}"] for d in data]) * 1e3  # -> us
                  for k, _, _ in _SHORT_T_BACKENDS}
        errs = {k: np.array([d.get(f"{k}_{suffix}_std", 0.0) for d in data]) * 1e3
                for k, _, _ in _SHORT_T_BACKENDS}
        for j, (key, label, color) in enumerate(_SHORT_T_BACKENDS):
            offset = (j - (n - 1) / 2) * w
            ax.bar(x + offset, series[key], w, color=color, edgecolor="white",
                   linewidth=0.4, zorder=3, label=label,
                   yerr=errs[key], error_kw=dict(elinewidth=0.5, ecolor="#555"))
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in Ts])
        ax.set_xlabel("Time steps $T$")
        ax.set_title(title, loc="left", fontweight="bold", pad=10)
        ax.set_ylim(0, max(s.max() for s in series.values()) * 1.14)
        ax.yaxis.grid(True, linewidth=0.3, color="#DDDDDD", zorder=0)
    axes[0].set_ylabel("Median kernel time (µs)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n,
               bbox_to_anchor=(0.5, 1.20), handlelength=1.2,
               columnspacing=1.4)
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.26, 1.12, letter, transform=ax.transAxes,
                fontsize=9, fontweight="bold")
    fig.text(0.5, 1.05, f"N = {data[0]['N']:,} neurons", ha="center",
             fontsize=6.5, color=P["neutral_dark"])
    fig.tight_layout(w_pad=2.4)
    save_pub(fig, out_dir, "fig1_short_t")


_VARIANTS = [  # key, label, color, marker
    ("flexsn", "FlexSN (Triton template)", P["teal"], "D"),
    ("unrolled", "inductor, unrolled", P["baseline_dark"], "o"),
    ("unrolled_cg", "unrolled + reduce-overhead", P["baseline_mid"], "s"),
    ("unrolled_manual_cg", "unrolled + manual CUDA graph",
     P["baseline_soft"], "^"),
    ("scan", "inductor, scan (host loop)", P["red_strong"], "o"),
    ("scan_manual_cg", "scan + manual CUDA graph", P["ours_large"], "^"),
    ("cupy", "CuPy (float4)", P["neutral_dark"], "v"),
]


def fig_sweep(json_file, out_dir):
    rows = json.loads(json_file.read_text())
    by = {}
    for r in rows:
        by.setdefault(r["variant"], {})[r["T"]] = r

    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 52 * MM))
    for key, label, color, marker in _VARIANTS:
        pts = by.get(key, {})
        Ts = sorted(pts)
        for kk, ax in [("inf_ms", axes[0]), ("train_ms", axes[1])]:
            xs = [t for t in Ts if kk in pts[t]]
            ys = [pts[t][kk] for t in xs]
            es = [pts[t].get(f"{kk}_std", 0.0) for t in xs]
            if xs:
                ax.errorbar(xs, ys, yerr=es, color=color, marker=marker, ms=3,
                            lw=1.0, label=label, zorder=3,
                            markeredgecolor="white", markeredgewidth=0.3,
                            elinewidth=0.5, capsize=1.5)
        if key in ("flexsn", "unrolled", "unrolled_cg", "scan", "cupy"):
            xs, ys = [], []
            for t in Ts:
                v = [pts[t].get("first_inf_s"), pts[t].get("first_train_s")]
                v = [u for u in v if u is not None]
                if v:
                    xs.append(t)
                    ys.append(max(v))
            if xs:
                axes[2].plot(xs, ys, color=color, marker=marker, ms=3, lw=1.0,
                             zorder=3, markeredgecolor="white",
                             markeredgewidth=0.3)

    titles = ["Inference", "Training (fwd + bwd)", "Compile / first-call cost"]
    for ax, title in zip(axes, titles):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Time steps $T$")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.yaxis.grid(True, which="major", linewidth=0.3, color="#DDDDDD",
                      zorder=0)
        ax.set_xticks([4, 16, 64, 256, 1024])
        ax.set_xticklabels(["4", "16", "64", "256", "1024"])
        ax.tick_params(which="minor", length=0)
    axes[0].set_ylabel("Median time (ms)")
    axes[2].set_ylabel("First call incl. compile (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.16), handlelength=1.6,
               columnspacing=1.4, labelspacing=0.4)
    for i, ax in enumerate(axes):
        ax.text(-0.22 if i == 0 else -0.18, 1.08, "abc"[i],
                transform=ax.transAxes, fontsize=9, fontweight="bold")
    N = next(iter(next(iter(by.values())).values()))["N"]
    fig.text(0.99, 1.02, f"N = {N:,} neurons", ha="right", fontsize=6.5,
             color=P["neutral_dark"])
    fig.tight_layout(w_pad=2.0)
    save_pub(fig, out_dir, "fig2_sweep")


def main():
    out_dir = fig_path()                       # this script's dir, for figures
    base = out_dir.parent                       # fig/benchmark/flexsn_vs_compile
    short_json = base / "bench_short_t" / "bench_short_t.json"
    sweep_json = base / "bench_sweep" / "bench_sweep.json"
    if short_json.exists():
        fig_short_t(short_json, out_dir)
    if sweep_json.exists():
        fig_sweep(sweep_json, out_dir)


if __name__ == "__main__":
    main()
