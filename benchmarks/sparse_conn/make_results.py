"""Render the benchmark JSON files as Markdown tables.

Reads the output of ``bench_sparse_conn.py`` and ``bench_rsnn.py`` and
prints the tables used in ``RESULTS.md``::

    python benchmarks/sparse_conn/make_results.py \
        --conn benchmarks/sparse_conn/results/conn_rtx5090.json \
        --rsnn benchmarks/sparse_conn/results/rsnn_rtx5090.json

Cell format: ``median`` in milliseconds. A trailing ``~`` marks a noisy cell
(median more than ``--noise`` times the fastest sample), in which case the
fastest sample follows in parentheses. A trailing ``!`` marks a throttled
cell: no sample could be taken while the GPU was free of other jobs' time
slices (the contention probe around every sample was more than
``--throttle`` times the best probe of the run). ``-`` means not run, with the reason
listed under the table.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


CONN_IMPLS = (
    "legacy[native]",
    "legacy[torch_sparse]",
    "dense",
    "new[eager]",
    "new[compile]",
    "new[push-hint]",
)
RSNN_IMPLS = (
    "legacy[native]",
    "legacy[torch_sparse]",
    "new[default]",
    "new[push-hint]",
    "dense",
)


def _fmt(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def cell(row: dict, key: str, min_key: str, probe_key: str, marks: dict) -> str:
    """``median`` or ``median~ (min)`` of one measurement, ``!`` if
    throttled."""
    value = row.get(key)
    if value is None or math.isnan(value):
        return "-"
    text = _fmt(value)
    probe, best_probe = row.get(probe_key), marks["best_probe"]
    throttled = row.get(probe_key.replace("_probe_us", "_throttled"))
    if throttled or (probe and best_probe and probe > marks["throttle"] * best_probe):
        text += "!"
    best = row.get(min_key)
    if best and value > marks["noise"] * best:
        text += f"~ ({_fmt(best)})"
    return text


def _reason(row: dict, *keys: str) -> str | None:
    for key in keys:
        if row.get(key):
            return str(row[key])
    return None


def conn_tables(payload: dict, marks: dict) -> str:
    rows = payload["rows"]
    out = []
    groups: dict = defaultdict(dict)
    for r in rows:
        groups[(r["workload"], r["n"], r["nnz"])][
            (r["batch"], r["density"], r["impl"])
        ] = r
    for (workload, n, nnz), cells in groups.items():
        out.append(f"\n#### `{workload}`: {n:,} neurons, {nnz:,} edges\n")
        impls = [i for i in CONN_IMPLS if any(k[2] == i for k in cells)]
        configs = sorted({(b, d) for b, d, _ in cells})
        notes: dict[str, set] = defaultdict(set)
        for title, key in (
            ("forward (ms)", "fwd"),
            ("forward + backward (ms)", "fwd_bwd"),
            ("peak GPU memory, forward / forward + backward (MB)", "peak"),
        ):
            out.append(f"**{title}**\n")
            out.append("| batch | density | " + " | ".join(impls) + " |")
            out.append("|---:|---:|" + "---:|" * len(impls))
            for batch, density in configs:
                line = [str(batch), f"{density:g}"]
                for impl in impls:
                    r = cells.get((batch, density, impl))
                    if r is None:
                        line.append("-")
                    elif key == "peak":
                        a, b = r.get("fwd_peak_mb"), r.get("fwd_bwd_peak_mb")
                        line.append(
                            "-"
                            if a is None and b is None
                            else f"{a:.0f} / {b:.0f}"
                            if a is not None and b is not None
                            else f"{a:.0f} / -"
                            if a is not None
                            else f"- / {b:.0f}"
                        )
                    else:
                        text = cell(
                            r, f"{key}_ms", f"{key}_min_ms", f"{key}_probe_us", marks
                        )
                        if (
                            key == "fwd"
                            and impl == "new[push-hint]"
                            and r.get("plan", "").startswith("pull")
                            and text != "-"
                        ):
                            text += " (pull)"
                        line.append(text)
                        why = _reason(r, "skipped", f"{key}_error")
                        if why:
                            notes[why].add(f"{impl} B={batch}")
                out.append("| " + " | ".join(line) + " |")
            out.append("")
        if notes:
            out.append("Not run:\n")
            for why, who in notes.items():
                out.append(f"- {', '.join(sorted(who))}: {why}")
            out.append("")
        out.append("**build / preprocessing and compilation (ms)**\n")
        out.append(
            "| implementation | build | compile fwd (B=1 / B=32) | compile fwd+bwd |"
        )
        out.append("|---|---:|---:|---:|")
        for impl in impls:
            mine = [r for k, r in cells.items() if k[2] == impl]
            build = next((r["build_ms"] for r in mine if "build_ms" in r), None)
            comp = {
                key: " / ".join(
                    f"{r[key]:.0f}"
                    for r in sorted(mine, key=lambda r: r["batch"])
                    if key in r
                )
                for key in ("compile_fwd_ms", "compile_fwd_bwd_ms")
            }
            out.append(
                f"| {impl} | {'-' if build is None else f'{build:.0f}'} | "
                f"{comp['compile_fwd_ms'] or '-'} | "
                f"{comp['compile_fwd_bwd_ms'] or '-'} |"
            )
        util = [r["gpu_util"] for r in cells.values() if r.get("gpu_util") is not None]
        other = [
            r["gpu_other_mb"]
            for r in cells.values()
            if r.get("gpu_other_mb") is not None
        ]
        if util:
            out.append(
                f"\nGPU load from other jobs while this workload ran: utilisation "
                f"{min(util):.0f}-{max(util):.0f} %, {min(other) / 1024:.1f}-"
                f"{max(other) / 1024:.1f} GB held by other processes."
            )
    return "\n".join(out)


def rsnn_tables(payload: dict, marks: dict) -> str:
    rows = payload["rows"]
    out = []
    by_n: dict = defaultdict(dict)
    for r in rows:
        by_n[(r["n"], r["nnz"])][(r["batch"], r["impl"])] = r
    for (n, nnz), cells in by_n.items():
        out.append(f"\n#### {n:,} neurons, {nnz:,} recurrent edges\n")
        impls = [i for i in RSNN_IMPLS if any(k[1] == i for k in cells)]
        notes: dict[str, set] = defaultdict(set)
        out.append(
            "| batch | phase | " + " | ".join(impls) + " | steps | spikes/step |"
        )
        out.append("|---:|---|" + "---:|" * len(impls) + "---:|---:|")
        for batch in sorted({b for b, _ in cells}):
            for phase in ("infer", "train"):
                line, steps, rates = [str(batch), phase], set(), []
                for impl in impls:
                    r = cells.get((batch, impl))
                    if r is None:
                        line.append("-")
                        continue
                    line.append(
                        cell(
                            r,
                            f"{phase}_ms_per_step",
                            f"{phase}_min_ms_per_step",
                            f"{phase}_probe_us",
                            marks,
                        )
                    )
                    if f"{phase}_steps" in r:
                        steps.add(r[f"{phase}_steps"])
                        rates.append(r[f"{phase}_rate"])
                    why = _reason(r, "skipped", f"{phase}_skipped", f"{phase}_error")
                    if why:
                        notes[why].add(f"{impl} B={batch} {phase}")
                line.append("/".join(str(s) for s in sorted(steps)) or "-")
                line.append(
                    f"{100 * min(rates):.1f}-{100 * max(rates):.1f} %" if rates else "-"
                )
                out.append("| " + " | ".join(line) + " |")
        out.append("")
        out.append("Peak GPU memory of the training phase (MB):\n")
        out.append("| batch | " + " | ".join(impls) + " |")
        out.append("|---:|" + "---:|" * len(impls))
        for batch in sorted({b for b, _ in cells}):
            line = [str(batch)]
            for impl in impls:
                r = cells.get((batch, impl), {})
                peak = r.get("train_peak_mb")
                line.append("-" if peak is None else f"{peak:.0f}")
            out.append("| " + " | ".join(line) + " |")
        if notes:
            out.append("\nNot run:\n")
            for why, who in notes.items():
                out.append(f"- {', '.join(sorted(who))}: {why}")
    return "\n".join(out)


def env_block(payload: dict) -> str:
    env = payload.get("env", {})
    keys = ("gpu", "cpu", "torch", "cuda", "triton", "torch_sparse", "python")
    lines = [f"- {k}: {env[k]}" for k in keys if env.get(k)]
    state = env.get("gpu_state_at_start") or {}
    sel = env.get("gpu_selection")
    if sel:
        lines.append(
            f"- GPU {sel['index']} chosen ({sel['free_gb_at_launch']:.1f} of "
            f"{sel['total_gb']:.1f} GB free, {sel['util_at_launch']:.0f} % util, "
            f"{len(sel['other_processes'])} other compute process(es)); memory "
            f"cap of the benchmark process: {sel['memory_cap_gb']:.1f} GB"
        )
    if payload.get("best_probe_us"):
        lines.append(
            f"- best contention probe of the run: {payload['best_probe_us']:.1f} us"
        )
    if state:
        lines.append(
            f"- GPU at start: {state.get('other_mb', 0) / 1024:.1f} GB held by other "
            f"processes, {state.get('free_mb', 0) / 1024:.1f} GB free, utilisation "
            f"{state.get('util', float('nan')):.0f} %"
        )
    return "\n".join(lines)


def _marks(payload: dict, args: argparse.Namespace) -> dict:
    return {
        "noise": args.noise,
        "throttle": args.throttle,
        "best_probe": payload.get("best_probe_us"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--conn", type=Path, nargs="*", default=[])
    p.add_argument("--rsnn", type=Path, nargs="*", default=[])
    p.add_argument("--noise", type=float, default=1.3)
    p.add_argument("--throttle", type=float, default=2.5)
    args = p.parse_args()
    for path in args.conn:
        payload = json.loads(path.read_text())
        print(f"\n### Connection benchmark ({path.name})\n")
        print(env_block(payload))
        print(conn_tables(payload, _marks(payload, args)))
    for path in args.rsnn:
        payload = json.loads(path.read_text())
        print(f"\n### RSNN benchmark ({path.name})\n")
        print(env_block(payload))
        print(rsnn_tables(payload, _marks(payload, args)))


if __name__ == "__main__":
    main()
