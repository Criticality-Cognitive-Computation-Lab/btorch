"""End-to-end benchmark: a recurrent spiking network with sparse recurrence.

The network is ``RecurrentNN(LIF, ExponentialPSC(linear=<connection>))``:
every step the spikes of ``N`` LIF neurons are propagated through the
recurrent connection into an exponential synaptic current. Only the
connection differs between implementations:

- ``legacy[native]`` / ``legacy[torch_sparse]``: frozen pre-refactor forward
  (``_legacy_baseline.py``);
- ``new[default]``: :class:`btorch.models.connection.SparseConnection` with
  the default plan;
- ``new[push-hint]``: the same with ``Hints(expected_density=--rate-hint)``,
  which lets the planner choose source-driven propagation;
- ``dense``: :class:`btorch.models.linear.DenseConn` (up to 8,192 neurons).

Two phases are timed per configuration and reported **per simulation step**:

- ``infer``: ``T`` steps under ``torch.no_grad``;
- ``train``: ``T`` steps with autograd, then the gradient of the mean spike
  count w.r.t. the recurrent weights (backward through time, surrogate
  gradients).

The neurons receive a constant heterogeneous drive plus the recurrent
current; ``rate`` in the output is the measured fraction of neurons spiking
per step (the spike density the connection sees).

The whole trajectory is kept on the device (stacked spikes for ``infer``,
the autograd graph for ``train``), so memory grows with ``T * B * N``. When
``T`` steps do not fit the budget (``--max-gpu-gb``) or the memory currently
free, the largest ``T`` that fits is used and recorded (``steps``); below
``--min-steps`` the configuration is skipped with the reason.

Run::

    python benchmarks/sparse_conn/bench_rsnn.py --neurons 4096 100000 \
        --batch 1 32 --steps 200 \
        --out benchmarks/sparse_conn/results/rsnn_rtx5090.json
"""

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
from _common import (
    Skipped,
    add_gpu_arguments,
    best_probe,
    contention_probe,
    env_info,
    gpu_state,
    is_oom,
    make_graph,
    require_memory,
    setup_device,
    sync,
    wait_for_quiet,
)
from _legacy_baseline import LegacySparseConn, available_legacy_backends
from bench_sparse_conn import DENSE_MAX_NEURONS, check_parity, estimate_bytes

from btorch.models import environ, functional
from btorch.models.connection import SparseConnection
from btorch.models.linear import DenseConn
from btorch.models.neurons.lif import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import ExponentialPSC
from btorch.sparse import Hints


# Scale of the recurrent weights: ``N(0, 1) * WEIGHT_SCALE / sqrt(indegree)``.
# Balanced (zero-mean) recurrence that perturbs the drive without taking the
# network out of the asynchronous low-rate regime.
WEIGHT_SCALE = 0.05
# The constant drive of each neuron is uniform in this range, in units of the
# rheobase (the constant input that just reaches threshold). With
# ``tau = 20 ms`` and ``dt = 1 ms`` it gives a few percent of the neurons
# spiking per step (tens of Hz).
DRIVE = (0.7, 1.6)


OOM = "skipped: exceeds memory cap (out of memory inside this process)"


def _connections(n: int, device: str, rate_hint: float) -> dict[str, Callable]:
    out: dict[str, Callable] = {}
    for backend in available_legacy_backends():
        out[f"legacy[{backend}]"] = lambda m, b=backend: LegacySparseConn(
            m, backend=b, device=device
        )
    out["new[default]"] = lambda m: SparseConnection.from_adjacency(m).to(device)
    out["new[push-hint]"] = lambda m: SparseConnection.from_adjacency(
        m, hints=Hints(expected_density=rate_hint)
    ).to(device)
    if n <= DENSE_MAX_NEURONS:
        out["dense"] = lambda m: DenseConn(
            n, n, weight=torch.tensor(m.toarray(), device=device), device=device
        )
    return out


def _conn_name(name: str) -> str:
    """Name used by ``bench_sparse_conn.estimate_bytes``."""
    return "new[eager]" if name.startswith("new") else name


def step_bytes(name: str, n: int, nnz: int, batch: int, train: bool) -> float:
    """Rough device memory one more simulation step keeps alive."""
    if not train:
        return 8.0 * batch * n  # the stacked spikes (and the list they come from)
    # Autograd keeps the neuron and synapse states and their intermediates.
    per_step = 80.0 * batch * n
    if name == "legacy[torch_sparse]":
        per_step += 8.0 * nnz * batch  # the saved [nnz, batch] products
    return per_step


def fit_steps(name, n, nnz, batch, train, args) -> int:
    """Largest number of steps (<= ``--steps``) that fits the memory."""
    base = estimate_bytes(_conn_name(name), n, nnz, batch)
    per_step = step_bytes(name, n, nnz, batch, train)
    require_memory(base, args.device, args.max_gb, name)
    if args.device != "cuda":
        return args.steps
    free, _ = torch.cuda.mem_get_info()
    cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    budget = min(args.max_gb * 2**30, 0.9 * (free + cached)) - base
    steps = int(min(args.steps, budget // per_step))
    if steps < args.min_steps:
        need = (base + per_step * args.min_steps) / 2**30
        raise Skipped(
            f"{name}: {args.min_steps} steps need ~{need:.1f} GB "
            f"({(free + cached) / 2**30:.1f} GB free, budget {args.max_gb:g} GB)"
        )
    return steps


def build_network(conn: torch.nn.Module, n: int, device: str) -> RecurrentNN:
    neuron = LIF(n_neuron=n, tau=20.0, step_mode="s", device=device)
    psc = ExponentialPSC(n_neuron=n, tau_syn=5.0, linear=conn, step_mode="s")
    return RecurrentNN(neuron=neuron, synapse=psc, step_mode="m").to(device)


def _time_runs(fn: Callable, device: str, repeats: int, max_wait: float) -> dict:
    """Seconds per run of ``fn`` after one warm-up run.

    Every run is bracketed by contention probes and only runs during which
    the GPU was not time-sliced with other jobs are used (see
    :func:`_common.time_calls`); without any such run within ``max_wait``
    extra seconds, all runs are used and the result is flagged.
    """
    fn()
    clean, dirty = [], []
    deadline = None
    before = wait_for_quiet(device, max_wait=max_wait)
    while len(clean) < repeats:
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        sample = time.perf_counter() - t0
        if deadline is None:
            deadline = time.perf_counter() + max_wait + repeats * sample
        after = contention_probe(device, 50)
        ok = device != "cuda" or max(before, after) <= 2.0 * best_probe()
        (clean if ok else dirty).append((sample, max(before, after)))
        before = after
        if time.perf_counter() > deadline and len(clean) + len(dirty) >= repeats:
            break
    used = clean or dirty
    samples = [s for s, _ in used]
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "probe_us": statistics.median(p for _, p in used),
        "throttled": not clean,
        "samples": len(samples),
    }


def bench_one(name, conn, n, nnz, batch, args) -> dict:
    device = args.device
    row = {"n": n, "nnz": nnz, "impl": name, "batch": batch, "device": device}
    state = gpu_state()
    row["gpu_util"], row["gpu_other_mb"] = state.get("util"), state.get("other_mb")
    if isinstance(conn, SparseConnection):
        row["plan"] = f"{conn._plan.algorithm}/{conn._plan.backend}"
    net = build_network(conn, n, device)
    params = [p for p in conn.parameters() if p.requires_grad]
    g = torch.Generator(device="cpu").manual_seed(0)
    # Constant drive per (sample, neuron), in units of the rheobase of the
    # LIF neuron: dV/dt = -(V - V_reset) / tau + I / c_m.
    neuron = net.neuron
    rheobase = float((neuron.v_threshold - neuron.v_reset) * neuron.c_m / neuron.tau)
    drive = DRIVE[0] + (DRIVE[1] - DRIVE[0]) * torch.rand(batch, n, generator=g)
    drive = (drive * rheobase).to(device)
    functional.init_net_state(net, batch_size=batch, device=device, dtype=torch.float32)

    for phase, train in (("infer", False), ("train", True)):
        try:
            steps = fit_steps(name, n, nnz, batch, train, args)
        except Skipped as e:
            row[f"{phase}_skipped"] = str(e)
            continue
        x = drive.expand(steps, batch, n)
        rate = {}

        def run(train=train, x=x, rate=rate):
            functional.reset_net(net, batch_size=batch)
            if train:
                spikes, _ = net(x)
                torch.autograd.grad(spikes.mean(), params)
            else:
                with torch.no_grad():
                    spikes, _ = net(x)
            rate["value"] = spikes.detach().mean()

        try:
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            res = _time_runs(run, device, args.repeats, args.max_wait)
            median, best = res["median"], res["min"]
            row[f"{phase}_probe_us"] = res["probe_us"]
            row[f"{phase}_throttled"] = res["throttled"]
            row[f"{phase}_samples"] = res["samples"]
        except Exception as e:  # report, keep going
            row[f"{phase}_error"] = (
                OOM if is_oom(e) else f"{type(e).__name__}: {str(e)[:80]}"
            )
            rate.clear()
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
        row[f"{phase}_steps"] = steps
        row[f"{phase}_ms_per_step"] = median * 1e3 / steps
        row[f"{phase}_min_ms_per_step"] = best * 1e3 / steps
        row[f"{phase}_rate"] = float(rate["value"])
        if device == "cuda":
            row[f"{phase}_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    functional.reset_net(net, batch_size=batch)
    return row


def run(args: argparse.Namespace) -> list[dict]:
    rows = []
    device = args.device
    for n in args.neurons:
        mat = make_graph(n, args.indegree)
        mat.data *= WEIGHT_SCALE / args.indegree**0.5
        nnz = int(mat.nnz)
        print(f"--- N={n} nnz={nnz}", flush=True)
        reference = None
        for name, build in _connections(n, device, args.rate_hint).items():
            if args.impl and not any(k in name for k in args.impl):
                continue
            conn = None
            try:
                require_memory(
                    estimate_bytes(_conn_name(name), n, nnz, min(args.batch)),
                    device,
                    args.max_gb,
                    name,
                )
                sync(device)
                t0 = time.perf_counter()
                conn = build(mat)
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
            if conn is None:
                print(f"N={n:<7d} {name:22s} SKIPPED: {reason}", flush=True)
                rows += [
                    {"n": n, "nnz": nnz, "impl": name, "batch": b, "skipped": reason}
                    for b in args.batch
                ]
                continue
            # The recurrent operator must be the same for every implementation.
            pair = {"ref": reference, name: conn} if reference else {name: conn}
            parity = check_parity(pair, n, device)[name]
            if parity > 1e-3:
                raise RuntimeError(f"{name} disagrees with the reference: {parity:.2e}")
            if reference is None and n <= DENSE_MAX_NEURONS:
                reference = conn
            for batch in args.batch:
                with environ.context(dt=1.0):
                    row = bench_one(name, conn, n, nnz, batch, args)
                row["build_ms"] = build_ms
                rows.append(row)
                nan = float("nan")
                print(
                    f"N={n:<7d} {name:22s} B={batch:<3d} "
                    f"infer {row.get('infer_ms_per_step', nan):8.4f} "
                    f"(min {row.get('infer_min_ms_per_step', nan):8.4f}) ms/step  "
                    f"train {row.get('train_ms_per_step', nan):8.4f} "
                    f"(min {row.get('train_min_ms_per_step', nan):8.4f}) ms/step  "
                    f"T={row.get('infer_steps')}/{row.get('train_steps')}  "
                    f"rate {row.get('infer_rate', nan):.3f}  "
                    f"peak {row.get('train_peak_mb', nan):7.0f} MB  "
                    f"util {row['gpu_util']}"
                    + (f"  [{row['plan']}]" if "plan" in row else "")
                    + "".join(
                        f"  [{k}: {v}]"
                        for k, v in row.items()
                        if k.endswith(("_error", "_skipped"))
                    ),
                    flush=True,
                )
            if conn is not reference:
                del conn
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        reference = None
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
    p.add_argument("--neurons", nargs="+", type=int, default=[4096])
    p.add_argument("--indegree", type=int, default=100)
    p.add_argument("--batch", nargs="+", type=int, default=[1, 32])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--min-steps", type=int, default=20)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--rate-hint",
        type=float,
        default=0.02,
        help="expected spike density given to the new[push-hint] connection",
    )
    p.add_argument("--impl", nargs="*", default=None, help="substring filter")
    p.add_argument(
        "--max-wait",
        type=float,
        default=20.0,
        help="extra seconds spent per measurement waiting for clean runs",
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
