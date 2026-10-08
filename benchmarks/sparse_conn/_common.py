"""Helpers shared by the sparse connection benchmarks.

Timing, GPU bookkeeping for a *shared* machine (choice of the GPU, a hard
memory cap for this process, free-memory gating, load snapshots) and the
synthetic graph generator. Benchmark fixture code, not library code.

The benchmarks must never push another user's job into an out-of-memory
failure. :func:`setup_device` therefore picks the GPU with the most free
memory, refuses to run when too little is free, and caps what this process
may allocate; see there.
"""

import argparse
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable

import numpy as np
import scipy.sparse
import torch


class Skipped(Exception):
    """A configuration that is not run; the message is the reported reason."""


def make_graph(n: int, indegree: int, seed: int = 0) -> scipy.sparse.coo_array:
    """Random recurrent graph (rows = source, cols = destination).

    Every destination draws ``indegree`` sources uniformly; duplicate edges
    are merged, so the stored number of edges is slightly below
    ``n * indegree``. Weights are standard normal ``float32``.
    """
    rng = np.random.default_rng(seed)
    nnz = n * indegree
    src = rng.integers(0, n, nnz)
    dst = np.repeat(np.arange(n), indegree)
    w = rng.standard_normal(nnz).astype(np.float32)
    mat = scipy.sparse.coo_array((w, (src, dst)), shape=(n, n))
    mat.sum_duplicates()
    return mat


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def time_calls(
    fn: Callable,
    device: str,
    *,
    repeats: int = 15,
    target_ms: float = 20.0,
    max_inner: int = 200,
    warmup: int = 3,
    max_wait: float = 20.0,
    quiet_factor: float = 2.0,
) -> dict[str, float]:
    """Steady-state latency of ``fn`` in milliseconds.

    ``inner`` calls are queued back to back and the device is synchronised
    once before and once after them, so one sample is the cost per call of a
    loop that never waits for a result (launch overhead included, the
    synchronisation amortised). ``inner`` is chosen so that one sample takes
    about ``target_ms``. The same procedure is used for every
    implementation; implementations that synchronise internally (host-side
    packing) simply cannot queue ahead.

    On a shared GPU other jobs are scheduled in between, in bursts that last
    from milliseconds to seconds and multiply every latency. Each sample is
    therefore bracketed by two contention probes
    (:func:`contention_probe`) and only kept if both are within
    ``quiet_factor`` of the best probe of the process. Sampling goes on
    until ``repeats`` clean samples exist or ``max_wait`` seconds have been
    spent beyond the nominal duration; if no clean sample was obtained, all
    samples are used and the result is flagged ``throttled``.

    Returns:
        ``{"median", "min", "max", "inner", "repeats", "clean", "probe_us",
        "throttled"}``: statistics over the samples used, how many of them
        were clean, and the median probe around them in microseconds.
    """
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    fn()
    sync(device)
    once = (time.perf_counter() - t0) * 1e3
    inner = int(max(1, min(max_inner, target_ms / max(once, 1e-3))))
    clean, dirty = [], []
    deadline = time.perf_counter() + max_wait + 2e-3 * repeats * target_ms
    before = contention_probe(device, 50)
    while len(clean) < repeats:
        sync(device)
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        sync(device)
        sample = (time.perf_counter() - t0) * 1e3 / inner
        after = contention_probe(device, 50)
        probe = max(before, after)
        before = after
        if device != "cuda" or probe <= quiet_factor * best_probe():
            clean.append((sample, probe))
        else:
            dirty.append((sample, probe))
        if time.perf_counter() > deadline and len(clean) + len(dirty) >= repeats:
            break
    used = clean or dirty
    samples = [s for s, _ in used]
    return {
        "median": float(statistics.median(samples)),
        "min": float(min(samples)),
        "max": float(max(samples)),
        "inner": inner,
        "repeats": len(samples),
        "clean": len(clean),
        "probe_us": float(statistics.median(p for _, p in used)),
        "throttled": not clean,
    }


_PROBE: dict = {}


def contention_probe(device: str, calls: int = 200) -> float:
    """Microseconds per call of a trivial queued kernel (an in-place add on
    4,096 floats), synchronised once.

    On an idle device this is the launch overhead of the host (about 5-15
    us). When other processes compute on the same GPU the driver time-slices
    the contexts and the figure jumps by an order of magnitude; latencies
    measured at such a moment are those of the scheduler, not of the code
    under test. The benchmarks record the probe next to every measurement
    and wait for a quiet moment before taking it (:func:`wait_for_quiet`).
    """
    if device != "cuda":
        return 0.0
    t = _PROBE.get("t")
    if t is None:
        t = _PROBE["t"] = torch.zeros(4096, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(calls):
        t.add_(0.0)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / calls * 1e6
    _PROBE["best"] = min(us, _PROBE.get("best", us))
    return us


def best_probe() -> float:
    """Smallest contention probe seen in this process (microseconds)."""
    return _PROBE.get("best", 0.0)


def wait_for_quiet(device: str, factor: float = 2.0, max_wait: float = 20.0) -> float:
    """Wait until the contention probe is within ``factor`` of the best probe
    seen in this process, for at most ``max_wait`` seconds.

    Returns:
        The last probe in microseconds (still high if the wait timed out).
    """
    if device != "cuda":
        return 0.0
    deadline = time.perf_counter() + max_wait
    while True:
        us = min(contention_probe(device) for _ in range(3))
        if us <= factor * _PROBE["best"] or time.perf_counter() > deadline:
            return us
        time.sleep(0.5)


def gpu_state() -> dict:
    """Load of the benchmark GPU as seen by ``nvidia-smi`` and by PyTorch.

    ``other_mb`` is device memory outside the PyTorch pool of this process
    (``total - free - memory_reserved``): an upper bound of what other
    processes hold, because it also contains the CUDA context of this process
    and anything it allocated outside the PyTorch allocator. ``util`` is the
    utilisation of the whole device in percent just before the measurement;
    it includes other jobs and the preceding work of this process.
    """
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    state = {"free_mb": free / 2**20, "total_mb": total / 2**20}
    state["other_mb"] = (total - free - torch.cuda.memory_reserved()) / 2**20
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        # ``CUDA_VISIBLE_DEVICES`` renumbers devices: match by UUID.
        uuid = torch.cuda.get_device_properties(0).uuid
        for line in out.strip().splitlines():
            gpu_uuid, util = (s.strip() for s in line.split(","))
            if str(uuid) in gpu_uuid:
                state["util"] = float(util)
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        # Monitoring is best-effort and must not invalidate benchmark results.
        pass
    return state


def require_memory(need_bytes: float, device: str, max_gb: float, what: str) -> None:
    """Raise :class:`Skipped` unless ``need_bytes`` fit the device.

    The estimate is compared with the user budget ``max_gb`` and with the
    memory that is free *now* (plus what this process already cached), so a
    run on a shared GPU never tries to take memory other jobs are using.
    """
    if device != "cuda":
        return
    need_gb = need_bytes / 2**30
    if need_gb > max_gb:
        raise Skipped(
            f"{what}: needs ~{need_gb:.1f} GB, exceeds memory cap {max_gb:.1f} GB"
        )
    free, _ = torch.cuda.mem_get_info()
    cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    avail_gb = (free + cached) / 2**30
    if need_gb > 0.9 * avail_gb:
        raise Skipped(
            f"{what}: needs ~{need_gb:.1f} GB, only {avail_gb:.1f} GB free on "
            "the shared GPU"
        )


def add_gpu_arguments(p: argparse.ArgumentParser) -> None:
    """Add the device and GPU-safety options used by :func:`setup_device`."""
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--max-gpu-gb",
        type=float,
        default=20.0,
        help="upper bound of the GPU memory cap of this process",
    )
    p.add_argument(
        "--min-free-gb",
        type=float,
        default=8.0,
        help="refuse to run unless the chosen GPU has this much memory free",
    )
    p.add_argument(
        "--gpu-margin-gb",
        type=float,
        default=4.0,
        help="free memory always left to other jobs (cap = free - margin)",
    )


def query_gpus() -> list[dict]:
    """Memory, utilisation and foreign compute processes of every GPU.

    Returns:
        One dict per GPU (``index``, ``uuid``, ``free_gb``, ``total_gb``,
        ``util``, ``processes`` as ``[(pid, used_mb), ...]``); empty when
        ``nvidia-smi`` is not available.
    """

    def smi(*query: str) -> list[list[str]]:
        out = subprocess.run(
            ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
        return [[c.strip() for c in ln.split(",")] for ln in out.strip().splitlines()]

    try:
        gpus = [
            {
                "index": int(index),
                "uuid": uuid,
                "free_gb": (float(total) - float(used)) / 1024,
                "total_gb": float(total) / 1024,
                "util": float(util),
                "processes": [],
            }
            for index, uuid, used, total, util in smi(
                "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu"
            )
        ]
        for uuid, pid, used in smi("--query-compute-apps=gpu_uuid,pid,used_memory"):
            for gpu in gpus:
                if gpu["uuid"] == uuid:
                    gpu["processes"].append((int(pid), float(used)))
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    return gpus


def setup_device(args: argparse.Namespace) -> None:
    """Choose a GPU and cap the memory of this process (shared machines).

    Must run before the first CUDA call. For ``--device cuda``:

    1. All GPUs (those listed in ``CUDA_VISIBLE_DEVICES`` if it is set) are
       queried with ``nvidia-smi``; the one with the most free memory is
       chosen, preferring GPUs without another compute process.
    2. If it has less than ``--min-free-gb`` free the benchmark exits
       without touching the device: with so little head room our
       allocations could make another job fail.
    3. The process is capped with
       ``torch.cuda.set_per_process_memory_fraction`` at ``min(--max-gpu-gb,
       free - --gpu-margin-gb)``. A workload that needs more fails with an
       out-of-memory error inside this process only, which the benchmarks
       record as a skipped cell.
    4. ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` is set so freed
       memory can be returned to the device between implementations.

    Sets ``args.max_gb`` (the cap, used by :func:`require_memory`) and
    ``args.gpu`` (what was chosen and what else was running, for the
    report).
    """
    args.gpu = None
    args.max_gb = args.max_gpu_gb
    if args.device != "cuda":
        return
    gpus = query_gpus()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        allowed = {v.strip() for v in visible.split(",")}
        gpus = [g for g in gpus if str(g["index"]) in allowed or g["uuid"] in allowed]
    if not gpus:
        raise SystemExit("No GPU found by nvidia-smi; use --device cpu.")
    best = max(gpus, key=lambda g: (not g["processes"], g["free_gb"], -g["util"]))
    if best["processes"]:  # no idle GPU: the freest one, whoever is on it
        best = max(gpus, key=lambda g: (g["free_gb"], -g["util"]))
    summary = "; ".join(
        f"GPU {g['index']}: {g['free_gb']:.1f} GB free, {g['util']:.0f} % util, "
        f"{len(g['processes'])} other process(es)"
        for g in gpus
    )
    if best["free_gb"] < args.min_free_gb:
        raise SystemExit(
            f"Not running: no GPU has {args.min_free_gb:g} GB free ({summary}). "
            "Wait for a free GPU; lower --min-free-gb only on a machine that "
            "nobody else is using."
        )
    cap_gb = min(args.max_gpu_gb, best["free_gb"] - args.gpu_margin_gb)
    if cap_gb <= 0:
        raise SystemExit(f"Not running: no memory left after the margin ({summary}).")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best["index"])
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gb / best["total_gb"]), 0)
    args.max_gb = cap_gb
    args.gpu = {
        "index": best["index"],
        "free_gb_at_launch": best["free_gb"],
        "total_gb": best["total_gb"],
        "util_at_launch": best["util"],
        "other_processes": best["processes"],
        "memory_cap_gb": cap_gb,
        "all_gpus": summary,
    }
    print(
        f"Using GPU {best['index']} ({best['free_gb']:.1f} GB free, "
        f"{best['util']:.0f} % util, other processes: {best['processes']}); "
        f"memory cap {cap_gb:.1f} GB",
        flush=True,
    )


def is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error)


def env_info(device: str) -> dict:
    """Hardware and software versions of this run."""
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "device": device,
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        # ``/proc/cpuinfo`` is absent on non-Linux benchmark hosts.
        pass
    try:
        import triton

        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = None
    try:
        import torch_sparse

        info["torch_sparse"] = torch_sparse.__version__
    except ImportError:
        info["torch_sparse"] = None
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["gpu_state_at_start"] = gpu_state()
    return info
