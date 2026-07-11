"""Benchmark GLIF3 + AlphaPSC RSNN providers on mice_column_v1.

This benchmark compares seven forward providers on the same recurrent
connectome graph:

* ``torch_compile``
* ``event_pre_span``
* ``event_post_span``
* the same three providers captured with CUDA Graphs
* ``persistent``

The recurrent graph is loaded in the same processed-parquet form used by the
``connectome_dataset`` mice_column_v1 loader. The sweep axis remains the number
of simulation time steps.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CONNECTOME_DATASET_ROOT = REPO_ROOT / "connectome_dataset"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CONNECTOME_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(CONNECTOME_DATASET_ROOT))

from btorch.backend.persistent import plain_version  # noqa: E402
from btorch.backend.persistent_snn import (  # noqa: E402
    EventCSRGraph,
    PersistentSNNParams,
    PersistentSNNState,
    WindowedSpikeEvents,
    make_empty_state,
    persistent_snn_forward,
)
from btorch.sparse import CSR, BinaryEvents, event_sparse_mm  # noqa: E402


Provider = Literal[
    "torch_compile",
    "event_pre_span",
    "event_post_span",
    "torch_compile_cudagraph",
    "event_pre_span_cudagraph",
    "event_post_span_cudagraph",
    "persistent",
]

PROVIDERS: tuple[Provider, ...] = (
    "persistent",
    "torch_compile",
    "event_pre_span",
    "event_post_span",
    "torch_compile_cudagraph",
    "event_pre_span_cudagraph",
    "event_post_span_cudagraph",
)


@dataclass(frozen=True)
class BenchCase:
    """GLIF3 + AlphaPSC benchmark parameter bundle."""

    batch_size: int
    t_steps: int
    event_rate: float
    dt: float = 1.0
    tau: float = 20.0
    tau_syn: float = 5.0
    v_threshold: float = -45.0
    v_reset: float = -60.0
    v_rest: float | None = None
    c_m: float = 2.0
    tau_ref: float = 2.0
    k: tuple[float, float] = (0.1, 0.2)
    asc_amps: tuple[float, float] = (1.0, -2.0)
    psc_g_max: float = 1.0
    input_amplitude: float = 30.0
    weight_scale: float = 0.1

    @property
    def resolved_v_rest(self) -> float:
        """Return GLIF3 resting potential, matching GLIF3's default."""

        return self.v_reset if self.v_rest is None else self.v_rest


@dataclass(frozen=True)
class RSNNResult:
    """Output and final state for one RSNN provider."""

    spikes: torch.Tensor
    v: torch.Tensor
    psc: torch.Tensor
    psc_h: torch.Tensor
    asc: torch.Tensor
    refractory: torch.Tensor


@dataclass(frozen=True)
class PersistentBenchmarkWorkspace:
    """Preallocated tensors for persistent dense benchmark replay."""

    events: WindowedSpikeEvents
    state: PersistentSNNState
    params: PersistentSNNParams
    dense_spikes: torch.Tensor
    v_out: torch.Tensor
    psc_out: torch.Tensor
    psc_h_out: torch.Tensor
    asc_out: torch.Tensor
    refractory_out: torch.Tensor
    input_current: torch.Tensor
    spike_queue_batch: torch.Tensor
    spike_queue_pre: torch.Tensor
    spike_count: torch.Tensor
    work_counter: torch.Tensor
    event_counts: torch.Tensor
    event_indices_full: torch.Tensor
    overflow: torch.Tensor


def benchmark_fig_path() -> Path:
    """Return this benchmark script's figure output directory."""

    path = REPO_ROOT / "fig" / "benchmark" / Path(__file__).with_suffix("").name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _is_cudagraph_provider(provider: Provider) -> bool:
    return provider.endswith("_cudagraph")


def provider_available(provider: Provider, device: torch.device) -> tuple[bool, str]:
    """Return whether a provider can run in this environment."""

    base = provider.removesuffix("_cudagraph")
    if _is_cudagraph_provider(provider) and device.type != "cuda":
        return False, "requires_cuda"
    if _is_cudagraph_provider(provider) and base.startswith("event_"):
        return False, "triton_cudagraph_unsupported"
    if base in ("event_pre_span", "event_post_span"):
        if device.type != "cuda":
            return False, "requires_cuda"
        if not _triton_available():
            return False, "requires_triton"
        return True, "ok"
    if base == "torch_compile":
        if not hasattr(torch, "compile"):
            return False, "torch_compile_missing"
        if platform.system() != "Linux":
            return False, "torch_compile_unsupported_platform"
        return True, "ok"
    if base == "persistent":
        if device.type != "cuda":
            return False, "requires_cuda"
        return True, "ok"
    return False, "unknown_provider"


def _cuda_device_index(device: torch.device) -> int:
    """Return the concrete CUDA device index for ``device``."""

    if device.type != "cuda":
        raise ValueError(f"Expected CUDA device, got {device}.")
    if device.index is not None:
        return device.index
    return torch.cuda.current_device()


def query_other_gpu_compute_processes(device: torch.device) -> list[str]:
    """Return non-current compute processes using the selected CUDA device."""

    device_index = _cuda_device_index(device)
    command = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to query GPU processes with nvidia-smi: {exc}")

    current_pid = os.getpid()
    processes = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=2)]
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid == current_pid:
            continue
        name = parts[1] if len(parts) > 1 else "unknown"
        memory = parts[2] if len(parts) > 2 else "unknown"
        processes.append(f"pid={pid} name={name} gpu_mem_mb={memory}")
    return processes


def wait_for_gpu_exclusive(
    device: torch.device,
    *,
    action: Literal["wait", "skip", "error", "off"],
    timeout_s: float,
    poll_interval_s: float,
) -> tuple[bool, str]:
    """Ensure no other compute process is using the benchmark GPU."""

    if action == "off":
        return True, "ok"
    deadline = time.monotonic() + timeout_s
    while True:
        processes = query_other_gpu_compute_processes(device)
        if not processes:
            return True, "ok"
        reason = "; ".join(processes)
        if action == "error":
            raise RuntimeError(f"GPU is busy: {reason}")
        if action == "skip":
            return False, f"gpu_busy:{reason}"
        if time.monotonic() >= deadline:
            return False, f"gpu_busy_timeout:{reason}"
        print(f"Waiting for exclusive GPU access: {reason}", flush=True)
        time.sleep(poll_interval_s)


def load_mice_column_v1_matrix(
    root: Path,
    *,
    use_weights: bool,
    max_neurons: int,
) -> sp.csr_array:
    """Load mice_column_v1 processed parquet files as a scipy CSR matrix."""

    from connectome_dataset.graph_loader import load_mice_column_v1

    matrix = load_mice_column_v1(root, use_weights=use_weights).tocsr()
    if max_neurons > 0:
        matrix = matrix[:max_neurons, :max_neurons].tocsr()
        matrix.eliminate_zeros()
    return sp.csr_array(matrix, dtype="float32")


def scipy_to_btorch_csr(matrix: sp.csr_array, device: torch.device) -> CSR:
    """Convert scipy CSR to btorch CSR with float32 values."""

    return CSR.from_scipy(matrix, device=device, dtype=torch.float32)


def csr_to_dense(matrix: CSR) -> torch.Tensor:
    """Materialize a CSR matrix as dense ``(N_pre, N_post)`` weights."""

    n_pre, n_post = matrix.shape
    dense = torch.zeros(
        n_pre,
        n_post,
        device=matrix.data.device,
        dtype=matrix.data.dtype,
    )
    indptr = matrix.indptr
    for pre in range(n_pre):
        start = int(indptr[pre].item())
        end = int(indptr[pre + 1].item())
        if end > start:
            dense[pre, matrix.indices[start:end]] = matrix.effective_values()[start:end]
    return dense


def csr_to_persistent_graph(matrix: CSR) -> EventCSRGraph:
    """Convert the benchmark CSR graph to the persistent graph contract."""

    return EventCSRGraph(
        indptr=matrix.indptr.to(torch.int32).contiguous(),
        indices=matrix.indices.to(torch.int32).contiguous(),
        weight=matrix.effective_values().to(torch.float32).contiguous(),
        delay=None,
        shape=matrix.shape,
    )


def scale_csr_weights(matrix: sp.csr_array, weight_scale: float) -> sp.csr_array:
    """Return a copy with normalized benchmark weights."""

    out = matrix.copy().astype("float32")
    if out.nnz == 0:
        return out
    out.data = out.data.astype("float32", copy=False)
    max_abs = float(abs(out.data).max())
    if max_abs > 0.0:
        out.data = out.data / max_abs * weight_scale
    return out


def make_input_sequence(
    case: BenchCase,
    n_neuron: int,
    device: torch.device,
) -> torch.Tensor:
    """Create deterministic sparse external input currents."""

    total = case.t_steps * case.batch_size * n_neuron
    n_active = max(0, min(total, int(round(total * case.event_rate))))
    flat = torch.zeros(total, device=device, dtype=torch.float32)
    if n_active > 0:
        stride = max(1, total // max(n_active, 1))
        active = (torch.arange(n_active, device=device) * stride) % total
        flat[active.long()] = case.input_amplitude
    return flat.reshape(case.t_steps, case.batch_size, n_neuron)


def dense_to_windowed_events(
    x_seq: torch.Tensor,
    threshold: float = 0.0,
) -> WindowedSpikeEvents:
    """Convert dense ``(T, B, N)`` events to time-batch bucketed events."""

    t_steps, batch_size, n_neuron = x_seq.shape
    active = x_seq > threshold
    counts = active.sum(dim=2, dtype=torch.int32).reshape(-1)
    offsets = torch.zeros(counts.numel() + 1, device=x_seq.device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts, dim=0)
    indices = torch.nonzero(active.reshape(-1, n_neuron), as_tuple=False)[:, 1]
    indices = indices.to(torch.int32).contiguous()
    values = x_seq[active].to(torch.float32).contiguous()
    return WindowedSpikeEvents(
        offsets=offsets.contiguous(),
        indices=indices,
        values=values,
        shape=(t_steps, batch_size, n_neuron),
    )


def _initial_state(
    batch_size: int,
    n_neuron: int,
    case: BenchCase,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    v = torch.full(
        (batch_size, n_neuron),
        case.v_reset,
        device=device,
        dtype=torch.float32,
    )
    psc = torch.zeros_like(v)
    psc_h = torch.zeros_like(v)
    asc = torch.zeros(batch_size, n_neuron, 2, device=device, dtype=torch.float32)
    refractory = torch.zeros_like(v)
    return v, psc, psc_h, asc, refractory


def glif_alpha_step(
    v: torch.Tensor,
    psc: torch.Tensor,
    psc_h: torch.Tensor,
    asc: torch.Tensor,
    refractory: torch.Tensor,
    external_current: torch.Tensor,
    case: BenchCase,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Advance one GLIF3 + AlphaPSC step."""

    syn_decay = math.exp(-case.dt / case.tau_syn)
    mem_decay = math.exp(-case.dt / case.tau)
    asc_decay = torch.tensor(
        [math.exp(-case.dt * case.k[0]), math.exp(-case.dt * case.k[1])],
        device=v.device,
        dtype=v.dtype,
    )
    asc_amps = torch.tensor(case.asc_amps, device=v.device, dtype=v.dtype)

    psc = syn_decay * psc + (1.0 - syn_decay) * psc_h
    psc_h = syn_decay * psc_h
    asc_before = asc
    asc = asc * asc_decay

    current = psc + external_current + asc_before.sum(dim=-1)
    v_inf = case.resolved_v_rest + case.tau * current / case.c_m
    v_pre = v_inf + (v - v_inf) * mem_decay
    can_fire = refractory <= 0.0
    spikes = ((v_pre >= case.v_threshold) & can_fire).to(v.dtype)
    v = v_pre - (case.v_threshold - case.v_reset) * spikes
    refractory = torch.clamp(refractory - case.dt, min=0.0)
    refractory = torch.where(
        spikes > 0,
        torch.full_like(refractory, max(case.tau_ref - case.dt, 0.0)),
        refractory,
    )
    asc = asc + asc_amps * spikes[..., None]
    return spikes, v, psc, psc_h, asc, refractory


def dense_rsnn_forward(
    x_seq: torch.Tensor,
    weight_dense: torch.Tensor,
    case: BenchCase,
) -> RSNNResult:
    """Dense GLIF3 + AlphaPSC RSNN forward."""

    batch_size, n_neuron = x_seq.shape[1], x_seq.shape[2]
    v, psc, psc_h, asc, refractory = _initial_state(
        batch_size,
        n_neuron,
        case,
        x_seq.device,
    )
    spikes = []
    for t in range(case.t_steps):
        z, v, psc, psc_h, asc, refractory = glif_alpha_step(
            v,
            psc,
            psc_h,
            asc,
            refractory,
            x_seq[t],
            case,
        )
        psc_h = psc_h + case.psc_g_max * (z @ weight_dense)
        spikes.append(z)
    return RSNNResult(torch.stack(spikes), v, psc, psc_h, asc, refractory)


def event_rsnn_forward(
    x_seq: torch.Tensor,
    matrix: CSR,
    case: BenchCase,
    *,
    schedule: Literal["pre_span", "post_span"],
    max_events: int | None = None,
) -> RSNNResult:
    """Event-driven forward using the main btorch sparse event path."""

    batch_size, n_neuron = x_seq.shape[1], x_seq.shape[2]
    v, psc, psc_h, asc, refractory = _initial_state(
        batch_size,
        n_neuron,
        case,
        x_seq.device,
    )
    spikes = []
    for t in range(case.t_steps):
        z, v, psc, psc_h, asc, refractory = glif_alpha_step(
            v,
            psc,
            psc_h,
            asc,
            refractory,
            x_seq[t],
            case,
        )
        recurrent = event_sparse_mm(
            matrix,
            BinaryEvents(z),
            schedule=schedule,
            max_events=max_events,
        )
        psc_h = psc_h + case.psc_g_max * recurrent
        spikes.append(z)
    return RSNNResult(torch.stack(spikes), v, psc, psc_h, asc, refractory)


def persistent_rsnn_forward(
    x_seq: torch.Tensor,
    graph: EventCSRGraph,
    case: BenchCase,
    *,
    backend: str,
) -> RSNNResult:
    """Persistent GLIF3 + AlphaPSC provider."""

    events = dense_to_windowed_events(x_seq)
    state = make_empty_state(
        case.batch_size,
        x_seq.shape[2],
        device=x_seq.device,
        refractory=True,
        psc_h=True,
        n_asc=2,
    )
    state.v.fill_(case.v_reset)
    params = PersistentSNNParams(
        dt=case.dt,
        tau=case.tau,
        tau_syn=case.tau_syn,
        v_threshold=case.v_threshold,
        v_reset=case.v_reset,
        v_rest=case.v_rest,
        c_m=case.c_m,
        tau_ref=case.tau_ref,
        k=case.k,
        asc_amps=case.asc_amps,
        psc_g_max=case.psc_g_max,
        hard_reset=False,
        window_size=case.t_steps,
    )
    out = persistent_snn_forward(
        events,
        graph,
        state,
        params,
        backend=backend,
        return_mode="dense",
    )
    assert out.spikes is not None
    assert out.state.psc_h is not None
    assert out.state.asc is not None
    assert out.state.refractory is not None
    return RSNNResult(
        out.spikes,
        out.state.v,
        out.state.psc,
        out.state.psc_h,
        out.state.asc,
        out.state.refractory,
    )


def make_persistent_workspace(
    x_seq: torch.Tensor,
    case: BenchCase,
) -> PersistentBenchmarkWorkspace:
    """Create reusable persistent CUDA output and workspace tensors."""

    events = dense_to_windowed_events(x_seq)
    state = make_empty_state(
        case.batch_size,
        x_seq.shape[2],
        device=x_seq.device,
        refractory=True,
        psc_h=True,
        n_asc=2,
    )
    state.v.fill_(case.v_reset)
    assert state.psc_h is not None
    assert state.asc is not None
    assert state.refractory is not None
    params = PersistentSNNParams(
        dt=case.dt,
        tau=case.tau,
        tau_syn=case.tau_syn,
        v_threshold=case.v_threshold,
        v_reset=case.v_reset,
        v_rest=case.v_rest,
        c_m=case.c_m,
        tau_ref=case.tau_ref,
        k=case.k,
        asc_amps=case.asc_amps,
        psc_g_max=case.psc_g_max,
        hard_reset=False,
        window_size=case.t_steps,
    )
    batch_size, n_neuron = x_seq.shape[1], x_seq.shape[2]
    dense_spikes = torch.empty_like(x_seq)
    event_counts = torch.empty(0, device=x_seq.device, dtype=torch.int32)
    event_indices_full = torch.empty(0, device=x_seq.device, dtype=torch.int32)
    return PersistentBenchmarkWorkspace(
        events=events,
        state=state,
        params=params,
        dense_spikes=dense_spikes,
        v_out=torch.empty_like(state.v),
        psc_out=torch.empty_like(state.psc),
        psc_h_out=torch.empty_like(state.psc_h),
        asc_out=torch.empty_like(state.asc),
        refractory_out=torch.empty_like(state.refractory),
        input_current=torch.empty_like(state.v),
        spike_queue_batch=torch.empty(
            batch_size * n_neuron,
            device=x_seq.device,
            dtype=torch.int32,
        ),
        spike_queue_pre=torch.empty(
            batch_size * n_neuron,
            device=x_seq.device,
            dtype=torch.int32,
        ),
        spike_count=torch.empty(1, device=x_seq.device, dtype=torch.int32),
        work_counter=torch.empty(1, device=x_seq.device, dtype=torch.int32),
        event_counts=event_counts,
        event_indices_full=event_indices_full,
        overflow=torch.empty(1, device=x_seq.device, dtype=torch.int32),
    )


def persistent_workspace_forward(
    workspace: PersistentBenchmarkWorkspace,
    graph: EventCSRGraph,
) -> RSNNResult:
    """Run persistent dense forward with reusable output/workspace tensors."""

    plain_version.load()
    graph_delay = (
        graph.delay
        if graph.delay is not None
        else torch.empty(
            (0,),
            device=graph.indices.device,
            dtype=graph.indices.dtype,
        )
    )
    event_values = (
        workspace.events.values
        if workspace.events.values is not None
        else torch.empty(
            (0,),
            device=workspace.events.indices.device,
            dtype=workspace.state.v.dtype,
        )
    )
    out = torch.ops.btorch_cuda.persistent_snn_forward_dense_workspace(
        workspace.events.offsets,
        workspace.events.indices,
        event_values,
        workspace.events.values is not None,
        graph.indptr,
        graph.indices,
        graph.weight,
        graph_delay,
        graph.delay is not None,
        workspace.state.v,
        workspace.state.psc,
        workspace.state.psc_h,
        workspace.state.asc,
        workspace.state.refractory,
        workspace.dense_spikes,
        workspace.v_out,
        workspace.psc_out,
        workspace.psc_h_out,
        workspace.asc_out,
        workspace.refractory_out,
        workspace.input_current,
        workspace.spike_queue_batch,
        workspace.spike_queue_pre,
        workspace.spike_count,
        workspace.work_counter,
        workspace.event_counts,
        workspace.event_indices_full,
        workspace.overflow,
        float(workspace.params.dt),
        float(workspace.params.tau),
        float(workspace.params.tau_syn),
        float(workspace.params.v_threshold),
        float(workspace.params.v_reset),
        float(
            workspace.params.v_reset
            if workspace.params.v_rest is None
            else workspace.params.v_rest
        ),
        float(workspace.params.c_m),
        float(workspace.params.tau_ref),
        float(workspace.params.k[0]),
        float(workspace.params.k[1]),
        float(workspace.params.asc_amps[0]),
        float(workspace.params.asc_amps[1]),
        float(workspace.params.psc_g_max),
        bool(workspace.params.hard_reset),
    )
    return RSNNResult(
        spikes=out[0],
        v=out[1],
        psc=out[2],
        psc_h=out[3],
        asc=out[4],
        refractory=out[5],
    )


class CudaGraphUnavailable(RuntimeError):
    """Raised when a provider cannot be captured safely."""


def _capture_cudagraph(op: Callable[[], RSNNResult]) -> Callable[[], RSNNResult]:
    """Capture a no-arg CUDA forward callable and return a replay callable."""

    try:
        torch.cuda.synchronize()
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                result = op()
        torch.cuda.current_stream().wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = op()
    except Exception as exc:
        raise CudaGraphUnavailable(str(exc)) from exc

    def replay() -> RSNNResult:
        graph.replay()
        return result

    try:
        replay()
        torch.cuda.synchronize()
    except Exception as exc:
        raise CudaGraphUnavailable(str(exc)) from exc
    return replay


def make_provider_runner(
    provider: Provider,
    x_seq: torch.Tensor,
    matrix: CSR,
    graph: EventCSRGraph,
    weight_dense: torch.Tensor,
    case: BenchCase,
    *,
    event_max_events: int | None,
    persistent_backend: str,
) -> Callable[[], RSNNResult]:
    """Create a stable no-arg forward callable for one provider."""

    base = provider.removesuffix("_cudagraph")
    if base == "torch_compile":
        compiled = torch.compile(dense_rsnn_forward)
        op = lambda: compiled(x_seq, weight_dense, case)
    elif base == "event_pre_span":
        op = lambda: event_rsnn_forward(
            x_seq,
            matrix,
            case,
            schedule="pre_span",
            max_events=event_max_events,
        )
    elif base == "event_post_span":
        op = lambda: event_rsnn_forward(
            x_seq,
            matrix,
            case,
            schedule="post_span",
            max_events=event_max_events,
        )
    elif base == "persistent":
        if persistent_backend == "cuda_persistent":
            workspace = make_persistent_workspace(x_seq, case)
            op = lambda: persistent_workspace_forward(workspace, graph)
        else:
            op = lambda: persistent_rsnn_forward(
                x_seq,
                graph,
                case,
                backend=persistent_backend,
            )
    else:
        raise ValueError(f"Unknown provider: {provider}.")

    if _is_cudagraph_provider(provider):
        return _capture_cudagraph(op)
    return op


def correctness_status(
    provider: Provider,
    result: RSNNResult | None,
    reference: RSNNResult,
) -> tuple[str, float]:
    """Compare a provider result to dense eager reference."""

    if result is None:
        return "skipped", float("nan")
    diffs = [
        (result.spikes - reference.spikes).abs().max().item(),
        (result.v - reference.v).abs().max().item(),
        (result.psc - reference.psc).abs().max().item(),
        (result.psc_h - reference.psc_h).abs().max().item(),
        (result.asc - reference.asc).abs().max().item(),
        (result.refractory - reference.refractory).abs().max().item(),
    ]
    max_diff = float(max(diffs))
    torch.testing.assert_close(result.spikes, reference.spikes, atol=0, rtol=0)
    torch.testing.assert_close(result.v, reference.v, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.psc, reference.psc, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.psc_h, reference.psc_h, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.asc, reference.asc, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        result.refractory,
        reference.refractory,
        atol=1e-5,
        rtol=1e-5,
    )
    return "passed", max_diff


def time_ms(
    fn: Callable[[], RSNNResult],
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[float, float, float]:
    """Benchmark a callable with CUDA event + wall-clock timing.

    Returns:
        (gpu_event_ms, wall_clock_ms, fn_body_ms)
    """

    import time

    for _ in range(warmup):
        fn()
    if device.type != "cuda":
        samples_wall = []
        for _ in range(repeat):
            start = time.perf_counter()
            fn()
            samples_wall.append((time.perf_counter() - start) * 1000.0)
        median = float(
            torch.median(torch.tensor(samples_wall, dtype=torch.float64)).item()
        )
        return median, median, median

    torch.cuda.synchronize()
    samples_event = []
    samples_wall = []
    samples_inside = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        end.record()
        end.synchronize()
        t2 = time.perf_counter()
        samples_event.append(start.elapsed_time(end))
        samples_wall.append((t2 - t0) * 1000.0)
        samples_inside.append((t1 - t0) * 1000.0)
    return (
        float(torch.median(torch.tensor(samples_event, dtype=torch.float64)).item()),
        float(torch.median(torch.tensor(samples_wall, dtype=torch.float64)).item()),
        float(torch.median(torch.tensor(samples_inside, dtype=torch.float64)).item()),
    )


def _empty_row(
    case: BenchCase,
    provider: Provider,
    device: torch.device,
    status: str,
    *,
    graph_name: str,
    n_neuron: int,
    edge_count: int,
) -> dict[str, float | int | str]:
    return {
        "provider": provider,
        "backend_mode": provider.removesuffix("_cudagraph"),
        "uses_cudagraph": int(_is_cudagraph_provider(provider)),
        "graph_name": graph_name,
        "device": device.type,
        "n_neuron": n_neuron,
        "batch_size": case.batch_size,
        "t_steps": case.t_steps,
        "event_rate": case.event_rate,
        "event_count": 0,
        "edge_count": edge_count,
        "correctness_status": status,
        "correctness_max_abs_diff": float("nan"),
        "latency_ms": float("nan"),
        "wall_clock_ms": float("nan"),
        "fn_body_ms": float("nan"),
        "speedup_vs_persistent": float("nan"),
    }


def bench_case(
    case: BenchCase,
    *,
    graph_name: str,
    matrix: CSR,
    graph: EventCSRGraph,
    weight_dense: torch.Tensor,
    device: torch.device,
    providers: tuple[Provider, ...],
    persistent_backend: str,
    gpu_exclusive_action: Literal["wait", "skip", "error", "off"],
    gpu_exclusive_timeout_s: float,
    gpu_exclusive_poll_interval_s: float,
    warmup: int,
    repeat: int,
    skip_correctness: bool,
) -> list[dict[str, float | int | str]]:
    """Measure all requested providers for one case."""

    n_neuron = matrix.shape[0]
    edge_count = int(matrix.indices.numel())
    x_seq = make_input_sequence(case, n_neuron, device)
    reference = dense_rsnn_forward(x_seq, weight_dense, case)
    event_max_events = max(1, int(reference.spikes.count_nonzero(dim=2).max().item()))
    event_count = int(torch.count_nonzero(x_seq).item())

    rows: list[dict[str, float | int | str]] = []
    for provider in providers:
        available, reason = provider_available(provider, device)
        if not available:
            row = _empty_row(
                case,
                provider,
                device,
                f"skipped:{reason}",
                graph_name=graph_name,
                n_neuron=n_neuron,
                edge_count=edge_count,
            )
            row["event_count"] = event_count
            rows.append(row)
            continue

        try:
            op = make_provider_runner(
                provider,
                x_seq,
                matrix,
                graph,
                weight_dense,
                case,
                event_max_events=event_max_events,
                persistent_backend=persistent_backend,
            )
            if provider == "persistent" and persistent_backend == "cuda_persistent":
                ok, reason = wait_for_gpu_exclusive(
                    device,
                    action=gpu_exclusive_action,
                    timeout_s=gpu_exclusive_timeout_s,
                    poll_interval_s=gpu_exclusive_poll_interval_s,
                )
                if not ok:
                    row = _empty_row(
                        case,
                        provider,
                        device,
                        f"skipped:{reason}",
                        graph_name=graph_name,
                        n_neuron=n_neuron,
                        edge_count=edge_count,
                    )
                    row["event_count"] = event_count
                    rows.append(row)
                    print(
                        f"Provider {provider} skipped for T={case.t_steps}: "
                        f"{reason}"
                    )
                    continue
            result = None if skip_correctness else op()
            if skip_correctness:
                status, max_diff = "not_checked", float("nan")
            else:
                status, max_diff = correctness_status(provider, result, reference)
            latency = time_ms(op, warmup=warmup, repeat=repeat, device=device)
            latency_ms, wall_ms, inside_ms = latency
        except CudaGraphUnavailable as exc:
            row = _empty_row(
                case,
                provider,
                device,
                f"skipped:cudagraph_unavailable:{type(exc.__cause__).__name__}",
                graph_name=graph_name,
                n_neuron=n_neuron,
                edge_count=edge_count,
            )
            row["event_count"] = event_count
            rows.append(row)
            print(f"Provider {provider} skipped for T={case.t_steps}: {exc}")
            continue
        except Exception as exc:
            row = _empty_row(
                case,
                provider,
                device,
                f"error:{type(exc).__name__}",
                graph_name=graph_name,
                n_neuron=n_neuron,
                edge_count=edge_count,
            )
            row["event_count"] = event_count
            rows.append(row)
            print(f"Provider {provider} failed for T={case.t_steps}: {exc}")
            continue

        rows.append(
            {
                "provider": provider,
                "backend_mode": provider.removesuffix("_cudagraph"),
                "uses_cudagraph": int(_is_cudagraph_provider(provider)),
                "graph_name": graph_name,
                "device": device.type,
                "n_neuron": n_neuron,
                "batch_size": case.batch_size,
                "t_steps": case.t_steps,
                "event_rate": case.event_rate,
                "event_count": event_count,
                "edge_count": edge_count,
                "correctness_status": status,
                "correctness_max_abs_diff": max_diff,
                "latency_ms": latency_ms,
                "wall_clock_ms": wall_ms,
                "fn_body_ms": inside_ms,
                "speedup_vs_persistent": float("nan"),
            }
        )

    persistent_latency = next(
        (
            float(row["latency_ms"])
            for row in rows
            if row["provider"] == "persistent"
            and not math.isnan(float(row["latency_ms"]))
        ),
        float("nan"),
    )
    for row in rows:
        latency = float(row["latency_ms"])
        if (
            not math.isnan(persistent_latency)
            and not math.isnan(latency)
            and latency > 0
        ):
            row["speedup_vs_persistent"] = persistent_latency / latency
    return rows


def sweep_cases(args: argparse.Namespace) -> list[BenchCase]:
    """Build the time-step sweep grid."""

    return [
        BenchCase(
            batch_size=args.batch_size,
            t_steps=t_steps,
            event_rate=event_rate,
            dt=args.dt,
            tau=args.tau,
            tau_syn=args.tau_syn,
            v_threshold=args.v_threshold,
            v_reset=args.v_reset,
            v_rest=args.v_rest,
            c_m=args.c_m,
            tau_ref=args.tau_ref,
            k=(args.k[0], args.k[1]),
            asc_amps=(args.asc_amps[0], args.asc_amps[1]),
            psc_g_max=args.psc_g_max,
            input_amplitude=args.input_amplitude,
            weight_scale=args.weight_scale,
        )
        for t_steps, event_rate in itertools.product(args.t_steps, args.event_rate)
    ]


def save_csv(rows: list[dict[str, float | int | str]], output_dir: Path) -> Path:
    """Save benchmark rows as CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "persistent_snn_latency.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _finite_rows(rows, y_key):
    return [row for row in rows if not math.isnan(float(row[y_key]))]


def plot_latency(rows: list[dict[str, float | int | str]], output_dir: Path) -> Path:
    """Plot latency and speedup comparisons across providers."""

    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {
        "torch_compile": "#2ca02c",
        "event_pre_span": "#1f77b4",
        "event_post_span": "#ff7f0e",
        "torch_compile_cudagraph": "#006d2c",
        "event_pre_span_cudagraph": "#08519c",
        "event_post_span_cudagraph": "#a63603",
        "persistent": "#7b3294",
    }
    markers = {
        "torch_compile": "^",
        "event_pre_span": "o",
        "event_post_span": "s",
        "torch_compile_cudagraph": "v",
        "event_pre_span_cudagraph": "D",
        "event_post_span_cudagraph": "P",
        "persistent": "x",
    }
    specs = [
        ("latency_ms", "Median latency (ms)"),
        ("speedup_vs_persistent", "Speedup vs persistent"),
    ]

    for ax, (y_key, ylabel) in zip(axes, specs):
        for provider in PROVIDERS:
            provider_rows = [
                row for row in _finite_rows(rows, y_key) if row["provider"] == provider
            ]
            if not provider_rows:
                continue
            fixed_event_rate = min({row["event_rate"] for row in provider_rows})
            provider_rows = [
                row for row in provider_rows if row["event_rate"] == fixed_event_rate
            ]
            provider_rows = sorted(provider_rows, key=lambda row: int(row["t_steps"]))
            ax.plot(
                [int(row["t_steps"]) for row in provider_rows],
                [float(row[y_key]) for row in provider_rows],
                label=provider,
                marker=markers[provider],
                linewidth=2,
                color=colors[provider],
            )
        ax.set_xlabel("Time steps")
        ax.set_ylabel(ylabel)
        if y_key == "latency_ms":
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8)

    fig.suptitle("mice_column_v1 GLIF3 + AlphaPSC Provider Comparison")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "persistent_snn_latency_sweep.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def print_latency_table(rows: list[dict[str, float | int | str]]) -> None:
    """Print per-time-step provider latencies to stdout."""

    print("\nLatency by time step (median GPU event / wall / fn_body, ms)")
    header = (
        f"{'T':>6}  {'provider':<28}  {'gpu_event':>12}  "
        f"{'wall_clock':>12}  {'fn_body':>12}  "
        f"{'vs_persistent':>14}  {'status':<18}"
    )
    print(header)
    print("-" * len(header))
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            int(row["t_steps"]),
            float(row["event_rate"]),
            str(row["provider"]),
        ),
    )
    for row in sorted_rows:
        latency = float(row["latency_ms"])
        wall = float(row.get("wall_clock_ms", float("nan")))
        inside = float(row.get("fn_body_ms", float("nan")))
        speedup = float(row["speedup_vs_persistent"])
        latency_text = "nan" if math.isnan(latency) else f"{latency:.6f}"
        wall_text = "nan" if math.isnan(wall) else f"{wall:.6f}"
        inside_text = "nan" if math.isnan(inside) else f"{inside:.6f}"
        speedup_text = "nan" if math.isnan(speedup) else f"{speedup:.6f}"
        print(
            f"{int(row['t_steps']):6d}  "
            f"{str(row['provider']):<28}  "
            f"{latency_text:>12}  "
            f"{wall_text:>12}  "
            f"{inside_text:>12}  "
            f"{speedup_text:>14}  "
            f"{str(row['correctness_status']):<18}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument(
        "--persistent-backend",
        default="cuda_persistent",
        choices=["auto", "torch_stub", "cuda_persistent"],
        help="Persistent operator backend.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
    )
    parser.add_argument(
        "--connectome-root",
        type=Path,
        default=CONNECTOME_DATASET_ROOT / "data" / "external" / "mice_column_v1",
        help="Directory containing mice_column_v1 processed parquet files.",
    )
    parser.add_argument("--use-weights", action="store_true")
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=0,
        help="Optional leading neuron subgraph limit; 0 uses the full graph.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--t-steps",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--event-rate",
        type=float,
        nargs="+",
        default=[0.005],
    )
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--tau-syn", type=float, default=5.0)
    parser.add_argument("--v-threshold", type=float, default=-45.0)
    parser.add_argument("--v-reset", type=float, default=-60.0)
    parser.add_argument("--v-rest", type=float, default=None)
    parser.add_argument("--c-m", type=float, default=2.0)
    parser.add_argument("--tau-ref", type=float, default=2.0)
    parser.add_argument("--k", type=float, nargs=2, default=[0.1, 0.2])
    parser.add_argument("--asc-amps", type=float, nargs=2, default=[1.0, -2.0])
    parser.add_argument("--psc-g-max", type=float, default=1.0)
    parser.add_argument("--input-amplitude", type=float, default=30.0)
    parser.add_argument("--weight-scale", type=float, default=0.1)
    parser.add_argument(
        "--gpu-exclusive-check",
        choices=["wait", "skip", "error", "off"],
        default="wait",
        help=(
            "How persistent should handle other compute processes on the "
            "selected GPU before correctness/timing."
        ),
    )
    parser.add_argument(
        "--gpu-exclusive-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for exclusive GPU access when using wait mode.",
    )
    parser.add_argument(
        "--gpu-exclusive-poll-interval",
        type=float,
        default=1.0,
        help="Seconds between nvidia-smi process checks in wait mode.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _device_from_arg(args.device)
    if device.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA providers.")
    output_dir = benchmark_fig_path()

    scipy_matrix = load_mice_column_v1_matrix(
        args.connectome_root,
        use_weights=args.use_weights,
        max_neurons=args.max_neurons,
    )
    scipy_matrix = scale_csr_weights(scipy_matrix, args.weight_scale)
    matrix = scipy_to_btorch_csr(scipy_matrix, device)
    graph = csr_to_persistent_graph(matrix)
    weight_dense = csr_to_dense(matrix)
    cases = sweep_cases(args)
    providers = tuple(args.providers)
    graph_name = "mice_column_v1"
    if args.max_neurons > 0:
        graph_name = f"{graph_name}_first_{args.max_neurons}"

    print(
        f"Loaded {graph_name}: N={matrix.shape[0]:,}, "
        f"E={matrix.indices.numel():,}, device={device.type}"
    )
    rows = []
    for idx, case in enumerate(cases, start=1):
        print(
            f"[{idx}/{len(cases)}] T={case.t_steps} "
            f"event_rate={case.event_rate}"
        )
        rows.extend(
            bench_case(
                case,
                graph_name=graph_name,
                matrix=matrix,
                graph=graph,
                weight_dense=weight_dense,
                device=device,
                providers=providers,
                persistent_backend=args.persistent_backend,
                gpu_exclusive_action=args.gpu_exclusive_check,
                gpu_exclusive_timeout_s=args.gpu_exclusive_timeout,
                gpu_exclusive_poll_interval_s=args.gpu_exclusive_poll_interval,
                warmup=args.warmup,
                repeat=args.repeat,
                skip_correctness=args.skip_correctness,
            )
        )

    print_latency_table(rows)
    csv_path = save_csv(rows, output_dir)
    fig_path_out = plot_latency(rows, output_dir)
    print(f"Benchmark CSV saved to {csv_path}")
    print(f"Latency sweep plot saved to {fig_path_out}")


if __name__ == "__main__":
    main()
