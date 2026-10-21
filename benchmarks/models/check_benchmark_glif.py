"""Check benchmark-branch GLIF primitives against current btorch references.

Each revision runs in a separate subprocess. The parent creates one
canonical CPU input artifact per case, then compares saved CUDA results
after every child has exited, avoiding mixed imports or shared mutable
neuron state.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


BRANCH_REVISION = "4bd0b0a649188173801549dcb970d50b2efd772c"
BRANCH_HASHES = {
    "benchmarks/glif_net/glif_common.py": (
        "547be79f2ce9a2778ecd89ab8959fe481c79e29223398f4c9a233e79bf3f550d"
    ),
    "benchmarks/glif_net/glif_triton.py": (
        "3c7a7e2580a1269d2fdbcc42b04caab773699ca381f5ca20f60de195b60ed6e2"
    ),
    "benchmarks/glif_net/glif_cupy.py": (
        "4fad22fb0c97e53ee9a2177770a1d709daee86cc8961a66a5595fe41d354087f"
    ),
    "benchmarks/glif_net/glif_tilelang.py": (
        "66530bff792fb68cd523bccf1c5477895797dc125d7263600e92dd2721bd016b"
    ),
}
BACKENDS = ("triton", "cupy", "tilelang")
OUTPUT_NAMES = ("spikes", "v_history", "final_v", "final_Iasc")
GRADIENT_NAMES = ("x", "v0", "Iasc0", "asc_amps")
CURRENT_SHARED_SOURCES = (
    "benchmarks/models/check_benchmark_glif.py",
    "btorch/models/neurons/glif.py",
    "btorch/models/surrogate/atan.py",
    "btorch/models/kernels/golden.py",
    "btorch/models/kernels/options.py",
    "btorch/models/kernels/contracts.py",
    "btorch/models/kernels/templates.py",
    "btorch/models/kernels/operators.py",
    "btorch/models/kernels/backends/engine.py",
    "btorch/models/kernels/backends/fragments.py",
)


def _distribution_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", nargs="+", type=int, default=(20, 128), help="Paired T values"
    )
    parser.add_argument(
        "--neurons",
        nargs="+",
        type=int,
        default=(1024, 4096),
        help="Paired neuron-lane values",
    )
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
    parser.add_argument(
        "--branch-root",
        type=Path,
        default=Path("/data1/fanqixuan/src/btorch.worktree/benchmark"),
    )
    parser.add_argument("--forward-atol", type=float, default=5e-6)
    parser.add_argument("--forward-rtol", type=float, default=5e-5)
    parser.add_argument("--gradient-atol", type=float, default=5e-5)
    parser.add_argument("--gradient-rtol", type=float, default=5e-4)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Measure implementations after they pass the numerical gate",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--child", choices=("current", "branch"), help=argparse.SUPPRESS
    )
    parser.add_argument("--implementation", help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        if not all((args.implementation, args.input, args.output, args.root)):
            parser.error(
                "child mode requires --implementation, --input, --output, --root"
            )
        return args
    if len(args.steps) != len(args.neurons):
        parser.error("--steps and --neurons must have equal lengths and are paired")
    if not args.steps or min(*args.steps, *args.neurons) < 1:
        parser.error("steps and neurons must be positive")
    if (
        min(
            args.forward_atol,
            args.forward_rtol,
            args.gradient_atol,
            args.gradient_rtol,
            args.timeout,
        )
        < 0
    ):
        parser.error("tolerances and timeout must be non-negative")
    if args.warmup < 3 or args.repeats < 30:
        parser.error("--warmup must be at least 3 and --repeats at least 30")
    args.backends = tuple(dict.fromkeys(args.backends))
    return args


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root, *arguments):
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def _identity(root, relative_paths):
    root = root.resolve()
    return {
        "root": str(root),
        "commit": _git(root, "rev-parse", "HEAD"),
        "dirty": bool(_git(root, "status", "--porcelain")),
        "sources": {relative: _sha256(root / relative) for relative in relative_paths},
    }


def _diagnostic(exc):
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "exception": type(exc).__name__,
        "message": str(exc)[-1200:],
        "traceback": trace[-5000:],
    }


def _pattern(torch, shape, phase):
    flat = torch.arange(int(torch.tensor(shape).prod().item()), dtype=torch.float32)
    value = torch.sin(flat * (0.011 + phase * 0.002))
    value = value + 0.5 * torch.cos(flat * (0.017 + phase * 0.001))
    return value.reshape(shape)


def _canonical_case(torch, steps, neurons):
    modes = 2
    lane = torch.arange(neurons, dtype=torch.float32)
    time = torch.arange(steps, dtype=torch.float32)[:, None]
    x = 0.48 + 0.42 * torch.sin(time * 0.37 + lane[None] * 0.013)
    x = x + 0.24 * torch.cos(time * 0.11 - lane[None] * 0.019)
    v0 = 0.08 + 0.34 * ((lane.remainder(29.0) + 0.5) / 29.0)
    mode = torch.arange(modes, dtype=torch.float32)[None]
    Iasc0 = 0.035 * torch.sin(lane[:, None] * 0.021 + mode * 1.7)
    k = 0.09 + 0.06 * mode + 0.025 * ((lane[:, None].remainder(17.0)) / 17.0)
    asc_amps = -0.09 + 0.12 * mode + 0.018 * torch.cos(lane[:, None] * 0.027)
    return {
        "steps": steps,
        "neurons": neurons,
        "modes": modes,
        "dt": 0.5,
        "x": x,
        "v0": v0,
        "Iasc0": Iasc0,
        "v_threshold": torch.ones(neurons),
        "v_reset": torch.zeros(neurons),
        "v_rest": torch.full((neurons,), -0.12),
        "c_m": torch.full((neurons,), 1.15),
        "tau": torch.full((neurons,), 3.8),
        "k": k,
        "asc_amps": asc_amps,
        "not_refrac": torch.ones(neurons),
        "cot_spikes": _pattern(torch, (steps, neurons), 0),
        "cot_v_history": _pattern(torch, (steps, neurons), 1),
        "cot_final_v": _pattern(torch, (neurons,), 2),
        "cot_final_Iasc": _pattern(torch, (neurons, modes), 3),
    }


def _leaf(torch, value):
    return value.detach().clone().to(device="cuda").requires_grad_()


def _child_tensors(torch, case):
    x = _leaf(torch, case["x"])
    v0 = _leaf(torch, case["v0"])
    Iasc0 = _leaf(torch, case["Iasc0"])
    asc_amps = _leaf(torch, case["asc_amps"])
    fixed = {
        name: case[name].detach().clone().to(device="cuda")
        for name in (
            "v_threshold",
            "v_reset",
            "v_rest",
            "c_m",
            "tau",
            "k",
            "not_refrac",
        )
    }
    cotangents = tuple(
        case[name].to(device="cuda")
        for name in (
            "cot_spikes",
            "cot_v_history",
            "cot_final_v",
            "cot_final_Iasc",
        )
    )
    return x, v0, Iasc0, asc_amps, fixed, cotangents


def _vjp(torch, outputs, inputs, cotangents):
    return torch.autograd.grad(outputs, inputs, grad_outputs=cotangents)


def _run_current(torch, case, implementation):
    from btorch.models.kernels import CellSpec, HistoryPlan, run_multi_step
    from btorch.models.kernels.golden import python_module_sequence

    x, v0, Iasc0, asc_amps, fixed, cotangents = _child_tensors(torch, case)
    parameters = (
        fixed["v_threshold"],
        fixed["v_reset"],
        fixed["c_m"],
        fixed["tau"],
        fixed["v_rest"],
        fixed["k"],
        asc_amps,
    )
    spec = CellSpec(
        "glif3",
        dt=float(case["dt"]),
        hard_reset=False,
        detach_reset=False,
        surrogate="atan",
        alpha=2.0,
        damping=1.0,
        refractory=False,
    )
    if implementation == "module":
        spikes, final, records = python_module_sequence(
            x, (v0, Iasc0), parameters, spec, ("v",)
        )
        outputs = (spikes, records[0], final[0], final[1])
    else:
        result = run_multi_step(
            x,
            (v0, Iasc0),
            parameters,
            spec,
            HistoryPlan(requested=("v",)),
            backend=implementation,
        )
        outputs = (result[0], result[3], result[1], result[2])
    gradients = _vjp(torch, outputs, (x, v0, Iasc0, asc_amps), cotangents)
    return outputs, gradients


def _run_branch(torch, case, backend):
    module = __import__(
        f"benchmarks.glif_net.glif_{backend}",
        fromlist=[f"glif3_step_{backend}"],
    )
    primitive = getattr(module, f"glif3_step_{backend}").multistep_fused
    x, v0, Iasc0, asc_amps, fixed, cotangents = _child_tensors(torch, case)
    params = {
        "v_th": fixed["v_threshold"],
        "v_reset": fixed["v_reset"],
        "v_rest": fixed["v_rest"],
        "c_m": fixed["c_m"],
        "tau": fixed["tau"],
        "k": fixed["k"].reshape(-1),
        "asc_amps": asc_amps.reshape(-1),
    }
    # Branch primitives are allowed to mutate working state, so every child
    # passes private clones of the canonical initial tensors.
    spikes, v_history, final_v, final_Iasc = primitive(
        x_seq=x,
        v=v0.clone(),
        Iasc=Iasc0.clone(),
        params=params,
        not_refrac=fixed["not_refrac"],
        dt=float(case["dt"]),
        M=int(case["modes"]),
        hard_reset=False,
        alpha=2.0,
    )
    outputs = (spikes, v_history, final_v, final_Iasc)
    gradients = _vjp(torch, outputs, (x, v0, Iasc0, asc_amps), cotangents)
    return outputs, gradients


def _current_workload(torch, case, implementation, training):
    from btorch.models.kernels import CellSpec, HistoryPlan, run_multi_step

    x, v0, Iasc0, asc_amps, fixed, cotangents = _child_tensors(torch, case)
    if not training:
        x, v0, Iasc0, asc_amps = (value.detach() for value in (x, v0, Iasc0, asc_amps))
    parameters = (
        fixed["v_threshold"],
        fixed["v_reset"],
        fixed["c_m"],
        fixed["tau"],
        fixed["v_rest"],
        fixed["k"],
        asc_amps,
    )
    spec = CellSpec(
        "glif3",
        dt=float(case["dt"]),
        hard_reset=False,
        detach_reset=False,
        surrogate="atan",
        alpha=2.0,
        damping=1.0,
        refractory=False,
    )

    def forward_call():
        result = run_multi_step(
            x,
            (v0, Iasc0),
            parameters,
            spec,
            HistoryPlan(requested=("v",)),
            backend=implementation,
        )
        return result[0], result[3], result[1], result[2]

    inputs = (x, v0, Iasc0, asc_amps)

    def training_call():
        outputs = forward_call()
        gradients = _vjp(torch, outputs, inputs, cotangents)
        return outputs, gradients

    return (training_call if training else forward_call), {
        "state_preparation": "none",
    }


def _branch_workload(torch, case, backend, training):
    module = __import__(
        f"benchmarks.glif_net.glif_{backend}",
        fromlist=[f"glif3_step_{backend}"],
    )
    primitive = getattr(module, f"glif3_step_{backend}").multistep_fused
    x, v0, Iasc0, asc_amps, fixed, cotangents = _child_tensors(torch, case)
    if not training:
        x, v0, Iasc0, asc_amps = (value.detach() for value in (x, v0, Iasc0, asc_amps))
    params = {
        "v_th": fixed["v_threshold"],
        "v_reset": fixed["v_reset"],
        "v_rest": fixed["v_rest"],
        "c_m": fixed["c_m"],
        "tau": fixed["tau"],
        "k": fixed["k"].reshape(-1),
        "asc_amps": asc_amps.reshape(-1),
    }

    def forward_call():
        # The branch API may mutate its state arguments. These private clones
        # are part of end-to-end latency and keep every replay at the canonical
        # initial state without changing the VJP path to v0 or Iasc0.
        return primitive(
            x_seq=x,
            v=v0.clone(),
            Iasc=Iasc0.clone(),
            params=params,
            not_refrac=fixed["not_refrac"],
            dt=float(case["dt"]),
            M=int(case["modes"]),
            hard_reset=False,
            alpha=2.0,
        )

    inputs = (x, v0, Iasc0, asc_amps)

    def training_call():
        outputs = forward_call()
        gradients = _vjp(torch, outputs, inputs, cotangents)
        return outputs, gradients

    return (training_call if training else forward_call), {
        "state_preparation": "v0/Iasc0 clones included in end-to-end timing",
    }


def _tensor_bytes(values):
    return sum(value.numel() * value.element_size() for value in values)


def _sample_summary(values):
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p10": ordered[int(0.1 * (len(ordered) - 1))],
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "samples": len(ordered),
    }


def _measure_eager(torch, make_workload, training, warmup, repeats):
    call, metadata = make_workload(training)
    for _ in range(warmup):
        result = call()
    del result
    torch.cuda.synchronize()
    host_us = []
    device_us = []
    result = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter_ns()
        start.record()
        result = call()
        end.record()
        host_us.append((time.perf_counter_ns() - host_start) / 1000.0)
        end.synchronize()
        device_us.append(start.elapsed_time(end) * 1000.0)
    outputs, gradients = result if training else (result, ())
    return {
        "host_enqueue_us": _sample_summary(host_us),
        "device_end_to_end_us": _sample_summary(device_us),
        "return_bytes": {
            "outputs": _tensor_bytes(outputs),
            "gradients": _tensor_bytes(gradients),
        },
        **metadata,
    }


def _measure_peak_allocated(torch, make_workload, training, warmup):
    call, _ = make_workload(training)
    for _ in range(warmup):
        result = call()
    del result
    torch.cuda.synchronize()
    gc.collect()
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        result = call()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        outputs, gradients = result if training else (result, ())
        measurement = {
            "baseline_bytes": baseline,
            "peak_bytes": peak,
            "incremental_peak_bytes": peak - baseline,
            "live_return_bytes": _tensor_bytes(outputs) + _tensor_bytes(gradients),
        }
    finally:
        if gc_was_enabled:
            gc.enable()
    return measurement


def _measure_captured(torch, make_workload, training, warmup, repeats):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Leaves, warmup graphs and capture all originate on this stream. This
        # avoids AccumulateGrad dependencies on the legacy default stream.
        call, metadata = make_workload(training)
        for _ in range(warmup):
            static_result = call()
        del static_result
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            static_result = call()
    stream.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    outputs, gradients = static_result if training else (static_result, ())
    return {
        "device_replay_us": _sample_summary(samples),
        "return_bytes": {
            "outputs": _tensor_bytes(outputs),
            "gradients": _tensor_bytes(gradients),
        },
        **metadata,
    }


def _measure_branch_state_clones(torch, case, warmup, repeats):
    _, v0, Iasc0, _, _, _ = _child_tensors(torch, case)
    for _ in range(warmup):
        clones = (v0.clone(), Iasc0.clone())
    del clones
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        clones = (v0.clone(), Iasc0.clone())
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return {
        "device_us": _sample_summary(samples),
        "bytes": _tensor_bytes(clones),
        "included_in_end_to_end": True,
    }


def _timing(torch, case, child, implementation, warmup, repeats):
    if child == "current":
        factory = lambda training: _current_workload(
            torch, case, implementation, training
        )
    else:
        factory = lambda training: _branch_workload(
            torch, case, implementation, training
        )
    return {
        "status": "pass",
        "warmup": warmup,
        "repeats": repeats,
        "forward": {
            "eager": _measure_eager(torch, factory, False, warmup, repeats),
            "captured": _measure_captured(torch, factory, False, warmup, repeats),
            "memory": _measure_peak_allocated(torch, factory, False, warmup),
        },
        "forward_backward": {
            "eager": _measure_eager(torch, factory, True, warmup, repeats),
            "captured": _measure_captured(torch, factory, True, warmup, repeats),
            "memory": _measure_peak_allocated(torch, factory, True, warmup),
        },
        "branch_state_clone": (
            _measure_branch_state_clones(torch, case, warmup, repeats)
            if child == "branch"
            else None
        ),
    }


def _child_sources(root, child, implementation):
    if child == "branch":
        return (
            "benchmarks/glif_net/glif_common.py",
            f"benchmarks/glif_net/glif_{implementation}.py",
        )
    backend_source = (
        (f"btorch/models/kernels/backends/{implementation}.py",)
        if implementation != "module"
        else ()
    )
    return (*CURRENT_SHARED_SOURCES, *backend_source)


def _child_main(args):
    import torch

    root = args.root.resolve()
    identity = _identity(root, _child_sources(root, args.child, args.implementation))
    if args.child == "branch":
        if identity["commit"] != BRANCH_REVISION:
            raise RuntimeError(
                f"branch revision {identity['commit']} != required {BRANCH_REVISION}"
            )
        mismatched = {
            name: {"actual": _sha256(root / name), "expected": expected}
            for name, expected in BRANCH_HASHES.items()
            if _sha256(root / name) != expected
        }
        if mismatched:
            raise RuntimeError(f"branch source hash mismatch: {mismatched}")
    import btorch

    loaded = Path(btorch.__file__).resolve()
    if not loaded.is_relative_to(root):
        raise ImportError(f"loaded btorch from {loaded}, expected under {root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    environment = {
        "torch": str(torch.__version__),
        "triton": _distribution_version("triton"),
        "tilelang": _distribution_version("tilelang"),
        "gpu": torch.cuda.get_device_name(),
        "loaded_btorch": str(loaded),
    }
    case = torch.load(args.input, map_location="cpu", weights_only=True)
    outputs, gradients = (
        _run_current(torch, case, args.implementation)
        if args.child == "current"
        else _run_branch(torch, case, args.implementation)
    )
    timing = (
        _timing(
            torch,
            case,
            args.child,
            args.implementation,
            args.warmup,
            args.repeats,
        )
        if args.timing
        else None
    )
    torch.cuda.synchronize()
    artifact = {
        "status": "pass",
        "implementation": args.implementation,
        "revision": args.child,
        "identity": identity,
        "environment": environment,
        "spike_count": int(outputs[0].detach().sum().item()),
        "num_spike_slots": outputs[0].numel(),
        "outputs": {
            name: value.detach().cpu() for name, value in zip(OUTPUT_NAMES, outputs)
        },
        "gradients": {
            name: value.detach().cpu() for name, value in zip(GRADIENT_NAMES, gradients)
        },
        "timing": timing,
    }
    torch.save(artifact, args.output)


def _metric(torch, actual, expected, atol, rtol):
    if actual.shape != expected.shape:
        return {
            "status": "fail",
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    finite = actual.isfinite() & expected.isfinite()
    difference = (actual - expected).abs()
    max_abs = difference[finite].max().item() if finite.any() else None
    denominator = expected.abs().clamp_min(torch.finfo(expected.dtype).eps)
    relative = difference / denominator
    max_rel = relative[finite].max().item() if finite.any() else None
    close = bool(
        actual.isfinite().all()
        and expected.isfinite().all()
        and actual.allclose(expected, atol=atol, rtol=rtol)
    )
    mismatch = ~actual.isclose(expected, atol=atol, rtol=rtol, equal_nan=False)
    locations = mismatch.nonzero()
    return {
        "status": "pass" if close else "fail",
        "max_abs": max_abs,
        "max_rel": max_rel,
        "first_mismatch": locations[0].tolist() if locations.numel() else None,
        "nonfinite_actual": int((~actual.isfinite()).sum().item()),
        "nonfinite_expected": int((~expected.isfinite()).sum().item()),
    }


def _compare(torch, candidate, reference, args):
    forward = {
        name: _metric(
            torch,
            candidate["outputs"][name],
            reference["outputs"][name],
            args.forward_atol,
            args.forward_rtol,
        )
        for name in OUTPUT_NAMES
    }
    backward = {
        name: _metric(
            torch,
            candidate["gradients"][name],
            reference["gradients"][name],
            args.gradient_atol,
            args.gradient_rtol,
        )
        for name in GRADIENT_NAMES
    }
    spike_status = (
        "pass" if candidate["spike_count"] == reference["spike_count"] else "fail"
    )
    activity_status = (
        "pass"
        if 0 < candidate["spike_count"] < candidate["num_spike_slots"]
        and 0 < reference["spike_count"] < reference["num_spike_slots"]
        else "fail"
    )
    status = (
        "pass"
        if all(
            value["status"] == "pass"
            for value in (*forward.values(), *backward.values())
        )
        and spike_status == "pass"
        and activity_status == "pass"
        else "fail"
    )
    return {
        "status": status,
        "spike_count": candidate["spike_count"],
        "expected_spike_count": reference["spike_count"],
        "spike_count_status": spike_status,
        "nontrivial_spike_activity_status": activity_status,
        "forward": forward,
        "backward": backward,
        "identity": candidate["identity"],
        "environment": candidate["environment"],
        "timing": candidate.get("timing"),
    }


def _launch(
    script,
    root,
    child,
    implementation,
    input_path,
    output_path,
    timeout,
    *,
    timing=False,
    warmup=3,
    repeats=30,
):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(root), environment.get("PYTHONPATH")))
    )
    command = (
        sys.executable,
        str(script),
        "--child",
        child,
        "--implementation",
        implementation,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--root",
        str(root),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
    )
    if timing:
        command = (*command, "--timing")
    try:
        process = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return {"status": "error", "diagnostic": _diagnostic(exc)}
    if process.returncode or not output_path.is_file():
        return {
            "status": "error",
            "returncode": process.returncode,
            "stdout": process.stdout[-4000:],
            "stderr": process.stderr[-6000:],
        }
    import torch

    return torch.load(output_path, map_location="cpu", weights_only=True)


def _timing_limit(reference_us):
    return max(1.2 * reference_us, reference_us + 5.0)


def _performance_comparison(candidate, reference, *, same_backend):
    if candidate.get("timing", {}).get("status") != "pass":
        return {"status": "error", "reason": "candidate timing unavailable"}
    if reference.get("timing", {}).get("status") != "pass":
        return {
            "status": "not_counted",
            "reason": "numerically valid reference timing unavailable",
        }
    checks = {}
    for phase in ("forward", "forward_backward"):
        candidate_us = candidate["timing"][phase]["captured"]["device_replay_us"][
            "median"
        ]
        reference_us = reference["timing"][phase]["captured"]["device_replay_us"][
            "median"
        ]
        limit_us = _timing_limit(reference_us)
        checks[f"captured_{phase}_latency"] = {
            "status": "pass" if candidate_us <= limit_us else "fail",
            "candidate_us": candidate_us,
            "reference_us": reference_us,
            "limit_us": limit_us,
        }
    candidate_inference = candidate["timing"]["forward"]["memory"][
        "incremental_peak_bytes"
    ]
    reference_inference = reference["timing"]["forward"]["memory"][
        "incremental_peak_bytes"
    ]
    checks["inference_peak_allocation"] = {
        "status": "pass" if candidate_inference <= reference_inference else "fail",
        "candidate_bytes": candidate_inference,
        "reference_bytes": reference_inference,
    }
    candidate_training = candidate["timing"]["forward_backward"]["memory"][
        "incremental_peak_bytes"
    ]
    reference_training = reference["timing"]["forward_backward"]["memory"][
        "incremental_peak_bytes"
    ]
    checks["training_peak_allocation"] = {
        "status": "pass" if candidate_training <= reference_training else "fail",
        "candidate_bytes": candidate_training,
        "reference_bytes": reference_training,
    }
    status = (
        "pass"
        if all(check["status"] == "pass" for check in checks.values())
        else "fail"
    )
    return {
        "status": status,
        "comparison": "same_backend" if same_backend else "cross_provider",
        "counts_for_acceptance": same_backend,
        "checks": checks,
    }


def _parent_main(args):
    import torch

    current_root = Path(__file__).resolve().parents[2]
    current_gate_identity = _identity(
        current_root, ("benchmarks/models/check_benchmark_glif.py",)
    )
    branch_root = args.branch_root.resolve()
    if not branch_root.is_dir():
        raise FileNotFoundError(branch_root)
    branch_identity = _identity(branch_root, BRANCH_HASHES)
    if branch_identity["commit"] != BRANCH_REVISION:
        raise RuntimeError(
            f"branch revision {branch_identity['commit']} != required {BRANCH_REVISION}"
        )
    if branch_identity["sources"] != BRANCH_HASHES:
        raise RuntimeError("benchmark branch source hashes do not match the gate")
    report = {
        "status": "pass",
        "contract": {
            "cases": [
                {"steps": steps, "neurons": neurons}
                for steps, neurons in zip(args.steps, args.neurons)
            ],
            "dtype": "float32",
            "dt": 0.5,
            "modes": 2,
            "reset": "soft",
            "refractory": False,
            "not_refrac": "all ones",
            "k": "positive per-neuron, per-mode tensor",
            "surrogate": {
                "name": "ATan",
                "normalized_voltage": True,
                "alpha": 2.0,
                "damping": 1.0,
                "detach_reset": False,
            },
            "outputs": list(OUTPUT_NAMES),
            "gradients": list(GRADIENT_NAMES),
            "vjp": "deterministic unit-scale cotangents; no mean reduction",
            "tolerances": {
                "forward": {"atol": args.forward_atol, "rtol": args.forward_rtol},
                "gradient": {
                    "atol": args.gradient_atol,
                    "rtol": args.gradient_rtol,
                },
            },
            "timing": (
                {
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "phases": ["forward", "forward_backward"],
                    "measurements": [
                        "eager host enqueue latency",
                        "eager device end-to-end latency",
                        "captured device replay latency",
                        "incremental peak allocated bytes",
                        "returned tensor bytes",
                    ],
                    "latency_acceptance": "current <= max(1.2 * branch, branch + 5us)",
                    "allocation_acceptance": "current <= numerically valid branch",
                    "invalid_reference": "not_counted",
                }
                if args.timing
                else None
            ),
        },
        "current_gate_identity": current_gate_identity,
        "branch_identity": branch_identity,
        "cases": [],
    }
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="btorch-glif-gate-") as temporary:
        temporary_root = Path(temporary)
        for case_index, (steps, neurons) in enumerate(zip(args.steps, args.neurons)):
            input_path = temporary_root / f"case-{case_index}.pt"
            torch.save(_canonical_case(torch, steps, neurons), input_path)
            module_path = temporary_root / f"case-{case_index}-module.pt"
            reference = _launch(
                script,
                current_root,
                "current",
                "module",
                input_path,
                module_path,
                args.timeout,
            )
            case_report = {
                "steps": steps,
                "neurons": neurons,
                "reference": {
                    key: reference.get(key)
                    for key in (
                        "status",
                        "spike_count",
                        "num_spike_slots",
                        "identity",
                        "environment",
                        "diagnostic",
                        "stderr",
                    )
                    if reference.get(key) is not None
                },
                "current": {},
                "branch": {},
            }
            if reference.get("status") != "pass":
                case_report["status"] = "error"
                report["status"] = "fail"
                report["cases"].append(case_report)
                continue
            for revision, root in (("current", current_root), ("branch", branch_root)):
                for backend in args.backends:
                    output_path = (
                        temporary_root / f"case-{case_index}-{revision}-{backend}.pt"
                    )
                    candidate = _launch(
                        script,
                        root,
                        revision,
                        backend,
                        input_path,
                        output_path,
                        args.timeout,
                    )
                    result = (
                        _compare(torch, candidate, reference, args)
                        if candidate.get("status") == "pass"
                        else candidate
                    )
                    if args.timing and result.get("status") == "pass":
                        timed_path = temporary_root / (
                            f"case-{case_index}-{revision}-{backend}-timed.pt"
                        )
                        timed_candidate = _launch(
                            script,
                            root,
                            revision,
                            backend,
                            input_path,
                            timed_path,
                            args.timeout,
                            timing=True,
                            warmup=args.warmup,
                            repeats=args.repeats,
                        )
                        timed_result = (
                            _compare(torch, timed_candidate, reference, args)
                            if timed_candidate.get("status") == "pass"
                            else timed_candidate
                        )
                        if timed_result.get("status") == "pass":
                            result["timing"] = timed_result["timing"]
                        else:
                            result = {
                                "status": "error",
                                "reason": "timed worker failed numerical revalidation",
                                "timed_worker": timed_result,
                            }
                    case_report[revision][backend] = result
            if args.timing:
                performance = {"same_backend": {}, "cross_provider": {}}
                for backend in args.backends:
                    current = case_report["current"].get(backend, {})
                    branch = case_report["branch"].get(backend, {})
                    if branch.get("status") != "pass":
                        performance["same_backend"][backend] = {
                            "status": "not_counted",
                            "reason": "branch reference failed numerical validation",
                            "reference_status": branch.get("status", "missing"),
                        }
                    elif current.get("status") != "pass":
                        performance["same_backend"][backend] = {
                            "status": "fail",
                            "reason": (
                                "current implementation failed validation or timing"
                            ),
                            "candidate_status": current.get("status", "missing"),
                        }
                    else:
                        performance["same_backend"][backend] = _performance_comparison(
                            current, branch, same_backend=True
                        )
                if (
                    neurons >= 4096
                    and "cupy" in case_report["current"]
                    and case_report["current"]["cupy"].get("status") == "pass"
                ):
                    for reference_backend in ("triton", "tilelang"):
                        reference_result = case_report["branch"].get(
                            reference_backend, {}
                        )
                        if reference_result.get("status") == "pass":
                            performance["cross_provider"][
                                f"current_cupy_vs_branch_{reference_backend}"
                            ] = _performance_comparison(
                                case_report["current"]["cupy"],
                                reference_result,
                                same_backend=False,
                            )
                case_report["performance"] = performance
            statuses = [
                result["status"]
                for revision in ("current", "branch")
                for result in case_report[revision].values()
            ]
            case_report["status"] = (
                "pass"
                if statuses and all(value == "pass" for value in statuses)
                else "fail"
            )
            if args.timing and any(
                value.get("status") == "fail"
                for value in case_report["performance"]["same_backend"].values()
            ):
                case_report["status"] = "fail"
            if case_report["status"] != "pass":
                report["status"] = "fail"
            report["cases"].append(case_report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        # Timing reports are intentionally detailed; keep terminal output small
        # and let automation consume the requested JSON artifact.
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(args.report.resolve()),
                    "cases": [
                        {
                            "steps": case["steps"],
                            "neurons": case["neurons"],
                            "status": case["status"],
                        }
                        for case in report["cases"]
                    ],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


def main():
    args = _parse_args()
    if args.child:
        try:
            _child_main(args)
        except Exception as exc:
            print(json.dumps({"status": "error", "diagnostic": _diagnostic(exc)}))
            return 1
        return 0
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
