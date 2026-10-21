"""Benchmark the 002HIP scalar case after explicit correctness qualification.

This optional benchmark requires explicit unchanged 002HIP and
compatible SpikingJelly roots. It validates numerical results before
measuring and does not install or modify either external dependency.
"""

import argparse
import importlib
import json
import math
import statistics
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import check_002hip as probe


LATENCY_RATIO_LIMIT = 1.2
LATENCY_NOISE_FLOOR_MS = 0.005
MIN_WARMUP = 3
MIN_REPEATS = 30
CURRENT_SOURCE_PATHS = (
    "benchmarks/models/bench_002hip.py",
    "benchmarks/models/check_002hip.py",
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
    parser.add_argument("--hip-root", type=Path, required=True)
    parser.add_argument("--spikingjelly-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--neurons", type=int, default=1024)
    parser.add_argument("--capacitance", type=float, default=100.0)
    parser.add_argument("--warmup", type=int, default=MIN_WARMUP)
    parser.add_argument("--repeats", type=int, default=MIN_REPEATS)
    parser.add_argument(
        "--mode",
        choices=("inference", "training"),
        default="training",
        help=(
            "inference excludes backward measurements and training acceptance; "
            "training requires the original scalar input VJP to pass validation"
        ),
    )
    parser.add_argument(
        "--histories",
        choices=("empty", "full", "both"),
        default="both",
        help="Measure no public records, full voltage/recovery records, or both",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("triton", "tilelang", "cupy"),
        default=("triton", "tilelang", "cupy"),
    )
    parser.add_argument("--forward-atol", type=float, default=1e-4)
    parser.add_argument("--forward-rtol", type=float, default=1e-4)
    parser.add_argument("--gradient-atol", type=float, default=5e-5)
    parser.add_argument("--gradient-rtol", type=float, default=5e-4)
    parser.add_argument(
        "--disable-fp-fusion",
        action="store_true",
        help=(
            "Disable compatible floating-point contraction in btorch and append "
            "a final --fmad=false to the 002HIP compiler options"
        ),
    )
    args = parser.parse_args()
    if min(args.steps, args.neurons) < 1:
        parser.error("steps and neurons must be positive")
    if not math.isfinite(args.capacitance) or args.capacitance <= 0:
        parser.error("capacitance must be finite and positive")
    if args.warmup < MIN_WARMUP:
        parser.error(f"warmup must be at least {MIN_WARMUP}")
    if args.repeats < MIN_REPEATS:
        parser.error(f"repeats must be at least {MIN_REPEATS}")
    if (
        min(
            args.forward_atol,
            args.forward_rtol,
            args.gradient_atol,
            args.gradient_rtol,
        )
        < 0
    ):
        parser.error("tolerances must be non-negative")
    args.backends = tuple(dict.fromkeys(args.backends))
    return args


def _summary(values):
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _return_bytes(values):
    return sum(value.numel() * value.element_size() for value in values)


def _loss(result, history):
    if history == "empty":
        return result[0].sum()
    return probe._loss((result[0], result[3], result[4]))


def _measure_device(torch, operation, repeats):
    values = []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        operation()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return _summary(values)


def _measure_host(torch, operation, repeats):
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000)
    return _summary(values)


def _measure_memory(torch, operation):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial = torch.cuda.memory_allocated()
    values = operation()
    torch.cuda.synchronize()
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() - initial,
        "returned_bytes": _return_bytes(values),
    }


def _measure_graph(torch, build, warmup, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        operation, captured, capture_x = build()
        for _ in range(warmup):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured[:] = operation()
    graph.replay()

    # A foreign kernel on another stream can execute during capture without
    # becoming a graph node. Validate replay itself, not just the warm eager
    # call; otherwise timing only allocations/memsets can look deceptively fast.
    def validate_replay():
        replayed = tuple(value.detach().clone() for value in captured)
        expected = operation()
        if len(replayed) != len(expected):
            raise AssertionError("Captured and eager outputs have different arity")
        for actual, reference in zip(replayed, expected):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    validate_replay()
    device = _measure_device(torch, graph.replay, repeats)
    host = _measure_host(torch, graph.replay, repeats)
    validate_replay()
    # Fixed-input equality alone can accept stale data left by a kernel that
    # ran during capture but was omitted from replay. Change the captured input
    # and require replay both to match eager and to produce a changed output.
    before = tuple(value.detach().clone() for value in captured)
    original_x = capture_x.detach().clone()
    try:
        with torch.no_grad():
            capture_x.add_(0.25)
        graph.replay()
        validate_replay()
        if not any(
            not torch.equal(previous, current)
            for previous, current in zip(before, captured)
        ):
            raise AssertionError("Captured outputs did not respond to changed input")
    finally:
        with torch.no_grad():
            capture_x.copy_(original_x)
    return {
        "device": device,
        "host_end_to_end": host,
        "replay_validation": (
            "bitexact against eager before/after timing and after input mutation; "
            "changed output required"
        ),
    }


def _first_calls(torch, inference, training=None):
    started = time.perf_counter()
    inference()
    torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - started) * 1000
    training_ms = None
    if training is not None:
        started = time.perf_counter()
        training()
        torch.cuda.synchronize()
        training_ms = (time.perf_counter() - started) * 1000
    return {
        "inference_host_ms": inference_ms,
        "training_host_ms": training_ms,
        "note": (
            "First frozen inference call after correctness validation; validation "
            "may have compiled only the grad-enabled dispatch variant"
        ),
    }


def _runner_measurements(torch, make_run, history, args, measure_training):
    frozen_x, _, _ = probe._make_inputs(torch, args.steps, args.neurons, "input")
    frozen_x = frozen_x.detach()
    train_x = None
    if measure_training:
        train_x, _, _ = probe._make_inputs(torch, args.steps, args.neurons, "input")

    def inference():
        result = make_run(frozen_x, history)
        if any(value.requires_grad or value.grad_fn is not None for value in result):
            raise AssertionError("Frozen inference created an autograd graph")
        return result

    def training():
        result = make_run(train_x, history)
        gradient = torch.autograd.grad(_loss(result, history), (train_x,))[0]
        return (*result, gradient)

    first_calls = _first_calls(torch, inference, training if measure_training else None)
    for _ in range(args.warmup):
        inference()
        if measure_training:
            training()
    torch.cuda.synchronize()

    inference_device = _measure_device(torch, inference, args.repeats)
    inference_host = _measure_host(torch, inference, args.repeats)
    inference_memory = _measure_memory(torch, inference)

    def build_inference():
        capture_x = frozen_x.clone()

        def operation():
            return make_run(capture_x, history)

        captured = list(operation())
        return operation, captured, capture_x

    def build_training():
        capture_x = train_x.detach().clone().requires_grad_()

        def operation():
            result = make_run(capture_x, history)
            gradient = torch.autograd.grad(_loss(result, history), (capture_x,))[0]
            return (*result, gradient)

        captured = list(operation())
        return operation, captured, capture_x

    inference_graph = _measure_graph(torch, build_inference, args.warmup, args.repeats)
    result = {
        "first_calls": first_calls,
        "steady_inference_device": inference_device,
        "steady_inference_host_end_to_end": inference_host,
        "inference_memory": inference_memory,
        "inference_graph": inference_graph,
    }
    if not measure_training:
        result["training"] = {
            "status": "not_run",
            "reason": "Inference mode excludes training measurements",
        }
        return result
    result.update(
        {
            "steady_training_host_end_to_end": _measure_host(
                torch, training, args.repeats
            ),
            "training_memory": _measure_memory(torch, training),
            "training_graph": _measure_graph(
                torch, build_training, args.warmup, args.repeats
            ),
        }
    )
    return result


def _latency_acceptance(candidate, reference):
    limit = max(
        LATENCY_RATIO_LIMIT * reference,
        reference + LATENCY_NOISE_FLOOR_MS,
    )
    return {
        "status": "pass" if candidate <= limit else "fail",
        "candidate_ms": candidate,
        "reference_ms": reference,
        "limit_ms": limit,
        "ratio": candidate / reference if reference else None,
    }


def _memory_acceptance(candidate, reference):
    return {
        "status": "pass" if candidate <= reference else "fail",
        "candidate_bytes": candidate,
        "reference_bytes": reference,
    }


def _acceptance(candidate, reference, include_training):
    checks = {
        "steady_inference_device": _latency_acceptance(
            candidate["steady_inference_device"]["median_ms"],
            reference["steady_inference_device"]["median_ms"],
        ),
        "inference_peak_memory": _memory_acceptance(
            candidate["inference_memory"]["peak_allocated_bytes"],
            reference["inference_memory"]["peak_allocated_bytes"],
        ),
    }
    if include_training:
        checks.update(
            {
                "training_graph_device": _latency_acceptance(
                    candidate["training_graph"]["device"]["median_ms"],
                    reference["training_graph"]["device"]["median_ms"],
                ),
                "training_peak_memory": _memory_acceptance(
                    candidate["training_memory"]["peak_allocated_bytes"],
                    reference["training_memory"]["peak_allocated_bytes"],
                ),
            }
        )
    inference_status = (
        "pass"
        if all(
            checks[name]["status"] == "pass"
            for name in ("steady_inference_device", "inference_peak_memory")
        )
        else "fail"
    )
    training_status = "not_run"
    if include_training:
        training_status = (
            "pass"
            if all(
                checks[name]["status"] == "pass"
                for name in ("training_graph_device", "training_peak_memory")
            )
            else "fail"
        )
    return {
        "status": (
            "pass"
            if all(value["status"] == "pass" for value in checks.values())
            else "fail"
        ),
        "inference_status": inference_status,
        "training_status": training_status,
        "checks": checks,
    }


def _setup(args, report):
    repo_root = Path(__file__).resolve().parents[2]
    report["sources"] = {
        "current_btorch": {
            "root": str(repo_root),
            "revision": probe._git_revision(repo_root),
            "dirty": probe._git_dirty(repo_root),
            "files": {
                relative: probe._sha256(repo_root / relative)
                for relative in CURRENT_SOURCE_PATHS
            },
        }
    }
    hip_root = args.hip_root.resolve()
    spikingjelly_root = args.spikingjelly_root.resolve()
    source = hip_root / "model" / "neuron_kernel_float.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    package_root = spikingjelly_root / "spikingjelly"
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(spikingjelly_root))
    cupy = importlib.import_module("cupy")
    spikingjelly = importlib.import_module("spikingjelly")
    torch = importlib.import_module("torch")
    configure = importlib.import_module("spikingjelly.configure")
    surrogate = importlib.import_module("spikingjelly.activation_based.surrogate")
    loaded = Path(spikingjelly.__file__).resolve()
    if not loaded.is_relative_to(package_root):
        raise ImportError(f"Loaded SpikingJelly from {loaded}, expected {package_root}")

    from btorch.models.kernels import (
        CellSpec,
        HistoryPlan,
        KernelOptions,
        run_multi_step,
    )
    from btorch.models.kernels.golden import python_module_sequence

    configure.save_spike_as_bool_in_neuron_kernel = False
    original_compiler_options, effective_compiler_options = (
        probe._configure_002hip_compiler(configure, args.disable_fp_fusion)
    )
    if not torch.cuda.is_available():
        report["environment"] = {
            "torch": torch.__version__,
            "cupy": cupy.__version__,
            "triton": _distribution_version("triton"),
            "tilelang": _distribution_version("tilelang"),
            "gpu": None,
            "spikingjelly_module": str(loaded),
            "002hip_compiler_options": {
                "original": list(original_compiler_options),
                "effective": list(effective_compiler_options),
            },
        }
        raise RuntimeError("CUDA is unavailable; this benchmark requires a host GPU")
    module = probe._load_module(source, "btorch_002hip_scalar_benchmark")
    case = probe.IzhikevichCase(capacitance=args.capacitance)
    parameters = tuple(
        torch.tensor(value, device="cuda", dtype=torch.float32)
        for value in case.btorch_parameters
    )
    states = (
        torch.full((args.neurons,), -65.0, device="cuda"),
        torch.zeros((args.neurons,), device="cuda"),
    )
    spec = CellSpec("izhikevich", hard_reset=True, alpha=100.0)
    histories = {
        "empty": HistoryPlan(),
        "full": HistoryPlan(requested=("v", "u")),
    }

    def hip_run(x, history):
        # The old SpikingJelly device context selects a device only. Bind CuPy
        # to the active Torch stream so CUDA Graph capture includes its kernel.
        with cupy.cuda.Device(x.device.index):
            with cupy.cuda.ExternalStream(
                torch.cuda.current_stream(x.device).cuda_stream
            ):
                spike, voltage, recovery = module.MultiStepIzhikevichNodePTT.apply(
                    x,
                    *states,
                    *case.hip_arguments,
                    False,
                    surrogate.Sigmoid(alpha=4.0).cuda_code,
                )
        if history == "full":
            return spike, voltage[-1], recovery[-1], voltage, recovery
        return spike, voltage[-1].clone(), recovery[-1].clone()

    def btorch_run(backend):
        def run(x, history):
            return run_multi_step(
                x,
                states,
                parameters,
                spec,
                histories[history],
                backend=KernelOptions(
                    neuron=backend,
                    enable_fp_fusion=not args.disable_fp_fusion,
                ),
            )

        return run

    spikingjelly_files = {
        str(path.relative_to(spikingjelly_root)): probe._sha256(path)
        for path in (
            package_root / "configure.py",
            package_root / "activation_based" / "cuda_utils.py",
            package_root / "activation_based" / "surrogate.py",
            package_root / "activation_based" / "tensor_cache.py",
        )
    }
    report.update(
        {
            "environment": {
                "torch": torch.__version__,
                "cupy": cupy.__version__,
                "triton": _distribution_version("triton"),
                "tilelang": _distribution_version("tilelang"),
                "gpu": torch.cuda.get_device_name(),
                "spikingjelly_module": str(loaded),
                "002hip_compiler_options": {
                    "original": list(original_compiler_options),
                    "effective": list(effective_compiler_options),
                },
            },
            "sources": {
                **report["sources"],
                "002hip_root": str(hip_root),
                "002hip_revision": probe._git_revision(hip_root),
                "scalar_kernel": probe._source_identity(source),
                "spikingjelly_root": str(spikingjelly_root),
                "spikingjelly_files": spikingjelly_files,
                "spikingjelly_verified_revision": (
                    probe.KNOWN_SPIKINGJELLY_REVISION
                    if spikingjelly_files == probe.KNOWN_SPIKINGJELLY_HASHES
                    else None
                ),
            },
        }
    )
    return {
        "torch": torch,
        "python_module_sequence": python_module_sequence,
        "run_multi_step": run_multi_step,
        "parameters": parameters,
        "states": states,
        "spec": spec,
        "case": case,
        "KernelOptions": KernelOptions,
        "histories": histories,
        "hip_module": module,
        "hip_run": hip_run,
        "btorch_run": btorch_run,
    }


def _validate(context, args):
    torch = context["torch"]
    check_backward = args.mode == "training"
    correctness_args = argparse.Namespace(
        steps=args.steps,
        neurons=args.neurons,
        forward_atol=args.forward_atol,
        forward_rtol=args.forward_rtol,
        gradient_atol=args.gradient_atol,
        gradient_rtol=args.gradient_rtol,
        gradient_scope="input",
    )
    reference = probe._module_reference(
        torch,
        context["python_module_sequence"],
        context["spec"],
        context["parameters"],
        correctness_args,
    )
    surrogate = importlib.import_module("spikingjelly.activation_based.surrogate")
    started = time.perf_counter()
    hip = probe._check_hip_variant(
        torch,
        surrogate,
        context["hip_module"],
        "scalar",
        reference,
        correctness_args,
        context["case"],
        check_backward=check_backward,
    )
    torch.cuda.synchronize()
    hip["correctness_and_lazy_compile_host_ms"] = (time.perf_counter() - started) * 1000
    backends = {}
    for backend in args.backends:
        started = time.perf_counter()
        backends[backend] = probe._check_btorch_backend(
            torch,
            context["run_multi_step"],
            context["histories"]["full"],
            context["spec"],
            context["parameters"],
            reference,
            context["KernelOptions"](
                neuron=backend,
                enable_fp_fusion=not args.disable_fp_fusion,
            ),
            correctness_args,
            check_backward=check_backward,
        )
        torch.cuda.synchronize()
        backends[backend]["correctness_and_lazy_compile_host_ms"] = (
            time.perf_counter() - started
        ) * 1000

    reference_x, _, _ = probe._make_inputs(torch, args.steps, args.neurons, "input")
    reference_output, reference_final, _ = context["python_module_sequence"](
        reference_x,
        context["states"],
        context["parameters"],
        context["spec"],
    )
    reference_gradient = None
    if check_backward:
        reference_gradient = torch.autograd.grad(reference_output.sum(), (reference_x,))
    empty_history = {}
    runners = {"002hip_scalar": context["hip_run"]}
    runners.update(
        {backend: context["btorch_run"](backend) for backend in args.backends}
    )
    for name, run in runners.items():
        x, _, _ = probe._make_inputs(torch, args.steps, args.neurons, "input")
        result = run(x, "empty")
        forward = probe._comparison(
            (result[0], result[1], result[2]),
            (reference_output, *reference_final),
            ("spikes", "final_voltage", "final_recovery"),
            args.forward_atol,
            args.forward_rtol,
        )
        backward = {"status": "not_run"}
        if check_backward:
            gradient = torch.autograd.grad(_loss(result, "empty"), (x,))
            backward = probe._comparison(
                gradient,
                reference_gradient,
                ("input",),
                args.gradient_atol,
                args.gradient_rtol,
            )
        empty_history[name] = {
            "status": (
                "pass"
                if forward["status"] == "pass"
                and (not check_backward or backward["status"] == "pass")
                else "fail"
            ),
            "forward": forward,
            "backward": backward,
        }
    full_inference = [
        hip["forward"]["status"],
        hip["continuation"]["status"],
        hip["spike_count_status"],
        hip["nontrivial_spike_activity_status"],
    ]
    full_inference.extend(
        status
        for value in backends.values()
        for status in (
            value["forward"]["status"],
            value["continuation"]["status"],
            value["spike_count_status"],
            value["nontrivial_spike_activity_status"],
        )
    )
    inference_status = (
        "pass"
        if all(value == "pass" for value in full_inference)
        and all(
            value["forward"]["status"] == "pass" for value in empty_history.values()
        )
        else "fail"
    )
    training_status = "not_run"
    if check_backward:
        training_status = (
            "pass"
            if hip["backward"]["status"] == "pass"
            and all(
                value["backward"]["status"] == "pass" for value in backends.values()
            )
            and all(
                value["backward"]["status"] == "pass"
                for value in empty_history.values()
            )
            else "fail"
        )
    return {
        "status": (
            "pass"
            if inference_status == "pass"
            and (not check_backward or training_status == "pass")
            else "fail"
        ),
        "inference_status": inference_status,
        "training_status": training_status,
        "full_history": {"002hip_scalar": hip, "btorch": backends},
        "empty_history": empty_history,
    }


def _run(args, report):
    context = _setup(args, report)
    validation = _validate(context, args)
    report["validation"] = validation
    if validation["inference_status"] != "pass":
        report["status"] = "fail"
        return
    measure_training = args.mode == "training"
    if measure_training and validation["training_status"] != "pass":
        report["acceptance"] = {
            "status": "not_run",
            "reason": "Original scalar input VJP failed strong validation",
        }
        report["status"] = "fail"
        return

    histories = ("empty", "full") if args.histories == "both" else (args.histories,)
    results = {}
    for history in histories:
        results[history] = {
            "002hip_scalar": _runner_measurements(
                context["torch"],
                context["hip_run"],
                history,
                args,
                measure_training,
            ),
            "btorch": {},
        }
        for backend in args.backends:
            results[history]["btorch"][backend] = _runner_measurements(
                context["torch"],
                context["btorch_run"](backend),
                history,
                args,
                measure_training,
            )
    report["measurements"] = results
    if "empty" not in results:
        report["acceptance"] = {
            "status": "not_run",
            "reason": "Acceptance requires the empty-history case",
        }
        report["status"] = "not_run"
        return
    reference = results["empty"]["002hip_scalar"]
    report["acceptance"] = {
        "status": "pass",
        "scope": args.mode,
        "backends": {
            backend: _acceptance(value, reference, measure_training)
            for backend, value in results["empty"]["btorch"].items()
        },
    }
    if any(
        value["status"] != "pass" for value in report["acceptance"]["backends"].values()
    ):
        report["acceptance"]["status"] = "fail"
    report["status"] = report["acceptance"]["status"]


def main():
    args = _parse_args()
    report = {
        "benchmark": "002HIP scalar input-gradient comparison",
        "status": "error",
        "contract": {
            "dtype": "float32",
            "shape_TN": [args.steps, args.neurons],
            "capacitance": args.capacitance,
            "gradient_scope": "input",
            "dt": 1.0,
            "hard_reset": True,
            "btorch_alpha": 100.0,
            "002hip_sigmoid_alpha": 4.0,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "mode": args.mode,
            "histories": args.histories,
            "floating_point_contraction": {
                "disable_requested": args.disable_fp_fusion,
                "btorch_enable_fp_fusion": not args.disable_fp_fusion,
                "002hip_policy": (
                    "append final --fmad=false"
                    if args.disable_fp_fusion
                    else "preserve original compiler options"
                ),
                "scope": (
                    "Controls compatible floating-point contraction only; it "
                    "is not a blanket fast-math switch"
                ),
            },
            "acceptance_fixed_before_measurement": {
                "latency_ratio": LATENCY_RATIO_LIMIT,
                "noise_floor_ms": LATENCY_NOISE_FLOOR_MS,
                "effective_limit": "max(1.2 * reference, reference + 0.005 ms)",
                "latency_metrics": [
                    "steady uncaptured inference device median",
                    "training CUDA Graph device median when training is enabled",
                ],
                "empty_history_peak_memory": "candidate <= original reference",
            },
            "original_empty_history_accounting": (
                "002HIP always emits voltage/recovery trajectories; they are "
                "counted as internal overhead when the wrapper returns no records"
            ),
        },
    }
    try:
        _run(args, report)
    except Exception as exc:
        report["diagnostic"] = probe._compact_diagnostic(exc)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
