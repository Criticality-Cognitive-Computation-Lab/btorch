"""Benchmark standalone native ConductancePSC after Module qualification.

The scientific reference is the current btorch ``ConductancePSC`` executed
through repeated public ``single_step_forward`` calls. This benchmark compares
only the exact-class standalone adapter with that reference; it does not cover
native recurrent scheduling or the scientifically distinct 002HIP models.

Run on a CUDA worker with the optional backends installed::

    PYTHONPATH=. python benchmarks/models/bench_conductance.py \
        --report conductance.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from btorch.models import environ
from btorch.models.connection import SparseConnection
from btorch.models.functional import init_net_state
from btorch.models.synapse import ConductancePSC


BACKENDS = ("triton", "cupy", "tilelang")
CONNECTIONS = ("dense", "sparse")
DT = 0.35
LATENCY_RATIO_LIMIT = 1.2
LATENCY_NOISE_FLOOR_US = 5.0
FORWARD_KW = {"atol": 3e-4, "rtol": 5e-4}
GRADIENT_KW = {"atol": 2e-3, "rtol": 2e-3}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--neurons", type=int, default=1024)
    parser.add_argument("--indegree", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
    parser.add_argument(
        "--connections", nargs="+", choices=CONNECTIONS, default=CONNECTIONS
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.neurons, args.indegree) < 1:
        parser.error("steps, batch, neurons and indegree must be positive")
    if args.indegree > args.neurons:
        parser.error("indegree cannot exceed neurons")
    if args.warmup < 3 or args.repeats < 30:
        parser.error("warmup must be at least 3 and repeats at least 30")
    args.backends = tuple(dict.fromkeys(args.backends))
    args.connections = tuple(dict.fromkeys(args.connections))
    return args


def _distribution_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _canonical_case(args):
    """Create backend-independent CPU values before one explicit CUDA copy."""
    generator = torch.Generator().manual_seed(1731)
    T, B, N = args.steps, args.batch, args.neurons
    z = torch.randn(T, B, N, generator=generator, dtype=torch.float32) * 0.08
    time = torch.arange(T, dtype=torch.float32)[:, None, None]
    lane = torch.arange(N, dtype=torch.float32)[None, None, :]
    voltage = -55.0 + 7.0 * torch.sin(time * 0.071 + lane * 0.013)
    voltage = voltage.expand(T, B, N).contiguous()
    psc = 0.18 * torch.sin(
        torch.arange(B * N, dtype=torch.float32).reshape(B, N) * 0.019
    )
    E = -12.0 + 25.0 * torch.cos(torch.arange(N, dtype=torch.float32) * 0.017)
    output_cotangent = torch.randn(T, B, N, generator=generator, dtype=torch.float32)
    state_cotangent = torch.randn(B, N, generator=generator, dtype=torch.float32)
    dense_weight = torch.randn(N, N, generator=generator, dtype=torch.float32)
    dense_weight.mul_(0.07 / args.neurons**0.5)

    # Every target receives exactly ``indegree`` distinct sources. A cyclic
    # construction avoids duplicate-edge merging changing the requested load.
    post = torch.arange(N, dtype=torch.long).repeat_interleave(args.indegree)
    offset = torch.arange(args.indegree, dtype=torch.long).repeat(N)
    pre = (post * 37 + offset) % N
    sparse_weight = torch.randn(
        N * args.indegree, generator=generator, dtype=torch.float32
    )
    sparse_weight.mul_(0.07 / args.indegree**0.5)
    values = {
        "z": z,
        "v": voltage,
        "psc": psc,
        "E": E,
        "output_cotangent": output_cotangent,
        "state_cotangent": state_cotangent,
        "dense_weight": dense_weight,
        "sparse_weight": sparse_weight,
        "pre": pre,
        "post": post,
    }
    return {name: value.cuda() for name, value in values.items()}


def _connection(kind, values, neurons):
    if kind == "dense":
        linear = torch.nn.Linear(
            neurons, neurons, bias=False, device="cuda", dtype=torch.float32
        )
        with torch.no_grad():
            linear.weight.copy_(values["dense_weight"])
        return linear
    return SparseConnection.from_edges(
        values["pre"],
        values["post"],
        neurons,
        neurons,
        values=values["sparse_weight"],
    ).cuda()


def _build(kind, backend, values, args, *, differentiable):
    module = ConductancePSC(
        args.neurons,
        tau_syn=torch.tensor(6.7, device="cuda"),
        linear=_connection(kind, values, args.neurons),
        reversal_potential=values["E"].clone(),
        step_mode="m",
        backend=backend,
    )
    init_net_state(
        module,
        batch_size=args.batch,
        dtype=torch.float32,
        device="cuda",
    )
    module.requires_grad_(differentiable)
    module.tau_syn.requires_grad_(differentiable)
    module.E.requires_grad_(differentiable)
    module.psc = values["psc"].clone().requires_grad_(differentiable)
    return module


def _step_sequence(module, z, v):
    """Execute only the public single-step Module API as the oracle."""
    return torch.stack(
        [module.single_step_forward(z[t], v[t]) for t in range(z.shape[0])]
    )


def _run(module, z, v, native):
    return module(z, v_seq=v) if native else _step_sequence(module, z, v)


def _roots(module, z, v, initial_psc):
    parameters = tuple(module.linear.parameters())
    if len(parameters) != 1:
        raise AssertionError(
            f"Expected one connection parameter, got {len(parameters)}"
        )
    return (z, v, initial_psc, parameters[0], module.tau_syn, module.E)


def _training_result(module, z, v, initial_psc, native, values):
    module.psc = initial_psc
    output = _run(module, z, v, native)
    final_psc = module.psc
    gradients = torch.autograd.grad(
        (output, final_psc),
        _roots(module, z, v, initial_psc),
        (values["output_cotangent"], values["state_cotangent"]),
    )
    return (output, final_psc, *gradients)


def _inference_result(module, z, v, initial_psc, native):
    module.psc = initial_psc
    output = _run(module, z, v, native)
    return output, module.psc


def _assert_close(actual, expected, *, gradients=False):
    options = GRADIENT_KW if gradients else FORWARD_KW
    if len(actual) != len(expected):
        raise AssertionError(f"Result arity differs: {len(actual)} != {len(expected)}")
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        if not torch.isfinite(value).all():
            raise AssertionError(f"Result {index} contains non-finite values")
        torch.testing.assert_close(value, reference, **options)


def _oracle_result(kind, values, args, z_value, v_value, *, training):
    module = _build(kind, "torch", values, args, differentiable=training)
    z = z_value.detach().clone().requires_grad_(training)
    v = v_value.detach().clone().requires_grad_(training)
    psc = values["psc"].detach().clone().requires_grad_(training)
    with environ.context(dt=DT):
        if training:
            return _training_result(module, z, v, psc, False, values)
        with torch.inference_mode():
            return _inference_result(module, z, v, psc, False)


def _correctness(kind, backend, values, args):
    native = _build(kind, backend, values, args, differentiable=True)
    reference = _build(kind, "torch", values, args, differentiable=True)
    z = values["z"].detach().clone().requires_grad_()
    zr = values["z"].detach().clone().requires_grad_()
    v = values["v"].detach().clone().requires_grad_()
    vr = values["v"].detach().clone().requires_grad_()
    psc = values["psc"].detach().clone().requires_grad_()
    pscr = values["psc"].detach().clone().requires_grad_()
    with environ.context(dt=DT):
        actual = _training_result(native, z, v, psc, True, values)
        expected = _training_result(reference, zr, vr, pscr, False, values)
    _assert_close(actual[:2], expected[:2])
    _assert_close(actual[2:], expected[2:], gradients=True)
    gradient_names = ("events", "voltage", "initial_g", "weight", "tau_syn", "E")
    gradient_norms = {}
    for name, gradient in zip(gradient_names, actual[2:], strict=True):
        norm = float(gradient.norm())
        if norm == 0.0:
            raise AssertionError(f"Expected a nonzero {name} VJP")
        gradient_norms[name] = norm

    # A split call must carry the physical conductance state without changing
    # event timing. The reference continues through actual single steps.
    split = args.steps // 2
    chunked = _build(kind, backend, values, args, differentiable=False)
    stepped = _build(kind, "torch", values, args, differentiable=False)
    with torch.no_grad(), environ.context(dt=DT):
        first = chunked(values["z"][:split], v_seq=values["v"][:split])
        second = chunked(values["z"][split:], v_seq=values["v"][split:])
        reference_output = _step_sequence(stepped, values["z"], values["v"])
    _assert_close(
        (torch.cat((first, second)), chunked.psc), (reference_output, stepped.psc)
    )
    return {
        "status": "pass",
        "oracle": "actual ConductancePSC repeated single_step_forward",
        "full_vjp_norms": gradient_norms,
        "continuation_split": split,
    }


def _make_workload(kind, backend, values, args, *, training, native):
    module = _build(kind, backend, values, args, differentiable=training)
    z = values["z"].detach().clone().requires_grad_(training)
    v = values["v"].detach().clone().requires_grad_(training)
    psc = values["psc"].detach().clone().requires_grad_(training)

    if training:
        call = lambda: _training_result(module, z, v, psc, native, values)
    else:

        def call():
            with torch.inference_mode():
                return _inference_result(module, z, v, psc, native)

    return call, z, v


def _summary(samples):
    ordered = sorted(samples)
    return {
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "samples": len(ordered),
    }


def _measure_eager(factory, warmup, repeats):
    call, _, _ = factory()
    for _ in range(warmup):
        result = call()
    del result
    torch.cuda.synchronize()
    host_us, device_us = [], []
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
    del result
    return {
        "host_enqueue_us": _summary(host_us),
        "device_end_to_end_us": _summary(device_us),
    }


def _measure_memory(factory, warmup):
    call, _, _ = factory()
    for _ in range(warmup):
        result = call()
    del result
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    result = call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    live_bytes = sum(value.numel() * value.element_size() for value in result)
    del result
    return {
        "baseline_bytes": baseline,
        "peak_bytes": peak,
        "incremental_peak_bytes": peak - baseline,
        "live_return_bytes": live_bytes,
        "measurement": "warmed eager workload; CUDA Graph pool allocation excluded",
    }


def _changed_inputs(z, v):
    return z * -0.43 + 0.05, v * 1.015 + 0.4


def _measure_captured(kind, factory, values, args, *, training, warmup, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call, static_z, static_v = factory()
        for _ in range(warmup):
            static_result = call()
        del static_result
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            static_result = call()
    stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    expected = _oracle_result(
        kind, values, args, values["z"], values["v"], training=training
    )
    _assert_close(static_result[:2], expected[:2], gradients=False)
    if training:
        _assert_close(static_result[2:], expected[2:], gradients=True)

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)

    before = tuple(value.detach().clone() for value in static_result)
    changed_z, changed_v = _changed_inputs(values["z"], values["v"])
    with torch.no_grad():
        static_z.copy_(changed_z)
        static_v.copy_(changed_v)
    graph.replay()
    torch.cuda.synchronize()
    changed_expected = _oracle_result(
        kind, values, args, changed_z, changed_v, training=training
    )
    _assert_close(static_result[:2], changed_expected[:2], gradients=False)
    if training:
        _assert_close(static_result[2:], changed_expected[2:], gradients=True)
    if torch.equal(before[0], static_result[0]):
        raise AssertionError("CUDA Graph replay did not respond to changed inputs")
    if training and not any(
        not torch.equal(previous, current)
        for previous, current in zip(before[2:], static_result[2:], strict=True)
    ):
        raise AssertionError("CUDA Graph replay VJPs did not respond to changed inputs")
    return {
        "device_replay_us": _summary(samples),
        "changed_input_output_and_vjp_replay": "pass",
    }


def _inference_storage_check(kind, backend, values, args, native):
    call, _, _ = _make_workload(
        kind, backend, values, args, training=False, native=native
    )
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda value: saved.append(value) or value, lambda value: value
    ):
        result = call()
    if saved:
        raise AssertionError(f"Inference saved {len(saved)} backward tensors")
    if any(value.requires_grad or value.grad_fn is not None for value in result):
        raise AssertionError("Inference unexpectedly created an autograd tape")
    return {"status": "pass", "saved_backward_tensors": 0}


def _measure_implementation(kind, backend, values, args, *, native):
    factory = lambda training: (
        lambda: _make_workload(
            kind, backend, values, args, training=training, native=native
        )
    )
    inference_factory = factory(False)
    training_factory = factory(True)
    return {
        "inference_storage": _inference_storage_check(
            kind, backend, values, args, native
        ),
        "inference": {
            "eager": _measure_eager(inference_factory, args.warmup, args.repeats),
            "captured": _measure_captured(
                kind,
                inference_factory,
                values,
                args,
                training=False,
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "memory": _measure_memory(inference_factory, args.warmup),
        },
        "training": {
            "eager": _measure_eager(training_factory, args.warmup, args.repeats),
            "captured": _measure_captured(
                kind,
                training_factory,
                values,
                args,
                training=True,
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "memory": _measure_memory(training_factory, args.warmup),
        },
    }


def _latency_check(candidate_us, reference_us):
    limit = max(
        LATENCY_RATIO_LIMIT * reference_us,
        reference_us + LATENCY_NOISE_FLOOR_US,
    )
    return {
        "status": "within_envelope" if candidate_us <= limit else "outside_envelope",
        "candidate_us": candidate_us,
        "reference_us": reference_us,
        "limit_us": limit,
        "ratio": candidate_us / reference_us if reference_us else None,
    }


def _latency_comparison(candidate, reference):
    checks = {}
    for phase in ("inference", "training"):
        checks[f"eager_{phase}_device"] = _latency_check(
            candidate[phase]["eager"]["device_end_to_end_us"]["median"],
            reference[phase]["eager"]["device_end_to_end_us"]["median"],
        )
        checks[f"captured_{phase}_device"] = _latency_check(
            candidate[phase]["captured"]["device_replay_us"]["median"],
            reference[phase]["captured"]["device_replay_us"]["median"],
        )
    return {
        "status": "informational",
        "all_within_envelope": all(
            check["status"] == "within_envelope" for check in checks.values()
        ),
        "latency_rule": "candidate <= max(1.2 * baseline, baseline + 5 us)",
        "counts_for_acceptance": False,
        "reason": (
            "the baseline is the actual repeated-step Torch Module oracle, "
            "not a matched native backend implementation"
        ),
        "checks": checks,
    }


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(1731)
    values = _canonical_case(args)
    report = {
        "sources": {
            name: hashlib.sha256(
                (Path(__file__).resolve().parents[2] / name).read_bytes()
            ).hexdigest()
            for name in (
                "benchmarks/models/bench_conductance.py",
                "btorch/models/synapse.py",
                "btorch/models/kernels/adapters.py",
                "btorch/models/kernels/operators.py",
                "btorch/models/kernels/backends/engine.py",
                "btorch/models/kernels/backends/fragments.py",
            )
        },
        "scope": (
            "current btorch standalone ConductancePSC; no recurrent COBA or "
            "002HIP performance claim"
        ),
        "environment": {
            "torch": torch.__version__,
            "triton": _distribution_version("triton"),
            "cupy": _distribution_version("cupy"),
            "tilelang": _distribution_version("tilelang"),
            "gpu": torch.cuda.get_device_name(),
        },
        "shape_TBN": [args.steps, args.batch, args.neurons],
        "dtype": "float32",
        "dt": DT,
        "indegree": args.indegree,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": [],
    }
    with environ.context(dt=DT):
        for kind in args.connections:
            baseline = _measure_implementation(
                kind, "torch", values, args, native=False
            )
            for backend in args.backends:
                started = time.perf_counter()
                correctness = _correctness(kind, backend, values, args)
                candidate = _measure_implementation(
                    kind, backend, values, args, native=True
                )
                performance = _latency_comparison(candidate, baseline)
                case = {
                    "status": "pass",
                    "backend": backend,
                    "connection": kind,
                    "edges": (
                        args.neurons * args.neurons
                        if kind == "dense"
                        else args.neurons * args.indegree
                    ),
                    "correctness": correctness,
                    "baseline": {
                        "name": "actual Module repeated single_step_forward",
                        **baseline,
                    },
                    "candidate": candidate,
                    "performance_comparison": performance,
                    "elapsed_s": time.perf_counter() - started,
                }
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
    report["status"] = "pass"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(report["cases"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
