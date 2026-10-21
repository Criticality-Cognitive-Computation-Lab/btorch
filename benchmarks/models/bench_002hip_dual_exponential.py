"""Benchmark the qualified historical 002HIP dual-exponential mapping.

The archived public Torch class is the scientific baseline. Native
btorch backends are compared for performance only with the identical
mapped btorch Torch composition, after the full output/state/VJP
qualification has passed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import bench_002hip as timing
import check_002hip_dual_exponential as qualification
import torch
from torch import Tensor
from torch.nn import functional as F

from btorch.models import environ


BACKENDS = ("triton", "cupy", "tilelang")
SEQUENCES = ("whole", "two_chunk")
DT = 1.0
ATOL = 2e-5
RTOL = 2e-5
LATENCY_RATIO_LIMIT = 1.2
LATENCY_NOISE_FLOOR_US = 5.0


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-repo",
        type=Path,
        default=Path("/data1/fanqixuan/src/002HIP"),
    )
    parser.add_argument(
        "--reference-revision", default=qualification.REFERENCE_REVISION
    )
    parser.add_argument(
        "--spikingjelly-root",
        type=Path,
        default=Path("/data1/fanqixuan/src/spikingjelly"),
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--split", type=int, default=64)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--parameter-layout",
        choices=qualification.PARAMETER_LAYOUTS,
        default="scalar",
        help="Original physical layout of each legacy parameter root",
    )
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
    parser.add_argument("--sequences", nargs="+", choices=SEQUENCES, default=SEQUENCES)
    parser.add_argument("--precise-exp", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if len(args.reference_revision) != 40:
        parser.error("--reference-revision must be a pinned full SHA")
    if min(args.steps, args.neurons) < 2:
        parser.error("steps and neurons must be at least two")
    if not 0 < args.split < args.steps:
        parser.error("split must be inside the sequence")
    if args.warmup < 3 or args.repeats < 30:
        parser.error("warmup must be at least 3 and repeats at least 30")
    args.backends = tuple(dict.fromkeys(args.backends))
    args.sequences = tuple(dict.fromkeys(args.sequences))
    return args


def _distribution_version(name: str):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _chunks(args, sequence: str) -> tuple[int, ...]:
    if sequence == "whole":
        return (args.steps,)
    return (args.split, args.steps - args.split)


def _loss(result: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    output, final_g, final_h = result
    scale = torch.linspace(
        0.3, 1.1, output.shape[0], device=output.device, dtype=output.dtype
    )[:, None, None]
    return (
        (output * scale).square().mean()
        + 0.13 * final_g.square().mean()
        + 0.17 * final_h.square().mean()
    )


class _Runner:
    """Own one module and its static leaves outside measured calls."""

    def __init__(
        self,
        kind: str,
        backend: str,
        reference_class,
        values: tuple[Tensor, ...],
        parameter_values: dict[str, Tensor],
        parameter_layout: str,
        chunks: tuple[int, ...],
        training: bool,
        *,
        precise_exp: bool = False,
    ):
        self.kind = kind
        self.backend = backend
        self.chunks = chunks
        source, weight, bias, g_init, h_init = values
        self.source = source.detach().clone().requires_grad_(training)
        self.g_init = g_init.detach().clone().requires_grad_(training)
        self.h_init = h_init.detach().clone().requires_grad_(training)
        self.parameter_roots = {
            name: value.detach().clone().requires_grad_(training)
            for name, value in parameter_values.items()
        }
        _, self.parameters = qualification._parameter_roots(
            parameter_values, parameter_layout, self.parameter_roots
        )
        self.parameter_layout = parameter_layout

        # The archived class owns the actual dense Wz+b parameters. The mapped
        # composition owns equivalent leaves and feeds the projection through
        # the qualified one-step pending-drive contract.
        if kind == "legacy":
            self.module = qualification._legacy_model(
                reference_class, weight, bias, self.parameters
            )
            self.module.linear.weight.requires_grad_(training)
            self.module.linear.bias.requires_grad_(training)
            self.weight = self.module.linear.weight
            self.bias = self.module.linear.bias
        else:
            self.weight = weight.detach().clone().requires_grad_(training)
            self.bias = bias.detach().clone().requires_grad_(training)
            mapped = qualification._mapping(self.parameters, validate=False)
            self.module = qualification._mapped_module(
                source.shape[-1],
                backend,
                source.device,
                source.dtype,
                mapped[2].detach(),
                mapped[3].detach(),
                mapped[4].detach(),
                precise_exp=precise_exp,
            )
        self.training = training

    @property
    def leaves(self) -> tuple[Tensor, ...]:
        return (
            self.source,
            self.weight,
            self.bias,
            self.g_init,
            self.h_init,
            *(self.parameter_roots[name] for name in qualification.PARAMETER_NAMES),
        )

    def _legacy(self) -> tuple[Tensor, Tensor, Tensor]:
        module = self.module
        for name, value in self.parameters.items():
            setattr(module, name, value)
        module.g = self.g_init
        module.h = self.h_init
        outputs = []
        start = 0
        for length in self.chunks:
            stop = start + length
            outputs.append(module.multi_step_forward(self.source[start:stop]))
            start = stop
        return torch.cat(outputs), module.g, module.h

    def _mapped(self) -> tuple[Tensor, Tensor, Tensor]:
        module = self.module
        qd, qr, tau_decay, tau_rise, amplitude = qualification._mapping(
            self.parameters, validate=False
        )
        module.tau_decay = tau_decay
        module.tau_rise = tau_rise
        module.a = (tau_decay - tau_rise) / tau_rise / tau_decay * amplitude
        g_max = self.parameters["g_max"]
        module.g_rise = self.h_init / (g_max * qr)
        module.g_decay = ((qd - qr) * self.g_init + self.h_init) / (g_max * qd)
        module.psc = (self.parameters["E_syn"] - self.parameters["v_r"]) * self.g_init
        pending = torch.zeros_like(self.g_init)
        outputs = []
        start = 0
        with environ.context(dt=DT):
            for length in self.chunks:
                stop = start + length
                projected = F.linear(self.source[start:stop], self.weight, self.bias)
                delayed = torch.cat((pending.unsqueeze(0), projected[:-1]), dim=0)
                if self.backend == "torch":
                    outputs.append(
                        torch.stack(
                            [module.single_step_forward(value) for value in delayed]
                        )
                    )
                else:
                    outputs.append(module.multi_step_forward(delayed))
                pending = projected[-1]
                start = stop
        h_lag = g_max * module.g_rise
        g_lag = g_max * (module.g_decay - module.g_rise) / (qd - qr)
        final_g = qd * g_lag + h_lag
        final_h = qr * h_lag + g_max * pending
        return torch.cat(outputs), final_g, final_h

    def forward(self) -> tuple[Tensor, Tensor, Tensor]:
        return self._legacy() if self.kind == "legacy" else self._mapped()

    def call(self) -> tuple[Tensor, ...]:
        if not self.training:
            with torch.inference_mode():
                return self.forward()
        result = self.forward()
        gradients = torch.autograd.grad(_loss(result), self.leaves)
        return (*result, *gradients)


def _factory(
    kind,
    backend,
    reference_class,
    values,
    parameter_values,
    parameter_layout,
    chunks,
    training,
    *,
    precise_exp=False,
):
    def build():
        runner = _Runner(
            kind,
            backend,
            reference_class,
            values,
            parameter_values,
            parameter_layout,
            chunks,
            training,
            precise_exp=precise_exp,
        )
        return runner.call, runner.source, runner

    return build


def _assert_results(actual, expected, label):
    if len(actual) != len(expected):
        raise AssertionError(f"{label}: result arity differs")
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        if not bool(torch.isfinite(value).all()):
            raise AssertionError(f"{label}[{index}] contains nonfinite values")
        torch.testing.assert_close(
            value, reference, atol=ATOL, rtol=RTOL, msg=f"{label}[{index}]"
        )


def _assert_inactive_parameter_gradients(result, parameter_layout, label):
    if parameter_layout != "strided":
        return
    parameter_offset = 3 + 5
    if len(result) == 3:
        return
    expected_arity = 3 + len(qualification.LEAF_NAMES)
    if len(result) != expected_arity:
        raise AssertionError(
            f"{label}: expected {expected_arity} training results, got {len(result)}"
        )
    for index, name in enumerate(qualification.PARAMETER_NAMES):
        gradient = result[parameter_offset + index]
        if bool(gradient[1::2].count_nonzero()):
            raise AssertionError(
                f"{label}: {name} inactive backing gradient is nonzero"
            )


def _tape(factory):
    call, _, _ = factory()
    saved = []

    def pack(value):
        saved.append(value)
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        result = call()
    del result
    storages = {
        (str(value.device), value.untyped_storage().data_ptr()): (
            value.untyped_storage().nbytes()
        )
        for value in saved
    }
    return {
        "saved_tensors": len(saved),
        "saved_reference_bytes": sum(
            value.numel() * value.element_size() for value in saved
        ),
        "unique_saved_storage_bytes": sum(storages.values()),
        "scope": (
            "includes required inputs, parameters, and outputs; not private tape only"
        ),
    }


def _milliseconds_to_microseconds(summary):
    return {
        "median": summary["median_ms"] * 1000.0,
        "min": summary["min_ms"] * 1000.0,
        "max": summary["max_ms"] * 1000.0,
    }


def _measure_eager(factory, warmup, repeats):
    call, _, _ = factory()
    for _ in range(warmup):
        result = call()
    del result
    torch.cuda.synchronize()
    return {
        "host_end_to_end_us": _milliseconds_to_microseconds(
            timing._measure_host(torch, call, repeats)
        ),
        "device_end_to_end_us": _milliseconds_to_microseconds(
            timing._measure_device(torch, call, repeats)
        ),
    }


def _measure_memory(factory, warmup):
    call, _, _ = factory()
    for _ in range(warmup):
        result = call()
    del result
    return timing._measure_memory(torch, call)


def _inference_storage(factory):
    call, _, _ = factory()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda value: saved.append(value) or value, lambda value: value
    ):
        result = call()
    if saved or any(
        value.requires_grad or value.grad_fn is not None for value in result
    ):
        raise AssertionError("Inference created an autograd tape")
    return {"saved_tensors": 0, "status": "pass"}


def _measure_captured(factory, expected_factory, parameter_layout, warmup, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call, static_source, _ = factory()
        for _ in range(warmup):
            captured = call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = call()
    stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    expected_call, _, _ = expected_factory()
    expected = expected_call()
    _assert_results(captured, expected, "captured initial replay")
    _assert_inactive_parameter_gradients(
        captured, parameter_layout, "captured initial replay"
    )
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))

    before = tuple(value.detach().clone() for value in captured)
    changed = static_source.detach() * -0.43 + 0.05
    with torch.no_grad():
        static_source.copy_(changed)
    graph.replay()
    torch.cuda.synchronize()
    changed_call, changed_source, _ = expected_factory()
    with torch.no_grad():
        changed_source.copy_(changed)
    changed_expected = changed_call()
    _assert_results(captured, changed_expected, "captured changed-input replay")
    _assert_inactive_parameter_gradients(
        captured, parameter_layout, "captured changed-input replay"
    )
    if torch.equal(before[0], captured[0]):
        raise AssertionError("Captured output did not respond to changed input")
    if len(captured) > 3 and not any(
        not torch.equal(old, new)
        for old, new in zip(before[3:], captured[3:], strict=True)
    ):
        raise AssertionError("Captured VJPs did not respond to changed input")
    return {
        "device_replay_us": _milliseconds_to_microseconds(timing._summary(samples)),
        "changed_input_output_and_vjp": "pass",
    }


def _measure(factory, oracle_factory, parameter_layout, warmup, repeats):
    inference = factory(False)
    training = factory(True)
    inference_oracle = oracle_factory(False)
    training_oracle = oracle_factory(True)
    return {
        "inference": {
            "storage": _inference_storage(inference),
            "eager": _measure_eager(inference, warmup, repeats),
            "captured": _measure_captured(
                inference,
                inference_oracle,
                parameter_layout,
                warmup,
                repeats,
            ),
            "memory": _measure_memory(inference, warmup),
        },
        "training": {
            "eager": _measure_eager(training, warmup, repeats),
            "captured": _measure_captured(
                training,
                training_oracle,
                parameter_layout,
                warmup,
                repeats,
            ),
            "memory": _measure_memory(training, warmup),
            "tape": _tape(training),
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
        "mapped_torch_us": reference_us,
        "limit_us": limit,
        "ratio": candidate_us / reference_us if reference_us else None,
    }


def _comparison(candidate, mapped_torch):
    checks = {}
    for phase in ("inference", "training"):
        checks[f"eager_{phase}"] = _latency_check(
            candidate[phase]["eager"]["device_end_to_end_us"]["median"],
            mapped_torch[phase]["eager"]["device_end_to_end_us"]["median"],
        )
        checks[f"captured_{phase}"] = _latency_check(
            candidate[phase]["captured"]["device_replay_us"]["median"],
            mapped_torch[phase]["captured"]["device_replay_us"]["median"],
        )
    return {
        "status": "informational",
        "counts_for_qualification": False,
        "all_within_envelope": all(
            item["status"] == "within_envelope" for item in checks.values()
        ),
        "rule": "candidate <= max(1.2 * mapped Torch, mapped Torch + 5 us)",
        "checks": checks,
    }


def _validate_mapping(parameters):
    # This performs the qualified regime checks, including finite/nonzero and
    # Euler-factor host decisions, before any CUDA Graph capture begins.
    qualification._mapping(parameters)
    return "pass"


def _source_hash(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(args, report):
    report["precise_exp"] = args.precise_exp
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    reference_repo = args.reference_repo.resolve()
    revision = qualification._git(
        reference_repo, "rev-parse", args.reference_revision
    ).strip()
    if revision != args.reference_revision:
        raise ValueError(f"Reference resolved to {revision}, expected pinned SHA")
    spikingjelly = qualification._enable_real_spikingjelly(args.spikingjelly_root)
    device = torch.device("cuda")
    values = qualification._inputs(args.steps, args.neurons, args.seed, device)
    parameter_values = qualification._parameter_inputs(
        args.neurons, args.parameter_layout, device
    )
    _, active_parameters = qualification._parameter_roots(
        parameter_values, args.parameter_layout, parameter_values
    )
    _validate_mapping(active_parameters)
    report.update(
        {
            "mapping_regime_validation_before_capture": "pass",
            "gradient_roots": list(qualification.LEAF_NAMES),
            "parameter_layout": args.parameter_layout,
            "parameter_roots": {
                name: {
                    "shape": list(root.shape),
                    "stride": list(root.stride()),
                    "active_shape": list(active_parameters[name].shape),
                    "active_stride": list(active_parameters[name].stride()),
                }
                for name, root in parameter_values.items()
            },
            "inactive_strided_backing_gradient": (
                "checked_zero"
                if args.parameter_layout == "strided"
                else "not_applicable"
            ),
            "reference": {
                "repo": str(reference_repo),
                "revision": revision,
                "connection_sha256": hashlib.sha256(
                    qualification._source_at(reference_repo, revision)
                ).hexdigest(),
                "path": "actual archived public backend='torch' class",
            },
            "environment": {
                "torch": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "triton": _distribution_version("triton"),
                "cupy": _distribution_version("cupy"),
                "tilelang": _distribution_version("tilelang"),
                "gpu": torch.cuda.get_device_name(),
                "gpu_capability": list(torch.cuda.get_device_capability()),
                "spikingjelly": spikingjelly,
            },
        }
    )
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "benchmarks/models/bench_002hip_dual_exponential.py",
        "benchmarks/models/check_002hip_dual_exponential.py",
        "benchmarks/models/bench_002hip.py",
        "btorch/models/synapse.py",
        "btorch/models/kernels/adapters.py",
        "btorch/models/kernels/operators.py",
        "btorch/models/kernels/backends/engine.py",
        "btorch/models/kernels/backends/fragments.py",
    ):
        report["sources"][relative] = _source_hash(root / relative)

    with tempfile.TemporaryDirectory(prefix="btorch-002hip-dual-bench-") as temporary:
        snapshot = Path(temporary) / "snapshot"
        snapshot.mkdir()
        qualification._archive_model(reference_repo, revision, snapshot)
        reference_class = qualification._load_snapshot_class(snapshot, revision)
        qualification_args = argparse.Namespace(
            device="cuda",
            steps=args.steps,
            neurons=args.neurons,
            split=args.split,
            seed=args.seed,
            atol=ATOL,
            rtol=RTOL,
            precise_exp=args.precise_exp,
        )
        for backend in ("torch", *args.backends):
            started = time.perf_counter()
            result = qualification._case(
                reference_class, backend, args.parameter_layout, qualification_args
            )
            report["qualification"][backend] = {
                **result,
                "elapsed_s": time.perf_counter() - started,
            }
        for sequence in args.sequences:
            chunks = _chunks(args, sequence)

            def make_factory(kind, backend):
                return lambda training: _factory(
                    kind,
                    backend,
                    reference_class,
                    values,
                    parameter_values,
                    args.parameter_layout,
                    chunks,
                    training,
                    precise_exp=args.precise_exp,
                )

            legacy_factory = make_factory("legacy", "torch")
            legacy = _measure(
                legacy_factory,
                legacy_factory,
                args.parameter_layout,
                args.warmup,
                args.repeats,
            )
            mapped_torch = _measure(
                make_factory("mapped", "torch"),
                legacy_factory,
                args.parameter_layout,
                args.warmup,
                args.repeats,
            )
            for backend in args.backends:
                started = time.perf_counter()
                candidate = _measure(
                    make_factory("mapped", backend),
                    legacy_factory,
                    args.parameter_layout,
                    args.warmup,
                    args.repeats,
                )
                case = {
                    "status": "pass",
                    "sequence": sequence,
                    "chunks": list(chunks),
                    "backend": backend,
                    "parameter_layout": args.parameter_layout,
                    "correctness": {
                        "status": report["qualification"][backend]["status"],
                        "qualification_report_key": backend,
                    },
                    "archived_torch_baseline": legacy,
                    "mapped_torch_baseline": mapped_torch,
                    "candidate": candidate,
                    "performance": _comparison(candidate, mapped_torch),
                    "legacy_performance_comparison": (
                        "not made: archived Torch is a scientific baseline, not "
                        "an identical native execution context"
                    ),
                    "elapsed_s": time.perf_counter() - started,
                }
                report["cases"].append(case)
                print(
                    json.dumps(
                        {
                            "status": "pass",
                            "sequence": sequence,
                            "backend": backend,
                            "parameter_layout": args.parameter_layout,
                            "performance": case["performance"],
                        }
                    ),
                    flush=True,
                )
                gc.collect()


def main():
    args = _args()
    report = {
        "status": "running",
        "scope": (
            "qualified archived 002HIP Torch class and mapped actual btorch "
            "DualExponentialPSC; archived CUDA paths are excluded"
        ),
        "shape_TBN": [args.steps, 1, args.neurons],
        "dtype": "float32",
        "dt": DT,
        "parameter_layout": args.parameter_layout,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "sources": {},
        "qualification": {},
        "cases": [],
    }
    try:
        _run(args, report)
        report["status"] = "pass"
    except Exception as error:
        report["status"] = "fail"
        report["diagnostic"] = {
            "exception": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error))[-12000:],
        }
    finally:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(report["cases"])}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
