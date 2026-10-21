"""Measure selective primitive reverse outputs against a preserved engine.

The baseline is a source snapshot supplied explicitly by the caller.
Both implementations execute the same complete native forward/reverse
sequence; actual btorch Modules provide output, final-state, and VJP
oracles. Each implementation runs in a fresh process so its JIT
factories, compiled-kernel caches and CUDA graph pools cannot affect the
other arm. This is an internal optimization comparison, not an external
002HIP performance gate.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import bench_002hip as timing
import torch

from btorch.models.kernels import CellSpec, get_template
from btorch.models.kernels.backends import engine
from btorch.models.kernels.golden import python_module_sequence


def _inputs(model, steps, batch, neurons):
    torch.manual_seed(542)
    x = torch.rand(steps, batch, neurons, device="cuda") * 0.4
    spec = CellSpec(model, dt=0.5)
    template = get_template(spec)
    states = tuple(
        torch.full((batch, neurons), 0.03 * (index + 1), device="cuda")
        for index in range(len(template.state_names))
    )
    values = {
        "lif": (1.0, 0.0, 1.2, 3.0),
        "alpha": (3.0, 0.7),
        "dual_exponential": (2.0, 4.0, 0.7),
    }[model]
    parameters = tuple(torch.tensor(value, device="cuda") for value in values)
    return spec, x, states, parameters


def _needs(scope, states, parameters):
    return (
        scope == "input",
        *(
            scope == "state" and index == len(states) - 1
            for index in range(len(states))
        ),
        *((scope == "parameters",) * len(parameters)),
    )


def _oracle(x, states, parameters, spec, needs):
    values = tuple(
        value.detach().clone().requires_grad_(need)
        for value, need in zip((x, *states, *parameters), needs, strict=True)
    )
    count = len(states)
    output, final, _ = python_module_sequence(
        values[0], values[1 : count + 1], values[count + 1 :], spec
    )
    roots = tuple(value for value in values if value.requires_grad)
    # A selected initial trace need not influence every final state. In the
    # dual-exponential cell, decay-only differentiation leaves the final rise
    # trace constant. Keep that final state in the value oracle, but supply
    # cotangents only for outputs that actually participate in autograd.
    differentiable = tuple(
        (value, gradient)
        for value, gradient in zip(
            (output, *final),
            (
                torch.ones_like(output),
                *(torch.full_like(value, 0.3) for value in final),
            ),
            strict=True,
        )
        if value.requires_grad
    )
    gradients = torch.autograd.grad(
        tuple(value for value, _ in differentiable),
        roots,
        tuple(gradient for _, gradient in differentiable),
        allow_unused=True,
    )
    return (
        output.detach(),
        *(value.detach() for value in final),
        *(
            torch.zeros_like(root) if gradient is None else gradient.detach()
            for root, gradient in zip(roots, gradients, strict=True)
        ),
    )


def _operation(implementation, backend, x, states, parameters, spec, needs):
    names = get_template(spec).state_names

    def run():
        with torch.no_grad():
            output, final, histories = implementation.forward(
                backend, x, states, parameters, spec, names
            )
            gx, gs, gp = implementation.backward(
                backend,
                x,
                states,
                parameters,
                spec,
                names,
                histories,
                torch.ones_like(output),
                tuple(torch.full_like(value, 0.3) for value in final),
                (None,) * len(histories),
                needs_input_grad=needs,
            )
        return (
            output,
            *final,
            *(value for value in (gx, *gs, *gp) if value is not None),
        )

    return run


def _check(actual, expected):
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=5e-5, rtol=5e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-engine", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--implementation",
        choices=("baseline", "candidate"),
        help="Measure one isolated arm; omitted runs both in separate processes",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("lif", "alpha", "dual_exponential"),
        default=("lif", "alpha", "dual_exponential"),
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=("input", "state", "parameters"),
        default=("input", "state", "parameters"),
    )
    parser.add_argument("--backends", nargs="+", default=("triton", "cupy", "tilelang"))
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--neurons", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if (
        min(args.steps, args.batch, args.neurons) < 1
        or args.warmup < 3
        or args.repeats < 30
    ):
        parser.error(
            "positive shapes, at least three warmups and thirty repeats required"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {"status": "running", "arguments": vars(args)}, default=str, indent=2
        )
        + "\n"
    )
    try:
        _run(args)
    except Exception:
        report = json.loads(args.report.read_text())
        report.update(status="fail", error=traceback.format_exc())
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        raise


def _run(args):
    if args.implementation is None:
        return _run_isolated(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Run this benchmark in a Slurm GPU allocation")
    label = args.implementation
    implementation = engine
    if label == "baseline":
        module_spec = importlib.util.spec_from_file_location(
            "btorch.models.kernels.backends._reverse_output_baseline",
            args.baseline_engine,
        )
        implementation = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = implementation
        module_spec.loader.exec_module(implementation)
    report = {
        "implementation": label,
        "pid": os.getpid(),
        "scope": (
            "complete raw primitive forward/reverse; excludes custom-op/autograd "
            "dispatch and graph pool memory"
        ),
        "shape_TBN": [args.steps, args.batch, args.neurons],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                args.baseline_engine,
                Path(engine.__file__),
                Path(__file__),
                Path(timing.__file__),
                Path("btorch/models/kernels/golden.py"),
                Path("btorch/models/kernels/templates.py"),
                Path("btorch/models/kernels/backends/fragments.py"),
                Path("btorch/models/neurons/lif.py"),
                Path("btorch/models/synapse.py"),
                Path("btorch/models/ode.py"),
            )
        },
        "cases": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for model in args.models:
        spec, x, states, parameters = _inputs(
            model, args.steps, args.batch, args.neurons
        )
        for scope in args.scopes:
            needs = _needs(scope, states, parameters)
            expected = _oracle(x, states, parameters, spec, needs)
            changed_expected = _oracle(x + 0.25, states, parameters, spec, needs)
            for backend in args.backends:
                case = {"model": model, "scope": scope, "backend": backend}
                report["active_case"] = case.copy()
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                run = _operation(
                    implementation, backend, x, states, parameters, spec, needs
                )
                _check(run(), expected)
                captured = {}

                def build():
                    capture_x = x.clone()
                    operation = _operation(
                        implementation,
                        backend,
                        capture_x,
                        states,
                        parameters,
                        spec,
                        needs,
                    )
                    values = list(operation())
                    captured["values"] = values
                    return operation, values, capture_x

                for _ in range(args.warmup):
                    run()
                torch.cuda.synchronize()
                measurements = {
                    "eager_device": timing._measure_device(torch, run, args.repeats),
                    "eager_host": timing._measure_host(torch, run, args.repeats),
                    "memory": timing._measure_memory(torch, run),
                    "captured": timing._measure_graph(
                        torch, build, args.warmup, args.repeats
                    ),
                }
                # The shared timing helper leaves graph outputs at the changed
                # drive after checking replay. Independently check those exact
                # captured tensors against the actual Module's changed-input VJP.
                _check(captured["values"], changed_expected)
                measurements["actual_module_changed_input_replay"] = "pass"
                case[label] = measurements
                case["status"] = "pass"
                report["cases"].append(case)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            key: case[key]
                            for key in ("model", "scope", "backend", "status")
                        }
                        | {"implementation": label}
                    ),
                    flush=True,
                )
    report.pop("active_case", None)
    report["status"] = "pass"
    args.report.write_text(json.dumps(report, indent=2) + "\n")


def _run_isolated(args):
    reports = {}
    for label in ("baseline", "candidate"):
        path = args.report.with_name(f"{args.report.stem}.{label}{args.report.suffix}")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--baseline-engine",
                str(args.baseline_engine.resolve()),
                "--report",
                str(path.resolve()),
                "--implementation",
                label,
                "--models",
                *args.models,
                "--scopes",
                *args.scopes,
                "--backends",
                *args.backends,
                "--steps",
                str(args.steps),
                "--batch",
                str(args.batch),
                "--neurons",
                str(args.neurons),
                "--warmup",
                str(args.warmup),
                "--repeats",
                str(args.repeats),
            ],
            check=True,
        )
        reports[label] = json.loads(path.read_text())
    baseline, candidate = reports["baseline"], reports["candidate"]
    identity = lambda case: (case["model"], case["scope"], case["backend"])
    if (
        baseline["status"] != "pass"
        or candidate["status"] != "pass"
        or baseline["source_sha256"] != candidate["source_sha256"]
        or [identity(case) for case in baseline["cases"]]
        != [identity(case) for case in candidate["cases"]]
    ):
        raise RuntimeError("Isolated benchmark arms disagree on source or workload")
    report = {
        key: value
        for key, value in candidate.items()
        if key not in ("implementation", "pid", "cases")
    }
    report["process_ids"] = {label: value["pid"] for label, value in reports.items()}
    report["isolation"] = "fresh process per implementation"
    report["cases"] = [
        {**old, "candidate": new["candidate"]}
        for old, new in zip(baseline["cases"], candidate["cases"], strict=True)
    ]
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
