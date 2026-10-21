"""Benchmark native stochastic recurrence against actual btorch Modules.

This benchmark uses the public :class:`RecurrentNN`, :class:`ALIF`,
:class:`ExponentialPSC`, and :class:`PoissonRandomSpike` implementations.  A
separate Torch-backed network is the pathwise correctness oracle; no neuron,
synapse, reset, or surrogate equations are reproduced here.

Run all installed native backends on CUDA::

    PYTHONPATH=. python benchmarks/models/bench_poisson_recurrent.py \
        --report run/parity/poisson-recurrent-benchmark.json

The correctness gate aligns the global CUDA generator before each candidate
and oracle call, compares whole and two-chunk continuation, and differentiates
inputs, initial memories, neuron parameters, PSC decay, and sparse connection
parameters. Timings cover independent whole-sequence calls without public
records. ``torch.compile(mode="reduce-overhead")`` supplies the captured
forward and backward measurements and is checked again after input mutation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import torch
from check_benchmark_glif import _identity
from torch import Tensor

from btorch.models import environ
from btorch.models.connection import SparseConnection, Synapse
from btorch.models.functional import init_net_state
from btorch.models.kernels.adapters import prepare_cell
from btorch.models.kernels.options import resolve_model_options
from btorch.models.kernels.registry import resolve_cell_backend
from btorch.models.neurons import ALIF
from btorch.models.rnn import RecurrentNN
from btorch.models.surrogate import PoissonRandomSpike
from btorch.models.synapse import ExponentialPSC


BACKENDS = ("triton", "tilelang", "cupy")
STATE_NAMES = ("v", "g_k", "refractory", "psc")
DT = 0.27
FORWARD_ATOL = 2e-5
FORWARD_RTOL = 2e-5
GRADIENT_ATOL = 8e-5
GRADIENT_RTOL = 8e-4
SOURCE_PATHS = (
    "benchmarks/models/bench_poisson_recurrent.py",
    "benchmarks/models/check_benchmark_glif.py",
    "btorch/models/rnn.py",
    "btorch/models/neurons/alif.py",
    "btorch/models/synapse.py",
    "btorch/models/surrogate/poisson_random.py",
    "btorch/models/kernels/recurrent.py",
    "btorch/models/kernels/recurrent_operator.py",
    "btorch/models/kernels/operators.py",
    "btorch/models/kernels/options.py",
    "btorch/models/kernels/backends/engine.py",
    "btorch/models/kernels/backends/fused.py",
    "btorch/models/kernels/backends/fragments.py",
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--neurons", type=int, default=256)
    parser.add_argument("--indegree", type=int, default=8)
    parser.add_argument("--split", type=int)
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
    parser.add_argument(
        "--precise-exp",
        action="store_true",
        help="Opt into accurate exponentials without changing the default",
    )
    parser.add_argument(
        "--no-fp-fusion",
        action="store_true",
        help="Opt out of the production floating-point contraction policy",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.neurons, args.indegree) < 1:
        parser.error("steps, batch, neurons and indegree must be positive")
    if args.indegree >= args.neurons:
        parser.error("indegree must be smaller than neurons")
    if args.warmup < 3 or args.repeats < 10:
        parser.error("warmup must be at least 3 and repeats at least 10")
    args.split = args.steps // 2 if args.split is None else args.split
    if not 0 < args.split < args.steps:
        parser.error("split must be inside the sequence")
    args.backends = tuple(dict.fromkeys(args.backends))
    return args


def _connection(neurons: int, indegree: int) -> SparseConnection:
    post = torch.arange(neurons, device="cuda").repeat_interleave(indegree)
    offsets = torch.arange(1, indegree + 1, device="cuda").repeat(neurons)
    pre = (post + offsets) % neurons
    edge = torch.arange(post.numel(), device="cuda", dtype=torch.float32)
    values = 0.035 * torch.sin(edge * 0.17) + 0.055
    bias = torch.linspace(-0.012, 0.018, neurons, device="cuda")
    return SparseConnection.from_edges(
        pre,
        post,
        neurons,
        neurons,
        Synapse(),
        values=values,
        bias=bias,
    )


def _build(args, backend: str) -> RecurrentNN:
    trainable = {
        "v_threshold",
        "v_reset",
        "c_m",
        "g_leak",
        "E_leak",
        "E_k",
        "tau_adapt",
        "dg_k",
        "tau_ref",
    }
    neuron = ALIF(
        args.neurons,
        v_threshold=0.82,
        v_reset=-0.12,
        c_m=1.1,
        g_leak=0.65,
        E_leak=-0.07,
        E_k=-0.8,
        tau_adapt=4.0,
        dg_k=0.10,
        tau_ref=0.75,
        trainable_param=trainable,
        surrogate_function=PoissonRandomSpike(
            tau=2.4,
            rho=0.72,
            leak=0.18,
            k=1.08,
            damping_factor=0.57,
        ),
        hard_reset=True,
        pre_spike_v=True,
    )
    if backend == "torch":
        options = "torch"
    else:
        defaults = resolve_model_options(backend, "alif")
        options = replace(
            defaults,
            connection="triton",
            enable_fp_fusion=(
                False if args.no_fp_fusion else defaults.enable_fp_fusion
            ),
            precise_exp=True if args.precise_exp else defaults.precise_exp,
        )
    model = RecurrentNN(
        neuron,
        ExponentialPSC(args.neurons, 2.7, _connection(args.neurons, args.indegree)),
        backend=options,
        update_state_names=(),
    ).to(device="cuda", dtype=torch.float32)
    init_net_state(model, batch_size=args.batch, device="cuda", dtype=torch.float32)
    lane = torch.arange(
        args.batch * args.neurons, device="cuda", dtype=torch.float32
    ).reshape(args.batch, args.neurons)
    model.neuron.v = (-0.08 + 0.025 * torch.sin(lane * 0.07)).requires_grad_()
    model.neuron.g_k = (0.025 + 0.006 * torch.cos(lane * 0.11)).requires_grad_()
    refractory = 0.05 * (lane.remainder(3.0))
    refractory[0, 0] = 3.0
    model.neuron.refractory = refractory.requires_grad_()
    model.synapse.psc = (0.018 * torch.sin(lane * 0.13)).requires_grad_()
    model.synapse.tau_syn.requires_grad_()
    return model


def _inputs(args) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(args.seed)
    drive = 0.30 + 0.30 * torch.rand(
        args.steps, args.batch, args.neurons, generator=generator
    )
    events = 0.028 * torch.randn(
        args.steps, args.batch, args.neurons, generator=generator
    )
    return drive.cuda(), events.cuda()


class _Runner:
    """Own one actual MemoryModule and stable independent-call roots."""

    def __init__(self, model: RecurrentNN, x: Tensor, events: Tensor, split: int):
        self.model = model
        self.x = x.detach().clone().requires_grad_()
        self.events = events.detach().clone().requires_grad_()
        self.split = split
        self.initial = (
            model.neuron.v,
            model.neuron.g_k,
            model.neuron.refractory,
            model.synapse.psc,
        )
        self.roots = OrderedDict((("input", self.x), ("events", self.events)))
        self.roots.update(
            (f"initial.{name}", value)
            for name, value in zip(STATE_NAMES, self.initial, strict=True)
        )
        self.roots["synapse.tau_syn"] = model.synapse.tau_syn
        seen = {id(value) for value in self.roots.values()}
        for name, parameter in model.named_parameters():
            if id(parameter) not in seen:
                self.roots[name] = parameter
                seen.add(id(parameter))
        generator = torch.Generator().manual_seed(8831)
        self.output_cotangent = torch.randn(
            x.shape, generator=generator, dtype=torch.float32
        ).cuda()
        self.state_cotangents = tuple(
            torch.randn(value.shape, generator=generator, dtype=torch.float32).cuda()
            for value in self.initial
        )

    def reset(self):
        (
            self.model.neuron.v,
            self.model.neuron.g_k,
            self.model.neuron.refractory,
            self.model.synapse.psc,
        ) = self.initial

    def run(
        self,
        sequence: str,
        x: Tensor | None = None,
        events: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        x = self.x if x is None else x
        events = self.events if events is None else events
        if sequence == "whole":
            output, records = self.model(x, events)
        else:
            left, left_records = self.model(x[: self.split], events[: self.split])
            right, right_records = self.model(x[self.split :], events[self.split :])
            if left_records or right_records:
                raise AssertionError("Performance path unexpectedly returned records")
            output, records = torch.cat((left, right)), {}
        if records:
            raise AssertionError("Performance path unexpectedly returned records")
        final = (
            self.model.neuron.v,
            self.model.neuron.g_k,
            self.model.neuron.refractory,
            self.model.synapse.psc,
        )
        return (output, *final)

    def loss(self, result: tuple[Tensor, ...]) -> Tensor:
        loss = (result[0] * self.output_cotangent).sum()
        return loss + sum(
            (state * cotangent).sum()
            for state, cotangent in zip(result[1:], self.state_cotangents, strict=True)
        )

    def training(self, sequence: str, *, check_backward_rng: bool = False):
        self.reset()
        result = self.run(sequence)
        rng_before_backward = torch.cuda.get_rng_state() if check_backward_rng else None
        gradients = torch.autograd.grad(
            self.loss(result), tuple(self.roots.values()), allow_unused=True
        )
        if check_backward_rng and not torch.equal(
            rng_before_backward, torch.cuda.get_rng_state()
        ):
            raise AssertionError("Backward consumed CUDA random numbers")
        return (*result, *gradients)


def _assert_values(actual, expected, label: str):
    if len(actual) != len(expected):
        raise AssertionError(f"{label}: result arity mismatch")
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        torch.testing.assert_close(
            value,
            reference,
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
            msg=f"{label} output {index}",
        )


def _assert_training(actual, expected, root_names, label: str):
    _assert_values(actual[:5], expected[:5], label)
    for name, value, reference in zip(
        root_names, actual[5:], expected[5:], strict=True
    ):
        if (value is None) != (reference is None):
            raise AssertionError(f"{label}: disconnected root mismatch for {name}")
        if reference is None:
            raise AssertionError(f"{label}: intended root {name} is disconnected")
        if not bool(torch.isfinite(value).all()):
            raise AssertionError(f"{label}: nonfinite gradient for {name}")
        torch.testing.assert_close(
            value,
            reference,
            atol=GRADIENT_ATOL,
            rtol=GRADIENT_RTOL,
            msg=f"{label} gradient {name}",
        )


def _pathwise_call(
    runner: _Runner,
    sequence: str,
    training: bool,
    *,
    check_backward_rng: bool = False,
):
    if training:
        return runner.training(sequence, check_backward_rng=check_backward_rng)
    runner.reset()
    return runner.run(sequence)


def _parity(candidate: _Runner, oracle: _Runner, sequence: str, training: bool):
    initial_rng = torch.cuda.get_rng_state()
    actual = _pathwise_call(
        candidate,
        sequence,
        training,
        check_backward_rng=training,
    )
    actual_rng = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(initial_rng)
    expected = _pathwise_call(
        oracle,
        sequence,
        training,
        check_backward_rng=training,
    )
    expected_rng = torch.cuda.get_rng_state()
    if not torch.equal(actual_rng, expected_rng):
        raise AssertionError(f"{sequence}: global CUDA RNG continuation differs")
    if training:
        _assert_training(actual, expected, tuple(candidate.roots), sequence)
    else:
        _assert_values(actual, expected, sequence)
    return actual


def _inference_autograd_tape(runner: _Runner):
    saved = []
    with (
        torch.autograd.graph.saved_tensors_hooks(
            lambda value: saved.append(value) or value, lambda value: value
        ),
        torch.inference_mode(),
    ):
        runner.reset()
        result = runner.run("whole")
    if saved or any(
        value.requires_grad or value.grad_fn is not None for value in result
    ):
        raise AssertionError("Inference retained a backward tape")
    return {"saved_tensors": 0, "status": "pass"}


def _tape(runner: _Runner):
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda value: saved.append(value) or value, lambda value: value
    ):
        runner.reset()
        result = runner.run("whole")
        loss = runner.loss(result)
    gradients = torch.autograd.grad(loss, tuple(runner.roots.values()))
    del gradients, loss, result
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
    }


def _summary(values):
    return {
        "median_us": statistics.median(values) * 1000.0,
        "min_us": min(values) * 1000.0,
        "max_us": max(values) * 1000.0,
    }


def _measure(operation, warmup: int, repeats: int):
    for _ in range(warmup):
        result = operation()
    del result
    torch.cuda.synchronize()
    device, host = [], []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        started = time.perf_counter()
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        device.append(start.elapsed_time(end))
        host.append((time.perf_counter() - started) * 1000.0)
    del result
    return {"device": _summary(device), "host_end_to_end": _summary(host)}


def _memory(operation):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    result = operation()
    torch.cuda.synchronize()
    absolute_peak = torch.cuda.max_memory_allocated()
    returned = sum(
        value.numel() * value.element_size()
        for value in result
        if isinstance(value, Tensor)
    )
    del result
    return {
        "baseline_allocated_bytes": baseline,
        "absolute_peak_allocated_bytes": absolute_peak,
        "peak_incremental_allocated_bytes": max(absolute_peak - baseline, 0),
        "returned_bytes": returned,
        "scope_note": (
            "Incremental peak excludes allocations, including compiled graph pools, "
            "that already existed when this measured call began"
        ),
    }


def _preload_backend(candidate: _Runner, backend: str):
    with environ.context(dt=DT):
        neuron_spec, *_ = prepare_cell(candidate.model.neuron, candidate.x)
        synapse_spec, *_ = prepare_cell(candidate.model.synapse, candidate.events)
    resolve_cell_backend(backend, neuron_spec, candidate.x.device)
    resolve_cell_backend(backend, synapse_spec, candidate.x.device)


def _compiled_checks(
    candidate: _Runner,
    oracle: _Runner,
    args,
    backend: str,
    training: bool,
):
    sequence = "whole"
    torch._dynamo.reset()
    _preload_backend(candidate, backend)

    def step(value, events):
        result = candidate.run(sequence, value, events)
        return (candidate.loss(result), *result) if training else result

    compiled = torch.compile(step, mode="reduce-overhead", fullgraph=True)

    def operation(*, check_backward_rng=False):
        candidate.reset()
        torch.compiler.cudagraph_mark_step_begin()
        if training:
            values = compiled(candidate.x, candidate.events)
            rng_before_backward = (
                torch.cuda.get_rng_state() if check_backward_rng else None
            )
            gradients = torch.autograd.grad(
                values[0], tuple(candidate.roots.values()), allow_unused=True
            )
            if check_backward_rng and not torch.equal(
                rng_before_backward, torch.cuda.get_rng_state()
            ):
                raise AssertionError("Captured backward consumed CUDA RNG")
            return (*values[1:], *gradients)
        with torch.inference_mode():
            return compiled(candidate.x, candidate.events)

    trials = (
        (candidate.x.detach().clone(), candidate.events.detach().clone()),
        (
            candidate.x.detach() * 0.73 + 0.07,
            candidate.events.detach() * -0.61 + 0.015,
        ),
        (
            candidate.x.detach().flip(0) * 1.09 - 0.04,
            candidate.events.detach().roll(1, 0) * 0.82,
        ),
        (
            candidate.x.detach().roll(2, 0) * 0.64 + 0.12,
            candidate.events.detach().flip(0) * -0.77,
        ),
    )
    before = None
    replay_calls = []
    replay_graphs = []
    original_replay = torch.cuda.CUDAGraph.replay

    def counted_replay(graph):
        replay_calls.append(graph)
        return original_replay(graph)

    torch.cuda.CUDAGraph.replay = counted_replay
    try:
        for trial_index, (x_value, event_value) in enumerate(trials):
            with torch.no_grad():
                candidate.x.copy_(x_value)
                candidate.events.copy_(event_value)
                oracle.x.copy_(x_value)
                oracle.events.copy_(event_value)
            initial_rng = torch.cuda.get_rng_state()
            replay_start = len(replay_calls)
            actual = operation(check_backward_rng=training)
            actual_rng = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(initial_rng)
            if training:
                expected = _pathwise_call(
                    oracle,
                    sequence,
                    True,
                    check_backward_rng=True,
                )
            else:
                with torch.inference_mode():
                    expected = _pathwise_call(oracle, sequence, False)
            if not torch.equal(actual_rng, torch.cuda.get_rng_state()):
                raise AssertionError("Captured call changed global RNG continuation")
            if training:
                _assert_training(
                    actual,
                    expected,
                    tuple(candidate.roots),
                    f"compiled trial {trial_index}",
                )
            else:
                _assert_values(actual, expected, f"compiled trial {trial_index}")
            if before is not None and all(
                torch.equal(old, new) for old, new in zip(before, actual, strict=True)
            ):
                raise AssertionError("Captured outputs and VJPs ignored changed inputs")
            before = tuple(value.detach().clone() for value in actual)
            replay_graphs.append(tuple(replay_calls[replay_start:]))
            del actual, expected
    finally:
        torch.cuda.CUDAGraph.replay = original_replay

    expected_replays = 2 if training else 1
    if not all(len(value) == expected_replays for value in replay_graphs[-2:]):
        raise AssertionError("Stable calls did not use captured CUDA Graphs")
    if not all(
        previous is current
        for previous, current in zip(replay_graphs[-2], replay_graphs[-1], strict=True)
    ):
        raise AssertionError("Changed inputs selected different CUDA Graphs")
    del before, event_value, trials, x_value

    timing = _measure(operation, args.warmup, args.repeats)
    memory = _memory(operation)
    return (
        timing,
        memory,
        {
            "changed_input_output_and_vjp": "pass" if training else "pass (outputs)",
            "global_rng_continuation": "pass",
            "backward_rng_unchanged": "pass" if training else "not applicable",
            "capture": "verified stable torch.compile CUDAGraph Tree replay identities",
            "graphs_per_replay": expected_replays,
        },
    )


def _backend_case(args, backend: str, x: Tensor, events: Tensor):
    candidate = _Runner(_build(args, backend), x, events, args.split)
    oracle = _Runner(_build(args, "torch"), x, events, args.split)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.cuda.manual_seed(args.seed + 101)
        whole = _parity(candidate, oracle, "whole", True)
        torch.cuda.manual_seed(args.seed + 101)
        chunked = _parity(candidate, oracle, "two_chunk", True)
        _assert_training(
            chunked,
            whole,
            tuple(candidate.roots),
            "native whole/two_chunk",
        )
        inference_tape = _inference_autograd_tape(candidate)
        tape = _tape(candidate)
        inference_timing, inference_memory, inference_capture = _compiled_checks(
            candidate, oracle, args, backend, False
        )
        training_timing, captured_training_memory, training_capture = _compiled_checks(
            candidate, oracle, args, backend, True
        )

        def eager_inference():
            with torch.inference_mode():
                return _pathwise_call(candidate, "whole", False)

        eager_training = lambda: _pathwise_call(candidate, "whole", True)
        measurements = {
            "eager_inference": _measure(eager_inference, args.warmup, args.repeats),
            "eager_training": _measure(eager_training, args.warmup, args.repeats),
            "captured_inference": inference_timing,
            "captured_training": training_timing,
            "inference_memory": _memory(eager_inference),
            "training_memory": _memory(eager_training),
            "captured_inference_memory": inference_memory,
            "captured_training_memory": captured_training_memory,
        }
    return {
        "backend": backend,
        "status": "pass",
        "kernel_policy": {
            "neuron": candidate.model.backend.neuron,
            "connection": candidate.model.backend.connection,
            "enable_fp_fusion": candidate.model.backend.enable_fp_fusion,
            "precise_division": candidate.model.backend.precise_division,
            "precise_exp": candidate.model.backend.precise_exp,
        },
        "checks": {
            "actual_module_whole_full_vjp": "pass",
            "actual_module_two_chunk_full_vjp": "pass",
            "whole_chunk_output_state_vjp": "pass",
            "backward_rng_unchanged": "pass",
            "inference_autograd_tape": inference_tape,
            "training_tape": tape,
            "captured_inference": inference_capture,
            "captured_training": training_capture,
        },
        "measurements": measurements,
        "differentiable_roots": list(candidate.roots),
    }


def _torch_baseline(args, x: Tensor, events: Tensor):
    runner = _Runner(_build(args, "torch"), x, events, args.split)

    def inference():
        with torch.inference_mode():
            return _pathwise_call(runner, "whole", False)

    training = lambda: _pathwise_call(runner, "whole", True)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.cuda.manual_seed(args.seed + 307)
        return {
            "inference_autograd_tape": _inference_autograd_tape(runner),
            "training_tape": _tape(runner),
            "eager_inference": _measure(inference, args.warmup, args.repeats),
            "eager_training": _measure(training, args.warmup, args.repeats),
            "inference_memory": _memory(inference),
            "training_memory": _memory(training),
        }


def _add_speedups(cases, baseline):
    inference = baseline["eager_inference"]["device"]["median_us"]
    training = baseline["eager_training"]["device"]["median_us"]
    for case in cases:
        measurements = case["measurements"]
        case["speedup_ratios_vs_actual_module_eager"] = {
            "comparison": (
                "Each ratio uses the matching actual btorch Module eager device "
                "median as denominator; captured native rows are not compared with "
                "a captured Torch or external 002HIP CUDA baseline"
            ),
            "ratios": {
                "native_eager_inference": inference
                / measurements["eager_inference"]["device"]["median_us"],
                "native_eager_training": training
                / measurements["eager_training"]["device"]["median_us"],
                "native_captured_inference": inference
                / measurements["captured_inference"]["device"]["median_us"],
                "native_captured_training": training
                / measurements["captured_training"]["device"]["median_us"],
            },
        }


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    root = Path(__file__).resolve().parents[2]
    x, events = _inputs(args)
    baseline = _torch_baseline(args, x, events)
    cases = []
    for backend in args.backends:
        case = _backend_case(args, backend, x, events)
        cases.append(case)
        print(json.dumps(case), flush=True)
    _add_speedups(cases, baseline)
    report = {
        "status": "pass",
        "scope": (
            "actual btorch recurrent ALIF+ExponentialPSC+PoissonRandomSpike; "
            "sparse projected recurrence; no copied equations"
        ),
        "performance_qualification": {
            "status": "informational",
            "acceptance_gate": None,
            "note": (
                "Correctness, RNG continuation, VJP, and graph-replay checks are "
                "gates; timings and speedup ratios have no fixed external native "
                "performance threshold"
            ),
        },
        "shape_TBN": [args.steps, args.batch, args.neurons],
        "edges": args.neurons * args.indegree,
        "split": args.split,
        "dt": DT,
        "precise_exp": args.precise_exp,
        "no_fp_fusion": args.no_fp_fusion,
        "tolerances": {
            "forward": [FORWARD_ATOL, FORWARD_RTOL],
            "gradient": [GRADIENT_ATOL, GRADIENT_RTOL],
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "identity": _identity(root, SOURCE_PATHS),
        "actual_module_baseline": baseline,
        "cases": cases,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    with environ.context(dt=DT):
        main()
