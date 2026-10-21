"""Qualify current GLIF3 RecurrentNN dense and sparse execution.

This benchmark compares generated recurrent operators with the ordinary
Torch Module using the same GLIF3 neuron, selected PSC dynamics and
connection. The benchmark-branch GLIF networks apply recurrent weights
directly to the previous spike and have no PSC, so they are recorded as
an excluded reference rather than used as a complete-network performance
baseline.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import torch

from btorch.models import environ
from btorch.models.connection import SparseConnection
from btorch.models.functional import init_net_state
from btorch.models.kernels import KernelOptions
from btorch.models.neurons import GLIF3
from btorch.models.rnn import RecurrentNN
from btorch.models.surrogate import ATan
from btorch.models.synapse import AlphaPSC, DualExponentialPSC


BACKENDS = ("triton", "tilelang", "cupy")
SOURCE_PATHS = (
    "benchmarks/models/check_recurrent_glif.py",
    "btorch/config.py",
    "btorch/models/neurons/glif.py",
    "btorch/models/ode.py",
    "btorch/models/surrogate/atan.py",
    "btorch/models/surrogate/base.py",
    "btorch/models/synapse.py",
    "btorch/models/kernels/recurrent.py",
    "btorch/models/kernels/recurrent_operator.py",
    "btorch/models/kernels/options.py",
    "btorch/models/kernels/contracts.py",
    "btorch/models/kernels/templates.py",
    "btorch/models/kernels/backends/engine.py",
    "btorch/models/kernels/backends/fragments.py",
    "btorch/models/kernels/backends/fused.py",
    "btorch/models/connection/sparse.py",
    "btorch/sparse/runtime/ops.py",
)
BRANCH_REVISION = "4bd0b0a649188173801549dcb970d50b2efd772c"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
    parser.add_argument(
        "--psc",
        choices=("alpha", "dual_exponential"),
        default="alpha",
        help="PSC model; the default preserves the established Alpha protocol",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=(
            "dense-1",
            "dense-2",
            "dense-12",
            "sparse-1",
            "sparse-2",
            "sparse-12",
            "sparse-20x256",
        ),
        help="Run only selected correctness cases; repeat the option as needed",
    )
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--no-fp-fusion", action="store_true")
    parser.add_argument("--precise-division", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--scaling-neurons",
        nargs="+",
        type=int,
        help="Add fixed-indegree sparse memory cases, intended for Slurm",
    )
    parser.add_argument("--scaling-steps", type=int, default=20)
    parser.add_argument("--indegree", type=int, default=32)
    parser.add_argument("--forward-atol", type=float, default=3e-5)
    parser.add_argument("--forward-rtol", type=float, default=3e-5)
    parser.add_argument("--gradient-atol", type=float, default=8e-5)
    parser.add_argument("--gradient-rtol", type=float, default=8e-4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.scaling_steps, args.indegree) < 1:
        parser.error("warmup, repeats, scaling steps and indegree must be positive")
    if args.scaling_neurons and min(args.scaling_neurons) < 1:
        parser.error("scaling neuron counts must be positive")
    args.backends = tuple(dict.fromkeys(args.backends))
    return args


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(root):
    def git(*arguments):
        return subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

    return {
        "root": str(root),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "sources": {name: _sha256(root / name) for name in SOURCE_PATHS},
    }


def _environment():
    packages = {}
    for name in ("cupy-cuda12x", "triton", "tilelang"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "torch": torch.__version__,
        "surrogate_execution": "plain_torch",
        "reference_backward_warmup": 0,
        "cuda_runtime": torch.version.cuda,
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory": properties.total_memory,
        "packages": packages,
    }


def _edges(neurons, indegree, device, seed):
    """Create deterministic post-major edges without duplicate pairs."""
    indegree = min(indegree, neurons)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    offsets = torch.randperm(neurons, generator=generator)[:indegree]
    post = torch.arange(neurons).repeat_interleave(indegree)
    pre = (post.reshape(neurons, indegree) + offsets).remainder(neurons).reshape(-1)
    scale = max(indegree, 1) ** -0.5
    values = scale * torch.sin(torch.arange(pre.numel(), dtype=torch.float32) * 0.17)
    return pre.to(device), post.to(device), values.to(device)


def _connection(kind, neurons, indegree, device, seed):
    pre, post, values = _edges(neurons, indegree, device, seed)
    bias = 0.02 * torch.cos(torch.arange(neurons, device=device) * 0.13)
    if kind == "sparse":
        connection = SparseConnection.from_edges(
            pre,
            post,
            neurons,
            neurons,
            values=values,
            bias=bias,
        )
        return connection, int(values.numel())
    weight = torch.zeros(neurons, neurons, device=device)
    weight[post, pre] = values
    connection = torch.nn.Linear(neurons, neurons, bias=True, device=device)
    with torch.no_grad():
        connection.weight.copy_(weight)
        connection.bias.copy_(bias)
    return connection, int(values.numel())


def _psc_state_names(psc):
    if psc == "alpha":
        return ("psc", "h")
    return ("psc", "g_rise", "g_decay")


def _psc_parameter_names(psc):
    if psc == "alpha":
        return ("tau_syn", "g_max")
    return ("tau_rise", "tau_decay", "a")


def _records(psc):
    if psc == "alpha":
        return (
            "neuron.v",
            "neuron.Iasc",
            "neuron.refractory",
            "synapse.psc",
            "synapse.h",
        )
    return (
        "neuron.v",
        "neuron.Iasc",
        "neuron.refractory",
        "synapse.psc",
        "synapse.g_rise",
        "synapse.g_decay",
    )


def _psc_contract(psc):
    if psc == "alpha":
        return "AlphaPSC tau_syn=3, g_max=1"
    return "DualExponentialPSC tau_decay=5, tau_rise=2, a=0.3"


def _psc_class_name(psc):
    return "AlphaPSC" if psc == "alpha" else "DualExponentialPSC"


def _build(kind, steps, neurons, batch, backend, indegree, seed, policy, psc):
    device = torch.device("cuda")
    lane = torch.arange(neurons, device=device, dtype=torch.float32)
    modes = torch.arange(2, device=device, dtype=torch.float32)
    connection, nnz = _connection(kind, neurons, indegree, device, seed)
    neuron = GLIF3(
        neurons,
        v_threshold=-50.0 + 0.3 * torch.sin(lane * 0.07),
        v_reset=-70.0 + 0.2 * torch.cos(lane * 0.05),
        v_rest=-70.0,
        c_m=0.05 + 0.002 * (lane.remainder(5) / 5),
        tau=20.0 + lane.remainder(7) * 0.1,
        k=0.1 + 0.03 * ((lane[:, None] + modes) % 5),
        asc_amps=0.03 * torch.sin(lane[:, None] * 0.11 + modes),
        tau_ref=0.0,
        hard_reset=False,
        surrogate_function=ATan(alpha=2.0, spiking=True),
        trainable_param={"v_threshold", "c_m", "tau", "k", "asc_amps"},
        device=device,
        dtype=torch.float32,
    )
    if psc == "alpha":
        synapse = AlphaPSC(neurons, 3.0, connection, g_max=1.0)
        original_psc_parameters = {}
    else:
        tau_decay = torch.tensor(5.0, device=device, requires_grad=True)
        tau_rise = torch.tensor(2.0, device=device, requires_grad=True)
        synapse = DualExponentialPSC(neurons, tau_decay, tau_rise, connection, A=1.0)
        synapse.a = synapse.a.detach().clone().requires_grad_()
        original_psc_parameters = {
            "tau_rise": tau_rise,
            "tau_decay": tau_decay,
            "a": synapse.a,
        }
    records = _records(psc)
    model = RecurrentNN(
        neuron,
        synapse,
        backend=(
            "torch" if backend == "eager" else KernelOptions(neuron=backend, **policy)
        ),
        update_state_names=records,
    ).to(device)
    for name, original in original_psc_parameters.items():
        if getattr(model.synapse, name) is not original:
            raise AssertionError(f"Dual-exponential {name} lost its original root")
    init_net_state(model, batch_size=batch, device=device, dtype=torch.float32)
    phase = lane[None] + torch.arange(batch, device=device)[:, None] * 0.37
    model.neuron.v = (-63.0 + 9.0 * torch.sin(phase * 0.19)).requires_grad_()
    model.neuron.Iasc = (
        0.02 * torch.sin(phase[..., None] * 0.13 + modes)
    ).requires_grad_()
    model.neuron.refractory = torch.zeros(batch, neurons, device=device)
    model.synapse.psc = (0.01 * torch.cos(phase * 0.17)).requires_grad_()
    if psc == "alpha":
        model.synapse.h = (0.01 * torch.sin(phase * 0.23)).requires_grad_()
        model.synapse.tau_syn.requires_grad_()
        model.synapse.g_max.requires_grad_()
    else:
        model.synapse.g_rise = (0.012 * torch.sin(phase * 0.23)).requires_grad_()
        model.synapse.g_decay = (0.015 * torch.cos(phase * 0.29)).requires_grad_()
    time_axis = torch.arange(steps, device=device)[:, None, None]
    x = 0.45 + 0.35 * torch.sin(time_axis * 0.31 + phase[None] * 0.07)
    x = x.requires_grad_()
    return model, x, nnz


def _states(model, psc):
    states = {
        "initial.neuron.v": model.neuron.v,
        "initial.neuron.Iasc": model.neuron.Iasc,
    }
    states.update(
        {
            f"initial.synapse.{name}": getattr(model.synapse, name)
            for name in _psc_state_names(psc)
        }
    )
    return states


def _leaves(model, x, psc):
    result = {"input": x, **_states(model, psc)}
    for name, value in model.named_parameters():
        result[f"parameter.{name}"] = value
    for name in _psc_parameter_names(psc):
        value = getattr(model.synapse, name)
        if value.requires_grad and all(
            value is not existing for existing in result.values()
        ):
            result[f"parameter.synapse.{name}"] = value
    return result


def _final_states(model, psc):
    states = {
        "neuron.v": model.neuron.v,
        "neuron.Iasc": model.neuron.Iasc,
        "neuron.refractory": model.neuron.refractory,
    }
    states.update(
        {
            f"synapse.{name}": getattr(model.synapse, name)
            for name in _psc_state_names(psc)
        }
    )
    return states


def _loss(output, records, final):
    loss = output.square().mean()
    for index, value in enumerate(records.values(), 1):
        loss = loss + index * 0.07 * value.square().mean()
    for index, value in enumerate(final.values(), 1):
        if value.is_floating_point():
            loss = loss + index * 0.03 * value.square().mean()
    return loss


def _run(model, x, psc, *, validate_records=True):
    output, records = model(x)
    record_names = _records(psc)
    if validate_records:
        record_keys = set(records)
        expected_keys = set(record_names)
        if record_keys != expected_keys:
            raise AssertionError(
                "Recurrent records differ from the declared contract: "
                f"missing={sorted(expected_keys - record_keys)}, "
                f"unexpected={sorted(record_keys - expected_keys)}"
            )
    # The eager loop and generated operator may construct dictionaries in
    # different orders. Record names define semantics, so normalize before
    # assigning deterministic loss coefficients or serializing results.
    records = {name: records[name] for name in record_names}
    final = _final_states(model, psc)
    return output, records, final, _loss(output, records, final)


def _compare_mapping(actual, expected, atol, rtol):
    actual_keys, expected_keys = set(actual), set(expected)
    if actual_keys != expected_keys:
        raise AssertionError(
            "Mapping keys differ: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    for name in sorted(actual_keys):
        torch.testing.assert_close(
            actual[name],
            expected[name],
            atol=atol,
            rtol=rtol,
            msg=lambda message, name=name: f"[{name}] {message}",
        )


def _require_gradient(name, gradient, implementation):
    if gradient is None:
        raise AssertionError(
            f"{implementation} disconnected declared gradient root {name!r}"
        )
    return gradient


def _first_spike_divergence(actual, expected):
    mismatch = actual != expected
    if not bool(mismatch.any()):
        return None
    flat = int(mismatch.flatten().nonzero()[0])
    coordinate = tuple(
        int(value)
        for value in torch.unravel_index(
            torch.tensor(flat, device=actual.device), actual.shape
        )
    )
    return {
        "index": list(coordinate),
        "actual": float(actual[coordinate]),
        "expected": float(expected[coordinate]),
        "mismatch_count": int(mismatch.sum()),
    }


def _tensor_diagnostic(actual, expected, atol, rtol):
    difference = (actual - expected).abs()
    flat = int(difference.argmax()) if difference.numel() else 0
    coordinate = (
        tuple(
            int(value)
            for value in torch.unravel_index(
                torch.tensor(flat, device=actual.device), actual.shape
            )
        )
        if difference.numel()
        else ()
    )
    close = torch.isclose(actual, expected, atol=atol, rtol=rtol)
    denominator = expected.abs().clamp_min(torch.finfo(expected.dtype).tiny)
    relative = difference / denominator
    return {
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "max_rel": float(relative.max()) if relative.numel() else 0.0,
        "max_abs_index": list(coordinate),
        "actual_at_max_abs": float(actual[coordinate]) if coordinate else None,
        "expected_at_max_abs": float(expected[coordinate]) if coordinate else None,
        "outside_tolerance": int((~close).sum()),
    }


def _mapping_diagnostics(actual, expected, atol, rtol):
    shared = sorted(set(actual) & set(expected))
    return {
        "missing": sorted(set(expected) - set(actual)),
        "unexpected": sorted(set(actual) - set(expected)),
        "values": {
            name: _tensor_diagnostic(actual[name], expected[name], atol, rtol)
            for name in shared
        },
    }


def _continuation(prototype, x, psc, split, atol, rtol):
    if x.shape[0] < 2:
        first = copy.deepcopy(prototype)
        second = copy.deepcopy(prototype)
        with torch.no_grad():
            first_result = _run(first, x, psc)
            second_result = _run(second, x, psc)
        torch.testing.assert_close(
            first_result[0], second_result[0], atol=atol, rtol=rtol
        )
        _compare_mapping(first_result[1], second_result[1], atol, rtol)
        _compare_mapping(first_result[2], second_result[2], atol, rtol)
        return
    full = copy.deepcopy(prototype)
    chunked = copy.deepcopy(prototype)
    with torch.no_grad():
        full_output, full_records = full(x)[:2]
        left_output, left_records = chunked(x[:split])[:2]
        right_output, right_records = chunked(x[split:])[:2]
    torch.testing.assert_close(
        torch.cat((left_output, right_output)), full_output, atol=atol, rtol=rtol
    )
    for name in full_records:
        torch.testing.assert_close(
            torch.cat((left_records[name], right_records[name])),
            full_records[name],
            atol=atol,
            rtol=rtol,
        )
    _compare_mapping(_final_states(chunked, psc), _final_states(full, psc), atol, rtol)


def _bptt_continuation(
    prototype,
    x,
    psc,
    split,
    forward_atol,
    forward_rtol,
    gradient_atol,
    gradient_rtol,
):
    full = copy.deepcopy(prototype)
    chunked = copy.deepcopy(prototype)
    full_x = x.detach().clone().requires_grad_()
    chunked_x = x.detach().clone().requires_grad_()
    full_leaves = _leaves(full, full_x, psc)
    chunked_leaves = _leaves(chunked, chunked_x, psc)

    full_result = _run(full, full_x, psc)
    full_gradients = torch.autograd.grad(
        full_result[-1], tuple(full_leaves.values()), allow_unused=True
    )
    left_output, left_records = chunked(chunked_x[:split])
    right_output, right_records = chunked(chunked_x[split:])
    output = torch.cat((left_output, right_output))
    records = {
        name: torch.cat((left_records[name], right_records[name]))
        for name in _records(psc)
    }
    final = _final_states(chunked, psc)
    chunked_loss = _loss(output, records, final)
    chunked_gradients = torch.autograd.grad(
        chunked_loss, tuple(chunked_leaves.values()), allow_unused=True
    )

    torch.testing.assert_close(
        output, full_result[0], atol=forward_atol, rtol=forward_rtol
    )
    _compare_mapping(records, full_result[1], forward_atol, forward_rtol)
    _compare_mapping(final, full_result[2], forward_atol, forward_rtol)
    for name, actual, expected in zip(
        chunked_leaves,
        chunked_gradients,
        full_gradients,
        strict=True,
    ):
        actual = _require_gradient(name, actual, "chunked Torch continuation")
        expected = _require_gradient(name, expected, "whole Torch continuation")
        torch.testing.assert_close(
            actual,
            expected,
            atol=gradient_atol,
            rtol=gradient_rtol,
            msg=lambda message, name=name: f"[{name}] {message}",
        )


def _saved_storage(model, x, psc):
    saved = []

    def pack(value):
        saved.append(
            {
                "shape": list(value.shape),
                "numel": value.numel(),
                "bytes": value.untyped_storage().nbytes(),
                "storage_ptr": value.untyped_storage().data_ptr(),
            }
        )
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        result = _run(model, x, psc)
    return result, saved


def _capture_check(prototype, x, psc, expected_loss, expected_gradient, atol, rtol):
    model = copy.deepcopy(prototype)
    value = x.detach().clone().requires_grad_()

    def loss_fn(argument):
        # Eager qualification already validates keys. Skipping the Python set
        # operation keeps full-graph capture focused on tensor computation.
        return _run(model, argument, psc, validate_records=False)[-1]

    # Module memories become graph outputs after each invocation. Restore
    # ordinary leaf tensors before starting the next graph step; copying into
    # the old output after mark_step_begin would access invalidated storage.
    initial = tuple(
        (
            owner,
            name,
            getattr(owner, name).detach().clone(),
            getattr(owner, name).requires_grad,
        )
        for owner in model.modules()
        if hasattr(owner, "_memories_rv")
        for name in owner._memories_rv
        if isinstance(getattr(owner, name), torch.Tensor)
    )
    compiled = torch.compile(loss_fn, mode="reduce-overhead", fullgraph=True)
    original_replay = torch.cuda.CUDAGraph.replay
    replay_counts = []

    def count_replay(graph):
        replay_counts[-1] += 1
        return original_replay(graph)

    with patch.object(torch.cuda.CUDAGraph, "replay", count_replay):
        for _ in range(3):
            # These initial leaves are immutable, so reuse their addresses.
            # Fresh clones per invocation could trigger capture again instead
            # of exercising graph replay with stable external state storage.
            for owner, name, original, needs_gradient in initial:
                setattr(owner, name, original.requires_grad_(needs_gradient))
            torch.compiler.cudagraph_mark_step_begin()
            replay_counts.append(0)
            loss = compiled(value)
            gradient = torch.autograd.grad(loss, value)[0]
            torch.testing.assert_close(loss, expected_loss, atol=atol, rtol=rtol)
            torch.testing.assert_close(
                gradient, expected_gradient, atol=atol, rtol=rtol
            )
            del loss, gradient
    if replay_counts[-1] == 0:
        raise AssertionError(f"No CUDA Graph replay on final trial: {replay_counts}")
    return replay_counts


def _eager_loss_gradients(prototype, x, psc):
    model = copy.deepcopy(prototype)
    model.backend = "torch"
    value = x.detach().clone().requires_grad_()
    leaves = _leaves(model, value, psc)
    loss = _run(model, value, psc)[-1]
    gradients = torch.autograd.grad(loss, tuple(leaves.values()), allow_unused=True)
    return loss.detach(), {
        name: _require_gradient(name, gradient, "eager Torch oracle").detach()
        for name, gradient in zip(leaves, gradients, strict=True)
    }


def _capture_full_vjp_check(prototype, x, psc, atol, rtol):
    model = copy.deepcopy(prototype)
    value = x.detach().clone().requires_grad_()
    original_x = value.detach().clone()
    initial = []
    for owner in model.modules():
        if not hasattr(owner, "_memories_rv"):
            continue
        for name in owner._memories_rv:
            current = getattr(owner, name)
            if not isinstance(current, torch.Tensor):
                continue
            original = current.detach().clone().requires_grad_(current.requires_grad)
            setattr(owner, name, original)
            initial.append((owner, name, original))
    leaves = _leaves(model, value, psc)

    def loss_fn(argument):
        return _run(model, argument, psc, validate_records=False)[-1]

    expected = {
        offset: _eager_loss_gradients(prototype, original_x + offset, psc)
        for offset in (0.0, 0.25)
    }
    compiled = torch.compile(loss_fn, mode="reduce-overhead", fullgraph=True)
    original_replay = torch.cuda.CUDAGraph.replay
    replay_counts = []

    def count_replay(graph):
        replay_counts[-1] += 1
        return original_replay(graph)

    with patch.object(torch.cuda.CUDAGraph, "replay", count_replay):
        for offset in (0.0, 0.0, 0.25):
            for owner, name, original in initial:
                setattr(owner, name, original)
            with torch.no_grad():
                value.copy_(original_x + offset)
            torch.compiler.cudagraph_mark_step_begin()
            replay_counts.append(0)
            loss = compiled(value)
            gradients = torch.autograd.grad(
                loss, tuple(leaves.values()), allow_unused=True
            )
            expected_loss, expected_gradients = expected[offset]
            torch.testing.assert_close(loss, expected_loss, atol=atol, rtol=rtol)
            for name, gradient in zip(leaves, gradients, strict=True):
                gradient = _require_gradient(name, gradient, "captured recurrent graph")
                torch.testing.assert_close(
                    gradient,
                    expected_gradients[name],
                    atol=atol,
                    rtol=rtol,
                    msg=lambda message, name=name: f"[{name}] {message}",
                )
            del loss, gradients
    if replay_counts[-1] == 0:
        raise AssertionError(f"No CUDA Graph replay on final trial: {replay_counts}")
    return replay_counts


def _timing(prototype, x, psc, warmup, repeats):
    values = []
    for _ in range(warmup + repeats):
        model = copy.deepcopy(prototype)
        value = x.detach().clone().requires_grad_()
        leaves = tuple(_leaves(model, value, psc).values())
        start, middle, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        start.record()
        loss = _run(model, value, psc)[-1]
        middle.record()
        torch.autograd.grad(loss, leaves, allow_unused=True)
        end.record()
        end.synchronize()
        values.append((start.elapsed_time(middle), middle.elapsed_time(end)))
    measured = values[warmup:]
    return {
        "forward_ms_median": statistics.median(item[0] for item in measured),
        "backward_ms_median": statistics.median(item[1] for item in measured),
        "note": (
            "Fresh-model complete recurrent calls; excludes construction "
            "and compilation"
        ),
    }


def _case(kind, steps, neurons, backend, args, *, scaling=False):
    started = time.perf_counter()
    seed = 610 + steps + neurons
    case_indegree = (
        args.indegree if scaling else min(args.indegree, max(4, neurons // 4))
    )
    policy = {
        "enable_fp_fusion": not args.no_fp_fusion,
        "precise_division": args.precise_division,
    }
    prototype, x, nnz = _build(
        kind, steps, neurons, 1, backend, case_indegree, seed, policy, args.psc
    )
    reference = copy.deepcopy(prototype)
    reference.backend = "torch"
    actual_model = copy.deepcopy(prototype)
    actual_x = x.detach().clone().requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    actual_leaves = _leaves(actual_model, actual_x, args.psc)
    reference_leaves = _leaves(reference, reference_x, args.psc)
    expected = _run(reference, reference_x, args.psc)
    expected_gradients = torch.autograd.grad(
        expected[-1], tuple(reference_leaves.values()), allow_unused=True
    )
    gc.collect()
    torch.cuda.empty_cache()
    gc_enabled = gc.isenabled()
    if gc_enabled:
        gc.disable()
    try:
        torch.cuda.reset_peak_memory_stats()
        baseline_bytes = torch.cuda.memory_allocated()
        actual, saved = _saved_storage(actual_model, actual_x, args.psc)
        actual_gradients = torch.autograd.grad(
            actual[-1], tuple(actual_leaves.values()), allow_unused=True
        )
        training_peak_bytes = torch.cuda.max_memory_allocated() - baseline_bytes
    finally:
        if gc_enabled:
            gc.enable()
    spike_divergence = _first_spike_divergence(actual[0], expected[0])
    diagnostics = {
        "output": _tensor_diagnostic(
            actual[0], expected[0], args.forward_atol, args.forward_rtol
        ),
        "records": _mapping_diagnostics(
            actual[1], expected[1], args.forward_atol, args.forward_rtol
        ),
        "final_states": _mapping_diagnostics(
            actual[2], expected[2], args.forward_atol, args.forward_rtol
        ),
        "first_spike_divergence": spike_divergence,
    }
    try:
        torch.testing.assert_close(
            actual[0], expected[0], atol=args.forward_atol, rtol=args.forward_rtol
        )
        _compare_mapping(actual[1], expected[1], args.forward_atol, args.forward_rtol)
        _compare_mapping(actual[2], expected[2], args.forward_atol, args.forward_rtol)
    except Exception as error:
        error.qualification_diagnostics = diagnostics
        raise
    spike_count = int(actual[0].detach().sum())
    if not 0 < spike_count < actual[0].numel():
        raise AssertionError(
            f"Expected mixed spike activity, got {spike_count}/{actual[0].numel()}"
        )
    gradient_norms = {}
    for name, value, target, source in zip(
        actual_leaves, actual_gradients, expected_gradients, actual_leaves.values()
    ):
        try:
            if args.psc == "dual_exponential":
                value = _require_gradient(name, value, "native recurrent reverse")
                target = _require_gradient(name, target, "Torch Module oracle")
            else:
                value = torch.zeros_like(source) if value is None else value
                target = torch.zeros_like(source) if target is None else target
            torch.testing.assert_close(
                value, target, atol=args.gradient_atol, rtol=args.gradient_rtol
            )
        except Exception as error:
            # Preserve forward evidence when the first failure is in reverse,
            # and name the actual Module leaf rather than inferring it from
            # a tensor shape in the traceback.
            error.qualification_diagnostics = {
                **diagnostics,
                "gradient_name": name,
                "actual_gradient_connected": value is not None,
                "reference_gradient_connected": target is not None,
            }
            if value is not None and target is not None:
                error.qualification_diagnostics["gradient"] = _tensor_diagnostic(
                    value, target, args.gradient_atol, args.gradient_rtol
                )
            raise
        gradient_norms[name] = float(value.norm())
    _continuation(
        prototype,
        x.detach(),
        args.psc,
        max(1, steps // 2),
        args.forward_atol,
        args.forward_rtol,
    )
    bptt_continuation = None
    if args.psc == "dual_exponential" and steps >= 2:
        _bptt_continuation(
            prototype,
            x,
            args.psc,
            max(1, steps // 2),
            args.forward_atol,
            args.forward_rtol,
            args.gradient_atol,
            args.gradient_rtol,
        )
        bptt_continuation = "pass"
    dense_saved = [
        item
        for item in saved
        if item["shape"] == [neurons, neurons]
        or item["bytes"] >= neurons * neurons * torch.finfo(torch.float32).bits // 8
        and item["numel"] == neurons * neurons
    ]
    if kind == "sparse" and dense_saved:
        raise AssertionError(f"Sparse reverse saved N-by-N storage: {dense_saved}")
    weight_grad_numel = None
    if kind == "sparse":
        candidates = [
            (name, gradient)
            for name, gradient in zip(actual_leaves, actual_gradients)
            if "synapse.linear.weight" in name and gradient is not None
        ]
        if len(candidates) != 1:
            raise AssertionError(
                f"Expected one sparse weight gradient, got {candidates}"
            )
        weight_grad_numel = candidates[0][1].numel()
        if weight_grad_numel != nnz:
            raise AssertionError(
                f"Sparse gradient has {weight_grad_numel} values, expected {nnz}"
            )
    twin = None
    if kind == "sparse" and not scaling:
        dense_twin, twin_x, _ = _build(
            "dense",
            steps,
            neurons,
            1,
            backend,
            case_indegree,
            seed,
            policy,
            args.psc,
        )
        with torch.no_grad():
            sparse_result = _run(copy.deepcopy(prototype), x.detach(), args.psc)
            dense_result = _run(dense_twin, twin_x.detach(), args.psc)
        torch.testing.assert_close(
            sparse_result[0],
            dense_result[0],
            atol=args.forward_atol,
            rtol=args.forward_rtol,
        )
        _compare_mapping(
            sparse_result[1], dense_result[1], args.forward_atol, args.forward_rtol
        )
        _compare_mapping(
            sparse_result[2], dense_result[2], args.forward_atol, args.forward_rtol
        )
        twin = "pass"
    capture = None
    capture_replays = None
    capture_vjp_scope = None
    capture_changed_input = None
    if args.capture and not scaling and steps in (2, 12) and neurons == 32:
        if args.psc == "alpha":
            input_gradient = expected_gradients[tuple(reference_leaves).index("input")]
            capture_replays = _capture_check(
                prototype,
                x,
                args.psc,
                expected[-1].detach(),
                input_gradient.detach(),
                args.gradient_atol,
                args.gradient_rtol,
            )
            capture_vjp_scope = ["input"]
            capture_changed_input = False
        else:
            capture_replays = _capture_full_vjp_check(
                prototype,
                x,
                args.psc,
                args.gradient_atol,
                args.gradient_rtol,
            )
            capture_vjp_scope = list(reference_leaves)
            capture_changed_input = True
        capture = "pass"
    timing = (
        _timing(prototype, x, args.psc, args.warmup, args.repeats)
        if args.timing
        else None
    )
    unique_saved = {}
    for item in saved:
        unique_saved[item["storage_ptr"]] = max(
            item["bytes"], unique_saved.get(item["storage_ptr"], 0)
        )
    public_saved = [
        {name: value for name, value in item.items() if name != "storage_ptr"}
        for item in saved
    ]
    return {
        "status": "pass",
        "backend": backend,
        "connection": kind,
        "psc": args.psc,
        "shape_TBN": [steps, 1, neurons],
        "nnz": nnz,
        "indegree": case_indegree,
        "compile_policy": policy,
        "records": list(_records(args.psc)),
        "gradient_norms": gradient_norms,
        "spike_count": spike_count,
        "first_spike_divergence": spike_divergence,
        "numerical_diagnostics": diagnostics,
        "training_peak_allocated_bytes": training_peak_bytes,
        "unique_saved_storage_bytes": sum(unique_saved.values()),
        "largest_saved_tensors": sorted(
            public_saved, key=lambda item: item["bytes"], reverse=True
        )[:12],
        "sparse_weight_gradient_numel": weight_grad_numel,
        "dense_sparse_twin": twin,
        "bptt_continuation": bptt_continuation,
        "capture": capture,
        "capture_replays_per_trial": capture_replays,
        "capture_vjp_scope": capture_vjp_scope,
        "capture_changed_input": capture_changed_input,
        "timing": timing,
        "elapsed_s": time.perf_counter() - started,
    }


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = Path(__file__).resolve().parents[2]
    selected = set(args.case or ())
    cases = [
        ("dense", 1, 32, "dense-1"),
        ("dense", 2, 32, "dense-2"),
        ("dense", 12, 32, "dense-12"),
        ("sparse", 1, 32, "sparse-1"),
        ("sparse", 2, 32, "sparse-2"),
        ("sparse", 12, 32, "sparse-12"),
        ("sparse", 20, 256, "sparse-20x256"),
    ]
    if selected:
        cases = [case for case in cases if case[3] in selected]
    report = {
        "qualification": (
            "current RecurrentNN GLIF3 dense/sparse"
            if args.psc == "alpha"
            else "current RecurrentNN GLIF3+dual_exponential dense/sparse"
        ),
        "current": _identity(root),
        "environment": _environment(),
        "reference": {
            "authority": "ordinary Torch RecurrentNN Module",
            "benchmark_branch_revision": BRANCH_REVISION,
            "branch_complete_network_status": "excluded",
            "reason": (
                "The branch applies W @ spike directly before GLIF3 and has no "
                "PSC or delay state; current RecurrentNN uses "
                f"{_psc_class_name(args.psc)} ordering."
            ),
        },
        "contract": {
            "dtype": "float32",
            "dt": 1.0,
            "neuron": "GLIF3 soft reset, tau_ref=0, two ASC modes, ATan alpha=2",
            "psc": _psc_contract(args.psc),
            "psc_state_names": list(_psc_state_names(args.psc)),
            "psc_parameter_roots": list(_psc_parameter_names(args.psc)),
            "sparse_orientation": "row=post, col=pre",
            "compile_policy": {
                "enable_fp_fusion": not args.no_fp_fusion,
                "precise_division": args.precise_division,
                "scope": (
                    "Generated recurrent kernels only; compilation policy does "
                    "not change CellSpec model semantics"
                ),
            },
            "checks": [
                "spikes",
                "all physical records and final states",
                "split continuation",
                "input/state/model/PSC/connection VJPs",
                *(
                    [
                        "whole versus chunked full BPTT continuation",
                        "captured changed-input full VJPs",
                    ]
                    if args.psc == "dual_exponential"
                    else []
                ),
                "sparse nnz-sized weight gradient",
                "no saved N-by-N sparse tensor",
            ],
        },
        "cases": [],
        "scaling_checks": [],
    }
    work = [(kind, steps, neurons, False) for kind, steps, neurons, _ in cases]
    if args.scaling_neurons:
        work.extend(
            ("sparse", args.scaling_steps, neurons, True)
            for neurons in args.scaling_neurons
        )
    for backend in args.backends:
        for kind, steps, neurons, scaling in work:
            try:
                with environ.context(dt=1.0):
                    row = _case(kind, steps, neurons, backend, args, scaling=scaling)
                row["scaling"] = scaling
            except Exception as error:
                row = {
                    "status": "fail",
                    "backend": backend,
                    "connection": kind,
                    "psc": args.psc,
                    "shape_TBN": [steps, 1, neurons],
                    "scaling": scaling,
                    "compile_policy": {
                        "enable_fp_fusion": not args.no_fp_fusion,
                        "precise_division": args.precise_division,
                    },
                    "exception": type(error).__name__,
                    "message": str(error),
                    "numerical_diagnostics": getattr(
                        error, "qualification_diagnostics", None
                    ),
                    "traceback": "".join(traceback.format_exception(error))[-6000:],
                }
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
    for backend in args.backends:
        rows = sorted(
            (
                row
                for row in report["cases"]
                if row.get("scaling")
                and row.get("backend") == backend
                and row.get("status") == "pass"
            ),
            key=lambda row: row["shape_TBN"][2],
        )
        for smaller, larger in zip(rows, rows[1:]):
            small_t, _, small_n = smaller["shape_TBN"]
            large_t, _, large_n = larger["shape_TBN"]
            if (
                large_n != 2 * small_n
                or large_t != small_t
                or larger["indegree"] != smaller["indegree"]
            ):
                continue
            ratio = (
                larger["training_peak_allocated_bytes"]
                / smaller["training_peak_allocated_bytes"]
            )
            check = {
                "backend": backend,
                "neurons": [small_n, large_n],
                "steps": small_t,
                "indegree": smaller["indegree"],
                "training_peak_ratio": ratio,
                "limit": 2.5,
                "status": "pass" if ratio <= 2.5 else "fail",
            }
            report["scaling_checks"].append(check)
    report["status"] = (
        "pass"
        if report["cases"]
        and all(row["status"] == "pass" for row in report["cases"])
        and all(check["status"] == "pass" for check in report["scaling_checks"])
        else "fail"
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    print(text)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
