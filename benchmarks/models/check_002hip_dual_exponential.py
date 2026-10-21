"""Qualify the restricted 002HIP dual-exponential mapping.

The reference is the unchanged public Torch ``DualExponentialPSC`` from the
002HIP ``main`` snapshot. The current 002HIP checkout is probed separately and
excluded because its public single-step route is broken. No legacy recurrence
is copied into this harness: outputs, states, and VJPs come from the archived
class itself.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import subprocess
import sys
import tarfile
import tempfile
import traceback
import types
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from btorch.models import environ
from btorch.models.kernels import KernelOptions
from btorch.models.synapse import DualExponentialPSC


REFERENCE_REVISION = "cf8f78e81eeb904409f03c3ba0fca2f92195e5f9"
BACKENDS = ("torch", "triton", "cupy", "tilelang")
PARAMETER_LAYOUTS = ("scalar", "per_neuron", "strided")
PARAMETER_NAMES = ("tau_decay", "tau_rise", "g_max", "E_syn", "v_r")
LEAF_NAMES = ("input", "weight", "bias", "g_init", "h_init", *PARAMETER_NAMES)


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-repo",
        type=Path,
        default=Path("/data1/fanqixuan/src/002HIP"),
    )
    parser.add_argument(
        "--reference-revision",
        default=REFERENCE_REVISION,
        help="Full 002HIP revision containing the executable public Torch class",
    )
    parser.add_argument(
        "--spikingjelly-root",
        type=Path,
        default=Path("/data1/fanqixuan/src/spikingjelly"),
    )
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=("torch",))
    parser.add_argument(
        "--parameter-layouts",
        nargs="+",
        choices=PARAMETER_LAYOUTS,
        default=PARAMETER_LAYOUTS,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--neurons", type=int, default=3)
    parser.add_argument("--split", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--precise-exp", action="store_true")
    parser.add_argument("--report", type=Path)
    result = parser.parse_args()
    if len(result.reference_revision) != 40:
        parser.error("--reference-revision must be a pinned full SHA")
    if min(result.steps, result.neurons) < 2:
        parser.error("steps and neurons must be at least two")
    if not 0 < result.split < result.steps:
        parser.error("split must be inside the sequence")
    if not all(
        math.isfinite(value) and value > 0 for value in (result.atol, result.rtol)
    ):
        parser.error("atol and rtol must be finite and positive")
    result.backends = tuple(dict.fromkeys(result.backends))
    result.parameter_layouts = tuple(dict.fromkeys(result.parameter_layouts))
    return result


def _git(repo: Path, *arguments: str, text: bool = True):
    return subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=True,
        capture_output=True,
        text=text,
        timeout=30,
    ).stdout


def _source_at(repo: Path, revision: str) -> bytes:
    return _git(repo, "show", f"{revision}:model/connection.py", text=False)


def _archive_model(repo: Path, revision: str, destination: Path) -> None:
    archive = _git(
        repo,
        "archive",
        "--format=tar",
        revision,
        "model/connection.py",
        text=False,
    )
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        handle.extractall(destination, filter="data")


def _enable_real_spikingjelly(root: Path) -> dict[str, str]:
    source_root = root.resolve()
    sys.path.insert(0, str(source_root))
    import spikingjelly
    from spikingjelly.activation_based import base

    package_source = Path(spikingjelly.__file__).resolve()
    base_source = Path(base.__file__).resolve()
    if not package_source.is_relative_to(source_root):
        raise RuntimeError(
            f"SpikingJelly resolved outside requested root: {package_source}"
        )
    if not base_source.is_relative_to(source_root):
        raise RuntimeError(
            f"SpikingJelly base resolved outside requested root: {base_source}"
        )

    # The checkout maps ``ops/`` to ``spikingjelly._ops`` at installation.
    # Recreate that package path so the real activation modules import from
    # source without installing or modifying the external checkout.
    if "spikingjelly._ops" not in sys.modules:
        operations = types.ModuleType("spikingjelly._ops")
        operations.__path__ = [str(source_root / "ops")]
        sys.modules["spikingjelly._ops"] = operations
    return {
        "root": str(source_root),
        "commit": _git(source_root, "rev-parse", "HEAD").strip(),
        "dirty": bool(_git(source_root, "status", "--porcelain").strip()),
        "base_source": str(base_source),
        "base_sha256": hashlib.sha256(base_source.read_bytes()).hexdigest(),
        "memory_module": f"{base.MemoryModule.__module__}.{base.MemoryModule.__name__}",
        "package_source": str(package_source),
    }


def _load_snapshot_class(archive_root: Path, revision: str):
    tag = revision[:12]
    package_name = f"_btorch_002hip_{tag}"
    model_name = f"{package_name}.model"
    package = types.ModuleType(package_name)
    package.__path__ = [str(archive_root)]
    model = types.ModuleType(model_name)
    model.__path__ = [str(archive_root / "model")]
    sys.modules[package_name] = package
    sys.modules[model_name] = model

    # These imports are required at module import time but are unreachable for
    # backend="torch". The archived scientific class and real framework base
    # remain unchanged; legacy CUDA kernels are explicitly outside this oracle.
    for suffix in ("connection_kernel", "connection_kernel_sparse"):
        name = f"{model_name}.{suffix}"
        sys.modules[name] = types.ModuleType(name)

    name = f"{model_name}.connection"
    path = archive_root / "model" / "connection.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import archived source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.DualExponentialPSC


def _device(requested: str, backend: str) -> torch.device:
    if requested != "auto":
        result = torch.device(requested)
    else:
        result = torch.device("cpu" if backend == "torch" else "cuda")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Backend {backend!r} requires CUDA in this run")
    if backend != "torch" and result.type != "cuda":
        raise ValueError(f"Backend {backend!r} must run on CUDA")
    return result


def _inputs(steps: int, neurons: int, seed: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = torch.randn(steps, 1, neurons, generator=generator) * 0.45
    weight = torch.randn(neurons, neurons, generator=generator) * 0.35
    bias = torch.linspace(0.03, -0.02, neurons)
    g_init = torch.randn(1, neurons, generator=generator) * 0.08
    h_init = torch.randn(1, neurons, generator=generator) * 0.05
    return tuple(
        value.to(device=device, dtype=torch.float32)
        for value in (source, weight, bias, g_init, h_init)
    )


def _leaf(value: Tensor) -> Tensor:
    return value.detach().clone().requires_grad_()


def _parameter_inputs(neurons: int, layout: str, device: torch.device):
    lane = torch.arange(neurons, dtype=torch.float32)
    values = {
        "tau_decay": 5.0 + 0.04 * lane,
        "tau_rise": 2.0 + 0.02 * lane,
        "g_max": 0.7 + 0.01 * lane,
        "E_syn": 3.0 + 0.03 * lane,
        "v_r": -0.1 + 0.01 * lane,
    }
    if layout == "scalar":
        values = {name: value[:1].squeeze(0) for name, value in values.items()}
    elif layout == "strided":
        strided = {}
        for index, (name, value) in enumerate(values.items()):
            backing = torch.full((2 * neurons,), -4.0 - index)
            backing[::2] = value
            strided[name] = backing
        values = strided
    return {name: value.to(device) for name, value in values.items()}


def _parameter_roots(
    values: dict[str, Tensor], layout: str, roots: dict[str, Tensor] | None = None
):
    if roots is None:
        roots = {name: _leaf(values[name]) for name in PARAMETER_NAMES}
    parameters = {
        name: root[::2] if layout == "strided" else root for name, root in roots.items()
    }
    return roots, parameters


def _legacy_model(cls, weight: Tensor, bias: Tensor, parameters):
    return cls(
        weight.shape[1],
        weight.shape[0],
        g_max=parameters["g_max"],
        E_syn=parameters["E_syn"],
        v_r=parameters["v_r"],
        tau_decay=parameters["tau_decay"],
        tau_rise=parameters["tau_rise"],
        weight0=weight.detach().clone(),
        bias0=bias.detach().clone(),
        step_mode="m",
        backend="torch",
    ).to(device=weight.device, dtype=weight.dtype)


def _legacy_run(
    cls,
    values,
    parameter_values,
    layout,
    chunks: tuple[int, ...],
    parameter_roots=None,
):
    source, weight, bias, g_init, h_init = map(_leaf, values)
    parameter_roots, parameters = _parameter_roots(
        parameter_values, layout, parameter_roots
    )
    model = _legacy_model(cls, weight, bias, parameters)
    with torch.no_grad():
        model.linear.weight.copy_(weight)
        model.linear.bias.copy_(bias)
    model.g = g_init
    model.h = h_init
    outputs = []
    start = 0
    for length in chunks:
        stop = start + length
        outputs.append(model.multi_step_forward(source[start:stop]))
        start = stop
    if start != source.shape[0]:
        raise AssertionError("Legacy chunks do not cover the sequence")
    return {
        "output": torch.cat(outputs),
        "g": model.g,
        "h": model.h,
        "leaves": (
            source,
            model.linear.weight,
            model.linear.bias,
            g_init,
            h_init,
            *(parameter_roots[name] for name in PARAMETER_NAMES),
        ),
    }


def _mapping(parameters, *, validate=True):
    tau_decay_legacy = parameters["tau_decay"]
    tau_rise_legacy = parameters["tau_rise"]
    q_decay = 1.0 - 1.0 / tau_decay_legacy
    q_rise = 1.0 - 1.0 / tau_rise_legacy
    valid = (
        (q_decay > 0)
        & (q_decay < 1)
        & (q_rise > 0)
        & (q_rise < 1)
        & (q_decay != q_rise)
    )
    if validate and not bool(valid.all()):
        raise ValueError(
            "Mapping requires distinct Euler factors strictly between 0 and 1"
        )
    if validate and not bool(
        ((parameters["g_max"] != 0) & torch.isfinite(parameters["g_max"])).all()
    ):
        raise ValueError("Mapping requires finite, nonzero g_max")
    tau_decay = -1.0 / torch.log(q_decay)
    tau_rise = -1.0 / torch.log(q_rise)
    difference = parameters["E_syn"] - parameters["v_r"]
    coefficient = difference * parameters["g_max"] / (q_decay - q_rise)
    amplitude = coefficient * tau_rise * tau_decay / (tau_decay - tau_rise)
    return q_decay, q_rise, tau_decay, tau_rise, amplitude


def _mapped_module(
    neurons,
    backend,
    device,
    dtype,
    tau_decay,
    tau_rise,
    amplitude,
    *,
    precise_exp=False,
):
    identity = nn.Identity()
    selected = (
        "torch"
        if backend == "torch"
        else KernelOptions(
            neuron=backend,
            enable_fp_fusion=False,
            precise_division=True,
            precise_exp=precise_exp,
        )
    )
    with environ.context(dt=1.0):
        model = DualExponentialPSC(
            neurons,
            tau_decay=tau_decay,
            tau_rise=tau_rise,
            linear=identity,
            A=amplitude,
            backend=selected,
        )
        model.init_state(batch_size=1, dtype=dtype, device=device)
    return model


def _mapped_run(
    values,
    parameter_values,
    layout,
    chunks: tuple[int, ...],
    backend: str,
    parameter_roots=None,
    *,
    precise_exp=False,
):
    source, weight, bias, g_init, h_init = map(_leaf, values)
    parameter_roots, parameters = _parameter_roots(
        parameter_values, layout, parameter_roots
    )
    qd, qr, tau_decay, tau_rise, amplitude = _mapping(parameters)
    model = _mapped_module(
        source.shape[-1],
        backend,
        source.device,
        source.dtype,
        tau_decay,
        tau_rise,
        amplitude,
        precise_exp=precise_exp,
    )
    g_max = parameters["g_max"]
    model.g_rise = h_init / (g_max * qr)
    model.g_decay = ((qd - qr) * g_init + h_init) / (g_max * qd)
    pending = torch.zeros_like(g_init)
    outputs = []
    start = 0
    with environ.context(dt=1.0):
        for length in chunks:
            stop = start + length
            projected = F.linear(source[start:stop], weight, bias)
            delayed = torch.cat((pending.unsqueeze(0), projected[:-1]), dim=0)
            if backend == "torch":
                outputs.append(
                    torch.stack([model.single_step_forward(value) for value in delayed])
                )
            else:
                outputs.append(model.multi_step_forward(delayed))
            pending = projected[-1]
            start = stop
    if start != source.shape[0]:
        raise AssertionError("Mapped chunks do not cover the sequence")

    # Native traces encode the legacy state before its final pending drive.
    h_lag = g_max * model.g_rise
    g_lag = g_max * (model.g_decay - model.g_rise) / (qd - qr)
    g_final = qd * g_lag + h_lag
    h_final = qr * h_lag + g_max * pending
    return {
        "output": torch.cat(outputs),
        "g": g_final,
        "h": h_final,
        "pending_projected_drive": pending,
        "mapped_g_rise": model.g_rise,
        "mapped_g_decay": model.g_decay,
        "leaves": (
            source,
            weight,
            bias,
            g_init,
            h_init,
            *(parameter_roots[name] for name in PARAMETER_NAMES),
        ),
        "mapping": {
            "q_decay": qd.detach().cpu().tolist(),
            "q_rise": qr.detach().cpu().tolist(),
            "tau_decay_btorch": tau_decay.detach().cpu().tolist(),
            "tau_rise_btorch": tau_rise.detach().cpu().tolist(),
            "A_btorch": amplitude.detach().cpu().tolist(),
            "drive_delay_steps": 1,
            "pending_drive": "unscaled complete projection Wz+b",
            "final_h": "g_max * (q_rise * mapped_g_rise + pending_drive)",
            "final_g": (
                "q_decay * g_max * (mapped_g_decay - mapped_g_rise) / "
                "(q_decay - q_rise) + g_max * mapped_g_rise"
            ),
        },
    }


def _loss(result):
    time = torch.linspace(
        0.3,
        1.1,
        result["output"].shape[0],
        device=result["output"].device,
        dtype=result["output"].dtype,
    )[:, None, None]
    return (
        (result["output"] * time).square().mean()
        + 0.13 * result["g"].square().mean()
        + 0.17 * result["h"].square().mean()
    )


def _difference(actual: Tensor, expected: Tensor):
    difference = (actual - expected).detach().abs()
    return {
        "max_abs": float(difference.max()),
        "l2": float(difference.norm()),
        "exact": bool(torch.equal(actual, expected)),
    }


def _assert(actual, expected, args, name):
    if not bool(torch.isfinite(actual).all()):
        raise AssertionError(f"{name}: actual tensor contains nonfinite values")
    if not bool(torch.isfinite(expected).all()):
        raise AssertionError(f"{name}: reference tensor contains nonfinite values")
    torch.testing.assert_close(
        actual, expected, atol=args.atol, rtol=args.rtol, msg=name
    )
    return _difference(actual, expected)


def _parameter_refresh(cls, values, parameter_values, layout, backend, args):
    roots, parameters = _parameter_roots(parameter_values, layout)
    identities = {name: root.data_ptr() for name, root in roots.items()}
    first_reference = _legacy_run(
        cls, values, parameter_values, layout, (args.steps,), roots
    )
    first_mapped = _mapped_run(
        values,
        parameter_values,
        layout,
        (args.steps,),
        backend,
        roots,
        precise_exp=getattr(args, "precise_exp", False),
    )
    result = {"first": {}}
    for name in ("output", "g", "h"):
        result["first"][name] = _assert(
            first_mapped[name], first_reference[name], args, f"refresh first {name}"
        )

    with torch.no_grad():
        parameters["tau_decay"].add_(0.17)
        parameters["tau_rise"].add_(0.05)
        parameters["g_max"].mul_(0.93)
        parameters["E_syn"].add_(0.11)
        parameters["v_r"].sub_(0.07)
    if any(root.data_ptr() != identities[name] for name, root in roots.items()):
        raise AssertionError("Parameter refresh replaced an original root tensor")

    second_reference = _legacy_run(
        cls, values, parameter_values, layout, (args.steps,), roots
    )
    second_mapped = _mapped_run(
        values,
        parameter_values,
        layout,
        (args.steps,),
        backend,
        roots,
        precise_exp=getattr(args, "precise_exp", False),
    )
    result["second"] = {}
    for name in ("output", "g", "h"):
        result["second"][name] = _assert(
            second_mapped[name],
            second_reference[name],
            args,
            f"refresh second {name}",
        )
    changed = _difference(second_reference["output"], first_reference["output"])
    if changed["max_abs"] == 0.0:
        raise AssertionError("Changed parameter roots did not change the reference")
    result["reference_change"] = changed
    result["root_identity_preserved"] = True
    return result


def _case(cls, backend, layout, args):
    device = _device(args.device, backend)
    values = _inputs(args.steps, args.neurons, args.seed, device)
    parameter_values = _parameter_inputs(args.neurons, layout, device)
    whole = (args.steps,)
    split = (args.split, args.steps - args.split)
    reference = _legacy_run(cls, values, parameter_values, layout, whole)
    reference_chunked = _legacy_run(cls, values, parameter_values, layout, split)
    mapped = _mapped_run(
        values,
        parameter_values,
        layout,
        whole,
        backend,
        precise_exp=getattr(args, "precise_exp", False),
    )
    mapped_chunked = _mapped_run(
        values,
        parameter_values,
        layout,
        split,
        backend,
        precise_exp=getattr(args, "precise_exp", False),
    )

    result = {
        "backend": backend,
        "device": str(device),
        "parameter_layout": layout,
        "parameter_roots": {
            name: {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "mapped_view_stride": (
                    list(value[::2].stride()) if layout == "strided" else None
                ),
            }
            for name, value in parameter_values.items()
        },
        "shape_TBN": [args.steps, 1, args.neurons],
        "precise_exp": getattr(args, "precise_exp", False),
        "mapping": mapped["mapping"],
        "connection": "actual archived dense nn.Linear",
        "sparse_reference": (
            "unsupported: archived backend='torch' ignores w_values and uses "
            "its dense linear; archived CUDA sparse is not a valid oracle"
        ),
        "forward": {},
        "continuation": {},
        "gradients": {},
    }
    for name in ("output", "g", "h"):
        result["forward"][name] = _assert(mapped[name], reference[name], args, name)
        result["continuation"][f"legacy_{name}"] = _assert(
            reference_chunked[name], reference[name], args, f"legacy chunk {name}"
        )
        result["continuation"][f"mapped_{name}"] = _assert(
            mapped_chunked[name], mapped[name], args, f"mapped chunk {name}"
        )
    result["continuation"]["pending_projected_drive"] = _assert(
        mapped_chunked["pending_projected_drive"],
        mapped["pending_projected_drive"],
        args,
        "pending projected drive",
    )

    reference_gradients = torch.autograd.grad(_loss(reference), reference["leaves"])
    mapped_gradients = torch.autograd.grad(_loss(mapped), mapped["leaves"])
    for name, actual, expected in zip(
        LEAF_NAMES,
        mapped_gradients,
        reference_gradients,
        strict=True,
    ):
        result["gradients"][name] = _assert(actual, expected, args, name)
    if layout == "strided":
        result["strided_backing_gradients"] = {}
        for offset, name in enumerate(PARAMETER_NAMES, 5):
            actual = mapped_gradients[offset]
            expected = reference_gradients[offset]
            if bool(actual[1::2].count_nonzero()) or bool(
                expected[1::2].count_nonzero()
            ):
                raise AssertionError(
                    f"{name}: inactive backing slots received gradient"
                )
            result["strided_backing_gradients"][name] = {
                "root_shape": list(actual.shape),
                "active_view_stride": list(actual[::2].stride()),
                "inactive_max_abs": float(actual[1::2].abs().max()),
            }
    # In particular, the pending projected drive must keep its graph across
    # the chunk boundary and into the final h loss, including its bias path.
    for label, chunked in (
        ("legacy_chunk_vjp", reference_chunked),
        ("mapped_chunk_vjp", mapped_chunked),
    ):
        chunk_gradients = torch.autograd.grad(_loss(chunked), chunked["leaves"])
        result["continuation"][label] = {
            name: _assert(actual, expected, args, f"{label} {name}")
            for name, actual, expected in zip(
                LEAF_NAMES,
                chunk_gradients,
                reference_gradients,
                strict=True,
            )
        }
    result["parameter_refresh"] = _parameter_refresh(
        cls, values, parameter_values, layout, backend, args
    )
    result["status"] = "pass"
    return result


def _probe_current_head(cls, neurons):
    weight = torch.eye(neurons)
    bias = torch.zeros(neurons)
    parameter_values = _parameter_inputs(neurons, "scalar", torch.device("cpu"))
    _, parameters = _parameter_roots(parameter_values, "scalar")
    model = _legacy_model(cls, weight, bias, parameters)
    source = torch.zeros(1, neurons)
    if hasattr(model, "reset_states"):
        model.reset_states(source)
    try:
        model.single_step_forward(source, torch.zeros_like(source))
    except TypeError as error:
        if "current_charge" not in str(error) or "positional argument" not in str(
            error
        ):
            raise
        return {
            "status": "excluded_broken_public_api",
            "exception": type(error).__name__,
            "message": str(error),
        }
    return {"status": "executable_not_qualified"}


def main():
    args = _args()
    repo = args.reference_repo.resolve()
    resolved = _git(repo, "rev-parse", args.reference_revision).strip()
    if resolved != args.reference_revision:
        raise ValueError(
            f"Pinned reference resolved to {resolved}, expected "
            f"{args.reference_revision}"
        )
    spikingjelly = _enable_real_spikingjelly(args.spikingjelly_root)
    current_revision = _git(repo, "rev-parse", "HEAD").strip()
    report = {
        "status": "pass",
        "contract": (
            "restricted output/state/VJP mapping of archived 002HIP Euler "
            "cascade to btorch DualExponentialPSC"
        ),
        "reference": {
            "repo": str(repo),
            "revision": resolved,
            "connection_sha256": hashlib.sha256(_source_at(repo, resolved)).hexdigest(),
            "public_path": "model.connection.DualExponentialPSC backend='torch'",
            "cuda_imports": "not loaded; unreachable from the reference path",
            "external_cuda_qualification": False,
            "working_tree_dirty": bool(_git(repo, "status", "--porcelain").strip()),
        },
        "current_head": {
            "revision": current_revision,
            "connection_sha256": hashlib.sha256(
                _source_at(repo, current_revision)
            ).hexdigest(),
        },
        "spikingjelly": spikingjelly,
        "btorch": {
            "commit": _git(Path.cwd(), "rev-parse", "HEAD").strip(),
            "synapse_sha256": hashlib.sha256(
                Path("btorch/models/synapse.py").read_bytes()
            ).hexdigest(),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "cases": [],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="btorch-002hip-dual-") as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            current = root / "current"
            snapshot.mkdir()
            current.mkdir()
            _archive_model(repo, resolved, snapshot)
            _archive_model(repo, current_revision, current)
            reference_class = _load_snapshot_class(snapshot, resolved)
            current_class = _load_snapshot_class(current, current_revision)
            report["current_head"]["probe"] = _probe_current_head(
                current_class, args.neurons
            )
            for backend in args.backends:
                for layout in args.parameter_layouts:
                    report["cases"].append(
                        _case(reference_class, backend, layout, args)
                    )
    except Exception as error:
        report["status"] = "fail"
        report["exception"] = type(error).__name__
        report["message"] = str(error)
        report["traceback"] = "".join(traceback.format_exception(error))[-12000:]
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    print(text)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
