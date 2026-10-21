"""Probe unchanged 002HIP Izhikevich kernels against btorch references.

This is an optional correctness probe, not a timing benchmark. It
imports the original 002HIP kernel file directly so unrelated package
initializers and their dependencies do not affect the comparison.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path


KNOWN_SPIKINGJELLY_REVISION = "7c596a05d039fbc4f51242021ca85def7ae6e30f"
KNOWN_SPIKINGJELLY_HASHES = {
    "spikingjelly/configure.py": (
        "2e0d986af7aa906c3c699c41b9ef59b76eafd2a2faed3496039549a165f4bccd"
    ),
    "spikingjelly/activation_based/cuda_utils.py": (
        "80b68f933eaf6da2c9e76e28b308f6cfcf52f49faa0411a2d666f45c067b8581"
    ),
    "spikingjelly/activation_based/surrogate.py": (
        "23485ac69bc84180464ee090a8ed46503b2519c02bcc75bb33efd710082df8b6"
    ),
    "spikingjelly/activation_based/tensor_cache.py": (
        "1d1fb6b5e81bc1fba7cdf6cf5f1f81831742ecf46ab876036294c2f1e4b69728"
    ),
}


@dataclass(frozen=True)
class IzhikevichCase:
    """Define one explicitly reported physical parameter workload."""

    v_threshold: float = -40.0
    v_reset: float = -65.0
    capacitance: float = 100.0
    k: float = 2.0
    v_rest: float = -60.0
    v_peak: float = 30.0
    a: float = 0.02
    b: float = 0.2
    d: float = 8.0

    @property
    def btorch_parameters(self):
        return (
            self.v_threshold,
            self.v_reset,
            self.capacitance,
            self.k,
            self.v_rest,
            self.v_peak,
            self.a,
            self.b,
            self.d,
        )

    @property
    def hip_arguments(self):
        return (
            self.capacitance,
            self.v_peak,
            self.v_reset,
            self.v_rest,
            self.b,
            self.d,
            self.a,
            self.v_threshold,
            self.k,
        )


DEFAULT_CASE = IzhikevichCase()
PARAMETER_VALUES = DEFAULT_CASE.btorch_parameters
HIP_ARGUMENT_VALUES = DEFAULT_CASE.hip_arguments


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hip-root",
        type=Path,
        required=True,
        help="Unmodified 002HIP checkout containing model/neuron_kernel_*.py",
    )
    parser.add_argument(
        "--spikingjelly-root",
        type=Path,
        required=True,
        help="Compatible checkout whose root contains the spikingjelly package",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("scalar", "tensor"),
        default=("scalar",),
        help="002HIP source variants to execute; tensor backward is known suspect",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("triton", "tilelang", "cupy"),
        default=("triton", "tilelang", "cupy"),
        help="btorch generated backends checked against its Python Module",
    )
    parser.add_argument(
        "--gradient-scope",
        choices=("input", "all"),
        default="all",
        help="Differentiate only input, or input plus both initial states",
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--neurons", type=int, default=128)
    parser.add_argument("--capacitance", type=float, default=100.0)
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
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Skip candidate VJPs and qualify forward trajectories only",
    )
    args = parser.parse_args()
    if min(args.steps, args.neurons) < 1:
        parser.error("steps and neurons must be positive")
    if not math.isfinite(args.capacitance) or args.capacitance <= 0:
        parser.error("capacitance must be finite and positive")
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
    args.variants = tuple(dict.fromkeys(args.variants))
    args.backends = tuple(dict.fromkeys(args.backends))
    return args


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path):
    try:
        result = subprocess.run(
            ("git", "-C", str(path), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _git_dirty(path):
    try:
        result = subprocess.run(
            ("git", "-C", str(path), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def _source_identity(path):
    return {"path": str(path), "sha256": _sha256(path)}


def _configure_002hip_compiler(configure, disable_fp_fusion):
    original = tuple(configure.cuda_compiler_options)
    effective = original
    if disable_fp_fusion:
        effective = tuple(value for value in original if value != "--fmad=false") + (
            "--fmad=false",
        )
        configure.cuda_compiler_options = effective
    return original, effective


def _compact_diagnostic(exc):
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    compact = "".join(lines[-8:]).strip()
    if len(compact) > 4000:
        compact = compact[-4000:]
    message = str(exc).strip()
    if len(message) > 1200:
        message = message[-1200:]
    return {
        "exception": type(exc).__name__,
        "message": message,
        "traceback": compact,
    }


def _load_module(path, name):
    declaration = importlib.util.spec_from_file_location(name, path)
    if declaration is None or declaration.loader is None:
        raise ImportError(f"Cannot load Python source from {path}")
    module = importlib.util.module_from_spec(declaration)
    declaration.loader.exec_module(module)
    return module


def load_scalar_cupy_kernel(hip_root, spikingjelly_root):
    """Load the unchanged 002HIP scalar CuPy Izhikevich source.

    This is a benchmark adapter boundary.  It imports only the original
    ``model/neuron_kernel_float.py`` source by path, while resolving its
    SpikingJelly dependency from the caller-selected checkout.  No 002HIP
    package import or compatibility implementation is used.

    Args:
        hip_root: 002HIP checkout containing ``model/neuron_kernel_float.py``.
        spikingjelly_root: Checkout whose root contains ``spikingjelly``.

    Returns:
        Tuple ``(kernel_module, surrogate_module, configure_module)``.

    Raises:
        FileNotFoundError: If either checkout does not contain the required
            source/package.
        ImportError: If Python resolves SpikingJelly outside the requested
            checkout.
    """
    hip_root = Path(hip_root).resolve()
    spikingjelly_root = Path(spikingjelly_root).resolve()
    source = hip_root / "model" / "neuron_kernel_float.py"
    package_root = spikingjelly_root / "spikingjelly"
    if not source.is_file():
        raise FileNotFoundError(source)
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)

    # Keep source import explicit: importing 002HIP.model would pull unrelated
    # optional dependencies and would no longer identify the raw kernel.
    if str(spikingjelly_root) not in sys.path:
        sys.path.insert(0, str(spikingjelly_root))
    import spikingjelly
    from spikingjelly import configure
    from spikingjelly.activation_based import surrogate

    loaded = Path(spikingjelly.__file__).resolve()
    if not loaded.is_relative_to(package_root):
        raise ImportError(f"Loaded SpikingJelly from {loaded}, expected {package_root}")
    configure.save_spike_as_bool_in_neuron_kernel = False
    module = _load_module(source, "btorch_002hip_raw_scalar_cupy")
    return module, surrogate, configure


def _first_mismatch(actual, expected, atol, rtol):
    mismatch = ~actual.isclose(expected, atol=atol, rtol=rtol, equal_nan=False)
    locations = mismatch.nonzero()
    return locations[0].tolist() if locations.numel() else None


def _metric(actual, expected, atol, rtol):
    if actual.shape != expected.shape:
        return {
            "status": "fail",
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "reason": "shape mismatch",
        }
    finite = actual.isfinite() & expected.isfinite()
    max_abs = (
        (actual[finite] - expected[finite]).abs().max().item() if finite.any() else None
    )
    close = bool(
        actual.isfinite().all()
        and expected.isfinite().all()
        and actual.allclose(expected, atol=atol, rtol=rtol)
    )
    return {
        "status": "pass" if close else "fail",
        "max_abs": max_abs,
        "first_mismatch": _first_mismatch(actual, expected, atol, rtol),
        "nonfinite_actual": int((~actual.isfinite()).sum().item()),
        "nonfinite_expected": int((~expected.isfinite()).sum().item()),
    }


def _comparison(actual, expected, names, atol, rtol):
    if len(actual) != len(expected) or len(actual) != len(names):
        return {
            "status": "fail",
            "reason": "result count mismatch",
            "actual_count": len(actual),
            "expected_count": len(expected),
            "name_count": len(names),
            "metrics": {},
        }
    metrics = {
        name: _metric(value, reference, atol, rtol)
        for name, value, reference in zip(names, actual, expected)
    }
    return {
        "status": (
            "pass"
            if len(metrics) == len(names)
            and all(metric["status"] == "pass" for metric in metrics.values())
            else "fail"
        ),
        "metrics": metrics,
    }


def _loss(trajectories):
    return sum(value.sum() for value in trajectories)


def _cotangents(torch, outputs):
    values = []
    for output_index, output in enumerate(outputs):
        flat = torch.arange(output.numel(), device=output.device, dtype=output.dtype)
        pattern = ((flat + 3 * output_index).remainder(23.0) - 11.0) / 11.0
        values.append(pattern.reshape(output.shape))
    return tuple(values)


def _vjp(torch, outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=_cotangents(torch, outputs),
    )


def _make_inputs(torch, steps, neurons, gradient_scope):
    offsets = torch.arange(steps * neurons, device="cuda", dtype=torch.float32)
    x = 200.0 + (offsets.remainder(40.0) + 0.5) / 40.0 * 40.0
    x = x.reshape(steps, neurons).requires_grad_()
    state_gradients = gradient_scope == "all"
    v = torch.full((neurons,), -65.0, device="cuda", requires_grad=state_gradients)
    u = torch.zeros((neurons,), device="cuda", requires_grad=state_gradients)
    return x, v, u


def _gradient_inputs(x, v, u, gradient_scope):
    return (x, v, u) if gradient_scope == "all" else (x,)


def _gradient_names(gradient_scope):
    return (
        ("input", "initial_voltage", "initial_recovery")
        if gradient_scope == "all"
        else ("input",)
    )


def _module_reference(torch, python_module_sequence, spec, parameters, args):
    x, v, u = _make_inputs(torch, args.steps, args.neurons, args.gradient_scope)
    output, final, records = python_module_sequence(
        x, (v, u), parameters, spec, ("v", "u")
    )
    trajectories = (output, *records)
    gradients = _vjp(
        torch,
        trajectories,
        _gradient_inputs(x, v, u, args.gradient_scope),
    )
    return {
        "trajectories": tuple(value.detach() for value in trajectories),
        "final": tuple(value.detach() for value in final),
        "gradients": tuple(value.detach() for value in gradients),
        "spike_count": int(output.detach().sum().item()),
    }


def _check_btorch_backend(
    torch,
    run_multi_step,
    history,
    spec,
    parameters,
    reference,
    backend,
    args,
    check_backward=True,
):
    try:
        x, v, u = _make_inputs(torch, args.steps, args.neurons, args.gradient_scope)
        result = run_multi_step(x, (v, u), parameters, spec, history, backend=backend)
        trajectories = (result[0], result[3], result[4])
        forward = _comparison(
            trajectories,
            reference["trajectories"],
            ("spikes", "voltage", "recovery"),
            args.forward_atol,
            args.forward_rtol,
        )
        continuation = _comparison(
            result[1:3],
            reference["final"],
            ("voltage", "recovery"),
            args.forward_atol,
            args.forward_rtol,
        )
        backward = {"status": "not_run"}
        if check_backward:
            gradients = _vjp(
                torch,
                trajectories,
                _gradient_inputs(x, v, u, args.gradient_scope),
            )
            backward = _comparison(
                gradients,
                reference["gradients"],
                _gradient_names(args.gradient_scope),
                args.gradient_atol,
                args.gradient_rtol,
            )
        spike_count = int(trajectories[0].detach().sum().item())
        spike_status = "pass" if spike_count == reference["spike_count"] else "fail"
        activity_status = (
            "pass" if 0 < reference["spike_count"] < trajectories[0].numel() else "fail"
        )
        return {
            "status": (
                "pass"
                if forward["status"] == "pass"
                and continuation["status"] == "pass"
                and (not check_backward or backward["status"] == "pass")
                and spike_status == "pass"
                and activity_status == "pass"
                else "fail"
            ),
            "spike_count": spike_count,
            "expected_spike_count": reference["spike_count"],
            "spike_count_status": spike_status,
            "nontrivial_spike_activity_status": activity_status,
            "forward": forward,
            "continuation": continuation,
            "backward": backward,
        }
    except Exception as exc:
        return {"status": "error", "diagnostic": _compact_diagnostic(exc)}


def _hip_parameters(torch, variant, x, case):
    if variant == "scalar":
        return case.hip_arguments
    # The original tensor source indexes every parameter with flattened time
    # index t, so full [T, N] storage is required to avoid out-of-bounds reads.
    return tuple(torch.full_like(x, value) for value in case.hip_arguments)


def _check_hip_variant(
    torch, surrogate, module, variant, reference, args, case, check_backward=True
):
    result = {
        "status": "error",
        "parameter_storage": (
            "scalar values"
            if variant == "scalar"
            else "constant [time, neuron] tensors required by source indexing"
        ),
    }
    try:
        x, v, u = _make_inputs(torch, args.steps, args.neurons, args.gradient_scope)
        candidate = module.MultiStepIzhikevichNodePTT.apply(
            x,
            v,
            u,
            *_hip_parameters(torch, variant, x, case),
            False,
            surrogate.Sigmoid(alpha=4.0).cuda_code,
        )
        forward = _comparison(
            candidate,
            reference["trajectories"],
            ("spikes", "voltage", "recovery"),
            args.forward_atol,
            args.forward_rtol,
        )
        continuation = _comparison(
            (candidate[1][-1], candidate[2][-1]),
            reference["final"],
            ("voltage", "recovery"),
            args.forward_atol,
            args.forward_rtol,
        )
        result.update(
            {
                "forward": forward,
                "continuation": continuation,
                "spike_count": int(candidate[0].detach().sum().item()),
                "expected_spike_count": reference["spike_count"],
            }
        )
        result["spike_count_status"] = (
            "pass" if result["spike_count"] == reference["spike_count"] else "fail"
        )
        result["nontrivial_spike_activity_status"] = (
            "pass" if 0 < reference["spike_count"] < candidate[0].numel() else "fail"
        )
    except Exception as exc:
        result["forward"] = {
            "status": "error",
            "diagnostic": _compact_diagnostic(exc),
        }
        result["backward"] = {"status": "not_run"}
        return result

    if not check_backward:
        result["backward"] = {"status": "not_run"}
    else:
        try:
            gradients = _vjp(
                torch,
                candidate,
                _gradient_inputs(x, v, u, args.gradient_scope),
            )
            backward = _comparison(
                gradients,
                reference["gradients"],
                _gradient_names(args.gradient_scope),
                args.gradient_atol,
                args.gradient_rtol,
            )
            result["backward"] = backward
        except Exception as exc:
            result["backward"] = {
                "status": "error",
                "diagnostic": _compact_diagnostic(exc),
            }
    result["status"] = (
        "pass"
        if all(result[name]["status"] == "pass" for name in ("forward", "continuation"))
        and (not check_backward or result["backward"]["status"] == "pass")
        and result["spike_count_status"] == "pass"
        and result["nontrivial_spike_activity_status"] == "pass"
        else "fail"
    )
    return result


def _run(args, report):
    hip_root = args.hip_root.resolve()
    spikingjelly_root = args.spikingjelly_root.resolve()
    scalar_source = hip_root / "model" / "neuron_kernel_float.py"
    tensor_source = hip_root / "model" / "neuron_kernel_tensor.py"
    required_sources = {"scalar": scalar_source, "tensor": tensor_source}
    for variant in args.variants:
        if not required_sources[variant].is_file():
            raise FileNotFoundError(required_sources[variant])
    package_root = spikingjelly_root / "spikingjelly"
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    spikingjelly_files = {
        str(path.relative_to(spikingjelly_root)): _sha256(path)
        for path in (
            package_root / "configure.py",
            package_root / "activation_based" / "cuda_utils.py",
            package_root / "activation_based" / "surrogate.py",
            package_root / "activation_based" / "tensor_cache.py",
        )
    }
    verified_spikingjelly_revision = (
        KNOWN_SPIKINGJELLY_REVISION
        if spikingjelly_files == KNOWN_SPIKINGJELLY_HASHES
        else None
    )

    case = IzhikevichCase(capacitance=args.capacitance)
    report.update(
        {
            "sources": {
                "002hip_root": str(hip_root),
                "002hip_revision": _git_revision(hip_root),
                "kernels": {
                    name: _source_identity(path)
                    for name, path in required_sources.items()
                },
                "spikingjelly_root": str(spikingjelly_root),
                "spikingjelly_revision": _git_revision(spikingjelly_root),
                "spikingjelly_verified_revision": verified_spikingjelly_revision,
                "spikingjelly_revision_verification": (
                    "SHA-256 match for the four imported compatibility files"
                ),
                "spikingjelly_files": spikingjelly_files,
            },
            "contract": {
                "dtype": "float32",
                "shape_TN": [args.steps, args.neurons],
                "input": "200 + ((flat_index mod 40) + 0.5)",
                "initial_voltage": -65.0,
                "initial_recovery": 0.0,
                "dt": 1.0,
                "reset": "hard",
                "forward_equations": (
                    "Euler voltage update, recovery update from charged voltage, "
                    "spike at v_peak, hard voltage reset and recovery increment d"
                ),
                "parameters": {
                    name: value
                    for name, value in zip(
                        (
                            "v_threshold",
                            "v_reset",
                            "c_m",
                            "k",
                            "v_rest",
                            "v_peak",
                            "a",
                            "b",
                            "d",
                        ),
                        case.btorch_parameters,
                    )
                },
                "vjp": (
                    "Deterministic unit-scale cotangents for spikes, voltage "
                    "and recovery; no reduction by T or N"
                ),
                "gradient_scope": args.gradient_scope,
                "backward_check": "not_run" if args.forward_only else "enabled",
                "surrogate": {
                    "002hip": "Sigmoid alpha=4 on physical h-v_peak",
                    "btorch": ("Sigmoid alpha=100 on (h-v_peak)/(v_threshold-v_reset)"),
                    "note": (
                        "Forward equations are surrogate-independent; VJPs "
                        "test derivative convention compatibility."
                    ),
                },
                "002hip_autograd_trigger": (
                    "Original forward saves backward context when x_seq or "
                    "v_init requires grad; u_init alone is not considered."
                ),
                "tolerance": {
                    "forward": {
                        "atol": args.forward_atol,
                        "rtol": args.forward_rtol,
                    },
                    "gradient": {
                        "atol": args.gradient_atol,
                        "rtol": args.gradient_rtol,
                    },
                },
            },
        }
    )

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(spikingjelly_root))

    import cupy
    import spikingjelly
    import torch
    from spikingjelly import configure
    from spikingjelly.activation_based import surrogate

    loaded_spikingjelly = Path(spikingjelly.__file__).resolve()
    if not loaded_spikingjelly.is_relative_to(package_root):
        raise ImportError(
            f"Loaded SpikingJelly from {loaded_spikingjelly}, expected {package_root}"
        )

    configure.save_spike_as_bool_in_neuron_kernel = False
    original_compiler_options, effective_compiler_options = _configure_002hip_compiler(
        configure, args.disable_fp_fusion
    )
    report["contract"]["floating_point_contraction"] = {
        "disable_requested": args.disable_fp_fusion,
        "btorch_enable_fp_fusion": not args.disable_fp_fusion,
        "002hip_original_compiler_options": list(original_compiler_options),
        "002hip_effective_compiler_options": list(effective_compiler_options),
        "scope": (
            "Controls compatible floating-point contraction only; it is not a "
            "blanket fast-math switch"
        ),
    }

    from btorch.models.kernels import (
        CellSpec,
        HistoryPlan,
        KernelOptions,
        run_multi_step,
    )
    from btorch.models.kernels.golden import python_module_sequence

    cuda_available = torch.cuda.is_available()
    report["environment"] = {
        "torch": torch.__version__,
        "cupy": cupy.__version__,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name() if cuda_available else None,
        "spikingjelly_module": str(loaded_spikingjelly),
    }
    if not cuda_available:
        raise RuntimeError("CUDA is unavailable; this probe requires a host GPU")
    parameters = tuple(
        torch.tensor(value, device="cuda", dtype=torch.float32)
        for value in case.btorch_parameters
    )
    # 002HIP applies Sigmoid(alpha=4) to physical voltage h-v_peak. The probe
    # follows the requested width conversion for btorch's normalized voltage:
    # alpha=4 * (v_threshold-v_reset)=100. Forward dynamics do not depend on
    # this surrogate choice; backward results determine whether the derivative
    # conventions also agree.
    spec = CellSpec("izhikevich", hard_reset=True, alpha=100.0)
    history = HistoryPlan(requested=("v", "u"))
    reference = _module_reference(torch, python_module_sequence, spec, parameters, args)

    report["btorch"] = {
        "oracle": "python_module_sequence",
        "backends": {
            backend: _check_btorch_backend(
                torch,
                run_multi_step,
                history,
                spec,
                parameters,
                reference,
                KernelOptions(
                    neuron=backend,
                    enable_fp_fusion=not args.disable_fp_fusion,
                ),
                args,
                check_backward=not args.forward_only,
            )
            for backend in args.backends
        },
    }
    report["btorch"]["status"] = (
        "pass"
        if all(
            value["status"] == "pass" for value in report["btorch"]["backends"].values()
        )
        else "fail"
    )

    report["002hip"] = {}
    for variant in args.variants:
        try:
            module = _load_module(
                required_sources[variant], f"btorch_002hip_{variant}_probe"
            )
            report["002hip"][variant] = _check_hip_variant(
                torch,
                surrogate,
                module,
                variant,
                reference,
                args,
                case,
                check_backward=not args.forward_only,
            )
        except Exception as exc:
            report["002hip"][variant] = {
                "status": "error",
                "diagnostic": _compact_diagnostic(exc),
            }
    report["002hip"]["status"] = (
        "pass"
        if all(
            report["002hip"][variant]["status"] == "pass" for variant in args.variants
        )
        else "fail"
    )
    report["status"] = (
        "pass"
        if report["btorch"]["status"] == "pass" and report["002hip"]["status"] == "pass"
        else "fail"
    )


def main():
    args = _parse_args()
    report = {
        "probe": "002HIP Izhikevich correctness",
        "status": "error",
        "requested_variants": list(args.variants),
        "requested_btorch_backends": list(args.backends),
        "gradient_scope": args.gradient_scope,
    }
    try:
        _run(args, report)
    except Exception as exc:
        report["diagnostic"] = _compact_diagnostic(exc)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
