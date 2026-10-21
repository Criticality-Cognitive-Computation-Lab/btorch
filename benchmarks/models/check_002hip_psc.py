"""Qualify unchanged 002HIP PSC sources against btorch model contracts."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import check_002hip as common


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hip-root", type=Path, required=True)
    parser.add_argument("--spikingjelly-root", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("triton", "tilelang", "cupy"),
        default=("triton", "tilelang", "cupy"),
    )
    parser.add_argument("--atol", type=float, default=5e-6)
    parser.add_argument("--rtol", type=float, default=5e-5)
    args = parser.parse_args()
    if min(args.atol, args.rtol) < 0:
        parser.error("tolerances must be non-negative")
    args.backends = tuple(dict.fromkeys(args.backends))
    return args


def _case_input(torch, impulse_step):
    value = torch.zeros(8, 2, device="cuda", dtype=torch.float32)
    value[impulse_step, 0] = 1.0
    return value


def _first_nonzero(torch, value):
    locations = torch.nonzero(value != 0)
    return locations[0].tolist() if locations.numel() else None


def _compare(common_module, actual, expected, names, args):
    return common_module._comparison(actual, expected, names, args.atol, args.rtol)


def _run(args, report):
    hip_root = args.hip_root.resolve()
    spikingjelly_root = args.spikingjelly_root.resolve()
    source = hip_root / "model" / "connection_kernel.py"
    package_root = spikingjelly_root / "spikingjelly"
    if not source.is_file():
        raise FileNotFoundError(source)
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(spikingjelly_root))
    torch = importlib.import_module("torch")
    spikingjelly = importlib.import_module("spikingjelly")
    configure = importlib.import_module("spikingjelly.configure")
    loaded = Path(spikingjelly.__file__).resolve()
    if not loaded.is_relative_to(package_root):
        raise ImportError(f"Loaded SpikingJelly from {loaded}, expected {package_root}")

    report["sources"] = {
        "002hip_root": str(hip_root),
        "002hip_revision": common._git_revision(hip_root),
        "connection_kernel": common._source_identity(source),
        "spikingjelly_root": str(spikingjelly_root),
        "spikingjelly_module": str(loaded),
        "spikingjelly_compiler_options": list(configure.cuda_compiler_options),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this probe requires a host GPU")

    from btorch.models.kernels import CellSpec, HistoryPlan, run_multi_step
    from btorch.models.kernels.golden import python_module_sequence

    configure.save_spike_as_bool_in_neuron_kernel = False
    module = common._load_module(source, "btorch_002hip_psc_probe")
    spec = CellSpec("alpha_billeh", dt=1.0)
    history = HistoryPlan(requested=("psc", "h"))
    states = (
        torch.zeros(2, device="cuda"),
        torch.zeros(2, device="cuda"),
    )
    tau = torch.tensor(5.5, device="cuda")
    parameters = (tau, torch.exp(-1.0 / tau))
    weight = torch.eye(2, device="cuda")
    cases = {}
    for impulse_step in (0, 3):
        x = _case_input(torch, impulse_step)
        output, final, records = python_module_sequence(
            x, states, parameters, spec, ("psc", "h")
        )
        expected = (output, *final, *records)
        original_raw = module.MultiStepCurrentBasedAlphaPSCPTT.apply(
            x, *states, 5.5, weight
        )
        original = (
            original_raw[0],
            original_raw[1][-1],
            original_raw[2][-1],
            original_raw[1],
            original_raw[2],
        )
        names = ("current", "final_current", "final_h", "current_history", "h")
        backend_results = {}
        for backend in args.backends:
            actual = run_multi_step(
                x, states, parameters, spec, history, backend=backend
            )
            backend_results[backend] = _compare(common, actual, expected, names, args)
        cases[f"impulse_t{impulse_step}"] = {
            "oracle_first_nonzero_current": _first_nonzero(torch, output),
            "original_first_nonzero_current": _first_nonzero(torch, original_raw[0]),
            "oracle_max_abs_current": output.abs().max().item(),
            "original_max_abs_current": original_raw[0].abs().max().item(),
            "original": _compare(common, original, expected, names, args),
            "btorch": backend_results,
        }

    btorch_status = (
        "pass"
        if all(
            result["status"] == "pass"
            for case in cases.values()
            for result in case["btorch"].values()
        )
        else "fail"
    )
    original_status = (
        "pass"
        if all(case["original"]["status"] == "pass" for case in cases.values())
        else "fail"
    )
    report.update(
        {
            "environment": {
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
            },
            "contract": {
                "dtype": "float32",
                "shape_TN": [8, 2],
                "tau_syn": 5.5,
                "weight": "2x2 identity",
                "initial_states": "zero current and rise state",
                "impulses": [0, 3],
                "btorch_model": "alpha_billeh",
                "original_source_observation": (
                    "The dense loop reads z_seq[i] without the timestep offset "
                    "and emits current before injecting the new event into h."
                ),
            },
            "current_based_alpha": {
                "status": original_status,
                "btorch_status": btorch_status,
                "cases": cases,
            },
            "conductance": {
                "status": "unsupported",
                "reason": (
                    "002HIP uses two Euler states g/h plus F and expects current "
                    "as (g + F) * (v - E); btorch ConductancePSC uses a single "
                    "exponential conductance state and current g * (E - v)."
                ),
            },
        }
    )
    report["status"] = (
        "pass" if btorch_status == "pass" and original_status == "pass" else "fail"
    )


def main():
    args = _parse_args()
    report = {"probe": "002HIP PSC qualification", "status": "error"}
    try:
        _run(args, report)
    except Exception as exc:
        report["diagnostic"] = common._compact_diagnostic(exc)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
