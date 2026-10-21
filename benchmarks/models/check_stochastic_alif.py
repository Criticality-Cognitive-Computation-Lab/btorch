"""Validate stochastic ALIF and ELIF Modules at three scientific scales.

This harness reuses the established ALIF/ELIF construction, stimulus,
tracing, and plotting helpers. The actual btorch Modules remain the only
simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.models.check_alif_elif import (  # noqa: E402
    DT,
    DTYPE,
    MODELS,
    STATE_NAMES,
    _build_neuron,
    _build_recurrent,
    _network_stimulus,
    _plot_activity,
    _run_neuron,
    _run_recurrent,
    _single_stimulus,
)
from btorch.models import environ  # noqa: E402
from btorch.models.functional import init_net_state  # noqa: E402
from btorch.models.surrogate import PoissonRandomSpike  # noqa: E402


SEED = 4103
POISSON = {
    "tau": 5.0,
    "rho": 3.0,
    "leak": 0.2,
    "k": 1.1,
    "damping_factor": 0.5,
}
SCALES = (("single", 1), ("pair", 2), ("network", 12))
ELIF_V_T = -48.0


def _install_poisson(module) -> None:
    neuron = module if hasattr(module, "surrogate_function") else module.neuron
    neuron.surrogate_function = PoissonRandomSpike(spiking=True, **POISSON)


def _build_case(model: str, scale: str, neurons: int, *, multistep=False):
    if scale == "single":
        # Hard reset bounds ELIF's exponential upswing after every sampled event.
        module = _build_neuron(
            model,
            neurons,
            hard_reset=model == "ELIF",
            step_mode="m" if multistep else "s",
        )
        init_net_state(module, batch_size=1, dtype=DTYPE)
    else:
        module = _build_recurrent(model, neurons, scale)
    if model == "ELIF":
        neuron = module if scale == "single" else module.neuron
        with torch.no_grad():
            neuron.v_T.fill_(ELIF_V_T)
    _install_poisson(module)
    return module


def _stimulus(model: str, scale: str, neurons: int) -> torch.Tensor:
    if scale == "single":
        return _single_stimulus(model)
    return _network_stimulus(model, neurons, scale)


def _snapshot(module, scale: str) -> dict[str, torch.Tensor]:
    if scale == "single":
        return {
            name: getattr(module, name).detach().clone()
            for name in ("v", "v_pre_spike", "g_k", "refractory")
        }
    return {
        name: getattr(
            module.neuron if name.startswith("neuron.") else module.synapse,
            name.split(".")[1],
        )
        .detach()
        .clone()
        for name in STATE_NAMES
    }


def _continuation(
    model: str, scale: str, neurons: int, stimulus: torch.Tensor, seed: int
) -> dict[str, Any]:
    whole = _build_case(model, scale, neurons, multistep=True)
    chunked = _build_case(model, scale, neurons, multistep=True)
    split = stimulus.shape[0] // 2 + 3

    with torch.random.fork_rng(devices=[]), torch.no_grad(), environ.context(dt=DT):
        torch.random.default_generator.manual_seed(seed)
        initial_rng = torch.get_rng_state().clone()
        if scale == "single":
            whole_spikes = whole(stimulus)
            whole_records: dict[str, torch.Tensor] = {}
        else:
            whole_spikes, whole_records = whole(stimulus)
        whole_rng = torch.get_rng_state().clone()

        torch.set_rng_state(initial_rng)
        if scale == "single":
            chunked_spikes = torch.cat(
                (chunked(stimulus[:split]), chunked(stimulus[split:]))
            )
            chunked_records: dict[str, torch.Tensor] = {}
        else:
            left_spikes, left_records = chunked(stimulus[:split])
            right_spikes, right_records = chunked(stimulus[split:])
            chunked_spikes = torch.cat((left_spikes, right_spikes))
            chunked_records = {
                name: torch.cat((left_records[name], right_records[name]))
                for name in STATE_NAMES
            }
        chunked_rng = torch.get_rng_state().clone()

    whole_state = _snapshot(whole, scale)
    chunked_state = _snapshot(chunked, scale)
    spike_equal = torch.equal(whole_spikes, chunked_spikes)
    records_equal = all(
        torch.equal(whole_records[name], chunked_records[name])
        for name in whole_records
    )
    state_equal = all(
        torch.equal(whole_state[name], chunked_state[name]) for name in whole_state
    )
    rng_equal = torch.equal(whole_rng, chunked_rng)
    return {
        "name": f"{model.lower()}.{scale}.whole_chunk_rng_state",
        "status": "PASS"
        if spike_equal and records_equal and state_equal and rng_equal
        else "FAIL",
        "split": split,
        "spikes_equal": spike_equal,
        "records_equal": records_equal,
        "final_state_equal": state_equal,
        "final_rng_equal": rng_equal,
    }


def _activity(
    model: str, scale: str, traces: dict[str, torch.Tensor]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spikes = traces["spike"]
    finite = all(bool(torch.isfinite(value).all()) for value in traces.values())
    if scale == "single":
        baseline_end, driven = 60, slice(80, 220)
        state_keys = ("v", "v_pre", "g_k", "refractory")
    else:
        baseline_end = 20
        driven = slice(20, 90 if scale == "pair" else 120)
        state_keys = STATE_NAMES
    total = int(spikes.sum())
    mixed = 0 < total < spikes.numel()
    checks = [
        {
            "name": f"{model.lower()}.{scale}.finite",
            "status": "PASS" if finite else "FAIL",
        },
        {
            "name": f"{model.lower()}.{scale}.mixed_activity",
            "status": "PASS" if mixed else "FAIL",
            "spike_count": total,
            "samples": spikes.numel(),
        },
    ]
    if scale != "single":
        psc_active = bool(traces["synapse.psc"].abs().max() > 0)
        checks.append(
            {
                "name": f"{model.lower()}.{scale}.psc_activity",
                "status": "PASS" if psc_active else "FAIL",
                "max_abs_psc": float(traces["synapse.psc"].abs().max()),
                "max_abs_h": float(traces["synapse.h"].abs().max()),
            }
        )
    metrics = {
        "model": model,
        "scale": scale,
        "baseline_spikes": int(spikes[:baseline_end].sum()),
        "driven_spikes": int(spikes[driven].sum()),
        "total_spikes": total,
        "active_neurons": int((spikes.sum(dim=(0, 1)) > 0).sum()),
        "state_ranges": {
            key: [float(traces[key].min()), float(traces[key].max())]
            for key in state_keys
        },
    }
    return checks, metrics


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stochastic ALIF / ELIF validation",
        "",
        f"Overall status: **{summary['status']}**",
        "",
        "The actual CPU FP32 btorch Modules are the sole simulator. The harness",
        "reuses the existing ALIF/ELIF factories, stimuli, tracing, and plotting.",
        "ELIF uses hard reset to keep the exponential trajectory finite.",
        f"Its validation envelope places exponential onset at `v_T={ELIF_V_T} mV`.",
        "",
        "`PoissonRandomSpike` parameters: "
        + ", ".join(f"`{key}={value}`" for key, value in POISSON.items())
        + ".",
        "",
        "Baseline spikes are reported as stochastic activity, not judged against",
        "the deterministic no-input silence contract. No rate equation or second",
        "statistical simulator is used.",
        "",
        "## Checks",
        "",
        "| Status | Check | Details |",
        "|---|---|---|",
    ]
    for check in summary["checks"]:
        details = {k: v for k, v in check.items() if k not in ("name", "status")}
        lines.append(
            f"| {check['status']} | `{check['name']}` | "
            f"`{json.dumps(details, sort_keys=True)}` |"
        )
    lines.extend(("", "## Activity", ""))
    for metric in summary["metrics"]:
        lines.append(
            f"- `{metric['model']}.{metric['scale']}`: baseline "
            f"{metric['baseline_spikes']}, driven {metric['driven_spikes']}, "
            f"total {metric['total_spikes']}, active neurons "
            f"{metric['active_neurons']}."
        )
    lines.extend(("", "## Artifacts", ""))
    lines.extend(f"- [{name}]({name})" for name in summary["artifacts"])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("run/parity/stochastic-alif-validation")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    traces: dict[str, dict[str, torch.Tensor]] = {}
    checks: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    artifacts: list[str] = []
    case_index = 0
    for model in MODELS:
        for scale, neurons in SCALES:
            stimulus = _stimulus(model, scale, neurons)
            module = _build_case(model, scale, neurons)
            seed = SEED + case_index * 101
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                values = (
                    _run_neuron(module, stimulus)
                    if scale == "single"
                    else _run_recurrent(module, stimulus)
                )
            key = f"{model.lower()}.{scale}"
            traces[key] = values
            activity_checks, activity_metrics = _activity(model, scale, values)
            checks.extend(activity_checks)
            checks.append(_continuation(model, scale, neurons, stimulus, seed))
            metrics.append(activity_metrics)
            figure = _plot_activity(
                args.output, key.replace(".", "-"), model, scale, values
            )
            artifacts.append(figure.name)
            case_index += 1

    trace_path = args.output / "traces.pt"
    torch.save(traces, trace_path)
    artifacts.append(trace_path.name)
    status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    summary = {
        "status": status,
        "command": f"python {Path(__file__)} --output {args.output}",
        "environment": {
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": str(DTYPE),
            "dt_ms": DT,
        },
        "poisson": POISSON,
        "elif_envelope": {"hard_reset": True, "v_T_mV": ELIF_V_T},
        "checks": checks,
        "metrics": metrics,
        "artifacts": [*artifacts, "summary.json", "report.md"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(args.output / "report.md", summary)
    print(json.dumps({"status": status, "output": str(args.output)}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
