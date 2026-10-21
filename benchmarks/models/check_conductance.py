"""Validate ConductancePSC and the public Torch recurrent COBA schedule.

The connected cases execute :class:`ConductanceRecurrentNN` with its explicit
previous-post-reset-voltage ordering. They provide reviewable scientific traces
for the Torch schedule and do not qualify native recurrent COBA execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import torch


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402, I001

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from btorch.models import environ  # noqa: E402
from btorch.models.functional import init_net_state  # noqa: E402
from btorch.models.neurons.lif import LIF  # noqa: E402
from btorch.models.rnn import ConductanceRecurrentNN  # noqa: E402
from btorch.models.synapse import ConductancePSC  # noqa: E402
from btorch.utils.file import save_fig  # noqa: E402


DT = 1.0
DTYPE = torch.float32


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _linear(weight: torch.Tensor, *, trainable: bool = False) -> torch.nn.Linear:
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=DTYPE)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.weight.requires_grad_(trainable)
    return layer


def _synapse(
    weight: torch.Tensor,
    reversal: float | torch.Tensor,
    *,
    trainable: bool = False,
) -> ConductancePSC:
    return ConductancePSC(
        weight.shape[0],
        tau_syn=6.0,
        linear=_linear(weight, trainable=trainable),
        reversal_potential=reversal,
    )


def _neuron(neurons: int) -> LIF:
    return LIF(
        neurons,
        v_threshold=-50.0,
        v_reset=-65.0,
        c_m=2.0,
        tau=20.0,
        tau_ref=2.0,
        hard_reset=True,
        pre_spike_v=True,
        dtype=DTYPE,
    )


def _network(
    weight: torch.Tensor, reversal: float | torch.Tensor
) -> ConductanceRecurrentNN:
    network = ConductanceRecurrentNN(
        _neuron(weight.shape[0]),
        _synapse(weight, reversal),
        backend="torch",
        unroll=8,
        update_state_names=("neuron.v", "neuron.v_pre_spike", "synapse.psc"),
    )
    init_net_state(network, batch_size=1, dtype=DTYPE)
    return network


def _run_psc(
    synapse: ConductancePSC,
    events: torch.Tensor,
    voltage: torch.Tensor,
    *,
    initial_conductance: float = 0.0,
) -> dict[str, torch.Tensor]:
    init_net_state(synapse, batch_size=events.shape[1], dtype=DTYPE)
    synapse.psc = torch.full_like(synapse.psc, initial_conductance)
    conductance = []
    current = []
    with torch.no_grad(), environ.context(dt=DT):
        for event, voltage_t in zip(events, voltage, strict=True):
            current.append(synapse.single_step_forward(event, voltage_t).clone())
            conductance.append(synapse.psc.clone())
    return {
        "input": events.clone(),
        "spike": events.clone(),
        "voltage_observed": voltage.clone(),
        "voltage_post": voltage.clone(),
        "voltage_charged": voltage.clone(),
        "conductance_updated": torch.stack(conductance),
        "current_used": torch.stack(current),
        "reversal_potential": synapse.E.expand_as(voltage).clone(),
    }


def _network_trace(
    network: ConductanceRecurrentNN, stimulus: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Derive schedule diagnostics from initial state and public records."""
    initial_voltage = network.neuron.v
    initial_conductance = network.synapse.psc
    spike, records = network(stimulus)
    voltage_post = records["neuron.v"]
    conductance_updated = records["synapse.psc"]
    voltage_observed = torch.cat((initial_voltage.unsqueeze(0), voltage_post[:-1]))
    conductance_used = torch.cat(
        (initial_conductance.unsqueeze(0), conductance_updated[:-1])
    )
    current_used = conductance_used * (network.synapse.E - voltage_observed)
    return {
        "input": stimulus.clone(),
        "spike": spike,
        "voltage_observed": voltage_observed,
        "voltage_post": voltage_post,
        "voltage_charged": records["neuron.v_pre_spike"],
        "conductance_used": conductance_used,
        "conductance_updated": conductance_updated,
        "current_used": current_used,
        "reversal_potential": network.synapse.E.expand_as(current_used),
    }


def _run_network(
    network: ConductanceRecurrentNN, stimulus: torch.Tensor
) -> dict[str, torch.Tensor]:
    with torch.no_grad(), environ.context(dt=DT):
        return _network_trace(network, stimulus)


def _add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, **details: Any
) -> None:
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", **details})


def _standalone_case(checks: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    steps = 140
    events = torch.zeros(steps, 1, 1, dtype=DTYPE)
    events[20] = 1.0
    events[50:70] = 0.25
    voltage = torch.full_like(events, -60.0)
    voltage[85:105] = -40.0
    trace = _run_psc(_synapse(torch.ones(1, 1), 0.0), events, voltage)
    g = trace["conductance_updated"][:, 0, 0]
    current = trace["current_used"][:, 0, 0]
    _add_check(
        checks,
        "conductance.single.finite",
        all(bool(torch.isfinite(value).all()) for value in trace.values()),
    )
    _add_check(
        checks,
        "conductance.single.delivery_step",
        float(g[20]) > 0 and float(current[20]) > 0,
        conductance=float(g[20]),
        current=float(current[20]),
    )
    _add_check(
        checks,
        "conductance.single.step_off_decay",
        float(g[-1]) < float(g[70]),
        conductance_at_off=float(g[70]),
        final_conductance=float(g[-1]),
    )
    _add_check(
        checks,
        "conductance.single.driving_force",
        float(current[90] / g[90]) < float(current[80] / g[80]),
        current_per_g_at_minus_60=float(current[80] / g[80]),
        current_per_g_at_minus_40=float(current[90] / g[90]),
    )
    return trace


def _reversal_checks(checks: list[dict[str, Any]]) -> None:
    event = torch.ones(1, 1, dtype=DTYPE)
    voltage = torch.full((1, 1), -60.0, dtype=DTYPE)
    exc = _synapse(torch.ones(1, 1), 0.0)
    inh = _synapse(torch.ones(1, 1), -80.0)
    init_net_state(exc, batch_size=1, dtype=DTYPE)
    init_net_state(inh, batch_size=1, dtype=DTYPE)
    with torch.no_grad(), environ.context(dt=DT):
        exc_current = exc(event, voltage)
        inh_current = inh(event, voltage)
    _add_check(
        checks,
        "conductance.reversal.direction",
        float(exc_current) > 0 and float(inh_current) < 0,
        excitatory_current=float(exc_current),
        inhibitory_current=float(inh_current),
    )
    equal = _synapse(torch.ones(1, 1), -60.0)
    init_net_state(equal, batch_size=1, dtype=DTYPE)
    with torch.no_grad(), environ.context(dt=DT):
        equal_current = equal(event, voltage)
    _add_check(
        checks,
        "conductance.reversal.zero_at_E",
        float(equal_current) == 0.0,
        current=float(equal_current),
    )


def _continuation_check(checks: list[dict[str, Any]]) -> None:
    events = torch.zeros(41, 1, 2, dtype=DTYPE)
    events[4, 0, 0] = 1.0
    events[19:24, 0, 1] = 0.3
    voltage = torch.full_like(events, -55.0)
    weight = torch.tensor([[0.8, 0.2], [0.1, 0.6]], dtype=DTYPE)

    def run_sequence(parts: tuple[slice, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        synapse = _synapse(weight, torch.tensor([0.0, -80.0], dtype=DTYPE))
        init_net_state(synapse, batch_size=1, dtype=DTYPE)
        synapse.psc = torch.full_like(synapse.psc, 0.7)
        with torch.no_grad(), environ.context(dt=DT):
            outputs = [
                synapse.multi_step_forward(events[part], v_seq=voltage[part])
                for part in parts
            ]
        return torch.cat(outputs), synapse.psc.clone()

    def run_single_step() -> tuple[torch.Tensor, torch.Tensor]:
        synapse = _synapse(weight, torch.tensor([0.0, -80.0], dtype=DTYPE))
        init_net_state(synapse, batch_size=1, dtype=DTYPE)
        synapse.psc = torch.full_like(synapse.psc, 0.7)
        with torch.no_grad(), environ.context(dt=DT):
            outputs = [
                synapse.single_step_forward(event, voltage_t)
                for event, voltage_t in zip(events, voltage, strict=True)
            ]
        return torch.stack(outputs), synapse.psc.clone()

    full_output, full_state = run_sequence((slice(None),))
    chunked_output, chunked_state = run_sequence((slice(None, 17), slice(17, None)))
    oracle_output, oracle_state = run_single_step()
    finite = bool(
        torch.isfinite(full_output).all()
        and torch.isfinite(chunked_output).all()
        and torch.isfinite(oracle_output).all()
        and torch.isfinite(full_state).all()
        and torch.isfinite(chunked_state).all()
        and torch.isfinite(oracle_state).all()
    )
    maximum = max(
        float((full_output - chunked_output).abs().max()),
        float((full_output - oracle_output).abs().max()),
        float((full_state - chunked_state).abs().max()),
        float((full_state - oracle_state).abs().max()),
    )
    _add_check(
        checks,
        "conductance.voltage_sequence_continuation.nonzero_initial",
        finite and maximum == 0.0,
        max_abs=maximum,
        split=17,
        initial_conductance=0.7,
    )


def _pair_case(checks: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    weight = torch.tensor([[0.0, 0.0], [0.8, 0.0]], dtype=DTYPE)
    stimulus = torch.zeros(150, 1, 2, dtype=DTYPE)
    stimulus[20:75, 0, 0] = 4.0
    trace = _run_network(_network(weight, 0.0), stimulus)
    pre = trace["spike"][:, 0, 0].nonzero().flatten()
    post = trace["spike"][:, 0, 1].nonzero().flatten()
    _add_check(
        checks,
        "conductance.pair.finite",
        all(bool(torch.isfinite(value).all()) for value in trace.values()),
    )
    propagated = len(pre) > 0 and len(post) > 0 and int(post[0]) > int(pre[0])
    _add_check(
        checks,
        "conductance.pair.recurrent_coupling",
        propagated,
        first_pre=int(pre[0]) if len(pre) else None,
        first_post=int(post[0]) if len(post) else None,
    )
    delivered = trace["conductance_updated"][:, 0, 1]
    _add_check(
        checks,
        "conductance.pair.delivery_after_emission",
        len(pre) > 0
        and float(delivered[int(pre[0])]) > 0
        and float(trace["current_used"][int(pre[0]), 0, 1]) == 0.0,
        conductance_on_emission=float(delivered[int(pre[0])]) if len(pre) else None,
        current_on_emission=(
            float(trace["current_used"][int(pre[0]), 0, 1]) if len(pre) else None
        ),
    )
    return trace


def _network_case(checks: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    neurons = 6
    weight = torch.zeros(neurons, neurons, dtype=DTYPE)
    for source in range(neurons):
        weight[(source + 1) % neurons, source] = 0.45
        weight[(source + 2) % neurons, source] = 0.2
    reversal = torch.tensor([0.0, 0.0, 0.0, -80.0, -80.0, -80.0], dtype=DTYPE)
    stimulus = torch.zeros(180, 1, neurons, dtype=DTYPE)
    stimulus[20:100, 0, :2] = 4.0
    trace = _run_network(_network(weight, reversal), stimulus)
    spike = trace["spike"][:, 0]
    active = int((spike.sum(0) > 0).sum())
    _add_check(
        checks,
        "conductance.network.finite",
        all(bool(torch.isfinite(value).all()) for value in trace.values()),
    )
    _add_check(
        checks,
        "conductance.network.mixed_activity",
        2 < active < neurons and 0 < int(spike.sum()) < spike.numel(),
        active_neurons=active,
        spikes=int(spike.sum()),
    )
    current = trace["current_used"]
    _add_check(
        checks,
        "conductance.network.mixed_reversal_direction",
        float(current.max()) > 0 and float(current.min()) < 0,
        minimum_current=float(current.min()),
        maximum_current=float(current.max()),
    )
    late_g = trace["conductance_updated"][-1].abs().mean()
    off_g = trace["conductance_updated"][100].abs().mean()
    _add_check(
        checks,
        "conductance.network.step_off_decay",
        float(late_g) < float(off_g),
        conductance_at_off=float(off_g),
        final_conductance=float(late_g),
    )
    return trace


def _gradient_checks(checks: list[dict[str, Any]]) -> dict[str, float]:
    weight = torch.tensor([[0.7, 0.2], [0.1, 0.5]], dtype=DTYPE)
    synapse = _synapse(weight, torch.tensor([0.0, -80.0]), trainable=True)
    init_net_state(synapse, batch_size=1, dtype=DTYPE)
    events = torch.full((12, 1, 2), 0.15, dtype=DTYPE, requires_grad=True)
    voltage = torch.full((12, 1, 2), -55.0, dtype=DTYPE, requires_grad=True)
    loss = torch.zeros((), dtype=DTYPE)
    with environ.context(dt=DT):
        for event, voltage_t in zip(events, voltage, strict=True):
            current = synapse.single_step_forward(event, voltage_t)
            loss = loss + current.square().mean()
    gradients = torch.autograd.grad(
        loss, (events, voltage, synapse.linear.weight), allow_unused=True
    )
    valid = all(
        value is not None
        and bool(torch.isfinite(value).all())
        and bool(value.abs().sum() > 0)
        for value in gradients
    )
    norms = [None if value is None else float(value.norm()) for value in gradients]
    _add_check(
        checks,
        "conductance.single.autograd",
        valid,
        input_grad_norm=norms[0],
        voltage_grad_norm=norms[1],
        linear_grad_norm=norms[2],
    )

    recurrent = _network(weight, torch.tensor([0.0, -80.0], dtype=DTYPE))
    recurrent.synapse.linear.weight.requires_grad_()
    drive = torch.zeros(30, 1, 2, dtype=DTYPE)
    drive[4:20, 0, 0] = 4.0
    drive.requires_grad_()
    with environ.context(dt=DT):
        trace = _network_trace(recurrent, drive)
        network_loss = trace["spike"].sum()
        network_loss = network_loss + 0.001 * trace["voltage_post"].square().mean()
        network_loss = network_loss + 0.001 * trace["current_used"].square().mean()
    drive_grad, weight_grad = torch.autograd.grad(
        network_loss, (drive, recurrent.synapse.linear.weight)
    )
    recurrent_valid = bool(
        torch.isfinite(drive_grad).all()
        and torch.isfinite(weight_grad).all()
        and drive_grad.abs().sum() > 0
        and weight_grad.abs().sum() > 0
    )
    _add_check(
        checks,
        "conductance.recurrent.autograd",
        recurrent_valid,
        input_grad_norm=float(drive_grad.norm()),
        linear_grad_norm=float(weight_grad.norm()),
    )
    return {
        "psc_input_grad_norm": norms[0] or 0.0,
        "psc_voltage_grad_norm": norms[1] or 0.0,
        "psc_linear_grad_norm": norms[2] or 0.0,
        "network_input_grad_norm": float(drive_grad.norm()),
        "network_linear_grad_norm": float(weight_grad.norm()),
    }


def _plot_activity(
    output: Path, name: str, title: str, trace: dict[str, torch.Tensor]
) -> Path:
    time = torch.arange(trace["input"].shape[0]) * DT
    fig, axes = plt.subplots(
        7, 1, figsize=(11, 15), sharex=True, constrained_layout=True
    )
    axes[0].plot(time, trace["input"][:, 0])
    axes[0].set_ylabel("external input")
    coordinates = trace["spike"][:, 0].nonzero()
    if coordinates.numel():
        axes[1].scatter(time[coordinates[:, 0]], coordinates[:, 1], marker="|", s=24)
    axes[1].set_ylabel("spikes/events")
    # Observation times share a neuron color; the legend describes the three
    # line styles once, rather than repeating entries for every neuron.
    for neuron in range(trace["voltage_post"].shape[-1]):
        (line,) = axes[2].plot(
            time,
            trace["voltage_post"][:, 0, neuron],
            label="post-reset" if neuron == 0 else None,
        )
        axes[2].plot(
            time,
            trace["voltage_observed"][:, 0, neuron],
            linestyle=":",
            color=line.get_color(),
            label="observed" if neuron == 0 else None,
        )
        axes[2].plot(
            time,
            trace["voltage_charged"][:, 0, neuron],
            linestyle="--",
            alpha=0.7,
            color=line.get_color(),
            label="charged" if neuron == 0 else None,
        )
    axes[2].set_ylabel("voltage (mV)")
    axes[2].legend(loc="upper right", fontsize="small")
    for neuron in range(trace["conductance_updated"].shape[-1]):
        (line,) = axes[3].plot(
            time,
            trace["conductance_updated"][:, 0, neuron],
            label="after spike update" if neuron == 0 else None,
        )
        if "conductance_used" in trace:
            axes[3].plot(
                time,
                trace["conductance_used"][:, 0, neuron],
                linestyle=":",
                color=line.get_color(),
                label="used for current" if neuron == 0 else None,
            )
    axes[3].set_ylabel("conductance")
    axes[3].legend(loc="upper right", fontsize="small")
    axes[4].plot(time, trace["current_used"][:, 0])
    axes[4].set_ylabel("current")
    axes[5].plot(time, trace["reversal_potential"][:, 0])
    axes[5].set_ylabel("E (mV)")
    axes[6].plot(
        time, trace["reversal_potential"][:, 0] - trace["voltage_observed"][:, 0]
    )
    axes[6].set_ylabel("E - observed v")
    axes[6].set_xlabel("time (ms)")
    fig.suptitle(title)
    path = save_fig(fig, name=name, path=output, file=__file__, suffix="png")
    plt.close(fig)
    return path


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ConductancePSC scientific validation",
        "",
        f"Overall status: **{summary['status']}**",
        "",
        "The standalone scale executes the existing `ConductancePSC` directly.",
        "Its explicit `v_seq` path is exercised against repeated single steps and",
        "across a stateful chunk boundary on the Torch backend.",
        "A separate actual-Module suite functionally qualifies the standalone",
        "CuPy, Triton, and TileLang composition; this report makes no performance",
        "claim and does not qualify native recurrent COBA scheduling.",
        "Connected scales execute actual `LIF` and `ConductancePSC` Modules through",
        "the public Torch-only `ConductanceRecurrentNN` schedule. Human review of",
        "the generated traces remains the validation gate for this schedule, and",
        "the report does not establish native recurrent COBA parity.",
        "",
        "## Checks",
        "",
        "| Status | Check | Details |",
        "|---|---|---|",
    ]
    for check in summary["checks"]:
        details = {
            key: value for key, value in check.items() if key not in ("name", "status")
        }
        lines.append(
            f"| {check['status']} | `{check['name']}` | "
            f"`{json.dumps(_json_value(details), sort_keys=True)}` |"
        )
    lines.extend(("", "## Supported existing paths", ""))
    for finding in summary["supported"]:
        lines.append(
            f"- **{finding['status']}** `{finding['name']}`: {finding['reason']}"
        )
    lines.extend(("", "## Unsupported existing paths", ""))
    for finding in summary["unsupported"]:
        lines.append(
            f"- **{finding['status']}** `{finding['name']}`: {finding['reason']}"
        )
    lines.extend(("", "## Artifacts", ""))
    lines.extend(f"- [{name}]({name})" for name in summary["artifacts"])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("run/parity/conductance-validation")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    artifacts: list[str] = []

    single = _standalone_case(checks)
    _reversal_checks(checks)
    _continuation_check(checks)
    pair = _pair_case(checks)
    network = _network_case(checks)
    gradient_metrics = _gradient_checks(checks)
    traces = {"single": single, "pair": pair, "network": network}

    for name, title, trace in (
        ("conductance-single", "Existing ConductancePSC standalone", single),
        ("conductance-pair", "ConductanceRecurrentNN connected pair", pair),
        ("conductance-network", "ConductanceRecurrentNN six-neuron network", network),
    ):
        artifacts.append(_plot_activity(args.output, name, title, trace).name)

    trace_path = args.output / "traces.pt"
    torch.save(traces, trace_path)
    artifacts.append(trace_path.name)
    metrics_path = args.output / "metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=gradient_metrics.keys())
        writer.writeheader()
        writer.writerow(gradient_metrics)
    artifacts.append(metrics_path.name)

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
        "schedule": {
            "classification": "PUBLIC_TORCH_GOLDEN_CANDIDATE",
            "voltage_observation": "previous step post-reset voltage",
            "current": "prior conductance times the observed driving force",
            "event_order": "current, neuron charge/fire/reset, conductance update",
        },
        "record_semantics": {
            "voltage_observed": "previous post-reset voltage used for current",
            "voltage_charged": "voltage after neuron charge and before reset",
            "voltage_post": "voltage after spike reset",
            "conductance_used": "prior conductance used for connected current",
            "conductance_updated": "conductance after decay and emitted-spike input",
            "current_used": "current delivered to the neuron on this step",
        },
        "checks": checks,
        "supported": [
            {
                "name": "ConductancePSC.multi_step_forward(z_seq, v_seq=v_seq)",
                "status": "SUPPORTED",
                "reason": (
                    "plain backend='torch' executes stateful repeated-single-step "
                    "COBA semantics when v_seq is explicit; omitting v_seq retains "
                    "the legacy stateless convolution"
                ),
            },
            {
                "name": "ConductancePSC exact-class standalone cell adapter",
                "status": "SUPPORTED",
                "reason": (
                    "KernelOptions(neuron='torch') and CuPy, Triton, and TileLang "
                    "reuse the existing exponential state operator for stateful "
                    "CUBA/COBA single and sequence calls, then apply g * (E - v) "
                    "in Torch; Slurm 1287 provides functional, not performance, "
                    "evidence"
                ),
            },
            {
                "name": "ConductanceRecurrentNN Torch schedule",
                "status": "SUPPORTED",
                "reason": (
                    "the public exact-class schedule uses previous post-reset "
                    "voltage and conductance, then updates conductance from the "
                    "emitted spike; this report provides CPU scientific traces "
                    "pending human figure confirmation"
                ),
            },
        ],
        "unsupported": [
            {
                "name": "ConductanceRecurrentNN native COBA scheduling",
                "status": "UNSUPPORTED",
                "reason": (
                    "the public recurrent schedule currently requires plain "
                    "backend='torch' for the network, neuron, and synapse"
                ),
            },
            {
                "name": "002HIP conductance model parity",
                "status": "UNSUPPORTED",
                "reason": (
                    "the audited 002HIP exponential and double-exponential models "
                    "have different state, event timing, sign, and edge semantics; "
                    "the standalone btorch adapter is not their lowering"
                ),
            },
        ],
        "gradient_metrics": gradient_metrics,
        "artifacts": [*artifacts, "summary.json", "report.md"],
    }
    (args.output / "summary.json").write_text(
        json.dumps(_json_value(summary), indent=2) + "\n"
    )
    _write_report(args.output / "report.md", summary)
    print(
        json.dumps(
            {"status": status, "checks": len(checks), "output": str(args.output)},
            indent=2,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
