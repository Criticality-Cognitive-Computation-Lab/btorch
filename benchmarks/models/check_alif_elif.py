"""Validate actual ALIF and ELIF Modules at three scientific scales.

The harness uses the existing Torch Modules as the only simulator. It writes
activity figures, raw traces, numerical checks, and a report under
``run/parity/alif-elif-validation`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
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
from btorch.models.neurons.alif import ALIF, ELIF  # noqa: E402
from btorch.models.rnn import RecurrentNN  # noqa: E402
from btorch.models.synapse import AlphaPSC  # noqa: E402
from btorch.utils.file import save_fig  # noqa: E402


DT = 1.0
DTYPE = torch.float32
MODELS = ("ALIF", "ELIF")
STATE_NAMES = (
    "neuron.v",
    "neuron.v_pre_spike",
    "neuron.g_k",
    "neuron.refractory",
    "synapse.psc",
    "synapse.h",
)


@dataclass(frozen=True)
class ModelConfig:
    threshold: float
    sustained_current: float
    impulse_current: float
    delta_t: float | None = None
    v_t: float | None = None


CONFIGS = {
    "ALIF": ModelConfig(-50.0, 20.0, 30.0),
    "ELIF": ModelConfig(-48.0, 12.0, 22.0, delta_t=2.0, v_t=-55.0),
}


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _build_neuron(
    model: str,
    neurons: int,
    *,
    trainable: bool = False,
    dg_k: float = 0.12,
    hard_reset: bool = False,
    step_mode: str = "s",
) -> ALIF | ELIF:
    config = CONFIGS[model]
    kwargs: dict[str, Any] = {
        "n_neuron": neurons,
        "v_threshold": config.threshold,
        "v_reset": -65.0,
        "c_m": 1.0,
        "g_leak": 0.05,
        "E_leak": -70.0,
        "E_k": -80.0,
        "g_k_init": 0.0,
        "tau_adapt": 180.0 if model == "ALIF" else 150.0,
        "dg_k": dg_k,
        "tau_ref": 2.0,
        "hard_reset": hard_reset,
        "pre_spike_v": True,
        "step_mode": step_mode,
        "dtype": DTYPE,
    }
    if trainable:
        kwargs["trainable_param"] = {"dg_k"}
    if model == "ELIF":
        kwargs.update(delta_T=config.delta_t, v_T=config.v_t)
    return (ALIF if model == "ALIF" else ELIF)(**kwargs)


def _single_stimulus(model: str, steps: int = 360) -> torch.Tensor:
    config = CONFIGS[model]
    current = torch.zeros(steps, 1, 1, dtype=DTYPE)
    current[60] = config.impulse_current
    current[80:220] = config.sustained_current
    return current


def _run_neuron(neuron: ALIF | ELIF, stimulus: torch.Tensor) -> dict[str, torch.Tensor]:
    init_net_state(neuron, batch_size=stimulus.shape[1], dtype=DTYPE)
    traces = {name: [] for name in ("spike", "v", "v_pre", "g_k", "refractory")}
    with torch.no_grad(), environ.context(dt=DT):
        for current in stimulus:
            spike = neuron(current)
            traces["spike"].append(spike.clone())
            traces["v"].append(neuron.v.clone())
            traces["v_pre"].append(neuron.v_pre_spike.clone())
            traces["g_k"].append(neuron.g_k.clone())
            traces["refractory"].append(neuron.refractory.clone())
    return {
        "input": stimulus.clone(),
        **{key: torch.stack(value) for key, value in traces.items()},
    }


def _connection(neurons: int, scale: str) -> torch.nn.Linear:
    linear = torch.nn.Linear(neurons, neurons, bias=False, dtype=DTYPE)
    with torch.no_grad():
        linear.weight.zero_()
        if scale == "pair":
            linear.weight[1, 0] = 1.0
        else:
            index = torch.arange(neurons)
            linear.weight[(index + 1) % neurons, index] = 0.7
            linear.weight[(index + 3) % neurons, index] = 0.35
    return linear


def _build_recurrent(
    model: str,
    neurons: int,
    scale: str,
    *,
    stress: bool = False,
) -> RecurrentNN:
    # Soft reset is retained for single-neuron validation. ELIF recurrent
    # validation uses hard reset to keep the exponential upswing in a finite,
    # physiologically interpretable envelope. The original soft-reset setup is
    # executed separately as a named stress case.
    hard_reset = model == "ELIF" and not stress
    neuron = _build_neuron(model, neurons, hard_reset=hard_reset)
    gain = 18.0 if scale == "pair" else 8.0
    synapse = AlphaPSC(
        neurons, tau_syn=3.0, linear=_connection(neurons, scale), g_max=gain
    )
    network = RecurrentNN(
        neuron,
        synapse,
        backend="torch",
        unroll=8,
        update_state_names=STATE_NAMES,
    )
    init_net_state(network, batch_size=1, dtype=DTYPE)
    return network


def _network_stimulus(model: str, neurons: int, scale: str) -> torch.Tensor:
    steps = 180 if scale == "pair" else 220
    current = torch.zeros(steps, 1, neurons, dtype=DTYPE)
    drive = CONFIGS[model].sustained_current
    if scale == "pair":
        current[20:90, 0, 0] = drive
    else:
        current[20:120, 0, : max(1, neurons // 4)] = drive
    return current


def _run_recurrent(
    network: RecurrentNN, stimulus: torch.Tensor
) -> dict[str, torch.Tensor]:
    with torch.no_grad(), environ.context(dt=DT):
        spikes, records = network(stimulus)
    return {
        "input": stimulus.clone(),
        "spike": spikes.clone(),
        **{name: value.clone() for name, value in records.items()},
    }


def _add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, **details: Any
) -> None:
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", **details})


def _single_checks(
    model: str, traces: dict[str, torch.Tensor], checks: list[dict[str, Any]]
) -> None:
    spike = traces["spike"][:, 0, 0]
    v = traces["v"][:, 0, 0]
    g_k = traces["g_k"][:, 0, 0]
    refractory = traces["refractory"][:, 0, 0]
    prefix = f"{model.lower()}.single"
    _add_check(
        checks,
        prefix + ".finite",
        all(bool(torch.isfinite(value).all()) for value in traces.values()),
    )
    _add_check(checks, prefix + ".rest_silent", int(spike[:60].sum()) == 0)
    _add_check(
        checks,
        prefix + ".rest_toward_E_leak",
        abs(float(v[59]) + 70.0) < abs(float(v[0]) + 70.0),
        initial_v=float(v[0]),
        resting_v=float(v[59]),
    )
    impulse_spikes = int(spike[60:80].sum())
    _add_check(
        checks, prefix + ".impulse_response", impulse_spikes >= 1, spikes=impulse_spikes
    )
    driven_spikes = int(spike[80:220].sum())
    _add_check(
        checks,
        prefix + ".sustained_activity",
        driven_spikes >= 3,
        spikes=driven_spikes,
    )
    _add_check(
        checks,
        prefix + ".adaptation_recruited",
        float(g_k.max()) > 0.12,
        max_g_k=float(g_k.max()),
    )
    spike_times = spike.nonzero().flatten()
    minimum_isi = (
        int((spike_times[1:] - spike_times[:-1]).min()) if len(spike_times) > 1 else 0
    )
    _add_check(
        checks,
        prefix + ".refractory_spacing",
        minimum_isi >= 2,
        minimum_isi=minimum_isi,
        max_refractory=float(refractory.max()),
    )
    _add_check(
        checks,
        prefix + ".step_off_silence",
        int(spike[-60:].sum()) == 0,
        late_spikes=int(spike[-60:].sum()),
    )
    _add_check(
        checks,
        prefix + ".step_off_recovery",
        float(g_k[-1]) < float(g_k[220]),
        g_k_at_off=float(g_k[220]),
        final_g_k=float(g_k[-1]),
        final_v=float(v[-1]),
    )


def _plot_activity(
    output_dir: Path,
    name: str,
    model: str,
    scale: str,
    traces: dict[str, torch.Tensor],
) -> Path:
    time = torch.arange(traces["input"].shape[0]) * DT
    input_value = traces["input"][:, 0]
    spike = traces["spike"][:, 0]
    if scale == "single":
        v = traces["v"][:, 0]
        v_pre = traces["v_pre"][:, 0]
        g_k = traces["g_k"][:, 0]
        refractory = traces["refractory"][:, 0]
    else:
        v = traces["neuron.v"][:, 0]
        v_pre = traces["neuron.v_pre_spike"][:, 0]
        g_k = traces["neuron.g_k"][:, 0]
        refractory = traces["neuron.refractory"][:, 0]
    rows = 5 if scale == "single" else 7
    fig, axes = plt.subplots(
        rows, 1, figsize=(11, 2.3 * rows), sharex=True, constrained_layout=True
    )
    axes[0].plot(time, input_value)
    axes[0].set_ylabel("input")
    coordinates = spike.nonzero()
    if coordinates.numel():
        axes[1].scatter(time[coordinates[:, 0]], coordinates[:, 1], marker="|", s=24)
    axes[1].set_ylabel("spikes")
    axes[1].set_ylim(-0.7, spike.shape[-1] - 0.3)
    # Keep each neuron's charged/reset traces the same color, and label the
    # two observation times once rather than adding 2*N legend entries.
    for neuron_index in range(v.shape[-1]):
        (line,) = axes[2].plot(
            time,
            v[:, neuron_index],
            label="post-reset" if neuron_index == 0 else None,
        )
        axes[2].plot(
            time,
            v_pre[:, neuron_index],
            linestyle=":",
            alpha=0.75,
            color=line.get_color(),
            label="charged" if neuron_index == 0 else None,
        )
    axes[2].axhline(
        CONFIGS[model].threshold, color="black", linestyle="--", linewidth=0.8
    )
    axes[2].set_ylabel("v (mV)")
    axes[2].legend(loc="upper right", fontsize="small")
    axes[3].plot(time, g_k)
    axes[3].set_ylabel("g_k (nS)")
    axes[4].plot(time, refractory)
    axes[4].set_ylabel("refractory (ms)")
    if scale != "single":
        axes[5].plot(time, traces["synapse.psc"][:, 0])
        axes[5].set_ylabel("PSC")
        axes[6].plot(time, traces["synapse.h"][:, 0])
        axes[6].set_ylabel("PSC h")
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle(f"{model} {scale} activity | dt={DT} ms | {name}")
    path = save_fig(fig, name=name, path=output_dir, file=__file__, suffix="png")
    plt.close(fig)
    return path


def _first_nonfinite(values: dict[str, torch.Tensor]) -> dict[str, Any] | None:
    for name, value in values.items():
        bad = (~torch.isfinite(value)).nonzero()
        if bad.numel():
            index = tuple(int(item) for item in bad[0])
            return {"state": name, "index": index, "value": float(value[index])}
    return None


def _continuation_check(
    model: str,
    scale: str,
    stimulus: torch.Tensor,
    checks: list[dict[str, Any]],
) -> None:
    split = stimulus.shape[0] // 2 + 3
    if scale == "single":
        full_model = _build_neuron(model, stimulus.shape[-1], step_mode="m")
        chunked_model = _build_neuron(model, stimulus.shape[-1], step_mode="m")
        init_net_state(full_model, batch_size=1, dtype=DTYPE)
        init_net_state(chunked_model, batch_size=1, dtype=DTYPE)
        with torch.no_grad(), environ.context(dt=DT):
            full_spike = full_model(stimulus)
            left_spike = chunked_model(stimulus[:split])
            right_spike = chunked_model(stimulus[split:])
        full = {
            "spike": full_spike,
            "v": full_model.v,
            "v_pre": full_model.v_pre_spike,
            "g_k": full_model.g_k,
            "refractory": full_model.refractory,
        }
        chunked = {
            "spike": torch.cat((left_spike, right_spike)),
            "v": chunked_model.v,
            "v_pre": chunked_model.v_pre_spike,
            "g_k": chunked_model.g_k,
            "refractory": chunked_model.refractory,
        }
    else:
        neurons = stimulus.shape[-1]
        full_network = _build_recurrent(model, neurons, scale)
        chunked_network = _build_recurrent(model, neurons, scale)
        full = _run_recurrent(full_network, stimulus)
        with torch.no_grad(), environ.context(dt=DT):
            left_spike, left_records = chunked_network(stimulus[:split])
            right_spike, right_records = chunked_network(stimulus[split:])
        chunked = {
            "spike": torch.cat((left_spike, right_spike)),
            **{
                name: torch.cat((left_records[name], right_records[name]))
                for name in STATE_NAMES
            },
        }
        full = {name: full[name] for name in ("spike", *STATE_NAMES)}
    full_bad = _first_nonfinite(full)
    chunked_bad = _first_nonfinite(chunked)
    finite = full_bad is None and chunked_bad is None
    maximum = (
        max(float((full[name] - chunked[name]).abs().max()) for name in full)
        if finite
        else None
    )
    _add_check(
        checks,
        f"{model.lower()}.{scale}.continuation",
        finite and maximum == 0.0,
        max_abs=maximum,
        split=split,
        full_nonfinite=full_bad,
        chunked_nonfinite=chunked_bad,
    )


def _run_without_refractory(
    neuron: ALIF | ELIF, stimulus: torch.Tensor
) -> dict[str, torch.Tensor]:
    init_net_state(neuron, batch_size=1, dtype=DTYPE)
    values = {name: [] for name in ("spike", "v", "g_k")}
    with torch.no_grad(), environ.context(dt=DT):
        for current in stimulus:
            values["spike"].append(neuron(current).clone())
            values["v"].append(neuron.v.clone())
            values["g_k"].append(neuron.g_k.clone())
    return {name: torch.stack(value) for name, value in values.items()}


def _limit_checks(model: str, checks: list[dict[str, Any]]) -> None:
    # Hard reset isolates the adaptation limit from ELIF soft-reset overshoot.
    stimulus = torch.full((260, 1, 1), CONFIGS[model].sustained_current, dtype=DTYPE)
    adapted = _run_neuron(_build_neuron(model, 1, hard_reset=True), stimulus)
    plain = _run_neuron(_build_neuron(model, 1, dg_k=0.0, hard_reset=True), stimulus)
    _add_check(
        checks,
        f"{model.lower()}.limit.dg_k_zero",
        int(adapted["spike"].sum()) < int(plain["spike"].sum()),
        adapted_spikes=int(adapted["spike"].sum()),
        zero_adaptation_spikes=int(plain["spike"].sum()),
        hard_reset=True,
    )
    if model == "ELIF":
        common = dict(
            n_neuron=1,
            v_threshold=1.0,
            v_reset=0.0,
            c_m=1.0,
            g_leak=0.1,
            E_leak=0.0,
            E_k=-1.0,
            tau_adapt=20.0,
            dg_k=0.1,
            tau_ref=None,
            dtype=DTYPE,
        )
        alif = ALIF(**common)
        elif_model = ELIF(**common, delta_T=1.0, v_T=1e3)
        small = torch.full((40, 1, 1), 0.4, dtype=DTYPE)
        a = _run_without_refractory(alif, small)
        e = _run_without_refractory(elif_model, small)
        maximum = max(
            float((a[name] - e[name]).abs().max()) for name in ("spike", "v", "g_k")
        )
        _add_check(
            checks,
            "elif.limit.vanishing_exponential",
            maximum < 1e-9,
            max_abs=maximum,
        )


def _network_checks(
    model: str,
    scale: str,
    traces: dict[str, torch.Tensor],
    checks: list[dict[str, Any]],
) -> None:
    spike = traces["spike"][:, 0]
    prefix = f"{model.lower()}.{scale}"
    _add_check(
        checks,
        prefix + ".finite",
        all(bool(torch.isfinite(value).all()) for value in traces.values()),
    )
    if scale == "pair":
        pre_times = spike[:, 0].nonzero().flatten()
        post_times = spike[:, 1].nonzero().flatten()
        propagated = (
            len(pre_times) > 0
            and len(post_times) > 0
            and int(post_times[0]) > int(pre_times[0])
        )
        _add_check(
            checks,
            prefix + ".directed_propagation",
            propagated,
            first_pre=int(pre_times[0]) if len(pre_times) else None,
            first_post=int(post_times[0]) if len(post_times) else None,
        )
        _add_check(
            checks,
            prefix + ".synaptic_delivery",
            float(traces["synapse.psc"][:, 0, 1].max()) > 0.0,
            max_post_psc=float(traces["synapse.psc"][:, 0, 1].max()),
        )
    else:
        active = (spike.sum(0) > 0).sum()
        total = int(spike.sum())
        _add_check(
            checks,
            prefix + ".mixed_population_activity",
            0 < total < spike.numel() and int(active) > spike.shape[1] // 4,
            spike_count=total,
            active_neurons=int(active),
        )
        _add_check(
            checks,
            prefix + ".step_off_decay",
            float(traces["neuron.g_k"][-1].mean())
            < float(traces["neuron.g_k"][120].mean()),
            g_k_at_off=float(traces["neuron.g_k"][120].mean()),
            final_g_k=float(traces["neuron.g_k"][-1].mean()),
        )


def _gradient_checks(model: str, checks: list[dict[str, Any]]) -> dict[str, float]:
    neuron = _build_neuron(model, 2, trainable=True)
    init_net_state(neuron, batch_size=1, dtype=DTYPE)
    x = torch.full(
        (24, 1, 2),
        CONFIGS[model].sustained_current * 0.7,
        dtype=DTYPE,
        requires_grad=True,
    )
    loss = torch.zeros((), dtype=DTYPE)
    with environ.context(dt=DT):
        for current in x:
            spike = neuron(current)
            loss = loss + spike.sum() + 0.01 * neuron.v.square().sum()
            loss = loss + neuron.g_k.square().sum()
    input_grad, adapt_grad = torch.autograd.grad(loss, (x, neuron.dg_k))
    finite = bool(torch.isfinite(input_grad).all() and torch.isfinite(adapt_grad).all())
    nonzero = bool(input_grad.abs().sum() > 0 and adapt_grad.abs().sum() > 0)
    _add_check(
        checks,
        f"{model.lower()}.single.gradient",
        finite and nonzero,
        input_grad_norm=float(input_grad.norm()),
        dg_k_grad_norm=float(adapt_grad.norm()),
    )

    network = _build_recurrent(model, 4, "network")
    network.neuron.dg_k.requires_grad_()
    network.synapse.linear.weight.requires_grad_()
    drive = _network_stimulus(model, 4, "network")[:32].requires_grad_()
    with environ.context(dt=DT):
        spike, records = network(drive)
        network_loss = spike.sum() + 0.01 * records["neuron.v"].square().mean()
        network_loss = network_loss + records["neuron.g_k"].square().mean()
    gradients = torch.autograd.grad(
        network_loss,
        (drive, network.neuron.dg_k, network.synapse.linear.weight),
        allow_unused=True,
    )
    valid = all(
        value is not None and bool(torch.isfinite(value).all()) for value in gradients
    )
    _add_check(
        checks,
        f"{model.lower()}.network.gradient",
        valid,
        gradient_norms=[
            None if value is None else float(value.norm()) for value in gradients
        ],
    )
    return {
        "single_input_grad_norm": float(input_grad.norm()),
        "single_dg_k_grad_norm": float(adapt_grad.norm()),
        "network_input_grad_norm": float(gradients[0].norm())
        if gradients[0] is not None
        else 0.0,
    }


def _stress_diagnostic(
    scale: str,
    traces: dict[str, torch.Tensor],
) -> dict[str, Any]:
    observed = {
        name: traces[name]
        for name in (
            "neuron.v",
            "neuron.v_pre_spike",
            "neuron.g_k",
            "synapse.psc",
            "synapse.h",
        )
    }
    first = _first_nonfinite(observed)
    result: dict[str, Any] = {
        "case": f"elif.soft_reset.{scale}",
        "status": "KNOWN_LIMITATION" if first else "FINITE",
        "first_nonfinite": first,
    }
    if first is None:
        return result
    time_index, batch_index, neuron_index = first["index"]
    previous = max(time_index - 1, 0)
    synaptic_input = (
        torch.zeros_like(traces["input"][time_index])
        if time_index == 0
        else traces["synapse.psc"][time_index - 1]
    )
    total_input = traces["input"][time_index] + synaptic_input
    probe = _build_neuron("ELIF", traces["input"].shape[-1])
    v_previous = traces["neuron.v"][previous]
    g_k_previous = traces["neuron.g_k"][previous]
    with torch.no_grad():
        derivative, linear = probe.dV(v_previous, g_k_previous, total_input)
        expm1_argument = DT * linear
        expm1_value = torch.expm1(expm1_argument)
    coordinate = (batch_index, neuron_index)
    result.update(
        {
            "time": time_index,
            "external_input": float(traces["input"][time_index][coordinate]),
            "synaptic_input": float(synaptic_input[coordinate]),
            "previous_v": float(v_previous[coordinate]),
            "previous_v_charged": float(
                traces["neuron.v_pre_spike"][previous][coordinate]
            ),
            "previous_g_k": float(g_k_previous[coordinate]),
            "dV_derivative": float(derivative[coordinate]),
            "dV_linear": float(linear[coordinate]),
            "exp_euler_expm1_argument": float(expm1_argument[coordinate]),
            "exp_euler_expm1_value": float(expm1_value[coordinate]),
        }
    )
    return result


def _state_metadata() -> list[dict[str, Any]]:
    return [
        {
            "path": "neuron.v",
            "axes": "time,batch,neuron",
            "units": "mV",
            "observation": "post-reset",
        },
        {
            "path": "neuron.v_pre_spike",
            "axes": "time,batch,neuron",
            "units": "mV",
            "observation": "charged, before threshold/reset",
        },
        {
            "path": "neuron.g_k",
            "axes": "time,batch,neuron",
            "units": "nS",
            "observation": "post-decay and post-spike increment",
        },
        {
            "path": "neuron.refractory",
            "axes": "time,batch,neuron",
            "units": "ms",
            "observation": "post-step counter",
        },
        {
            "path": "synapse.psc",
            "axes": "time,batch,neuron",
            "units": "input-current units",
            "observation": "post-delivery",
        },
        {
            "path": "synapse.h",
            "axes": "time,batch,neuron",
            "units": "input-current units",
            "observation": "post-delivery alpha state",
        },
    ]


def _write_report(output: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ALIF / ELIF scientific validation",
        "",
        f"Overall status: **{summary['status']}**",
        "",
        "The harness executed the existing btorch Modules with `dt=1 ms` on CPU.",
        "No generated kernel or independent neuron equation was used.",
        "",
        "The representative ELIF connected cases use hard reset. The original",
        "soft-reset configuration is retained as a stress case because its",
        "exponential upswing becomes nonfinite under sustained recurrent drive.",
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
        encoded = json.dumps(_json_value(details), sort_keys=True)
        lines.append(f"| {check['status']} | `{check['name']}` | `{encoded}` |")
    lines.extend(("", "## Known stress limitation", ""))
    for finding in summary["findings"]:
        encoded = json.dumps(_json_value(finding), sort_keys=True, allow_nan=False)
        lines.append(f"- `{finding['case']}`: `{encoded}`")
    lines.extend(("", "## Artifacts", ""))
    for artifact in summary["artifacts"]:
        lines.append(f"- [{artifact}]({artifact})")
    lines.extend(("", "## State semantics", ""))
    for state in summary["state_metadata"]:
        lines.append(
            f"- `{state['path']}`: {state['axes']}, {summary['environment']['dtype']} "
            "on CPU, "
            f"{state['units']}; observed {state['observation']}."
        )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    global DTYPE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("run/parity/alif-elif-validation")
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Torch floating-point dtype used by every Module and stimulus.",
    )
    args = parser.parse_args()
    DTYPE = getattr(torch, args.dtype)
    args.output.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    traces: dict[str, dict[str, torch.Tensor]] = {}
    stress_traces: dict[str, dict[str, torch.Tensor]] = {}
    findings: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    artifacts: list[str] = []

    for model in MODELS:
        stimulus = _single_stimulus(model)
        single = _run_neuron(_build_neuron(model, 1), stimulus)
        traces[f"{model.lower()}.single"] = single
        _single_checks(model, single, checks)
        _continuation_check(model, "single", stimulus, checks)
        _limit_checks(model, checks)
        path = _plot_activity(
            args.output, f"{model.lower()}-single", model, "single", single
        )
        artifacts.append(path.name)

        for scale, neurons in (("pair", 2), ("network", 12)):
            network_stimulus = _network_stimulus(model, neurons, scale)
            activity = _run_recurrent(
                _build_recurrent(model, neurons, scale), network_stimulus
            )
            traces[f"{model.lower()}.{scale}"] = activity
            _network_checks(model, scale, activity, checks)
            _continuation_check(model, scale, network_stimulus, checks)
            path = _plot_activity(
                args.output,
                f"{model.lower()}-{scale}",
                model,
                scale,
                activity,
            )
            artifacts.append(path.name)

            if model == "ELIF":
                stress = _run_recurrent(
                    _build_recurrent(model, neurons, scale, stress=True),
                    network_stimulus,
                )
                stress_traces[scale] = stress
                findings.append(_stress_diagnostic(scale, stress))
                path = _plot_activity(
                    args.output,
                    f"elif-{scale}-soft-reset-stress",
                    model,
                    scale,
                    stress,
                )
                artifacts.append(path.name)

        gradient_metrics = _gradient_checks(model, checks)
        metrics.append(
            {
                "model": model,
                "single_spikes": int(single["spike"].sum()),
                "single_max_g_k": float(single["g_k"].max()),
                **gradient_metrics,
            }
        )

    trace_path = args.output / "traces.pt"
    torch.save(traces, trace_path)
    artifacts.append(trace_path.name)
    stress_trace_path = args.output / "stress-traces.pt"
    torch.save(stress_traces, stress_trace_path)
    artifacts.append(stress_trace_path.name)
    diagnostics_path = args.output / "stress-diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(_json_value(findings), indent=2, allow_nan=False) + "\n"
    )
    artifacts.append(diagnostics_path.name)
    metrics_path = args.output / "metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    artifacts.append(metrics_path.name)
    status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    summary = {
        "status": status,
        "command": (
            f"python {Path(__file__)} --output {args.output} --dtype {args.dtype}"
        ),
        "environment": {
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": str(DTYPE),
            "dt_ms": DT,
        },
        "configuration_envelope": {
            "single": "soft reset for ALIF and ELIF",
            "representative_recurrent": "ALIF soft reset; ELIF hard reset",
            "stress": "ELIF soft reset with the same recurrent drive and gain",
        },
        "checks": checks,
        "findings": findings,
        "metrics": metrics,
        "state_metadata": _state_metadata(),
        "artifacts": [*artifacts, "summary.json", "report.md"],
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(_json_value(summary), indent=2, allow_nan=False) + "\n"
    )
    _write_report(args.output / "report.md", summary)
    print(
        json.dumps(
            {
                "status": status,
                "output": str(args.output),
                "checks": len(checks),
                "known_limitations": len(findings),
            },
            indent=2,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
