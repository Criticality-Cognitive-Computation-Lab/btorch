"""Generate three-scale GLIF3 activity evidence using existing btorch models.

Run ``python examples/kernel_generation.py --output run/figures``. The suite
can also be called by the kernel-gen workflow's verify command. This is model
validation, not a replacement simulator or an optimized kernel.
"""

import argparse
import inspect
import json
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from btorch.models import functional
from btorch.models.connection import SparseConnection
from btorch.models.kernels.response import (
    ResponsePlotOptions,
    describe_response,
    plot_response,
    simulate_response,
    validate_response,
)
from btorch.models.neurons import GLIF3
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import AlphaPSC


_FACTORY_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "skills/btorch-kernel-gen/assets/templates/glif3_rsnn.py"
)


def run_suite(output: str | Path = "run/figures") -> dict:
    """Run single, pair and network activity with the same actual model
    factory.

    Args:
        output: User-selected OOT figure directory.

    Returns:
        Workflow checks, figures, source files and execution metadata.
    """
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    build_model = runpy.run_path(str(_FACTORY_SOURCE))["build_model"]
    checks, figures, artifacts = [], [], []
    units = {
        "dt": "ms",
        "input": "pA",
        "neuron.v": "mV",
        "neuron.Iasc": "pA",
        "synapse.psc": "pA",
    }
    for level, n_neuron in (("single", 1), ("pair", 2), ("network", 8)):
        for protocol in ("zero", "impulse", "step"):
            model = build_model(n_neuron)
            x_seq = torch.zeros(400, 1, n_neuron)
            # Drive only the first neuron: the pair/network response must propagate
            # through the very same SparseConnection/AlphaPSC/RecurrentNN code.
            if protocol == "impulse":
                x_seq[20, :, 0] = 12000.0
            elif protocol == "step":
                x_seq[20:220, :, 0] = 300.0
            functional.init_net_state(model, batch_size=1, dtype=x_seq.dtype)
            trace = simulate_response(
                model,
                x_seq,
                dt=1.0,
                state_names=("neuron.v", "neuron.Iasc", "synapse.psc"),
            )
            issues = validate_response(trace)
            # These checks validate this configured example, not a universal rule
            # that every computational-neuroscience model must start silent.
            if trace.spikes[:20].count_nonzero():
                issues.append("Configured resting initial state spiked before input")
            if protocol != "zero" and not trace.spikes[20:220, :, 0].count_nonzero():
                issues.append("Driven neuron did not fire during sustained input")
            if (
                protocol != "zero"
                and n_neuron > 1
                and not trace.spikes[:, :, 1].count_nonzero()
            ):
                issues.append("Downstream neuron did not respond to upstream drive")
            if protocol == "zero" and trace.spikes.count_nonzero():
                issues.append("No-input model unexpectedly fired")
            fig = plot_response(
                trace,
                title=f"GLIF3 · {level} · {protocol}",
                options=ResponsePlotOptions(units=units, time_unit="ms", batch_index=0),
            )
            path = destination / f"glif3_{level}_{protocol}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            figures.append(str(path.resolve()))
            data_path = destination / f"glif3_{level}_{protocol}.pt"
            torch.save(
                {
                    "input": trace.x_seq,
                    "spikes": trace.spikes,
                    "states": trace.states,
                    "dt": trace.dt,
                },
                data_path,
            )
            description_path = data_path.with_suffix(".json")
            description_path.write_text(
                json.dumps(
                    describe_response(
                        trace,
                        units=units,
                    ),
                    indent=2,
                    allow_nan=False,
                )
            )
            artifacts.extend(
                [str(data_path.resolve()), str(description_path.resolve())]
            )

            checks.append(
                {
                    "name": f"glif3_{level}_{protocol}_response",
                    "level": level,
                    "figures": [str(path.resolve())],
                    "status": "FAIL" if issues else "PASS",
                    "detail": "; ".join(issues)
                    or (
                        "Smoke checks only: finite states, resting silence "
                        f"and spike presence ({protocol})"
                    ),
                }
            )
    return {
        "checks": checks,
        "figures": figures,
        "artifacts": artifacts,
        "source_files": [
            str(Path(__file__).resolve()),
            str(_FACTORY_SOURCE),
            inspect.getfile(simulate_response),
        ],
        "model_source_files": [
            str(_FACTORY_SOURCE),
            *[
                inspect.getfile(cls)
                for cls in (GLIF3, AlphaPSC, SparseConnection, RecurrentNN)
            ],
        ],
        "required_checks": [check["name"] for check in checks],
        "requires_model_confirmation": True,
        "metadata": {
            "dt_unit": "ms",
            "torch_version": str(torch.__version__),
            "cases": "zero/impulse/step-on-off, levels single/pair/network",
            "execution": "Actual GLIF3/AlphaPSC/SparseConnection via RecurrentNN",
            "gradient": "Not tested: this example runs inference only",
            "coverage": (
                "Finite values, pre-input silence, spike presence; "
                "not a dynamics, delay or kernel parity oracle"
            ),
            "model_confirmation_required": True,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="run/figures")
    arguments = parser.parse_args()
    result = run_suite(arguments.output)
    for check in result["checks"]:
        print(f"{check['status']}: {check['name']}: {check['detail']}")
    raise SystemExit(any(check["status"] != "PASS" for check in result["checks"]))
