"""Plotting helpers for two-compartment fit reports.

Re-exported from :mod:`btorch.analysis.two_compartment_fit`.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np


if TYPE_CHECKING:
    from .evaluation import FitEvaluation


def plot_two_compartment_fit(
    evaluation: "FitEvaluation",
    history: Sequence[dict[str, float | str]] | None = None,
    *,
    output_path: str | Path,
) -> Path:
    """Create a compact fit-quality figure with traces and loss history."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    time_ms = evaluation.traces["time_ms"]

    axes[0].plot(time_ms, evaluation.traces["i_soma"], color="tab:blue", lw=1.2)
    axes[0].set_ylabel("I soma")
    axes[0].set_title(
        f"Specimen {evaluation.specimen_id}, sweep {evaluation.sweep_number}"
    )

    axes[1].plot(time_ms, evaluation.traces["v_true"], label="Recorded", lw=1.5)
    axes[1].plot(time_ms, evaluation.traces["v_pred"], label="Predicted", lw=1.2)
    axes[1].set_ylabel("Voltage")
    axes[1].legend(loc="best")

    spike_true_t = evaluation.traces["spike_true"]
    spike_pred_t = evaluation.traces["spike_pred"]
    axes[2].eventplot(
        [time_ms[spike_true_t > 0.5], time_ms[spike_pred_t > 0.5]],
        lineoffsets=[1, 0],
        linelengths=0.8,
        colors=["tab:green", "tab:red"],
    )
    axes[2].set_yticks([0, 1], ["Pred", "True"])
    axes[2].set_ylabel("Spikes")

    if history:
        x = np.arange(len(history))
        axes[3].plot(x, [float(row["total_loss"]) for row in history], label="Total")
        axes[3].plot(
            x,
            [float(row["voltage_loss"]) for row in history],
            label="Voltage",
        )
        axes[3].plot(
            x,
            [float(row["spike_loss"]) for row in history],
            label="Spike",
        )
        axes[3].legend(loc="best")
        axes[3].set_xlabel("Optimization step")
    else:
        axes[3].axis("off")

    axes[3].set_ylabel("Loss")
    metrics_text = "\n".join(
        f"{name}: {value:.4f}" for name, value in evaluation.metrics.items()
    )
    fig.text(
        0.99,
        0.5,
        metrics_text,
        va="center",
        ha="right",
        fontsize=9,
        family="monospace",
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
