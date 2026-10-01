"""Direct tests of the two-compartment fit report writer and plot helper.

They use a tiny hand-made :class:`FitEvaluation` (no model rollout, no
fitting) so that the file layout and the figure content can be asserted
exactly. Matplotlib runs with the non-interactive ``Agg`` backend.
"""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from btorch.fitting.two_compartment import (
    FitEvaluation,
    _plots,
    plot_two_compartment_fit,
    save_fit_report,
)


# Select the backend before any figure is created.
matplotlib.use("Agg")


def _evaluation(
    specimen_id: int = 7, sweep_number: int = 3, n_spikes: int = 2, seed: int = 0
) -> FitEvaluation:
    """Synthetic 20-step evaluation with ``n_spikes`` true and predicted
    spikes."""
    rng = np.random.default_rng(seed)
    steps = 20
    spike_true = np.zeros(steps)
    spike_true[[5, 12][:n_spikes]] = 1.0
    spike_pred = np.zeros(steps)
    spike_pred[[6, 12][:n_spikes]] = 1.0
    v_true = -65.0 + rng.normal(size=steps)
    return FitEvaluation(
        specimen_id=specimen_id,
        sweep_number=sweep_number,
        dt=0.5,
        metrics={"spike_count_true": float(n_spikes), "voltage_rmse": 1.25},
        traces={
            "time_ms": np.arange(steps) * 0.5,
            "i_soma": rng.normal(size=steps),
            "v_true": v_true,
            "v_pred": v_true + 0.1,
            "spike_true": spike_true,
            "spike_pred": spike_pred,
        },
    )


HISTORY = [
    {"total_loss": 3.0, "voltage_loss": 1.0, "spike_loss": 2.0, "phase": "global"},
    {"total_loss": 2.0, "voltage_loss": 0.5, "spike_loss": 1.5, "phase": "local"},
]


@pytest.fixture
def captured_figures(monkeypatch):
    """Keep the figures that the plot helper closes so tests can inspect
    them."""
    figures = []
    real_close = plt.close

    def close(fig=None):
        if fig is not None and fig != "all":
            figures.append(fig)
        real_close(fig)

    # Closing only invalidates the canvas, axes/lines stay inspectable.
    monkeypatch.setattr(_plots.plt, "close", close)
    return figures


def test_plot_writes_png_and_draws_all_panels(tmp_path, captured_figures):
    """The figure has four panels with traces, spike raster and loss curves."""
    out = plot_two_compartment_fit(
        _evaluation(), HISTORY, output_path=tmp_path / "nested" / "fit.png"
    )

    assert out == tmp_path / "nested" / "fit.png"  # parent dir is created
    assert out.read_bytes().startswith(b"\x89PNG")

    (fig,) = captured_figures
    ax_i, ax_v, ax_spikes, ax_loss = fig.axes[:4]
    assert ax_i.get_title() == "Specimen 7, sweep 3"
    # Recorded and predicted voltage are two labelled lines.
    assert [line.get_label() for line in ax_v.lines] == ["Recorded", "Predicted"]
    # One event collection per spike train (true / predicted).
    assert len(ax_spikes.collections) == 2
    # Loss panel: one curve per tracked loss, one point per history row.
    assert [line.get_label() for line in ax_loss.lines] == [
        "Total",
        "Voltage",
        "Spike",
    ]
    np.testing.assert_allclose(ax_loss.lines[0].get_ydata(), [3.0, 2.0])
    # Metrics are written as text next to the axes.
    assert any("voltage_rmse: 1.2500" in t.get_text() for t in fig.texts)


def test_plot_without_history_hides_loss_panel(tmp_path, captured_figures):
    """``history=None`` skips the loss curves and turns the axis off."""
    plot_two_compartment_fit(_evaluation(), output_path=tmp_path / "a.png")
    ax_loss = captured_figures[0].axes[3]
    assert not ax_loss.lines
    assert not ax_loss.axison


def test_save_fit_report_layout_and_content(tmp_path):
    """JSON payloads mirror the inputs; plots are named per evaluation."""
    model = torch.nn.Module()
    model.w_Ca = torch.nn.Parameter(torch.tensor([0.25, 0.5]))
    model.tau_s = torch.nn.Parameter(torch.tensor(12.0))
    silent = _evaluation(specimen_id=1, sweep_number=1, n_spikes=0)
    spiking = _evaluation(specimen_id=2, sweep_number=9, n_spikes=2)

    paths = save_fit_report(
        model,
        [silent, spiking],
        {"mean_voltage_rmse": 1.25},
        HISTORY,
        output_dir=tmp_path / "report",
    )

    assert paths["output_dir"] == tmp_path / "report"
    assert json.loads(paths["parameters"].read_text()) == {
        "w_Ca": [0.25, 0.5],
        "tau_s": [12.0],
    }
    metrics = json.loads(paths["metrics"].read_text())
    assert metrics["aggregate"] == {"mean_voltage_rmse": 1.25}
    assert [(m["specimen_id"], m["sweep_number"]) for m in metrics["per_sweep"]] == [
        (1, 1),
        (2, 9),
    ]
    assert metrics["per_sweep"][0]["dt_ms"] == 0.5
    assert json.loads(paths["history"].read_text()) == HISTORY

    # One PNG per evaluation, named by specimen/sweep/index. ``primary_plot``
    # is the first sweep with recorded spikes, ``plot`` the last one.
    assert paths["primary_plot"].name == "fit_specimen_2_sweep_9_1.png"
    assert paths["plot"] == paths["primary_plot"]
    assert (tmp_path / "report" / "fit_specimen_1_sweep_1_0.png").exists()


def test_save_fit_report_primary_plot_falls_back_to_last(tmp_path):
    """Without any recorded spike the last figure is also the primary one."""
    model = torch.nn.Linear(1, 1)
    evals = [_evaluation(n_spikes=0, sweep_number=i) for i in range(2)]
    paths = save_fit_report(model, evals, {}, [], output_dir=tmp_path)
    assert paths["primary_plot"].name.endswith("_1.png")
    assert paths["primary_plot"] == paths["plot"]


def test_save_fit_report_without_evaluations_points_to_directory(tmp_path):
    """No evaluations: JSON is still written and plot entries are the dir."""
    model = torch.nn.Linear(1, 1)
    paths = save_fit_report(model, [], {}, [], output_dir=tmp_path / "empty")
    assert paths["plot"] == paths["primary_plot"] == paths["output_dir"]
    assert json.loads(paths["history"].read_text()) == []
    assert not list((tmp_path / "empty").glob("*.png"))
