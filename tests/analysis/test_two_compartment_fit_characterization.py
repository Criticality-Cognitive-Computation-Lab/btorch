"""Characterization tests for the two-compartment fitting helpers.

These tests pin the numerical behaviour of the loss, the per-sweep
evaluation helper and the public fit entry points on tiny synthetic
data, so internal refactors (config objects, shared helpers, module
splits) can be verified to be behaviour preserving.
"""

import pytest
import torch

import btorch.analysis.two_compartment_fit as tcf
from btorch.analysis.two_compartment_fit import (
    AllenSweepBatch,
    evaluate_fit_across_sweeps,
    fit_two_compartment_model,
    plot_two_compartment_fit,
    save_fit_report,
    two_compartment_loss,
)
from btorch.models.neurons.two_compartment import TwoCompartmentGLIF


def _tiny_traces():
    """Build (T=6, B=1, N=1) traces with one true spike and one predicted."""
    v_true = torch.tensor([0.0, 1.0, 10.0, 0.5, 0.2, 0.1]).view(6, 1, 1)
    v_pred = torch.tensor([0.1, 0.8, 9.0, 0.7, 0.1, 0.0]).view(6, 1, 1)
    spike_true = torch.zeros(6, 1, 1)
    spike_true[2] = 1.0
    spike_pred = torch.zeros(6, 1, 1)
    spike_pred[3] = 1.0  # one-bin timing offset
    return v_pred, spike_pred, v_true, spike_true


def _sweep(seed: int = 0, steps: int = 12, spike: bool = False) -> AllenSweepBatch:
    g = torch.Generator().manual_seed(seed)
    spike_true = torch.zeros(steps, 1, 1)
    if spike:
        spike_true[4] = 1.0
    return AllenSweepBatch(
        specimen_id=1,
        sweep_number=seed,
        dt=1.0,
        i_soma=torch.randn(steps, 1, 1, generator=g),
        v_true=torch.zeros(steps, 1, 1),
        spike_true=spike_true,
        i_apical=torch.zeros(steps, 1, 1),
        metadata={},
    )


def test_loss_total_is_weighted_sum_of_components():
    v_pred, spike_pred, v_true, spike_true = _tiny_traces()
    kwargs = dict(
        v_pred=v_pred,
        spike_pred=spike_pred,
        v_true=v_true,
        spike_true=spike_true,
        dt=1.0,
        w_Ca=torch.tensor([2.0]),
    )
    losses = two_compartment_loss(
        **kwargs,
        voltage_weight=0.5,
        spike_weight=2.0,
        spike_count_weight=0.3,
        spike_timing_weight=0.1,
        sparsity_weight=0.25,
    )
    expected = (
        0.5 * losses["voltage"]
        + 2.0 * losses["spike"]
        + 0.3 * losses["spike_count"]
        + 0.1 * losses["spike_timing"]
        + 0.25 * losses["sparsity"]
    )
    torch.testing.assert_close(losses["total"], expected)
    # |w_Ca| mean is 2.0, and one predicted spike is one bin off the truth.
    torch.testing.assert_close(losses["sparsity"], torch.tensor(2.0))
    assert losses["spike_timing"].item() == pytest.approx(1.0)
    # Count is matched (1 vs 1) so the count loss vanishes.
    assert losses["spike_count"].item() == 0.0
    # Default weights: voltage=1, spike=1, sparsity=1e-4, others 0.
    default = two_compartment_loss(**kwargs)
    torch.testing.assert_close(
        default["total"],
        default["voltage"] + default["spike"] + 1e-4 * default["sparsity"],
    )


def test_loss_rejects_nonpositive_count_weights():
    v_pred, spike_pred, v_true, spike_true = _tiny_traces()
    with pytest.raises(ValueError):
        two_compartment_loss(
            v_pred=v_pred,
            spike_pred=spike_pred,
            v_true=v_true,
            spike_true=spike_true,
            dt=1.0,
            spike_count_over_weight=0.0,
        )


def test_evaluate_fit_across_sweeps_is_deterministic():
    model = TwoCompartmentGLIF(n_neuron=1)
    sweeps = [_sweep(0), _sweep(1, spike=True)]
    _, agg1 = evaluate_fit_across_sweeps(model, sweeps, spike_count_weight=0.1)
    _, agg2 = evaluate_fit_across_sweeps(model, sweeps, spike_count_weight=0.1)
    assert agg1 == agg2
    assert agg1["n_sweeps"] == 2.0
    assert agg1["n_silent_sweeps"] == 1.0
    assert agg1["n_spiking_sweeps"] == 1.0


def test_fit_sweeps_once_matches_manual_loss_average():
    """`_fit_sweeps_once` averages the per-sweep loss terms."""
    torch.manual_seed(0)
    model = TwoCompartmentGLIF(n_neuron=1)
    sweeps = [_sweep(0), _sweep(1, spike=True)]
    metrics = tcf._fit_sweeps_once(
        model, sweeps, loss=tcf.FitLossConfig(voltage_weight=0.3)
    )

    totals = []
    for sw in sweeps:
        ev, _ = evaluate_fit_across_sweeps(model, [sw])
        rollout_losses = ev[0].metrics
        totals.append(rollout_losses)
    assert set(metrics) == {
        "total_loss",
        "voltage_loss",
        "spike_loss",
        "spike_count_loss",
        "spike_timing_loss",
        "sparsity_loss",
    }
    # Voltage loss is weight independent, so it must equal the evaluation one.
    mean_voltage = sum(m["voltage_loss"] for m in totals) / len(totals)
    assert metrics["voltage_loss"] == pytest.approx(mean_voltage, rel=1e-5)
    with pytest.raises(ValueError):
        tcf._fit_sweeps_once(model, [], loss=tcf.FitLossConfig())


def test_fit_sweeps_once_weights_change_total_only():
    model = TwoCompartmentGLIF(n_neuron=1)
    sweeps = [_sweep(2, spike=True)]
    a = tcf._fit_sweeps_once(model, sweeps, loss=tcf.FitLossConfig())
    b = tcf._fit_sweeps_once(model, sweeps, loss=tcf.FitLossConfig(voltage_weight=0.0))
    assert a["voltage_loss"] == pytest.approx(b["voltage_loss"])
    assert a["total_loss"] - b["total_loss"] == pytest.approx(a["voltage_loss"])


@pytest.mark.parametrize("method", ["tbptt", "global", "hybrid", "staged"])
def test_fit_entry_point_history_schema(method):
    torch.manual_seed(3)
    model = TwoCompartmentGLIF(n_neuron=1, trainable_param={"w_Ca", "w_as", "tau_s"})
    history = fit_two_compartment_model(
        model,
        [_sweep(0), _sweep(1, spike=True)],
        method=method,
        epochs=1,
        chunk_size=6,
        global_maxiter=1,
        global_popsize=4,
        local_maxiter=1,
        spike_count_weight=0.1,
        spike_timing_weight=0.1,
    )
    assert history
    for row in history:
        assert "total_loss" in row and "phase" in row
    phases = {row["phase"] for row in history}
    if method == "tbptt":
        assert phases == {"tbptt"} and len(history) == 4
    if method == "global":
        assert phases == {"global", "local"} and len(history) == 2
    if method == "hybrid":
        assert phases == {"global", "local", "tbptt"}


def test_plot_and_report_reexports(tmp_path):
    """Plot helpers stay importable from the fit module and write files."""
    model = TwoCompartmentGLIF(n_neuron=1)
    evals, agg = evaluate_fit_across_sweeps(model, [_sweep(0, spike=True)])
    history = [{"total_loss": 1.0, "voltage_loss": 0.5, "spike_loss": 0.2}]
    out = plot_two_compartment_fit(evals[0], history, output_path=tmp_path / "a.png")
    assert out.exists()
    paths = save_fit_report(model, evals, agg, history, output_dir=tmp_path / "r")
    assert paths["plot"].exists() and paths["metrics"].exists()


def test_fit_loss_config_matches_public_loss_kwargs():
    """The private config path and the public keyword path agree exactly."""
    v_pred, spike_pred, v_true, spike_true = _tiny_traces()
    cfg = tcf.FitLossConfig(
        voltage_weight=0.4, spike_count_weight=0.2, spike_timing_weight=0.3
    )
    via_config = tcf._loss_from_config(
        v_pred=v_pred,
        spike_pred=spike_pred,
        v_true=v_true,
        spike_true=spike_true,
        dt=1.0,
        w_Ca=None,
        config=cfg,
    )
    via_kwargs = two_compartment_loss(
        v_pred=v_pred,
        spike_pred=spike_pred,
        v_true=v_true,
        spike_true=spike_true,
        dt=1.0,
        voltage_weight=0.4,
        spike_count_weight=0.2,
        spike_timing_weight=0.3,
    )
    for key, value in via_kwargs.items():
        torch.testing.assert_close(via_config[key], value)
    # Config defaults equal the public keyword defaults.
    assert tcf.FitLossConfig().sparsity_weight == 1e-4


def test_prepare_sweep_converts_dtype_and_resets_state():
    model = TwoCompartmentGLIF(n_neuron=1)
    sweep = _sweep(0)
    i_soma, v_true, spike_true, i_apical = tcf._prepare_sweep(
        model, sweep, device="cpu", dtype=torch.float64
    )
    assert i_soma.dtype == v_true.dtype == spike_true.dtype == torch.float64
    assert i_apical is not None and i_apical.dtype == torch.float64
    assert model.v.shape[0] == i_soma.shape[1]
