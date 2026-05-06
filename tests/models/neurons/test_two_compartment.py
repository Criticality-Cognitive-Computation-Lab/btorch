import torch

from btorch.analysis.two_compartment_fit import (
    AllenSweepBatch,
    evaluate_fit_across_sweeps,
    exponential_filter_spike_train,
    fit_two_compartment_model,
    mask_post_spike_voltage_samples,
    save_fit_report,
    spike_timing_stats,
    two_compartment_loss,
)
from btorch.models import environ
from btorch.models.functional import init_net_state, reset_net
from btorch.models.neurons.two_compartment import TwoCompartmentGLIF


def test_two_compartment_single_step_updates_voltage_and_apical_state():
    neuron = TwoCompartmentGLIF(
        n_neuron=2,
        w_Ca=2.0,
        theta_Ca=-0.5,
        w_sa=0.7,
        delta_th=1.5,
    )
    init_net_state(neuron, batch_size=3, dtype=torch.float32)

    i_soma = torch.full((3, 2), 50.0)
    i_apical = torch.full((3, 2), 1.5)

    with environ.context(dt=1.0):
        spike, voltage, state = neuron.single_step_forward(
            i_soma,
            i_apical,
            return_state=True,
        )

    assert spike.shape == (3, 2)
    assert voltage.shape == (3, 2)
    assert state["i_a"].shape == (3, 2)
    assert torch.all(state["i_a"] > 0.0)
    assert torch.equal(neuron.i_a, state["i_a"])
    assert state["theta_th"].shape == (3, 2)
    assert torch.all(state["v_threshold_eff"] >= neuron.v_threshold)


def test_two_compartment_reset_restores_deterministic_rollout():
    torch.manual_seed(4)
    neuron = TwoCompartmentGLIF(n_neuron=1, step_mode="m")
    i_soma = torch.randn(12, 2, 1)
    i_apical = torch.randn(12, 2, 1)

    init_net_state(neuron, batch_size=2, dtype=i_soma.dtype)
    with environ.context(dt=1.0):
        first_spike, first_v = neuron.multi_step_forward(i_soma, i_apical)
        continued_spike, continued_v = neuron.multi_step_forward(i_soma, i_apical)

        reset_net(neuron, batch_size=2)
        repeated_spike, repeated_v = neuron.multi_step_forward(i_soma, i_apical)

    assert not torch.allclose(first_v, continued_v)
    torch.testing.assert_close(first_spike, repeated_spike, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(first_v, repeated_v, atol=1e-6, rtol=0.0)


def test_two_compartment_dynamic_threshold_accumulates_after_spike():
    neuron = TwoCompartmentGLIF(
        n_neuron=1,
        tau_s=5.0,
        R_s=1.0,
        E_L=-70.0,
        v_threshold=-55.0,
        v_reset=-70.0,
        tau_th=20.0,
        delta_th=5.0,
    )
    init_net_state(neuron, batch_size=1, dtype=torch.float32)

    with environ.context(dt=1.0):
        spike_1, _, state_1 = neuron.single_step_forward(
            torch.full((1, 1), 200.0),
            torch.zeros(1, 1),
            return_state=True,
        )
        spike_2, _, state_2 = neuron.single_step_forward(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            return_state=True,
        )

    assert float(spike_1.item()) > 0.0
    assert float(state_1["theta_th"].item()) > 0.0
    assert float(state_2["v_threshold_eff"].item()) > float(neuron.v_threshold.item())
    assert float(spike_2.item()) == 0.0


def test_two_compartment_exponential_initiation_boosts_pre_spike_voltage():
    base = TwoCompartmentGLIF(
        n_neuron=1,
        tau_s=10.0,
        R_s=0.2,
        E_L=-70.0,
        v_threshold=-50.0,
        v_reset=-65.0,
        delta_T=0.0,
    )
    boosted = TwoCompartmentGLIF(
        n_neuron=1,
        tau_s=10.0,
        R_s=0.2,
        E_L=-70.0,
        v_threshold=-50.0,
        v_reset=-65.0,
        delta_T=3.0,
    )
    init_net_state(base, batch_size=1, dtype=torch.float32)
    init_net_state(boosted, batch_size=1, dtype=torch.float32)
    base.v = torch.full_like(base.v, -51.0)
    boosted.v = torch.full_like(boosted.v, -51.0)

    with environ.context(dt=1.0):
        _, _, base_state = base.single_step_forward(
            torch.full((1, 1), 10.0),
            torch.zeros(1, 1),
            return_state=True,
        )
        _, _, boosted_state = boosted.single_step_forward(
            torch.full((1, 1), 10.0),
            torch.zeros(1, 1),
            return_state=True,
        )

    assert float(boosted_state["v_pre_spike"].item()) > float(
        base_state["v_pre_spike"].item()
    )


def test_two_compartment_loss_masks_post_spike_samples_and_regularizes_w_ca():
    v_true = torch.tensor([0.0, 1.0, 10.0, 0.8, 0.2]).view(5, 1, 1)
    v_pred = torch.tensor([0.0, 1.2, -10.0, 0.6, 0.1]).view(5, 1, 1)
    spike_true = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]).view(5, 1, 1)
    spike_pred = spike_true.clone()

    losses = two_compartment_loss(
        v_pred=v_pred,
        spike_pred=spike_pred,
        v_true=v_true,
        spike_true=spike_true,
        dt_ms=1.0,
        w_Ca=torch.tensor([2.0]),
        post_spike_mask_ms=2.0,
        sparsity_weight=0.5,
    )
    mask = mask_post_spike_voltage_samples(spike_true, refractory_bins=2)

    assert mask.squeeze(-1).squeeze(-1).tolist() == [True, False, False, False, True]
    expected_voltage = torch.mean((v_pred[mask] - v_true[mask]) ** 2)
    torch.testing.assert_close(losses["voltage"], expected_voltage)
    torch.testing.assert_close(losses["spike_timing"], torch.tensor(0.0))
    torch.testing.assert_close(losses["sparsity"], torch.tensor(2.0))


def test_exponential_filter_spike_train_is_causal_and_stable():
    spikes = torch.tensor([0.0, 1.0, 0.0, 0.0]).view(4, 1, 1)
    filtered = exponential_filter_spike_train(spikes, tau_ms=2.0, dt_ms=1.0)

    assert filtered.shape == spikes.shape
    assert filtered[0].item() == 0.0
    assert filtered[1].item() > 0.0
    assert filtered[2].item() < filtered[1].item()


def test_spike_timing_stats_reward_close_matches():
    spike_true = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).view(6, 1, 1)
    spike_pred = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 1.0]).view(6, 1, 1)

    stats = spike_timing_stats(
        spike_true,
        spike_pred,
        dt_ms=1.0,
        match_window_ms=1.5,
    )

    assert stats["matched_spikes"] == 2.0
    assert stats["false_positive_spikes"] == 0.0
    assert stats["false_negative_spikes"] == 0.0
    assert stats["timing_f1"] == 1.0
    assert stats["mean_timing_error_ms"] == 0.5


def test_tbptt_fit_loop_runs_on_synthetic_sweep():
    torch.manual_seed(9)
    model = TwoCompartmentGLIF(
        n_neuron=1,
        trainable_param={"w_Ca", "w_as", "tau_s"},
    )
    sweep = AllenSweepBatch(
        specimen_id=1,
        sweep_number=1,
        dt_ms=1.0,
        i_soma=torch.randn(16, 1, 1),
        v_true=torch.zeros(16, 1, 1),
        spike_true=torch.zeros(16, 1, 1),
        i_apical=torch.zeros(16, 1, 1),
        metadata={"synthetic": True},
    )

    history = fit_two_compartment_model(
        model,
        [sweep],
        method="tbptt",
        lr=1e-3,
        epochs=1,
        chunk_size=5,
    )

    assert len(history) == 4
    assert all("total_loss" in row for row in history)


def test_global_fit_improves_tau_s_from_poor_initialization():
    torch.manual_seed(7)
    target_model = TwoCompartmentGLIF(
        n_neuron=1,
        tau_s=30.0,
        R_s=1.0,
        E_L=-70.0,
        tau_a=120.0,
        tau_th=50.0,
        delta_th=0.0,
        delta_T=0.0,
        w_Ca=0.0,
        theta_Ca=2.0,
        w_sa=0.0,
        w_as=0.0,
        v_threshold=1000.0,
        step_mode="m",
    )
    init_net_state(target_model, batch_size=1, dtype=torch.float32)

    i_soma = torch.linspace(0.0, 25.0, 20).view(20, 1, 1)
    i_apical = torch.zeros_like(i_soma)
    with environ.context(dt=1.0):
        spike_true, v_true = target_model.multi_step_forward(i_soma, i_apical)

    fit_model = TwoCompartmentGLIF(
        n_neuron=1,
        tau_s=5.0,
        R_s=1.0,
        E_L=-70.0,
        tau_a=120.0,
        tau_th=50.0,
        delta_th=0.0,
        delta_T=0.0,
        w_Ca=0.0,
        theta_Ca=2.0,
        w_sa=0.0,
        w_as=0.0,
        v_threshold=1000.0,
        step_mode="m",
        trainable_param={"tau_s"},
    )
    sweep = AllenSweepBatch(
        specimen_id=1,
        sweep_number=1,
        dt_ms=1.0,
        i_soma=i_soma,
        v_true=v_true.detach(),
        spike_true=spike_true.detach(),
        i_apical=i_apical,
        metadata={"synthetic": True},
    )

    history = fit_two_compartment_model(
        fit_model,
        [sweep],
        method="global",
        global_maxiter=4,
        global_popsize=4,
        local_maxiter=15,
        param_bounds={"tau_s": (1.0, 60.0)},
        seed=0,
    )

    fitted_tau_s = float(fit_model.tau_s.detach().cpu())
    assert history
    assert abs(fitted_tau_s - 30.0) < abs(5.0 - 30.0)


def test_staged_fit_runs_with_mixed_sweeps():
    torch.manual_seed(3)
    target_model = TwoCompartmentGLIF(
        n_neuron=1,
        tau_s=25.0,
        R_s=0.15,
        E_L=-72.0,
        tau_a=120.0,
        tau_th=50.0,
        delta_th=0.0,
        delta_T=0.0,
        w_Ca=0.0,
        theta_Ca=3.0,
        w_sa=0.0,
        w_as=0.0,
        v_threshold=-45.0,
        v_reset=-60.0,
        step_mode="m",
    )
    init_net_state(target_model, batch_size=1, dtype=torch.float32)

    silent_i = torch.full((20, 1, 1), 10.0)
    spiking_i = torch.full((20, 1, 1), 220.0)
    with environ.context(dt=1.0):
        silent_spike, silent_v = target_model.multi_step_forward(
            silent_i,
            torch.zeros_like(silent_i),
        )
        reset_net(target_model, batch_size=1)
        spiking_spike, spiking_v = target_model.multi_step_forward(
            spiking_i,
            torch.zeros_like(spiking_i),
        )

    fit_model = TwoCompartmentGLIF(
        n_neuron=1,
        tau_s=10.0,
        R_s=0.05,
        E_L=-68.0,
        tau_a=120.0,
        tau_th=50.0,
        delta_th=0.0,
        delta_T=2.0,
        w_Ca=0.0,
        theta_Ca=3.0,
        w_sa=0.0,
        w_as=0.0,
        v_threshold=-40.0,
        v_reset=-55.0,
        step_mode="m",
        trainable_param={
            "tau_s",
            "R_s",
            "E_L",
            "v_threshold",
            "v_reset",
            "tau_th",
            "delta_th",
            "delta_T",
        },
    )
    sweeps = [
        AllenSweepBatch(
            specimen_id=1,
            sweep_number=1,
            dt_ms=1.0,
            i_soma=silent_i,
            v_true=silent_v.detach(),
            spike_true=silent_spike.detach(),
            i_apical=torch.zeros_like(silent_i),
        ),
        AllenSweepBatch(
            specimen_id=1,
            sweep_number=2,
            dt_ms=1.0,
            i_soma=spiking_i,
            v_true=spiking_v.detach(),
            spike_true=spiking_spike.detach(),
            i_apical=torch.zeros_like(spiking_i),
        ),
    ]

    history = fit_two_compartment_model(
        fit_model,
        sweeps,
        method="staged",
        global_maxiter=2,
        global_popsize=3,
        local_maxiter=5,
        seed=0,
    )

    assert history
    assert any(str(row["phase"]).startswith("threshold_reset:") for row in history)


def test_fit_evaluation_and_report_outputs(tmp_path):
    model = TwoCompartmentGLIF(
        n_neuron=1,
        tau_s=20.0,
        R_s=1.0,
        E_L=-70.0,
        tau_a=120.0,
        tau_th=50.0,
        delta_th=0.0,
        delta_T=0.0,
        w_Ca=0.0,
        theta_Ca=1.0,
        w_sa=0.0,
        w_as=0.0,
        v_threshold=1000.0,
        step_mode="m",
    )
    init_net_state(model, batch_size=1, dtype=torch.float32)

    i_soma = torch.ones(12, 1, 1)
    i_apical = torch.zeros_like(i_soma)
    with environ.context(dt=1.0):
        spike_true, v_true = model.multi_step_forward(i_soma, i_apical)

    sweep = AllenSweepBatch(
        specimen_id=11,
        sweep_number=7,
        dt_ms=1.0,
        i_soma=i_soma,
        v_true=v_true.detach(),
        spike_true=spike_true.detach(),
        i_apical=i_apical,
    )
    history = [
        {
            "phase": "global",
            "epoch": 0.0,
            "specimen_id": 11.0,
            "sweep_number": 7.0,
            "chunk_start": -1.0,
            "total_loss": 0.0,
            "voltage_loss": 0.0,
            "spike_loss": 0.0,
            "sparsity_loss": 0.0,
        }
    ]

    evaluations, aggregate = evaluate_fit_across_sweeps(model, [sweep])
    paths = save_fit_report(
        model,
        evaluations,
        aggregate,
        history,
        output_dir=tmp_path,
    )

    assert aggregate["mean_total_loss"] == 0.0
    assert paths["parameters"].exists()
    assert paths["metrics"].exists()
    assert paths["history"].exists()
    assert paths["plot"].exists()
