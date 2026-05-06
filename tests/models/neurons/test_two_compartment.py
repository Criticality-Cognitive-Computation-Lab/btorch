import matplotlib.pyplot as plt
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
from btorch.models.linear import DenseConn
from btorch.models.neurons.glif import GLIF3
from btorch.models.neurons.mixed import MixedNeuronPopulation
from btorch.models.neurons.two_compartment import TwoCompartmentGLIF
from btorch.models.rnn import ApicalRecurrentNN
from btorch.models.synapse import AlphaPSC
from btorch.utils.file import save_fig


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def test_draw_glif3_vs_two_compartment():
    """Visual comparison of GLIF3 and TwoCompartmentGLIF via ApicalRecurrentNN.

    A single ApicalRecurrentNN wraps a MixedNeuronPopulation (1 GLIF3 + 1 TC)
    with a shared recurrent synapse (weights zeroed so dynamics are input-driven).

    Input layout (last dim):  index 0 → GLIF3,  index 1 → TwoCompartmentGLIF
    Apical layout (last dim): index 0 → ignored, index 1 → TC apical input

    Four stimulus phases reveal TC-exclusive features vs. the GLIF3 baseline:

      Phase 1 (silence):      both neurons quiescent — establishes baseline.
      Phase 2 (soma only):    both neurons fire at comparable baseline rates.
      Phase 3 (soma+apical):  apical pulse triggers Ca²⁺ plateau in TC (I_a
                              crosses θ_Ca), boosting TC firing via w_as coupling.
                              GLIF3 is unaffected.
      Phase 4 (soma only):    apical pulse removed — TC I_a stays in the high
                              plateau state (bistable self-sustaining), so TC
                              continues firing faster than in phase 2.  GLIF3
                              immediately returns to baseline.

    The five-row plot captures:
      Row 0 — external inputs (soma step + apical pulse)
      Row 1 — spike rasters + per-phase firing rates
      Row 2 — somatic membrane voltage
      Row 3 — GLIF3: after-spike current (Iasc) | TC: slow apical current I_a
               (dashed line at θ_Ca marks the plateau threshold)
      Row 4 — GLIF3: N/A              | TC: back-propagating AP current I_bap
               (each somatic spike injects a delayed current into the apical
               compartment, visible as brief pulses aligned with spikes)

    Saved to fig/tests/models/neurons/test_two_compartment/.
    """
    dt = 1.0
    # Stimulus phases (ms)
    t_silence = 100
    t_soma_only_1 = 300  # soma-only baseline
    t_apical = 150  # soma + apical (plateau trigger)
    t_soma_only_2 = 350  # soma-only again (plateau sustained)
    t_trail = 100
    T = t_silence + t_soma_only_1 + t_apical + t_soma_only_2 + t_trail
    time = torch.arange(0, T, dt)
    n_total = 2  # 1 GLIF3 + 1 TwoCompartmentGLIF

    glif = GLIF3(
        n_neuron=1,
        v_threshold=-45.0,
        v_reset=-65.0,
        c_m=200.0,
        tau=20.0,
        k=[0.05],
        asc_amps=[-50],
        tau_ref=2.0,
        step_mode="s",
        device=DEVICE,
    )
    # TwoCompartmentGLIF parameters chosen to exhibit clear bistability:
    #   Low plateau state:  I_a ≈ 0.07  (w_Ca * sigmoid(-θ_Ca) ≈ 0)
    #   High plateau state: I_a ≈ 9.9   (self-sustained by w_Ca term)
    #   Unstable boundary:  I_a = θ_Ca = 5.0
    # A 5-unit apical pulse for 150 ms pushes I_a from low → past threshold →
    # high state, which then persists after the pulse is removed.
    tc = TwoCompartmentGLIF(
        n_neuron=1,
        v_threshold=-50.0,
        v_reset=-65.0,
        tau_s=20.0,
        R_s=1.0,
        E_L=-70.0,
        tau_a=80.0,
        w_Ca=10.0,
        theta_Ca=5.0,
        w_sa=1.0,
        w_as=2.5,
        step_mode="s",
        device=DEVICE,
    )
    mixed = MixedNeuronPopulation({"glif": (1, glif), "tc": (1, tc)}, step_mode="s")
    conn = DenseConn(n_total, n_total, bias=None, device=DEVICE)
    psc = AlphaPSC(n_neuron=n_total, tau_syn=5.0, linear=conn, step_mode="s")
    with torch.no_grad():
        conn.weight.zero_()
    rnn = ApicalRecurrentNN(neuron=mixed, synapse=psc, step_mode="s")
    init_net_state(rnn, batch_size=1, dtype=torch.float32, device=DEVICE)

    # Build stimulus tensors for each phase
    def _phase(*pairs):
        return torch.cat([torch.full((n,), v) for n, v in pairs])

    glif_drive = _phase(
        (t_silence, 0.0),
        (t_soma_only_1, 250.0),
        (t_apical, 250.0),
        (t_soma_only_2, 250.0),
        (t_trail, 0.0),
    ).to(DEVICE)
    tc_drive = _phase(
        (t_silence, 0.0),
        (t_soma_only_1, 30.0),
        (t_apical, 30.0),
        (t_soma_only_2, 30.0),
        (t_trail, 0.0),
    ).to(DEVICE)
    tc_apical = _phase(
        (t_silence, 0.0),
        (t_soma_only_1, 0.0),
        (t_apical, 5.0),  # strong pulse: pushes I_a past bistable threshold
        (t_soma_only_2, 0.0),
        (t_trail, 0.0),
    ).to(DEVICE)

    glif_spikes, glif_v, glif_iasc = [], [], []
    tc_spikes, tc_v, tc_ia, tc_ibap = [], [], [], []

    with torch.no_grad():
        with environ.context(dt=dt):
            for i_g, i_t, i_a in zip(glif_drive, tc_drive, tc_apical):
                x = torch.stack([i_g, i_t]).view(1, n_total)
                x_a = torch.stack([torch.zeros_like(i_a), i_a]).view(1, n_total)

                spikes, _ = rnn.single_step_forward(x, x_apical=x_a)

                glif_spikes.append(spikes[0, 0].cpu())
                tc_spikes.append(spikes[0, 1].cpu())
                glif_v.append(rnn.neuron.glif.v[0, 0].cpu())
                tc_v.append(rnn.neuron.tc.v[0, 0].cpu())
                glif_iasc.append(
                    rnn.neuron.glif.Iasc[0, 0, 0].cpu()
                    if hasattr(rnn.neuron.glif, "Iasc")
                    else torch.tensor(0.0)
                )
                tc_ia.append(rnn.neuron.tc.i_a[0, 0].cpu())
                tc_ibap.append(rnn.neuron.tc.i_bap[0, 0].cpu())

    glif_spikes = torch.stack(glif_spikes)
    tc_spikes = torch.stack(tc_spikes)
    glif_v = torch.stack(glif_v)
    tc_v = torch.stack(tc_v)
    glif_iasc = torch.stack(glif_iasc)
    tc_ia = torch.stack(tc_ia)
    tc_ibap = torch.stack(tc_ibap)

    # --- Plot ---
    t = time.numpy()
    # Phase boundary x-positions for shading
    p1 = t_silence
    p2 = p1 + t_soma_only_1
    p3 = p2 + t_apical
    p4 = p3 + t_soma_only_2
    phase_labels = [
        (0, p1, "silence"),
        (p1, p2, "soma only"),
        (p2, p3, "soma+apical"),
        (p3, p4, "soma only\n(plateau sustained)"),
        (p4, T, "silence"),
    ]
    phase_colors = ["white", "#e8f4f8", "#fde8d8", "#e8f4e8", "white"]

    fig, axes = plt.subplots(5, 2, figsize=(14, 13), sharex=True)

    def _shade(ax):
        for (x0, x1, label), color in zip(phase_labels, phase_colors):
            ax.axvspan(x0, x1, alpha=0.25, color=color, linewidth=0)

    # Phase shading on all rows; labels placed inside the top row axes
    for ax in axes.flat:
        _shade(ax)
    for (x0, x1, label), color in zip(phase_labels, phase_colors):
        mid = (x0 + x1) / 2
        for ax in axes[0]:
            ax.text(
                mid,
                0.97,
                label,
                ha="center",
                va="top",
                fontsize=6,
                transform=ax.get_xaxis_transform(),
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
            )

    # Row 0: input currents
    axes[0, 0].plot(t, glif_drive.cpu().numpy(), color="steelblue")
    axes[0, 0].set_ylabel("I_soma (pA)")
    axes[0, 0].set_title("GLIF3  (point neuron)")

    axes[0, 1].plot(t, tc_drive.cpu().numpy(), color="steelblue", label="soma")
    axes[0, 1].plot(t, tc_apical.cpu().numpy(), color="coral", label="apical")
    axes[0, 1].set_ylabel("Input current")
    axes[0, 1].set_title("TwoCompartmentGLIF  (soma + apical compartment)")
    axes[0, 1].legend(fontsize=7)

    # Per-phase firing rates annotated on spike raster
    phase_edges = [0, p1, p2, p3, p4, T]

    # Row 1: spikes
    for col, (ax, spk) in enumerate(
        [(axes[1, 0], glif_spikes), (axes[1, 1], tc_spikes)]
    ):
        nz = spk.nonzero(as_tuple=False)
        if nz.numel() > 0:
            ax.scatter(
                t[nz[:, 0]],
                torch.ones(nz.shape[0]).numpy(),
                marker="|",
                linewidths=0.8,
                color="k",
                s=50,
            )
        ax.set_ylabel("Spikes")
        ax.set_ylim(0, 2)
        for i in range(len(phase_edges) - 1):
            a, b = phase_edges[i], phase_edges[i + 1]
            dur = (b - a) * dt / 1000  # s
            n_spk = spk[a:b].sum().item()
            fr = n_spk / dur if dur > 0 else 0.0
            ax.text(
                (a + b) / 2,
                1.7,
                f"{fr:.0f} Hz",
                ha="center",
                va="center",
                fontsize=6,
                color="navy",
            )

    # Row 2: somatic voltage
    for ax, v, thr in [
        (axes[2, 0], glif_v, -45.0),
        (axes[2, 1], tc_v, -50.0),
    ]:
        ax.plot(t, v.numpy(), linewidth=0.7, color="steelblue")
        ax.axhline(thr, color="r", linestyle="--", linewidth=0.7, label=f"V_th={thr}")
        ax.set_ylabel("V_soma (mV)")
        ax.legend(fontsize=7)

    # Row 3: GLIF3 after-spike current | TC slow apical current I_a
    axes[3, 0].plot(t, glif_iasc.numpy(), linewidth=0.8, color="olive")
    axes[3, 0].set_ylabel("After-spike current\n(Iasc, pA)")

    axes[3, 1].plot(t, tc_ia.numpy(), linewidth=0.8, color="coral")
    axes[3, 1].axhline(
        float(tc.theta_Ca.item()),
        color="purple",
        linestyle="--",
        linewidth=0.8,
        label=f"θ_Ca={float(tc.theta_Ca.item()):.0f} (plateau threshold)",
    )
    axes[3, 1].set_ylabel("Apical current I_a\n(slow, bistable)")
    axes[3, 1].legend(fontsize=7)

    # Row 4: GLIF3 N/A | TC back-propagating AP current I_bap
    axes[4, 0].axis("off")
    axes[4, 0].text(
        0.5,
        0.5,
        "N/A\n(point neuron has no apical compartment)",
        ha="center",
        va="center",
        transform=axes[4, 0].transAxes,
        fontsize=9,
        color="gray",
        style="italic",
    )

    axes[4, 1].plot(t, tc_ibap.numpy(), linewidth=0.8, color="darkorange")
    axes[4, 1].set_ylabel("bAP current I_bap\n(spike→apical feedback)")
    axes[4, 1].set_xlabel("Time (ms)")
    axes[2, 0].set_xlabel("")
    axes[3, 0].set_xlabel("")

    fig.suptitle(
        "GLIF3 vs TwoCompartmentGLIF  ·  ApicalRecurrentNN + MixedNeuronPopulation\n"
        "Phase 3 apical pulse triggers bistable Ca\u00b2\u207a plateau in TC"
        " that persists in Phase 4",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, name="glif3_vs_two_compartment")
    plt.close("all")
