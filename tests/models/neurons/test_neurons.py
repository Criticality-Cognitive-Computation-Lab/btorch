import matplotlib.pyplot as plt
import pytest
import torch

from btorch.fitting.tuning import compute_fi_vi_curve
from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.neurons.alif import ALIF, ELIF
from btorch.models.neurons.glif import GLIF3
from btorch.utils.file import save_fig
from btorch.visualisation.tuning import plot_fi_vi_curve


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DT = 1.0
T = 1000
TIME = torch.arange(0, T, DT)


@pytest.fixture(scope="module")
def time_axis():
    return DT, TIME


def _simulate(neuron, stimulus, dt: float):
    """Run a single-neuron simulation and collect spikes, voltage, and
    adaptation."""
    traces = {"spike": [], "v": [], "adapt": []}
    with torch.no_grad():
        with environ.context(dt=float(dt)):
            for current in stimulus:
                step_input = current.expand(neuron.n_neuron)
                spike = neuron(step_input)
                traces["spike"].append(spike.detach().cpu())
                traces["v"].append(neuron.v.detach().cpu())

                adapt = None
                if hasattr(neuron, "Iasc"):
                    adapt = neuron.Iasc
                elif hasattr(neuron, "g_k"):
                    adapt = neuron.g_k
                traces["adapt"].append(
                    adapt.detach().cpu() if adapt is not None else None
                )

    traces["spike"] = torch.stack(traces["spike"])
    traces["v"] = torch.stack(traces["v"])
    # Some neurons (e.g., GLIF) have multiple adaptation channels; keep the first.
    adapt_stack = [
        torch.zeros_like(traces["v"][0])
        if a is None
        else (a if a.ndim == 1 else a[..., 0])
        for a in traces["adapt"]
    ]
    traces["adapt"] = torch.stack(adapt_stack)
    return traces


def _plot(time, traces, v_threshold: float, title: str, adapt_label: str, name: str):
    spikes = traces["spike"].squeeze(-1)
    v = traces["v"].squeeze(-1)
    adapt = traces["adapt"].squeeze(-1)

    firing_rate = spikes.sum() / len(time) * 1000

    fig, axes = plt.subplots(3, 1, sharex=True)

    spk_nz = spikes.nonzero(as_tuple=False)
    # assert False
    axes[0].scatter(
        time[spk_nz[:, 0]], torch.ones(spk_nz.shape[0]), marker="|", linewidths=0.8
    )
    axes[0].set_ylabel("Spikes")
    axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0].set_title(f"Firing Rate: {firing_rate.item():.1f} Hz")

    axes[1].plot(time, v, linewidth=0.9)
    axes[1].axhline(y=v_threshold, color="r", linestyle="--", label="Threshold")
    axes[1].set_ylabel("Membrane Potential")

    axes[2].plot(time, adapt, linewidth=0.9)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel(adapt_label)

    fig.suptitle(title)
    fig.legend()
    fig.tight_layout()
    save_fig(fig, name=name)


@pytest.mark.parametrize(
    "case",
    [
        {
            "name": "glif3_single_neuron",
            "title": "GLIF3 Neuron Dynamics",
            "adapt_label": "After-Spike Current",
            "v_threshold": -45.0,
            "v_reset": -65.0,
            # Deterministic simulation: 12 spikes while driven, none after.
            "n_spikes_driven": (8, 16),
            "n_spikes_after": (0, 0),
            "adapts": False,
            "stimulus": lambda steps: torch.cat(
                (torch.full((steps // 2,), 250.0), torch.zeros((steps // 2,)))
            ),
            "build": lambda: GLIF3(
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
            ),
        },
        {
            "name": "alif_single_neuron",
            "title": "ALIF Neuron Dynamics",
            "adapt_label": "Adaptation Conductance",
            "v_threshold": -50.0,
            "v_reset": -65.0,
            "n_spikes_driven": (13, 21),  # 17 observed
            "n_spikes_after": (1, 6),  # 3 observed: weaker drive, still firing
            "adapts": True,
            "stimulus": lambda steps: torch.cat(
                (torch.full((steps // 2,), 20.0), torch.full((steps // 2,), 8.0))
            ),
            "build": lambda: ALIF(
                n_neuron=1,
                v_threshold=-50.0,
                v_reset=-65.0,
                c_m=1.0,
                g_leak=0.05,
                E_leak=-70.0,
                E_k=-80.0,
                g_k_init=0.0,
                tau_adapt=250.0,
                dg_k=0.12,
                tau_ref=2.0,
                step_mode="s",
                device=DEVICE,
            ),
        },
        {
            "name": "elif_single_neuron",
            "title": "ELIF Neuron Dynamics",
            "adapt_label": "Adaptation Conductance",
            "v_threshold": -48.0,
            "v_reset": -65.0,
            "n_spikes_driven": (13, 23),  # 18 observed
            "n_spikes_after": (1, 9),  # 5 observed
            "adapts": True,
            "stimulus": lambda steps: torch.cat(
                (torch.full((steps // 2,), 12.0), torch.full((steps // 2,), 5.0))
            ),
            "build": lambda: ELIF(
                n_neuron=1,
                v_threshold=-48.0,
                v_reset=-65.0,
                c_m=1.0,
                g_leak=0.05,
                E_leak=-70.0,
                E_k=-80.0,
                g_k_init=0.0,
                tau_adapt=150.0,
                dg_k=0.1,
                tau_ref=2.0,
                delta_T=2.0,
                v_T=-55.0,
                step_mode="s",
                device=DEVICE,
            ),
        },
    ],
    ids=lambda case: case["name"],
)
def test_draw_single_neuron(case, time_axis):
    dt, time = time_axis
    neuron = case["build"]()
    init_net_state(neuron, device=DEVICE)

    stimulus = case["stimulus"](len(time))
    stimulus = stimulus.to(device=DEVICE, dtype=torch.float32)

    traces = _simulate(neuron, stimulus, dt=dt)
    _plot(
        time,
        traces,
        v_threshold=case["v_threshold"],
        title=case["title"],
        adapt_label=case["adapt_label"],
        name=case["name"],
    )
    plt.close("all")

    # ---- Assertions (all runs are deterministic; ranges leave a safety margin).
    spikes = traces["spike"].squeeze(-1)
    v = traces["v"].squeeze(-1)
    adapt = traces["adapt"].squeeze(-1)
    half = len(time) // 2
    tau_ref = 2.0

    # Numerical sanity: no NaN/Inf anywhere in the recorded traces.
    assert torch.isfinite(v).all()
    assert torch.isfinite(adapt).all()

    # Spikes are binary events.
    assert set(spikes.unique().tolist()) <= {0.0, 1.0}

    # Spike counts in the driven and the following (weaker / zero) phase.
    n_driven = int(spikes[:half].sum())
    n_after = int(spikes[half:].sum())
    assert case["n_spikes_driven"][0] <= n_driven <= case["n_spikes_driven"][1]
    assert case["n_spikes_after"][0] <= n_after <= case["n_spikes_after"][1]

    # Refractory period: consecutive spikes are at least tau_ref ms apart.
    spike_idx = spikes.nonzero().squeeze(-1)
    isi = spike_idx.diff() * dt
    assert (isi >= tau_ref).all()

    if not case["adapts"]:
        # GLIF3: stimulus off -> silent, and the after-spike current decays to ~0.
        assert n_after == 0
        assert adapt[-1].abs() < 1e-3
        # Reset: the step after a spike V is within a few mV of v_reset (far
        # below threshold), not at threshold.
        after_spike = v[spike_idx + 1]
        assert (after_spike - case["v_reset"]).abs().max() < 3.0
        # Constant drive -> regular firing: ISIs agree within 1 ms (skip the
        # first, which includes the initial charging from rest).
        driven_isi = isi[spike_idx[1:] < half]
        assert driven_isi[1:].max() - driven_isi[1:].min() <= 1
    else:
        # Adaptation conductance builds up under sustained firing.
        assert adapt.max() > 0.3
        assert adapt.min() >= 0.0
        # Spike-frequency adaptation: with a constant stimulus the rate in
        # the first 100 ms exceeds the rate in 400-500 ms (8 vs 2 observed).
        early = int(spikes[:100].sum())
        late = int(spikes[400:half].sum())
        assert early > late
        # ISIs lengthen: the last driven ISI is longer than the first.
        driven = spike_idx[spike_idx < half]
        assert (driven[-1] - driven[-2]) > (driven[1] - driven[0])


@pytest.mark.parametrize(
    "case",
    [
        {
            "name": "glif3_fi_vi_curve",
            "neuron_cls": GLIF3,
            "neuron_params": {
                "v_threshold": -45.0,
                "v_reset": -65.0,
                "c_m": 200.0,
                "tau": 20.0,
                "k": [0.05],
                "asc_amps": [-50],
                "tau_ref": 2.0,
                "step_mode": "s",
            },
            "current_start": 100.0,
            "current_end": 300.0,
            "steps": 20,
        },
        {
            "name": "alif_fi_vi_curve",
            "neuron_cls": ALIF,
            "neuron_params": {
                "v_threshold": -50.0,
                "v_reset": -65.0,
                "c_m": 1.0,
                "g_leak": 0.05,
                "E_leak": -70.0,
                "E_k": -80.0,
                "g_k_init": 0.0,
                "tau_adapt": 200.0,
                "dg_k": 0.12,
                "tau_ref": 2.0,
                "step_mode": "s",
            },
            "current_start": 0.0,
            "current_end": 30.0,
            "steps": 20,
        },
        {
            "name": "glif3_population_fi_vi",
            "neuron_cls": GLIF3,
            "neuron_params": {
                "n_neuron": 5,
                "v_threshold": -45.0,
                "v_reset": -65.0,
                "c_m": 200.0,
                "tau": 20.0,
                "k": [0.05],
                "asc_amps": [-50],
                "tau_ref": 2.0,
                "step_mode": "s",
            },
            "current_start": 100.0,
            "current_end": 300.0,
            "steps": 10,
        },
    ],
    ids=lambda case: case["name"],
)
def test_plot_fi_vi_curves(case):
    fig = plot_fi_vi_curve(
        get_data_func=compute_fi_vi_curve,
        data_func_kwargs={
            "neuron_cls": case["neuron_cls"],
            "neuron_params": case["neuron_params"],
            "current_start": case["current_start"],
            "current_end": case["current_end"],
            "steps": case["steps"],
            "device": DEVICE,
        },
        name=case["name"],
    )
    save_fig(fig, name=case["name"])
    plt.close("all")

    # ---- Assertions on the underlying curve data (deterministic simulation).
    data = compute_fi_vi_curve(
        neuron_cls=case["neuron_cls"],
        neuron_params=case["neuron_params"],
        current_start=case["current_start"],
        current_end=case["current_end"],
        steps=case["steps"],
        device=DEVICE,
    )
    freq = data["frequencies"].cpu()  # (steps, n_neuron), Hz
    n_neuron = case["neuron_params"].get("n_neuron", 1)
    assert freq.shape == (case["steps"], n_neuron)
    assert data["currents"].shape == (case["steps"],)
    assert torch.isfinite(data["voltages"]).all()

    # f-I curve is monotone non-decreasing in the input current.
    assert (freq.diff(dim=0) >= 0).all()
    # Rates are bounded by the refractory period: f <= 1000 / tau_ref Hz.
    assert freq.max() <= 1000.0 / case["neuron_params"]["tau_ref"]
    # Strongest drive fires clearly more than the weakest.
    assert (freq[-1] > freq[0]).all()
    assert (freq[-1] > 10.0).all()
    # Sub-threshold (below rheobase) weakest current gives exactly zero spikes.
    assert (freq[0] == 0).all()
    # Identical neurons driven by the same current fire identically.
    assert torch.equal(freq, freq[:, :1].expand_as(freq))
