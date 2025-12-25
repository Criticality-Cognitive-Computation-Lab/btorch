import matplotlib.pyplot as plt
import torch

from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.neurons.izhikevich import Izhikevich

from ...utils.file import save_fig


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DT = 0.1  # ms


def _boxcar(values, durations, dt=DT):
    """Create a boxcar (step) current following brainstate section_input."""
    assert len(values) == len(durations)
    segments = [torch.full((int(d / dt),), v) for v, d in zip(values, durations)]
    return torch.cat(segments)


CLASSIC_PHENOTYPES = [
    {
        "name": "tonic_spiking",
        "title": "Tonic Spiking",
        "params": {"a": 0.02, "b": 0.20, "v_reset": -65.0, "d": 8.0},
        "stimulus": lambda dt=DT: _boxcar([0.0, 10.0], [50, 150], dt=dt),
    },
    {
        "name": "phasic_spiking",
        "title": "Phasic Spiking",
        "params": {"a": 0.02, "b": 0.25, "v_reset": -65.0, "d": 6.0},
        "stimulus": lambda dt=DT: _boxcar([0.0, 1.0], [50, 150], dt=dt),
    },
    {
        "name": "tonic_bursting",
        "title": "Tonic Bursting",
        "params": {"a": 0.02, "b": 0.20, "v_reset": -50.0, "d": 2.0},
        "stimulus": lambda dt=DT: _boxcar([0.0, 15.0], [50, 150], dt=dt),
    },
    {
        "name": "phasic_bursting",
        "title": "Phasic Bursting",
        "params": {"a": 0.02, "b": 0.25, "v_reset": -55.0, "d": 0.05},
        "stimulus": lambda dt=DT: _boxcar([0.0, 1.0], [50, 150], dt=dt),
    },
    {
        "name": "mixed_mode",
        "title": "Mixed Mode",
        "params": {"a": 0.02, "b": 0.20, "v_reset": -55.0, "d": 4.0},
        "stimulus": lambda dt=DT: _boxcar([0.0, 10.0], [50, 150], dt=dt),
    },
    {
        "name": "spike_frequency_adaptation",
        "title": "Spike Frequency Adaptation",
        "params": {"a": 0.01, "b": 0.20, "v_reset": -65.0, "d": 8.0},
        "stimulus": lambda dt=DT: _boxcar([0.0, 30.0], [50, 150], dt=dt),
    },
]


def _simulate(neuron: Izhikevich, stimulus: torch.Tensor, dt: float):
    traces = {"spike": [], "v": [], "u": [], "input": []}
    with torch.no_grad():
        with environ.context(dt=float(dt)):
            for current in stimulus:
                spike = neuron(current.expand(neuron.n_neuron))
                traces["spike"].append(spike.detach().cpu())
                traces["v"].append(neuron.v.detach().cpu())
                traces["u"].append(neuron.u.detach().cpu())
                traces["input"].append(current.detach().cpu())

    traces = {k: torch.stack(v) for k, v in traces.items()}
    return traces


def _plot_grid(results):
    n = len(results)
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(12, 3 * n), sharex=False)
    if n == 1:
        axes = axes.reshape(1, -1)
    legend_handles, legend_labels = [], []

    for idx, case_result in enumerate(results):
        time_axis = case_result["time"]
        case, traces = case_result["case"], case_result["traces"]
        spikes = traces["spike"].squeeze(-1)
        v = traces["v"].squeeze(-1)
        u = traces["u"].squeeze(-1)
        inp = traces["input"].squeeze(-1)

        firing_rate = spikes.sum() / (len(time_axis) * (DT / 1000.0))

        ax_spk, ax_v, ax_u = axes[idx]
        spk_nz = spikes.nonzero(as_tuple=False)
        if len(spk_nz) > 0:
            ax_spk.scatter(
                time_axis[spk_nz[:, 0]],
                torch.ones(spk_nz.shape[0]),
                marker="|",
                linewidths=0.8,
            )
        ax_spk.set_ylabel("Spikes")
        ax_spk.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax_spk.set_title(f"{case['title']} — {firing_rate.item():.1f} Hz")

        ax_v.plot(time_axis, v, linewidth=0.9, label="v")
        ax_v.axhline(
            y=case_result["v_threshold"], color="r", linestyle="--", label="Threshold"
        )
        ax_v.axhline(
            y=case_result["v_rest"],
            color="b",
            linestyle=":",
            linewidth=0.9,
            label="v_rest",
        )
        ax_v.axhline(
            y=case_result["v_reset"],
            color="orange",
            linestyle="-.",
            linewidth=0.9,
            label="v_reset/v_min",
        )
        ax_v.axhline(
            y=case_result["v_peak"],
            color="m",
            linestyle="--",
            linewidth=0.9,
            label="v_peak",
        )
        ax_v.set_ylabel("Membrane Potential")
        ax_in = ax_v.twinx()
        ax_in.plot(
            time_axis, inp, color="g", linestyle=":", linewidth=0.9, label="Input (pA)"
        )
        ax_in.set_ylabel("Input (pA)", color="g")
        ax_in.tick_params(axis="y", labelcolor="g")

        ax_u.plot(time_axis, u, linewidth=0.9)
        ax_u.set_ylabel("Recovery (u)")
        ax_u.set_xlabel("Time (ms)")

        if idx == 0:
            h_v, l_v = ax_v.get_legend_handles_labels()
            h_i, l_i = ax_in.get_legend_handles_labels()
            legend_handles.extend(h_v + h_i)
            legend_labels.extend(l_v + l_i)

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper right")
    fig.suptitle("Izhikevich 2003 Phenotypes (brainstate-inspired)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, name="izhikevich_phenotypes_grid")
    plt.close(fig)


def test_izhikevich_2003_phenotypes():
    base_kwargs = {
        "n_neuron": 1,
        "p1": 0.04,
        "p2": 5.0,
        # "p3": 140.0,
        "v_rest": -65.0,
        "v_peak": 30.0,
        "c_m": 1.0,
        "detach_reset": False,
        "step_mode": "s",
        "backend": "torch",
        "device": DEVICE,
    }

    results = []
    for case in CLASSIC_PHENOTYPES:
        params = base_kwargs.copy()
        params.update(case["params"])

        neuron = Izhikevich.from_canonical_quadratic(**params)
        init_net_state(neuron, device=DEVICE)

        stimulus = case["stimulus"](dt=DT).to(device=DEVICE, dtype=torch.float32)
        traces = _simulate(neuron, stimulus, dt=DT)
        time = torch.arange(0, len(stimulus) * DT, DT)
        results.append(
            {
                "case": case,
                "traces": traces,
                "v_threshold": float(torch.as_tensor(neuron.v_threshold).flatten()[0]),
                "v_rest": float(torch.as_tensor(neuron.v_rest).flatten()[0]),
                "v_reset": float(torch.as_tensor(neuron.v_reset).flatten()[0]),
                "v_peak": float(torch.as_tensor(neuron.v_peak).flatten()[0]),
                "time": time,
            }
        )

    # Use time from first case (all share durations)
    if results:
        _plot_grid(results)
