import matplotlib.pyplot as plt
import torch

from btorch.models import environ, linear, rnn, synapse
from btorch.models.functional import init_net_state
from btorch.models.init import build_sparse_mat, uniform_v_
from btorch.models.neurons.glif import GLIF3
from btorch.models.scale import scale_state_
from btorch.utils.file import save_fig


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_single_neuron(neuron_params, dtype=torch.bfloat16):
    # Create a GLIF3 instance
    scaler = torch.GradScaler(device=device)
    neuron = GLIF3(
        n_neuron=1,
        **neuron_params,
        # hard_reset=True,
        detach_reset=False,
        step_mode="s",
        backend="torch",
        dtype=dtype,
        device=device,
        pre_spike_v=True,  # capture mem voltage before spike reset
    )
    # scale_net(neuron)
    init_net_state(neuron, dtype=dtype, device=device)
    # Define the simulation parameters
    dt = 1  # Time step
    T = 1000  # Total simulation time

    time = torch.arange(0, T, dt)

    with torch.autograd.detect_anomaly():
        # Generate a stimulus
        x_seq = torch.cat((torch.full((T // 2,), 5), torch.zeros((T // 2,))))
        x_seq = x_seq.to(device=device, dtype=dtype).requires_grad_(True)
        neuron_rnn = rnn.make_rnn(
            neuron, update_state_names=("v", "Iasc", "v_pre_spike")
        )
        with environ.context(dt=float(dt)):
            # spike, ret = neuron_rnn(x_seq / neuron.neuron_scale)
            spike, ret = neuron_rnn(x_seq)
            ret["spike"] = spike
            # ret = unscale_state(neuron, ret)

            spikes, membrane_potential, membrane_potential_pre, after_spike_current = (
                ret["spike"].float().detach().cpu(),
                ret["v"].float().detach().cpu(),
                ret["v_pre_spike"].float().detach().cpu(),
                ret["Iasc"].float().detach().cpu(),
            )

        loss = ret["v"].mean()
        scaler.scale(loss).backward()

    print(torch.isnan(x_seq.grad).nonzero())

    # Calculate firing rate
    firing_rate = spikes.sum() / T * 1000

    # Plot the spikes and membrane potential vs time
    fig, axes = plt.subplots(3, 1, sharex=True)
    # Plot spikes
    spk_nz = spikes.nonzero()
    axes[0].scatter(time[spk_nz[:, 0]], torch.ones_like(spk_nz[:, 0]), marker="|")
    axes[0].set_ylabel("Spikes")
    axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0].set_title(f"Firing Rate: {firing_rate.item()} Hz")

    # Plot membrane potential
    axes[1].plot(time, membrane_potential)
    axes[1].plot(time, membrane_potential_pre)
    axes[1].axhline(
        y=neuron_params["v_threshold"],
        color="r",
        linestyle="--",
        label="Firing Threshold",
    )
    axes[1].set_ylabel("Membrane Potential")

    # Plot membrane potential
    axes[2].plot(time, after_spike_current[..., 0])
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("After Spike Current")

    fig.suptitle("GLIF3 Neuron Dynamics")
    fig.legend()
    save_fig(fig, "single_neuron")


def test_multi_neuron(neuron_params: dict, n_neuron=10, dtype=torch.bfloat16):
    # Create a GLIF3 instance
    threshold = neuron_params["v_threshold"]
    neuron_params = neuron_params.copy()
    scale, zeropoint = scale_state_(neuron_params)
    neuron = GLIF3(
        n_neuron=n_neuron,
        **neuron_params,
        # hard_reset=True,
        detach_reset=False,
        step_mode="s",
        backend="torch",
    )
    init_net_state(neuron)
    uniform_v_(neuron, set_reset_value=False)
    neuron = torch.compile(neuron)
    neuron = neuron.to(device=device, dtype=dtype)
    # Define the simulation parameters
    dt = 1  # Time step
    T = 1000  # Total simulation time

    time = torch.arange(0, T, dt)

    # Generate a stimulus
    x_seq = torch.cat((torch.full((T // 2,), 2), torch.zeros((T // 2,))))
    x_seq = x_seq.to(device=device, dtype=dtype).requires_grad_()

    # better alternative to for loop
    # neuron_rnn = rnn.make_rnn(
    #     neuron, update_state_names=("v", "Iasc")
    # )
    with environ.context(dt=float(dt)):
        ret = {"spike": [], "v": [], "Iasc": []}
        for t, x in enumerate(x_seq):
            s = neuron(x / torch.as_tensor(scale).to(x.device))
            ret["spike"].append(s)
            ret["v"].append(neuron.v)
            ret["Iasc"].append(neuron.Iasc)
        ret = {k: torch.stack(v) for k, v in ret.items()}

    ret = {k: v.float().numpy(force=True) for k, v in ret.items()}

    scale_state_(
        ret,
        unscale=True,
        scale=scale,
        zeropoint=zeropoint,
    )
    spikes, membrane_potential, after_spike_current = (
        ret["spike"],
        ret["v"],
        ret["Iasc"],
    )

    # Calculate firing rate
    firing_rate = spikes.sum() / n_neuron / T * 1000

    # Plot the spikes and membrane potential vs time
    fig, axes = plt.subplots(3, 1, sharex=True)
    # Plot spikes
    spk_nz = spikes.nonzero()
    axes[0].scatter(time[spk_nz[0]], spk_nz[1], marker="|", linewidths=0.5)
    axes[0].set_ylabel("Spikes")
    axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0].set_title(f"Firing Rate: {firing_rate.item()} Hz")

    # Plot membrane potential
    axes[1].plot(time, membrane_potential, linewidth=0.5)
    axes[1].axhline(
        y=threshold,
        color="r",
        linestyle="--",
        label="Firing Threshold",
    )
    axes[1].set_ylabel("Membrane Potential")

    # Plot membrane potential
    axes[2].plot(time[:, None], after_spike_current[..., 0], linewidth=0.5)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("After Spike Current")

    fig.suptitle("GLIF3 Neuron Dynamics")
    fig.legend()
    save_fig(fig, "multi_neuron")


def test_exact_no_spike(neuron_params, n_neuron=10, dtype=torch.float32):
    # Create GLIF3 instance (your discrete step model)
    neuron = GLIF3(
        n_neuron=n_neuron,
        **neuron_params,
        detach_reset=False,
        step_mode="s",
        backend="torch",
    )
    init_net_state(neuron, dtype=dtype)
    uniform_v_(neuron, set_reset_value=False)
    neuron.Iasc = torch.rand_like(neuron.Iasc)
    neuron = neuron.to(device=device, dtype=dtype)

    # Simulation params
    dt = 1.0
    duration = 200
    T = int(duration / dt)
    time = torch.arange(0, T, device=device) * dt

    # Input stimulus
    x_seq = torch.full((T,), 3, device=device, dtype=dtype)

    # run exact first because it doesn't change neuron's internal state,
    # as long as v0 and Iasc0 are explicitly passed in.
    # --- Exact solver (vectorized, not step-by-step) ---
    v_exact, Iasc_exact = neuron.forward_exact_no_spike(
        x_seq[0], v0=neuron.v, Iasc0=neuron.Iasc, dt=time
    )
    v_exact, Iasc_exact = (
        v_exact.float().detach().cpu(),
        Iasc_exact.float().detach().cpu(),
    )

    # --- Discrete simulation ---
    with environ.context(dt=float(dt)):
        ret = {"v": [], "Iasc": []}
        for x in x_seq:
            ret["v"].append(neuron.v.clone())
            ret["Iasc"].append(neuron.Iasc.clone())
            neuron.neuronal_charge(x)
            neuron.neuronal_adaptation()
        ret = {k: torch.stack(v) for k, v in ret.items()}

    v_disc = ret["v"].float().detach().cpu()
    Iasc_disc = ret["Iasc"].float().detach().cpu()

    # --- Comparison ---
    v_err = torch.norm(v_disc - v_exact) / torch.norm(v_exact)
    Iasc_err = torch.norm(Iasc_disc - Iasc_exact) / torch.norm(Iasc_exact)

    print(f"Relative error: v={v_err:.2e}, Iasc={Iasc_err:.2e}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(time.cpu(), v_disc, label="Discrete", linewidth=0.5)
    axes[0].plot(time.cpu(), v_exact, label="Exact", linestyle="--", linewidth=0.8)
    axes[0].axhline(
        y=neuron_params["v_threshold"],
        color="r",
        linestyle="-.",
        label="Threshold",
    )
    axes[0].set_ylabel("Membrane Potential")

    axes[1].plot(time.cpu(), Iasc_disc[..., 0], label="Discrete", linewidth=0.5)
    axes[1].plot(
        time.cpu(), Iasc_exact[..., 0], label="Exact", linestyle="--", linewidth=0.8
    )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("After Spike Current")

    fig.suptitle("GLIF3 Neuron Dynamics (No spike, Discrete vs Exact)")
    handles, labels = [], []
    for ax in axes:
        handle, label = ax.get_legend_handles_labels()
        for hh, ll in zip(handle, label):
            if ll not in labels:  # only add if not already there
                handles.append(hh)
                labels.append(ll)
    axes[0].legend(handles, labels)
    save_fig(fig, "compare_exact_vs_discrete")


def test_network(neuron_params, dtype=torch.float32):
    # scaler = torch.GradScaler(device=device)
    n_neurons = 28 * 28
    e_i_neuron_number_ratio = 0.8
    n_e_neurons = int(n_neurons * e_i_neuron_number_ratio)
    n_i_neurons = n_neurons - n_e_neurons

    neuron_module = GLIF3(
        n_neuron=n_neurons,
        **neuron_params,
        detach_reset=False,
        step_mode="s",
        backend="torch",
    )
    # neuron_module = neuron_module.to(device)
    # rec_weights, _, _ = build_dense_mat(n_e_neurons, n_i_neurons)
    # conn = linear.DenseConn(
    #         in_features=n_neurons, out_features=n_neurons, weight=rec_weights
    # )

    rec_weights, _, _ = build_sparse_mat(n_e_neurons, n_i_neurons, i_e_ratio=1)
    conn = linear.SparseConn(
        # conn = rec_weights /
        # (neuron_params["v_threshold"] - neuron_params["v_reset"]),
        device=device,
        conn=rec_weights,
    )

    tau_syn = torch.cat([torch.ones(n_e_neurons) * 5.8, torch.ones(n_i_neurons) * 6.5])
    tau_syn = tau_syn
    psc_module = synapse.AlphaPSCBilleh(
        n_neurons,
        tau_syn=tau_syn,
        linear=conn,
        step_mode="s",
    )
    layer_module = rnn.RecurrentNN(
        neuron=neuron_module,
        synapse=psc_module,
        step_mode="m",
        update_state_names=("neuron.v", "synapse.psc"),
        grad_checkpoint=False,
    )
    # scale_net(layer_module)
    init_net_state(layer_module)
    uniform_v_(layer_module.neuron, set_reset_value=False)
    layer_module = layer_module.to(device=device, dtype=dtype)
    layer_module = torch.compile(layer_module, fullgraph=False)

    # Define the simulation parameters
    dt = 1  # Time step
    T = 1000  # Total simulation time
    time = torch.arange(0, T, dt)

    # Generate a stimulus
    x = torch.ones([T, n_neurons]) * 2
    x = x.to(device=device, dtype=dtype)  # .requires_grad_()

    # with torch.autograd.detect_anomaly():
    with environ.context(dt=float(dt)):
        # x = x / layer_module.neuron.neuron_scale
        spikes, states = layer_module(x)
        # states = unscale_state(layer_module, states)

    membrane_potential, I_syn = states["neuron.v"], states["synapse.psc"]

    # loss = membrane_potential.mean()
    # scaler.scale(loss).backward()
    # constrain_net(layer_module)

    # print(f"x: {torch.isnan(x.grad).nonzero()}")
    # print(
    #     [
    #         f"{n}: {torch.isnan(p.grad).nonzero()}"
    #         for n, p in layer_module.named_parameters()
    #     ]
    # )

    spikes = spikes.float().detach().cpu()
    membrane_potential = membrane_potential.float().detach().cpu()
    I_syn = I_syn.float().detach().cpu()

    fig, ax = plt.subplots(3, 1, sharex=True)

    # Plot spiking raster
    spike_nz = spikes.nonzero(as_tuple=True)
    ax[0].scatter(
        time[spike_nz[0]],
        spike_nz[1],
        marker=".",
        color="black",
        # linewidths=0.5,
        s=0.1,
    )
    ax[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax[0].set_ylabel("Neuron")
    firing_rate = spikes.sum() / n_neurons / T * 1000
    ax[0].set_title(f"Firing Rate: {firing_rate.item()} Hz")

    # Select three random neurons
    random_neurons = torch.randint(0, n_neurons, (5,))
    # random_neurons = torch.randint(n_e_neurons, n_neurons, (2,))
    # Plot membrane potential vs time for the selected neurons
    for neuron_idx in random_neurons:
        ax[1].plot(
            time, membrane_potential[:, neuron_idx], label=f"Neuron {neuron_idx.item()}"
        )

    ax[1].axhline(
        y=neuron_params["v_threshold"],
        color="r",
        linestyle="--",
        label="Firing Threshold",
    )
    ax[1].set_ylabel("Membrane Potential")
    # ax[1].set_ylim([neuron_params['v_rest'], neuron_params['v_peak']])
    ax[1].legend()

    # Plot synaptic currents vs time for the selected neurons
    for neuron_idx in random_neurons:
        ax[2].plot(time, I_syn[:, neuron_idx], label=f"Neuron {neuron_idx.item()}")

    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Synaptic Current")
    ax[2].legend()

    fig.tight_layout()

    save_fig(fig, "network")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Set the neuron parameters, using coherent units: msec+mV+nF+miuS+nA

    neuron_params = {
        "v_threshold": -45.0,  # mV
        "v_reset": -60.0,  # mV
        "c_m": 2.0,  # pfarad
        "tau": 20.0,  # ms
        "k": [1.0 / 80],  # ms^-1
        "asc_amps": [-0.2],  # pA
        "tau_ref": 2.0,  # ms
    }

    test_single_neuron(neuron_params)
    test_multi_neuron(neuron_params)
    test_exact_no_spike(neuron_params)
    test_network(neuron_params)
