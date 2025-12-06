import gzip
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse
from btorch.models.init import build_sparse_mat

from tests.utils.file import fig_path


np.random.seed(0)
fdtype = np.float32


def euler_step(f, y, *args, dt):
    """Explicit Euler step for ODE y' = f(y, *args)."""
    return y + dt * f(y, *args)


def exp_euler_step(f, linear, y, *args, dt):
    """Exponential Euler step for ODE y' = f(y, *args)."""
    derivative = f(y, *args)
    return y + np.expm1(dt * linear) / linear * derivative


@dataclass
class GLIF3:
    n_neuron: int

    v_th: Union[float, np.ndarray]
    E_L: Optional[Union[float, np.ndarray]]
    C: Union[float, np.ndarray]
    tau: Union[float, np.ndarray]
    v_reset: Union[float, np.ndarray] = None
    fm: Union[float, Sequence[float], np.ndarray] = field(default_factory=lambda: [0.2])
    delta_Im: Union[float, Sequence[float], np.ndarray] = field(
        default_factory=lambda: [0.0]
    )
    refractory_count: Union[float, np.ndarray] = 0.0

    hard_reset: bool = False

    # # Internal state, initialized later
    # v: np.ndarray = field(init=False)
    # Iasc: np.ndarray = field(init=False)
    # refractory: np.ndarray = field(init=False)
    # n_Iasc: int = field(init=False)

    def __post_init__(self):
        self.v_th = np.array(self.v_th, dtype=fdtype)
        self.E_L = np.array(self.E_L, dtype=fdtype)
        if self.v_reset is None:
            self.v_reset = self.E_L.copy()
        else:
            self.v_reset = np.array(self.v_reset, dtype=fdtype)

        self.C = np.array(self.C, dtype=fdtype)
        self.tau = np.array(self.tau, dtype=fdtype)
        self.refractory_count = np.array(self.refractory_count, dtype=fdtype)
        self.fm = np.array(self.fm, dtype=fdtype)

        self.delta_Im = np.array(self.delta_Im, dtype=fdtype)
        if self.delta_Im.ndim == 1:
            self.delta_Im = self.delta_Im.reshape(1, -1)
        self.n_Iasc = self.delta_Im.shape[-1]

        self.v = np.full(self.n_neuron, self.E_L, dtype=fdtype)
        self.Iasc = np.zeros((self.n_neuron, self.n_Iasc), dtype=fdtype)
        self.refractory = np.zeros(self.n_neuron, dtype=fdtype)

    def dIasc(self, Iasc):
        return -self.fm * Iasc

    def dV(self, V, Iasc, x):
        Isum = x
        return -(V - self.E_L) / self.tau + (Isum + Iasc.sum(axis=-1)) / self.C

    def neuronal_charge(self, x, dt):
        self.v = exp_euler_step(self.dV, -1.0 / self.tau, self.v, self.Iasc, x, dt=dt)

    def neuronal_adaptation(self, dt):
        self.Iasc = exp_euler_step(self.dIasc, -self.fm, self.Iasc, dt=dt)

    def neuronal_fire(self):
        not_in_refractory = self.refractory == 0
        spike_raw = self.v > self.v_th
        spike = spike_raw & not_in_refractory
        return spike

    def neuronal_reset(self, spike, dt):
        if self.hard_reset:
            self.v -= (self.v - self.v_reset) * spike
        else:
            self.v -= (self.v_th - self.v_reset) * spike

        self.Iasc += self.delta_Im * spike[:, None]

        self.refractory = np.maximum(
            self.refractory + spike * self.refractory_count - dt, 0.0
        )

    def step(self, x, dt):
        self.neuronal_charge(x, dt)
        self.neuronal_adaptation(dt)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike, dt)
        return spike


@dataclass
class SparseConn:
    weight: scipy.sparse.sparray  # (in, out)

    def step(self, inp):
        return inp @ self.weight


@dataclass
class ExponentialPSC:
    n_neuron: int
    tau_syn: float
    linear: SparseConn

    # I_syn: np.ndarray = None  # shape (n_neuron,)

    def __post_init__(self):
        self.I_syn = np.zeros(self.n_neuron, dtype=fdtype)

    def dI_syn(self, I_syn):
        return -I_syn / self.tau_syn

    def conductance_charge(self, dt: float):
        self.I_syn = euler_step(self.dI_syn, self.I_syn, dt=dt)

    def adaptation_charge(self, z: np.ndarray, dt: float):
        wz = self.linear.step(z)
        self.I_syn += wz

    def step(self, z: np.ndarray, dt: float):
        self.conductance_charge(dt)
        self.adaptation_charge(z, dt)
        return self.I_syn


@dataclass
class AlphaPSCBilleh:
    n_neuron: int
    tau_syn: float
    linear: SparseConn

    # I_syn: np.ndarray = None  # shape (n_neuron,)
    # C_rise: np.ndarray = None  # shape (n_neuron,)

    def __post_init__(self):
        self.I_syn = np.zeros(self.n_neuron, dtype=fdtype)
        self.C_rise = np.zeros(self.n_neuron, dtype=fdtype)
        self.syn_decay = np.exp(-1.0 / self.tau_syn)

    def conductance_charge(self, dt: float):
        self.I_syn = self.syn_decay * self.I_syn + self.syn_decay * self.C_rise
        return self.I_syn

    def adaptation_charge(self, z: np.ndarray, dt: float):
        wz = self.linear.step(z)
        self.C_rise = self.syn_decay * self.C_rise + np.e / self.tau_syn * wz

    def step(self, z: np.ndarray, dt: float):
        self.conductance_charge(dt)
        self.adaptation_charge(z, dt)
        return self.I_syn


@dataclass
class AlphaPSC:
    n_neuron: int
    tau_syn: float
    linear: SparseConn
    g_max: float = 1.0

    # I_syn: np.ndarray = None  # shape (n_neuron,)
    # C_rise: np.ndarray = None  # shape (n_neuron,)

    def __post_init__(self):
        self.I_syn = np.zeros(self.n_neuron, dtype=fdtype)
        self.C_rise = np.zeros(self.n_neuron, dtype=fdtype)

    def dI_syn(self, I_syn, C_rise):
        return -I_syn / self.tau_syn + C_rise / self.tau_syn

    def dC_rise(self, C_rise):
        return -C_rise / self.tau_syn

    def conductance_charge(self, dt: float):
        self.I_syn = exp_euler_step(
            self.dI_syn, -1.0 / self.tau_syn, self.I_syn, self.C_rise, dt=dt
        )

    def adaptation_charge(self, z: np.ndarray, dt: float):
        wz = self.g_max * self.linear.step(z)
        self.C_rise = (
            exp_euler_step(self.dC_rise, -1.0 / self.tau_syn, self.C_rise, dt=dt) + wz
        )

    def step(self, z: np.ndarray, dt: float):
        self.conductance_charge(dt)
        self.adaptation_charge(z, dt)
        return self.I_syn


@dataclass
class Network:
    w_in: SparseConn
    synapse: dict[str, AlphaPSCBilleh]  # E, I
    neuron: GLIF3

    def step(self, I_e: np.ndarray, dt):
        I_syn = np.zeros_like(next(iter(self.synapse.values())).I_syn)
        for syn in self.synapse.values():
            I_syn += syn.I_syn
        z = self.neuron.step(I_syn + I_e, dt)
        for syn in self.synapse.values():
            _ = syn.step(z, dt)
        return z


# TODO: use SONATA here


def save_sparray(path: str, mat: scipy.sparse.sparray):
    if not isinstance(mat, scipy.sparse.coo_array):
        mat: scipy.sparse.coo_array = mat.tocoo()
    df = pd.DataFrame({"row": mat.row, "col": mat.col, "value": mat.data})
    df.to_csv(path, index=False)


def load_sparray(path: str, shape) -> scipy.sparse.coo_array:
    df = pd.read_csv(path)
    return scipy.sparse.coo_array((df["value"], (df["row"], df["col"])), shape=shape)


def recurse_dict(d: dict, mapper: Callable, include_sequence: bool = False) -> dict:
    def _f(d, k):
        if isinstance(d, dict):
            return {k: _f(v, k) for k, v in d.items()}
        if include_sequence:
            if isinstance(d, tuple):
                return tuple(_f(ve, None) for ve in d)
            elif isinstance(d, list):
                return list(_f(ve, None) for ve in d)
        return mapper(k, d)

    return _f(d, None)


def save_network(network: Network, base_dir: str, json_file: str = "network.json"):
    data = {"neuron": asdict(network.neuron), "synapse": {}, "w_in": {}}
    data = recurse_dict(
        data, lambda k, v: v.tolist() if isinstance(v, np.ndarray) else v
    )

    os.makedirs(os.path.join(base_dir, "synapse"), exist_ok=True)
    for k, syn in network.synapse.items():
        syn_data = asdict(syn)
        weight = syn_data["linear"]["weight"]
        weight_path = f"synapse/{k}.csv.gz"
        save_sparray(os.path.join(base_dir, weight_path), weight)
        syn_data["linear"]["weight"] = weight_path
        syn_data["linear"]["shape"] = weight.shape
        data["synapse"][k] = syn_data

    weight_path = "w_in.csv.gz"
    w_in_data = {"weight": weight_path, "shape": network.w_in.weight.shape}
    save_sparray(os.path.join(base_dir, weight_path), network.w_in.weight)
    data["w_in"] = w_in_data

    json_path = os.path.join(base_dir, json_file)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def load_network(base_dir: str, json_file: str = "network.json") -> Network:
    json_path = os.path.join(base_dir, json_file)
    with open(json_path, "r") as f:
        data = json.load(f)

    neuron = GLIF3(**data["neuron"])

    synapse = {}
    for k, syn_data in data["synapse"].items():
        weight_path = syn_data["linear"]["weight"]
        weight_shape = syn_data["linear"]["shape"]
        weight = load_sparray(os.path.join(base_dir, weight_path), weight_shape)
        syn_data["linear"] = SparseConn(weight=weight)
        synapse[k] = AlphaPSCBilleh(**syn_data)

    w_in_data = data["w_in"]
    weight_path = w_in_data["weight"]
    weight_shape = w_in_data["shape"]
    weight = load_sparray(os.path.join(base_dir, weight_path), weight_shape)
    w_in = SparseConn(weight=weight)

    return Network(synapse=synapse, neuron=neuron, w_in=w_in)


def _glif_config():
    return {
        "E_L": -71.5881938934326,  # mV
        "v_th": 22.647237682932017,  # mV
        "refractory_count": 4,
        "fm": [0.3333333333333333],
        "delta_Im": [-10.862661485260188],  # pA
        "C": 81.50097665577195,  # pFarad
        "tau": 8.150097665577195e-11 * 79640729.43958016 * 1e3,  # ms
    }


def create_example_network():
    n_neuron = 200
    e_i_neuron_number_ratio = 0.8
    n_e_neuron = int(n_neuron * e_i_neuron_number_ratio)
    E_matrix, I_matrix, e_idx, i_idx = build_sparse_mat(
        n_e_neuron, n_neuron - n_e_neuron, split=True, density=0.08
    )
    n_inp = 100
    Wji_in_count = 200
    Wji_in: scipy.sparse.coo_array = scipy.sparse.random_array(
        (n_inp, n_neuron),
        density=Wji_in_count / (n_neuron * n_inp),
        format="coo",
        data_sampler=lambda size: np.ones(shape=size),
        random_state=np.random.default_rng(seed=8),
    )
    synapse_config = {
        "tau_syn": [5.0, 5.0],
        "E_matrix": E_matrix,
        "I_matrix": I_matrix,
    }

    glif = GLIF3(n_neuron=n_neuron, **_glif_config())
    w_in = SparseConn(Wji_in)

    # tau_syn depends on E, I neuron type
    # tau_syn = synapse_config["tau_syn"][0]*np.ndarray(n_neuron)
    # tau_syn[i_idx] = synapse_config["tau_syn"][1]

    w_rec = {
        "E": {"weight": E_matrix, "tau_syn": synapse_config["tau_syn"][0]},
        "I": {"weight": -I_matrix, "tau_syn": synapse_config["tau_syn"][1]},
    }
    synapses = {
        k: AlphaPSCBilleh(
            n_neuron=n_neuron, linear=SparseConn(v["weight"]), tau_syn=v["tau_syn"]
        )
        for k, v in w_rec.items()
    }
    network = Network(synapse=synapses, neuron=glif, w_in=w_in)
    return network


def uniform_v_(neuron: GLIF3):
    neuron.v = (
        np.random.random(neuron.v.shape) * (neuron.v_th - neuron.E_L) + neuron.E_L
    )


def test_multi_neuron(n_neuron=10):
    import matplotlib.pyplot as plt

    # Create a GLIF3 instance
    glif = GLIF3(n_neuron=n_neuron, **_glif_config())
    uniform_v_(glif)

    T = 1000
    dt = 1.0
    x_seq = np.concatenate((np.full((T // 2,), 1200), np.zeros((T // 2,))))
    x_seq = x_seq[..., None]
    z, v = np.empty((T, n_neuron)), np.empty((T, n_neuron))
    asc = np.empty((T,) + (n_neuron, glif.n_Iasc))
    for i in range(T):
        z[i] = glif.step(x_seq[i], dt=dt)
        v[i] = glif.v
        asc[i] = glif.Iasc
    t = np.arange(T) * dt

    # Calculate firing rate
    firing_rate = z.sum() / n_neuron / T * 1000

    # Plot the spikes and membrane potential vs time
    fig, axes = plt.subplots(3, 1, sharex=True)
    # Plot spikes
    spk_nz = z.nonzero()
    axes[0].scatter(t[spk_nz[0]], spk_nz[1], marker="|", linewidths=0.5)
    axes[0].set_ylabel("Spikes")
    axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0].set_title(f"Firing Rate: {firing_rate} Hz")

    # Plot membrane potential
    axes[1].plot(t, v, linewidth=0.5)
    axes[1].axhline(
        y=glif.v_th,
        color="r",
        linestyle="--",
        label="Firing Threshold",
    )
    axes[1].set_ylabel("Membrane Potential")

    # Plot membrane potential
    axes[2].plot(t[:, None], asc[..., 0], linewidth=0.5)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("After Spike Current")

    fig.suptitle("GLIF3 Neuron Dynamics")
    fig.legend()

    output_dir = fig_path(__file__)
    fig.savefig(output_dir / "multi_neuron.pdf")


def run(net: Network, x, T: Optional[int] = None, dt=1.0, skip_w_in=False):
    x = np.asarray(x)
    if T is None:
        assert isinstance(x, np.ndarray) and x.shape[0] > 1
        T = x.shape[0]

    if skip_w_in:
        x = np.broadcast_to(x, (T, net.neuron.n_neuron))
    else:
        x = np.broadcast_to(x, (T, net.w_in.weight.shape[0]))

    n_neuron = net.neuron.n_neuron
    z, v = np.empty((T, n_neuron)), np.empty((T, n_neuron))
    for i in range(T):
        I_e = net.w_in.step(x[i]) if not skip_w_in else x[i]
        z[i] = net.step(I_e, dt)
        v[i] = net.neuron.v

    return z, v, np.arange(T) * dt


def test_network():
    import matplotlib.pyplot as plt

    # network = create_example_network()
    # save_network(network, fig_path(__file__))
    output_dir = fig_path(__file__)
    network = load_network(output_dir)

    T, dt = 1000, 1.0
    z, v, t = run(network, 600, T=T, dt=dt)

    fig, ax = plt.subplots(2, 1, sharex=True)

    # Plot spiking raster
    spike_nz = z.nonzero()

    n_neurons = network.neuron.n_neuron
    ax[0].scatter(
        t[spike_nz[0]],
        spike_nz[1],
        marker="|",
        color="black",
        linewidths=0.5,
    )
    ax[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax[0].set_ylabel("Neuron")
    firing_rate = z.sum() / n_neurons / T * 1000
    ax[0].set_title(f"Firing Rate: {firing_rate.item()} Hz")

    # Select three random neurons
    random_neurons = np.random.randint(0, n_neurons, (5,))
    # random_neurons = torch.randint(n_e_neurons, n_neurons, (2,))
    # Plot membrane potential vs time for the selected neurons
    for neuron_idx in random_neurons:
        ax[1].plot(t, v[:, neuron_idx], label=f"Neuron {neuron_idx.item()}")

    ax[1].axhline(
        y=network.neuron.v_th,
        color="r",
        linestyle="--",
        label="Firing Threshold",
    )
    ax[1].set_ylabel("Membrane Potential")
    # ax[1].set_ylim([neuron_params['v_rest'], neuron_params['v_peak']])
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "network.pdf")

    with gzip.open(output_dir / "v.csv.gz", "wt") as f:
        np.savetxt(f, v, delimiter=",")


if __name__ == "__main__":
    test_multi_neuron()
    test_network()
