import os

import torch
from btorch.models import environ, linear, rnn, synapse
from btorch.models.functional import init_net_state
from btorch.models.glif import GLIF3
from btorch.models.init import build_dense_mat, uniform_v_
from torchview import draw_graph
from torchviz import make_dot

from tests.utils.file import fig_path


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def show_autograd_graph(output, params, name):
    digraph = make_dot(output, params=params, show_attrs=True)

    output_dir = fig_path(__file__)
    digraph.render(format="svg", outfile=os.path.join(output_dir, f"{name}_grad.svg"))


def show_forward_graph(model, input, name):
    output_dir = fig_path(__file__)
    _ = draw_graph(
        model,
        input,
        graph_dir="TB",
        directory=str(output_dir),
        filename=f"{name}",
        save_graph=True,
    )


def vis_neuron(neuron_params, dtype=torch.float32, n_neuron=3):
    def _make_net():
        neuron = GLIF3(
            n_neuron=n_neuron,
            **neuron_params,
            # hard_reset=True,
            detach_reset=False,
            step_mode="s",
            backend="torch",
            device=device,
            dtype=dtype,
        )
        init_net_state(neuron, dtype=dtype, device=device)
        uniform_v_(neuron, set_reset_value=False)

        x = torch.full((n_neuron,), 1.0, device=device, dtype=dtype, requires_grad=True)
        return neuron, x

    neuron, x = _make_net()
    with environ.context(dt=1.0):
        show_forward_graph(neuron, x, "single")

    neuron, x = _make_net()
    with environ.context(dt=1.0):
        neuron(x)
    show_autograd_graph(neuron.v, {"input": x}, "single")


def vis_network(neuron_params, dtype=torch.float32):
    dt = 1  # Time step

    def _make_net():
        n_neurons = 5000
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

        rec_weights = build_dense_mat(n_e_neurons, n_i_neurons)
        conn = linear.DenseConn(
            in_features=n_neurons, out_features=n_neurons, weight=rec_weights
        )

        # rec_weights = build_sparse_mat(n_e_neurons, n_i_neurons)
        # conn = linear.SparseConn(conn=rec_weights)

        tau_syn = torch.cat(
            [torch.ones(n_e_neurons) * 5.8, torch.ones(n_i_neurons) * 6.5]
        )
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
        )
        init_net_state(layer_module)
        layer_module = layer_module.to(dtype=dtype)
        # layer_module = torch.compile(layer_module, fullgraph=False)

        # Define the simulation parameters
        T = 3  # Total simulation time

        # Generate a stimulus
        x = torch.ones([T, n_neurons]) * 2
        x = x.to(dtype=dtype).requires_grad_()
        return layer_module, x

    net, x = _make_net()
    with environ.context(dt=float(dt)):
        show_forward_graph(net, x, "network")

    net, x = _make_net()
    with environ.context(dt=float(dt)):
        spikes, states = net(x)
    show_autograd_graph(
        states["neuron.v"],
        {"input": x, **dict(net.named_parameters())},
        "network",
    )


if __name__ == "__main__":
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
    vis_neuron(neuron_params)
    vis_network(neuron_params)
