import pytest
import torch

from btorch.models import environ, linear, rnn, synapse
from btorch.models.functional import (
    init_net_state,
    named_hidden_states,
    named_memory_reset_values,
    set_hidden_states,
    set_memory_reset_values,
)
from btorch.models.init import uniform_v_
from btorch.models.neurons.glif import GLIF3
from btorch.models.neurons.lif import LIF
from tests.utils.conn import build_sparse_mat


# --- Fixtures and helpers --- #
@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def dtype():
    return torch.float32


@pytest.fixture(scope="module")
def neuron_params():
    return {
        "v_threshold": -45.0,  # mV
        "v_reset": -60.0,  # mV
        "c_m": 2.0,  # pfarad
        "tau": 20.0,  # ms
        "k": [1.0 / 80],  # ms^-1
        "asc_amps": [-0.2],  # pA
        "tau_ref": 2.0,  # ms
    }


def def_model(neuron_params, device, dtype):
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

    rec_weights, _, _ = build_sparse_mat(n_e_neurons, n_i_neurons, i_e_ratio=1)
    conn = linear.SparseConn(conn=rec_weights, device=device)

    tau_syn = torch.cat([torch.ones(n_e_neurons) * 5.8, torch.ones(n_i_neurons) * 6.5])
    psc_module = synapse.AlphaPSCBilleh(
        n_neurons,
        tau_syn=tau_syn,
        linear=conn,
        step_mode="s",
    )

    module = rnn.RecurrentNN(
        neuron=neuron_module,
        synapse=psc_module,
        step_mode="m",
        update_state_names=("neuron.v", "synapse.psc"),
        grad_checkpoint=False,
    )

    # scale_net(module)
    init_net_state(module)
    uniform_v_(module.neuron, set_reset_value=True)
    module = module.to(device=device, dtype=dtype)

    return module


def run_sim(module, device, dtype):
    dt = 1
    T = 50  # shorter sim for test speed
    x = torch.ones([T, *module.neuron.n_neuron]) * 2
    x = x.to(device=device, dtype=dtype)

    # reset_net_state(module)

    with environ.context(dt=float(dt)):
        # x = x / module.neuron.neuron_scale
        spikes, states = module(x)
        # states = unscale_state(module, states)

    return spikes, states


# --- The actual test --- #
def test_torch_save_load(tmp_path, neuron_params, device, dtype):
    module = def_model(neuron_params, device, dtype)

    hidden_states_before = named_hidden_states(module)
    # Run initial simulation
    _, states_before = run_sim(module, device, dtype)

    # Capture states
    memories_rv_before = named_memory_reset_values(module)

    # Save everything
    model_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": module.state_dict(),
            "memories_rv": memories_rv_before,
            "hidden_states": hidden_states_before,
        },
        model_path,
    )

    # Recreate and load
    module_loaded = def_model({}, device, dtype)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    module_loaded.load_state_dict(checkpoint["model_state"])
    set_memory_reset_values(module_loaded, checkpoint["memories_rv"])
    set_hidden_states(module_loaded, checkpoint["hidden_states"])

    # Validate internal state restoration
    memories_rv_after = named_memory_reset_values(module_loaded)
    hidden_states_after = named_hidden_states(module_loaded)

    for k in memories_rv_before:
        assert torch.allclose(
            torch.as_tensor(memories_rv_before[k].value, device=device),
            torch.as_tensor(memories_rv_after[k].value, device=device),
        ), f"Memory reset value mismatch for {k}"

    for k in hidden_states_before:
        assert torch.allclose(
            hidden_states_before[k], hidden_states_after[k]
        ), f"Hidden state mismatch for {k}"

    # Run again and compare some output
    _, states_after = run_sim(module_loaded, device, dtype)

    assert torch.allclose(
        states_before["neuron.v"], states_after["neuron.v"], atol=1e-6
    ), "Membrane potential mismatch after save/load"

    assert torch.allclose(
        states_before["synapse.psc"], states_after["synapse.psc"], atol=1e-6
    ), "Synaptic current mismatch after save/load"


def test_checkpoint_param_shape_transition_for_uniform_and_non_uniform(tmp_path):
    """Checkpoint loading should follow uniform/non-uniform shape semantics.

    This test validates the shape transition policy introduced by ParamBufferMixin:
    1. Full but uniform checkpoint tensors can be loaded as compact scalar params.
    2. Full non-uniform checkpoint tensors must remain full and not be collapsed.

    The test persists checkpoints to disk to mirror real training/inference
    workflows where shape mismatches appear across runs.
    """

    n_neuron = 4

    # Destination model starts with compact scalar tau.
    dest_uniform = LIF(n_neuron=n_neuron, tau=20.0)
    assert tuple(dest_uniform.tau.shape) == ()

    # Source model stores tau as full, but uniform across neurons.
    src_full_uniform = LIF(n_neuron=n_neuron, tau=torch.full((n_neuron,), 20.0))
    ckpt_uniform = tmp_path / "lif_uniform.pt"
    torch.save(src_full_uniform.state_dict(), ckpt_uniform)
    dest_uniform.load_state_dict(torch.load(ckpt_uniform, weights_only=False))

    # Uniform full values are compacted to scalar storage.
    assert tuple(dest_uniform.tau.shape) == ()
    assert torch.allclose(
        dest_uniform.tau, torch.tensor(20.0, dtype=dest_uniform.tau.dtype)
    )

    # Source model stores tau as full and non-uniform.
    src_full_non_uniform = LIF(
        n_neuron=n_neuron, tau=torch.tensor([1.0, 2.0, 3.0, 4.0])
    )
    ckpt_non_uniform = tmp_path / "lif_non_uniform.pt"
    torch.save(src_full_non_uniform.state_dict(), ckpt_non_uniform)

    dest_non_uniform = LIF(n_neuron=n_neuron, tau=20.0)
    dest_non_uniform.load_state_dict(torch.load(ckpt_non_uniform, weights_only=False))

    # Non-uniform values must remain full to preserve per-neuron information.
    assert tuple(dest_non_uniform.tau.shape) == (n_neuron,)
    assert torch.allclose(dest_non_uniform.tau, torch.tensor([1.0, 2.0, 3.0, 4.0]))
