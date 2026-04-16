import torch

from btorch.models import environ
from btorch.models.functional import init_net_state, reset_net_state
from btorch.models.neurons.glif import GLIF3
from btorch.models.neurons.lif import LIF
from btorch.models.synapse import DelayedPSC, DualExponentialPSC


def test_lif_multi_dim_batch_and_neuron_axes():
    # Multi-dim batch and neuron axes should be preserved across init/reset/forward.
    batch_shape = (2, 3)
    neuron_shape = (4, 5)
    neuron = LIF(n_neuron=neuron_shape, v_threshold=1.0, v_reset=0.0)

    # init_state should add batch dims in front of neuron axes.
    init_net_state(neuron, batch_size=batch_shape)
    assert neuron.v.shape == batch_shape + neuron_shape

    # Forward should keep the same batch + neuron shape.
    x = torch.zeros(batch_shape + neuron_shape)
    with environ.context(dt=1.0):
        spike = neuron(x)
    assert spike.shape == batch_shape + neuron_shape

    # reset should accept tuple batch_size and preserve shapes.
    reset_net_state(neuron, batch_size=batch_shape)
    assert neuron.v.shape == batch_shape + neuron_shape


def test_glif_multi_dim_state_shapes():
    # GLIF uses extra trailing dims for after-spike currents (n_Iasc).
    batch_shape = (3,)
    neuron_shape = (2, 2)
    n_iasc = 2
    neuron = GLIF3(
        n_neuron=neuron_shape,
        v_threshold=-50.0,
        v_reset=-65.0,
        c_m=0.05,
        tau=20.0,
        k=[0.1, 0.2],
        asc_amps=[1.0, 0.5],
        tau_ref=2.0,
        step_mode="s",
    )

    # Iasc should be (*batch, *neuron_shape, n_Iasc).
    init_net_state(neuron, batch_size=batch_shape)
    assert neuron.Iasc.shape == batch_shape + neuron_shape + (n_iasc,)

    # Forward should keep the batch + neuron axes intact.
    x = torch.zeros(batch_shape + neuron_shape)
    with environ.context(dt=1.0):
        spike = neuron(x)
    assert spike.shape == batch_shape + neuron_shape


def test_glif_scalar_tensor_k_broadcasts_to_asc_channels():
    # Regression test for GLIF ASC shape inference.
    #
    # A scalar tensor is a common way to provide a shared decay constant `k`
    # while still using a multi-channel `asc_amps`. In this configuration,
    # GLIF should infer `n_Iasc` from the largest trailing ASC dimension and
    # broadcast scalar `k` across all ASC channels.
    #
    # This case previously raised `IndexError: tuple index out of range` during
    # initialization when the code assumed non-sequence inputs always had a
    # trailing axis.
    neuron_shape = (4,)
    neuron = GLIF3(
        n_neuron=neuron_shape,
        v_threshold=-50.0,
        v_reset=-65.0,
        c_m=0.05,
        tau=20.0,
        k=torch.tensor(0.2),
        asc_amps=[1.0, 0.5],
        tau_ref=2.0,
        step_mode="s",
    )

    assert neuron.n_Iasc == 2
    assert neuron.k.shape == neuron_shape + (2,)
    assert neuron.asc_amps.shape == neuron_shape + (2,)

    # The scalar `k` should broadcast to both ASC channels for every neuron.
    expected_k = torch.full(neuron_shape + (2,), 0.2, dtype=neuron.k.dtype)
    assert torch.allclose(neuron.k, expected_k)

    init_net_state(neuron, batch_size=(3,))
    assert neuron.Iasc.shape == (3,) + neuron_shape + (2,)


def test_synapse_delay_buffer_multi_dim_axes():
    # Delay buffers keep time first, then batch, then neuron axes.
    batch_shape = (2, 1)
    neuron_shape = (3, 4)
    max_delay_steps = 3
    linear = torch.nn.Identity()

    with environ.context(dt=1.0):
        synapse = DelayedPSC(
            DualExponentialPSC(
                n_neuron=neuron_shape,
                tau_decay=5.0,
                tau_rise=1.0,
                linear=linear,
            ),
            max_delay_steps=max_delay_steps,
        )

    init_net_state(synapse, batch_size=batch_shape)

    # Access delay buffer through SpikeHistory interface
    # History stores buffer as (*batch, max_delay, *n_neuron)
    # DelayedPSC allocates max_delay_steps + 1 to support get_delay(max_delay_steps)
    expected_history_shape = (*batch_shape, max_delay_steps + 1, *neuron_shape)
    assert synapse.history.history.shape == expected_history_shape
    assert synapse.psc.shape == batch_shape + neuron_shape
