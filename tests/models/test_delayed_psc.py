import torch

from btorch.models import environ
from btorch.models.functional import init_net_state, reset_net
from btorch.models.neurons import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import (
    DelayedPSC,
    DualExponentialPSC,
    ExponentialPSC,
)


def _expected_delayed_exponential_psc(
    z_seq: torch.Tensor,
    dt: float,
    tau_syn: float,
    max_delay_steps: int,
) -> torch.Tensor:
    """Manually compute expected PSC for DelayedPSC(ExponentialPSC).

    ExponentialPSC evolves as:
        psc_{t+1} = psc_t * exp(-dt / tau_syn) + spike_{t - delay}
    where spike is zero for t < delay.
    """
    decay = torch.exp(torch.tensor(-dt / tau_syn, dtype=z_seq.dtype))
    psc = torch.zeros_like(z_seq[0])
    outputs = []
    for t in range(z_seq.shape[0]):
        if t >= max_delay_steps:
            spike = z_seq[t - max_delay_steps]
        else:
            spike = torch.zeros_like(psc)
        psc = psc * decay + spike
        outputs.append(psc.clone())
    return torch.stack(outputs, dim=0)


def test_delayed_psc_exact_delay_matches_manual_computation():
    """DelayedPSC should delay spikes by exactly max_delay_steps."""
    dt = 1.0
    tau_syn = 2.0
    max_delay_steps = 3
    n_neuron = 4

    # Deterministic spike sequence
    z_seq = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    linear = torch.nn.Linear(n_neuron, n_neuron, bias=False)
    torch.nn.init.eye_(linear.weight)

    with environ.context(dt=dt):
        delayed = DelayedPSC(
            ExponentialPSC(
                n_neuron=n_neuron,
                tau_syn=tau_syn,
                linear=linear,
                step_mode="m",
            ),
            max_delay_steps=max_delay_steps,
        )
        init_net_state(delayed, dtype=torch.float32)
        out = delayed(z_seq)

    expected = _expected_delayed_exponential_psc(
        z_seq, dt=dt, tau_syn=tau_syn, max_delay_steps=max_delay_steps
    )
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0.0)


def test_delayed_psc_passthrough_when_max_delay_steps_one():
    """max_delay_steps=1 should behave identically to the raw PSC (no
    delay)."""
    dt = 1.0
    tau_syn = 2.0
    n_neuron = 3

    z_seq = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    linear = torch.nn.Linear(n_neuron, n_neuron, bias=False)
    torch.nn.init.eye_(linear.weight)

    with environ.context(dt=dt):
        raw_psc = ExponentialPSC(
            n_neuron=n_neuron, tau_syn=tau_syn, linear=linear, step_mode="m"
        )
        delayed = DelayedPSC(
            ExponentialPSC(
                n_neuron=n_neuron, tau_syn=tau_syn, linear=linear, step_mode="m"
            ),
            max_delay_steps=1,
        )
        init_net_state(raw_psc, dtype=torch.float32)
        init_net_state(delayed, dtype=torch.float32)

        out_raw = raw_psc(z_seq)
        out_delayed = delayed(z_seq)

    torch.testing.assert_close(out_delayed, out_raw, atol=1e-6, rtol=0.0)


def test_delayed_psc_cat_and_circular_buffers_agree():
    """Cat-mode and circular-buffer-mode DelayedPSC must produce identical
    outputs."""
    dt = 1.0
    tau_syn = 3.0
    max_delay_steps = 4
    n_neuron = 5
    steps = 10

    torch.manual_seed(42)
    z_seq = torch.randn(steps, 2, n_neuron)

    linear = torch.nn.Linear(n_neuron, n_neuron, bias=False)
    torch.nn.init.xavier_uniform_(linear.weight)

    with environ.context(dt=dt):
        cat_delayed = DelayedPSC(
            ExponentialPSC(
                n_neuron=n_neuron,
                tau_syn=tau_syn,
                linear=linear,
                step_mode="m",
            ),
            max_delay_steps=max_delay_steps,
            use_circular_buffer=False,
        )
        circ_delayed = DelayedPSC(
            ExponentialPSC(
                n_neuron=n_neuron,
                tau_syn=tau_syn,
                linear=linear,
                step_mode="m",
            ),
            max_delay_steps=max_delay_steps,
            use_circular_buffer=True,
        )
        # Share weights so dynamics are identical
        circ_delayed.psc_module.linear.weight.data.copy_(
            cat_delayed.psc_module.linear.weight.data
        )

        init_net_state(cat_delayed, batch_size=2, dtype=torch.float32)
        init_net_state(circ_delayed, batch_size=2, dtype=torch.float32)

        out_cat = cat_delayed(z_seq)
        out_circ = circ_delayed(z_seq)

    torch.testing.assert_close(out_cat, out_circ, atol=1e-6, rtol=0.0)


def test_delayed_psc_initial_steps_are_zero_due_to_delay():
    """Before max_delay_steps have elapsed, output should be pure decay from
    zero."""
    dt = 1.0
    tau_syn = 2.0
    max_delay_steps = 3
    n_neuron = 2

    z_seq = torch.ones(3, 1, n_neuron, dtype=torch.float32)

    linear = torch.nn.Identity()

    with environ.context(dt=dt):
        delayed = DelayedPSC(
            ExponentialPSC(
                n_neuron=n_neuron,
                tau_syn=tau_syn,
                linear=linear,
                step_mode="m",
            ),
            max_delay_steps=max_delay_steps,
        )
        init_net_state(delayed, batch_size=1, dtype=torch.float32)
        out = delayed(z_seq)

    # For t < max_delay_steps, no spikes have arrived yet, so PSC should be zero
    for t in range(min(max_delay_steps, out.shape[0])):
        assert torch.allclose(
            out[t], torch.zeros_like(out[t]), atol=1e-6
        ), f"PSC at step {t} should be zero before delay elapses"


def test_delayed_psc_gradients_flow_and_are_nonzero():
    """Gradients should flow through DelayedPSC history and weight updates."""
    dt = 1.0
    tau_syn = 2.0
    max_delay_steps = 3
    n_neuron = 3
    steps = 5

    torch.manual_seed(7)
    z_seq = torch.randn(steps, 1, n_neuron, requires_grad=True)

    linear = torch.nn.Linear(n_neuron, n_neuron, bias=False)
    torch.nn.init.xavier_uniform_(linear.weight)

    with environ.context(dt=dt):
        delayed = DelayedPSC(
            ExponentialPSC(
                n_neuron=n_neuron,
                tau_syn=tau_syn,
                linear=linear,
                step_mode="m",
            ),
            max_delay_steps=max_delay_steps,
            use_circular_buffer=False,
        )
        init_net_state(delayed, batch_size=1, dtype=torch.float32)
        out = delayed(z_seq)

    loss = out.sum()
    loss.backward()

    assert z_seq.grad is not None
    assert delayed.psc_module.linear.weight.grad is not None
    # Because spikes from early timesteps affect later outputs through the delay,
    # input gradients should be non-zero.
    assert z_seq.grad.abs().sum() > 0
    assert delayed.psc_module.linear.weight.grad.abs().sum() > 0


def test_delayed_psc_finite_difference_weight_gradient():
    """Finite-difference check on a single linear weight element."""
    dt = 1.0
    max_delay_steps = 2
    n_neuron = 2
    steps = 4

    torch.manual_seed(3)
    z_seq = torch.randn(steps, 1, n_neuron)

    def make_model():
        linear = torch.nn.Linear(n_neuron, n_neuron, bias=False)
        torch.nn.init.xavier_uniform_(linear.weight)
        with environ.context(dt=dt):
            delayed = DelayedPSC(
                ExponentialPSC(
                    n_neuron=n_neuron,
                    tau_syn=1.5,
                    linear=linear,
                    step_mode="m",
                ),
                max_delay_steps=max_delay_steps,
                use_circular_buffer=False,
            )
        return delayed

    model = make_model()
    init_net_state(model, batch_size=1, dtype=torch.float32)
    with environ.context(dt=dt):
        out = model(z_seq)
    loss = out.sum()
    loss.backward()
    analytical_grad = model.psc_module.linear.weight.grad[0, 0].item()

    # Finite difference
    eps = 1e-4
    model_plus = make_model()
    model_plus.psc_module.linear.weight.data.copy_(
        model.psc_module.linear.weight.data.detach()
    )
    model_plus.psc_module.linear.weight.data[0, 0] += eps
    init_net_state(model_plus, batch_size=1, dtype=torch.float32)
    with environ.context(dt=dt):
        out_plus = model_plus(z_seq)
    loss_plus = out_plus.sum()

    model_minus = make_model()
    model_minus.psc_module.linear.weight.data.copy_(
        model.psc_module.linear.weight.data.detach()
    )
    model_minus.psc_module.linear.weight.data[0, 0] -= eps
    init_net_state(model_minus, batch_size=1, dtype=torch.float32)
    with environ.context(dt=dt):
        out_minus = model_minus(z_seq)
    loss_minus = out_minus.sum()

    fd_grad = (loss_plus.item() - loss_minus.item()) / (2 * eps)
    assert abs(analytical_grad - fd_grad) < 1e-3


def test_delayed_psc_multi_dim_neuron_axes_exact_values():
    """Verify exact delayed values with multi-dimensional neuron axes."""
    neuron_shape = (2, 3)
    dt = 1.0
    tau_syn = 1.0  # fast decay for easy mental math
    max_delay_steps = 2

    # Single batch, deterministic spikes
    z_seq = torch.zeros(4, 1, *neuron_shape, dtype=torch.float32)
    z_seq[0, 0, 0, 0] = 1.0
    z_seq[1, 0, 1, 2] = 2.0
    z_seq[2, 0, 0, 1] = 3.0
    z_seq[3, 0, 1, 1] = 4.0

    linear = torch.nn.Identity()

    with environ.context(dt=dt):
        delayed = DelayedPSC(
            ExponentialPSC(
                n_neuron=neuron_shape,
                tau_syn=tau_syn,
                linear=linear,
                step_mode="m",
            ),
            max_delay_steps=max_delay_steps,
        )
        init_net_state(delayed, batch_size=1, dtype=torch.float32)
        out = delayed(z_seq)

    # With tau_syn=1.0 and dt=1.0, decay factor = exp(-1) ~ 0.3679
    decay = torch.exp(torch.tensor(-1.0, dtype=torch.float32))

    # t=0: no spike arrived yet -> 0
    assert torch.allclose(out[0], torch.zeros_like(out[0]), atol=1e-6)
    # t=1: no spike arrived yet -> 0
    assert torch.allclose(out[1], torch.zeros_like(out[1]), atol=1e-6)
    # t=2: spike from t=0 arrives: [1,0,0,0,0,0] at neuron (0,0)
    assert torch.allclose(out[2, 0, 0, 0], torch.tensor(1.0), atol=1e-6)
    # t=3: spike from t=1 arrives (2.0 at 1,2), plus decay of previous
    expected_3_1_2 = 2.0
    expected_3_0_0 = 1.0 * decay
    assert torch.allclose(out[3, 0, 1, 2], torch.tensor(expected_3_1_2), atol=1e-6)
    assert torch.allclose(out[3, 0, 0, 0], torch.as_tensor(expected_3_0_0), atol=1e-6)


def test_delayed_psc_reset_clears_history_and_psc():
    """Reset must zero out both the inner PSC and the history buffer."""
    dt = 1.0
    max_delay_steps = 3
    psc = ExponentialPSC(n_neuron=5, tau_syn=2.0, linear=torch.nn.Identity())
    delayed = DelayedPSC(psc, max_delay_steps=max_delay_steps, use_circular_buffer=True)

    init_net_state(delayed, batch_size=1, dtype=torch.float32)

    z = torch.ones(1, 5)
    with environ.context(dt=dt):
        # Run enough steps so the delayed spike reaches the PSC
        for _ in range(max_delay_steps + 1):
            delayed(z)

    # History contains spikes and PSC is non-zero after delay elapses
    assert delayed.history.history.abs().sum() > 0
    assert delayed.psc.abs().sum() > 0

    reset_net(delayed, batch_size=1)

    assert delayed.history.history.abs().sum() == 0
    assert delayed.psc.abs().sum() == 0

    # After reset, running again from the same input should give the same result
    with environ.context(dt=dt):
        delayed.init_state(batch_size=1, dtype=torch.float32)
        for _ in range(max_delay_steps + 1):
            out_after_reset = delayed(z)

    # Re-init fresh and compare
    delayed.init_state(batch_size=1, dtype=torch.float32)
    with environ.context(dt=dt):
        for _ in range(max_delay_steps + 1):
            out_fresh = delayed(z)

    torch.testing.assert_close(out_after_reset, out_fresh, atol=1e-6, rtol=0.0)


def test_delayed_psc_in_recurrent_nn_end_to_end():
    """DelayedPSC should work transparently inside RecurrentNN.

    Note: neuron and PSC use step_mode='s' because RecurrentNN handles the
    time loop; RecurrentNN itself uses step_mode='m'.
    """
    dt = 1.0
    n_neuron = 4

    neuron = LIF(n_neuron=n_neuron, v_threshold=1.0, v_reset=0.0, step_mode="s")
    psc = DelayedPSC(
        ExponentialPSC(
            n_neuron=n_neuron,
            tau_syn=2.0,
            linear=torch.nn.Linear(n_neuron, n_neuron, bias=False),
            step_mode="s",
        ),
        max_delay_steps=2,
    )
    torch.nn.init.eye_(psc.psc_module.linear.weight)

    brain = RecurrentNN(
        neuron=neuron,
        synapse=psc,
        step_mode="m",
        update_state_names=("neuron.v", "synapse.psc"),
    )

    init_net_state(brain, batch_size=2, dtype=torch.float32)

    # Strong input to guarantee spikes
    x = torch.ones(5, 2, n_neuron) * 2.0

    with environ.context(dt=dt):
        spikes, states = brain(x)

    assert spikes.shape == (5, 2, n_neuron)

    # Compare with a non-delayed brain: voltage traces should diverge once the
    # non-delayed synapse starts contributing current while the delayed one
    # still reads zero.
    neuron2 = LIF(n_neuron=n_neuron, v_threshold=1.0, v_reset=0.0, step_mode="s")
    psc2 = ExponentialPSC(
        n_neuron=n_neuron,
        tau_syn=2.0,
        linear=torch.nn.Linear(n_neuron, n_neuron, bias=False),
        step_mode="s",
    )
    torch.nn.init.eye_(psc2.linear.weight)
    brain_no_delay = RecurrentNN(
        neuron=neuron2,
        synapse=psc2,
        step_mode="m",
        update_state_names=("neuron.v",),
    )
    init_net_state(brain_no_delay, batch_size=2, dtype=torch.float32)

    with environ.context(dt=dt):
        spikes_no_delay, states_no_delay = brain_no_delay(x)

    # Both brains spike at t=0 because external input x=2.0 exceeds threshold.
    # At t=1 the delayed brain still receives zero synaptic current, while the
    # non-delayed brain receives current from the t=0 spikes -> voltages differ.
    v_delayed = states["neuron.v"]
    v_no_delay = states_no_delay["neuron.v"]
    assert torch.allclose(v_delayed[0], v_no_delay[0], atol=1e-6)
    assert not torch.allclose(v_delayed[1], v_no_delay[1], atol=1e-3)


def test_delayed_psc_with_dual_exponential_exact_values():
    """DualExponentialPSC wrapped in DelayedPSC should produce expected
    delays."""
    dt = 1.0
    max_delay_steps = 2
    n_neuron = 2

    # Only one spike at t=0 for neuron 0
    z_seq = torch.zeros(5, 1, n_neuron, dtype=torch.float32)
    z_seq[0, 0, 0] = 1.0

    linear = torch.nn.Identity()

    with environ.context(dt=dt):
        delayed = DelayedPSC(
            DualExponentialPSC(
                n_neuron=n_neuron,
                tau_decay=5.0,
                tau_rise=1.0,
                linear=linear,
                step_mode="m",
            ),
            max_delay_steps=max_delay_steps,
        )
        init_net_state(delayed, batch_size=1, dtype=torch.float32)
        out = delayed(z_seq)

    # Before delay elapses, no spike has reached PSC -> PSC should be 0
    assert torch.allclose(out[0], torch.zeros_like(out[0]), atol=1e-6)
    assert torch.allclose(out[1], torch.zeros_like(out[1]), atol=1e-6)
    # At t=2 the spike finally arrives. For DualExponentialPSC with identity
    # weights, g_rise and g_decay both increase, so immediate psc is zero;
    # the PSC becomes non-zero in the next step as rise/decay evolve.
    psc_module = delayed.psc_module
    assert psc_module.g_rise[0, 0].item() != 0.0
    assert psc_module.g_decay[0, 0].item() != 0.0
    # Neuron 1 never received any spike -> should stay 0
    assert torch.allclose(out[:, 0, 1], torch.zeros(out.shape[0]), atol=1e-6)


def test_delayed_psc_single_neuron_edge_case():
    """DelayedPSC should work with a single neuron (n_neuron=1)."""
    dt = 1.0
    max_delay_steps = 3

    z_seq = torch.tensor(
        [
            [[1.0]],
            [[0.0]],
            [[0.0]],
            [[0.0]],
            [[2.0]],
        ],
        dtype=torch.float32,
    )

    linear = torch.nn.Identity()

    with environ.context(dt=dt):
        delayed = DelayedPSC(
            ExponentialPSC(
                n_neuron=1,
                tau_syn=1.0,
                linear=linear,
                step_mode="m",
            ),
            max_delay_steps=max_delay_steps,
        )
        init_net_state(delayed, batch_size=1, dtype=torch.float32)
        out = delayed(z_seq)

    # t=0,1,2: no spike arrived yet
    for t in range(3):
        assert torch.allclose(out[t], torch.zeros_like(out[t]), atol=1e-6)
    # t=3: spike from t=0 arrives
    assert torch.allclose(out[3], torch.tensor([[[1.0]]]), atol=1e-6)
    # t=4: spike from t=1 is zero, previous decays
    decay = torch.exp(torch.tensor(-1.0, dtype=torch.float32))
    assert torch.allclose(out[4], torch.tensor([[[1.0 * decay]]]), atol=1e-6)


def test_delayed_psc_batch_with_different_spike_patterns():
    """Different batch elements should evolve independently."""
    dt = 1.0
    max_delay_steps = 2
    n_neuron = 3

    # Batch of 2 with completely different spike patterns
    z_seq = torch.zeros(4, 2, n_neuron, dtype=torch.float32)
    z_seq[0, 0, 0] = 1.0  # batch 0 spikes at t=0, neuron 0
    z_seq[1, 1, 2] = 2.0  # batch 1 spikes at t=1, neuron 2

    linear = torch.nn.Identity()

    with environ.context(dt=dt):
        delayed = DelayedPSC(
            ExponentialPSC(
                n_neuron=n_neuron,
                tau_syn=1.0,
                linear=linear,
                step_mode="m",
            ),
            max_delay_steps=max_delay_steps,
        )
        init_net_state(delayed, batch_size=2, dtype=torch.float32)
        out = delayed(z_seq)

    # Batch 0: first non-zero PSC at t=2 (delay of spike at t=0)
    assert torch.allclose(out[0:2, 0], torch.zeros_like(out[0:2, 0]), atol=1e-6)
    assert torch.allclose(out[2, 0, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(out[2, 0, 1:], torch.zeros(2), atol=1e-6)

    # Batch 1: first non-zero PSC at t=3 (delay of spike at t=1)
    assert torch.allclose(out[0:3, 1], torch.zeros_like(out[0:3, 1]), atol=1e-6)
    assert torch.allclose(out[3, 1, 2], torch.tensor(2.0), atol=1e-6)
    assert torch.allclose(out[3, 1, 0:2], torch.zeros(2), atol=1e-6)


def test_delayed_psc_step_mode_s_vs_m_consistency():
    """Single-step and multi-step modes should produce identical sequences."""
    dt = 1.0
    max_delay_steps = 2
    n_neuron = 3
    steps = 5

    torch.manual_seed(99)
    z_seq = torch.randn(steps, 1, n_neuron)

    linear = torch.nn.Linear(n_neuron, n_neuron, bias=False)
    torch.nn.init.xavier_uniform_(linear.weight)

    with environ.context(dt=dt):
        delayed_m = DelayedPSC(
            ExponentialPSC(
                n_neuron=n_neuron,
                tau_syn=2.0,
                linear=linear,
                step_mode="m",
            ),
            max_delay_steps=max_delay_steps,
        )
        init_net_state(delayed_m, batch_size=1, dtype=torch.float32)
        out_m = delayed_m(z_seq)

    # Single-step mode: same weights, same init, process step-by-step
    linear_s = torch.nn.Linear(n_neuron, n_neuron, bias=False)
    linear_s.weight.data.copy_(linear.weight.data)

    with environ.context(dt=dt):
        delayed_s = DelayedPSC(
            ExponentialPSC(
                n_neuron=n_neuron,
                tau_syn=2.0,
                linear=linear_s,
                step_mode="s",
            ),
            max_delay_steps=max_delay_steps,
        )
        init_net_state(delayed_s, batch_size=1, dtype=torch.float32)
        out_s = []
        for t in range(steps):
            out_s.append(delayed_s(z_seq[t]))
        out_s = torch.stack(out_s)

    torch.testing.assert_close(out_m, out_s, atol=1e-6, rtol=0.0)
