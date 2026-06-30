import platform

import pytest
import torch

from btorch.models import environ
from btorch.models.base import MemoryModule
from btorch.models.functional import init_net_state
from btorch.models.history import SpikeHistory
from btorch.models.synapse import (
    AlphaPSC,
    AlphaPSCBilleh,
    DelayedPSC,
    DualExponentialPSC,
    ExponentialPSC,
    HeterSynapsePSC,
)
from tests.utils.compile import compile_or_skip


def _expected_exponential_psc(
    z_seq: torch.Tensor, dt: float, tau_syn: float, latency_steps: int
) -> torch.Tensor:
    # ExponentialPSC uses psc_{t+1} = psc_t * exp(-dt / tau_syn) + spike_{t-l}.
    decay = torch.exp(torch.tensor(-dt / tau_syn, dtype=z_seq.dtype))
    psc = torch.zeros_like(z_seq[0])
    expected = []

    for t in range(z_seq.shape[0]):
        if t >= latency_steps:
            spike = z_seq[t - latency_steps]
        else:
            spike = torch.zeros_like(psc)
        psc = psc * decay + spike
        expected.append(psc.clone())

    return torch.stack(expected, dim=0)


def test_exponential_psc_latency_matches_manual():
    # Use identity weights to isolate delay + exponential decay behavior.
    dt = 1.0
    tau_syn = 2.0
    max_delay_steps = 3
    n_neuron = 3

    z_seq = torch.tensor(
        [
            [1.0, 1.0, 0.5],
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )

    linear = torch.nn.Linear(n_neuron, n_neuron, bias=False)
    torch.nn.init.eye_(linear.weight)

    with environ.context(dt=dt):
        synapse = DelayedPSC(
            ExponentialPSC(
                n_neuron=n_neuron,
                tau_syn=tau_syn,
                linear=linear,
                step_mode="m",
            ),
            max_delay_steps=max_delay_steps,
        )
        init_net_state(synapse, dtype=torch.float32)
        out = synapse(z_seq)

    expected = _expected_exponential_psc(
        z_seq,
        dt=dt,
        tau_syn=tau_syn,
        latency_steps=max_delay_steps,
    )

    # _plot_exponential_psc(out, expected, dt=dt, name="exponential_psc_latency")

    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
def test_exponential_psc_latency_grad_matches_compile():
    # Compare eager vs compiled results and gradients under identical state.
    torch.manual_seed(123)

    dt = 1.0
    tau_syn = 3.0
    max_delay_steps = 3
    n_neuron = 4
    steps = 6

    z_seq = torch.randn(steps, n_neuron, requires_grad=True)
    z_seq_compiled = z_seq.clone().detach().requires_grad_(True)

    with environ.context(dt=dt):
        eager = DelayedPSC(
            ExponentialPSC(
                n_neuron=n_neuron,
                tau_syn=tau_syn,
                linear=torch.nn.Linear(n_neuron, n_neuron, bias=False),
                step_mode="m",
            ),
            max_delay_steps=max_delay_steps,
            use_circular_buffer=False,
        )
        compiled = DelayedPSC(
            ExponentialPSC(
                n_neuron=n_neuron,
                tau_syn=tau_syn,
                linear=torch.nn.Linear(n_neuron, n_neuron, bias=False),
                step_mode="m",
            ),
            max_delay_steps=max_delay_steps,
            use_circular_buffer=False,
        )

    compiled.psc_module.linear.weight.data.copy_(eager.psc_module.linear.weight.data)

    compiled = compile_or_skip(compiled)

    with environ.context(dt=dt):
        init_net_state(eager)
        init_net_state(compiled)

        out_eager = eager(z_seq)
        out_compiled = compiled(z_seq_compiled)

    torch.testing.assert_close(out_eager, out_compiled, atol=1e-6, rtol=0.0)

    loss_eager = out_eager.sum()
    loss_compiled = out_compiled.sum()

    loss_eager.backward()
    loss_compiled.backward()

    torch.testing.assert_close(z_seq.grad, z_seq_compiled.grad, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(
        eager.psc_module.linear.weight.grad,
        compiled.psc_module.linear.weight.grad,
        atol=1e-5,
        rtol=0.0,
    )


def _reference_loop_output(psc: MemoryModule, z_seq: torch.Tensor) -> torch.Tensor:
    T = z_seq.shape[0]
    y_seq = []
    for t in range(T):
        y = psc.single_step_forward(z_seq[t])
        y_seq.append(y)
    return torch.stack(y_seq)


def _conv_output(
    psc: MemoryModule, z_seq: torch.Tensor, kernel_len: int
) -> torch.Tensor:
    return psc.multi_step_forward(z_seq, kernel_len=kernel_len)


def _make_psc(psc_cls, n_neuron=4, **kwargs):
    linear = torch.nn.Linear(n_neuron, n_neuron, bias=False)
    torch.nn.init.eye_(linear.weight)
    if psc_cls is ExponentialPSC:
        return psc_cls(
            n_neuron=n_neuron, tau_syn=kwargs.get("tau_syn", 5.0), linear=linear
        )
    elif psc_cls is AlphaPSC:
        return psc_cls(
            n_neuron=n_neuron, tau_syn=kwargs.get("tau_syn", 5.0), linear=linear
        )
    elif psc_cls is AlphaPSCBilleh:
        return psc_cls(
            n_neuron=n_neuron, tau_syn=kwargs.get("tau_syn", 5.0), linear=linear
        )
    elif psc_cls is DualExponentialPSC:
        return psc_cls(
            n_neuron=n_neuron,
            tau_decay=kwargs.get("tau_decay", 20.0),
            tau_rise=kwargs.get("tau_rise", 5.0),
            linear=linear,
        )
    else:
        raise ValueError(f"Unknown PSC class: {psc_cls}")


PSC_CLASSES = [ExponentialPSC, AlphaPSC, AlphaPSCBilleh, DualExponentialPSC]
KERNEL_LENS = [64, 128]
BATCH_SHAPES = [(), (2,), (2, 3)]
TIME_STEPS = [1, 16, 64]


@pytest.mark.parametrize("psc_cls", PSC_CLASSES)
@pytest.mark.parametrize("kernel_len", KERNEL_LENS)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("T", TIME_STEPS)
def test_psc_multistep_conv_parity(psc_cls, kernel_len, batch_shape, T):
    n_neuron = 4
    torch.manual_seed(42)

    dt = 1.0
    batch_dims = batch_shape
    z_seq = torch.randn(T, *batch_dims, n_neuron)
    batch_size = batch_dims if len(batch_dims) > 0 else None

    with environ.context(dt=dt):
        psc = _make_psc(psc_cls, n_neuron=n_neuron)
        init_net_state(
            psc, batch_size=batch_size, dtype=z_seq.dtype, device=z_seq.device
        )
        loop_out = _reference_loop_output(psc, z_seq)

        psc.reset()
        init_net_state(
            psc, batch_size=batch_size, dtype=z_seq.dtype, device=z_seq.device
        )
        conv_out = _conv_output(psc, z_seq, kernel_len)

    torch.testing.assert_close(loop_out, conv_out, atol=1e-5, rtol=1e-5)


def test_delayed_psc_multistep_conv_parity():
    n_neuron = 4
    T = 16
    kernel_len = 64
    max_delay_steps = 3
    torch.manual_seed(42)

    dt = 1.0
    linear = torch.nn.Linear(n_neuron, n_neuron, bias=False)
    torch.nn.init.eye_(linear.weight)
    base_psc = ExponentialPSC(n_neuron=n_neuron, tau_syn=5.0, linear=linear)
    psc = DelayedPSC(base_psc, max_delay_steps=max_delay_steps)

    z_seq = torch.randn(T, n_neuron)

    with environ.context(dt=dt):
        init_net_state(psc, batch_size=None, dtype=z_seq.dtype, device=z_seq.device)
        loop_out = _reference_loop_output(psc, z_seq)

        psc.reset()
        init_net_state(psc, batch_size=None, dtype=z_seq.dtype, device=z_seq.device)
        conv_out = _conv_output(psc, z_seq, kernel_len)

    torch.testing.assert_close(loop_out, conv_out, atol=1e-5, rtol=1e-5)


def test_heter_synapse_psc_multistep_conv_parity():
    n_neuron = 4
    n_receptor = 3
    T = 16
    kernel_len = 64
    torch.manual_seed(42)

    dt = 1.0
    linear = torch.nn.Linear(n_neuron * n_receptor, n_neuron * n_receptor, bias=False)
    torch.nn.init.eye_(linear.weight)

    psc = HeterSynapsePSC(
        n_neuron=n_neuron,
        n_receptor=n_receptor,
        receptor_type_index=None,
        linear=linear,
        base_psc=AlphaPSC,
        max_delay_steps=1,
        tau_syn=5.0,
    )

    z_seq = torch.randn(T, n_neuron, n_receptor)

    with environ.context(dt=dt):
        init_net_state(psc, batch_size=None, dtype=z_seq.dtype, device=z_seq.device)
        loop_out = _reference_loop_output(psc, z_seq)

        psc.reset()
        init_net_state(psc, batch_size=None, dtype=z_seq.dtype, device=z_seq.device)
        conv_out = _conv_output(psc, z_seq, kernel_len)

    torch.testing.assert_close(loop_out, conv_out, atol=1e-5, rtol=1e-5)


def test_heter_synapse_psc_multistep_conv_parity_raw_input():
    """Validate convolution parity for raw-spike heterosynapse inputs.

    This covers the path where HeterSynapsePSC receives spikes in shape
    ``(*batch, n_neuron)`` (without an explicit receptor axis) when
    ``max_delay_steps == 1``.

    Why this matters:
    - There are two supported input layouts for non-delayed heter synapses:
      raw ``n_neuron`` spikes and receptor-expanded ``n_neuron x n_receptor``.
    - The convolution-based ``multi_step_forward`` must match the reference
      single-step rollout for both layouts.
    """

    n_neuron = 5
    n_receptor = 3
    T = 20
    kernel_len = 64
    batch_shape = (2,)
    torch.manual_seed(7)

    dt = 1.0
    linear = torch.nn.Linear(n_neuron, n_neuron * n_receptor, bias=False)

    psc = HeterSynapsePSC(
        n_neuron=n_neuron,
        n_receptor=n_receptor,
        receptor_type_index=None,
        linear=linear,
        base_psc=ExponentialPSC,
        max_delay_steps=1,
        tau_syn=4.0,
    )

    # Raw spike layout: (T, *batch, n_neuron)
    z_seq = torch.randn(T, *batch_shape, n_neuron)

    with environ.context(dt=dt):
        init_net_state(
            psc, batch_size=batch_shape, dtype=z_seq.dtype, device=z_seq.device
        )
        loop_out = _reference_loop_output(psc, z_seq)

        psc.reset()
        init_net_state(
            psc, batch_size=batch_shape, dtype=z_seq.dtype, device=z_seq.device
        )
        conv_out = _conv_output(psc, z_seq, kernel_len)

    torch.testing.assert_close(loop_out, conv_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("use_circular_buffer", [True, False])
def test_heter_synapse_psc_delay_multistep_matches_manual(use_circular_buffer):
    """Check delayed heterosynapse multi-step against manual history pipeline.

    This test verifies the full delayed execution path used by
    ``HeterSynapsePSC.multi_step_forward``:
    1) push raw spikes into SpikeHistory,
    2) flatten delayed spikes,
    3) apply base PSC dynamics and linear map,
    4) sum receptor channels per neuron.

    It runs in both buffer modes (circular and cat) to ensure implementation
    details of history storage do not change numerical behavior.
    """

    n_neuron = 6
    n_receptor = 2
    max_delay_steps = 3
    T = 12
    batch_shape = (3,)
    torch.manual_seed(11)

    dt = 1.0
    linear = torch.nn.Linear(
        n_neuron * max_delay_steps,
        n_neuron * n_receptor,
        bias=False,
    )

    hetero_psc = HeterSynapsePSC(
        n_neuron=n_neuron,
        n_receptor=n_receptor,
        receptor_type_index=None,
        linear=linear,
        base_psc=ExponentialPSC,
        max_delay_steps=max_delay_steps,
        use_circular_buffer=use_circular_buffer,
        tau_syn=3.0,
    )

    manual_history = SpikeHistory(
        n_neuron=n_neuron,
        max_delay_steps=max_delay_steps,
        use_circular_buffer=use_circular_buffer,
    )
    manual_base = ExponentialPSC(
        n_neuron=n_neuron * n_receptor,
        tau_syn=3.0,
        linear=linear,
    )

    z_seq = torch.randn(T, *batch_shape, n_neuron)

    with environ.context(dt=dt):
        init_net_state(
            hetero_psc,
            batch_size=batch_shape,
            dtype=z_seq.dtype,
            device=z_seq.device,
        )
        init_net_state(
            manual_history,
            batch_size=batch_shape,
            dtype=z_seq.dtype,
            device=z_seq.device,
        )
        init_net_state(
            manual_base,
            batch_size=batch_shape,
            dtype=z_seq.dtype,
            device=z_seq.device,
        )

        # Path under test: class-provided multi-step API.
        out_multi = hetero_psc.multi_step_forward(z_seq, kernel_len=64)

        # Reference path: explicit step-by-step SpikeHistory + BasePSC.
        out_manual = []
        for t in range(T):
            manual_history.update(z_seq[t])
            z_delayed = manual_history.get_flattened(max_delay_steps)
            psc_flat = manual_base.single_step_forward(z_delayed)
            psc = psc_flat.view(*psc_flat.shape[:-1], n_neuron, n_receptor).sum(-1)
            out_manual.append(psc)
        out_manual = torch.stack(out_manual)

    torch.testing.assert_close(out_multi, out_manual, atol=1e-6, rtol=0.0)


def test_heter_synapse_psc_delay_rejects_receptor_axis_input():
    """Ensure delayed heterosynapse enforces raw-spike input layout.

    Delayed HeterSynapsePSC internally owns SpikeHistory over raw spikes
    ``(*batch, n_neuron)``. Passing receptor-expanded tensors would represent
    a different semantic object and can silently produce wrong delay shapes.

    The model should fail fast for both single-step and multi-step calls.
    """

    n_neuron = 4
    n_receptor = 2
    max_delay_steps = 3
    dt = 1.0

    linear = torch.nn.Linear(
        n_neuron * max_delay_steps,
        n_neuron * n_receptor,
        bias=False,
    )
    psc = HeterSynapsePSC(
        n_neuron=n_neuron,
        n_receptor=n_receptor,
        receptor_type_index=None,
        linear=linear,
        base_psc=AlphaPSC,
        tau_syn=2.0,
        max_delay_steps=max_delay_steps,
    )

    z_single = torch.randn(2, n_neuron, n_receptor)
    z_multi = torch.randn(5, 2, n_neuron, n_receptor)

    with environ.context(dt=dt):
        init_net_state(psc, batch_size=2, dtype=torch.float32)

        with pytest.raises(RuntimeError, match="expects input without receptor axis"):
            psc.single_step_forward(z_single)

        with pytest.raises(RuntimeError, match="expects input without receptor axis"):
            psc.multi_step_forward(z_multi, kernel_len=64)


@pytest.mark.skipif(
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
@pytest.mark.parametrize("psc_cls", [ExponentialPSC, DualExponentialPSC])
def test_psc_multistep_conv_compile_parity(psc_cls, T=16, n_neuron=4, kernel_len=64):
    torch.manual_seed(123)
    dt = 1.0

    if psc_cls is ExponentialPSC:
        linear = torch.nn.Linear(n_neuron, n_neuron, bias=False)
        torch.nn.init.eye_(linear.weight)
        psc = psc_cls(n_neuron=n_neuron, tau_syn=5.0, linear=linear)
    else:
        linear = torch.nn.Linear(n_neuron, n_neuron, bias=False)
        torch.nn.init.eye_(linear.weight)
        psc = psc_cls(n_neuron=n_neuron, tau_decay=20.0, tau_rise=5.0, linear=linear)

    psc_compiled = compile_or_skip(psc)

    z_seq = torch.randn(T, n_neuron, requires_grad=True)
    z_seq_compiled = z_seq.clone().detach().requires_grad_(True)

    with environ.context(dt=dt):
        init_net_state(psc)
        init_net_state(psc_compiled)

        out_eager = psc.multi_step_forward(z_seq, kernel_len=kernel_len)
        out_compiled = psc_compiled.multi_step_forward(
            z_seq_compiled, kernel_len=kernel_len
        )

    torch.testing.assert_close(out_eager, out_compiled, atol=1e-6, rtol=0.0)

    loss_eager = out_eager.sum()
    loss_compiled = out_compiled.sum()

    loss_eager.backward()
    loss_compiled.backward()

    torch.testing.assert_close(z_seq.grad, z_seq_compiled.grad, atol=1e-5, rtol=0.0)
