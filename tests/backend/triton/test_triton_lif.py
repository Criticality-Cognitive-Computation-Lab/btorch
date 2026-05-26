import pytest
import torch

from btorch.backend.triton.lif import triton_lif_single_step
from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.neurons.lif import LIF


def _build_input(
    *,
    steps: int,
    batch_size: int,
    n_neuron: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x_seq = torch.randn(
        (steps, batch_size, n_neuron),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return 0.25 * x_seq + 0.3


def _build_loss_weight(
    *,
    steps: int,
    batch_size: int,
    n_neuron: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        (steps, batch_size, n_neuron),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def _torch_reference_soft_lif_multistep(
    x_seq: torch.Tensor,
    *,
    dt: float,
    v_threshold: float,
    v_reset: float,
    c_m: float,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This is the native PyTorch reference for the exact mode implemented by the
    # Triton backward in this test: soft reset, no refractory period, and
    # gradients only through the spike sequence. The forward uses a hard
    # threshold, while the backward is supplied by the smooth sigmoid branch via
    # the standard detach-based surrogate trick, so autograd is entirely native
    # PyTorch here.
    v = torch.full_like(x_seq[0], v_reset)
    spikes = []
    scale = v_threshold - v_reset

    for x_t in x_seq:
        dv = -(v - v_reset) / tau + x_t / c_m
        v = v + dt * dv
        h = (v - v_threshold) / scale
        spike_hard = (h >= 0).to(v.dtype)
        spike_soft = torch.sigmoid(h)
        spike = spike_hard.detach() + spike_soft - spike_soft.detach()
        v = v - scale * spike
        spikes.append(spike)

    return torch.stack(spikes), v


def _project_lif_multistep(
    x_seq: torch.Tensor,
    *,
    dt: float,
    v_threshold: float,
    v_reset: float,
    c_m: float,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    neuron = LIF(
        n_neuron=x_seq.shape[-1],
        v_threshold=v_threshold,
        v_reset=v_reset,
        c_m=c_m,
        tau=tau,
        tau_ref=None,
        hard_reset=False,
        step_mode="m",
        device=x_seq.device,
        dtype=x_seq.dtype,
    )
    init_net_state(
        neuron,
        batch_size=x_seq.shape[1],
        device=x_seq.device,
        dtype=x_seq.dtype,
    )
    with environ.context(dt=dt):
        spikes = neuron(x_seq)
    return spikes, neuron.v


def _triton_lif_multistep(
    x_seq: torch.Tensor,
    *,
    dt: float,
    v_threshold: float,
    v_reset: float,
    c_m: float,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    neuron = LIF(
        n_neuron=x_seq.shape[-1],
        v_threshold=v_threshold,
        v_reset=v_reset,
        c_m=c_m,
        tau=tau,
        tau_ref=None,
        hard_reset=False,
        backend="triton",
        step_mode="m",
        device=x_seq.device,
        dtype=x_seq.dtype,
    )
    init_net_state(
        neuron,
        batch_size=x_seq.shape[1],
        device=x_seq.device,
        dtype=x_seq.dtype,
    )
    with environ.context(dt=dt):
        spikes = neuron(x_seq)
    return spikes, neuron.v


def _torch_lif_single_step(
    x: torch.Tensor,
    v: torch.Tensor,
    *,
    dt: float,
    v_threshold: float,
    v_reset: float,
    c_m: float,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dv = -(v - v_reset) / tau + x / c_m
    v_next = v + dt * dv
    spike = (v_next >= v_threshold).to(v.dtype)
    v_next = v_next - (v_threshold - v_reset) * spike
    return spike, v_next

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_lif_soft_reset_x_grad_matches_pytorch_and_project_lif():
    device = "cuda"
    dtype = torch.float32
    steps = 20
    batch_size = 2
    n_neuron = 100
    dt = 1.0
    v_threshold = 1.0
    v_reset = 0.0
    c_m = 1.0
    tau = 20.0

    # The random weight makes the loss depend on every time step, which ensures
    # the input gradient exercises the temporal recurrence instead of only a
    # trivial final-step path.
    x_ref = _build_input(
        steps=steps,
        batch_size=batch_size,
        n_neuron=n_neuron,
        device=device,
        dtype=dtype,
        seed=0,
    ).requires_grad_(True)
    x_project = x_ref.detach().clone().requires_grad_(True)
    x_triton = x_ref.detach().clone().requires_grad_(True)
    loss_weight = _build_loss_weight(
        steps=steps,
        batch_size=batch_size,
        n_neuron=n_neuron,
        device=device,
        dtype=dtype,
        seed=1,
    )

    spikes_ref, v_ref = _torch_reference_soft_lif_multistep(
        x_ref,
        dt=dt,
        v_threshold=v_threshold,
        v_reset=v_reset,
        c_m=c_m,
        tau=tau,
    )
    loss_ref = (spikes_ref * loss_weight).sum()
    loss_ref.backward()

    spikes_project, v_project = _project_lif_multistep(
        x_project,
        dt=dt,
        v_threshold=v_threshold,
        v_reset=v_reset,
        c_m=c_m,
        tau=tau,
    )
    loss_project = (spikes_project * loss_weight).sum()
    loss_project.backward()

    spikes_triton, v_triton = _triton_lif_multistep(
        x_triton,
        dt=dt,
        v_threshold=v_threshold,
        v_reset=v_reset,
        c_m=c_m,
        tau=tau,
    )
    loss_triton = (spikes_triton * loss_weight).sum()
    loss_triton.backward()

    torch.testing.assert_close(spikes_project, spikes_ref, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(spikes_triton, spikes_ref, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(v_project, v_ref, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(v_triton, v_ref, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(x_project.grad, x_ref.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(x_triton.grad, x_ref.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        x_triton.grad,
        x_project.grad,
        atol=1e-6,
        rtol=1e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_lif_single_step_matches_torch():
    device = "cuda"
    dtype = torch.float32
    batch_size = 4
    n_neuron = 257
    dt = 1.0
    v_threshold = 1.0
    v_reset = 0.0
    c_m = 1.0
    tau = 20.0

    generator = torch.Generator(device=device)
    generator.manual_seed(3)
    x = 0.2 + 0.3 * torch.randn(
        (batch_size, n_neuron),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    v = 0.1 * torch.randn(
        (batch_size, n_neuron),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    spike_ref, v_ref = _torch_lif_single_step(
        x,
        v,
        dt=dt,
        v_threshold=v_threshold,
        v_reset=v_reset,
        c_m=c_m,
        tau=tau,
    )
    spike_triton, v_triton = triton_lif_single_step(
        x.reshape(-1),
        v.reshape(-1),
        torch.full_like(v.reshape(-1), v_threshold),
        torch.full_like(v.reshape(-1), v_reset),
        torch.full_like(v.reshape(-1), c_m),
        torch.full_like(v.reshape(-1), tau),
        dt=dt,
        hard_reset=False,
    )

    torch.testing.assert_close(
        spike_triton.reshape_as(x),
        spike_ref,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        v_triton.reshape_as(x),
        v_ref,
        atol=1e-6,
        rtol=0.0,
    )
