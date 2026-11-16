"""Plug a handwritten Triton forward kernel into btorch's existing LIF.

Run from the repository root: ``python -m examples.oot_cell_backend``.
The example contains three parts: the kernel, its tensor callback, and normal
Module usage. It reuses btorch's existing LIF backward kernel rather than
implementing a second surrogate-gradient system. CUDA and Triton are required.
"""

import torch
import triton
import triton.language as tl

from btorch.models import environ
from btorch.models.kernels import (
    HistoryPlan,
    NeuronSpec,
    register_backend,
    resolve_cell_backend,
)
from btorch.models.kernels.testing import (
    ParityCase,
    ParityOptions,
    check_backend_parity,
)
from btorch.models.neurons import LIF


@triton.jit
def lif_kernel(
    x_ptr,
    initial_ptr,
    output_ptr,
    final_ptr,
    voltage_ptr,
    charged_ptr,
    parameters,
    parameter_strides: tl.constexpr,
    dt_ptr,
    steps,
    BATCH: tl.constexpr,
    NEURONS: tl.constexpr,
    X_STRIDES: tl.constexpr,
    STATE_STRIDES: tl.constexpr,
    HARD_RESET: tl.constexpr,
    RECORD_V: tl.constexpr,
    RECORD_CHARGED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One lane owns a neuron across the sequence; no cross-block synchronization.
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = lane < BATCH * NEURONS
    batch, neuron = lane // NEURONS, lane % NEURONS
    voltage = tl.load(
        initial_ptr + batch * STATE_STRIDES[0] + neuron * STATE_STRIDES[1],
        mask,
        other=0,
    )
    threshold = tl.load(
        parameters[0]
        + batch * parameter_strides[0][0]
        + neuron * parameter_strides[0][1],
        mask,
        other=1,
    )
    reset = tl.load(
        parameters[1]
        + batch * parameter_strides[1][0]
        + neuron * parameter_strides[1][1],
        mask,
        other=0,
    )
    capacitance = tl.load(
        parameters[2]
        + batch * parameter_strides[2][0]
        + neuron * parameter_strides[2][1],
        mask,
        other=1,
    )
    tau = tl.load(
        parameters[3]
        + batch * parameter_strides[3][0]
        + neuron * parameter_strides[3][1],
        mask,
        other=1,
    )
    dt = tl.load(dt_ptr)
    for time in range(steps):
        drive = tl.load(
            x_ptr + time * X_STRIDES[0] + batch * X_STRIDES[1] + neuron * X_STRIDES[2],
            mask,
            other=0,
        )
        charged = voltage + dt * (-(voltage - reset) / tau + drive / capacitance)
        # Use btorch's normalized threshold so its existing surrogate VJP
        # differentiates the same firing convention as this forward kernel.
        spike = (charged - threshold) / (threshold - reset) >= 0.0
        if HARD_RESET:
            voltage = tl.where(spike, reset, charged)
        else:
            voltage = charged - (threshold - reset) * spike
        # Persistent state rounds to its declared storage precision each step.
        voltage = voltage.to(final_ptr.dtype.element_ty)
        offset = time * BATCH * NEURONS + lane
        tl.store(output_ptr + offset, spike, mask)
        if RECORD_V:
            tl.store(voltage_ptr + offset, voltage, mask)
        if RECORD_CHARGED:
            tl.store(charged_ptr + offset, charged, mask)
    tl.store(final_ptr + lane, voltage, mask)


def forward(x_seq, states, parameters, spec, history_names):
    """Adapt btorch's ordered tensors to the native kernel without input
    copies.

    This teaching kernel handles time/batch/neuron inputs, scalar or
    broadcast parameters, and strided tensors. Refractory and stochastic
    LIF variants are deliberately outside its capability declaration.
    """
    if x_seq.ndim != 3:
        raise ValueError("This example expects (time, batch, neuron) input")
    steps, batch, neurons = x_seq.shape
    dtype = x_seq.dtype
    for value in (*states, *parameters):
        dtype = torch.promote_types(dtype, value.dtype)
    initial = states[0].expand(batch, neurons)
    coefficients = tuple(value.expand(batch, neurons) for value in parameters)
    output = torch.empty(x_seq.shape, device=x_seq.device, dtype=dtype)
    histories = {name: torch.empty_like(output) for name in history_names}
    if not steps:
        # No integration occurred: preserve state precision and its identity
        # derivative even when the empty drive has a wider calculation dtype.
        return (
            output,
            (initial.clone(),),
            tuple(histories[name] for name in history_names),
        )
    final = torch.empty((batch, neurons), device=x_seq.device, dtype=dtype)
    dt = torch.as_tensor(spec.dt, device=x_seq.device, dtype=dtype)
    lif_kernel[(triton.cdiv(batch * neurons, 128),)](
        x_seq,
        initial,
        output,
        final,
        histories.get("v", output),
        histories.get("v_charged", output),
        coefficients,
        tuple(value.stride() for value in coefficients),
        dt,
        steps,
        batch,
        neurons,
        x_seq.stride(),
        initial.stride(),
        spec.hard_reset,
        "v" in histories,
        "v_charged" in histories,
        128,
        enable_fp_fusion=False,
    )
    return output, (final,), tuple(histories[name] for name in history_names)


def register_example_backend():
    """Register our forward and reuse btorch's matching surrogate-gradient
    VJP."""
    builtin = resolve_cell_backend("triton", NeuronSpec("lif"), "cuda")
    register_backend(
        "tutorial_lif",
        forward=forward,
        backward=builtin.backward,
        supports=lambda spec, device: (
            spec.model == "lif"
            and device.type == "cuda"
            and not spec.refractory
            and spec.poisson is None
        ),
        replace=True,
    )


def run_example():
    """Validate against Torch, then train through the ordinary LIF Module."""
    register_example_backend()
    # A partial block, strided drive, heterogeneous parameters and nonzero
    # initial voltage make this a numerical check of the kernel itself.
    drive = torch.linspace(0.15, 0.85, 12 * 2 * 129 * 2, device="cuda").reshape(
        12, 2, 258
    )[..., ::2]
    initial = torch.full((2, 129), 0.15, device="cuda")
    parameters = tuple(
        torch.as_tensor(value, device="cuda")
        for value in (1.0, 0.0, torch.linspace(0.9, 1.1, 129), 4.0)
    )
    report = check_backend_parity(
        "tutorial_lif",
        [
            ParityCase(
                drive,
                (initial,),
                parameters,
                NeuronSpec("lif"),
                HistoryPlan(requested=("v", "v_charged")),
            )
        ],
        options=ParityOptions(compile=True, capture=True),
    )
    report.assert_close()

    neuron = LIF(129, tau=4.0, step_mode="m", backend="tutorial_lif", device="cuda")
    neuron.init_state(batch_size=2, device="cuda", dtype=drive.dtype)
    scale = torch.tensor(0.8, device="cuda", requires_grad=True)
    with environ.context(dt=1.0):
        spikes = neuron(scale * drive)
        loss = spikes.sum() + neuron.v.square().sum()
        loss.backward()
    assert scale.grad is not None and torch.isfinite(scale.grad)
    print(f"Kernel parity passed ({len(report.comparisons)} comparisons)")
    print(
        f"LIF output: {tuple(spikes.shape)}; encoder gradient: {scale.grad.item():.6f}"
    )
    return report


if __name__ == "__main__":
    run_example()
