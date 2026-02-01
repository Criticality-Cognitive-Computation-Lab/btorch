from collections.abc import Callable

import matplotlib.pyplot as plt
import pytest
import torch

from benchmark.dense_glif_net.glif_common import GLIFDenseNet, build_neuron, make_inputs
from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.neurons.glif import GLIF3
from btorch.models.surrogate import ATan, ATanApprox
from btorch.models.surrogate.base import SurrogateFunctionBase
from btorch.utils.file import save_fig


def _base_inputs(B: int, M: int, device: torch.device, dtype: torch.dtype):
    # Deterministic inputs so kernel vs. reference comparisons are stable.
    gen = torch.Generator(device=device).manual_seed(123)
    v = -60.0 + 5.0 * torch.randn(B, generator=gen, device=device, dtype=dtype)
    x = 20.0 * torch.randn(B, generator=gen, device=device, dtype=dtype)
    Iasc = 0.05 * torch.randn(B, M, generator=gen, device=device, dtype=dtype)

    v_th = torch.full((B,), -50.0, device=device, dtype=dtype)
    v_reset = torch.full((B,), -70.0, device=device, dtype=dtype)
    v_rest = v_reset.clone()

    c_m = torch.full((B,), 0.05, device=device, dtype=dtype)
    tau = torch.full((B,), 20.0, device=device, dtype=dtype)
    k = 0.1 + 0.2 * torch.rand((B, M), generator=gen, device=device, dtype=dtype)
    asc_amps = 0.05 * torch.randn((B, M), generator=gen, device=device, dtype=dtype)

    return {
        "v": v,
        "x": x,
        "Iasc": Iasc,
        "v_th": v_th,
        "v_reset": v_reset,
        "v_rest": v_rest,
        "c_m": c_m,
        "tau": tau,
        "k": k,
        "asc_amps": asc_amps,
    }


def _reference_step(
    base,
    dt: float,
    hard_reset: bool,
    surrogate: SurrogateFunctionBase,
):
    B, M = base["Iasc"].shape
    glif = GLIF3(
        n_neuron=B,
        v_threshold=base["v_th"],
        v_reset=base["v_reset"],
        v_rest=base["v_rest"],
        c_m=base["c_m"],
        tau=base["tau"],
        k=base["k"],
        asc_amps=base["asc_amps"],
        tau_ref=0.0,
        hard_reset=hard_reset,
        surrogate_function=surrogate,
        trainable_param={"asc_amps"},
        device=base["v"].device,
        dtype=base["v"].dtype,
    )

    # Feed leaf tensors to the stateful module so autograd can track gradients.
    glif.v = base["v"]
    glif.Iasc = base["Iasc"]
    glif.refractory = torch.zeros_like(base["v"])

    with environ.context(dt=dt):
        spike = glif.single_step_forward(base["x"])

    return glif.v, glif.Iasc, spike, glif.asc_amps


def _reference_multistep(
    base,
    x_seq: torch.Tensor,
    dt: float,
    hard_reset: bool,
    surrogate: SurrogateFunctionBase,
):
    B, M = base["Iasc"].shape
    glif = GLIF3(
        n_neuron=B,
        v_threshold=base["v_th"],
        v_reset=base["v_reset"],
        v_rest=base["v_rest"],
        c_m=base["c_m"],
        tau=base["tau"],
        k=base["k"],
        asc_amps=base["asc_amps"],
        tau_ref=0.0,
        hard_reset=hard_reset,
        surrogate_function=surrogate,
        trainable_param={"asc_amps"},
        device=base["v"].device,
        dtype=base["v"].dtype,
    )

    glif.v = base["v"]
    glif.Iasc = base["Iasc"]
    glif.refractory = torch.zeros_like(base["v"])

    spikes = []
    v_seq = []
    with environ.context(dt=dt):
        for x_t in x_seq:
            s = glif.single_step_forward(x_t)
            spikes.append(s)
            v_seq.append(glif.v)

    spike_seq = torch.stack(spikes)
    v_seq = torch.stack(v_seq)
    return v_seq, glif.Iasc, spike_seq, glif.asc_amps


def _make_upstream(v, Iasc, s):
    # Use a deterministic upstream signal to make gradient comparisons stable.
    gen = torch.Generator(device=v.device).manual_seed(999)
    g_v = torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=gen)
    g_I = torch.randn(Iasc.shape, device=Iasc.device, dtype=Iasc.dtype, generator=gen)
    g_s = torch.randn(s.shape, device=s.device, dtype=s.dtype, generator=gen)
    return g_v, g_I, g_s


def _plot_base_inputs(device: torch.device, dtype: torch.dtype):
    v = torch.full((1,), -65.0, device=device, dtype=dtype)
    Iasc = torch.zeros((1, 1), device=device, dtype=dtype)

    v_th = torch.full((1,), -50.0, device=device, dtype=dtype)
    v_reset = torch.full((1,), -70.0, device=device, dtype=dtype)
    v_rest = v_reset.clone()

    c_m = torch.full((1,), 1.0, device=device, dtype=dtype)
    tau = torch.full((1,), 20.0, device=device, dtype=dtype)
    k = torch.full((1, 1), 0.1, device=device, dtype=dtype)
    asc_amps = torch.full((1, 1), -0.05, device=device, dtype=dtype)

    return {
        "v": v,
        "Iasc": Iasc,
        "v_th": v_th,
        "v_reset": v_reset,
        "v_rest": v_rest,
        "c_m": c_m,
        "tau": tau,
        "k": k,
        "asc_amps": asc_amps,
    }


def _simulate_reference_trace(
    base,
    stimulus: torch.Tensor,
    dt: float,
    hard_reset: bool,
    surrogate: SurrogateFunctionBase,
):
    B, M = base["Iasc"].shape
    glif = GLIF3(
        n_neuron=B,
        v_threshold=base["v_th"],
        v_reset=base["v_reset"],
        v_rest=base["v_rest"],
        c_m=base["c_m"],
        tau=base["tau"],
        k=base["k"],
        asc_amps=base["asc_amps"],
        tau_ref=0.0,
        hard_reset=hard_reset,
        surrogate_function=surrogate,
        device=base["v"].device,
        dtype=base["v"].dtype,
    )

    glif.v = base["v"].clone()
    glif.Iasc = base["Iasc"].clone()
    glif.refractory = torch.zeros_like(base["v"])

    traces = {"v": [], "Iasc": [], "spike": []}
    with torch.no_grad():
        with environ.context(dt=dt):
            for current in stimulus:
                spike = glif.single_step_forward(current)
                traces["v"].append(glif.v.detach().cpu())
                traces["Iasc"].append(glif.Iasc.detach().cpu())
                traces["spike"].append(spike.detach().cpu())

    return {k: torch.stack(v) for k, v in traces.items()}


def _simulate_kernel_trace(
    step_fn,
    base,
    stimulus: torch.Tensor,
    dt: float,
    hard_reset: bool,
    alpha: float,
):
    B, M = base["Iasc"].shape
    v = base["v"].clone()
    Iasc_flat = base["Iasc"].reshape(-1).clone()
    not_refrac = torch.ones_like(v)

    params = {
        "v_th": base["v_th"].clone(),
        "v_reset": base["v_reset"].clone(),
        "v_rest": base["v_rest"].clone(),
        "c_m": base["c_m"].clone(),
        "tau": base["tau"].clone(),
        "k": base["k"].reshape(-1).clone(),
        "asc_amps": base["asc_amps"].reshape(-1).clone(),
    }

    traces = {"v": [], "Iasc": [], "spike": []}
    with torch.no_grad():
        for current in stimulus:
            v, Iasc_flat, spike = step_fn(
                v=v,
                Iasc=Iasc_flat,
                x=current,
                params=params,
                not_refrac=not_refrac,
                dt=dt,
                M=M,
                hard_reset=hard_reset,
                alpha=alpha,
            )
            traces["v"].append(v.detach().cpu())
            traces["Iasc"].append(Iasc_flat.view(B, M).detach().cpu())
            traces["spike"].append(spike.detach().cpu())

    return {k: torch.stack(v) for k, v in traces.items()}


def _plot_kernel_traces(time: torch.Tensor, traces: dict[str, dict[str, torch.Tensor]]):
    fig, axes = plt.subplots(3, 1, sharex=True)

    labels = list(traces.keys())
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    color_map = {label: colors[idx % len(colors)] for idx, label in enumerate(labels)}

    for idx, label in enumerate(labels):
        color = color_map[label]
        spike = traces[label]["spike"].squeeze(-1)
        spike_vis = spike >= 0.5
        spk_nz = spike_vis.nonzero(as_tuple=False).flatten()
        if spk_nz.numel() > 0:
            axes[0].scatter(
                time[spk_nz],
                torch.full((spk_nz.numel(),), float(idx)),
                marker="|",
                linewidths=0.8,
                color=color,
                label=label,
            )

        axes[1].plot(
            time,
            traces[label]["v"].squeeze(-1),
            label=label,
            linewidth=0.9,
            color=color,
        )
        axes[2].plot(
            time,
            traces[label]["Iasc"].squeeze(-1),
            label=label,
            linewidth=0.9,
            color=color,
        )

    axes[0].set_ylabel("Spikes")
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[1].set_ylabel("Membrane Potential")
    axes[2].set_ylabel("After-Spike Current")
    axes[2].set_xlabel("Time (ms)")

    fig.suptitle("GLIF3 Kernel Comparison")
    # fig.legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, name="glif3_kernel_comparison")
    plt.close(fig)


@pytest.mark.parametrize("hard_reset", [False, True])
@pytest.mark.parametrize("backend", ["warp", "triton", "cupy"])
def test_glif3_step_matches_reference(backend: str, hard_reset: bool):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for warp/triton GLIF3 kernels.")

    if backend == "warp":
        pytest.importorskip("warp")
        from benchmark.dense_glif_net.glif_warp import glif3_step_warp as step_fn
    elif backend == "triton":
        pytest.importorskip("triton")
        from benchmark.dense_glif_net.glif_triton import glif3_step_triton as step_fn
    else:
        pytest.importorskip("cupy")
        from benchmark.dense_glif_net.glif_cupy import glif3_step_cupy as step_fn

    device = torch.device("cuda")
    dtype = torch.float32
    B, M = 64, 3
    dt = 0.5
    alpha = 2.0

    base = _base_inputs(B, M, device, dtype)

    # Kernel inputs as leaf tensors (flattened Iasc/asc_amps for kernels).
    v = base["v"].clone().requires_grad_(True)
    x = base["x"].clone().requires_grad_(True)
    Iasc_flat = base["Iasc"].reshape(-1).clone().requires_grad_(True)
    asc_flat = base["asc_amps"].reshape(-1).clone().requires_grad_(True)
    not_refrac = torch.ones_like(v)

    params = {
        "v_th": base["v_th"].clone(),
        "v_reset": base["v_reset"].clone(),
        "v_rest": base["v_rest"].clone(),
        "c_m": base["c_m"].clone(),
        "tau": base["tau"].clone(),
        "k": base["k"].reshape(-1).clone(),
        "asc_amps": asc_flat,
    }

    v_out, I_out, s_out = step_fn(
        v=v,
        Iasc=Iasc_flat,
        x=x,
        params=params,
        not_refrac=not_refrac,
        dt=dt,
        M=M,
        hard_reset=hard_reset,
        alpha=alpha,
    )

    # Reference GLIF3 step with the same surrogate as the kernel.
    if backend in ("warp", "cupy"):
        surrogate = ATan(alpha=alpha, spiking=True)
    else:
        surrogate = ATanApprox(alpha=alpha, spiking=True)
    ref_base = {
        "v": base["v"].clone().requires_grad_(True),
        "x": base["x"].clone().requires_grad_(True),
        "Iasc": base["Iasc"].clone().requires_grad_(True),
        "v_th": base["v_th"].clone(),
        "v_reset": base["v_reset"].clone(),
        "v_rest": base["v_rest"].clone(),
        "c_m": base["c_m"].clone(),
        "tau": base["tau"].clone(),
        "k": base["k"].clone(),
        "asc_amps": base["asc_amps"].clone(),
    }
    v_ref, I_ref, s_ref, asc_ref = _reference_step(
        ref_base, dt=dt, hard_reset=hard_reset, surrogate=surrogate
    )

    torch.testing.assert_close(v_out, v_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(I_out.view(B, M), I_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(s_out, s_ref, rtol=1e-4, atol=1e-4)

    g_v, g_I_flat, g_s = _make_upstream(v_out, I_out, s_out)
    loss = (v_out * g_v).sum() + (I_out * g_I_flat).sum() + (s_out * g_s).sum()
    loss.backward()

    g_v_ref, g_I_ref, g_s_ref = _make_upstream(v_ref, I_ref, s_ref)
    loss_ref = (
        (v_ref * g_v_ref).sum() + (I_ref * g_I_ref).sum() + (s_ref * g_s_ref).sum()
    )
    loss_ref.backward()

    torch.testing.assert_close(v.grad, ref_base["v"].grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        Iasc_flat.grad.view(B, M), ref_base["Iasc"].grad, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(x.grad, ref_base["x"].grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        asc_flat.grad.view(B, M), asc_ref.grad, rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("backend", ["triton", "warp", "cupy"])
def test_glif_dense_multistep_fused_matches_reference(backend: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused kernels.")

    if backend == "triton":
        pytest.importorskip("triton")
    elif backend == "warp":
        pytest.importorskip("warp")
    else:
        pytest.importorskip("cupy")

    device = torch.device("cuda")
    dtype = torch.float32
    T, N = 16, 16

    weight, bias, x_seq, params = make_inputs(T, N, device, require_grad=False)
    neuron_ref = build_neuron("torch_eager", N, params, require_grad=False)
    model_ref = GLIFDenseNet(N, neuron_ref, unroll=16)
    init_net_state(model_ref, device=device, dtype=dtype)
    model_ref.linear.weight.data.copy_(weight)
    model_ref.linear.bias.data.copy_(bias)

    neuron_kernel = build_neuron(backend, N, params, require_grad=False)
    model_kernel = GLIFDenseNet(N, neuron_kernel, unroll=16)
    init_net_state(model_kernel, device=device, dtype=dtype)
    model_kernel.linear.weight.data.copy_(weight)
    model_kernel.linear.bias.data.copy_(bias)

    with torch.no_grad():
        with environ.context(dt=1.0):
            spike_ref, _ = model_ref(x_seq)
            spike_kernel, _ = model_kernel(x_seq)

    torch.testing.assert_close(spike_kernel, spike_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("backend", ["triton", "warp", "cupy"])
def test_glif_multistep_fused_matches_reference(backend: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused kernels.")

    if backend == "triton":
        pytest.importorskip("triton")
        from benchmark.dense_glif_net.glif_triton import glif3_step_triton as step_fn
    elif backend == "warp":
        pytest.importorskip("warp")
        from benchmark.dense_glif_net.glif_warp import glif3_step_warp as step_fn
    else:
        pytest.importorskip("cupy")
        from benchmark.dense_glif_net.glif_cupy import glif3_step_cupy as step_fn

    device = torch.device("cuda")
    dtype = torch.float32
    T, B, M = 16, 32, 3
    dt = 0.5
    hard_reset = False

    base = _base_inputs(B, M, device, dtype)
    x_seq = torch.randn((T, B), device=device, dtype=dtype)

    v = base["v"].clone()
    Iasc = base["Iasc"].clone()
    not_refrac = torch.ones_like(v)

    with torch.no_grad():
        s_seq, v_seq, v_out, I_out = step_fn.multistep_fused(
            x_seq=x_seq,
            v=v,
            Iasc=Iasc,
            params={
                "v_th": base["v_th"].clone(),
                "v_reset": base["v_reset"].clone(),
                "v_rest": base["v_rest"].clone(),
                "c_m": base["c_m"].clone(),
                "tau": base["tau"].clone(),
                "k": base["k"].reshape(-1).clone(),
                "asc_amps": base["asc_amps"].reshape(-1).clone(),
            },
            not_refrac=not_refrac,
            dt=dt,
            M=M,
            hard_reset=hard_reset,
            alpha=2.0,
        )

    surrogate = (
        ATan(alpha=2.0, spiking=True)
        if backend in ("warp", "cupy")
        else ATanApprox(alpha=2.0, spiking=True)
    )
    v_ref_seq, I_ref, s_ref_seq, _ = _reference_multistep(
        {
            "v": base["v"].clone(),
            "Iasc": base["Iasc"].clone(),
            "v_th": base["v_th"].clone(),
            "v_reset": base["v_reset"].clone(),
            "v_rest": base["v_rest"].clone(),
            "c_m": base["c_m"].clone(),
            "tau": base["tau"].clone(),
            "k": base["k"].clone(),
            "asc_amps": base["asc_amps"].clone(),
        },
        x_seq=x_seq,
        dt=dt,
        hard_reset=hard_reset,
        surrogate=surrogate,
    )

    torch.testing.assert_close(s_seq, s_ref_seq, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_seq, v_ref_seq, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_out, v_ref_seq[-1], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(I_out, I_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("backend", ["triton", "warp", "cupy"])
def test_glif_multistep_fused_grad_matches_reference(backend: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused kernels.")

    if backend == "triton":
        pytest.importorskip("triton")
        from benchmark.dense_glif_net.glif_triton import glif3_step_triton as step_fn
    elif backend == "warp":
        pytest.importorskip("warp")
        from benchmark.dense_glif_net.glif_warp import glif3_step_warp as step_fn
    else:
        pytest.importorskip("cupy")
        from benchmark.dense_glif_net.glif_cupy import glif3_step_cupy as step_fn

    device = torch.device("cuda")
    dtype = torch.float32
    T, B, M = 8, 6, 2
    dt = 0.5
    hard_reset = False

    base = _base_inputs(B, M, device, dtype)
    x_seq = torch.randn((T, B), device=device, dtype=dtype, requires_grad=True)

    v = base["v"].clone().requires_grad_(True)
    Iasc = base["Iasc"].clone().requires_grad_(True)
    asc_flat = base["asc_amps"].reshape(-1).clone().requires_grad_(True)
    not_refrac = torch.ones_like(v)

    s_seq, v_seq, v_out, I_out = step_fn.multistep_fused(
        x_seq=x_seq,
        v=v,
        Iasc=Iasc,
        params={
            "v_th": base["v_th"].clone(),
            "v_reset": base["v_reset"].clone(),
            "v_rest": base["v_rest"].clone(),
            "c_m": base["c_m"].clone(),
            "tau": base["tau"].clone(),
            "k": base["k"].reshape(-1).clone(),
            "asc_amps": asc_flat,
        },
        not_refrac=not_refrac,
        dt=dt,
        M=M,
        hard_reset=hard_reset,
        alpha=2.0,
    )

    loss = s_seq.sum() + v_seq.sum() + v_out.sum() + I_out.sum()
    loss.backward()

    x_seq_ref = x_seq.detach().clone().requires_grad_(True)
    v_ref = base["v"].clone().requires_grad_(True)
    Iasc_ref = base["Iasc"].clone().requires_grad_(True)
    asc_ref = base["asc_amps"].clone().requires_grad_(True)
    surrogate = (
        ATan(alpha=2.0, spiking=True)
        if backend in ("warp", "cupy")
        else ATanApprox(alpha=2.0, spiking=True)
    )
    v_ref_seq, I_ref, s_ref_seq, asc_ref_param = _reference_multistep(
        {
            "v": v_ref,
            "Iasc": Iasc_ref,
            "v_th": base["v_th"].clone(),
            "v_reset": base["v_reset"].clone(),
            "v_rest": base["v_rest"].clone(),
            "c_m": base["c_m"].clone(),
            "tau": base["tau"].clone(),
            "k": base["k"].clone(),
            "asc_amps": asc_ref,
        },
        x_seq=x_seq_ref,
        dt=dt,
        hard_reset=hard_reset,
        surrogate=surrogate,
    )
    loss_ref = s_ref_seq.sum() + v_ref_seq.sum() + v_ref_seq[-1].sum() + I_ref.sum()
    loss_ref.backward()

    torch.testing.assert_close(x_seq.grad, x_seq_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v.grad, v_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(Iasc.grad, Iasc_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        asc_flat.grad.view(B, M), asc_ref_param.grad, rtol=1e-4, atol=1e-4
    )


def test_glif3_step_cupy_stress_noncontig_and_casts():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CuPy GLIF3 kernels.")
    pytest.importorskip("cupy")

    from benchmark.dense_glif_net.glif_cupy import glif3_step_cupy

    device = torch.device("cuda")
    B, M = 256, 4
    dt = 0.25

    base = _base_inputs(B, M, device, torch.float32)

    # Create non-contiguous inputs and higher precision to exercise casts.
    v = base["v"].view(16, 16).t().reshape(-1).double()
    x = base["x"].view(16, 16).t().reshape(-1).double()
    Iasc_flat = base["Iasc"].transpose(0, 1).reshape(-1).double()
    not_refrac = torch.ones_like(v)

    params = {
        "v_th": base["v_th"].double(),
        "v_reset": base["v_reset"].double(),
        "v_rest": base["v_rest"].double(),
        "c_m": base["c_m"].double(),
        "tau": base["tau"].double(),
        "k": base["k"].reshape(-1).double(),
        "asc_amps": base["asc_amps"].reshape(-1).double(),
    }

    v_out, I_out, s_out = glif3_step_cupy(
        v=v,
        Iasc=Iasc_flat,
        x=x,
        params=params,
        not_refrac=not_refrac,
        dt=dt,
        M=M,
    )

    assert v_out.dtype == torch.float32
    assert I_out.dtype == torch.float32
    assert s_out.dtype == torch.float32
    assert torch.isfinite(v_out).all()


def test_draw_glif3_kernel_comparison():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for kernel comparison plots.")

    device = torch.device("cuda")
    dtype = torch.float32
    dt = 1.0
    T = 200
    alpha = 2.0
    hard_reset = False

    stimulus = torch.zeros(T, 1, device=device, dtype=dtype)
    stimulus[50:150] = 10.0
    time = torch.arange(0, T, dtype=torch.float32) * dt

    base = _plot_base_inputs(device, dtype)
    traces = {
        "reference (atan)": _simulate_reference_trace(
            base,
            stimulus,
            dt=dt,
            hard_reset=hard_reset,
            surrogate=ATan(alpha=alpha, spiking=True),
        )
    }

    backends: dict[str, Callable] = {}
    try:
        from benchmark.dense_glif_net.glif_warp import glif3_step_warp

        backends["warp"] = glif3_step_warp
    except Exception:
        pass

    try:
        from benchmark.dense_glif_net.glif_triton import glif3_step_triton

        backends["triton"] = glif3_step_triton
        traces["reference (approx)"] = _simulate_reference_trace(
            base,
            stimulus,
            dt=dt,
            hard_reset=hard_reset,
            surrogate=ATanApprox(alpha=alpha, spiking=True),
        )
    except Exception:
        pass

    try:
        from benchmark.dense_glif_net.glif_cupy import glif3_step_cupy

        backends["cupy"] = glif3_step_cupy
    except Exception:
        pass

    if not backends:
        pytest.skip("No CUDA kernel backends available for plotting.")

    for name, step_fn in backends.items():
        traces[name] = _simulate_kernel_trace(
            step_fn,
            base,
            stimulus,
            dt=dt,
            hard_reset=hard_reset,
            alpha=alpha,
        )

    _plot_kernel_traces(time, traces)
