"""Correctness tests for the GLIF3 CUDA kernels (triton / warp / cupy).

Every backend is checked against the eager ``btorch.models.neurons.glif.GLIF3``
reference for three entry points:

- single step               ``glif3_step_<backend>``
- neuron multistep          ``.multistep_fused``
- dense multistep           ``.dense_multistep_fused`` (inference + training)
- sparse multistep          ``.sparse_multistep_fused`` (scale-free SpMV; the
                            fused kernel and its gradients are checked against
                            the dense-equivalent weight)

Spikes, voltages, after-spike currents and gradients must all match the eager
reference. Inputs come in ``constant`` and ``random`` flavours, and every case
asserts the outputs are *non-trivial* (spikes actually happen but not on every
step/neuron, gradients are finite and non-zero) so a kernel that silently
returned zeros could never pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from benchmarks.glif_net.glif_common import GLIFDenseNet, build_neuron
from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.neurons.glif import GLIF3
from btorch.models.surrogate import ATan


DEVICE = torch.device("cuda")
DTYPE = torch.float32
DT = 0.5
ALPHA = 2.0
RTOL = 1e-4
ATOL = 1e-4

BACKENDS = ("triton", "warp", "cupy", "tilelang")
INPUT_KINDS = ("constant", "random")


# ---------------------------------------------------------------------------
# Backend loading
# ---------------------------------------------------------------------------
def _load_step_fn(name: str):
    pytest.importorskip(name)
    if name == "triton":
        from benchmarks.glif_net.glif_triton import glif3_step_triton as fn
    elif name == "warp":
        from benchmarks.glif_net.glif_warp import glif3_step_warp as fn
    elif name == "cupy":
        from benchmarks.glif_net.glif_cupy import glif3_step_cupy as fn
    else:
        from benchmarks.glif_net.glif_tilelang import glif3_step_tilelang as fn
    return fn


@pytest.fixture(params=BACKENDS)
def step_fn(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GLIF3 kernels.")
    return _load_step_fn(request.param)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass
class Glif3Case:
    """A GLIF3 state + parameter set, chosen to spike non-trivially."""

    v: torch.Tensor  # (B,)
    Iasc: torch.Tensor  # (B, M)
    x: torch.Tensor  # (B,)
    v_th: torch.Tensor  # (B,)
    v_reset: torch.Tensor  # (B,)
    v_rest: torch.Tensor  # (B,)
    c_m: torch.Tensor  # (B,)
    tau: torch.Tensor  # (B,)
    k: torch.Tensor  # (B, M)
    asc_amps: torch.Tensor  # (B, M)

    @property
    def B(self) -> int:
        return self.v.numel()

    @property
    def M(self) -> int:
        return self.Iasc.shape[1]


def make_case(B: int, M: int, kind: str, *, seed: int = 0) -> Glif3Case:
    """Build a GLIF3 case chosen to fire non-trivially.

    The membrane is stiff (``a = exp(-dt/tau) ~ 0.975``), so two things matter:
    the initial ``v`` straddles threshold, so a *single* step already produces a
    mix of firing / non-firing neurons; and the input drive is strong enough to
    sustain repeated firing over a short sequence (with reset-induced gaps).

    ``kind`` controls the *input* (``x``): ``constant`` is a DC drive shared by
    every step, ``random`` varies per neuron/step. Both use a heterogeneous
    initial ``v`` so the single-step case is meaningful.
    """
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=DEVICE, dtype=DTYPE)

    def rand(*shape):
        return torch.rand(*shape, generator=gen, device=DEVICE, dtype=DTYPE)

    def full(value, *shape):
        return torch.full(shape, value, device=DEVICE, dtype=DTYPE)

    if kind == "random":
        v = -55.0 + 12.0 * randn(B)  # straddles threshold (-50)
        x = 0.3 + 0.1 * randn(B)
        Iasc = 0.03 * randn(B, M)
        k = 0.1 + 0.2 * rand(B, M)
        asc_amps = 0.05 * randn(B, M)
    elif kind == "constant":
        v = torch.linspace(-72.0, -44.0, B, device=DEVICE, dtype=DTYPE)
        x = full(0.35, B)
        Iasc = full(0.0, B, M)
        k = full(0.15, B, M)
        asc_amps = full(0.03, B, M)
    else:
        raise ValueError(f"unknown input kind: {kind!r}")

    return Glif3Case(
        v=v,
        Iasc=Iasc,
        x=x,
        v_th=full(-50.0, B),
        v_reset=full(-70.0, B),
        v_rest=full(-70.0, B),
        c_m=full(0.05, B),
        tau=full(20.0, B),
        k=k,
        asc_amps=asc_amps,
    )


def make_input_sequence(T: int, B: int, kind: str, *, seed: int) -> torch.Tensor:
    if kind == "constant":
        return torch.full((T, B), 0.35, device=DEVICE, dtype=DTYPE)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return 0.3 + 0.1 * torch.randn(T, B, generator=gen, device=DEVICE, dtype=DTYPE)


# ---------------------------------------------------------------------------
# Non-triviality guards
# ---------------------------------------------------------------------------
def assert_nontrivial_spikes(spike: torch.Tensor) -> None:
    frac = spike.float().mean().item()
    assert 0.0 < frac < 1.0, f"spikes are trivial (fraction firing = {frac})"
    assert spike.float().std().item() > 0.0, "spikes have no variation"


def assert_nontrivial_grad(grad: torch.Tensor, name: str) -> None:
    assert grad is not None, f"missing gradient for {name}"
    assert torch.isfinite(grad).all(), f"non-finite gradient for {name}"
    assert grad.abs().sum().item() > 0.0, f"all-zero gradient for {name}"


# ---------------------------------------------------------------------------
# Eager reference (btorch GLIF3)
# ---------------------------------------------------------------------------
def _build_glif(case: Glif3Case, hard_reset: bool) -> GLIF3:
    return GLIF3(
        n_neuron=case.B,
        v_threshold=case.v_th,
        v_reset=case.v_reset,
        v_rest=case.v_rest,
        c_m=case.c_m,
        tau=case.tau,
        k=case.k,
        asc_amps=case.asc_amps,
        tau_ref=0.0,
        hard_reset=hard_reset,
        surrogate_function=ATan(alpha=ALPHA, spiking=True),
        trainable_param={"asc_amps"},
        device=DEVICE,
        dtype=DTYPE,
    )


def reference_step(glif: GLIF3, v, Iasc, x):
    glif.v = v
    glif.Iasc = Iasc
    glif.refractory = torch.zeros_like(v)
    with environ.context(dt=DT):
        spike = glif.single_step_forward(x)
    return glif.v, glif.Iasc, spike, glif.asc_amps


def reference_neuron_multistep(glif: GLIF3, v, Iasc, x_seq):
    glif.v = v
    glif.Iasc = Iasc
    glif.refractory = torch.zeros_like(v)
    spikes, voltages = [], []
    with environ.context(dt=DT):
        for x_t in x_seq:
            spikes.append(glif.single_step_forward(x_t))
            voltages.append(glif.v)
    return torch.stack(voltages), glif.Iasc, torch.stack(spikes), glif.asc_amps


def reference_dense_multistep(glif: GLIF3, v, Iasc, weight, bias, x_seq):
    glif.v = v.clone()
    glif.Iasc = Iasc.clone()
    glif.refractory = torch.zeros_like(v)
    B = v.numel()
    spike = torch.zeros(B, device=DEVICE, dtype=DTYPE)
    spikes, voltages = [], []
    with torch.no_grad(), environ.context(dt=DT):
        for x_t in x_seq:
            x_in = x_t + bias + weight @ spike
            spike = glif.single_step_forward(x_in)
            spikes.append(spike)
            voltages.append(glif.v.clone())
    return torch.stack(spikes), torch.stack(voltages), glif.v, glif.Iasc


def reference_dense_multistep_grad(glif: GLIF3, v, Iasc, weight, bias, x_seq):
    """Grad-enabled eager dense multistep: autograd flows through the eager
    neuron and the torch matmul, mirroring the kernel training path."""
    glif.v = v
    glif.Iasc = Iasc
    glif.refractory = torch.zeros_like(v)
    s_prev = torch.zeros(v.numel(), device=DEVICE, dtype=DTYPE)
    spikes, voltages = [], []
    with environ.context(dt=DT):
        for x_t in x_seq:
            s_prev = glif.single_step_forward(x_t + bias + torch.mv(weight, s_prev))
            spikes.append(s_prev)
            voltages.append(glif.v)
    return (
        torch.stack(spikes), torch.stack(voltages), glif.v, glif.Iasc, glif.asc_amps
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def leaf(t: torch.Tensor, requires_grad: bool) -> torch.Tensor:
    return t.detach().clone().requires_grad_(requires_grad)


def kernel_params(case: Glif3Case, asc_flat: torch.Tensor) -> dict:
    return {
        "v_th": case.v_th,
        "v_reset": case.v_reset,
        "v_rest": case.v_rest,
        "c_m": case.c_m,
        "tau": case.tau,
        "k": case.k.reshape(-1),
        "asc_amps": asc_flat,
    }


def assert_close(actual, expected, name: str) -> None:
    torch.testing.assert_close(
        actual, expected, rtol=RTOL, atol=ATOL, msg=lambda m: f"[{name}] {m}"
    )


# ---------------------------------------------------------------------------
# Single step: forward + backward
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", INPUT_KINDS)
@pytest.mark.parametrize("hard_reset", [False, True])
@pytest.mark.parametrize("M", [1, 3])
def test_step_matches_reference(step_fn, M, hard_reset, kind):
    B = 64
    case = make_case(B, M, kind, seed=1)

    # Kernel leaves (flattened Iasc / asc_amps).
    v = leaf(case.v, True)
    x = leaf(case.x, True)
    Iasc = leaf(case.Iasc.reshape(-1), True)
    asc = leaf(case.asc_amps.reshape(-1), True)
    params = kernel_params(case, asc)
    not_refrac = torch.ones(B, device=DEVICE, dtype=DTYPE)

    v_out, I_out, s_out = step_fn(
        v=v, Iasc=Iasc, x=x, params=params, not_refrac=not_refrac,
        dt=DT, M=M, hard_reset=hard_reset, alpha=ALPHA,
    )

    # Reference leaves.
    v_r = leaf(case.v, True)
    x_r = leaf(case.x, True)
    Iasc_r = leaf(case.Iasc, True)
    glif = _build_glif(case, hard_reset)
    v_ref, I_ref, s_ref, asc_ref = reference_step(glif, v_r, Iasc_r, x_r)

    assert_nontrivial_spikes(s_out)
    assert_close(v_out, v_ref, "v")
    assert_close(I_out.view(B, M), I_ref, "Iasc")
    assert_close(s_out, s_ref, "spike")

    # Deterministic upstream so both sides see identical cotangents.
    gen = torch.Generator(device=DEVICE).manual_seed(7)
    g_v = torch.randn(v_out.shape, generator=gen, device=DEVICE, dtype=DTYPE)
    g_I = torch.randn(I_out.shape, generator=gen, device=DEVICE, dtype=DTYPE)
    g_s = torch.randn(s_out.shape, generator=gen, device=DEVICE, dtype=DTYPE)

    ((v_out * g_v).sum() + (I_out * g_I).sum() + (s_out * g_s).sum()).backward()
    (
        (v_ref * g_v).sum() + (I_ref * g_I.view(B, M)).sum() + (s_ref * g_s).sum()
    ).backward()

    for name in ("v", "x", "Iasc", "asc"):
        assert_nontrivial_grad(locals()[name].grad, name)
    assert_close(v.grad, v_r.grad, "grad_v")
    assert_close(x.grad, x_r.grad, "grad_x")
    assert_close(Iasc.grad.view(B, M), Iasc_r.grad, "grad_Iasc")
    assert_close(asc.grad.view(B, M), asc_ref.grad, "grad_asc")


# ---------------------------------------------------------------------------
# Neuron multistep: forward + backward
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", INPUT_KINDS)
@pytest.mark.parametrize("hard_reset", [False, True])
@pytest.mark.parametrize("M", [1, 3])
def test_neuron_multistep_matches_reference(step_fn, M, hard_reset, kind):
    T, B = 16, 32
    case = make_case(B, M, kind, seed=2)
    x_seq_vals = make_input_sequence(T, B, kind, seed=3)

    x_seq = leaf(x_seq_vals, True)
    v = leaf(case.v, True)
    Iasc = leaf(case.Iasc, True)
    asc = leaf(case.asc_amps.reshape(-1), True)
    params = kernel_params(case, asc)
    not_refrac = torch.ones(B, device=DEVICE, dtype=DTYPE)

    s_seq, v_seq, v_out, I_out = step_fn.multistep_fused(
        x_seq=x_seq, v=v, Iasc=Iasc, params=params, not_refrac=not_refrac,
        dt=DT, M=M, hard_reset=hard_reset, alpha=ALPHA,
    )

    x_seq_r = leaf(x_seq_vals, True)
    v_r = leaf(case.v, True)
    Iasc_r = leaf(case.Iasc, True)
    glif = _build_glif(case, hard_reset)
    v_ref_seq, I_ref, s_ref_seq, asc_ref = reference_neuron_multistep(
        glif, v_r, Iasc_r, x_seq_r
    )

    assert_nontrivial_spikes(s_seq)
    assert_close(s_seq, s_ref_seq, "spike_seq")
    assert_close(v_seq, v_ref_seq, "v_seq")
    assert_close(v_out, v_ref_seq[-1], "v_out")
    assert_close(I_out, I_ref, "Iasc")

    (s_seq.sum() + v_seq.sum() + v_out.sum() + I_out.sum()).backward()
    (s_ref_seq.sum() + v_ref_seq.sum() + v_ref_seq[-1].sum() + I_ref.sum()).backward()

    for name in ("x_seq", "v", "Iasc", "asc"):
        assert_nontrivial_grad(locals()[name].grad, name)
    assert_close(x_seq.grad, x_seq_r.grad, "grad_x_seq")
    assert_close(v.grad, v_r.grad, "grad_v")
    assert_close(Iasc.grad, Iasc_r.grad, "grad_Iasc")
    assert_close(asc.grad.view(B, M), asc_ref.grad, "grad_asc")


# ---------------------------------------------------------------------------
# Dense multistep (neuron + recurrent connection): inference and training
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", INPUT_KINDS)
@pytest.mark.parametrize("fused_matmul", [True, False])
@pytest.mark.parametrize("M", [1, 3])
def test_dense_multistep_matches_reference(step_fn, M, fused_matmul, kind):
    # B = 34 is not a multiple of the warp TILE_M (16), so the fused path here
    # exercises the single-thread fallback; test_dense_fused_matches_matmul uses
    # a tile-aligned B to cover the tile-matmul path.
    T, B = 16, 34
    hard_reset = False
    case = make_case(B, M, kind, seed=4)
    x_seq = make_input_sequence(T, B, kind, seed=5)

    gen = torch.Generator(device=DEVICE).manual_seed(6)
    weight = torch.randn(B, B, generator=gen, device=DEVICE, dtype=DTYPE) / B**0.5
    bias = 0.02 * torch.randn(B, generator=gen, device=DEVICE, dtype=DTYPE)

    params = kernel_params(case, case.asc_amps.reshape(-1))
    not_refrac = torch.ones(B, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        s_seq, v_seq, v_out, I_out = step_fn.dense_multistep_fused(
            x_seq=x_seq,
            weight=weight,
            bias=bias,
            v=case.v.clone(),
            Iasc=case.Iasc.reshape(-1).clone(),
            params=params,
            not_refrac=not_refrac,
            dt=DT,
            M=M,
            hard_reset=hard_reset,
            alpha=ALPHA,
            fused_matmul=fused_matmul,
        )

    glif = _build_glif(case, hard_reset)
    s_ref, v_ref, v_out_ref, I_ref = reference_dense_multistep(
        glif, case.v, case.Iasc, weight, bias, x_seq
    )

    assert_nontrivial_spikes(s_seq)
    assert_close(s_seq, s_ref, "spike_seq")
    assert_close(v_seq, v_ref, "v_seq")
    assert_close(v_out, v_out_ref, "v_out")
    assert_close(I_out, I_ref, "Iasc")


@pytest.mark.parametrize("kind", INPUT_KINDS)
def test_dense_fused_matches_matmul(step_fn, kind):
    """The two dense paths (in-kernel matmul vs cuBLAS) must agree exactly."""
    T, B, M = 16, 48, 3
    case = make_case(B, M, kind, seed=8)
    x_seq = make_input_sequence(T, B, kind, seed=9)

    gen = torch.Generator(device=DEVICE).manual_seed(10)
    weight = torch.randn(B, B, generator=gen, device=DEVICE, dtype=DTYPE) / B**0.5
    bias = 0.02 * torch.randn(B, generator=gen, device=DEVICE, dtype=DTYPE)
    not_refrac = torch.ones(B, device=DEVICE, dtype=DTYPE)

    def run(fused_matmul: bool):
        with torch.no_grad():
            return step_fn.dense_multistep_fused(
                x_seq=x_seq,
                weight=weight,
                bias=bias,
                v=case.v.clone(),
                Iasc=case.Iasc.reshape(-1).clone(),
                params=kernel_params(case, case.asc_amps.reshape(-1)),
                not_refrac=not_refrac,
                dt=DT,
                M=M,
                hard_reset=False,
                alpha=ALPHA,
                fused_matmul=fused_matmul,
            )

    fused = run(True)
    matmul = run(False)
    assert_nontrivial_spikes(fused[0])
    for name, a, b in zip(("spike_seq", "v_seq", "v_out", "Iasc"), fused, matmul):
        assert_close(a, b, f"fused_vs_matmul_{name}")


# ---------------------------------------------------------------------------
# Sparse (scale-free recurrent) multistep: inference and training
# ---------------------------------------------------------------------------
def _sparse_case(B, M, kind, seed):
    """Sparse CSR weight + its dense equivalent (same values at the nonzeros)."""
    from benchmarks.glif_net.glif_common import scale_free_csr
    W = scale_free_csr(B, 0.05, DEVICE, seed=seed)
    rows, cols = W.coo_indices[0], W.coo_indices[1]
    dense = torch.zeros(B, B, device=DEVICE, dtype=DTYPE)
    dense[rows, cols] = W.val
    case = make_case(B, M, kind, seed=seed)
    x_seq = make_input_sequence(16, B, kind, seed=seed + 1)
    bias = 0.02 * torch.randn(B, generator=torch.Generator(device=DEVICE).manual_seed(seed + 2),
                              device=DEVICE, dtype=DTYPE)
    return W, dense, case, x_seq, bias, torch.ones(B, device=DEVICE, dtype=DTYPE)


@pytest.mark.parametrize("kind", INPUT_KINDS)
def test_sparse_multistep_matches_dense(step_fn, kind):
    """The fused sparse SpMV multistep must match the dense-equivalent weight and
    spike non-trivially."""
    B, M = 512, 3
    W, dense, case, x_seq, bias, not_refrac = _sparse_case(B, M, kind, seed=30)
    kw = dict(bias=bias, not_refrac=not_refrac, dt=DT, M=M, hard_reset=False, alpha=ALPHA)
    with torch.no_grad():
        sparse = step_fn.sparse_multistep_fused(
            x_seq=x_seq, weight=W, v=case.v.clone(),
            Iasc=case.Iasc.reshape(-1).clone(),
            params=kernel_params(case, case.asc_amps.reshape(-1)), **kw)
        ref = step_fn.dense_multistep_fused(
            x_seq=x_seq, weight=dense, v=case.v.clone(),
            Iasc=case.Iasc.reshape(-1).clone(),
            params=kernel_params(case, case.asc_amps.reshape(-1)),
            fused_matmul=True, **kw)
    assert_nontrivial_spikes(sparse[0])
    for name, a, b in zip(("spike_seq", "v_seq", "v_out", "Iasc"), sparse, ref):
        assert_close(a, b, f"sparse_vs_dense_{name}")


@pytest.mark.parametrize("kind", INPUT_KINDS)
def test_sparse_multistep_grad_matches_dense(step_fn, kind):
    """Sparse training gradients must match the dense-equivalent's — grad w.r.t.
    the sparse values equals the dense weight gradient at the nonzeros."""
    from benchmarks.glif_net.glif_common import SparseWeight
    B, M = 256, 3
    W0, dense0, case, x_seq0, bias0, not_refrac = _sparse_case(B, M, kind, seed=40)
    rows, cols = W0.coo_indices[0], W0.coo_indices[1]

    def run_sparse():
        val = leaf(W0.val, True)
        W = SparseWeight(W0.crow, W0.col, val, W0.coo_indices, W0.N)
        x, bias = leaf(x_seq0, True), leaf(bias0, True)
        asc = leaf(case.asc_amps.reshape(-1), True)
        out = step_fn.sparse_multistep_fused(
            x_seq=x, weight=W, bias=bias, v=case.v.clone(),
            Iasc=case.Iasc.reshape(-1).clone(), params=kernel_params(case, asc),
            not_refrac=not_refrac, dt=DT, M=M, hard_reset=False, alpha=ALPHA)
        sum(o.sum() for o in out).backward()
        return val, x, bias

    def run_dense():
        weight = leaf(dense0, True)
        x, bias = leaf(x_seq0, True), leaf(bias0, True)
        asc = leaf(case.asc_amps.reshape(-1), True)
        out = step_fn.dense_multistep_fused(
            x_seq=x, weight=weight, bias=bias, v=case.v.clone(),
            Iasc=case.Iasc.reshape(-1).clone(), params=kernel_params(case, asc),
            not_refrac=not_refrac, dt=DT, M=M, hard_reset=False, alpha=ALPHA)
        sum(o.sum() for o in out).backward()
        return weight, x, bias

    val, xs, bs = run_sparse()
    wd, xd, bd = run_dense()
    for name in ("val", "xs", "bs"):
        assert_nontrivial_grad(locals()[name].grad, name)
    assert_close(val.grad, wd.grad[rows, cols], "grad_val_vs_dense")
    assert_close(xs.grad, xd.grad, "grad_x")
    assert_close(bs.grad, bd.grad, "grad_bias")


# ---------------------------------------------------------------------------
# Integration: the GLIFDenseNet wrapper picks up the fused inference path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_dense_net_matches_eager(backend):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GLIF3 kernels.")
    pytest.importorskip(backend)

    T, N = 16, 32
    gen = torch.Generator(device=DEVICE).manual_seed(11)
    x_seq = make_input_sequence(T, N, "random", seed=12)
    case = make_case(N, 2, "random", seed=12)
    params = {
        "v_th": case.v_th, "v_reset": case.v_reset, "v_rest": case.v_rest,
        "c_m": case.c_m, "tau": case.tau, "k": case.k, "asc_amps": case.asc_amps,
    }
    weight = torch.randn(N, N, generator=gen, device=DEVICE, dtype=DTYPE) / N**0.5
    bias = 0.02 * torch.randn(N, generator=gen, device=DEVICE, dtype=DTYPE)

    def build(provider):
        neuron = build_neuron(provider, N, params, require_grad=False)
        model = GLIFDenseNet(N, neuron, unroll=T)
        init_net_state(model, device=DEVICE, dtype=DTYPE)
        model.linear.weight.data.copy_(weight)
        model.linear.bias.data.copy_(bias)
        return model

    # GLIFDenseNet's fused inference path uses the neuron's own dt (1.0), so the
    # eager reference must run at the same dt.
    with torch.no_grad(), environ.context(dt=1.0):
        spike_ref, _ = build("torch_eager")(x_seq)
        spike_kernel, _ = build(backend)(x_seq)

    assert_nontrivial_spikes(spike_kernel)
    assert_close(spike_kernel, spike_ref, "dense_net_spike")


@pytest.mark.parametrize("kind", INPUT_KINDS)
@pytest.mark.parametrize("M", [1, 3])
def test_dense_multistep_grad_matches_reference(step_fn, M, kind):
    """Training path: gradients w.r.t. the recurrent weight/bias, inputs and
    state must match the eager dense reference."""
    T, B = 12, 32
    hard_reset = False
    case = make_case(B, M, kind, seed=13)
    x_seq_vals = make_input_sequence(T, B, kind, seed=14)
    gen = torch.Generator(device=DEVICE).manual_seed(15)
    weight_vals = torch.randn(B, B, generator=gen, device=DEVICE, dtype=DTYPE) / B**0.5
    bias_vals = 0.02 * torch.randn(B, generator=gen, device=DEVICE, dtype=DTYPE)
    not_refrac = torch.ones(B, device=DEVICE, dtype=DTYPE)

    x_seq = leaf(x_seq_vals, True)
    weight = leaf(weight_vals, True)
    bias = leaf(bias_vals, True)
    v = leaf(case.v, True)
    Iasc = leaf(case.Iasc, True)
    asc = leaf(case.asc_amps.reshape(-1), True)
    params = kernel_params(case, asc)

    s_seq, v_seq, v_out, I_out = step_fn.dense_multistep_fused(
        x_seq=x_seq, weight=weight, bias=bias, v=v, Iasc=Iasc, params=params,
        not_refrac=not_refrac, dt=DT, M=M, hard_reset=hard_reset, alpha=ALPHA,
    )
    assert_nontrivial_spikes(s_seq)
    (s_seq.sum() + v_seq.sum() + v_out.sum() + I_out.sum()).backward()

    x_seq_r = leaf(x_seq_vals, True)
    weight_r = leaf(weight_vals, True)
    bias_r = leaf(bias_vals, True)
    v_r = leaf(case.v, True)
    Iasc_r = leaf(case.Iasc, True)
    glif = _build_glif(case, hard_reset)
    s_ref, v_ref, v_out_ref, I_ref, asc_ref = reference_dense_multistep_grad(
        glif, v_r, Iasc_r, weight_r, bias_r, x_seq_r
    )
    (s_ref.sum() + v_ref.sum() + v_out_ref.sum() + I_ref.sum()).backward()

    assert_close(s_seq, s_ref, "dense_spike_seq")
    assert_close(v_seq, v_ref, "dense_v_seq")
    for name in ("x_seq", "weight", "bias", "asc"):
        assert_nontrivial_grad(locals()[name].grad, name)
    assert_close(x_seq.grad, x_seq_r.grad, "grad_x_seq")
    assert_close(weight.grad, weight_r.grad, "grad_weight")
    assert_close(bias.grad, bias_r.grad, "grad_bias")
    assert_close(asc.grad.view(B, M), asc_ref.grad, "grad_asc")
    assert_close(v.grad, v_r.grad, "grad_v")
    assert_close(Iasc.grad, Iasc_r.grad, "grad_Iasc")
