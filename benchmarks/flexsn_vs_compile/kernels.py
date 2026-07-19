"""Multistep "complicated LIF" implementations used by the FlexSN-vs-compile
benchmark.

Backends (all numerically identical in forward and input gradients):

- ``eager_loop``: plain Python time loop (reference).
- ``scan_compiled``: ``torch.compile`` of the ``scan`` higher-order op.
- ``unrolled_compiled``: ``torch.compile`` of an unrolled Python time loop.
- FlexSN: spikingjelly's ``FlexSN`` (Triton backend), via ``make_flexsn``.
- ``cupy_impl``: hand-written CuPy ``RawKernel`` (forward + analytical BPTT).

Per-step core dynamics::

    h   = beta * v + x
    s1  = sg(h - (rho + 1))          # adaptive-threshold spike
    s2  = sg(h - 1)                  # fixed-threshold spike
    rho = gamma * rho + s1
    yy  = sigmoid(y)
    v   = (h * (1 - s1)) * yy + (h - s2) * (1 - yy)

Surrogate is btorch ``surrogate.ATan(alpha=2)``: forward ``H(x) = (x >= 0)``,
backward ``g'(x) = (alpha / 2) / (1 + (pi / 2 * alpha * x) ** 2)``.
"""
import math

import torch
import torch.nn as nn
from torch._higher_order_ops.scan import scan


BETA = 0.9
GAMMA = 0.8
ALPHA = 2.0  # ATan alpha


# ---- surrogate: pure-torch straight-through mirror of btorch's ATan ----
def atan_sg(x: torch.Tensor) -> torch.Tensor:
    # Heaviside forward, ATan-shaped backward -- identical to
    # btorch.models.surrogate.ATan(alpha=ALPHA), inlined as a pure-torch
    # straight-through so it stays traceable through torch.compile + scan (the
    # autograd.Function form cannot be lowered by the scan HOP).
    spike = (x >= 0.0).to(x)
    soft = 0.5 + torch.atan(0.5 * math.pi * ALPHA * x) / math.pi
    return soft + (spike - soft).detach()


def core_step(x, y, v, rho, spike_fn=atan_sg):
    h = BETA * v + x
    s1 = spike_fn(h - (rho + 1.0))
    s2 = spike_fn(h - 1.0)
    rho = GAMMA * rho + s1
    v1 = h * (1.0 - s1)
    v2 = h - s2
    yy = torch.sigmoid(y)
    v = v1 * yy + v2 * (1.0 - yy)
    return s1, s2, v, rho


def make_core(spike_fn):
    """Return a single-step core with ``spike_fn`` bound in a closure.

    FlexSN traces the core with ``make_fx``, so the surrogate must be captured
    by closure (not passed as a keyword) for the trace to see it.
    """
    def core(x, y, v, rho):
        return core_step(x, y, v, rho, spike_fn=spike_fn)
    return core


# ---------- eager loop ----------
def eager_loop(x_seq, y_seq, v0, rho0):
    T = x_seq.shape[0]
    v, rho = v0, rho0
    s1_l, s2_l = [], []
    for t in range(T):
        s1, s2, v, rho = core_step(x_seq[t], y_seq[t], v, rho)
        s1_l.append(s1)
        s2_l.append(s2)
    return torch.stack(s1_l), torch.stack(s2_l), v, rho


# ---------- scan HOP ----------
def _combine(carry, xs):
    v, rho = carry
    x, y = xs
    s1, s2, v, rho = core_step(x, y, v, rho)
    # clone: inductor's HOP aliasing guard rejects outputs that alias the
    # saved-for-backward tensors
    return (v, rho), (s1.clone(), s2.clone())


def _scan_run(x_seq, y_seq, v0, rho0):
    (v, rho), (s1_seq, s2_seq) = scan(_combine, (v0, rho0), (x_seq, y_seq))
    return s1_seq, s2_seq, v, rho


_scan_run_compiled = torch.compile(_scan_run, fullgraph=True)


def _fix_init(x_seq, y_seq, v0, rho0):
    # torch 2.11 bug: scan autograd drops all cross-step carry gradients when the
    # init carry has requires_grad=False. Force it on during training.
    if torch.is_grad_enabled() and (x_seq.requires_grad or y_seq.requires_grad):
        if not v0.requires_grad:
            v0 = v0.detach().requires_grad_(True)
        if not rho0.requires_grad:
            rho0 = rho0.detach().requires_grad_(True)
    return v0, rho0


def scan_impl(x_seq, y_seq, v0, rho0):
    v0, rho0 = _fix_init(x_seq, y_seq, v0, rho0)
    return _scan_run(x_seq, y_seq, v0, rho0)


def scan_compiled(x_seq, y_seq, v0, rho0):
    v0, rho0 = _fix_init(x_seq, y_seq, v0, rho0)
    return _scan_run_compiled(x_seq, y_seq, v0, rho0)


unrolled_compiled = torch.compile(eager_loop, fullgraph=True, dynamic=False)


# --------------------------------------------------------------------------
# CuPy backend
#
# A ``float4``-vectorized CuPy implementation in the style of
# ``spikingjelly.activation_based.neuron_cupy``: the CUDA source is a raw string
# whose neuron constants are injected per instance as ``constexpr`` values,
# compiled lazily into ``cupy.RawKernel`` objects and driven by a
# ``torch.autograd.Function`` (forward + analytical BPTT) wrapped in an
# ``nn.Module``.
#
# Each thread owns ``VEC`` neurons (one ``float4``) and walks all ``T`` steps
# with the recurrent state ``(v, rho)`` held in registers, so no state ever
# round-trips through global memory between timesteps. The backward pass replays
# the recurrence in reverse, recomputing the spikes and surrogate gradients from
# the saved ``h`` and ``rho`` sequences.
# --------------------------------------------------------------------------

_VEC = 4                 # neurons per thread (float4 load/store width)
_THREADS_PER_BLOCK = 256

_FORWARD_KERNEL_NAME = "complicated_lif_forward"
_BACKWARD_KERNEL_NAME = "complicated_lif_backward"

# Forward kernel body. Neuron constants (``membrane_leak``, ``adapt_decay``,
# the surrogate coefficients) and the ``vec`` width / ``as_float4`` helper come
# from the injected header. Signature is
# ``(x_seq, y_seq, v_init, rho_init, s1_seq, s2_seq, h_seq, rho_prev_seq,
#    v_final, rho_final, numel, time_step)``. ``h_seq`` and ``rho_prev_seq`` are
# the residuals the backward pass replays from.
_FORWARD_KERNEL_BODY = r"""
extern "C" __global__ void complicated_lif_forward(
    const float* __restrict__ x_seq,
    const float* __restrict__ y_seq,
    const float* __restrict__ v_init,
    const float* __restrict__ rho_init,
    float* __restrict__ s1_seq,
    float* __restrict__ s2_seq,
    float* __restrict__ h_seq,
    float* __restrict__ rho_prev_seq,
    float* __restrict__ v_final,
    float* __restrict__ rho_final,
    const int numel, const int time_step, const int save_residuals)
{
    const int base = (blockDim.x * blockIdx.x + threadIdx.x) * vec;
    if (base >= numel) return;
    // float4 is only valid when every timestep row (offset numel*t) is 16-byte
    // aligned, i.e. numel is a multiple of vec; otherwise fall back to scalar.
    const bool aligned = (base + vec - 1 < numel) && (numel % vec == 0);
    const int tail = (numel - base < vec) ? (numel - base) : vec;  // valid lanes

    float v[vec], rho[vec], x[vec], y[vec];
    float s1[vec], s2[vec], h[vec], rho_prev[vec];

    if (aligned) {
        as_float4(v[0])   = as_float4(v_init[base]);
        as_float4(rho[0]) = as_float4(rho_init[base]);
    } else {
        for (int i = 0; i < tail; i++) {
            v[i] = v_init[base + i];
            rho[i] = rho_init[base + i];
        }
    }

    for (int t = 0; t < time_step; t++) {
        const int idx = base + numel * t;
        if (aligned) {
            as_float4(x[0]) = as_float4(x_seq[idx]);
            as_float4(y[0]) = as_float4(y_seq[idx]);
        } else {
            for (int i = 0; i < tail; i++) {
                x[i] = x_seq[idx + i];
                y[i] = y_seq[idx + i];
            }
        }

        #pragma unroll
        for (int i = 0; i < vec; i++) {
            const float membrane = membrane_leak * v[i] + x[i];
            const float adaptive_arg = membrane - (rho[i] + 1.0f);
            const float fixed_arg = membrane - 1.0f;
            const float spike_adaptive = adaptive_arg >= 0.0f ? 1.0f : 0.0f;
            const float spike_fixed = fixed_arg >= 0.0f ? 1.0f : 0.0f;
            const float modulation = 1.0f / (1.0f + __expf(-y[i]));
            h[i] = membrane;       // residual for backward
            rho_prev[i] = rho[i];  // residual for backward
            s1[i] = spike_adaptive;
            s2[i] = spike_fixed;
            rho[i] = adapt_decay * rho[i] + spike_adaptive;  // adaptation update
            // modulated reset (hard reset blended with soft reset by modulation)
            v[i] = (membrane * (1.0f - spike_adaptive)) * modulation
                   + (membrane - spike_fixed) * (1.0f - modulation);
        }

        // residuals (h, rho_prev) are only needed for the backward pass
        if (aligned) {
            as_float4(s1_seq[idx]) = as_float4(s1[0]);
            as_float4(s2_seq[idx]) = as_float4(s2[0]);
            if (save_residuals) {
                as_float4(h_seq[idx])        = as_float4(h[0]);
                as_float4(rho_prev_seq[idx]) = as_float4(rho_prev[0]);
            }
        } else {
            for (int i = 0; i < tail; i++) {
                s1_seq[idx + i] = s1[i];
                s2_seq[idx + i] = s2[i];
            }
            if (save_residuals) {
                for (int i = 0; i < tail; i++) {
                    h_seq[idx + i] = h[i];
                    rho_prev_seq[idx + i] = rho_prev[i];
                }
            }
        }
    }

    if (aligned) {
        as_float4(v_final[base])   = as_float4(v[0]);
        as_float4(rho_final[base]) = as_float4(rho[0]);
    } else {
        for (int i = 0; i < tail; i++) {
            v_final[base + i] = v[i];
            rho_final[base + i] = rho[i];
        }
    }
}
"""

# Reverse-time BPTT kernel body. ``grad_v_final`` / ``grad_rho_final`` are the
# incoming gradients on the returned final states.
_BACKWARD_KERNEL_BODY = r"""
// Surrogate gradient  d(spike)/d(arg) = scale / (1 + curvature * arg^2)
// (matches btorch.models.surrogate.ATan.derivative).
__device__ __forceinline__ float surrogate_grad(float a) {
    return surrogate_scale / (1.0f + surrogate_curvature * a * a);
}

extern "C" __global__ void complicated_lif_backward(
    const float* __restrict__ grad_s1_seq,
    const float* __restrict__ grad_s2_seq,
    const float* __restrict__ grad_v_final,
    const float* __restrict__ grad_rho_final,
    const float* __restrict__ y_seq,
    const float* __restrict__ h_seq,
    const float* __restrict__ rho_prev_seq,
    float* __restrict__ grad_x_seq,
    float* __restrict__ grad_y_seq,
    float* __restrict__ grad_v_init,
    float* __restrict__ grad_rho_init,
    const int numel, const int time_step)
{
    const int base = (blockDim.x * blockIdx.x + threadIdx.x) * vec;
    if (base >= numel) return;
    // float4 is only valid when every timestep row (offset numel*t) is 16-byte
    // aligned, i.e. numel is a multiple of vec; otherwise fall back to scalar.
    const bool aligned = (base + vec - 1 < numel) && (numel % vec == 0);
    const int tail = (numel - base < vec) ? (numel - base) : vec;

    // grad flowing back into the carried state, seeded from the final-state grads
    float gv[vec], grho[vec];
    if (aligned) {
        as_float4(gv[0])   = as_float4(grad_v_final[base]);
        as_float4(grho[0]) = as_float4(grad_rho_final[base]);
    } else {
        for (int i = 0; i < tail; i++) {
            gv[i] = grad_v_final[base + i];
            grho[i] = grad_rho_final[base + i];
        }
    }

    float h[vec], rho_prev[vec], y[vec], gs1[vec], gs2[vec], gx[vec], gy[vec];

    for (int t = time_step - 1; t >= 0; t--) {
        const int idx = base + numel * t;
        if (aligned) {
            as_float4(h[0])        = as_float4(h_seq[idx]);
            as_float4(rho_prev[0]) = as_float4(rho_prev_seq[idx]);
            as_float4(y[0])        = as_float4(y_seq[idx]);
            as_float4(gs1[0])      = as_float4(grad_s1_seq[idx]);
            as_float4(gs2[0])      = as_float4(grad_s2_seq[idx]);
        } else {
            for (int i = 0; i < tail; i++) {
                h[i] = h_seq[idx + i];
                rho_prev[i] = rho_prev_seq[idx + i];
                y[i] = y_seq[idx + i];
                gs1[i] = grad_s1_seq[idx + i];
                gs2[i] = grad_s2_seq[idx + i];
            }
        }

        #pragma unroll
        for (int i = 0; i < vec; i++) {
            const float adaptive_arg = h[i] - (rho_prev[i] + 1.0f);
            const float fixed_arg = h[i] - 1.0f;
            const float spike_adaptive = adaptive_arg >= 0.0f ? 1.0f : 0.0f;
            const float spike_fixed = fixed_arg >= 0.0f ? 1.0f : 0.0f;
            const float modulation = 1.0f / (1.0f + __expf(-y[i]));
            const float grad_adaptive = surrogate_grad(adaptive_arg);
            const float grad_fixed = surrogate_grad(fixed_arg);
            // adjoints on the two spikes: direct output grad + paths through
            // rho (adaptive spike only) and v (both spikes)
            const float d_spike_adaptive =
                gs1[i] + grho[i] - gv[i] * h[i] * modulation;
            const float d_spike_fixed = gs2[i] - gv[i] * (1.0f - modulation);
            // adjoint on membrane; grad wrt x equals it (h = membrane_leak*v + x)
            const float d_membrane =
                gv[i] * ((1.0f - spike_adaptive) * modulation + (1.0f - modulation))
                + d_spike_adaptive * grad_adaptive
                + d_spike_fixed * grad_fixed;
            gy[i] = gv[i]
                    * (h[i] * (1.0f - spike_adaptive) - (h[i] - spike_fixed))
                    * modulation * (1.0f - modulation);
            gx[i] = d_membrane;
            gv[i] = membrane_leak * d_membrane;  // propagate to previous v
            grho[i] = adapt_decay * grho[i]
                      - d_spike_adaptive * grad_adaptive;  // propagate to rho
        }

        if (aligned) {
            as_float4(grad_x_seq[idx]) = as_float4(gx[0]);
            as_float4(grad_y_seq[idx]) = as_float4(gy[0]);
        } else {
            for (int i = 0; i < tail; i++) {
                grad_x_seq[idx + i] = gx[i];
                grad_y_seq[idx + i] = gy[i];
            }
        }
    }

    if (aligned) {
        as_float4(grad_v_init[base])   = as_float4(gv[0]);
        as_float4(grad_rho_init[base]) = as_float4(grho[0]);
    } else {
        for (int i = 0; i < tail; i++) {
            grad_v_init[base + i] = gv[i];
            grad_rho_init[base + i] = grho[i];
        }
    }
}
"""


def _kernel_source(body: str, *, membrane_leak: float, adapt_decay: float,
                   surrogate_scale: float, surrogate_curvature: float) -> str:
    """Prepend the header that injects this neuron's constants and helpers.

    Constants are emitted as ``constexpr`` so the compiler folds them into
    immediates (no macro namespace pollution). ``as_float4`` is the vectorized
    load/store helper; ``vec`` is the neurons-per-thread width.

    Args:
        body: raw CUDA kernel body referencing ``membrane_leak``,
            ``adapt_decay``, ``surrogate_scale``, ``surrogate_curvature``,
            ``vec`` and ``as_float4``.
        membrane_leak: membrane leak factor (``beta``).
        adapt_decay: adaptation decay factor (``gamma``).
        surrogate_scale, surrogate_curvature: surrogate-gradient coefficients.

    Returns:
        Full compilable CUDA source string.
    """
    header = (
        f"constexpr int vec = {_VEC};\n"
        f"constexpr float membrane_leak = {membrane_leak}f;\n"
        f"constexpr float adapt_decay = {adapt_decay}f;\n"
        f"constexpr float surrogate_scale = {surrogate_scale}f;\n"
        f"constexpr float surrogate_curvature = {surrogate_curvature}f;\n"
        # vectorized float4 view of an aligned address; const overload for reads
        "__device__ __forceinline__ float4& as_float4(float& p) {\n"
        "    return reinterpret_cast<float4*>(&p)[0];\n"
        "}\n"
        "__device__ __forceinline__ const float4& as_float4(const float& p) {\n"
        "    return reinterpret_cast<const float4*>(&p)[0];\n"
        "}\n"
    )
    return header + body


def _grid(numel: int) -> int:
    """Blocks needed so every vec-wide lane is covered (ceil, not floor)."""
    lanes = (numel + _VEC - 1) // _VEC
    return (lanes + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK


def _surrogate_grad_coeffs(surrogate_function) -> tuple[float, float]:
    """Coefficients ``(scale, curvature)`` so the surrogate gradient is
    ``scale / (1 + curvature * a^2)``.

    Matches :class:`btorch.models.surrogate.ATan`, whose derivative is
    ``damping * alpha / (2 * (1 + (pi/2 * alpha * a)^2))``.
    """
    from btorch.models import surrogate

    if not isinstance(surrogate_function, surrogate.ATan):
        raise NotImplementedError(
            "ComplicatedLIFNode currently supports btorch surrogate.ATan only; "
            f"got {type(surrogate_function).__name__}."
        )
    alpha = surrogate_function.alpha
    damping = surrogate_function.damping_factor
    scale = damping * alpha / 2.0
    curvature = (math.pi / 2.0 * alpha) ** 2
    return scale, curvature


class ComplicatedLIFNodeCuPy(torch.autograd.Function):
    """Autograd bridge to the CuPy forward / backward kernels.

    Operates on flattened ``[T, numel]`` inputs. Following the spikingjelly
    convention, the compiled kernels are passed in as arguments so the owning
    module controls compilation and caching.
    """

    @staticmethod
    def forward(ctx, x_seq, y_seq, v_init, rho_init,
                forward_kernel, backward_kernel):
        import cupy as cp

        time_step, numel = x_seq.shape
        s1_seq = torch.empty_like(x_seq)
        s2_seq = torch.empty_like(x_seq)
        v_final = torch.empty_like(v_init)
        rho_final = torch.empty_like(rho_init)
        need_grad = any(ctx.needs_input_grad)
        # residuals (h, rho_prev) are only saved when a backward is needed; in
        # inference the kernel skips those stores and these dummy pointers
        # (aliased to s1_seq) are never written.
        h_seq = torch.empty_like(x_seq) if need_grad else s1_seq
        rho_prev_seq = torch.empty_like(x_seq) if need_grad else s1_seq

        with cp.cuda.Device(x_seq.device.index):
            forward_kernel(
                (_grid(numel),), (_THREADS_PER_BLOCK,),
                (x_seq.data_ptr(), y_seq.data_ptr(),
                 v_init.data_ptr(), rho_init.data_ptr(),
                 s1_seq.data_ptr(), s2_seq.data_ptr(),
                 h_seq.data_ptr(), rho_prev_seq.data_ptr(),
                 v_final.data_ptr(), rho_final.data_ptr(),
                 cp.int32(numel), cp.int32(time_step),
                 cp.int32(1 if need_grad else 0)),
            )

        if need_grad:
            ctx.save_for_backward(y_seq, h_seq, rho_prev_seq)
            ctx.backward_kernel = backward_kernel
        return s1_seq, s2_seq, v_final, rho_final

    @staticmethod
    def backward(ctx, grad_s1, grad_s2, grad_v_final, grad_rho_final):
        import cupy as cp

        y_seq, h_seq, rho_prev_seq = ctx.saved_tensors
        time_step, numel = h_seq.shape
        grad_x = torch.empty_like(h_seq)
        grad_y = torch.empty_like(h_seq)
        grad_v_init = torch.empty(numel, dtype=h_seq.dtype, device=h_seq.device)
        grad_rho_init = torch.empty_like(grad_v_init)

        # incoming grads on the returned outputs; None means zero
        grad_s1 = grad_s1.contiguous()
        grad_s2 = grad_s2.contiguous()
        grad_v_final = (torch.zeros_like(grad_v_init) if grad_v_final is None
                        else grad_v_final.contiguous())
        grad_rho_final = (torch.zeros_like(grad_rho_init) if grad_rho_final is None
                          else grad_rho_final.contiguous())

        with cp.cuda.Device(h_seq.device.index):
            ctx.backward_kernel(
                (_grid(numel),), (_THREADS_PER_BLOCK,),
                (grad_s1.data_ptr(), grad_s2.data_ptr(),
                 grad_v_final.data_ptr(), grad_rho_final.data_ptr(),
                 y_seq.data_ptr(), h_seq.data_ptr(), rho_prev_seq.data_ptr(),
                 grad_x.data_ptr(), grad_y.data_ptr(),
                 grad_v_init.data_ptr(), grad_rho_init.data_ptr(),
                 cp.int32(numel), cp.int32(time_step)),
            )

        return grad_x, grad_y, grad_v_init, grad_rho_init, None, None


class ComplicatedLIFNode(nn.Module):
    """CuPy-accelerated multistep "complicated LIF" neuron.

    Written in the style of ``spikingjelly.activation_based.neuron_cupy``: the
    CUDA source is generated with the neuron constants injected as ``constexpr``
    values and compiled lazily into ``cupy.RawKernel`` objects on the first
    forward call.

    The single-step dynamics (per neuron) are::

        h   = beta * v + x
        s1  = Theta(h - (rho + 1))     # adaptive-threshold spike
        s2  = Theta(h - 1)             # fixed-threshold spike
        rho = gamma * rho + s1         # threshold adaptation
        yy  = sigmoid(y)               # modulation factor
        v   = (h * (1 - s1)) * yy + (h - s2) * (1 - yy)

    where ``Theta`` is the Heaviside step, differentiated with the surrogate
    gradient during backprop.

    Args:
        beta: membrane leak factor.
        gamma: adaptation decay factor.
        surrogate_function: spiking surrogate; only
            ``btorch.models.surrogate.ATan`` is supported. Defaults to
            ``ATan(alpha=ALPHA)``.
        store_state_seqs: if ``True``, ``forward`` also returns the final
            ``(v, rho)`` states.

    Shape:
        Input ``x_seq``, ``y_seq``: ``[T, *]`` on a CUDA device; the trailing
        dims are flattened internally. Output spike sequences match that shape.
    """

    def __init__(self, beta: float = BETA, gamma: float = GAMMA,
                 surrogate_function=None, store_state_seqs: bool = False):
        super().__init__()
        if surrogate_function is None:
            from btorch.models import surrogate
            surrogate_function = surrogate.ATan(alpha=ALPHA)
        self.beta = beta
        self.gamma = gamma
        self.surrogate_function = surrogate_function
        self.store_state_seqs = store_state_seqs
        self._surrogate_scale, self._surrogate_curvature = (
            _surrogate_grad_coeffs(surrogate_function))
        self._forward_kernel = None
        self._backward_kernel = None

    def extra_repr(self) -> str:
        return f"beta={self.beta}, gamma={self.gamma}, backend=cupy"

    def _build_kernels(self, dtype: torch.dtype) -> None:
        import cupy as cp

        if dtype != torch.float32:
            raise NotImplementedError(
                f"ComplicatedLIFNode CuPy backend supports float32, got {dtype}."
            )
        kwargs = dict(membrane_leak=self.beta, adapt_decay=self.gamma,
                      surrogate_scale=self._surrogate_scale,
                      surrogate_curvature=self._surrogate_curvature)
        self._forward_kernel = cp.RawKernel(
            _kernel_source(_FORWARD_KERNEL_BODY, **kwargs),
            _FORWARD_KERNEL_NAME, backend="nvrtc")
        self._backward_kernel = cp.RawKernel(
            _kernel_source(_BACKWARD_KERNEL_BODY, **kwargs),
            _BACKWARD_KERNEL_NAME, backend="nvrtc")

    def forward(self, x_seq: torch.Tensor, y_seq: torch.Tensor,
                v_init: torch.Tensor | None = None,
                rho_init: torch.Tensor | None = None):
        if self._forward_kernel is None:
            self._build_kernels(x_seq.dtype)

        shape = x_seq.shape
        T = shape[0]
        x_flat = x_seq.reshape(T, -1).contiguous()
        y_flat = y_seq.reshape(T, -1).contiguous()
        numel = x_flat.shape[1]
        v_init = (x_flat.new_zeros(numel) if v_init is None
                  else v_init.reshape(-1).contiguous())
        rho_init = (x_flat.new_zeros(numel) if rho_init is None
                    else rho_init.reshape(-1).contiguous())

        s1, s2, v_final, rho_final = ComplicatedLIFNodeCuPy.apply(
            x_flat, y_flat, v_init, rho_init,
            self._forward_kernel, self._backward_kernel)

        s1 = s1.reshape(shape)
        s2 = s2.reshape(shape)
        if self.store_state_seqs:
            v_final = v_final.reshape(shape[1:])
            rho_final = rho_final.reshape(shape[1:])
            return s1, s2, v_final, rho_final
        return s1, s2


_default_cupy_node: "ComplicatedLIFNode | None" = None


def cupy_impl(x_seq, y_seq, v0, rho0):
    """Functional wrapper matching the other backends' calling convention.

    Reuses a single lazily-built :class:`ComplicatedLIFNode` so the kernels are
    compiled only once across calls.
    """
    global _default_cupy_node
    if _default_cupy_node is None:
        _default_cupy_node = ComplicatedLIFNode(store_state_seqs=True)
    return _default_cupy_node(x_seq, y_seq, v0, rho0)


# ---------- FlexSN ----------
def make_flexsn(example_shape, device, store_state_seqs=False):
    # FlexSN builds its kernels by tracing the core with make_fx, which needs a
    # spikingjelly surrogate to recover the backward. spikingjelly ATan(alpha)
    # is numerically identical to btorch's ATan(alpha), so this stays consistent
    # with the other backends.
    from spikingjelly.activation_based import surrogate
    from spikingjelly.activation_based.neuron.flexsn import FlexSN
    sg = surrogate.ATan(alpha=ALPHA)

    def flexsn_core(x, y, v, rho):
        return core_step(x, y, v, rho, spike_fn=sg)

    ex = tuple(torch.zeros(example_shape, device=device) for _ in range(4))
    return FlexSN(core=flexsn_core, num_inputs=2, num_states=2, num_outputs=2,
                  example_inputs=ex, backend="triton",
                  store_state_seqs=store_state_seqs)
