"""CuPy GLIF3 kernels: single-step, neuron multistep, and dense (neuron +
recurrent connection) multistep.

The GLIF3 update per step is

    v_inf   = v_rest + tau * (x + sum I_asc) / c_m
    v'      = v_inf + (v - v_inf) * exp(-dt / tau)
    spike   = heaviside(v' - v_th) * not_refrac
    v_post  = v' - (v' - v_reset) * spike   (hard reset)
            = v' - (v_th - v_reset) * spike (soft reset)
    I_asc'  = I_asc * exp(-k dt) + asc_amps * spike

The device-side dynamics are written as small ``__device__`` helper functions
(``neuronal_charge`` / ``neuronal_fire`` / ``neuronal_reset`` /
``neuronal_adaptation`` / ``surrogate_grad``) that mirror the eager
``btorch.models.neurons.glif.GLIF3`` methods, so each kernel reads as a
composition of those steps.

Organization:
- single-step ``glif3_step_cupy`` (autograd, the training primitive)
- neuron multistep ``glif3_multistep_fused_cupy`` (autograd for training,
  a lean forward-only path for inference)
- dense multistep ``glif3_dense_multistep_fused_cupy`` (inference or training; the
  recurrent matmul is either fused into the single kernel or delegated to
  cuBLAS via ``fused_matmul=False``)

Conventions (flattened, contiguous, fp32 on CUDA):
- ``v``, ``x``, ``not_refrac``: shape ``(N,)``
- ``Iasc``, ``k``, ``asc_amps``: shape ``(N*M,)`` with base ``i*M``
- surrogate gradient is btorch's exact ``ATan`` (damping = 1)
"""

from __future__ import annotations

from functools import lru_cache

import torch
from jaxtyping import Float

from benchmarks.dense_glif_net.glif_common import (
    GLIF3StepOps,
    dense_multistep_autograd,
)


try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None



# Threads cooperating on the recurrent matmul tile in the fused dense kernel.
_FUSED_TILE_K = 128

# ----------------------------------------------------------------------------
# CUDA-C device helpers shared by every kernel below. Each mirrors one step of
# the eager GLIF3 update so the kernels read top-to-bottom like the math.
# ----------------------------------------------------------------------------
_DEVICE_HELPERS = r"""
#define GLIF_PI 3.141592653589793f

// One exponential-Euler step of dx/dt = A*x + N, given the derivative f(x) at x
// and the linear term A: x_next = x + expm1(dt*A)/A * f(x). Mirrors
// btorch.models.ode.exp_euler_step.
__device__ __forceinline__ float exp_euler(
    float x, float derivative, float linear, float dt
) {
    return x + expm1f(dt * linear) / linear * derivative;
}

// Exponential-Euler membrane update -> v' (pre-spike voltage):
// dv/dt = -(v - v_rest)/tau + (x + sum I_asc)/c_m, linear term A = -1/tau.
__device__ __forceinline__ float neuronal_charge(
    float v, float x, float i_sum,
    float v_rest, float c_m, float tau, float dt
) {
    float dv = -(v - v_rest) / tau + (x + i_sum) / c_m;
    return exp_euler(v, dv, -1.0f / tau, dt);
}

// Hard-threshold spike (heaviside), gated by the stop-grad refractory mask.
__device__ __forceinline__ float neuronal_fire(
    float v_prime, float v_th, float mask
) {
    return (v_prime >= v_th ? 1.0f : 0.0f) * mask;
}

// Membrane reset after a spike (hard or soft).
__device__ __forceinline__ float neuronal_reset(
    float v_prime, float spike, float v_th, float v_reset, int hard_reset
) {
    float drop = hard_reset ? (v_prime - v_reset) : (v_th - v_reset);
    return v_prime - drop * spike;
}

// After-spike current for one mode: exp-Euler decay (dI/dt = -k*I) + spike jump.
__device__ __forceinline__ float neuronal_adaptation(
    float i_asc, float k, float asc_amp, float spike, float dt
) {
    return exp_euler(i_asc, -k * i_asc, -k, dt) + asc_amp * spike;
}

// Surrogate gradient d spike / d v' (btorch ATan: 1 / (1 + (alpha*u)^2),
// damping = 1, with u = (v' - v_th) / (v_th - v_reset)).
__device__ __forceinline__ float surrogate_grad(
    float v_prime, float v_th, float v_reset, float mask, float alpha
) {
    float denom = v_th - v_reset;
    float u = (v_prime - v_th) / denom;
    float ds_du = mask / (1.0f + (alpha * u) * (alpha * u));
    return ds_du / denom;
}
"""

_STEP_FORWARD_SRC = _DEVICE_HELPERS + r"""
extern "C" __global__
void glif3_step_forward(
    const float* v,
    const float* Iasc,
    const float* x,
    const float* v_th,
    const float* v_reset,
    const float* v_rest,
    const float* c_m,
    const float* tau,
    const float* k,
    const float* asc,
    const float* mask,
    float* v_out,
    float* I_out,
    float* s_out,
    int N,
    int M,
    float dt,
    int hard_reset,
    float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int base = i * M;

    float i_sum = 0.0f;
    for (int m = 0; m < M; ++m) i_sum += Iasc[base + m];

    float v_prime = neuronal_charge(
        v[i], x[i], i_sum, v_rest[i], c_m[i], tau[i], dt);
    float spike = neuronal_fire(v_prime, v_th[i], mask[i]);
    float v_post = neuronal_reset(v_prime, spike, v_th[i], v_reset[i], hard_reset);

    for (int m = 0; m < M; ++m) {
        I_out[base + m] = neuronal_adaptation(
            Iasc[base + m], k[base + m], asc[base + m], spike, dt);
    }
    v_out[i] = v_post;
    s_out[i] = spike;
}
"""

_STEP_BACKWARD_SRC = _DEVICE_HELPERS + r"""
extern "C" __global__
void glif3_step_backward(
    const float* v,
    const float* Iasc,
    const float* x,
    const float* v_th,
    const float* v_reset,
    const float* v_rest,
    const float* c_m,
    const float* tau,
    const float* k,
    const float* asc,
    const float* mask,
    const float* s_out,
    const float* dv_out,
    const float* dI_out,
    const float* ds_out,
    float* dv,
    float* dI,
    float* dx,
    float* dasc,
    int N,
    int M,
    float dt,
    int hard_reset,
    float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int base = i * M;

    float i_sum = 0.0f;
    for (int m = 0; m < M; ++m) i_sum += Iasc[base + m];

    float decay = expf(-dt / tau[i]);
    float v_prime = neuronal_charge(
        v[i], x[i], i_sum, v_rest[i], c_m[i], tau[i], dt);
    float ds_dvprime = surrogate_grad(v_prime, v_th[i], v_reset[i], mask[i], alpha);

    float spike = s_out[i];
    float dvprime = hard_reset ? dv_out[i] * (1.0f - spike) : dv_out[i];
    float ds_from_v = dv_out[i]
        * (hard_reset ? -(v_prime - v_reset[i]) : -(v_th[i] - v_reset[i]));

    float dI_s_sum = 0.0f;
    for (int m = 0; m < M; ++m) {
        float dI_post_m = dI_out[base + m];
        dI_s_sum += dI_post_m * asc[base + m];
        dasc[base + m] = dI_post_m * spike;
    }

    float ds_total = ds_out[i] + ds_from_v + dI_s_sum;
    dvprime += ds_total * ds_dvprime;

    float dv_inf = dvprime * (1.0f - decay);
    float dI_common = dv_inf * (tau[i] / c_m[i]);
    dx[i] = dI_common;
    dv[i] = dvprime * decay;

    for (int m = 0; m < M; ++m) {
        float b = expf(-k[base + m] * dt);
        dI[base + m] = dI_out[base + m] * b + dI_common;
    }
}
"""

_NEURON_MULTISTEP_FORWARD_SRC = _DEVICE_HELPERS + r"""
// One thread per neuron; the whole time loop runs in-register.
extern "C" __global__
void glif3_neuron_multistep_forward(
    const float* x_seq,  // (T*N,)
    float* v,            // (N,) working state, updated in place
    float* Iasc,         // (N*M,) working state, updated in place
    const float* v_th,
    const float* v_reset,
    const float* v_rest,
    const float* c_m,
    const float* tau,
    const float* k,
    const float* asc,
    const float* mask,
    float* s_seq,        // (T*N,)
    float* v_seq,        // (T*N,)
    float* v_out,        // (N,)
    float* I_out,        // (N*M,)
    int T,
    int N,
    int M,
    float dt,
    int hard_reset,
    float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int base = i * M;

    float v_i = v[i];

    for (int t = 0; t < T; ++t) {
        float i_sum = 0.0f;
        for (int m = 0; m < M; ++m) i_sum += Iasc[base + m];

        float v_prime = neuronal_charge(
            v_i, x_seq[t * N + i], i_sum, v_rest[i], c_m[i], tau[i], dt);
        float spike = neuronal_fire(v_prime, v_th[i], mask[i]);
        float v_post = neuronal_reset(
            v_prime, spike, v_th[i], v_reset[i], hard_reset);

        for (int m = 0; m < M; ++m) {
            Iasc[base + m] = neuronal_adaptation(
                Iasc[base + m], k[base + m], asc[base + m], spike, dt);
        }
        v_i = v_post;
        s_seq[t * N + i] = spike;
        v_seq[t * N + i] = v_post;
    }

    v[i] = v_i;
    v_out[i] = v_i;
    for (int m = 0; m < M; ++m) I_out[base + m] = Iasc[base + m];
}
"""

_NEURON_MULTISTEP_BACKWARD_SRC = _DEVICE_HELPERS + r"""
// Reverse-time BPTT for the neuron-only multistep. I_post/dI are per-neuron
// working buffers; I_asc at each step is reconstructed by inverse decay.
extern "C" __global__
void glif3_neuron_multistep_backward(
    const float* x_seq,  // (T*N,)
    const float* v0,     // (N,)
    const float* v_seq,  // (T*N,)
    const float* s_seq,  // (T*N,)
    float* I_post,       // (N*M,) working buffer, seeded with final I_asc
    float* dI,           // (N*M,) working buffer, seeded with dI_out
    const float* v_th,
    const float* v_reset,
    const float* v_rest,
    const float* c_m,
    const float* tau,
    const float* k,
    const float* asc,
    const float* mask,
    const float* ds_seq, // (T*N,)
    const float* dv_seq, // (T*N,)
    const float* dv_out, // (N,)
    float* dv0,          // (N,)
    float* dx_seq,       // (T*N,)
    float* dasc,         // (N*M,)
    int T,
    int N,
    int M,
    float dt,
    int hard_reset,
    float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int base = i * M;

    float decay = expf(-dt / tau[i]);
    float tau_over_c = tau[i] / c_m[i];
    float dv_post = dv_out[i];

    for (int t = T - 1; t >= 0; --t) {
        dv_post += dv_seq[t * N + i];
        float spike = s_seq[t * N + i];
        float v_pre = (t == 0) ? v0[i] : v_seq[(t - 1) * N + i];

        // Reconstruct I_asc(t) from I_post(t) by undoing decay + jump.
        float i_sum = 0.0f;
        for (int m = 0; m < M; ++m) {
            float b = expf(-k[base + m] * dt);
            float i_pre = (I_post[base + m] - asc[base + m] * spike) / b;
            I_post[base + m] = i_pre;
            i_sum += i_pre;
        }

        float v_prime = neuronal_charge(
            v_pre, x_seq[t * N + i], i_sum, v_rest[i], c_m[i], tau[i], dt);
        float ds_dvprime = surrogate_grad(
            v_prime, v_th[i], v_reset[i], mask[i], alpha);

        float dvprime = hard_reset ? dv_post * (1.0f - spike) : dv_post;
        float ds_from_v = dv_post
            * (hard_reset ? -(v_prime - v_reset[i]) : -(v_th[i] - v_reset[i]));

        float dI_s_sum = 0.0f;
        for (int m = 0; m < M; ++m) dI_s_sum += dI[base + m] * asc[base + m];

        float ds_total = ds_seq[t * N + i] + ds_from_v + dI_s_sum;
        dvprime += ds_total * ds_dvprime;

        float dv_inf = dvprime * (1.0f - decay);
        float dI_common = dv_inf * tau_over_c;
        dx_seq[t * N + i] = dI_common;

        for (int m = 0; m < M; ++m) {
            float b = expf(-k[base + m] * dt);
            float dI_old = dI[base + m];
            dasc[base + m] += dI_old * spike;
            dI[base + m] = dI_old * b + dI_common;
        }
        dv_post = dvprime * decay;
    }
    dv0[i] = dv_post;
}
"""

_DENSE_MULTISTEP_FORWARD_SRC = (
    "#define TILE_K " + str(_FUSED_TILE_K) + "\n"
    + _DEVICE_HELPERS + r"""
// Fully fused dense (neuron + recurrent connection) multistep, inference only.
// The recurrent term lin_i = sum_j w[i,j] * s_prev[j] is computed inside the
// kernel with a shared-memory tiled matmul over the previous step's spikes.
extern "C" __global__
void glif3_dense_multistep_forward(
    const float* x_seq,  // (T*N,)
    const float* w,      // (N*N,)
    const float* b,      // (N,)
    float* v,            // (N,)
    float* Iasc,         // (N*M,)
    const float* v_th,
    const float* v_reset,
    const float* v_rest,
    const float* c_m,
    const float* tau,
    const float* k,
    const float* asc,
    const float* mask,
    float* s_seq,        // (T*N,)
    float* v_seq,        // (T*N,)
    float* v_out,        // (N,)
    float* I_out,        // (N*M,)
    int T,
    int N,
    int M,
    float dt,
    int hard_reset,
    float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = i < N;
    int base = i * M;
    int w_base = i * N;

    float v_i = active ? v[i] : 0.0f;

    extern __shared__ float s_prev_tile[];

    for (int t = 0; t < T; ++t) {
        // Tiled recurrent matmul over the previous step's spikes.
        float lin = 0.0f;
        for (int k0 = 0; k0 < N; k0 += TILE_K) {
            int col = k0 + threadIdx.x;
            if (threadIdx.x < TILE_K) {
                float s_prev = (t > 0 && col < N) ? s_seq[(t - 1) * N + col] : 0.0f;
                s_prev_tile[threadIdx.x] = s_prev;
            }
            __syncthreads();
            int tile = min(TILE_K, N - k0);
            if (active) {
                for (int j = 0; j < tile; ++j)
                    lin += w[w_base + k0 + j] * s_prev_tile[j];
            }
            __syncthreads();
        }
        if (!active) continue;

        float i_sum = 0.0f;
        for (int m = 0; m < M; ++m) i_sum += Iasc[base + m];

        float x_in = x_seq[t * N + i] + b[i] + lin;
        float v_prime = neuronal_charge(
            v_i, x_in, i_sum, v_rest[i], c_m[i], tau[i], dt);
        float spike = neuronal_fire(v_prime, v_th[i], mask[i]);
        float v_post = neuronal_reset(
            v_prime, spike, v_th[i], v_reset[i], hard_reset);

        for (int m = 0; m < M; ++m) {
            Iasc[base + m] = neuronal_adaptation(
                Iasc[base + m], k[base + m], asc[base + m], spike, dt);
        }
        v_i = v_post;
        s_seq[t * N + i] = spike;
        v_seq[t * N + i] = v_post;
    }

    if (!active) return;
    v[i] = v_i;
    v_out[i] = v_i;
    for (int m = 0; m < M; ++m) I_out[base + m] = Iasc[base + m];
}
"""
)

_DENSE_STEP_SRC = _DEVICE_HELPERS + r"""
// One dense step given a precomputed total input x_in = x + bias + W @ s_prev.
// Used by the non-fused (fused_matmul=False) path where the recurrent matmul
// is delegated to cuBLAS between launches.
extern "C" __global__
void glif3_dense_step(
    const float* x_in,   // (N,) already includes bias + recurrent term
    float* v,            // (N,)
    float* Iasc,         // (N*M,)
    const float* v_th,
    const float* v_reset,
    const float* v_rest,
    const float* c_m,
    const float* tau,
    const float* k,
    const float* asc,
    const float* mask,
    float* s_out,        // (N,) this step's spikes
    float* v_out,        // (N,) this step's voltage
    int N,
    int M,
    float dt,
    int hard_reset,
    float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int base = i * M;

    float i_sum = 0.0f;
    for (int m = 0; m < M; ++m) i_sum += Iasc[base + m];

    float v_prime = neuronal_charge(
        v[i], x_in[i], i_sum, v_rest[i], c_m[i], tau[i], dt);
    float spike = neuronal_fire(v_prime, v_th[i], mask[i]);
    float v_post = neuronal_reset(v_prime, spike, v_th[i], v_reset[i], hard_reset);

    for (int m = 0; m < M; ++m) {
        Iasc[base + m] = neuronal_adaptation(
            Iasc[base + m], k[base + m], asc[base + m], spike, dt);
    }
    v[i] = v_post;
    s_out[i] = spike;
    v_out[i] = v_post;
}
"""


# ----------------------------------------------------------------------------
# Python-side helpers
# ----------------------------------------------------------------------------
def _as_fp32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor.contiguous()


def _check_shapes(
    N: int,
    M: int,
    v: torch.Tensor,
    Iasc: torch.Tensor,
    x: torch.Tensor,
    params: dict,
    not_refrac: torch.Tensor,
) -> None:
    if v.ndim != 1 or x.ndim != 1 or not_refrac.ndim != 1:
        raise ValueError("v, x, not_refrac must be 1D tensors.")
    if v.numel() != x.numel() or v.numel() != not_refrac.numel():
        raise ValueError("v, x, not_refrac must have the same length.")
    if Iasc.numel() != N * M:
        raise ValueError("Iasc must have shape (N*M,) flattened.")
    for key in ("v_th", "v_reset", "v_rest", "c_m", "tau"):
        if params[key].numel() != N:
            raise ValueError(f"params['{key}'] must have shape (N,).")
    for key in ("k", "asc_amps"):
        if params[key].numel() != N * M:
            raise ValueError(f"params['{key}'] must have shape (N*M,).")


def _require_cupy() -> None:
    if cp is None:  # pragma: no cover - optional dependency
        raise RuntimeError("cupy is required for GLIF3 cupy kernels.")


def _ptr(tensor: torch.Tensor) -> int:
    # SpikingJelly-style interop: hand raw device pointers to RawKernel instead
    # of wrapping in CuPy ndarrays (no DLPack), which avoids per-step overhead.
    if not tensor.is_cuda:
        raise RuntimeError("Expected CUDA tensor for CuPy pointer view.")
    return int(tensor.data_ptr())


def _current_stream() -> "cp.cuda.Stream":
    _require_cupy()
    return cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)


@lru_cache(maxsize=None)
def _compile_kernels(device: int):
    _require_cupy()
    with cp.cuda.Device(device):
        return {
            "step_forward": cp.RawKernel(_STEP_FORWARD_SRC, "glif3_step_forward"),
            "step_backward": cp.RawKernel(_STEP_BACKWARD_SRC, "glif3_step_backward"),
            "neuron_forward": cp.RawKernel(
                _NEURON_MULTISTEP_FORWARD_SRC, "glif3_neuron_multistep_forward"
            ),
            "neuron_backward": cp.RawKernel(
                _NEURON_MULTISTEP_BACKWARD_SRC, "glif3_neuron_multistep_backward"
            ),
            "dense_forward": cp.RawKernel(
                _DENSE_MULTISTEP_FORWARD_SRC, "glif3_dense_multistep_forward"
            ),
            "dense_step": cp.RawKernel(_DENSE_STEP_SRC, "glif3_dense_step"),
        }


def _kernels():
    _require_cupy()
    return _compile_kernels(int(cp.cuda.runtime.getDevice()))


def _scalars(N: int, M: int, dt: float, hard_reset: bool, alpha: float):
    return (
        cp.int32(N),
        cp.int32(M),
        cp.float32(dt),
        cp.int32(1 if hard_reset else 0),
        cp.float32(alpha),
    )


def _grid(n: int, block: int):
    return ((n + block - 1) // block,)


# ----------------------------------------------------------------------------
# Single-step (training primitive)
# ----------------------------------------------------------------------------
class GLIF3StepCuPy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v: torch.Tensor,
        Iasc: torch.Tensor,
        x: torch.Tensor,
        v_th: torch.Tensor,
        v_reset: torch.Tensor,
        v_rest: torch.Tensor,
        c_m: torch.Tensor,
        tau: torch.Tensor,
        k: torch.Tensor,
        asc_amps: torch.Tensor,
        not_refrac: torch.Tensor,
        dt: float,
        M: int,
        hard_reset: bool,
        alpha: float,
        block: int = 256,
    ):
        _require_cupy()
        if not (v.is_cuda and Iasc.is_cuda and x.is_cuda):
            raise RuntimeError("GLIF3StepCuPy requires CUDA tensors.")

        v, Iasc, x = _as_fp32(v), _as_fp32(Iasc), _as_fp32(x)
        v_th, v_reset, v_rest = _as_fp32(v_th), _as_fp32(v_reset), _as_fp32(v_rest)
        c_m, tau, k = _as_fp32(c_m), _as_fp32(tau), _as_fp32(k)
        asc_amps, not_refrac = _as_fp32(asc_amps), _as_fp32(not_refrac)

        N = v.numel()
        params = {
            "v_th": v_th, "v_reset": v_reset, "v_rest": v_rest,
            "c_m": c_m, "tau": tau, "k": k, "asc_amps": asc_amps,
        }
        _check_shapes(N, int(M), v, Iasc, x, params, not_refrac)

        v_out = torch.empty_like(v)
        I_out = torch.empty_like(Iasc)
        s_out = torch.empty_like(v)

        with _current_stream():
            _kernels()["step_forward"](
                _grid(N, block), (block,),
                (
                    _ptr(v), _ptr(Iasc), _ptr(x),
                    _ptr(v_th), _ptr(v_reset), _ptr(v_rest),
                    _ptr(c_m), _ptr(tau), _ptr(k), _ptr(asc_amps), _ptr(not_refrac),
                    _ptr(v_out), _ptr(I_out), _ptr(s_out),
                    *_scalars(N, int(M), dt, hard_reset, alpha),
                ),
            )

        ctx.save_for_backward(
            v, Iasc, x, v_th, v_reset, v_rest, c_m, tau, k, asc_amps, not_refrac, s_out
        )
        ctx.dt, ctx.M = float(dt), int(M)
        ctx.hard_reset = bool(hard_reset)
        ctx.alpha, ctx.block = float(alpha), int(block)
        return v_out, I_out, s_out

    @staticmethod
    def backward(ctx, dv_out, dI_out, ds_out):
        (v, Iasc, x, v_th, v_reset, v_rest, c_m, tau, k, asc_amps, not_refrac,
         s_out) = ctx.saved_tensors
        N, M, block = v.numel(), ctx.M, ctx.block

        dv_out = _as_fp32(dv_out if dv_out is not None else torch.zeros_like(v))
        dI_out = _as_fp32(dI_out if dI_out is not None else torch.zeros_like(Iasc))
        ds_out = _as_fp32(ds_out if ds_out is not None else torch.zeros_like(v))

        dv = torch.empty_like(v)
        dI = torch.empty_like(Iasc)
        dx = torch.empty_like(x)
        dasc = torch.empty_like(asc_amps)

        with _current_stream():
            _kernels()["step_backward"](
                _grid(N, block), (block,),
                (
                    _ptr(v), _ptr(Iasc), _ptr(x),
                    _ptr(v_th), _ptr(v_reset), _ptr(v_rest),
                    _ptr(c_m), _ptr(tau), _ptr(k), _ptr(asc_amps), _ptr(not_refrac),
                    _ptr(s_out), _ptr(dv_out), _ptr(dI_out), _ptr(ds_out),
                    _ptr(dv), _ptr(dI), _ptr(dx), _ptr(dasc),
                    *_scalars(N, M, ctx.dt, ctx.hard_reset, ctx.alpha),
                ),
            )

        return (dv, dI, dx, None, None, None, None, None, None, dasc,
                None, None, None, None, None, None)


def _glif3_step_cupy(
    v: Float[torch.Tensor, " N"],
    Iasc: Float[torch.Tensor, " N M"],
    x: Float[torch.Tensor, " N"],
    params: dict,
    not_refrac: Float[torch.Tensor, " N"],
    dt: float,
    M: int,
    hard_reset: bool = False,
    alpha: float = 2.0,
    block: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-step GLIF3 update (autograd) using CuPy."""
    return GLIF3StepCuPy.apply(
        v, Iasc, x,
        params["v_th"], params["v_reset"], params["v_rest"],
        params["c_m"], params["tau"], params["k"], params["asc_amps"], not_refrac,
        float(dt), int(M), bool(hard_reset), float(alpha), int(block),
    )


# ----------------------------------------------------------------------------
# Neuron multistep
# ----------------------------------------------------------------------------
def _neuron_multistep_forward(
    x_seq, v_work, I_work, params, not_refrac, dt, M, hard_reset, alpha, block
):
    """Launch the forward neuron-multistep kernel; returns (s_seq, v_seq, v_out, I_out).

    ``v_work`` / ``I_work`` are updated in place, so callers that must preserve
    the originals (training) should pass clones.
    """
    T, N = x_seq.shape
    v_seq = torch.empty((T, N), device=v_work.device, dtype=v_work.dtype)
    s_seq = torch.empty((T, N), device=v_work.device, dtype=v_work.dtype)
    v_out = torch.empty_like(v_work)
    I_out = torch.empty_like(I_work)

    with _current_stream():
        _kernels()["neuron_forward"](
            _grid(N, block), (block,),
            (
                _ptr(x_seq.reshape(-1)), _ptr(v_work), _ptr(I_work),
                _ptr(params["v_th"]), _ptr(params["v_reset"]), _ptr(params["v_rest"]),
                _ptr(params["c_m"]), _ptr(params["tau"]),
                _ptr(params["k"]), _ptr(params["asc_amps"]), _ptr(not_refrac),
                _ptr(s_seq), _ptr(v_seq), _ptr(v_out), _ptr(I_out),
                cp.int32(T), *_scalars(N, int(M), dt, hard_reset, alpha),
            ),
        )
    return s_seq, v_seq, v_out, I_out


class GLIF3NeuronMultiStepCuPy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_seq,
        v,
        Iasc,
        v_th,
        v_reset,
        v_rest,
        c_m,
        tau,
        k,
        asc_amps,
        not_refrac,
        dt: float,
        M: int,
        hard_reset: bool,
        alpha: float,
        block: int = 256,
    ):
        if not x_seq.is_cuda:
            raise RuntimeError("Neuron multistep CuPy requires CUDA tensors.")

        x_seq, v, Iasc = _as_fp32(x_seq), _as_fp32(v), _as_fp32(Iasc)
        v_th, v_reset, v_rest = _as_fp32(v_th), _as_fp32(v_reset), _as_fp32(v_rest)
        c_m, tau, k = _as_fp32(c_m), _as_fp32(tau), _as_fp32(k)
        asc_amps, not_refrac = _as_fp32(asc_amps), _as_fp32(not_refrac)

        T, N = x_seq.shape
        ctx.Iasc_was_flat = Iasc.ndim == 1
        ctx.asc_was_flat = asc_amps.ndim == 1
        if Iasc.numel() != N * int(M):
            raise ValueError("Iasc must have N*M elements for multistep.")
        I_flat = Iasc.reshape(-1)

        params = {
            "v_th": v_th, "v_reset": v_reset, "v_rest": v_rest,
            "c_m": c_m, "tau": tau, "k": k, "asc_amps": asc_amps,
        }
        # Training path preserves the initial state for the backward replay.
        s_seq, v_seq, v_out, I_out = _neuron_multistep_forward(
            x_seq, v.clone(), I_flat.clone(), params, not_refrac,
            dt, int(M), hard_reset, alpha, int(block),
        )

        ctx.save_for_backward(
            x_seq, v, v_th, v_reset, v_rest, c_m, tau, k, asc_amps, not_refrac,
            s_seq, v_seq, I_out,
        )
        ctx.dt, ctx.M = float(dt), int(M)
        ctx.hard_reset = bool(hard_reset)
        ctx.alpha, ctx.block = float(alpha), int(block)
        return s_seq, v_seq, v_out, I_out.view(N, int(M))

    @staticmethod
    def backward(ctx, ds_seq, dv_seq, dv_out, dI_out):
        (x_seq, v0, v_th, v_reset, v_rest, c_m, tau, k, asc_amps, not_refrac,
         s_seq, v_seq, I_out) = ctx.saved_tensors
        T, N, M, block = *x_seq.shape, ctx.M, ctx.block

        ds_seq = ds_seq if ds_seq is not None else torch.zeros_like(s_seq)
        dv_seq = dv_seq if dv_seq is not None else torch.zeros_like(v_seq)
        dv_out = dv_out if dv_out is not None else torch.zeros_like(v0)
        dI_out = dI_out if dI_out is not None else torch.zeros_like(I_out)

        # Working buffers seeded with the final I_asc and its upstream gradient.
        I_post = _as_fp32(I_out.clone())
        dI = _as_fp32(dI_out.reshape(-1).clone())
        dv0 = torch.empty_like(v0)
        dx_seq = torch.empty_like(x_seq)
        dasc = torch.zeros_like(dI)

        tensors = dict(
            x_seq=_as_fp32(x_seq.reshape(-1)), v0=_as_fp32(v0),
            v_seq=_as_fp32(v_seq.reshape(-1)), s_seq=_as_fp32(s_seq.reshape(-1)),
            ds_seq=_as_fp32(ds_seq.reshape(-1)), dv_seq=_as_fp32(dv_seq.reshape(-1)),
            dv_out=_as_fp32(dv_out),
        )
        with _current_stream():
            _kernels()["neuron_backward"](
                _grid(N, block), (block,),
                (
                    _ptr(tensors["x_seq"]), _ptr(tensors["v0"]),
                    _ptr(tensors["v_seq"]), _ptr(tensors["s_seq"]),
                    _ptr(I_post), _ptr(dI),
                    _ptr(v_th), _ptr(v_reset), _ptr(v_rest),
                    _ptr(c_m), _ptr(tau), _ptr(k), _ptr(asc_amps), _ptr(not_refrac),
                    _ptr(tensors["ds_seq"]), _ptr(tensors["dv_seq"]),
                    _ptr(tensors["dv_out"]),
                    _ptr(dv0), _ptr(dx_seq), _ptr(dasc),
                    cp.int32(T), *_scalars(N, M, ctx.dt, ctx.hard_reset, ctx.alpha),
                ),
            )

        dI0 = dI if ctx.Iasc_was_flat else dI.view(N, M)
        dasc_out = dasc if ctx.asc_was_flat else dasc.view(N, M)
        return (dx_seq, dv0, dI0, None, None, None, None, None, None, dasc_out,
                None, None, None, None, None, None)


def glif3_multistep_fused_cupy(
    x_seq: Float[torch.Tensor, " T N"],
    v: Float[torch.Tensor, " N"],
    Iasc: Float[torch.Tensor, " N M"],
    params: dict,
    not_refrac: Float[torch.Tensor, " N"],
    dt: float,
    M: int,
    hard_reset: bool = False,
    alpha: float = 2.0,
    block: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Neuron-only multistep. Uses a lean forward-only path for inference and
    the autograd path (saving intermediates) when gradients are required."""
    _require_cupy()
    need_grad = torch.is_grad_enabled() and (
        x_seq.requires_grad or v.requires_grad or Iasc.requires_grad
        or params["asc_amps"].requires_grad
    )
    if need_grad:
        return GLIF3NeuronMultiStepCuPy.apply(
            x_seq, v, Iasc,
            params["v_th"], params["v_reset"], params["v_rest"],
            params["c_m"], params["tau"], params["k"], params["asc_amps"], not_refrac,
            float(dt), int(M), bool(hard_reset), float(alpha), int(block),
        )

    if not x_seq.is_cuda:
        raise RuntimeError("Neuron multistep CuPy requires CUDA tensors.")
    x_seq = _as_fp32(x_seq)
    N = v.numel()
    fp32 = {key: _as_fp32(params[key]) for key in params}
    s_seq, v_seq, v_out, I_out = _neuron_multistep_forward(
        x_seq, _as_fp32(v), _as_fp32(Iasc.reshape(-1)), fp32, _as_fp32(not_refrac),
        float(dt), int(M), bool(hard_reset), float(alpha), int(block),
    )
    return s_seq, v_seq, v_out, I_out.view(N, int(M))


# ----------------------------------------------------------------------------
# Dense (neuron + recurrent connection) multistep — inference only
# ----------------------------------------------------------------------------
def _dense_multistep_fused(
    x_seq, weight, bias, v, Iasc, params, not_refrac, dt, M, hard_reset, alpha, block
):
    """Single fused kernel that also computes the recurrent matmul in-kernel."""
    if block < _FUSED_TILE_K:
        raise ValueError("block must be >= TILE_K for the fused dense CuPy kernel.")
    T, N = x_seq.shape
    v_seq = torch.empty((T, N), device=v.device, dtype=v.dtype)
    s_seq = torch.empty((T, N), device=v.device, dtype=v.dtype)
    v_out = torch.empty_like(v)
    I_out = torch.empty_like(Iasc)

    with _current_stream():
        _kernels()["dense_forward"](
            _grid(N, block), (block,),
            (
                _ptr(x_seq.reshape(-1)), _ptr(weight.reshape(-1)), _ptr(bias),
                _ptr(v), _ptr(Iasc),
                _ptr(params["v_th"]), _ptr(params["v_reset"]), _ptr(params["v_rest"]),
                _ptr(params["c_m"]), _ptr(params["tau"]),
                _ptr(params["k"]), _ptr(params["asc_amps"]), _ptr(not_refrac),
                _ptr(s_seq), _ptr(v_seq), _ptr(v_out), _ptr(I_out),
                cp.int32(T), *_scalars(N, int(M), dt, hard_reset, alpha),
            ),
            shared_mem=int(_FUSED_TILE_K * 4),
        )
    return s_seq, v_seq, v_out, I_out


def _dense_multistep_matmul(
    x_seq, weight, bias, v, Iasc, params, not_refrac, dt, M, hard_reset, alpha, block
):
    """Non-fused path: cuBLAS matmul for the recurrent term, neuron step kernel
    for the dynamics. One launch pair per timestep."""
    T, N = x_seq.shape
    v_seq = torch.empty((T, N), device=v.device, dtype=v.dtype)
    s_seq = torch.empty((T, N), device=v.device, dtype=v.dtype)
    v_out = torch.empty_like(v)
    I_out = torch.empty_like(Iasc)
    step_v = torch.empty_like(v)
    step_s = torch.empty_like(v)
    s_prev = torch.zeros((N,), device=v.device, dtype=v.dtype)
    scalars = _scalars(N, int(M), dt, hard_reset, alpha)
    step = _kernels()["dense_step"]

    with _current_stream():
        for t in range(T):
            # lin_i = sum_j weight[i, j] * s_prev[j]  (cuBLAS matrix-vector).
            x_in = torch.addmv(x_seq[t] + bias, weight, s_prev)
            step(
                _grid(N, block), (block,),
                (
                    _ptr(x_in), _ptr(v), _ptr(Iasc),
                    _ptr(params["v_th"]), _ptr(params["v_reset"]),
                    _ptr(params["v_rest"]), _ptr(params["c_m"]), _ptr(params["tau"]),
                    _ptr(params["k"]), _ptr(params["asc_amps"]), _ptr(not_refrac),
                    _ptr(step_s), _ptr(step_v), *scalars,
                ),
            )
            s_seq[t] = step_s
            v_seq[t] = step_v
            s_prev = step_s.clone()

    v_out.copy_(v)
    I_out.copy_(Iasc)
    return s_seq, v_seq, v_out, I_out


def glif3_dense_multistep_fused_cupy(
    x_seq: Float[torch.Tensor, " T N"],
    weight: torch.Tensor,
    bias: torch.Tensor,
    v: Float[torch.Tensor, " N"],
    Iasc: Float[torch.Tensor, " N M"],
    params: dict,
    not_refrac: Float[torch.Tensor, " N"],
    dt: float,
    M: int,
    hard_reset: bool = False,
    alpha: float = 2.0,
    block: int = 256,
    fused_matmul: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense (neuron + recurrent connection) multistep (inference or training).

    With ``fused_matmul=True`` the recurrent matmul is computed inside the single
    neuron kernel; with ``fused_matmul=False`` it is delegated to cuBLAS and the
    neuron dynamics run in a per-step kernel.
    """
    _require_cupy()
    need_grad = torch.is_grad_enabled() and (
        weight.requires_grad or bias.requires_grad or x_seq.requires_grad
        or v.requires_grad or Iasc.requires_grad or params["asc_amps"].requires_grad
    )
    if need_grad:
        return dense_multistep_autograd(
            _glif3_step_cupy, x_seq, weight, bias, v, Iasc, params, not_refrac,
            float(dt), int(M), bool(hard_reset), float(alpha),
        )
    if not x_seq.is_cuda:
        raise RuntimeError("Fused dense multistep CuPy requires CUDA tensors.")
    N = v.numel()
    if weight.shape != (N, N):
        raise ValueError("weight must have shape (N, N).")
    if bias.shape != (N,):
        raise ValueError("bias must have shape (N,).")

    x_seq, weight, bias = _as_fp32(x_seq), _as_fp32(weight), _as_fp32(bias)
    v, Iasc = _as_fp32(v), _as_fp32(Iasc.reshape(-1))
    fp32 = {key: _as_fp32(params[key]) for key in params}
    not_refrac = _as_fp32(not_refrac)

    path = _dense_multistep_fused if fused_matmul else _dense_multistep_matmul
    s_seq, v_seq, v_out, I_out = path(
        x_seq, weight, bias, v, Iasc, fp32, not_refrac,
        float(dt), int(M), bool(hard_reset), float(alpha), int(block),
    )
    return s_seq, v_seq, v_out, I_out.view(N, int(M))


glif3_step_cupy = GLIF3StepOps(
    step=_glif3_step_cupy,
    multistep_fused=glif3_multistep_fused_cupy,
    dense_multistep_fused=glif3_dense_multistep_fused_cupy,
)


class GLIF3CuPy(torch.nn.Module):
    """Thin module wrapper exposing a ``step`` API."""

    def __init__(
        self, M: int, hard_reset: bool = False, alpha: float = 2.0, block: int = 256
    ):
        super().__init__()
        self.M = int(M)
        self.hard_reset = bool(hard_reset)
        self.alpha = float(alpha)
        self.block = int(block)

    def step(
        self,
        v: Float[torch.Tensor, " N"],
        Iasc: Float[torch.Tensor, " N M"],
        x: Float[torch.Tensor, " N"],
        params: dict,
        not_refrac: Float[torch.Tensor, " N"],
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _glif3_step_cupy(
            v=v, Iasc=Iasc, x=x, params=params, not_refrac=not_refrac,
            dt=float(dt), M=self.M, hard_reset=self.hard_reset,
            alpha=self.alpha, block=self.block,
        )
