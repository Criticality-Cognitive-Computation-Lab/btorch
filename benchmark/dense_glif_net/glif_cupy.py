from __future__ import annotations

import math
from functools import lru_cache

import torch
from jaxtyping import Float


try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None


_PI = math.pi
_ATAN_SCALE = 0.5 * math.pi

_FWD_SRC = r"""
extern "C" __global__
void glif3_fwd(
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
    int B,
    int M,
    float dt,
    int hard_reset,
    float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    float v_i = v[i];
    int base = i * M;

    float I_sum = 0.0f;
    for (int m = 0; m < M; ++m) {
        I_sum += Iasc[base + m];
    }

    float tau_i = tau[i];
    float a = expf(-dt / tau_i);

    float v_inf = v_rest[i] + tau_i * (x[i] + I_sum) / c_m[i];
    float v_prime = v_inf + (v_i - v_inf) * a;

    float denom = v_th[i] - v_reset[i];
    float u = (v_prime - v_th[i]) / denom;
    float s = (u > 0.0f) ? 1.0f : 0.0f;
    s *= mask[i];

    float v_post = v_prime - (hard_reset ? (v_prime - v_reset[i])
                                         : (v_th[i] - v_reset[i])) * s;

    for (int m = 0; m < M; ++m) {
        float k_m = k[base + m];
        float I_dec = Iasc[base + m] * expf(-k_m * dt);
        float I_post = I_dec + asc[base + m] * s;
        I_out[base + m] = I_post;
    }

    v_out[i] = v_post;
    s_out[i] = s;
}
"""

_BWD_SRC = r"""
extern "C" __global__
void glif3_bwd(
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
    int B,
    int M,
    float dt,
    int hard_reset,
    float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    int base = i * M;

    float v_i = v[i];
    float x_i = x[i];
    float v_th_i = v_th[i];
    float v_reset_i = v_reset[i];
    float v_rest_i = v_rest[i];
    float c_m_i = c_m[i];
    float tau_i = tau[i];
    float mask_i = mask[i];

    float I_sum = 0.0f;
    for (int m = 0; m < M; ++m) {
        I_sum += Iasc[base + m];
    }

    float a = expf(-dt / tau_i);
    float v_inf = v_rest_i + tau_i * (x_i + I_sum) / c_m_i;
    float v_prime = v_inf + (v_i - v_inf) * a;

    float denom = v_th_i - v_reset_i;
    float u = (v_prime - v_th_i) / denom;
    float scale = 0.5f * 3.141592653589793f * alpha;
    float s = s_out[i];

    float ds_du = mask_i * (0.5f * alpha)
                  / (1.0f + (scale * u) * (scale * u));
    float ds_dvprime = ds_du / denom;

    float dvprime = hard_reset ? dv_out[i] * (1.0f - s) : dv_out[i];
    float ds_from_v = dv_out[i]
        * (hard_reset ? (-(v_prime - v_reset_i)) : (-(v_th_i - v_reset_i)));

    float dI_s_sum = 0.0f;
    for (int m = 0; m < M; ++m) {
        float dI_post_m = dI_out[base + m];
        dI_s_sum += dI_post_m * asc[base + m];
        dasc[base + m] = dI_post_m * s;
    }

    float ds_total = ds_out[i] + ds_from_v + dI_s_sum;
    dvprime += ds_total * ds_dvprime;

    float dv_i = dvprime * a;
    float dv_inf = dvprime * (1.0f - a);

    dx[i] = dv_inf * (tau_i / c_m_i);
    float dI_common = dv_inf * (tau_i / c_m_i);

    for (int m = 0; m < M; ++m) {
        float k_m = k[base + m];
        float b = expf(-k_m * dt);
        dI[base + m] = dI_out[base + m] * b + dI_common;
    }

    dv[i] = dv_i;
}
"""

_FUSED_TILE_K = 128

_FUSED_SRC = (
    "#define TILE_K "
    + str(_FUSED_TILE_K)
    + "\n"
    + r"""
extern "C" __global__
void glif3_dense_fwd(
    const float* x_seq,  // (T*B,)
    const float* w,      // (B*B,)
    const float* b,      // (B,)
    float* v,            // (B,)
    float* Iasc,         // (B*M,)
    const float* v_th,
    const float* v_reset,
    const float* v_rest,
    const float* c_m,
    const float* tau,
    const float* k,      // (B*M,)
    const float* asc,    // (B*M,)
    const float* mask,
    float* s_seq,        // (T*B,)
    float* v_seq,        // (T*B,)
    float* v_out,        // (B,)
    float* I_out,        // (B*M,)
    int T,
    int B,
    int M,
    float dt,
    int hard_reset,
    float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    int base = i * M;
    float v_i = v[i];
    float v_th_i = v_th[i];
    float v_reset_i = v_reset[i];
    float v_rest_i = v_rest[i];
    float c_m_i = c_m[i];
    float tau_i = tau[i];
    float mask_i = mask[i];

    int w_base = i * B;
    extern __shared__ float s_sh[];

    for (int t = 0; t < T; ++t) {
        float I_sum = 0.0f;
        for (int m = 0; m < M; ++m) {
            I_sum += Iasc[base + m];
        }

        float lin = 0.0f;
        for (int k0 = 0; k0 < B; k0 += TILE_K) {
            int k = k0 + threadIdx.x;
            if (threadIdx.x < TILE_K) {
                float s_prev = 0.0f;
                if (t > 0 && k < B) {
                    s_prev = s_seq[(t - 1) * B + k];
                }
                s_sh[threadIdx.x] = s_prev;
            }
            __syncthreads();
            int k_max = B - k0;
            if (k_max > TILE_K) {
                k_max = TILE_K;
            }
            for (int j = 0; j < k_max; ++j) {
                lin += w[w_base + k0 + j] * s_sh[j];
            }
            __syncthreads();
        }

        float x = x_seq[t * B + i] + b[i] + lin;
        float a = expf(-dt / tau_i);
        float v_inf = v_rest_i + tau_i * (x + I_sum) / c_m_i;
        float v_prime = v_inf + (v_i - v_inf) * a;

        float denom = v_th_i - v_reset_i;
        float u = (v_prime - v_th_i) / denom;
        float s = (u > 0.0f) ? 1.0f : 0.0f;
        s *= mask_i;

        float v_post = v_prime - (hard_reset ? (v_prime - v_reset_i)
                                             : (v_th_i - v_reset_i)) * s;

        for (int m = 0; m < M; ++m) {
            float k_m = k[base + m];
            float I_dec = Iasc[base + m] * expf(-k_m * dt);
            float I_post = I_dec + asc[base + m] * s;
            Iasc[base + m] = I_post;
            I_out[base + m] = I_post;
        }

        v_i = v_post;
        s_seq[t * B + i] = s;
        v_seq[t * B + i] = v_post;
    }

    v[i] = v_i;
    v_out[i] = v_i;
}
"""
)

_MULTI_SRC = r"""
extern "C" __global__
void glif3_multistep_fwd(
    const float* x_seq,  // (T*B,)
    float* v,            // (B,)
    float* Iasc,         // (B*M,)
    const float* v_th,
    const float* v_reset,
    const float* v_rest,
    const float* c_m,
    const float* tau,
    const float* k,      // (B*M,)
    const float* asc,    // (B*M,)
    const float* mask,
    float* s_seq,        // (T*B,)
    float* v_seq,        // (T*B,)
    float* v_out,        // (B,)
    float* I_out,        // (B*M,)
    int T,
    int B,
    int M,
    float dt,
    int hard_reset,
    float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    int base = i * M;
    float v_i = v[i];
    float v_th_i = v_th[i];
    float v_reset_i = v_reset[i];
    float v_rest_i = v_rest[i];
    float c_m_i = c_m[i];
    float tau_i = tau[i];
    float mask_i = mask[i];

    for (int t = 0; t < T; ++t) {
        float I_sum = 0.0f;
        for (int m = 0; m < M; ++m) {
            I_sum += Iasc[base + m];
        }

        float x = x_seq[t * B + i];
        float a = expf(-dt / tau_i);
        float v_inf = v_rest_i + tau_i * (x + I_sum) / c_m_i;
        float v_prime = v_inf + (v_i - v_inf) * a;

        float denom = v_th_i - v_reset_i;
        float u = (v_prime - v_th_i) / denom;
        float s = (u > 0.0f) ? 1.0f : 0.0f;
        s *= mask_i;

        float v_post = v_prime - (hard_reset ? (v_prime - v_reset_i)
                                             : (v_th_i - v_reset_i)) * s;

        for (int m = 0; m < M; ++m) {
            float k_m = k[base + m];
            float I_dec = Iasc[base + m] * expf(-k_m * dt);
            float I_post = I_dec + asc[base + m] * s;
            Iasc[base + m] = I_post;
            I_out[base + m] = I_post;
        }

        v_i = v_post;
        s_seq[t * B + i] = s;
        v_seq[t * B + i] = v_post;
    }

    v[i] = v_i;
    v_out[i] = v_i;
}
"""


def _as_fp32(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor.contiguous()


def _check_shapes(
    B: int,
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
    if Iasc.numel() != B * M:
        raise ValueError("Iasc must have shape (B*M,) flattened.")
    for key in ("v_th", "v_reset", "v_rest", "c_m", "tau"):
        if params[key].numel() != B:
            raise ValueError(f"params['{key}'] must have shape (B,).")
    for key in ("k", "asc_amps"):
        if params[key].numel() != B * M:
            raise ValueError(f"params['{key}'] must have shape (B*M,).")


def _to_cupy_ptr(tensor: torch.Tensor) -> int:
    # NOTE: SpikingJelly-style interop: pass raw device pointers directly to
    # RawKernel. We do not create CuPy ndarrays here (no DLPack), which avoids
    # per-step wrapper overhead and reduces stream sync costs in tight loops.
    if not tensor.is_cuda:
        raise RuntimeError("Expected CUDA tensor for CuPy pointer view.")
    return int(tensor.data_ptr())


def _current_stream() -> "cp.cuda.ExternalStream":
    if cp is None:  # pragma: no cover - optional dependency
        raise RuntimeError("cupy is required for GLIF3 cupy kernels.")
    return cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)


@lru_cache(maxsize=None)
def _get_kernels_cached(device: int):
    if cp is None:  # pragma: no cover - optional dependency
        raise RuntimeError("cupy is required for GLIF3 cupy kernels.")
    with cp.cuda.Device(device):
        fwd = cp.RawKernel(_FWD_SRC, "glif3_fwd")
        bwd = cp.RawKernel(_BWD_SRC, "glif3_bwd")
        fused = cp.RawKernel(_FUSED_SRC, "glif3_dense_fwd")
        multi = cp.RawKernel(_MULTI_SRC, "glif3_multistep_fwd")
    return fwd, bwd, fused, multi


def _get_kernels():
    if cp is None:  # pragma: no cover - optional dependency
        raise RuntimeError("cupy is required for GLIF3 cupy kernels.")
    device = cp.cuda.runtime.getDevice()
    return _get_kernels_cached(int(device))


class GLIF3StepCupy(torch.autograd.Function):
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
        if cp is None:  # pragma: no cover - optional dependency
            raise RuntimeError("cupy is required for GLIF3StepCupy.")
        if not (v.is_cuda and Iasc.is_cuda and x.is_cuda):
            raise RuntimeError("GLIF3StepCupy requires CUDA tensors.")

        v = _as_fp32(v, "v")
        Iasc = _as_fp32(Iasc, "Iasc")
        x = _as_fp32(x, "x")
        v_th = _as_fp32(v_th, "v_th")
        v_reset = _as_fp32(v_reset, "v_reset")
        v_rest = _as_fp32(v_rest, "v_rest")
        c_m = _as_fp32(c_m, "c_m")
        tau = _as_fp32(tau, "tau")
        k = _as_fp32(k, "k")
        asc_amps = _as_fp32(asc_amps, "asc_amps")
        not_refrac = _as_fp32(not_refrac, "not_refrac")

        B = v.numel()
        params = {
            "v_th": v_th,
            "v_reset": v_reset,
            "v_rest": v_rest,
            "c_m": c_m,
            "tau": tau,
            "k": k,
            "asc_amps": asc_amps,
        }
        _check_shapes(B, int(M), v, Iasc, x, params, not_refrac)

        v_out = torch.empty_like(v)
        I_out = torch.empty_like(Iasc)
        s_out = torch.empty_like(v)

        v_cu = _to_cupy_ptr(v)
        I_cu = _to_cupy_ptr(Iasc)
        x_cu = _to_cupy_ptr(x)
        v_th_cu = _to_cupy_ptr(v_th)
        v_reset_cu = _to_cupy_ptr(v_reset)
        v_rest_cu = _to_cupy_ptr(v_rest)
        c_m_cu = _to_cupy_ptr(c_m)
        tau_cu = _to_cupy_ptr(tau)
        k_cu = _to_cupy_ptr(k)
        asc_cu = _to_cupy_ptr(asc_amps)
        mask_cu = _to_cupy_ptr(not_refrac)
        v_out_cu = _to_cupy_ptr(v_out)
        I_out_cu = _to_cupy_ptr(I_out)
        s_out_cu = _to_cupy_ptr(s_out)

        fwd, _, _, _ = _get_kernels()
        grid = ((B + block - 1) // block,)
        with _current_stream():
            fwd(
                grid,
                (block,),
                (
                    v_cu,
                    I_cu,
                    x_cu,
                    v_th_cu,
                    v_reset_cu,
                    v_rest_cu,
                    c_m_cu,
                    tau_cu,
                    k_cu,
                    asc_cu,
                    mask_cu,
                    v_out_cu,
                    I_out_cu,
                    s_out_cu,
                    cp.int32(B),
                    cp.int32(M),
                    cp.float32(dt),
                    cp.int32(1 if hard_reset else 0),
                    cp.float32(alpha),
                ),
            )

        ctx.save_for_backward(
            v,
            Iasc,
            x,
            v_th,
            v_reset,
            v_rest,
            c_m,
            tau,
            k,
            asc_amps,
            not_refrac,
            s_out,
        )
        ctx.dt = float(dt)
        ctx.M = int(M)
        ctx.hard_reset = bool(hard_reset)
        ctx.alpha = float(alpha)
        ctx.block = int(block)

        return v_out, I_out, s_out

    @staticmethod
    def backward(ctx, dv_out, dI_out, ds_out):
        (
            v,
            Iasc,
            x,
            v_th,
            v_reset,
            v_rest,
            c_m,
            tau,
            k,
            asc_amps,
            not_refrac,
            s_out,
        ) = ctx.saved_tensors

        if cp is None:  # pragma: no cover - optional dependency
            raise RuntimeError("cupy is required for GLIF3StepCupy.")

        B = v.numel()
        M = ctx.M
        block = ctx.block

        dv_out = _as_fp32(
            dv_out if dv_out is not None else torch.zeros_like(v), "dv_out"
        )
        dI_out = _as_fp32(
            dI_out if dI_out is not None else torch.zeros_like(Iasc), "dI_out"
        )
        ds_out = _as_fp32(
            ds_out if ds_out is not None else torch.zeros_like(v), "ds_out"
        )

        dv = torch.empty_like(v)
        dI = torch.empty_like(Iasc)
        dx = torch.empty_like(x)
        dasc = torch.empty_like(asc_amps)

        v_cu = _to_cupy_ptr(v)
        I_cu = _to_cupy_ptr(Iasc)
        x_cu = _to_cupy_ptr(x)
        v_th_cu = _to_cupy_ptr(v_th)
        v_reset_cu = _to_cupy_ptr(v_reset)
        v_rest_cu = _to_cupy_ptr(v_rest)
        c_m_cu = _to_cupy_ptr(c_m)
        tau_cu = _to_cupy_ptr(tau)
        k_cu = _to_cupy_ptr(k)
        asc_cu = _to_cupy_ptr(asc_amps)
        mask_cu = _to_cupy_ptr(not_refrac)
        s_cu = _to_cupy_ptr(s_out)
        dv_out_cu = _to_cupy_ptr(dv_out)
        dI_out_cu = _to_cupy_ptr(dI_out)
        ds_out_cu = _to_cupy_ptr(ds_out)
        dv_cu = _to_cupy_ptr(dv)
        dI_cu = _to_cupy_ptr(dI)
        dx_cu = _to_cupy_ptr(dx)
        dasc_cu = _to_cupy_ptr(dasc)

        _, bwd, _, _ = _get_kernels()
        grid = ((B + block - 1) // block,)
        with _current_stream():
            bwd(
                grid,
                (block,),
                (
                    v_cu,
                    I_cu,
                    x_cu,
                    v_th_cu,
                    v_reset_cu,
                    v_rest_cu,
                    c_m_cu,
                    tau_cu,
                    k_cu,
                    asc_cu,
                    mask_cu,
                    s_cu,
                    dv_out_cu,
                    dI_out_cu,
                    ds_out_cu,
                    dv_cu,
                    dI_cu,
                    dx_cu,
                    dasc_cu,
                    cp.int32(B),
                    cp.int32(M),
                    cp.float32(ctx.dt),
                    cp.int32(1 if ctx.hard_reset else 0),
                    cp.float32(ctx.alpha),
                ),
            )

        return (
            dv,
            dI,
            dx,
            None,
            None,
            None,
            None,
            None,
            None,
            dasc,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def glif3_step_cupy(
    v: Float[torch.Tensor, " B"],
    Iasc: Float[torch.Tensor, " B M"],
    x: Float[torch.Tensor, " B"],
    params: dict,
    not_refrac: Float[torch.Tensor, " B"],
    dt: float,
    M: int,
    hard_reset: bool = False,
    alpha: float = 2.0,
    block: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return GLIF3StepCupy.apply(
        v,
        Iasc,
        x,
        params["v_th"],
        params["v_reset"],
        params["v_rest"],
        params["c_m"],
        params["tau"],
        params["k"],
        params["asc_amps"],
        not_refrac,
        float(dt),
        int(M),
        bool(hard_reset),
        float(alpha),
        int(block),
    )


def glif3_dense_multistep_fused_cupy(
    x_seq: Float[torch.Tensor, " T B"],
    weight: torch.Tensor,
    bias: torch.Tensor,
    v: Float[torch.Tensor, " B"],
    Iasc: Float[torch.Tensor, " B M"],
    params: dict,
    not_refrac: Float[torch.Tensor, " B"],
    dt: float,
    M: int,
    hard_reset: bool = False,
    alpha: float = 2.0,
    block: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cp is None:  # pragma: no cover - optional dependency
        raise RuntimeError("cupy is required for fused GLIF3 cupy kernels.")
    if torch.is_grad_enabled():
        raise RuntimeError("Fused multistep CuPy is forward-only.")
    if not x_seq.is_cuda:
        raise RuntimeError("Fused multistep CuPy requires CUDA tensors.")
    if block < _FUSED_TILE_K:
        raise ValueError("block must be >= TILE_K for fused CuPy kernel.")

    T, B = x_seq.shape
    x_seq = _as_fp32(x_seq.reshape(-1), "x_seq")
    weight = _as_fp32(weight.reshape(-1), "weight")
    bias = _as_fp32(bias, "bias")
    v = _as_fp32(v, "v")
    Iasc = _as_fp32(Iasc.reshape(-1), "Iasc")
    v_th = _as_fp32(params["v_th"], "v_th")
    v_reset = _as_fp32(params["v_reset"], "v_reset")
    v_rest = _as_fp32(params["v_rest"], "v_rest")
    c_m = _as_fp32(params["c_m"], "c_m")
    tau = _as_fp32(params["tau"], "tau")
    k = _as_fp32(params["k"], "k")
    asc_amps = _as_fp32(params["asc_amps"], "asc_amps")
    not_refrac = _as_fp32(not_refrac, "not_refrac")

    v_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    s_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    v_out = torch.empty_like(v)
    I_out = torch.empty_like(Iasc)

    x_cu = _to_cupy_ptr(x_seq)
    w_cu = _to_cupy_ptr(weight)
    b_cu = _to_cupy_ptr(bias)
    v_cu = _to_cupy_ptr(v)
    I_cu = _to_cupy_ptr(Iasc)
    v_th_cu = _to_cupy_ptr(v_th)
    v_reset_cu = _to_cupy_ptr(v_reset)
    v_rest_cu = _to_cupy_ptr(v_rest)
    c_m_cu = _to_cupy_ptr(c_m)
    tau_cu = _to_cupy_ptr(tau)
    k_cu = _to_cupy_ptr(k)
    asc_cu = _to_cupy_ptr(asc_amps)
    mask_cu = _to_cupy_ptr(not_refrac)
    s_seq_cu = _to_cupy_ptr(s_seq)
    v_seq_cu = _to_cupy_ptr(v_seq)
    v_out_cu = _to_cupy_ptr(v_out)
    I_out_cu = _to_cupy_ptr(I_out)

    _, _, fused, _ = _get_kernels()
    grid = ((B + block - 1) // block,)
    shared_mem = int(_FUSED_TILE_K * 4)
    with _current_stream():
        fused(
            grid,
            (int(block),),
            (
                x_cu,
                w_cu,
                b_cu,
                v_cu,
                I_cu,
                v_th_cu,
                v_reset_cu,
                v_rest_cu,
                c_m_cu,
                tau_cu,
                k_cu,
                asc_cu,
                mask_cu,
                s_seq_cu,
                v_seq_cu,
                v_out_cu,
                I_out_cu,
                cp.int32(T),
                cp.int32(B),
                cp.int32(M),
                cp.float32(dt),
                cp.int32(1 if hard_reset else 0),
                cp.float32(alpha),
            ),
            shared_mem=shared_mem,
        )

    return s_seq, v_seq, v_out, I_out.view(B, M)


glif3_step_cupy.dense_multistep_fused = glif3_dense_multistep_fused_cupy


def glif3_multistep_fused_cupy(
    x_seq: Float[torch.Tensor, " T B"],
    v: Float[torch.Tensor, " B"],
    Iasc: Float[torch.Tensor, " B M"],
    params: dict,
    not_refrac: Float[torch.Tensor, " B"],
    dt: float,
    M: int,
    hard_reset: bool = False,
    alpha: float = 2.0,
    block: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cp is None:  # pragma: no cover - optional dependency
        raise RuntimeError("cupy is required for fused GLIF3 cupy kernels.")
    if torch.is_grad_enabled():
        raise RuntimeError("Fused multistep CuPy is forward-only.")
    if not x_seq.is_cuda:
        raise RuntimeError("Fused multistep CuPy requires CUDA tensors.")

    T, B = x_seq.shape
    x_seq = _as_fp32(x_seq.reshape(-1), "x_seq")
    v = _as_fp32(v, "v")
    Iasc = _as_fp32(Iasc.reshape(-1), "Iasc")
    v_th = _as_fp32(params["v_th"], "v_th")
    v_reset = _as_fp32(params["v_reset"], "v_reset")
    v_rest = _as_fp32(params["v_rest"], "v_rest")
    c_m = _as_fp32(params["c_m"], "c_m")
    tau = _as_fp32(params["tau"], "tau")
    k = _as_fp32(params["k"], "k")
    asc_amps = _as_fp32(params["asc_amps"], "asc_amps")
    not_refrac = _as_fp32(not_refrac, "not_refrac")

    v_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    s_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    v_out = torch.empty_like(v)
    I_out = torch.empty_like(Iasc)

    x_cu = _to_cupy_ptr(x_seq)
    v_cu = _to_cupy_ptr(v)
    I_cu = _to_cupy_ptr(Iasc)
    v_th_cu = _to_cupy_ptr(v_th)
    v_reset_cu = _to_cupy_ptr(v_reset)
    v_rest_cu = _to_cupy_ptr(v_rest)
    c_m_cu = _to_cupy_ptr(c_m)
    tau_cu = _to_cupy_ptr(tau)
    k_cu = _to_cupy_ptr(k)
    asc_cu = _to_cupy_ptr(asc_amps)
    mask_cu = _to_cupy_ptr(not_refrac)
    s_seq_cu = _to_cupy_ptr(s_seq)
    v_seq_cu = _to_cupy_ptr(v_seq)
    v_out_cu = _to_cupy_ptr(v_out)
    I_out_cu = _to_cupy_ptr(I_out)

    _, _, _, multi = _get_kernels()
    grid = ((B + block - 1) // block,)
    with _current_stream():
        multi(
            grid,
            (int(block),),
            (
                x_cu,
                v_cu,
                I_cu,
                v_th_cu,
                v_reset_cu,
                v_rest_cu,
                c_m_cu,
                tau_cu,
                k_cu,
                asc_cu,
                mask_cu,
                s_seq_cu,
                v_seq_cu,
                v_out_cu,
                I_out_cu,
                cp.int32(T),
                cp.int32(B),
                cp.int32(M),
                cp.float32(dt),
                cp.int32(1 if hard_reset else 0),
                cp.float32(alpha),
            ),
        )

    return s_seq, v_seq, v_out, I_out.view(B, M)


glif3_step_cupy.multistep_fused = glif3_multistep_fused_cupy


class GLIF3Cupy(torch.nn.Module):
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
        v: Float[torch.Tensor, " B"],
        Iasc: Float[torch.Tensor, " B M"],
        x: Float[torch.Tensor, " B"],
        params: dict,
        not_refrac: Float[torch.Tensor, " B"],
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return glif3_step_cupy(
            v=v,
            Iasc=Iasc,
            x=x,
            params=params,
            not_refrac=not_refrac,
            dt=float(dt),
            M=self.M,
            hard_reset=self.hard_reset,
            alpha=self.alpha,
            block=self.block,
        )
