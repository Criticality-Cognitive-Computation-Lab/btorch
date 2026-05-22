import triton
import triton.language as tl


@triton.jit
def lif_single_step_fwd_kernel(
    x_ptr,
    spike_ptr,
    v_init_ptr,
    v_threshold_ptr,
    v_reset_ptr,
    c_m_ptr,
    tau_ptr,
    v_next_ptr,
    numel,
    dt,
    BLOCK_SIZE: tl.constexpr,
    HARD_RESET: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    v = tl.load(v_init_ptr + offs, mask=mask, other=0.0)
    v_threshold = tl.load(v_threshold_ptr + offs, mask=mask, other=0.0)
    v_reset = tl.load(v_reset_ptr + offs, mask=mask, other=0.0)
    c_m = tl.load(c_m_ptr + offs, mask=mask, other=1.0)
    tau = tl.load(tau_ptr + offs, mask=mask, other=1.0)

    dv = -(v - v_reset) / tau + x / c_m
    v = v + dt * dv

    spike = tl.where(v >= v_threshold, 1.0, 0.0).to(v.dtype)
    if HARD_RESET:
        v = v - (v - v_reset) * spike
    else:
        v = v - (v_threshold - v_reset) * spike

    tl.store(spike_ptr + offs, spike, mask=mask)
    tl.store(v_next_ptr + offs, v, mask=mask)


@triton.jit
def lif_multistep_fwd_kernel(
    x_ptr,
    spike_ptr,
    v_init_ptr,
    refractory_init_ptr,
    v_threshold_ptr,
    v_reset_ptr,
    c_m_ptr,
    tau_ptr,
    tau_ref_ptr,
    v_final_ptr,
    refractory_final_ptr,
    u_pre_spike_ptr,
    stride_xt,
    stride_xn,
    stride_st,
    stride_sn,
    stride_ut,
    stride_un,
    numel,
    dt,
    BLOCK_SIZE: tl.constexpr,
    T: tl.constexpr,
    HARD_RESET: tl.constexpr,
    USE_REFRACTORY: tl.constexpr,
    SAVE_U_PRE_SPIKE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    v = tl.load(v_init_ptr + offs, mask=mask, other=0.0)
    v_threshold = tl.load(v_threshold_ptr + offs, mask=mask, other=0.0)
    v_reset = tl.load(v_reset_ptr + offs, mask=mask, other=0.0)
    c_m = tl.load(c_m_ptr + offs, mask=mask, other=1.0)
    tau = tl.load(tau_ptr + offs, mask=mask, other=1.0)
    #不应期
    if USE_REFRACTORY:
        refractory = tl.load(refractory_init_ptr + offs, mask=mask, other=0.0)
        tau_ref = tl.load(tau_ref_ptr + offs, mask=mask, other=0.0)

    for t in tl.static_range(0, T):
        x = tl.load(x_ptr + t * stride_xt + offs * stride_xn, mask=mask, other=0.0)

        dv = -(v - v_reset) / tau + x / c_m
        v = v + dt * dv
        #存储
        if SAVE_U_PRE_SPIKE:
            tl.store(
                u_pre_spike_ptr + t * stride_ut + offs * stride_un,
                v,
                mask=mask,
            )

        spike = tl.where(v >= v_threshold, 1.0, 0.0).to(v.dtype)
        if USE_REFRACTORY:
            active = (refractory == 0).to(v.dtype)
            spike = spike * active

        if HARD_RESET:
            v = v - (v - v_reset) * spike
        else:
            v = v - (v_threshold - v_reset) * spike

        if USE_REFRACTORY:
            refractory = tl.maximum(refractory + spike * tau_ref - dt, 0.0)

        tl.store(spike_ptr + t * stride_st + offs * stride_sn, spike, mask=mask)

    tl.store(v_final_ptr + offs, v, mask=mask)
    if USE_REFRACTORY:
        tl.store(refractory_final_ptr + offs, refractory, mask=mask)


@triton.jit
def lif_multistep_soft_noref_bwd_kernel(
    grad_spike_ptr,
    grad_v_final_ptr,
    u_pre_spike_ptr,
    v_threshold_ptr,
    v_reset_ptr,
    c_m_ptr,
    tau_ptr,
    grad_x_ptr,
    stride_gst,
    stride_gsn,
    stride_gxt,
    stride_gxn,
    stride_ut,
    stride_un,
    numel,
    dt,
    BLOCK_SIZE: tl.constexpr,
    T: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    v_threshold = tl.load(v_threshold_ptr + offs, mask=mask, other=0.0)
    v_reset = tl.load(v_reset_ptr + offs, mask=mask, other=0.0)
    c_m = tl.load(c_m_ptr + offs, mask=mask, other=1.0)
    tau = tl.load(tau_ptr + offs, mask=mask, other=1.0)

    v_scale = v_threshold - v_reset
    one = tl.full([BLOCK_SIZE], 1.0, tl.float32).to(v_threshold.dtype)
    grad_v = tl.load(grad_v_final_ptr + offs, mask=mask, other=0.0)

    for t in tl.static_range(T - 1, -1, -1):
        grad_spike = tl.load(
            grad_spike_ptr + t * stride_gst + offs * stride_gsn,
            mask=mask,
            other=0.0,
        )
        u_pre_spike = tl.load(
            u_pre_spike_ptr + t * stride_ut + offs * stride_un,
            mask=mask,
            other=0.0,
        )

        h = (u_pre_spike - v_threshold) / v_scale
        sigma = tl.sigmoid(h)
        surrogate_derivative = sigma * (one - sigma)
        ds_du = surrogate_derivative / v_scale

        grad_u = grad_spike * ds_du + grad_v * (one - surrogate_derivative)
        tl.store(
            grad_x_ptr + t * stride_gxt + offs * stride_gxn,
            grad_u * (dt / c_m),
            mask=mask,
        )
        grad_v = grad_u * (one - dt / tau)
