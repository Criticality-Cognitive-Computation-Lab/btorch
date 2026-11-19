"""Fixed-weight FP32 conductance/Izhikevich chunks with existing Triton push.

The reference keeps g(E-v) before neuron mutation and adds emitted events only
at the end of that frame. Physical module states remain complete at every
chunk boundary; there is no hidden pending conductance increment.
"""

from __future__ import annotations

import math
import time

import torch

from .. import environ
from ..ode import _exp_euler_coefficient
from .adapters import connection_scope
from .options import KernelOptions


def _tensor_signature(t):
    if t is None:
        return None
    return (id(t), t.data_ptr(), t._version, tuple(t.shape), t.dtype, t.device)


def _signature(module):
    neuron, synapse = module.neuron, module.synapse
    connection = synapse.linear
    parameters = [
        getattr(neuron, k)
        for k in (
            "c_m",
            "k",
            "v_rest",
            "v_threshold",
            "v_peak",
            "a",
            "b",
            "d",
            "v_reset",
        )
    ]
    parameters += [
        synapse.base_psc.tau_syn,
        synapse.base_psc.E,
        connection.weight.value,
    ]
    states = [neuron.v, neuron.u, synapse.base_psc.psc, synapse.psc]
    return (
        float(environ.get("dt")),
        connection.topology_version,
        connection.routing_version,
        tuple(_tensor_signature(t) for t in parameters),
        tuple((id(t), t.data_ptr(), tuple(t.shape), t.dtype, t.device) for t in states),
    )


def _indices(value, *, n, device, groups=False):
    if value is None:
        return (
            torch.zeros(n, dtype=torch.int32, device=device)
            if groups
            else torch.empty(0, dtype=torch.int32, device=device)
        )
    x = torch.as_tensor(value, device=device)
    if x.ndim != 1 or x.dtype not in (torch.int32, torch.int64):
        raise ValueError("Recording IDs must be an integer vector")
    if groups:
        if len(x) != n or bool((x < 0).any()):
            raise ValueError(
                "group_index must label every neuron with a nonnegative integer"
            )
    elif len(x) and (
        bool((x < 0).any()) or bool((x >= n).any()) or len(x.unique()) != len(x)
    ):
        raise ValueError("probe_ids must be unique valid neuron IDs")
    return x.to(torch.int32).contiguous()


def run_chunk(
    module,
    inputs,
    *,
    hold_steps,
    probe_ids,
    group_index,
    current_stride,
    cudagraph,
    check,
):
    if torch.is_grad_enabled():
        raise RuntimeError(
            "Fused conductance execution is inference-only; disable gradients"
        )
    if (
        inputs.device.type != "cuda"
        or inputs.dtype != torch.float32
        or not inputs.is_contiguous()
    ):
        raise ValueError("Expected contiguous FP32 CUDA input")
    n = module.neuron.v.numel()
    if (
        module.neuron.v.shape != (1, n)
        or inputs.ndim not in (2, 3)
        or inputs.shape[-1] != n
        or (inputs.ndim == 3 and inputs.shape[1] != 1)
    ):
        raise ValueError(
            "Expected one network batch and input [frames,1,N] or [frames,N]"
        )
    if type(hold_steps) is not int or hold_steps < 1 or len(inputs) < 1:
        raise ValueError(
            "Positive integer hold_steps and at least one input frame required"
        )
    steps = len(inputs) * hold_steps
    if (
        type(current_stride) is not int
        or current_stride < 0
        or (current_stride and steps % current_stride)
    ):
        raise ValueError(
            "current_stride must be zero or a positive divisor of chunk steps"
        )
    if (
        module.neuron.v.dtype != torch.float32
        or module.neuron.u.dtype != torch.float32
        or module.synapse.base_psc.psc.dtype != torch.float32
    ):
        raise ValueError("Fused physical states must be FP32")
    if module.neuron.v.device != inputs.device:
        raise ValueError("Input and physical states must share a CUDA device")
    # IDs are normally reused tensors. Their versions avoid repeated device
    # validation and CPU synchronization after the first preparation.
    key = (
        len(inputs),
        hold_steps,
        current_stride,
        bool(cudagraph),
        _tensor_signature(probe_ids)
        if isinstance(probe_ids, torch.Tensor)
        else repr(probe_ids),
        _tensor_signature(group_index)
        if isinstance(group_index, torch.Tensor)
        else repr(group_index),
    )
    signature = _signature(module)
    runtime = module._fused_chunks.get(key)
    if runtime is None or runtime.signature != signature:
        probes = _indices(probe_ids, n=n, device=inputs.device)
        groups = _indices(group_index, n=n, device=inputs.device, groups=True)
        ng = int(groups.max()) + 1 if group_index is not None else 0
        if ng > 32:
            raise ValueError("At most32 recording groups are supported")
        runtime = _Chunk(
            module,
            inputs,
            hold_steps,
            probes,
            groups,
            ng,
            current_stride,
            bool(cudagraph),
        )
        if key not in module._fused_chunks and len(module._fused_chunks) >= 4:
            # Bounded layouts keep varied interactive durations from retaining
            # unbounded graphs and recording buffers on the device.
            module._fused_chunks.pop(next(iter(module._fused_chunks)))
        module._fused_chunks[key] = runtime
    return runtime(inputs, check)


class _Chunk:
    def __init__(self, module, inputs, hold, probes, groups, ng, stride, graph):
        import triton

        from ...sparse.runtime.kernels_triton_spikelist import SPIKE_BLOCK
        from ...sparse.runtime.ops import prepare_bound_forward

        started = time.perf_counter()
        self.module, self.hold = module, hold
        self.n, self.steps, self.r = (
            module.neuron.v.numel(),
            len(inputs) * hold,
            module.synapse.n_receptor,
        )
        self.ng, self.p, self.stride = ng, len(probes), stride
        self.static_input = inputs.reshape(len(inputs), 1, self.n).clone()
        self.probes, self.groups = probes, groups
        self.probe_slot = torch.full(
            (self.n,), -1, dtype=torch.int32, device=inputs.device
        )
        self.probe_slot[probes.long()] = torch.arange(
            self.p, dtype=torch.int32, device=inputs.device
        )
        n, t, p, r = self.n, self.steps, self.p, self.r

        def empty(*shape, dtype=torch.float32):
            return torch.empty(shape, device=inputs.device, dtype=dtype)

        self.output = dict(
            spikes=empty(t, 1, n),
            packed_spikes=empty(t, (n + 7) // 8, dtype=torch.uint8),
            pre_v_mv=empty(t, p),
            post_v_mv=empty(t, p),
            post_u=empty(t, p),
            pre_g_ns=empty(t, p, r),
            post_g_ns=empty(t, p, r),
            pre_current_pa=empty(t, p, r),
            external_pa=empty(t, p),
            group_spikes=empty(t, ng, dtype=torch.int32),
            group_current_sum_pa=empty(t, ng, r),
            cell_current_sum_pa=empty(n, r, dtype=torch.float64),
            epsc_pa=empty(t // stride if stride else 0, n),
            ipsc_pa=empty(t // stride if stride else 0, n),
            errors=torch.zeros((), device=inputs.device, dtype=torch.int32),
        )
        self.old_v = empty(n)
        self.charged_v, self.charged_u = empty(n), empty(n)
        self.parameters = [
            getattr(module.neuron, k).reshape(-1).expand(n).contiguous()
            for k in (
                "c_m",
                "k",
                "v_rest",
                "v_threshold",
                "v_peak",
                "a",
                "b",
                "d",
                "v_reset",
            )
        ]
        base = module.synapse.base_psc
        self.tau = base.tau_syn.reshape(-1).expand(n * r).contiguous()
        self.reversal = base.E.reshape(-1).expand(n * r).contiguous()
        self.dt = float(environ.get("dt"))
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Positive finite dt required")
        self.phi = _exp_euler_coefficient(-1.0 / self.tau, self.dt)[0]
        connection = module.synapse.linear
        options = KernelOptions(connection="triton", connection_method="push")
        with connection_scope(options):
            connection._begin_sparse_run(1, torch.float32, stable_addresses=True)
            try:
                cache = connection.cache
                self.prepared = prepare_bound_forward(
                    connection._route_token,
                    (
                        cache.crow,
                        cache.col,
                        connection._run_weight,
                        self.output["spikes"][0],
                        cache.t_crow,
                        cache.t_col,
                        cache.t_perm,
                        connection._max_density,
                        connection._run_task_values,
                    ),
                )
                if (
                    self.prepared is None
                    or self.prepared.spikes is None
                    or self.prepared.prepare_sequence is None
                ):
                    raise RuntimeError(
                        "Triton source-CSR push with a prepared spike list is required"
                    )
                self.route = connection.explain()
            finally:
                connection._end_sparse_run()
        self.block, self.grid = SPIKE_BLOCK, (triton.cdiv(n, SPIKE_BLOCK),)
        self.graph = None
        physical = [module.neuron.v, module.neuron.u, base.psc, module.synapse.psc]
        saved = [x.clone() for x in physical]

        def restore():
            for value, snapshot in zip(physical, saved, strict=True):
                value.copy_(snapshot)

        stream = torch.cuda.Stream(device=inputs.device)
        stream.wait_stream(torch.cuda.current_stream(inputs.device))
        try:
            with torch.cuda.stream(stream):
                self.body()
                restore()
                self.body()
                restore()
            torch.cuda.current_stream(inputs.device).wait_stream(stream)
            if graph:
                self.graph = torch.cuda.CUDAGraph()
                # The process-wide CUDA generator's capture bookkeeping may
                # have been initialized by an earlier inference-mode graph.
                # Keep capture/replay compatible with both public no-grad modes.
                with (
                    torch.inference_mode(),
                    torch.cuda.graph(self.graph, stream=stream),
                ):
                    self.body()
                torch.cuda.current_stream(inputs.device).wait_stream(stream)
                restore()
            torch.cuda.synchronize(inputs.device)
        except BaseException:
            torch.cuda.current_stream(inputs.device).wait_stream(stream)
            restore()
            raise
        self.setup_s = time.perf_counter() - started
        self.signature = _signature(module)
        self.calls = 0

    def body(self):
        cell, finish = _kernels()
        m, o = self.module, self.output
        queue = self.prepared.spikes
        # Bind to the CURRENT warmup/capture/replay stream, never to the
        # stream that happened to allocate the runtime.
        consume = self.prepared.prepare_sequence(o["spikes"])
        o["errors"].zero_()
        o["group_spikes"].zero_()
        o["group_current_sum_pa"].zero_()
        o["cell_current_sum_pa"].zero_()
        for step in range(self.steps):
            cell[self.grid](
                m.neuron.v,
                m.neuron.u,
                m.synapse.base_psc.psc,
                m.synapse.psc,
                self.static_input,
                self.tau,
                self.phi,
                self.reversal,
                *self.parameters,
                self.old_v,
                self.charged_v,
                self.charged_u,
                o["spikes"],
                o["packed_spikes"],
                self.probe_slot,
                self.groups,
                o["pre_v_mv"],
                o["post_v_mv"],
                o["post_u"],
                o["pre_g_ns"],
                o["pre_current_pa"],
                o["external_pa"],
                o["epsc_pa"],
                o["ipsc_pa"],
                o["group_spikes"],
                o["group_current_sum_pa"],
                o["cell_current_sum_pa"],
                o["errors"],
                queue.degree,
                queue.indices,
                queue.cumulative,
                queue.counts,
                queue.output,
                self.n,
                self.r,
                self.p,
                self.ng,
                self.stride,
                self.hold,
                step,
                self.dt,
                self.block,
                enable_fp_fusion=False,
            )
            consume(step)
            finish[self.grid](
                m.synapse.base_psc.psc,
                queue.output,
                self.reversal,
                self.old_v,
                m.synapse.psc,
                self.probe_slot,
                o["post_g_ns"],
                o["errors"],
                self.n,
                self.r,
                self.p,
                step,
                self.block,
                enable_fp_fusion=False,
            )

    def __call__(self, inputs, check):
        self.static_input.copy_(inputs.reshape_as(self.static_input))
        if self.graph is None:
            self.body()
        else:
            with torch.inference_mode():
                self.graph.replay()
        self.calls += 1
        if self.module.neuron.pre_spike_v:
            self.module.neuron.v_pre_spike = self.charged_v.reshape(1, self.n)
            self.module.neuron.u_pre_spike = self.charged_u.reshape(1, self.n)
        if check and int(self.output["errors"]):
            raise FloatingPointError(
                "Invalid V/U/conductance observed within fused chunk"
            )
        return dict(
            self.output,
            metadata=dict(
                cudagraph=self.graph is not None,
                steps=self.steps,
                hold_steps=self.hold,
                setup_s=self.setup_s,
                prepared_calls=self.calls,
                route=self.route,
            ),
        )


_KERNELS = None


def _kernels():
    global _KERNELS, triton, tl, _emit_spike_list
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl

    from ...sparse.runtime.kernels_triton_spikelist import _emit_spike_list

    @triton.jit
    def cell(
        V,
        U,
        G,
        PSC,
        X,
        TAU,
        PHI,
        REV,
        C,
        K,
        VR,
        VT,
        VP,
        A,
        B,
        D,
        VM,
        OLDV,
        CHARGEDV,
        CHARGEDU,
        Z,
        PACKED,
        PSLOT,
        GROUP,
        PV,
        QV,
        QU,
        PG,
        PI,
        PX,
        FE,
        FI,
        GC,
        GI,
        CELL_CURRENT_SUM,
        ERROR,
        DEGREE,
        ACTIVE,
        CUMULATIVE,
        COUNTS,
        ACCUMULATION,
        N: tl.constexpr,
        R: tl.constexpr,
        P: tl.constexpr,
        NG: tl.constexpr,
        STRIDE: tl.constexpr,
        HOLD: tl.constexpr,
        T,
        DT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < N
        v = tl.load(V + i, valid, 0)
        u = tl.load(U + i, valid, 0)
        x = tl.load(X + (T // HOLD) * N + i, valid, 0)
        slot = tl.load(PSLOT + i, valid, -1)
        total = tl.full((BLOCK,), 0.0, tl.float32)
        exc = total
        inh = total
        for r in tl.static_range(R):
            g = tl.load(G + i * R + r, valid, 0)
            rev = tl.load(REV + i * R + r, valid, 0)
            cur = g * (rev - v)
            previous_sum = tl.load(CELL_CURRENT_SUM + i * R + r, valid, 0.0)
            tl.store(
                CELL_CURRENT_SUM + i * R + r,
                previous_sum + cur.to(tl.float64),
                valid,
            )
            total = total + cur
            if r == 0:
                exc = cur
            else:
                inh = inh + cur
            if P > 0:
                tl.store(PG + (T * P + slot) * R + r, g, valid & (slot >= 0))
                tl.store(PI + (T * P + slot) * R + r, cur, valid & (slot >= 0))
            if NG > 0:
                group = tl.load(GROUP + i, valid, 0)
                for j in tl.static_range(NG):
                    value = tl.sum(tl.where(valid & (group == j), cur, 0.0))
                    tl.atomic_add(GI + (T * NG + j) * R + r, value, sem="relaxed")
            tau = tl.load(TAU + i * R + r, valid, 1.0)
            phi = tl.load(PHI + i * R + r, valid, 0.0)
            derivative = tl.div_rn(-g, tau)
            decayed = g + phi * derivative
            tl.store(G + i * R + r, decayed, valid)
            tl.store(ACCUMULATION + i * R + r, 0.0, valid)
        if STRIDE > 0:
            if T % STRIDE == 0:
                tl.store(FE + (T // STRIDE) * N + i, exc, valid)
                tl.store(FI + (T // STRIDE) * N + i, inh, valid)
        c = tl.load(C + i, valid, 1.0)
        k = tl.load(K + i, valid, 0.0)
        vr = tl.load(VR + i, valid, 0.0)
        vt = tl.load(VT + i, valid, 1.0)
        vp = tl.load(VP + i, valid, 1.0)
        a = tl.load(A + i, valid, 0.0)
        b = tl.load(B + i, valid, 0.0)
        d = tl.load(D + i, valid, 0.0)
        vm = tl.load(VM + i, valid, 0.0)
        quadratic = (k * (v - vr)) * (v - vt)
        drive = x + total
        vc = v + DT * tl.div_rn((drive + quadratic) - u, c)
        uc = u + DT * (a * (b * (vc - vr) - u))
        z = (tl.div_rn(vc - vp, vt - vm) >= 0.0).to(tl.float32)
        vn = vc - (vc - vm) * z
        un = uc + d * z
        tl.store(OLDV + i, v, valid)
        tl.store(CHARGEDV + i, vc, valid)
        tl.store(CHARGEDU + i, uc, valid)
        tl.store(V + i, vn, valid)
        tl.store(U + i, un, valid)
        tl.store(Z + T * N + i, z, valid)
        if P > 0:
            tl.store(PV + T * P + slot, v, valid & (slot >= 0))
            tl.store(QV + T * P + slot, vn, valid & (slot >= 0))
            tl.store(QU + T * P + slot, un, valid & (slot >= 0))
            tl.store(PX + T * P + slot, x, valid & (slot >= 0))
        bits = (z.to(tl.int32) << (i % 8)).reshape((BLOCK // 8, 8))
        packed = tl.sum(bits, 1)
        byte = tl.program_id(0) * (BLOCK // 8) + tl.arange(0, BLOCK // 8)
        tl.store(PACKED + T * tl.cdiv(N, 8) + byte, packed, byte < tl.cdiv(N, 8))
        _emit_spike_list(z, i, valid, DEGREE, ACTIVE, CUMULATIVE, COUNTS, N, BLOCK)
        if NG > 0:
            group = tl.load(GROUP + i, valid, 0)
            for j in tl.static_range(NG):
                tl.atomic_add(
                    GC + T * NG + j,
                    tl.sum(tl.where(valid & (group == j), z.to(tl.int32), 0)),
                    sem="relaxed",
                )
        bad = valid & (
            (vn != vn) | (un != un) | (tl.abs(vn) > 500.0) | (tl.abs(un) > 1.0e6)
        )
        if tl.sum(bad.to(tl.int32)) > 0:
            tl.atomic_or(ERROR, 1)

    @triton.jit
    def finish(
        G,
        DRIVE,
        REV,
        OLDV,
        PSC,
        PSLOT,
        POSTG,
        ERROR,
        N: tl.constexpr,
        R: tl.constexpr,
        P: tl.constexpr,
        T,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < N
        v = tl.load(OLDV + i, valid, 0.0)
        slot = tl.load(PSLOT + i, valid, -1)
        current = tl.full((BLOCK,), 0.0, tl.float32)
        bad = tl.full((BLOCK,), False, tl.int1)
        for r in tl.static_range(R):
            g = tl.load(G + i * R + r, valid, 0.0) + tl.load(
                DRIVE + i * R + r, valid, 0.0
            )
            tl.store(G + i * R + r, g, valid)
            rev = tl.load(REV + i * R + r, valid, 0.0)
            current = current + g * (rev - v)
            if P > 0:
                tl.store(POSTG + (T * P + slot) * R + r, g, valid & (slot >= 0))
            bad = bad | (
                valid & ((g != g) | (tl.abs(g) == float("inf")) | (g < -1.0e-6))
            )
        tl.store(PSC + i, current, valid)
        if tl.sum(bad.to(tl.int32)) > 0:
            tl.atomic_or(ERROR, 2)

    _KERNELS = (cell, finish)
    return _KERNELS
