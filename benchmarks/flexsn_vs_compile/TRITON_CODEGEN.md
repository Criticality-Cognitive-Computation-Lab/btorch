# Triton codegen: torch.compile vs FlexSN vs hand-written spikingjelly

The same multistep spiking-neuron workload, as written by four different
compilers/authors, so the codegen can be compared directly.

Generate the snapshots with:

```bash
python benchmarks/flexsn_vs_compile/dump_triton.py --T 4 --N 32768
```

Raw copy-paste files live in [`codegen/`](codegen/):

| file | source |
|------|--------|
| `neuron_python.py` | reference neuron (torch) |
| `inductor_inference_triton.py` / `inductor_training_triton.py` | `torch.compile(eager_loop)` |
| `inductor_save_history_triton.py` | `torch.compile(eager_loop_save)` |
| `flexsn_inference_triton.py` / `flexsn_training_triton.py` | FlexSN `backend="triton"` |

Snapshot: `T=4`, `N=32768`, fp32, RTX 5090 (sm_120), torch 2.11.0, triton 3.6.0.

---

# 1. The reference neuron (stripped down)

`codegen/neuron_python.py`; per-step dynamics, called in a Python time loop:

```python
def core(x, y, v, rho):
    h = 0.9 * v + x
    s1 = sg(h - (rho + 1.0))        # adaptive-threshold spike
    s2 = sg(h - 1.0)                # fixed-threshold spike
    rho = 0.8 * rho + s1            # threshold adaptation
    yy = torch.sigmoid(y)           # modulation
    v = (h * (1.0 - s1)) * yy + (h - s2) * (1.0 - yy)
    return s1, s2, v, rho

def eager_loop(x_seq, y_seq, v0, rho0):
    v, rho = v0, rho0
    s1_l, s2_l = [], []
    for t in range(T):
        s1, s2, v, rho = core(x_seq[t], y_seq[t], v, rho)
        s1_l.append(s1); s2_l.append(s2)
    return torch.stack(s1_l), torch.stack(s2_l), v, rho
```

`sg` is the straight-through ATan surrogate (Heaviside forward, `0.5 + atan(pi x)/pi`
backward). The state `(v, rho)` is carried; the spikes are stacked into `[T,N]`.

---

# 2. FlexSN generated Triton

FlexSN traces **one step** with `make_fx`, emits it as a `@triton.jit` core, and
wraps it in a `tl.static_range(T)` scan. The whole pass is **one kernel**
(`codegen/flexsn_inference_triton.py:17-165`):

```python
@triton.jit
def flexsn_core_inductor_scan_<h>(x_1, y_1, v_1, rho_1):   # the traced single step
    mul = v_1 * 0.9
    add = mul + x_1
    add_1 = rho_1 + 1.0
    sub = add - add_1
    _to_copy = (sub >= 0).to(tl.float32)
    sub_1 = add - 1.0
    _to_copy_1 = (sub_1 >= 0).to(tl.float32)
    add_2 = (rho_1 * 0.8) + _to_copy
    mul_2 = add * (1.0 - _to_copy)
    sub_2 = add - _to_copy_1
    sigmoid = tl.sigmoid(y_1.to(tl.float32)).to(y_1.dtype)
    add_3 = (mul_2 * sigmoid) + (sub_2 * (1.0 - sigmoid))
    return _to_copy, _to_copy_1, add_3, add_2

@triton.autotune(configs=[triton.Config({"BLOCK_NCL": f*w*32}, num_warps=w)
                          for f in [1, 2] for w in [2, 4]],
                 key=["T", "dtype"], restore_value=[...])
@triton.jit
def flexsn_inference_kernel_<h>(x0_seq_ptr, x1_seq_ptr, v0_init_ptr, v1_init_ptr,
                                s0_seq_ptr, s1_seq_ptr, v0_seq_ptr, v1_seq_ptr,
                                T: tl.constexpr, NCL: tl.constexpr,
                                BLOCK_NCL: tl.constexpr, dtype: tl.constexpr):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL
    v0 = tl.load(<block_ptr v0_init>, boundary_check=(1,), padding_option="zero")
    v1 = tl.load(<block_ptr v1_init>, boundary_check=(1,), padding_option="zero")

    for t in tl.static_range(0, T, 1):              # T: constexpr -> Triton unrolls
        x0 = tl.load(<block_ptr x0_seq at t>, boundary_check=(1,), padding_option="zero")
        x1 = tl.load(<block_ptr x1_seq at t>, boundary_check=(1,), padding_option="zero")
        s0, s1, v0, v1 = flexsn_core_inductor_scan_<h>(x0, x1, v0, v1)
        convert_and_store(<block_ptr s0_seq at t>, s0, boundary_check=(1,))   # outputs
        convert_and_store(<block_ptr s1_seq at t>, s1, boundary_check=(1,))   # inline
        convert_and_store(<block_ptr v0_seq at t>, v0, boundary_check=(1,))
        convert_and_store(<block_ptr v1_seq at t>, v1, boundary_check=(1,))
```

Key traits: state `v0/v1` are SSA values carried across the unrolled loop
(registers); every access is a `tl.make_block_ptr` with `boundary_check` mask;
`tl.static_range` means Triton unrolls `T`; no explicit `tl.fma` (unlike the
hand-written backward). The training version adds a forward that checkpoints
`(rho_prev, h, sigmoid)` and one reverse `tl.static_range(T-1,-1,-1)` backward
kernel.

---

# 3. torch.compile (inductor) generated Triton

Dynamo **unrolls the Python loop at trace time**; there is no `tl.static_range`
anywhere. Inductor then splits by iteration space into two kernels
(`codegen/inductor_inference_triton.py`), launched back-to-back:

```python
# call(): two launches, grid 32768 then 131072  (lines 892, 898)
k0.run(v0, x_seq, rho0, y_seq, buf0..buf7, 32768, stream=stream0)
k1.run(v0, x_seq, rho0, buf1, buf2, buf3, buf5, buf0, buf4, out_s1, out_s2, 131072, ...)
```

Kernel 0, `xnumel = N = 32768` — the register-resident recurrence (real head shown):

```python
@triton.jit
def triton_poi_fused_..._sigmoid_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, *out, xnumel, XBLOCK):
    xnumel = 32768
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]          # no mask: N is aligned
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)                 # v0
    tmp3 = tl.load(in_ptr1 + (x0), None)                 # x[0]
    tmp5 = tl.load(in_ptr2 + (x0), None)                 # rho0
    tmp23 = tl.load(in_ptr3 + (x0), None)                # y[0]
    tmp40 = tl.load(in_ptr1 + (32768 + x0), None)        # x[1]  (T dim = const offset)
    tmp59 = tl.load(in_ptr3 + (32768 + x0), None)        # y[1]
    tmp76 = tl.load(in_ptr1 + (65536 + x0), None)        # x[2]
    tmp107 = tl.load(in_ptr1 + (98304 + x0), None)       # x[3]
    ...
    # tmp1..tmp140: four copies of the neuron body, v/rho as SSA registers
    tl.store(out_ptr0 + (x0), tmp38, None)               # per-step scalars
    ...
    tl.store(out_ptr7 + (x0), tmp140, None)              # final v (and rho)
```

Kernel 1, `xnumel = T·N = 131072` — output materialization (real head shown):

```python
@triton.jit
def triton_poi_fused_..._stack_sub_1(in_ptr0, ..., in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK):
    xnumel = 131072
    x0 = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:]
    tmp2 = x0 >= 0;  tmp4 = x0 < 32768                  # which timestep am I?
    tmp5 = tl.load(in_ptr0 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    ...
    # recompute that step's spike from k0's saved scalars, then stack-store s1, s2
```

Key traits: no loop construct (T baked into straight-line `tmpNN`); flat pointers
with `tt.divisibility 16` and a dead `xmask` (maximally vectorizable); `stack`
becomes a separate `T·N`-wide kernel; `eviction_policy='evict_last'` hints.

---

# 4. spikingjelly hand-written Triton (different neuron: LIF)

`triton_kernel/neuron_kernel/lif.py:38-154` — a *different* neuron (plain LIF),
hand-written rather than generated, but structurally the FlexSN pattern plus
compile-time flags and a precision plan:

```python
@triton.autotune(configs=[triton.Config({"BLOCK_NCL": f*w*32}, num_warps=w)
                          for f in [1, 2] for w in [4, 8]],
                 key=["T", "NCL", "compute_dtype", "soft_reset",
                      "save_intermediates", "store_v_seq"],
                 restore_value=["s_seq_ptr", "h_seq_ptr", "v_seq_ptr"])
@triton.jit
def _multistep_lif_forward_kernel_static(
        x_seq_ptr, v_init_ptr, s_seq_ptr, h_seq_ptr, v_seq_ptr,
        tau, v_threshold, v_reset,
        T: tl.constexpr, NCL: tl.constexpr, BLOCK_NCL: tl.constexpr,
        compute_dtype: tl.constexpr, decay_input: tl.constexpr,
        soft_reset: tl.constexpr, save_intermediates: tl.constexpr,
        store_v_seq: tl.constexpr):
    ncl_offset = tl.program_id(0) * BLOCK_NCL
    v_threshold = tl.full([1], v_threshold, dtype=compute_dtype)
    v_reset = tl.full([1], v_reset, dtype=compute_dtype)
    r_tau = tl.full([1], 1.0 / tau, dtype=compute_dtype)
    v = tl.load(<block_ptr v_init>, boundary_check=(1,), padding_option="zero").to(compute_dtype)

    for t in tl.static_range(0, T, 1):
        x = tl.load(<block_ptr x_seq at t>, boundary_check=(1,), padding_option="zero").to(compute_dtype)
        if decay_input:
            h = v + r_tau * (v_reset - v + x)
        else:
            h = v + r_tau * (v_reset - v) + x
        s = tl.where(h >= v_threshold, 1.0, 0.0).to(compute_dtype)
        v = h - s * v_threshold if soft_reset else s * v_reset + (1.0 - s) * h
        convert_and_store(<block_ptr s_seq at t>, s, boundary_check=(1,))
        if store_v_seq:                       # tl.constexpr -> dead-code-eliminated
            convert_and_store(<block_ptr v_seq at t>, v, boundary_check=(1,))
        if save_intermediates:                # checkpoint h only if backward needs it
            convert_and_store(<block_ptr h_seq at t>, h, boundary_check=(1,))
    if not store_v_seq:
        convert_and_store(<block_ptr v_seq>, v, boundary_check=(1,))   # only final v
```

Key traits: same `tl.static_range` scan template and `block_ptr`+`boundary_check`
as FlexSN, plus `compute_dtype` (mixed precision), `save_intermediates` /
`store_v_seq` constexpr flags, `tl.full` constants, and (in the backward)
`tl.fma`. Above `triton_neuron_kernel_static_range_max_T = 64` a twin `_dynamic`
kernel with `tl.range` is selected to cap Triton compile cost.

---

# 5. The four at a glance

| | reference Python | FlexSN (generated) | torch.compile (generated) | spikingjelly (hand-written) |
|---|---|---|---|---|
| multistep structure | Python loop | `tl.static_range(T)` in-kernel | Dynamo unroll at trace time | `tl.static_range(T)` in-kernel |
| kernels, inference | n/a (eager: T launches) | **1** | **2** (N + T·N) | **1** |
| residual state | Python locals | registers (SSA) | registers within k0; buffered between k0/k1 | registers (SSA) |
| spike output store | `torch.stack` -> eager kernels | inline in the scan | separate T·N recompute kernel | inline in the scan |
| addressing | — | `make_block_ptr` + `boundary_check` (masked) | flat ptr + `tt.divisibility 16` (unmasked) | `make_block_ptr` + `boundary_check` (masked) |
| precision | fp32 | `dtype` only | fp32 (graph dtype) | `compute_dtype` plan (fp16/bf16/fp8) |
| unroll control | — | always static | always unrolled by Dynamo | static≤64 / `tl.range`>64 |
| backward | autograd | 1 AOT-generated reverse scan | AOTAutograd, rematerializes | `tl.fma`, explicit reverse scan |
| spike storage (bwd) | fp32 | fp32 | fp32 | optional bool / 1-bit packed |

Note the hand-written file is a *plain LIF* (different dynamics), included to show
the hand-written codegen style, not a numerical match to the reference neuron.

---

# 6. What FlexSN / spikingjelly do that torch.compile does not

| optimization | who | real? |
|---|---|---|
| one fused scan kernel, outputs stored inline | FlexSN, SJ | real but bounded; shrinks as N dominates |
| explicit storage-vs-compute `compute_dtype` plan, fp8/bf16 storage | SJ | real for mixed precision |
| static↔dynamic unroll switch at `T=64` | SJ | real compile-cost/robustness win; **FlexSN lacks it** |
| bool / 1-bit packed spike storage for backward | SJ | real memory win (8×/32×) |
| `save_intermediates` / `store_v_seq` constexpr flags | SJ | specialization/convenience |
| per-channel-vs-scalar threshold specialization | SJ | real for per-channel params |
| explicit half2 intrinsics (`__hfma2`, `__hgeu2`) | SJ CUDA | redundant on Triton (auto `fma.rn.f16x2`) |
| explicit `tl.fma` | SJ | cosmetic (LLVM/`enable_fp_fusion`) |
| `__shared__` reduction (PLIF `tau` grad) | SJ CUDA | real, but only non-pointwise op |

And the converse — things inductor does that the others do not:

- **iteration-space split + rematerialization** (O(N) backward memory vs FlexSN's
  O(T·N) residual checkpointing);
- **mask-free vectorized addressing** (`tt.divisibility 16`, dead `xmask`) —
  FlexSN/SJ's `boundary_check` makes every load/store predicated;
- **constant folding** (`1/pi` precomputed) and `eviction_policy` hints;
- **AOTAutograd generality** (arbitrary graphs, not just scan neurons).

Two cautions when reading the numbers:

1. **"Inductor can't fuse the scan" is false.** At `T=16` inductor emits **one**
   fused forward kernel; at `T=32` it fragments (~28 kernels), `T=64` ~97. The
   2-vs-1 split is the scheduler's iteration-space heuristic, not a hard limit.
   No exposed flag (`aggressive_fusion`, `max_fusion_size`, `max-autotune`,
   `dynamic=True`) changes it at `T=4`.
2. **Single-kernel fusion is not the real axis.** The differences that scale are
   the rematerialize-vs-checkpoint training strategy (inductor O(N) memory,
   FlexSN O(T·N)) and the unroll-vs-`tl.range` compile-time wall (SJ's switch).

---

# 7. Does Triton pack half2 for us?

Yes. Measured on this box (triton 3.6, sm_120), a plain fp16
`c = a * b + 1.0` elementwise kernel compiles to

```
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];   // 128-bit = 8x fp16
fma.rn.f16x2 ...                                          // x4, packed half2
st.global.v4.b32 [ %rd3 + 0 ], { %r9, %r10, %r11, %r12 };
```

with zero scalar `mul.f16`/`add.f16`. So Triton autovectorizes memory to 128-bit
and LLVM's NVPTX backend fuses the fp16 arithmetic into `fma.rn.f16x2` on its own
— the same instruction the hand-written CUDA path reaches via `__hfma2`/`__hgeu2`.
Caveats: packing needs contiguity/alignment and pair-able ops (masked/strided
access or fp32-accumulate paths fall back toward scalar f16), and the
`compute_dtype` trick (load f16 → fp32 → math → store f16) deliberately prevents
half2 compute when fp32 accumulation is wanted.

---

# 8. Regenerating

```bash
python benchmarks/flexsn_vs_compile/dump_triton.py --T 4 --N 32768
```

writes `inf_*`, `train_*`, `save_*` (inductor) and `flexsn_{inf,train}_*` into
the resolved figure directory; the `codegen/` copies are trimmed snapshots of
those with a `# ruff: noqa` header.

---

# Appendix A — spikingjelly's hand-written kernels, and why CuPy wins

Paths are relative to the spikingjelly checkout
`/home/fanqixuan/src/spikingjelly/spikingjelly/activation_based/` (written `sj/`
below) and to this benchmark dir (written `BT/`). Every claim carries a
`file:line` reference.

## A.1 Catalogue

There are **no `.cu`/`.cuh` files** — all CUDA is embedded as Python strings
compiled at runtime via CuPy `RawKernel`/`RawModule` (NVRTC) or `load_inline`.

| area | files (`sj/…`) | kernels |
|---|---|---|
| bit-pack | `cuda_kernel/tensor_cache.py` | 4: `float2bool`, `half2bool`, `bool2float`, `bool2half` (`:17-89`) |
| spike GEMM | `cuda_kernel/spike_linear.py` | 4: pack (`:42-59`), bit-packed tiled dense GEMM (`:78-180`), row-index sparse (`:185-209`), sparse-Wᵀ (`:210-231`); `--use_fast_math` (`:253`) |
| neurons (CUDA) | `cuda_kernel/neuron_kernel/multi_step/{lif,integrate_and_fire,plif,izhikevich,qif,eif}.py`, `single_step/{lif,integrate_and_fire}.py` | hand-written fwd+bwd, fp32 and fp16 **half2** |
| CUDA codegen | `cuda_kernel/auto_cuda/{base,generator,cfunction}.py`, `cuda_kernel/neuron_kernel/cuda_code.py` | emit `__global__` per subclass |
| host C++ | `cuda_kernel/spike_op.py` | 0 kernels (cuDNN conv wrapper) |
| neurons (Triton) | `triton_kernel/neuron_kernel/{lif,integrate_and_fire,plif,ilif,stbif,activation_aware_if}.py` | static+dynamic pairs |
| Triton misc | `triton_kernel/{surrogate_kernel,compress,fp8_capability,triton_utils}.py` | surrogate dispatch (`surrogate_kernel.py:37-95`), bit-spike pack/unpack (`compress.py:71-111`), FP8 probes (`fp8_capability.py:42-98`) |
| TileLang | `tilelang_kernel/` | **sources absent — only `.pyc`** (~24 inferred kernels) |

## A.2 What a generic Triton compiler would not do automatically

1. **Guaranteed SIMD width + alignment contract.** The CUDA path is written in
   half2: `__hfma2`/`__hmul2`/`__hsub2`/`__hadd2`/`__hgeu2`/`h2exp`
   (`sj/cuda_kernel/neuron_kernel/multi_step/qif.py:92-119,205-264`,
   `…/eif.py:92-119,206-265`), fire/reset in `…/neuron_kernel/cuda_code.py:9-25`,
   expression snippets in `…/auto_cuda/cfunction.py:54,81,218,228`. It also
   enforces an even-N / alignment contract (`…/auto_cuda/base.py:692-746,785-822`)
   with runtime padding (`…/cuda_kernel/neuron_kernel/multi_step/base.py:424-426`).
   Triton only does this best-effort.
2. **Global fast-math.** `-use_fast_math` is injected into every NVRTC compile
   (`sj/configure.py:30,40`), consumed at `sj/cuda_kernel/auto_cuda/base.py:472`
   and `sj/cuda_kernel/tensor_cache.py:172,266`; `--use_fast_math` for the GEMM
   RawModule (`sj/cuda_kernel/spike_linear.py:253`). Triton has no global
   precision knob.
3. **Shared-memory tiled bit-packed GEMM** with register accumulators and
   `__syncthreads`, extracting bits *inside* the inner product:
   `sj/cuda_kernel/spike_linear.py:78-84` (tile consts), `:92-93` (`__shared__`),
   `:102-107` (`float acc[TM][TN]`), `:139,167` (`__syncthreads`),
   `:141-165` (bit-extract, `(s_bits[i] >> b) & 1` at `:162`). No compiler
   invents a bit-serial smem GEMM.
4. **Block reduction + `atomicAdd`** for PLIF's scalar `decay` gradient
   (`sj/cuda_kernel/neuron_kernel/multi_step/plif.py:114` `__shared__ sdata`,
   `:219-233` stride-halving reduce + `atomicAdd`, power-of-two contract `:244-250`).
   The Triton path can't and instead exports per-lane partials and sums on host
   (`sj/triton_kernel/neuron_kernel/plif.py:276,372`, comment `:383-384`, sum `:1237`).
5. **Bit-packing as the data format** (8 spikes/byte), in memory and in math:
   `sj/cuda_kernel/tensor_cache.py:26-31,46-51,63-69,83-89`,
   `sj/triton_kernel/compress.py:71-111`, config `sj/configure.py:54-68`.
6. **Sparse, data-dependent execution** with a fixed-capacity workspace to avoid
   host sync and a transposed-weight layout for coalescing:
   `sj/cuda_kernel/spike_linear.py:184-231` (two-kernel split, block `(1,)`),
   workspace `:510-511`, `W_T[k*N+n]` `:528`.
7. **Hand-derived analytical BPTT fused into one reverse kernel** instead of
   autograd recompute: `sj/cuda_kernel/neuron_kernel/multi_step/base.py:137-239`
   (`NeuronBPTTKernel`, `grad_h`/`grad_s_to_h`) and `:292-409` (register-carried
   reverse loop); Triton equivalent `sj/triton_kernel/neuron_kernel/lif.py:333-411`;
   surrogate inlined `sj/triton_kernel/surrogate_kernel.py:37-95`.
8. **Temporal layout trick (CUDA only).** The state *sequence* lives in a `T+1`
   buffer with `v_init` in row 0, so one array holds `v[t-1]` at `v_v_seq[t]`
   and `v[t]` at `v_v_seq[t+dt]` — no per-step shift/clone, only a one-row
   `v_v_seq[0].copy_(v_init)` at setup
   (`sj/cuda_kernel/neuron_kernel/multi_step/lif.py:180-181`; index convention
   `sj/cuda_kernel/neuron_kernel/multi_step/base.py:67,115`; the backward
   recovers the shifted view `sj/cuda_kernel/neuron_kernel/multi_step/plif.py:353-361`,
   falling back to `torch.cat` `:380`). **Triton / FlexSN / inductor do NOT do
   this** — they carry `v` in a register overwritten each step
   (`sj/triton_kernel/neuron_kernel/lif.py:88-113`), so neither `v[t-1]` nor
   `v[t]` is copied at all.
9. **Thread-coarsened time loop:** one thread owns a whole neuron's time series
   (`sj/cuda_kernel/auto_cuda/base.py:1176-1197`).
10. **Hand-authored unroll/precision policy:** static-vs-dynamic unroll switch at
    `T=64` (`sj/configure.py:70-85`, `sj/triton_kernel/triton_utils.py:460-464`,
    kernel pairs `sj/triton_kernel/neuron_kernel/lif.py:55,173`) and FP8
    storage/compute decoupling (`sj/triton_kernel/neuron_kernel/utils.py:155-228`).

**Absent repo-wide** (verified): `__ldg`, warp shuffles/`__ballot_sync`/`__popc`,
`__launch_bounds__`, `cp.async`/`__pipeline`, PDL/`griddepcontrol`, cooperative
groups/`grid.sync`, persistent/grid-stride kernels, and `float4` (the CUDA path
vectorizes only at the half2 level).

## A.3 Why the benchmark's CuPy kernel is fast

All refs are `BT/kernels.py` unless noted; Triton/inductor rivals are
`BT/codegen/*`.

1. **Analytical hand-derived BPTT (dominant in training).** The backward replays
   the recurrence from three saved residuals and recomputes the adjoints in
   ~20 FMAs/step: `kernels.py:293` (`surrogate_grad`), `:357-374`
   (`d_membrane`, `gv`, `grho` updates), residuals saved at `:335-339` and
   written at `:258-260`. Contrast inductor's AOT-expanded graph
   (`BT/codegen/inductor_training_triton.py:967-1089`, 14 `neg`, `mul_28…mul_81`)
   which saves 8 buffers (`:882-889`). ⇒ ~2–3× fewer backward FLOPs.
2. **One forward + one backward kernel, outputs stored inline (dominant short-T).**
   Spikes/intermediates written in-kernel: `kernels.py:254-278`. Inductor emits a
   second T·N "stack" forward (`BT/codegen/inductor_inference_triton.py:694-857`,
   `xnumel=131072`); FlexSN also writes inline
   (`BT/codegen/flexsn_inference_triton.py:120-165`).
3. **Guaranteed float4 (128-bit) transactions.** `_VEC = 4` (`kernels.py:178`),
   explicit `as_float4` helper (`kernels.py:385,391-396`), every load/store
   `kernels.py:215-216,227-228,256-260,277-278,321-322,335-339`. Triton's
   `make_block_ptr`+`boundary_check` is predicated/best-effort
   (`BT/codegen/flexsn_inference_triton.py:76-78`); inductor *does* hint
   `tt.divisibility 16` (`BT/codegen/inductor_inference_triton.py:329`).
4. **Register-resident recurrence + thread coarsening** (4 neurons/thread):
   `kernels.py:211-212` (`float v[vec], rho[vec]`, registers), `_VEC=4` `:178`.
   Triton's autotune searches `BLOCK_NCL ∈ {64,128,256}`
   (`BT/codegen/flexsn_inference_triton.py:41-48`).
5. **No hot-path masking.** One thread-exit and a scalar tail only:
   `kernels.py:205` (`if (base >= numel) return`), `:208-209`
   (`aligned`/`tail` scalar fallback). FlexSN masks every access.
6. **Fast intrinsics / flags.** `__expf` sigmoid `kernels.py:243`; `#pragma unroll`
   `:236,350`; `__restrict__` on all pointers `:192-201,298-308`; neuron constants
   as `constexpr` immediates `kernels.py:385`.
7. **Launch/compile overhead** — small at N=32768 (≈2–5% of a ~50–100 µs kernel);
   only dominant in the launch-bound regime (see `README.md`, PDL study).
   CuPy nvrtc one-time compile ~0.4 s vs FlexSN ~1.7–1.9 s (`README.md:61-64`).
8. **Occupancy/launch config** — negligible; Triton's autotune is at least as
   thorough. `_THREADS_PER_BLOCK=256` (`kernels.py:179`).

## A.4 Caveats

- `tilelang_kernel/` has no sources in the checkout (only `.pyc`), so its kernels
  are inferred and not quoted.
- Correspondence table above is for *this* fp32 neuron; SJ's `compute_dtype`,
  FP8, and bit-packed-spike wins apply to mixed-precision / network-scale runs.

