# 002HIP coverage inventory

Source inspection: 2026-10-19; implementation evidence updated 2026-10-20. This is a coverage inventory, not new GPU
qualification. The expanded execution goal is active; new families still require the validation stages described below. Existing measured evidence
lives in [002HIP_PARITY.md](002HIP_PARITY.md).

002HIP reference checkout: `/data1/fanqixuan/src/002HIP`, HEAD
`f64cd8e002af0e4285125e54d275cf7eae6ef0bf`. The checkout has preexisting
modifications; observations below describe the inspected working tree, not an
assertion that HEAD contains identical code. Do not modify that checkout.

## Cell and PSC coverage

002HIP paths in this table are relative to its checkout; btorch paths are
relative to this repository. Native means generated CuPy, Triton and TileLang.

| 002HIP capability/source | btorch coverage | Missing work |
| --- | --- | --- |
| `model/ALIF.py:StochasticALIFNode` | Existing `ALIF` MemoryModule has native adapters/templates on all three backends; scoped primitive/recurrent tests pass | Complete stochastic capture/performance qualification, then qualify a compatible external case. 002HIP uses Euler with implicit dt=1 and stochastic firing; btorch uses configurable-dt exponential Euler. Adaptation/reset ordering and refractory behavior must be mapped explicitly. The advertised CuPy path has commented-out implementation; validate executability before timing. |
| `model/ELIF.py:ELIFNode` | Existing `ELIF` MemoryModule has native adapters/templates; scoped tests pass, long-sequence VJP remains open | Resolve long-sequence Module VJP operation order. The reference omits adaptation, uses a hard-coded 0.1 voltage step and clamps refractory voltage; btorch ELIF inherits adaptation and exponential Euler. Its multistep override also lacks a return. Do not claim an executable matched custom kernel merely from the class name. |
| `model/ALIF.py:PoissonRandomSpike` | Existing Poisson surrogate is lowered natively with explicit uniforms or opaque Torch-owned per-step RNG; forward raster is authoritative for reverse | Frozen Slurm 1381 passes 61 primitive/recurrent/RNG/VJP tests; three compiled capture cases fail and remain open. Finish graph lifecycle qualification and benchmark performance; matching seeds alone is insufficient. |
| `model/neuron_{float,tensor,hybrid}.py` and `neuron_kernel_{float,tensor}.py` Izhikevich | Native Izhikevich exists; the regular scalar kernel has scoped external qualification | `v_t` maps directly to btorch `v_threshold`; no new scientific Module is needed for that parameter. The float-only `MultiStepIzhikevichNodeVtPTT` instead changes recovery ordering, is not dispatched by the float Module, and has an invalid backward. Keep it excluded rather than adding a model option or native kernel. Tensor and hybrid layouts have separate source blockers described below. |
| `model/GLIF.py:GLIF3` | Native GLIF3 exists; branch and Module comparisons have scoped evidence | Broader 002HIP qualification remains distinct from benchmark-branch qualification. Keep failed sparse numerical cases and invalid branch VJPs excluded. |
| `model/synapse.py` exponential/alpha families | Corresponding btorch PSC Modules and native templates exist | Qualify equations, event timing and all requested states/gradients per case; newer and legacy PSC families are separate references. |
| `model/connection.py:DualExponentialPSC`; dense/sparse `MultiStepDualExponentialPSCPTT` | Btorch Module and all three native kernels pass a restricted mapping against the unchanged historical public Torch Module (`cf8f78e81eeb904409f03c3ba0fca2f92195e5f9`), through T=128/N=64 with ten-root scalar/per-neuron/strided VJPs (Slurm 1299) | Mapping requires finite positive distinct Euler factors, transformed states/amplitude/time constants and a one-step delay of the complete projected drive including bias. Outputs, final states, whole/chunk VJPs and pending-drive continuation pass. Complete mapped whole/chunk measurements and changed-input graph checks pass at N=64 (1306). Current HEAD public API remains broken; legacy CUDA/sparse-reference performance qualification remains excluded. See CONDUCTANCE_CONTRACT.md. |
| `model/connection.py:AlphaPSC, CurrentBasedPSC`; dense/sparse kernels | AlphaPSC/AlphaPSCBilleh provide related native building blocks | Legacy Alpha updates g from the old h before adding the event to h. Establish exact temporal/integration/current-scaling mapping; isolated impulse similarity is insufficient for state/VJP/continuation qualification. |
| `model/connection.py:ConductanceBasedExponentialPSC` | Existing ConductancePSC provides voltage readout and standalone native state kernels; HeterSynapsePSC supplies heterogeneous receptor states, but their voltage-aware composition is not wired through | The btorch adapter reuses its exponential conductance-state operator and applies `g*(E-v)` in Torch; CuPy, Triton and TileLang standalone execution is covered by Slurm 1287 and benchmark run 1293. A separate public Torch-only ConductanceRecurrentNN schedule passes targeted tests/activity checks (1307); golden confirmation and native recurrent COBA scheduling remain outstanding. The 002HIP reference instead has two Euler states, per-edge weights, external F and current `(g+F)*(v-E)`, so this evidence does not establish external parity or lower its dynamics. |
| `model/connection.py:ConductanceBasedDoubleExponentialPSC` | Existing DualExponentialPSC dynamics and HeterSynapsePSC receptor layout are the reuse candidates; heterogeneous COBA composition and exact legacy mapping remain unqualified | Reference keeps rise/decay states with dt=0.1, divides their difference by tau_d-tau_r, adds F and uses v-E. Audit equal/near-equal constants, per-edge memory and voltage gradients. Its custom dispatch is commented out: distinguish executable Python behavior from available custom kernels. |

Existing native model registration can be checked in
`btorch/models/kernels/adapters.py` and `templates.py`. Existing backend tests
include dual-exponential primitive, paired and recurrent configurations; those
are Module correctness evidence, not an external performance-parity claim.
All new correctness goldens must execute actual btorch Module steps and Torch
autograd. Preserve existing equations rather than changing them to fit a legacy
reference. If no equivalent configuration exists, record the mismatch or design
and validate a separate Module before lowering it.

### Izhikevich variant qualification

The external `v_t` is the instantaneous threshold in the quadratic voltage
term. It maps to `Izhikevich.v_threshold`; external `v_peak` maps separately to
the spike cutoff. Both float kernels use that same voltage equation
(`model/neuron_kernel_float.py:325` for `MultiStepIzhikevichNodeVtPTT`), so the
presence of `v_t` does not define a missing btorch model.

The `VtPTT` suffix denotes a different recovery update. Its forward uses the
previous voltage (`model/neuron_kernel_float.py:327`), while the regular kernel
and btorch use the charged voltage. The float Module exposes this ordering as
`store_vt=True` for Torch single steps (`model/neuron_float.py:250-273`), but
its CuPy multistep route always calls regular `MultiStepIzhikevichNodePTT`
(`model/neuron_float.py:284-306`). No Module route dispatches `VtPTT`.

The isolated `VtPTT` backward is not a training oracle. Its quadratic
derivative reads `v_v_seq[t + neuron_num]`, the next post-reset voltage, rather
than the previous voltage used by its forward
(`model/neuron_kernel_float.py:417-424`). It also activates saved context from
`x_seq` or `v_init`, excluding
the recovery-only gradient case (`model/neuron_kernel_float.py:438,504-524`),
and returns no parameter gradients. The external-Module probe in
`check_002hip_izh_variant.py` reports forward, backward and observed dispatch
separately; a backward failure must not invalidate a forward-only observation.

The actual float Module is not executable unchanged in the inspected checkout.
`model/base.py:12` imports the removed `model.torch_utils` module (the tree now
contains `shape.py`), and its `register_memory(name, value, sizes)` API at line
51 is incompatible with the two-argument calls in `neuron_float.py:60,94-98`.
The probe records this as `module_import` or oracle construction failure and
stops; it does not insert compatibility aliases or substitute another base
class. Consequently, the current checkout supplies no actual-Module dispatch
qualification for `VtPTT`.

There is no tensor `VtPTT`. `IzhikevichTensorNode` also routes CuPy multistep
execution to regular `MultiStepIzhikevichNodePTT`
(`model/neuron_tensor.py:166-184`); its flattened-time parameter indexing and
invalid generated backward remain the qualifications recorded in
`002HIP_PARITY.md`. The hybrid Module asserts a Torch-only backend and the
inspected working-tree file is modified (`model/neuron_hybrid.py:282-284`), so
it is excluded from immutable external CuPy evidence. These exclusions do not
justify a new btorch update-order option or kernel.

Immutable source identities for this classification:

```text
922d618081223f932253fc1a713e261726bc31b4bc2c63c7ac468452b6bbdd20  model/neuron_float.py
7fab1a917a16556bf6afef31409c8adb79704fc2aa6d334f453602a637baf7b0  model/neuron_tensor.py
2b4278b45e2ff546623fd53e58b15340e36b90912761caec34c747ee667ba2da  model/neuron_hybrid.py (modified working tree)
16591ea380f9d73653be41f5c1cbd0f4819cf37b071249edb4ac2da3f611d1ca  model/neuron_kernel_float.py
9dcade8e303c52943722040529de57549a639ecaffbbeffbe7ad52f041b76f36  model/neuron_kernel_tensor.py
ef820ffcbf15480acfaa0dce0f941a37c09a0390c95ea8869cb7e4816526eea1  model/base.py
52071599b1823c5dc6d165b532abff41e6dd931a875374604a3162cc13074c19  model/shape.py
```

## Other library features

| 002HIP feature | btorch status | Priority |
| --- | --- | --- |
| `plasticity.py:STDPTLearner, PlasticityVoltageDependence, ShortTermPlasticity` | Corresponding online learning-rule integration is missing; `Synapse.plasticity` is reserved and sparse connections reject non-None values | Real library gap, separate from cell/PSC kernel parity. Reference learning rules also have restricted execution paths. |
| `model/deeprewire.py:DeepRewiring` hard/soft modes | `btorch/models/connection/rewire.py:HardDeepR` implements sparse fixed-slot hard rewiring with different semantics; explicitly no Soft Deep R | Separate algorithm/API track; do not count different hard-rewiring algorithms as parity. |
| Constraints, spatial connections, delay/receptors, convolutions | Related functionality already exists in connection rule/weight/synapse/sparse modules, `history.py`, `conv.py` and `hex/conv.py` | Qualify selected workloads, not blanket missing-feature claims. Position-generation/histogram-resampling helpers are utility gaps rather than kernel blockers. |

This inventory covers inspected computational families and material supporting
features; it does not claim a line-by-line port of every experiment or utility.
Keep independent connection backends and sparse nnz complexity. Required
outputs, continuation state and requested records are allowed during inference;
backward-only residual buffers, arguments, stores and tape are not.

## Working-tree provenance

SHA-256 of the inspected external files:

```text
a77be2a6464cf4b22d084d8944adcaf9f793d96033d06fbc45205697a13cfa4f  model/ALIF.py
534c467ede8bc9adb3a5ed6bfe3ce20c1dfc99680c43c075f501dffd03bd1a29  model/ELIF.py
1ea72e3a4501312845ba80fc0577f7741c83b8575c6ac3cb4549465415192f85  model/connection.py
16591ea380f9d73653be41f5c1cbd0f4819cf37b071249edb4ac2da3f611d1ca  model/neuron_kernel_float.py
```
