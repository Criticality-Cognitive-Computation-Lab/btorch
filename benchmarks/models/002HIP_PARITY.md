# 002HIP comparison contract

Status: scoped evidence only. The qualified scalar `c_m=128` inference gate
passes for all three generated backends; broader cross-repository parity and
training performance remain unestablished. See `plan.md` for delivery
requirements.

## Latest recurrent qualification (jobs 1252–1257)

The recurrent GLIF coefficient candidate shares the actual Module's
exponential-Euler coefficient helper. Job 1252 passes all T=20 sparse
GLIF3+AlphaPSC cases at N=256/1024/2048/4096 on all three native backends,
including full VJPs, continuation and capture. Job 1250 retains T=128
gradient failures despite passing forward state checks; those cases remain
unqualified. These are Module comparisons, not matched 002HIP network claims.

Job 1253 passes 880 combined regression tests with 6 expected skips before
the subsequent exponential PSC operation-order update. The updated native
PSC follows the Module's exponential-Euler graph. Job 1256 then passes
LIF+ExponentialPSC correctness at T=1/16/17/128 on all three backends with
unchanged tolerances, including sparse-weight and external synaptic-drive
VJPs. Both full-history and checkpoint modes are compared to actual Modules.

For that fixed sparse N=1024, B=1 workload, Triton and TileLang checkpoint
training pass the fixed eager and captured latency envelopes at every tested
length. CuPy passes captured timing but fails eager timing at T=16/17/128.
At T=128, all three reduce measured training peak from 5,229,056 to 3,451,392
bytes. This is evidence for the narrow LIF+exponential checkpoint plan, not
all-family residual or performance parity. Job 1257 passes 895 regression
tests with 6 expected skips after the PSC update. Detailed reports reside under `run/parity/` with names
`recurrent-glif-hoisted-matched-control.json`,
`recurrent-glif-hoisted-coefficients.json`, and
`recurrent-checkpoint-exponential-module-order.json`.

The subsequent shared CuPy raw-pointer launch removes per-argument DLPack
wrappers from fused execution. Job 1258 passes all 12 fixed checkpoint cases,
including eager and captured latency on CuPy. Its T=128 checkpoint training
time is 84.486 ms versus 71.225 ms full history (limit 85.470 ms); captured
times are 4.409 versus 4.358 ms. The earlier CuPy eager failures above remain
provenance for the superseded launch path. Report:
`run/parity/recurrent-checkpoint-cupy-raw-launch.json`. Post-launch combined
regression passes 895 tests with 6 expected skips in job 1259; the full
cross-repository scope remains unqualified.

**Captured-timing correction (jobs 1205 and 1208):** earlier 002HIP captured
medians around 7.5–7.7 us are invalid. The legacy SpikingJelly context selected
the device but left CuPy on its default stream, outside the Torch capture
stream. Those graph results did not include the complete neuron computation.
The harness now binds CuPy to the active Torch stream and checks captured
outputs against a fresh eager call after replay, bit for bit. On the qualified
`c_m=128`, T=128, N=1024 case, job 1208 measured 57.056 us for 002HIP versus
26.224 us Triton, 28.256 us CuPy and 27.584 us TileLang. The unchanged eager
latency/memory gates pass too. Earlier eager measurements are not invalidated by
this capture-specific flaw. Captured inference remains diagnostic under the
original acceptance contract; no retrospective threshold was added.

The experimental Izhikevich reciprocal/IEEE multiplication helpers and their
tests were removed: their small improvement did not justify the complexity,
and the apparent threefold deficit came from the benchmark. Production retains
the original division expressions and declared compilation policy.

## Latest functional qualification

The compiler-aware TileLang cache candidate passes **820 tests, 6 expected skips** (Slurm 1234). Its selected NVCC/ptxas/libdevice identity now enters both TileLang disk-cache layers; Slurm 1236 validates shared-cache switching between CUDA 12.9 and 13.2 without deleting shared artifacts. This fixes stale artifact reuse, not the independent CUDA 13.2 `expm1f` numerical difference. See `profile/glif-expm1-20261019/REPORT.md`.

Slurm 1238 reruns the qualified original `c_m=128`, T=128, N=1024 inference workload after the cache patch. All three fixed eager-latency and peak-memory gates pass. Verified captured medians are 56.256 us reference versus 25.424 / 27.520 / 27.488 us Triton/CuPy/TileLang. Captured inference remains diagnostic; original 002HIP training is still excluded by its failed VJP.

Slurm 1239 retains every valid same-backend branch GLIF latency/memory pass: all three short cases and Triton/TileLang long cases. The long branch CuPy VJP remains invalid, making the report overall FAIL; it is excluded rather than counted. Long current CuPy versus branch TileLang is a non-acceptance cross-provider diagnostic and narrowly misses its timing limit (112.976 versus 112.915 us); it is not relabeled a pass. Reports: `run/parity/izh-cache-fix-benchmark.json` and `run/parity/glif-cache-fix-benchmark.json`.

Slurm job 1220 passed **815 tests, 6 expected skips** for the simplified
candidate with stable GLIF and AlphaPSC integration. Job 1217 separately passed
23 AlphaPSC, fused-layout and benchmark-capture regression checks. GLIF and
AlphaPSC forward and reverse now use the Module's stable `expm1` integration,
including zero adaptation rates and infinite membrane time constants. Checkpoint
replay uses the identical adaptation update. Adaptation currents are summed
before adding external current, preserving the Module's FP32 grouping.

Job 1201 passed all 21 dense/sparse GLIF+AlphaPSC qualification cases against
the Torch Module: physical records, final states, continuation, input/state/
parameter/connection VJPs, sparse nnz gradients and dense twins. Its 12 capture
cases verified replay counts `[0, 2, 2]` and checked losses/VJPs on every trial.
This report predates the additional adaptation-sum grouping regression; job
1203 passed that focused check on every backend. Larger sparse cases remain
incomplete: jobs 1200/1202/1204 expose a few voltage values just outside the
fixed 3e-5 gate. They are not counted as numerical passes.

With the AlphaPSC update, job 1218 passes Triton's full T=20 sparse matrix at
N=1024/2048/4096. Training peaks are 1,969,664 / 3,935,744 / 7,867,904 bytes,
approximately 2x per neuron-count doubling at fixed indegree 32. CuPy and TileLang
still fail one voltage value in the N=1024 and N=4096 cases; the largest error
at a failing value is 4.58e-5. Explicit contraction-disabled/precise-division runs in job 1219 retain
small failures, so that policy is not represented as a complete fix.

Job 1223 reran the 21-case dense/sparse suite with actual graph-replay checks on
the final GLIF+AlphaPSC candidate and passed every case. Job 1222 passed the
regression that accepts a stream-bound CuPy recurrence and rejects an unbound
foreign-stream kernel after changing its input. Job 1216 additionally validated
changed-input replay against the original qualified 002HIP workload itself:
56.128 us reference versus 25.488 / 27.488 / 27.456 us for Triton/CuPy/TileLang.

Job 1199 retained every valid same-backend GLIF performance/memory pass after
the stable integration change (microseconds, current/reference):

| T, N | Backend | Captured forward | Captured forward+backward |
| --- | --- | --- | --- |
| 20, 1024 | Triton | 7.632 / 9.312 | 15.920 / 29.664 |
| 20, 1024 | CuPy | 8.864 / 17.168 | 19.408 / 42.144 |
| 20, 1024 | TileLang | 7.936 / 9.184 | 19.328 / 19.456 |
| 128, 4096 | Triton | 26.496 / 34.880 | 83.376 / 151.664 |
| 128, 4096 | TileLang | 32.256 / 28.528 | 87.904 / 94.160 |

The long branch CuPy VJP remains invalid and excluded; it accounts for the
report's overall failure. Current training allocation remains 287,232 bytes
short and 6,652,416 bytes long. Branch CuPy explicitly uses Torch's current
stream, so the separate 002HIP capture-stream bug does not affect these results.

Superseded development failures are curated here before their generated logs
are removed: job 1187 exposed generated-helper lookup and test-fixture errors;
1190 exposed unsupported Triton `tl.all` and CuPy's forced FTZ compilation.
The experimental arithmetic path was subsequently rejected. Job 1188 exposed
record-dictionary ordering in the new harness; 1189 exposed zero-step harness
handling plus the genuine GLIF `1-exp` numerical gap. Job 1195 exposed restoration
into invalidated CUDA Graph output storage. These are superseded by the focused
checks and the 21-case qualification above, not silently treated as passes.

## Environment observed on 2026-10-19

The managed sandbox hides CUDA devices: Torch reports zero available devices
there and `nvidia-smi` cannot communicate with the driver. The host has three
NVIDIA GeForce RTX 5090 GPUs. Torch 2.12.0 and the CuPy, Triton and TileLang
packages are installed.

An escalated host smoke run on physical GPU 2 passed the btorch Module parity
checks for Triton, TileLang and CuPy with scalar FP32 Izhikevich parameters,
`[T, B, N] = [4, 1, 16]`, one warmup and one measured repetition. Both
`--gradient-scope input` and `--gradient-scope all` passed. This validates the
small benchmark workflow, including complete voltage/recovery trajectories,
continuation states and VJPs; it is not performance evidence.

A local SpikingJelly checkout exists at `/data1/fanqixuan/src/spikingjelly`, but
its migrated API does not provide the `activation_based/cuda_utils.py` and
`tensor_cache.py` modules imported by 002HIP. Adding that checkout to
`PYTHONPATH` therefore does not make the external reference executable. A
compatible snapshot can instead be supplied explicitly to the optional probe
described below; it is not a package or runtime dependency of btorch.

## Optional original-source probe

`benchmarks/models/check_002hip.py` is a correctness probe, separate from the
timing benchmark. It loads `neuron_kernel_float.py` or
`neuron_kernel_tensor.py` directly with `importlib`, avoiding 002HIP's
unrelated package initializer and its `torch_sparse` dependency. It does not
modify or mock either external source tree.

The probe uses fixed FP32 input and initial-state values. It first checks the
Triton, TileLang and CuPy btorch trajectories, continuation states and
input/initial-state VJPs against the Python Module. It then compares the
requested unchanged 002HIP variant against the same oracle. Forward equations
are matched independently of the surrogate: Euler voltage is updated first,
recovery uses the charged voltage, firing observes `v_peak`, and hard reset
sets `v_reset` and increments recovery by `d`. For reverse comparison, 002HIP
uses `Sigmoid(alpha=4)` on physical `h-v_peak`; btorch uses the requested
normalized mapping `alpha=100` across the 25 mV threshold-to-reset gap.

Every invocation emits one JSON report. Missing optional dependencies, an
unavailable GPU, compilation errors and numerical mismatches return nonzero;
they are not converted to skips or parity passes. Tracebacks are retained in a
bounded diagnostic field. The report records checkout revisions and SHA-256
hashes. For the inspected 002HIP checkout at revision
`f64cd8e002af0e4285125e54d275cf7eae6ef0bf`, the original source hashes are:

| Source | SHA-256 |
| --- | --- |
| `model/neuron_kernel_float.py` | `16591ea380f9d73653be41f5c1cbd0f4819cf37b071249edb4ac2da3f611d1ca` |
| `model/neuron_kernel_tensor.py` | `9dcade8e303c52943722040529de57549a639ecaffbbeffbe7ad52f041b76f36` |

The compatible SpikingJelly snapshot is identified as revision
`7c596a05d039fbc4f51242021ca85def7ae6e30f` only when the four imported files
match the probe's pinned hashes. A directory name alone is not accepted as
revision evidence.

`--forward-only` qualifies trajectories, continuation, spike counts and
nontrivial spike activity without invoking candidate VJPs. It is used for
inference-reference qualification when the unchanged source backward is known
invalid; it does not convert that backward into a supported training path.

Example scalar and tensor invocations:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python benchmarks/models/check_002hip.py \
  --hip-root /data1/fanqixuan/src/002HIP \
  --spikingjelly-root /tmp/btorch-parity-spikingjelly-7c596a05 \
  --variants scalar --gradient-scope input --steps 128 --neurons 128
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python benchmarks/models/check_002hip.py \
  --hip-root /data1/fanqixuan/src/002HIP \
  --spikingjelly-root /tmp/btorch-parity-spikingjelly-7c596a05 \
  --variants tensor --steps 4 --neurons 16
```

Both probe and timing CLIs accept `--disable-fp-fusion`. By default, btorch
keeps `KernelOptions(enable_fp_fusion=True)` and 002HIP keeps the compatible
SpikingJelly snapshot's original compiler option, `-use_fast_math`. When the
flag is selected, btorch uses `KernelOptions(enable_fp_fusion=False)` and the
probe appends a final `--fmad=false` to the unchanged 002HIP compilation
options. Every JSON report records the requested policy plus the original and
effective 002HIP option lists. This controls compatible floating-point
contraction only; it does not claim that broader fast-math behavior is
disabled, and it does not modify either external source tree.

The original tensor backward source references loop-local `t` after its loop,
so a compiler rejection is an expected reference qualification and must remain
a reported failure. The tensor forward also indexes parameter arrays with the
flattened time index; the probe supplies explicit constant `[T, N]` storage and
reports that layout rather than altering the source.

The probe uses separate tolerances for forward values (`atol=1e-4`,
`rtol=1e-4`) and gradients (`atol=5e-5`, `rtol=5e-4`). VJPs use fixed
unit-scale cotangents for every spike, voltage and recovery trajectory element,
without division by time or neuron count. The report includes the exact spike
count and fails a case with zero spikes or spikes at every element, because
such a case does not exercise both sides of reset dynamics.

The normal child sandbox execution on 2026-10-19 remained `NOT_RUN`: Torch
reported zero devices and `nvidia-smi` could not access the driver. A host run
on GPU 1 then executed both variants with `[T, N] = [20, 128]` and the full
input-plus-initial-state gradient scope. The then-current probe reported passes
for all three btorch generated backends and reported exact 002HIP spikes with a
maximum voltage difference of `3.8147e-6`. The gradient evidence from that run
is superseded by the stronger method below.

That earlier run used a mean-square loss, whose division by `T*N` can hide a
VJP defect, and it did not record whether any spikes occurred. Its scalar input
VJP maximum difference (`2.1388e-6`) is retained as historical diagnostic
evidence only. The same run failed the initial-voltage VJP (`0.1415682`) and
initial-recovery VJP (`0.0076697`). It does not establish a validated training
scope. The tensor backward failed compilation with `CompileException` because
`t` is undefined at the generated source line that uses it after the loop. The
combined probe therefore correctly returned exit code 1 and establishes no
broad parity claim.

Use `--gradient-scope input` to test the common published scope independently.
Input-only training becomes eligible for timing only after the stronger
unit-cotangent probe passes with nontrivial spike activity. Use
`--gradient-scope all` for the stricter initial-state diagnostic.

The stronger host run on GPU 1 used `[T, N] = [128, 128]`, input-only
gradients, the fixed unit-scale cotangents and the gradient tolerances above.
It produced 256 spikes in the Python Module and every candidate, so both spike
and reset branches were exercised. The unchanged scalar 002HIP input VJP
failed at input `[0, 0]` with maximum absolute difference `356.347015`.
Therefore this original scalar kernel is not an accepted training reference
for the case.

The same run also rejected full-trajectory inference at the fixed forward
tolerance: the unchanged scalar kernel and all three generated backends had
exact spike counts but a maximum voltage difference of `0.00535774`, first
observed at trajectory index `[45, 20]`. The Triton input VJP passed with
maximum absolute difference `0.00047326` under the fixed relative tolerance;
CuPy and TileLang failed with maximum absolute difference `0.02630186`, first
observed at input `[0, 26]`. These are actionable numerical mismatches. The
tolerances were not relaxed, and this case is excluded from performance
acceptance.

The explicit contraction-disabled host probe repeated the same Izhikevich
`[T, N] = [128, 128]` case with 256 spikes. The unchanged 002HIP forward then
passed: maximum voltage difference was `0.001094818` and maximum recovery
difference was `2.384e-5`. Its input VJP still failed with maximum difference
`356.3469238`, so it remains invalid as a training reference.

All three btorch generated backends passed forward and input-VJP checks under
the same selected policy. CuPy and TileLang were forward bit-exact and had
maximum gradient difference `3.33786e-6`. Triton had maximum voltage
difference `0.001094818`, maximum input-gradient difference `5.93662e-5` and
maximum final-voltage difference `3.8147e-6`; it passed the fixed relative
tolerance. The 002HIP option list still begins with `-use_fast_math`, so
approximate division remains possible even though final `--fmad=false`
controls contraction. These results do not claim that all fast-math
transformations are disabled. This selected configuration qualifies the case
for inference timing only.

## Optional scalar performance gate

`benchmarks/models/bench_002hip.py` is limited to the unchanged scalar case.
It rejects fewer than three warmups or 30 measured repetitions and rechecks
trajectories and continuation before any timing; training mode additionally
checks input VJPs. Forward-valid inference can be selected explicitly with
`--mode inference`; that mode omits backward measurements and reports
inference acceptance only. `--mode training` requires the original input VJP
to pass before any measurements. Both modes require forward, continuation,
spike-count and nontrivial-activity validation,
so the default-policy `[128, 128]` failure is not timed. The
contraction-disabled configuration passes the inference gate but remains
excluded from training timing because the unchanged 002HIP VJP fails.
The tensor variant, initial-state gradients and parameter gradients are
excluded because the correctness probe did not validate them.

The acceptance envelope was fixed before measurement. For the empty-public-
history case, each measured latency must satisfy:

```text
candidate median <= max(1.2 * original median, original median + 0.005 ms)
```

Inference mode applies it to steady inference and requires isolated inference
peak allocated bytes no larger than the original scalar reference. Training
mode additionally applies it to captured forward-plus-backward replay and
requires training peak allocation no larger than the reference. Host
end-to-end latency, first post-validation calls, graph replay latency, ranges
and return sizes are reported separately and do not replace the device gate.

002HIP always emits voltage and recovery trajectories. When no public records
are requested, the benchmark adapter returns only spikes and cloned final
states, but peak memory still counts those unavoidable internal trajectories.
For btorch, frozen grad-enabled inference asserts that no autograd graph or
reverse-only tape is created. `--histories both` additionally reports the full
voltage/recovery history case, while acceptance remains tied to empty public
histories.

Large GPU timing runs must use Slurm. Create the scheduler log directory before
submission; `/tmp/btorch-bench-parity.sbatch` creates a separate
`run/parity/$SLURM_JOB_ID` directory, inherits the active Python environment,
and does not override the scheduler's `CUDA_VISIBLE_DEVICES` assignment.

The qualified `c_m=128` Izhikevich inference workload can be submitted with:

```bash
mkdir -p run/parity
/usr/bin/sbatch /tmp/btorch-bench-parity.sbatch \
  benchmarks/models/bench_002hip.py \
  --hip-root /data1/fanqixuan/src/002HIP \
  --spikingjelly-root /tmp/btorch-parity-spikingjelly-7c596a05 \
  --mode inference --disable-fp-fusion --capacitance 128 \
  --steps 128 --neurons 1024 \
  --warmup 10 --repeats 30 --histories empty
```

The harness records GPU/software versions, external revisions and source
hashes in its JSON output. A report claims gate success only after a
host-visible GPU run completes. This scalar primitive case does not cover
recurrent execution, PSCs or benchmark-branch GLIF cases.

The contraction-disabled inference gate was attempted on GPU 1 with
`[T, N] = [128, 4096]`, 10 warmups and 30 requested repetitions. It was
rejected before timing, so no latency or memory samples were collected. All
implementations produced 8,192 matching spikes, but unchanged 002HIP and
Triton failed full voltage-history validation with maximum difference
`0.0024719`, first observed at `[45, 28]`; CuPy and TileLang passed. The result
is a failed acceptance with no performance claim. The remaining drift is not
attributed to contraction alone; approximate division is a separate numerical
factor.

A shorter `[T, N] = [32, 1024]` run with the same published physical
parameters passed forward-value and continuation checks, but all
implementations produced zero spikes. The nontrivial-activity gate rejected it
before timing. Unchanged 002HIP and Triton had maximum voltage difference
`1.91e-5` and recovery difference `3.58e-7`; TileLang and CuPy were bit-exact.
It is not used as a performance substitute for the long case.

Both Izhikevich CLIs accept `--capacitance`, defaulting to the published
`c_m=100`. The value is carried by one frozen case definition into the Python
Module parameters, generated backends, unchanged 002HIP argument order and
JSON metadata. `c_m=128` is an explicitly labeled arithmetic edge workload:
its power-of-two reciprocal separates kernel overhead from the published
case's reciprocal-rounding drift. It does not replace the documented
`c_m=100` failures or long-horizon coverage, and it must still pass strict
forward and nontrivial-spike qualification before inference timing.

The GPU 1 `c_m=128`, `[T, N] = [128, 1024]` forward-only qualifier passed
overall with 1,280 spikes. Unchanged 002HIP had maximum voltage difference
`7.629e-6`, maximum recovery difference `9.537e-7`, final-voltage difference
`3.815e-6` and exact final recovery. Triton, TileLang and CuPy were bit-exact
for all trajectories and continuation states. This qualifies only the labeled
arithmetic edge workload for inference timing; it says nothing about the
unchanged source backward and does not rehabilitate the published `c_m=100`
case.

Slurm job 1163 then measured the qualified `c_m=128` inference workload with
10 warmups and 30 repetitions. Numerical validation passed, including 1,280
matching spikes. The fixed acceptance gate nevertheless failed every generated
backend on steady uncaptured device latency:

| Backend | Median (ms) | 002HIP median (ms) | Ratio | Limit (ms) |
| --- | ---: | ---: | ---: | ---: |
| Triton | 0.337584 | 0.198224 | 1.703 | 0.237869 |
| CuPy | 0.525872 | 0.198224 | 2.653 | 0.237869 |
| TileLang | 0.694624 | 0.198224 | 3.504 | 0.237869 |

All generated backends passed the peak-allocation check at 532,992 bytes,
compared with 3,162,112 bytes for unchanged 002HIP. Captured inference replay
was reported separately: 002HIP was 0.007472 ms, while Triton, CuPy and
TileLang were 0.031536, 0.033680 and 0.033632 ms. Those values would also fail
the same noise-aware envelope, whose graph-scale limit would be 0.012472 ms.

The acceptance contract remains the steady uncaptured inference median. It
must not be reinterpreted after seeing job 1163. A future captured-performance
gate would need to be declared separately before collecting new evidence.
The job also exposed a reporting-only issue: empty-history inference rows
combined a passing forward status with backward `not_run` and displayed a
false row failure. The benchmark now treats that row as passing in inference
mode; the job's overall performance failure is unchanged.

Job 1163 recorded the external 002HIP and SpikingJelly identities but predated
the current-candidate source manifest. It therefore does not contain immutable
hash evidence for the dirty btorch checkout that produced the measurements.
New `bench_002hip.py` reports capture the current commit, dirty indicator and
SHA-256 values for the benchmark, `check_002hip.py`, kernel options, contracts,
templates, operators, backend engine and backend fragments before validation
or timing. They also record Triton and TileLang versions. Job 1163 remains
useful measured evidence, but this provenance gap must accompany it.

Slurm job 1176 reran the same qualified workload after the frozen-dispatch
and launch-overhead changes. Numerical qualification again passed with 1,280
matching spikes, and all three generated backends passed the unchanged steady
inference and peak-allocation gates:

| Backend | Job 1163 median (ms) | Job 1176 median (ms) | 002HIP median (ms) | Ratio | Limit (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Triton | 0.337584 | 0.185264 | 0.195984 | 0.945 | 0.235181 |
| CuPy | 0.525872 | 0.196720 | 0.195984 | 1.004 | 0.235181 |
| TileLang | 0.694624 | 0.171840 | 0.195984 | 0.877 | 0.235181 |

Each generated backend still peaked at 532,992 allocated bytes versus
3,162,112 bytes for unchanged 002HIP. Captured replay remained a separately
reported diagnostic at 0.031584 ms for Triton, 0.033696 ms for CuPy and
0.033760 ms for TileLang, versus 0.007456 ms for 002HIP; it is not the fixed
acceptance metric. The retained `run/parity/slurm-1176.log` report records the
dirty btorch revision `c8b36604c0229279e43bc669a9750ad3e3b253c5`, hashes of
the benchmark and current kernel dependencies, external source identities,
and Torch, Triton and TileLang versions before validation and timing. This is
performance evidence only for the labeled `c_m=128` scalar inference case;
002HIP training remains excluded by its failed input VJP.

Slurm job 1183 measured the subsequent integrated-validation build under the
same contract. All three gates passed again, with lower steady inference
medians than job 1176:

| Backend | Median (ms) | 002HIP median (ms) | Ratio | Captured diagnostic (ms) |
| --- | ---: | ---: | ---: | ---: |
| Triton | 0.147456 | 0.197376 | 0.747 | 0.025504 |
| CuPy | 0.157792 | 0.197376 | 0.799 | 0.027520 |
| TileLang | 0.142144 | 0.197376 | 0.720 | 0.027744 |

Peak allocation remained 532,992 bytes for each generated backend versus
3,162,112 bytes for 002HIP; its captured diagnostic was 0.007552 ms. The
retained `run/parity/slurm-1183.log` includes the current engine and operator
hashes. Job 1181 separately passed five CuPy tests for integrated timestep
validation, including invalid-device values and CUDA Graph replay. The check
now executes inside the primitive kernel, preserving rejection semantics and
capture behavior without separate validation launches. This does not change
the `c_m=100` reference failures or the inference-only qualification scope.

## Benchmark-branch GLIF qualification

`benchmarks/models/check_benchmark_glif.py` separately qualifies the current
generated GLIF backends and unchanged benchmark-branch primitives against the
current Python Module. It uses FP32, `dt=0.5`, two after-spike-current modes,
soft reset, deterministic unit-scale cotangents and nontrivial spike activity.
The inspected benchmark branch was commit
`4bd0b0a649188173801549dcb970d50b2efd772c`; source hashes and the complete
current-source identity are retained in
`/tmp/btorch-glif-reference-current.json`.

After correctness qualification, submit GLIF timing through the same wrapper:

```bash
mkdir -p run/parity
/usr/bin/sbatch /tmp/btorch-bench-parity.sbatch \
  benchmarks/models/check_benchmark_glif.py \
  --branch-root /data1/fanqixuan/src/btorch.worktree/benchmark \
  --steps 20 128 --neurons 1024 4096 \
  --backends triton tilelang cupy --timing --warmup 10 --repeats 30
```

For `[T, N] = [20, 1024]`, all Triton, TileLang and CuPy implementations in
both current btorch and the benchmark branch passed forward and VJP checks,
with 2,074 spikes. For `[T, N] = [128, 4096]`, all three current generated
backends passed, and benchmark-branch Triton and TileLang passed, with 44,239
spikes. Benchmark-branch CuPy retained exact forward spikes but failed its
long-horizon inverse-recurrence backward: maximum absolute differences were
`0.00058254` for input, `0.00125361` for initial voltage, `0.00132596` for
initial after-spike current and `0.00098157` for after-spike-current
amplitudes. The long benchmark-branch CuPy training case is therefore not
acceptance evidence.

Slurm job 1165 measured only numerically valid same-backend pairs, using 10
warmups, 30 repetitions, captured forward and captured forward-plus-backward
device medians, and incremental peak allocation. The fixed latency envelope
was `current <= max(1.2 * branch, branch + 5 us)`; current allocation had to be
no larger than the branch reference.

| Case | Backend | Result | Captured forward, current/reference/limit (us) | Captured forward+backward, current/reference (us) | Allocation result |
| --- | --- | --- | ---: | ---: | --- |
| T20 N1024 | Triton | Pass | 13.264 / 9.248 / 14.248 | 19.408 / 29.776 | Both pass |
| T20 N1024 | CuPy | Fail | 14.112 / 17.152 / 22.152 | 21.408 / 42.032 | Training fails: 442,880 > 290,816 bytes |
| T20 N1024 | TileLang | Fail | 14.992 / 9.168 / 14.168 | 21.264 / 19.456 | Both pass |
| T128 N4096 | Triton | Pass | 32.592 / 34.736 / 41.683 | 63.424 / 152.592 | Both pass |
| T128 N4096 | CuPy | Not counted | Branch CuPy failed numerical validation | Not counted | Not counted |
| T128 N4096 | TileLang | Fail | 35.936 / 28.640 / 34.368 | 73.456 / 94.032 | Both pass |

Current CuPy at T128 N4096 also passed every check when compared to the valid
branch Triton reference, including 36.704 us captured forward, 73.536 us
captured forward-plus-backward and both allocation limits. That cross-provider
diagnostic does not count toward same-backend acceptance. The overall GLIF
timing report therefore remains failed despite the two Triton passes.

Job 1184 used the GC-corrected allocation measurement and records the
benchmark script plus shared options, contracts, templates, model, surrogate,
operators, engine, fragments and selected backend source in every current
child identity. This closes the provenance and zero-byte memory-baseline gaps
in job 1165. Every numerically valid same-backend gate passed:

| Case | Backend | Captured forward, current/reference (us) | Captured forward+backward, current/reference (us) | Training allocation, current/reference (bytes) |
| --- | --- | ---: | ---: | ---: |
| T20 N1024 | Triton | 7.648 / 9.184 | 17.328 / 29.792 | 287,232 / 446,464 |
| T20 N1024 | CuPy | 7.712 / 17.296 | 25.536 / 42.000 | 287,232 / 290,816 |
| T20 N1024 | TileLang | 7.648 / 9.072 | 19.152 / 19.632 | 287,232 / 446,464 |
| T128 N4096 | Triton | 26.592 / 34.912 | 106.176 / 151.632 | 6,652,416 / 10,633,216 |
| T128 N4096 | TileLang | 28.928 / 28.640 | 83.776 / 94.192 | 6,652,416 / 10,633,216 |

Inference allocation also passed at 176,640 versus 188,416 bytes for every
short case and 4,243,968 versus 4,292,608 bytes for both valid long cases. The
report at `run/parity/glif-integrated-dt-1182.json` remains overall failed only
because long benchmark-branch CuPy failed its previously documented VJP and
is excluded from performance acceptance; that failure is not masked by the
five passing comparisons.

Slurm job 1175 passed 740 tests with 6 skips for the earlier checkpoint build.
The subsequent frozen pipeline in job 1182 passed 761 tests with 6 skips in
301 seconds, including fused source caching and integrated primitive timestep
validation. Those checks support the current implementation state; they do
not convert rejected external-reference cases into parity or performance
passes.

Jobs 1166 and 1171 exercised intermediate checkpoint candidates and failed
14 and 7 tests respectively; they are development failures, not parity
evidence. Debug job 1170 traced the TileLang defect to generated CUDA using a
negative checkpoint index after simplifying `(time // 16) - 1`. Rewriting the
shift, adding the CuPy integer clamp and correcting the empty-workload oracle
led to all 18 focused residual tests passing in job 1174 and the 740-test full
pass in job 1175. The obsolete failed/debug logs can be removed after this
chronology is retained; accepted baseline and performance reports remain
separate.

Slurm job 1178 profiled the qualified Izhikevich validation path. Hardware
counter collection was denied with `ERR_NVGPUCTRPERM`, so no occupancy,
throughput or stall claim is available. Torch profiling attributed seven
device timestep-validation launches totaling 7.488 us and one scan launch of
24.224 us; the warmed native scan diagnostic was 24.065 us. These figures are
diagnostic launch attribution, not a replacement for the repeated acceptance
gate. Integrated validation checks and fused rendering caches remain under
development. CPU-only cache observations reduced a forward setup path from
about 613 us to 7 us and a reverse setup path from 1.68 ms to 7.9 us, but they
are not GPU performance evidence until the integrated implementation passes
GPU correctness and benchmark runs.

The follow-up job 1185 traced one kernel per warmed native call: 25.376 us for
Triton, 28.608 us for CuPy and 28.160 us for TileLang, each with 1,280 spikes.
This confirms that the integrated timestep check no longer appears as seven
separate launches. The trace remains diagnostic and does not replace job
1183's repeated end-to-end acceptance measurements.

## Optional PSC source qualification

`benchmarks/models/check_002hip_psc.py` executes the unchanged
`model/connection_kernel.py` current-based Alpha CUDA source against btorch's
`alpha_billeh` Python Module and all three generated backends. The inspected
source SHA-256 is
`37b5ad842a491bbaa19107b9fe16975ee8dc1f9919d170f22e28847f83995ddf`.
The fixed FP32 cases use `[T, N] = [8, 2]`, identity weights, zero initial
states, `tau_syn=5.5`, and a unit impulse at either step 0 or step 3.

The source qualification is intentionally a failure detector. The original
dense loop reads `z_seq[i]` without adding the current timestep offset, so a
step-3 impulse is not delivered. It also emits current from the old rise state
before injecting the current event, while `AlphaPSCBilleh` advances current
from the updated rise state so the event acts in its delivery step. The two
impulse placements expose both defects without replacing the original source
with a simulator or mock.

The GPU 1 probe returned exit code 1 as required. For the step-0 impulse, the
oracle's first nonzero current was `[0, 0]`, while unchanged 002HIP first became
nonzero at `[1, 0]` and reached maximum current `5.87448` because it reused the
first input row at every step. For the step-3 impulse, the oracle first became
nonzero at `[3, 0]`, while unchanged 002HIP remained identically zero. Triton,
TileLang and CuPy passed both cases; their maximum current difference was
approximately `5.96e-8`. The original current-based Alpha source is therefore
not an accepted forward reference.

The same source file's conductance primitive is classified `unsupported`
rather than compared by class name. It evolves two Euler states `g/h`, adds a
constant `F`, and defines current using `(g + F) * (v - E)`. Btorch's
`ConductancePSC` uses one exponentially decayed conductance state and
`g * (E - v)`. Those equations and state contracts are not equivalent.

Example invocation:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python \
  benchmarks/models/check_002hip_psc.py \
  --hip-root /data1/fanqixuan/src/002HIP \
  --spikingjelly-root /tmp/btorch-parity-spikingjelly-7c596a05
```

## Initial case matrix

| Case | Shared comparison contract | Required validation |
| --- | --- | --- |
| Izhikevich scalar | FP32, dt=1, hard reset, fixed scalar parameters | Spikes, voltage/recovery trajectories, final states, input/initial-state VJPs |
| Izhikevich per-neuron | Same equations with independent neuron parameters | Same checks, then genuinely heterogeneous values |
| Current-based Alpha PSC | Map coefficients and update order before execution | Impulse/step response, all states, input/state VJPs |
| Recurrent dense | Validated cell pair and identical topology/schedule | Complete trajectories, continuation, input/state/weight VJPs where supported |
| Recurrent sparse | Validated propagation and cell pair | As above, including orientation and delayed arrival timing |
| Conductance | Inventory voltage dependence and temporal observation point first | Current plus conductance/voltage gradients; not implied by PSC name |

Use time lengths 20 and 128 initially, neuron counts 1024 and 4096, and
batch sizes 1 and 4 where both implementations support the same layout.
Record unsupported batch/layout combinations explicitly. Test empty/single-step
and chunk boundaries for correctness separately from timing cases.

002HIP's published Izhikevich benchmark uses c=100, k=2, v_t=-40,
v_peak=30, v_min=-65, v_r=-60, a=0.02, b=0.2, d=8 and input in [200,240).
Map these respectively to btorch c_m, k, v_threshold, v_peak, v_reset,
v_rest, a, b and d. Initialize v=-65 and u=0. Match the surrogate and its
normalization explicitly before comparing VJPs; matching forward equations
alone is insufficient. Test reset-detach choices separately.

## Known reference qualifications

- 002HIP's inspected Izhikevich custom backward returns no parameter gradients.
  Its commonly published training benchmark differentiates the input only,
  while the kernel can also return initial-state gradients. Full btorch
  initial-state and parameter training is a separate capability and
  measurement.
- The original forward decides whether to save backward context from
  `x_seq.requires_grad or v_init.requires_grad`; `u_init` alone does not
  activate the training path. In the inspected scalar source this expression
  is at line 147 and guards `save_for_backward` at lines 213-219; the tensor
  source has the same branch at lines 145 and 211-217. These locations are
  tied to the source hashes above. The probe records this source contract
  rather than treating recovery-only differentiation as supported.
- Its tensor kernel has suspect generated backward expressions (including
  uses of loop-local `t` after the loop); compilation and independent VJP
  validation are prerequisites, not assumed passes.
- Its sparse PSC forward scans every COO edge per destination and accesses
  `z_seq[row]` without a temporal offset. Validate time-varying inputs before
  accepting it as a numerical reference. Do not reproduce a reference bug.
- Some PSC implementations use Euler coefficients rather than btorch's
  exponential integration. Establish an equivalent model or classify the
  comparison as unsupported; do not change btorch dynamics for a timing win.
- The btorch benchmark supports explicit `input` and `all` gradient scopes.
  `input` differentiates only the drive tensor; `all` also differentiates
  initial states and parameters. Before timing, it checks output, complete
  state trajectories, continuation states and participating VJPs against the
  Python Modules. Timed inference requests no state histories and runs with
  grad mode enabled but all tensors frozen, asserting that no autograd graph
  is created. Inference and training peak allocations are reported separately.
  These are backend regression measurements, not the complete
  cross-repository gate specified above.

## Reproduction and acceptance

From the btorch root, after restoring CUDA availability:

```bash
mkdir -p run/parity
/usr/bin/sbatch /tmp/btorch-bench-parity.sbatch \
  benchmarks/models/bench_cell_backends.py \
  --model izhikevich --parameter-layout scalar \
  --gradient-scope input --steps 20 --batch 1 --neurons 1024 \
  --warmup 10 --repeats 30
/usr/bin/sbatch /tmp/btorch-bench-parity.sbatch \
  benchmarks/models/bench_cell_backends.py \
  --model izhikevich --parameter-layout neuron \
  --gradient-scope input --steps 128 --batch 4 --neurons 4096 \
  --warmup 10 --repeats 30
```

Run `--gradient-scope all` separately to measure btorch's initial-state and
parameter gradients. Do not compare that result with a 002HIP input-only run.

Before optimizing, collect equivalent 002HIP runs using the fixed latency and
memory thresholds above. The qualified `c_m=128` scalar inference gate passes;
no 002HIP training gate passes. Record GPU identity, package versions,
warmup, repeats, compilation cost, gradient scope and requested records.
Measure eager and captured calls separately and include preparation costs.
Report numerical tolerance and first divergent spike independently of timing.
Missing dependencies, unavailable devices and rejected reference kernels are
NOT_RUN/UNSUPPORTED, never successful parity evidence.

The integrated-validation development run 1179 correctly rejected CuPy NVRTC's
missing system `assert.h`; the final implementation uses the headerless CUDA
`__assertfail` intrinsic, compiled in probe 1180 and verified with real invalid
values and captured replay in job 1181, then the full suite in 1182. Its failed
intermediate log is superseded. The invalid zero-memory report from job 1177 is
also superseded by the GC-controlled report from job 1184; neither contributes
acceptance evidence.

### Recurrent storage and paired layouts: focused verification

Slurm job 1186 passed 28 targeted tests in 30.52 seconds. The tests exercise
paired forward/VJPs on CuPy, Triton and TileLang with original parameter spans,
offset/strided and zero-stride views, receptor-axis broadcasting and shared
low-precision leaves. Recurrent final charged-memory losses, continuation and
capture checks also pass. Unrequested charged trajectories are omitted during
training; physical state histories are still retained. Parameter reverse still
uses lane-sized logical gradient buffers followed by broadcast reductions.

This is functional evidence, not a new performance comparison. The retained log
is `run/parity/slurm-1186.log`. Source SHA-256 at verification:

| File | SHA-256 |
| --- | --- |
| `backends/fused.py` | `69ea2596773141047d28e21c904bfd8c2a42c163307f3f621538e1b22de10f4c` |
| `recurrent.py` | `293e4081a7f2f9d89708743cb4c8d65a1167d0b866bf2447da9eb1a6abca4d7e` |
| `recurrent_operator.py` | `95e4255e4b757be16c2b24001db10f204d2a685d22b784a40983493b5f8ff5fa` |

### Network reference boundary

The benchmark branch's GLIF dense/sparse recurrence delivers
`x_t + bias + W @ spike_(t-1)` directly to the neuron. Current `RecurrentNN`
delivers the prior PSC current to the neuron and then updates the PSC from the
new spike. AlphaPSC adds two physical states and a different temporal filter.
Consequently, branch full-network latency/memory is not a matched baseline for
current RecurrentNN. Its GLIF cell and dense/CSR propagation remain useful
component references; neither proves PSC or delayed-network parity.

Current dense/sparse GLIF qualification must compare identical GLIF3+AlphaPSC
models against the ordinary Torch Module, including physical trajectories,
initial-state and connection VJPs, split continuation and capture replay. Sparse
checks must distinguish required dense neuron cotangents from connection
gradients, whose storage should scale with stored edges rather than N squared.

### Standalone conductance and long GLIF follow-up

Slurm 1287 passes 21 standalone conductance tests on CuPy, Triton and TileLang.
The exact `ConductancePSC` adapter reuses the existing exponential recurrence
and ordinary Torch `g * (E - v)`. Actual Module single steps establish the
output, state, delay and full-VJP oracle. Changed-input CUDA Graph training
replay and inference without autograd tape pass. These are functional results;
voltage-aware recurrent scheduling, standalone performance, and the distinct
legacy 002HIP conductance equations remain unqualified. See
[the conductance contract](CONDUCTANCE_CONTRACT.md).

Slurm 1288 reports 916 passed and 6 skipped combined regression tests before
the subsequent Alpha record-association and recurrent COBA guard changes.
The shared CuPy direct/fused launcher uses raw tensor pointers on Torch's
current CUDA stream, without DLPack conversion.

The corrected cut-graph diagnostic in Slurm 1285 identifies a genuine FP32
association difference at reverse boundary 126: `(carried + record) + neuron_gx`
differs from the actual Module's `(carried + neuron_gx) + record` in 251 PSC
lanes, maximum 1.1920929e-7. Moving only the Alpha PSC record addition after
neuron coupling yields 29/30 passing cases in Slurm 1289. All CuPy/TileLang
T=128 scaling cases through N=4096 pass, as do Triton N=2048/4096. Triton
N=1024 retains one failing sparse-weight gradient lane (absolute difference
2.3651123e-4); the overall report remains FAIL. Tolerances and the benchmark's
reference execution protocol were unchanged. This does not establish broad
GLIF parity or a new performance result.

### Restricted historical dual-exponential qualification

The executable external reference is the unchanged public Torch
`DualExponentialPSC` at 002HIP revision
`cf8f78e81eeb904409f03c3ba0fca2f92195e5f9`, with the real SpikingJelly
`MemoryModule`. The current HEAD has a separately reproduced public-method
arity error. Its CUDA kernels and purported sparse Torch route are not used
as scientific or performance goldens.

Slurm 1295 passes all four btorch paths (Torch plus the three native backends)
at T=32/N=16. Jobs 1296/1297 extend to T=128/N=64; 1297 checks whole and chunked
VJPs for inputs, dense weights, bias and initial legacy g/h. The mapping carries
one pending projected drive, preserving its bias and gradient path across
chunks. All checks pass unchanged atol=rtol=2e-5. Native maximum differences
are 6.676e-6 for outputs/final states, 1.073e-6 for whole VJPs and 1.193e-6
for chunk VJPs.

Run the following command inside a Slurm GPU allocation:

```bash
PYTHONPATH=. python benchmarks/models/check_002hip_dual_exponential.py \
  --backends torch triton cupy tilelang --device cuda \
  --steps 128 --neurons 64 --split 53 \
  --report run/parity/002hip-dual-exponential-chunk-vjp.json
```

The report records full external/dependency revisions, source hashes and dirty
state. The reference is extracted to temporary storage; the external checkout
is not changed. Unused external CUDA imports are placeholders only, explicitly
outside qualification. The btorch implementation uses the actual Module with
an identity projection after the explicit delayed drive, not a copied golden
recurrence or a dense identity matrix. The finite-pole parameter/state mapping,
limits and remaining exclusions are specified in
[CONDUCTANCE_CONTRACT.md](CONDUCTANCE_CONTRACT.md). No external CUDA/sparse or
performance acceptance claim follows from this functional slice.

Slurm 1299 extends that historical mapping to scalar, per-neuron and strided
roots for `tau_decay`, `tau_rise`, `g_max`, `E_syn` and `v_r`, alongside the five
input/connection/initial-state roots. All 12 backend/layout cases pass whole
and chunk VJPs at T=128/B=1/N=64 with unchanged atol=rtol=2e-5. Inactive strided
backing slots have exactly zero gradients; same-root parameter updates change
the reference and retain parity after recomputing the mapping. Maximum native
absolute parameter-VJP error is 5.723e-5, within the combined absolute/relative
gate. This wider matrix does not inherit the smaller fixed-scalar error bound.

The dual-exponential mode of `bench_cell_backends.py` measures the
already-projected-drive cell separately. Serialized Slurm jobs 1302/1303 cover
scalar/neuron parameter layouts, FP32 T=128/B=1/N=1024, three warmups and 30
repeats. All ten backend rows pass actual-Module forward/trace/VJP and
changed-input graph checks. Captured native inference ranges from 7.20 to
22.98 us; captured training ranges from 48.13 to 61.78 us. Native eager
inference incremental peak is 537088 bytes versus 536576 required return
bytes. Timing and allocation scopes, full measurements and source hashes are
in `run/parity/dual-exponential-component.md` and JSON. Projection, projected
delay, pending carry and final legacy-state transforms are excluded; no
external CUDA acceptance follows. Concurrent initial jobs 1300/1301 were
cancelled and excluded from timing evidence.

Slurm 1298 consolidates the related regression: **962 passed, 6 skipped**.
Repository-wide Ruff and pre-commit for the changed benchmark/test files pass.


### Complete mapped dual-exponential measurement and public COBA schedule

Slurm 1306 passes all six native whole/two-chunk cases at FP32
T=128/B=1/N=64, including projection, pending-drive delay, parameter/initial-
state transforms, final legacy-state reconstruction and all ten-root VJPs.
Changed-input captured execution is checked against the unchanged archived
Torch Module. Native captured inference is 56.98–78.74 us and training is
238.51–290.29 us. Inference saves no autograd tape. Incremental eager training
peak is lower than the archived Torch case, but unique saved storage is higher;
these are different metrics, and neither alone identifies private residual
storage. Full scope and exclusions are in
`run/parity/002hip-dual-exponential-end-to-end.md`. Run 1304 has an independent-
call benchmark reset bug; 1305 has invalid captured timing units. Neither is
accepted timing evidence. Their original artifacts remain available.

The public Torch-only `ConductanceRecurrentNN` reuses the existing recurrent
scheduler and actual neuron/PSC Modules. Current uses previous conductance and
previous post-reset voltage; newly emitted spikes update conductance for the
following step. Slurm 1307 passes 85 targeted public/standalone tests, including
stateful graph replay and recapture after parameter replacement. The activity
harness passes all 16 checks at single/pair/network scales. Default-surrogate
public `torch.compile` continuation/VJPs additionally pass on CPU; strict
fullgraph mode retains the same custom-JVP limitation as ordinary RecurrentNN.
The golden confirmation is pending; no new coupled native kernel is claimed.

The former proposal artifact directory was overwritten during the public
harness update. Seven authentic pre-public files were recovered byte-for-byte
from a surviving original temporary run, with hashes and provenance in
`run/parity/conductance-validation/RECOVERY.json`. They are not proven identical
to the overwritten directory instance. Their 26 raw tensor channels (13540
values) agree bitwise with the new public validation. Both the overwritten
public run and current public validation are preserved separately.


Slurm 1309 extends complete mapped dual-exponential measurement to
T=128/B=1/N=256 with uneven chunks 47/81. All six native whole/chunk cases
pass unchanged actual-Module/tolerance/ten-root checks and changed-input graph
validation. Captured inference is 62.34–82.96 us and training 254.86–301.97 us;
inference saves no autograd tape. The scoped report is
`run/parity/002hip-dual-exponential-end-to-end-n256.md`. This does not validate
legacy CUDA/sparse reference implementations.

Broadening regression to the entire RNN folder in Slurm 1308 reveals
**39 failed, 1081 passed, 6 skipped**. The failures are 38 gradient-history
checks and one compiled CPU-offload context re-entry; classification against
an unchanged baseline is pending. Do not describe the expanded suite as green
or replace this result with the earlier narrower 962-test pass.


Clean-HEAD controls classify all 39 expanded-suite failures as existing defects:
38 gradient-history failures reproduce in an exact archive of c8b3660, and
Slurm 1312 reproduces the same compiled CPU-offload context re-entry in that
archive with a fresh Inductor cache. These failures remain open; the expanded
suite remains failed. Ordinary output/input/parameter-gradient checks in the
gradient-history matrix pass.

Independent capture review found and fixed two real issues: grad-enabled
inference capture now rejects trainable buffers/initial memories as well as
parameters/inputs, and CPU tensor timestep mutations invalidate captured
constants. CUDA tensor timestep updates retain live-storage replay. Slurm 1311
passes all 34 public conductance tests, including CPU inference-tensor fallback,
continued state/output agreement, no saved inference tensors and actual replay.
Targeted pre-commit and repository-wide Ruff pass.


### Reverse output pruning and parameter-layout measurements (October 20)

Historical dual-exponential complete mapping now also passes per-neuron and
strided physical-root measurements (Slurm 1313/1314), six whole/chunk cases per
layout. All ten original roots, changed-input captured outputs/states/VJPs and
exact-zero inactive strided backing slots pass. Derived mapping coefficients
may be contiguous; these runs establish original physical-root gradients, not
a claim about every transformed native parameter's strides. Curated report:
`run/parity/002hip-dual-exponential-end-to-end-layouts.md`.

Primitive reverse now omits unrequested input/state gradient allocations,
pointers and stores, preserving internal adjoints and fragment arithmetic. The
new target mask enters source caching. Public and private source/launch
positional argument order is preserved. Empty sequences return only requested
identity state VJPs and zero parameter VJPs. Focused Slurm 1315 passes 92 tests,
including GLIF3+Alpha inference with no hidden trajectories/tape and a frozen
cell driven by a trainable encoder. The later full kernel/all-RNN/history/ODE
run 1319 reports **1158 passed, 6 skipped, 39 existing failures**; those same
39 failures reproduce in clean-HEAD controls.

The raw primitive comparison in 1317 passes all 27 model/scope/backend cases
and changed-input actual-Module replay checks at T=128/B=4/N=1024. Parameter-
only and initial-state-only scopes save approximately 2.1 MB. A real Triton
Alpha parameter-only captured regression remains: 58.208 to 74.384 us in the
repeat 1318, beyond the inherited 20%/+5us allowance. The candidate is not
accepted on memory improvement alone. Run 1316 failed because the new benchmark
supplied a cotangent to a constant final dual-exponential trace; corrected
oracle setup filters nondifferentiable outputs while retaining every final
value check. That harness error did not require a kernel/tolerance change.

NCU counter collection 1320 is unavailable (`ERR_NVGPUCTRPERM`); no host setting
was changed. The isolated reverse target comparison 1321 proves the slowdown
appears when the input-gradient sink is removed: 29.3–29.6 us with it versus
45.8–46.0 us without. All variants spill zero registers, and slower variants
use fewer registers. This is compiler-output/timing evidence, not a claimed
hardware-counter diagnosis. Reports live under
`profile/reverse-output-alpha-20261020/` and
`profile/reverse-output-alpha-targets-20261020/`.


### Reverse-output pruning gate restored

The measured launch change uses two Triton warps only for Alpha reverse with
no input VJP and a requested tau_syn VJP. All equations, FP policy and parameter
reductions remain unchanged. Inference, input/encoder VJPs, other models and
other backends retain their existing launch paths. Public/private source and
launch positional signatures remain compatible; Triton specializes its binary
cache on warp count.

Slurm 1328 passes all 27 raw complete forward/reverse correctness and changed-
input actual-Module cases, all 108 fixed 20%/+5us latency checks, and all 27
nonincreasing memory checks. Triton Alpha parameter-only captured latency is
60.288 versus 58.032 us. Slurm 1329 confirms public custom-op/autograd timing:
61.98 versus 59.78 us; eager peak decreases 8456704 to 6326784 bytes. CuPy and
TileLang public controls pass. Raw and public measurement boundaries are kept
separate in `run/parity/reverse-output-pruning.md`.

Successful privileged NCU verification in 1331 reduces long-scoreboard waiting
from the regressed 19.42 to 6.21 cycles per issued instruction and raises issue-
active slots from 4.56% to 12.37%, with zero spills. Source and PM reports plus
the reproducible NVIDIA Python-API parser are preserved in
`profile/reverse-output-alpha-two-warps-ncu-20261020/`.

Slurm 1330 passes **1072 tests, 6 skipped** across kernels, public conductance,
RNN topology, history and ODE. The separately expanded full-RNN result remains
**1158 passed, 6 skipped, 39 clean-HEAD failures** (1319). The latter is not
reclassified as passing. Family-wide parity and pending new-family golden
confirmations remain open.

### Expanded recurrent dual PSC qualification (1333–1336)

Slurm 1333 passes all 15 GLIF3 + DualExponentialPSC native dense/sparse cases
against actual btorch Modules. Twelve short cases also pass changed-input
CUDA Graph replay for every declared VJP root; larger N=256 cases are eager.
The harness uses original PSC parameter tensors and independent nonzero initial
psc/rise/decay states. Review identified None-to-zero normalization in some
comparisons; strict connectivity checks are being added and require a rerun.

Slurm 1334 passes all three T=128/N=1024 sparse cases but fails voltage-record
tolerance at N=2048 and N=4096 on every backend. Spikes remain identical; PSC
record errors remain within their existing tolerance. This is an open default
forward gate, not a full parity result. Slurm 1336 runs unfused/precise-division
diagnostics without changing default tolerances. ALIF/ELIF native whole/chunk
full-VJP matrix is running in Slurm 1335.

### Native ALIF/ELIF first matrix (1335)

Slurm 1335 passes 67 tests (19 CPU + 48 GPU) against actual unchanged Modules:
three backends, two models, reset/detach/refractory combinations, whole/chunk
state and full VJPs. See `run/parity/alif-elif-native.md`. Slurm 1337 extends
surrogates and exact integration limits; recurrent/inference/performance gates
remain open. Slurm 1336 confirms unfused precise division does not resolve the
large recurrent dual PSC voltage-record failure; default tolerance stays fixed.

### ALIF/ELIF extended gates and isolated default failure (1337–1341)

Slurm 1337 reports 113 passed and two failed: scalar `v_T` VJPs for Triton
ELIF with sigmoid and erf, respectively 1.2755e-5 and 1.6153e-5 absolute error
at fixed 1e-5/1e-4 tolerance. All backend/model zero-linear and infinite-tau
cases pass. Slurm 1340 isolates both failures to the default scalar `v_T` path:
per-neuron roots pass, and unfused precise-division scalar/per-neuron cases
pass. This is diagnostic evidence only; default support is not fully qualified,
and neither tolerance relaxation nor a new numerical helper is introduced.

Slurm 1341 passes all 30 ALIF/ELIF recurrent cases: 24 dense/sparse full-VJP
cases cover default and strict policies across all backends; six inference
cases assert the inference operator route, no training route, no private
state trajectories and two-call continuation against actual Modules. Temporal
cotangents before the chunk boundary are zero, so nonzero first-chunk input
gradients necessarily cross that boundary. The previous 18-case result1338
had strict policies only and weaker checks; use1341 for these claims.

Source review finds dual PSC native state decay collapsed to exponential
multiplication despite the actual Module using exponential Euler. A Module-order
state-charge/reverse correction is in progress; the separate exponential
readout coefficients remain part of the actual scientific Module. Large dual
network forward qualification remains open.

### Module-order dual correction and model precision policy (1343–1354)

Dual PSC state decay now follows the unchanged Module's exponential-Euler
operation graph and analytic reverse; its separate exponential readout remains
unchanged. Slurm 1343 uses strict gradient-connectivity checks and passes 29/30
recurrent cases: CuPy/TileLang pass every size through T128/N4096 with ~2x
training allocation per neuron doubling. Triton T128/N2048 still fails 16
voltage-record lanes. Precise division alone leaves one lane (1346); strict
unfused/precise passes (1350). These diagnostics do not replace the failed
default gate. Twelve standalone/continued dual tests pass1345, with later
stronger None-status assertions queued for regression.

The ALIF/ELIF shorthand policy now requests accurate division on built-in GPU
backends; explicit options retain their exact choices and existing families
keep their defaults. Slurm1347 passes all 115 primitive tests. ALIF T128/B4/N1024
full-root and changed-input graph benchmark passes all three backends1348. ELIF
long-sequence input VJP fails1349; accurate-exp diagnostic1351 and CuPy1352
alsofail, and strict policy1354 fails. This requires actual-Module analytic
reverse/operation-order diagnosis, not tolerance changes.

Slurm1344 validates the unchanged Poisson primitive's Torch-owned T-draw
schedule, contiguous/strided layouts, whole/chunk RNG continuation, refractory
draws and changed-input CUDA Graph RNG continuation. Native stochastic primitive
lowering is now in progress; no native stochastic qualification is claimed.

Slurm1353 passes 1279 tests with six skips in all kernels, conductance, topology,
history and ODE scope. It excludes the broader all RNN suite with39 independently
reproduced clean HEAD defects. This green scope does not close those defects.

### Qualification isolation and capture policy

Job1355 passes49 policy/capture checks, but its six standalone dual cases
load an in-progress stochastic ABI edit with incompatible argument positions.
These are mixed-source setup failures, not numerical qualification. Rerun
from an immutable source snapshot after the ABI integration is coherent.
Capture-version resolution now distinguishes ALIF/ELIF shorthand accurate
division from explicit approximate options, preventing stale graph replay.
The long ELIF gradient and default Triton dual large-network gates remain open.

### Frozen primitive qualification and external dual refresh (1356–1364)

1356 runs from immutable snapshot `by3_ti0a`:135 checks pass and three RNG
tests fail because their own cotangent generation advances the checked CUDA
generator. The test now uses a separate CPU generator. Snapshot `eg8qhmnj`
(1360) passes all17 stochastic primitive and six dual numerical tests; its
model-policy cases expose five spies needing the new optional ABI argument and
six snapshots of the still-in-progress recurrent uniform integration. These
are not a completed recurrent qualification. Exact source manifests are retained
in `run/parity/qualification-source-*.json`.

ELIF first and second reverse-association stages (1357/1359) still fail nine
long input-VJP lanes at unchanged tolerance. Sigmoid operation-order diagnosis
is next. Stochastic native graph-capture and recurrence remain under development.

1358 passes all12 pinned external dual backend/layout cases at T32/N16,
including original ten-root VJPs. Corrected T128/N64 scalar end-to-end mapping
passes six cases (1362), while strict Triton per-neuron mapping fails the
external output gate before timing.1361 is a runner import-path setup failure;
1362 fixes the runner without changing measured source.1363 tests a standard
accurate-exp overlay;1364 separates CuPy/TileLang per-neuron qualification.
These diagnostic jobs retain unchanged2e-5 tolerances and do not reintroduce
collapsed state decay.

### Accurate exponential diagnostic scope (1363–1365)

1363 replaces Triton `tl.exp` with standard `libdevice.exp` in a retained
out-of-tree overlay. The strict external per-neuron T128/N64 qualification,
all ten VJPs, two sequence modes and changed-input capture pass.1364 passes
four CuPy/TileLang per-neuron cases without the overlay. This supports a
scoped accurate-exp implementation for Dual PSC readout, not coefficient
precomputation or a new arithmetic helper.

1365 applies the same overlay to default Triton GLIF3 + dual recurrence.
Nine of ten cases pass; N2048 still has16 outside-tolerance voltage lanes.
The external readout issue and default recurrent rounding issue are distinct.
Retain the latter failed gate and the separately passing strict configuration.

### User-directed exponential policy

Accurate exponentials are an explicit opt-in `KernelOptions.precise_exp=False`
by default. No model shorthand enables them automatically. This supersedes the
earlier proposal to select accurate exp automatically for Dual PSC or Poisson.
Division precision and compiler contraction remain independent settings.
The accurate-exp diagnostics prove a useful qualified option, not a requirement
for every workload. Runtime/cache/capture threading is complete and validated; see the option report below.
The cell benchmark exposes `--precise-exp` and records the selected policy.

### Optional accurate exp complete (1367–1370)

`KernelOptions.precise_exp=False` is the default, and every model shorthand
keeps it disabled. Explicit True selects accurate exp independently of division
and FMA; primitive/recurrent forward/reverse source caches and capture versions
carry the flag. Slurm1370 passes all133 targeted tests, and1368 passes the
public opt-in benchmark/capture/VJP path on allthreebackends. The original
1367 TileLang division-capture error is preserved; isolated1369 and the same
full batch1370 pass without semantic changes. See `run/parity/precise-exp-option.md`.

### Stochastic native and optional-exp follow-up (1377–1395)

The explicit `precise_exp` option remains False by default. External dual
actual-Module mapping passes all six strided and per-neuron T128/N64 cases
with opt-in accurate exp (1377/1379), and six scalar T128/N256 default-exp
cases (1380). These complete-composition checks do not qualify invalid
external CUDA implementations. The default Triton GLIF+Dual N2048 voltage
case remains open.

Frozen run 1383 passes all 222 focused stochastic and ALIF/ELIF tests.
Stochastic compiled tests retain exact RNG, full-VJP and same-object graph
replay checks after correcting compiler isolation and output lifetime.
The broader 1389 run exposes a new GLIF parameter-layout validation regression
(61 failures) plus one missing frozen skill resource. Model-aware validation
and snapshot resource copying are corrected; broad rerun 1395 is pending.
Native stochastic precision is explicitly FP32; reduced-precision calls fail
before draws, while deterministic native and plain Torch paths retain their
existing dtype behavior.

Stochastic sparse recurrence smoke (1391) and T128/B4/N1024 (1392) pass all
three native backends under default fusion and approximate exp. Complete
changed-input captured inference is 1.03–1.07 ms, training 8.41–8.48 ms.
These measurements compare against actual btorch Modules and carry no fixed
external native performance gate. Captured speedups use eager Torch as the
denominator. Memory increments exclude pre-existing graph pools; zero new
allocation during replay is not zero total graph memory. See
`run/parity/poisson-recurrent-performance.md`.

ELIF spike-cotangent association now follows the actual Module's direct,
refractory, adaptation, reset order. Its regression passes, but the public
long default benchmark still fails nine input-gradient elements (1384).
No-contraction passes the original-input local trace but changed-input replay
still fails seven elements (1387/1390); adding accurate exp instead leaves
four original-input elements failing (1394). The native-state/cotangent probe
(1393) identifies a terminal one-ULP voltage adjoint difference amplified near
a zero linear coefficient. No tolerance, model equation or default math
policy is changed to hide these failures.

Corrected broad rerun 1395 reports 1386 passed, 6 skipped and one failed
CuPy accurate-exp policy-recapture case with streamCaptureInvalidated. All
previous 61 GLIF layout failures and the missing skill asset case pass. Exact
isolated case 1396 passes without source changes. The mixed-suite capture
lifecycle remains an open issue; do not claim a fully green broad regression.
