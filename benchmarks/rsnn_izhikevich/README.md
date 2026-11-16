# Izhikevich RSNN: cold and warm execution

Run from the repository root in the btorch environment:

```bash
python -m benchmarks.rsnn_izhikevich.run
```

The default matrix uses 4096 and 60,000 neurons, the existing example's edge
density (exactly 33,072,348 edges at 60K), seed 42, two heterogeneous
DualExponentialPSC receptors, 1 ms timesteps, 5000 steps and fixed input scale
0.6640625. The large static network's drive was calibrated to approximately
5 Hz. Each result reports its actual firing rate; STDP can change activity.
The 4096-neuron case is the small launch-bound comparison.

The models come directly from `examples.rsnn_izhikevich` (BPTT) and
`examples.rsnn_izhikevich_stdp` (static inference and online learning).
No benchmark-specific neuron, PSC, connectivity or learning equations are used.
All strategies retain the examples' voltage, adaptation and PSC recordings.

## Select comparisons

Torch native is measured **both with and without CUDA Graph**, once per
population/mode. Native Triton and TileLang are measured separately. Available
strategies are `torch-eager`, `torch-graph`, `push-eager`, `push-graph` and
`helion-graph`; push and HELION both explicitly consume spike lists.

```bash
# Matched static / STDP comparison, including both Torch baselines.
python -m benchmarks.rsnn_izhikevich.run \
    --modes inference stdp --backends triton tilelang \
    --strategies torch-eager torch-graph push-graph helion-graph

# Full cross-chunk BPTT, not truncated learning.
python -m benchmarks.rsnn_izhikevich.run --modes training \
    --strategies torch-eager torch-graph push-eager push-graph

# Quick end-to-end smoke run; a remainder exercises RecurrentNN's chunk policy.
python -m benchmarks.rsnn_izhikevich.run --populations 128 \
    --steps 9 --chunk-size 4 --repeats 1

# CPU reference only.
python -m benchmarks.rsnn_izhikevich.run --populations 64 --steps 8 \
    --chunk-size 4 --repeats 1 --device cpu --strategies torch-eager
```

Training uses the example's sparse surrogate, checkpointing and rate/voltage
loss, followed by one SGD update with no momentum. `--cpu-offload` and
`--no-grad-checkpoint` select the existing RecurrentNN mechanisms. Online STDP
runs under `no_grad`, updates its weights throughout the sequence and does not
perform backward. HELION BPTT and Torch-native manual-Graph BPTT are explicitly
reported as unsupported: the existing bounded forward/reverse graph implementation
requires native neuron/PSC kernels. Torch Graph inference and STDP are measured;
Torch eager BPTT remains the training reference. The script does not silently
replace Torch with Inductor. Failures,
including allocation failures and timeouts, remain visible and make the script
exit nonzero; they never trigger a different implementation. Large Torch STDP
and full recorded BPTT can be expensive; use `--timeout` or select cases to fit
the allocated GPU and execution budget.

## Timing contract

Each case runs in a **fresh subprocess**, retaining disk compiler caches.
Construction and input allocation are measured separately. The first timed
call is cold: it includes JIT/preparation, bounded chunk warmup, capture and
simulation. No full-sequence warmup occurs before it. Graphs use the existing
RecurrentNN scheduler, with `--chunk-size 200 --chunk-mode split` by default;
`--chunk-mode chunk` is also available. Remainder handling follows the selected
RecurrentNN execution path. Forward capture counts are recorded after each call
so recapture is visible; this counter excludes native reverse captures.

Warm time is the median of three complete subsequent calls, including Python
submission and output assembly. Training includes zeroing gradients, loss,
backward and the optimizer update. Timers use synchronized wall time, not sums
of kernel durations. Every trial starts with identical parameters, weights,
neuron/PSC states and learning traces. Reset and finite-value/rate validation
are outside the timers; graphs are retained. The input uses one stored current
row expanded over time, as in the examples. CUDA allocated-memory peaks include
model, snapshot, recording and graph storage; they are not device DRAM traffic.

Raw results, individual warm samples, failure logs and a Markdown table are
saved under ignored `profile/rsnn_izhikevich/<timestamp>/` by default. Choose a
new `--output` directory for every run; existing results are never overwritten.
The summary reports matched HELION/push graph ratios and their geometric mean,
including slowdowns. Cold times remain separate. Saving results is untimed.
Atomic push propagation is not bitwise reproducible. Finite-value checks are
health checks; numerical parity is covered by the repository's model and
plasticity tests, rather than claimed from a timing run.

## Measured reference: RTX 5090

Measured on 2026-11-16 using the implementation at `21c6d71`: FP32, batch 1,
**200 timesteps** at 1 ms, split chunks of 200, seed 42 and input scale
0.6640625. These are synchronized end-to-end **warm milliseconds**, the median
of three complete calls after the first cold call. Construction, reset,
validation and disk writes are excluded. Disk compiler caches were retained.
The script defaults to 5000 steps; these short-run measurements must not be
interpreted as 5000-step timings.

The 4096-neuron circuit has 154,128 edges and mean activity 5.280 Hz; the
60,000-neuron circuit has 33,072,348 edges and mean activity 4.966 Hz. Both use
the example's two heterogeneous DualExponentialPSC receptors and full neuron/PSC
recordings. Native push and HELION explicitly use spike lists. Training includes
loss, full backward and one SGD step; STDP runs online without backward.

| N | Mode | Torch eager | Torch Graph | Triton push Graph | Triton HELION Graph |
|---:|---|---:|---:|---:|---:|
| 4096 | Inference | 133.440 | 19.736 | 1.492 | 2.058 |
| 4096 | STDP | 183.591 | 27.589 | 3.673 | 3.224 |
| 4096 | Training | 427.971 | Unsupported | 7.673 | Unsupported |
| 60000 | Inference | 133.164 | 88.539 | 4.518 | 8.656 |
| 60000 | STDP | 890.573 | 885.180 | 19.215 | 25.846 |
| 60000 | Training | 812.081 | Unsupported | 32.463 | Unsupported |

Torch-native manual Graph BPTT and HELION BPTT are unsupported in this
implementation. Their entries are not timings or silent fallbacks. HELION
improves the small STDP case by 1.139x over push Graph but regresses the large
STDP case to 0.743x; fusion does not provide a universal speedup.

Reproduce the table's configurations with:

```bash
python -m benchmarks.rsnn_izhikevich.run --steps 200 --repeats 3 \
    --backends triton \
    --strategies torch-eager torch-graph push-graph helion-graph
```

Cold timings, individual warm samples, memory peaks and capture counts are
retained in the local, untracked report
`profile/rsnn-benchmark-20261116/COMPARISON.md` and its referenced JSON results.
The small and large measurements are also recorded in the untracked append-only
STDP performance journal.
