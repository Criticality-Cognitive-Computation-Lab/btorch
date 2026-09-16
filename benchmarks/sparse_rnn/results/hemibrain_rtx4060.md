# Native vs Triton sparse benchmark on RTX 4060 Laptop

## Workload

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB, compute capability 8.9.
- Environment: `ml-py312`, PyTorch 2.10.0, CUDA 12.9, Triton 3.6.0.
- Dataset: real Hemibrain connectome, 21,739 neurons and 3,550,403 edges
  (mean fan-out 163.32).
- Shape: 128 time steps, batch size one, float32 inference.
- Timing: CUDA Graph replay, 10 warmups and 30 CUDA-event samples; values below
  are medians.
- Native baseline: the original `torch.sparse.mm` backend.
- Triton: default full configuration (`reorder`, `block`, and `hash` enabled).
- Packed-weight gathering and workspace allocation happen once, before graph
  capture. One-off construction/preprocessing and state reset are outside
  steady-state timings.

The Hemibrain edge list is unsigned. Its mean absolute recurrent fan-out
strength is normalized to 0.01 so that the closed-loop network remains
subcritical and measured SNN firing rate follows the requested sweep.

## Pure SpMSpV

| Spike rate | Native (us/step) | Triton (us/step) | Speedup |
|---:|---:|---:|---:|
| 0.1% | 396.93 | 46.25 | 8.58x |
| 0.2% | 396.94 | 38.19 | 10.39x |
| 0.5% | 396.90 | 55.34 | 7.17x |
| 1.0% | 396.98 | 75.33 | 5.27x |
| 2.0% | 396.90 | 118.08 | 3.36x |
| 5.0% | 396.91 | 247.03 | 1.61x |
| 10.0% | 396.88 | 445.70 | 0.89x |

The event-driven path wins through 5% activity on this graph. At 10%, task
packing, hash-table work, and atomics exceed the fixed-cost native SpMV.

## End-to-end recurrent SNN

This path runs the project's existing `RecurrentNN`, `LIF`, and
`ExponentialPSC` modules. External impulses control activity; recurrent spikes
feed the same Hemibrain connection into the PSC state for the following step.

| Requested | Measured SNN | Native (us/step) | Triton (us/step) | Speedup |
|---:|---:|---:|---:|---:|
| 0.1% | 0.1010% | 434.86 | 72.56 | 5.99x |
| 0.2% | 0.1983% | 435.30 | 75.02 | 5.80x |
| 0.5% | 0.4980% | 434.67 | 88.82 | 4.89x |
| 1.0% | 1.0014% | 434.91 | 108.74 | 4.00x |
| 2.0% | 2.0008% | 434.69 | 152.05 | 2.86x |
| 5.0% | 5.0123% | 434.99 | 279.18 | 1.56x |
| 10.0% | 10.0201% | 434.46 | 480.68 | 0.90x |

CUDA Graph capture removes the per-timestep Python/kernel-launch penalty, so
the end-to-end result now follows the operator crossover: large gains at low
activity, diminishing gains toward 5%, and a loss at 10%. The replay still
includes public-wrapper input/state copies and output cloning.

## One-off cost and correctness

For the pure-operator construction, native took 475.54 ms and Triton took
1,684.31 ms; Triton preparation then took 15.11 ms. The additional Triton
setup cost is roughly 1.22 s. Depending on spike rate, the pure-operator gain
amortizes after approximately 3,400--8,200 steps at rates up to 5%; it does not
amortize at 10% in this configuration.

All SpMSpV comparisons passed `atol=1e-6, rtol=1e-5`; the largest absolute
difference was 1.34e-7. Native and Triton produced identical spike sequences
for every end-to-end rate.

Raw measurements are in `hemibrain_rtx4060.csv`; runtime metadata and the full
Triton configuration are in `hemibrain_rtx4060.json`.
