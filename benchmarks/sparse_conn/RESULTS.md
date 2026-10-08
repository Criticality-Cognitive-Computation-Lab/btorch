# Sparse connection benchmark results

Latency of `SparseConnection` against the pre-refactor layer (a frozen copy
in `_legacy_baseline.py`) and a dense layer, on synthetic recurrent graphs.
Cell values are medians in milliseconds; `~` marks a noisy cell with the
fastest sample in parentheses; `-` means not run, with the reason under the
table.

## Where the numbers come from

| Section | Machine | State of the GPU | Code |
|---|---|---|---|
| 1, 2 | RTX 5090 (cccl3) | no other compute process at launch; not verified idle for the whole run (see below) | commit `b0d12c5` |
| 3 | RTX 5090 (cccl2) | shared with another job | section 3 states the commit per table |

The environment used for sections 1 and 2 has no `torch_sparse`, so the
legacy `torch_sparse` path is absent there. Section 3 keeps the earlier
comparison against it, taken on a shared GPU.

What was recorded about the GPU for sections 1 and 2
(`results/conn_rtx5090_cccl3.json`, `results/rsnn_rtx5090_cccl3.json`):

- **At launch**, `nvidia-smi` reported GPU 0 with 31.8 of 31.8 GB free, 0 %
  utilisation and no other compute process. The benchmark chose that GPU.
- **At start**, after the benchmark process had created its CUDA context:
  30.8 GB free, 0 % utilisation, and 0.5 GB of device memory outside the
  PyTorch pool of the benchmark process.
- **During the run**, two figures were sampled immediately before every
  measurement and are printed under each workload:

  | Run | Workload | Device utilisation | Device memory outside the benchmark's PyTorch pool |
  |---|---|---:|---:|
  | connection | `debug` | 0-100 % | 0.6-4.9 GB |
  | connection | `v1` | 0-100 % | 0.6-4.9 GB |
  | connection | `large` | 0-99 % | 0.6-0.6 GB |
  | connection | `xlarge` | 0-99 % | 0.6-1.7 GB |
  | recurrent network | 4,096 neurons | 0-67 % | 0.6-1.6 GB |
  | recurrent network | 100,000 neurons | 0-78 % | 0.6-0.6 GB |

  Neither figure separates the benchmark from other jobs. The utilisation is
  that of the whole device and includes the work the benchmark itself did
  just before the sample. The memory figure is `total - free - reserved by
  PyTorch in this process`; it includes the CUDA context of the benchmark
  process and any memory it holds outside the PyTorch allocator, as well as
  memory of other processes. The generated tables label the two figures "GPU
  load from other jobs" and "held by other processes", which is an upper
  bound and not a measurement of other jobs. The readings of 4.0-4.9 GB span
  18 consecutive measurements of the connection run, from `debug` / `dense` /
  batch 32 / density 0.5 to `v1` / `dense` / batch 1 / density 0.01. Which
  process held that memory was not recorded.
- **Contention.** A probe (the time per call of a trivial queued kernel) is
  taken around every sample. Only samples taken while the probe was within 2
  times the best probe of the process are used, and a measurement for which
  no such sample could be taken is flagged as throttled. The best probe was
  3.6 us (connection run) and 3.5 us (recurrent run), the probes recorded
  with the measurements lie between 3.6 and 6.6 us, and no measurement of
  either run was flagged as throttled.

In short: the card was free of other compute processes when the run started,
no measurement shows time slicing, and the recorded data cannot rule out that
another process used memory on the card during part of the `debug`, `v1` and
`xlarge` workloads.

Every run picks the GPU with the most free memory, caps this process at
`min(--max-gpu-gb, free - --gpu-margin-gb)` (20 GB here) and records a cell as
skipped when a workload does not fit.

## Summary

Connection alone (section 1):

- **4,096 neurons.** Forward 0.016 ms (batch 1) and 0.033 ms (batch 32)
  against 0.068 / 0.093 ms for the legacy `native` path and 0.022 / 0.030 ms
  for dense. Forward + backward 0.17-0.23 ms against 0.54-1.4 ms legacy and
  0.13-0.14 ms dense. A dense layer is still the fastest to train at this
  size.
- **100,000 neurons.** Forward 0.105 ms (batch 1) and 0.28 ms (batch 32)
  against 0.235 / 3.04 ms legacy. Forward + backward 0.42 ms and 1.03 ms; the
  legacy `native` path cannot run a backward pass (it needs N x N floats).
- **1,000,000 neurons, 50M edges.** Forward 0.54 ms (batch 1) and 2.1 ms
  (batch 32); forward + backward 2.4 ms and 9.8 ms, with 3.5-4.0 GB peak
  memory. The legacy path runs only the batch-1 forward (1.39 ms).
- **Push hint** (`Hints(expected_density=...)`) at 1 % spike density: forward
  0.030 ms against 0.105 ms at 100k neurons and 0.031 ms against 0.54 ms at
  1M neurons (batch 1); at batch 32, 0.103 against 0.279 ms and 1.29 against
  2.08 ms. At 4,096 neurons it makes no useful difference. At 50 % density the
  planner falls back to pull and the numbers are equal.
- **`torch.compile`** of a single connection equals eager from 100k neurons
  up. At 4,096 neurons and below it is slower (forward 0.052 against
  0.016 ms, forward + backward 0.24-0.44 against 0.17-0.23 ms): the
  registered-operator boundary costs more per call than the eager autograd
  node.

Recurrent network, 200 steps of LIF + exponential synapse (section 2):

- **4,096 neurons.** The connection is not the bottleneck: inference is
  0.23-0.25 ms per step for the new path and for dense, 0.27-0.29 ms for
  legacy; training is 0.44-0.49 ms per step for new and dense, 0.84 ms for
  legacy.
- **100,000 neurons.** Inference 0.26 ms (batch 1) and 0.47 ms (batch 32) per
  step against 0.29 / 3.27 ms legacy; training 0.66 ms and 1.37 ms per step,
  which the legacy path cannot run. The push hint gives about 10 % at batch
  32 (0.43 ms inference, 1.30 ms training) at the 2 % firing rate of this
  network.

## 1. Connection benchmark and 2. recurrent network


### Connection benchmark (conn_rtx5090_cccl3.json)

- gpu: NVIDIA GeForce RTX 5090
- cpu: AMD EPYC 9965 192-Core Processor
- torch: 2.11.0
- cuda: 12.9
- triton: 3.6.0
- python: 3.12.14
- GPU 0 chosen (31.8 of 31.8 GB free, 0 % util, 0 other compute process(es)); memory cap of the benchmark process: 20.0 GB
- best contention probe of the run: 3.6 us
- GPU at start: 0.5 GB held by other processes, 30.8 GB free, utilisation 0 %

#### `debug`: 256 neurons, 3,987 edges

**forward (ms)**

| batch | density | legacy[native] | dense | new[eager] | new[compile] | new[push-hint] |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 | 0.043 | 0.012 | 0.016 | 0.072 | 0.020 |
| 1 | 0.5 | 0.047 | 0.012 | 0.016 | 0.051 | 0.017 (pull) |
| 32 | 0.01 | 0.045 | 0.010 | 0.027 | 0.074 | 0.020 |
| 32 | 0.5 | 0.048 | 0.011 | 0.028 | 0.075 | 0.023 (pull) |

**forward + backward (ms)**

| batch | density | legacy[native] | dense | new[eager] | new[compile] | new[push-hint] |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 | 0.442 | 0.114 | 0.111 | 0.355~ (0.242) | 0.121 |
| 1 | 0.5 | 0.419 | 0.122 | 0.186 | 0.349 | 0.201 |
| 32 | 0.01 | 0.453 | 0.089 | 0.208 | 0.408 | 0.200 |
| 32 | 0.5 | 0.449 | 0.126 | 0.205 | 0.392 | 0.213 |

**peak GPU memory, forward / forward + backward (MB)**

| batch | density | legacy[native] | dense | new[eager] | new[compile] | new[push-hint] |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 | 0 / 9 | 17 / 17 | 17 / 17 | 17 / 17 | 17 / 17 |
| 1 | 0.5 | 8 / 9 | 17 / 17 | 17 / 17 | 17 / 17 | 17 / 17 |
| 32 | 0.01 | 8 / 9 | 17 / 17 | 17 / 17 | 17 / 17 | 17 / 17 |
| 32 | 0.5 | 8 / 9 | 17 / 17 | 17 / 17 | 17 / 17 | 17 / 17 |

**build / preprocessing and compilation (ms)**

| implementation | build | compile fwd (B=1 / B=32) | compile fwd+bwd |
|---|---:|---:|---:|
| legacy[native] | 6 | - | - |
| dense | 2 | - | - |
| new[eager] | 24 | - | - |
| new[compile] | 24 | 414 / 283 | 105 / 100 |
| new[push-hint] | 24 | - | - |

GPU load from other jobs while this workload ran: utilisation 0-100 %, 0.6-4.9 GB held by other processes.

#### `v1`: 4,096 neurons, 404,681 edges

**forward (ms)**

| batch | density | legacy[native] | dense | new[eager] | new[compile] | new[push-hint] |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 | 0.068 | 0.022 | 0.016 | 0.052 | 0.020 |
| 1 | 0.5 | 0.069 | 0.045 | 0.015 | 0.051 | 0.016 (pull) |
| 32 | 0.01 | 0.093 | 0.030 | 0.033 | 0.066 | 0.029 |
| 32 | 0.5 | 0.094 | 0.031 | 0.024 | 0.092~ (0.065) | 0.024 (pull) |

**forward + backward (ms)**

| batch | density | legacy[native] | dense | new[eager] | new[compile] | new[push-hint] |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 | 1.39~ (1.06) | 0.130 | 0.170~ (0.115) | 0.242 | 0.206 |
| 1 | 0.5 | 1.34~ (0.521) | 0.129 | 0.195 | 0.371 | 0.197 |
| 32 | 0.01 | 0.552 | 0.134 | 0.185 | 0.439 | 0.219~ (0.138) |
| 32 | 0.5 | 0.538 | 0.140 | 0.232 | 0.423 | 0.176 |

**peak GPU memory, forward / forward + backward (MB)**

| batch | density | legacy[native] | dense | new[eager] | new[compile] | new[push-hint] |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 | 26 / 97 | 88 / 152 | 46 / 52 | 46 / 52 | 47 / 55 |
| 1 | 0.5 | 26 / 97 | 88 / 152 | 50 / 52 | 50 / 52 | 53 / 55 |
| 32 | 0.01 | 28 / 98 | 89 / 153 | 52 / 54 | 52 / 54 | 54 / 57 |
| 32 | 0.5 | 28 / 98 | 89 / 153 | 52 / 54 | 52 / 54 | 55 / 57 |

**build / preprocessing and compilation (ms)**

| implementation | build | compile fwd (B=1 / B=32) | compile fwd+bwd |
|---|---:|---:|---:|
| legacy[native] | 24 | - | - |
| dense | 13 | - | - |
| new[eager] | 58 | - | - |
| new[compile] | 44 | 59 / 86 | 78 / 107 |
| new[push-hint] | 57 | - | - |

GPU load from other jobs while this workload ran: utilisation 0-100 %, 0.6-4.9 GB held by other processes.

#### `large`: 100,000 neurons, 9,995,065 edges

**forward (ms)**

| batch | density | legacy[native] | new[eager] | new[compile] | new[push-hint] |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 | 0.235 | 0.105 | 0.097 | 0.030 |
| 1 | 0.5 | 0.230 | 0.101 | 0.097 | 0.097 (pull) |
| 32 | 0.01 | 3.04 | 0.279 | 0.279 | 0.103 |
| 32 | 0.5 | 3.04 | 0.286 | 0.284 | 0.287 (pull) |

**forward + backward (ms)**

| batch | density | legacy[native] | new[eager] | new[compile] | new[push-hint] |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 | - | 0.420 | 0.410 | 0.330 |
| 1 | 0.5 | - | 0.411 | 0.409 | 0.403 |
| 32 | 0.01 | - | 1.03 | 1.02 | 0.829 |
| 32 | 0.5 | - | 1.05 | 1.05 | 1.05 |

**peak GPU memory, forward / forward + backward (MB)**

| batch | density | legacy[native] | new[eager] | new[compile] | new[push-hint] |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 | 247 / - | 743 / 897 | 743 / 897 | 782 / 974 |
| 1 | 0.5 | 247 / - | 858 / 897 | 858 / 897 | 936 / 974 |
| 32 | 0.01 | 294 / - | 894 / 944 | 894 / 944 | 959 / 1022 |
| 32 | 0.5 | 294 / - | 894 / 944 | 894 / 944 | 971 / 1022 |

Not run:

- legacy[native] B=1, legacy[native] B=32: skipped: backward needs N x N floats (37 GB), exceeds memory cap

**build / preprocessing and compilation (ms)**

| implementation | build | compile fwd (B=1 / B=32) | compile fwd+bwd |
|---|---:|---:|---:|
| legacy[native] | 1058 | - | - |
| new[eager] | 743 | - | - |
| new[compile] | 805 | 76 / 72 | 79 / 103 |
| new[push-hint] | 779 | - | - |

GPU load from other jobs while this workload ran: utilisation 0-99 %, 0.6-0.6 GB held by other processes.

#### `xlarge`: 1,000,000 neurons, 49,998,792 edges

**forward (ms)**

| batch | density | legacy[native] | new[eager] | new[compile] | new[push-hint] |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 | 1.39 | 0.536 | 0.510 | 0.031 |
| 1 | 0.5 | 1.39 | 0.510 | 0.512 | 0.514 (pull) |
| 32 | 0.01 | - | 2.08 | 2.09 | 1.29 |
| 32 | 0.5 | - | 2.11 | 2.11 | 2.11 (pull) |

**forward + backward (ms)**

| batch | density | legacy[native] | new[eager] | new[compile] | new[push-hint] |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 | - | 2.40 | 2.41 | 1.94 |
| 1 | 0.5 | - | 2.41 | 2.42 | 2.41 |
| 32 | 0.01 | - | 9.75 | 9.76 | 8.95 |
| 32 | 0.5 | - | 9.79 | 9.83 | 9.78 |

**peak GPU memory, forward / forward + backward (MB)**

| batch | density | legacy[native] | new[eager] | new[compile] | new[push-hint] |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 | 1180 / - | 2713 / 3484 | 2713 / 3484 | 2912 / 3873 |
| 1 | 0.5 | 1180 / - | 3289 / 3484 | 3289 / 3484 | 3682 / 3877 |
| 32 | 0.01 | - | 3648 / 3961 | 3648 / 3961 | 3919 / 4353 |
| 32 | 0.5 | - | 3648 / 3961 | 3648 / 3961 | 4041 / 4353 |

Not run:

- legacy[native] B=32: legacy[native]: needs ~21.9 GB, exceeds memory cap 20.0 GB
- legacy[native] B=1: skipped: backward needs N x N floats (3725 GB), exceeds memory cap

**build / preprocessing and compilation (ms)**

| implementation | build | compile fwd (B=1 / B=32) | compile fwd+bwd |
|---|---:|---:|---:|
| legacy[native] | 7270 | - | - |
| new[eager] | 3795 | - | - |
| new[compile] | 4490 | 93 / 81 | 85 / 110 |
| new[push-hint] | 4338 | - | - |

GPU load from other jobs while this workload ran: utilisation 0-99 %, 0.6-1.7 GB held by other processes.

### RSNN benchmark (rsnn_rtx5090_cccl3.json)

- gpu: NVIDIA GeForce RTX 5090
- cpu: AMD EPYC 9965 192-Core Processor
- torch: 2.11.0
- cuda: 12.9
- triton: 3.6.0
- python: 3.12.14
- GPU 0 chosen (31.8 of 31.8 GB free, 0 % util, 0 other compute process(es)); memory cap of the benchmark process: 20.0 GB
- best contention probe of the run: 3.5 us
- GPU at start: 0.5 GB held by other processes, 30.8 GB free, utilisation 0 %

#### 4,096 neurons, 404,681 recurrent edges

| batch | phase | legacy[native] | new[default] | new[push-hint] | dense | steps | spikes/step |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | infer | 0.273 | 0.230 | 0.242 | 0.226 | 200 | 2.2-2.2 % |
| 1 | train | 0.835 | 0.451 | 0.453 | 0.449 | 200 | 2.2-2.2 % |
| 32 | infer | 0.287 | 0.254 | 0.249 | 0.229 | 200 | 2.3-2.3 % |
| 32 | train | 0.845 | 0.489 | 0.484 | 0.495 | 200 | 2.3-2.3 % |

Peak GPU memory of the training phase (MB):

| batch | legacy[native] | new[default] | new[push-hint] | dense |
|---:|---:|---:|---:|---:|
| 1 | 110 | 64 | 67 | 299 |
| 32 | 716 | 742 | 746 | 881 |

#### 100,000 neurons, 9,995,065 recurrent edges

| batch | phase | legacy[native] | new[default] | new[push-hint] | steps | spikes/step |
|---:|---|---:|---:|---:|---:|---:|
| 1 | infer | 0.291 | 0.264 | 0.265 | 200 | 2.2-2.2 % |
| 1 | train | - | 0.663 | 0.660 | 200 | 2.2-2.2 % |
| 32 | infer | 3.26 | 0.472 | 0.427 | 200 | 2.2-2.2 % |
| 32 | train | - | 1.37 | 1.30 | 79 | 1.9-1.9 % |

Peak GPU memory of the training phase (MB):

| batch | legacy[native] | new[default] | new[push-hint] |
|---:|---:|---:|---:|
| 1 | - | 1202 | 1279 |
| 32 | - | 7417 | 7494 |

Not run:

- legacy[native] B=1 train, legacy[native] B=32 train: skipped: exceeds memory cap (out of memory inside this process)

## 3. Earlier comparison against `torch_sparse` (shared GPU)

Taken on cccl2 while another job used the same card, so sub-0.1 ms values are
subject to time slicing. Kept because it is the only comparison against the
legacy `torch_sparse` path.

4,096 neurons, commit `02c653e` (`results/conn_rtx5090_small.json`):

| batch | legacy[torch_sparse] fwd | new[eager] fwd | legacy[torch_sparse] fwd+bwd | new[eager] fwd+bwd |
|---:|---:|---:|---:|---:|
| 1 | 0.058 | 0.023 | 0.278 | 0.250 |
| 32 | 0.186 | 0.052 | 0.655 | 0.297 |

100,000 neurons, commit `99dbbba`, before the per-call overhead work
(`results/final_rtx5090.json`):

| batch | legacy[torch_sparse] fwd | new[eager] fwd | legacy[torch_sparse] fwd+bwd | new[eager] fwd+bwd | legacy peak (MB) | new peak (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.286 | 0.110 | 0.594 | 0.659 | 364 | 710 |
| 32 | 5.60 | 0.285 | 17.4 | 1.06 | 5141 | 757 |

The batch-1 forward + backward at 100,000 neurons, where the new path was
slower than `torch_sparse` in that run, now measures 0.42 ms in the run of
section 1. The two runs are on different machines, so this is not a
like-for-like comparison with the 0.594 ms above.

## 4. Kernel-level measurements

Standalone kernel timings and the push/pull crossover densities behind the
planner thresholds are in `results/kernels_pull_rtx5090.json` and
`results/kernels_push_rtx5090.json` (`bench_kernels_pull.py`,
`bench_kernels_push.py`), taken on a shared GPU with the first version of the
kernels.

## 5. Reproduce

```bash
python benchmarks/sparse_conn/bench_sparse_conn.py --device cuda --compile \
    --workloads debug v1 large xlarge --batch 1 32 --density 0.01 0.5 \
    --out benchmarks/sparse_conn/results/conn.json
python benchmarks/sparse_conn/bench_rsnn.py --device cuda \
    --neurons 4096 100000 --batch 1 32 \
    --out benchmarks/sparse_conn/results/rsnn.json
python benchmarks/sparse_conn/make_results.py \
    --conn benchmarks/sparse_conn/results/conn.json \
    --rsnn benchmarks/sparse_conn/results/rsnn.json
```

Not measured: a like-for-like `torch_sparse` comparison on an unshared GPU, and
anything on a real connectome (all graphs here are synthetic, uniform
in-degree).
