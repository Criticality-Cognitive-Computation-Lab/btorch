# Sparse RNN Profiling

This folder contains a simple profiler entry point for sparse recurrent
connections. The script records a Chrome trace, memory stacks for a flamegraph,
and summary tables.

## Usage

```bash
python benchmark/sparse_rnn/profile_sparse_rnn.py
```

To compare the native sparse path with the event-driven Triton backend on the
local Hemibrain connectome:

```bash
micromamba run -n ml-py312 \
  python benchmarks/sparse_rnn/benchmark_triton_spmspv.py
```

To reproduce the Praxist fused CUDA interval-RSNN result without downloading
data:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/fanqixuan/micromamba/envs/ml-py312/bin/python \
  benchmarks/sparse_rnn/reproduce_praxist_cuda_rsnn.py \
  --mode smoke --output /tmp/praxist_cuda_rsnn.jsonl
```

Use `--mode aligned` for FlyWire, Hemibrain, MICrONS, and multiarea, or
`--mode complete` for every locally cataloged graph at least as large as
FlyWire plus the mandatory connectomes. The complete protocol evaluates
whole-network activity rates `{1, 3, 10, 30}` Hz and horizons
`{8, 128, 2048}`. Results contain synchronized CUDA-event latency, per-step
latency, speedup against the unchanged Torch CSR recurrence, final-state
error, reference-vs-reference error, and registers per thread.

The fused kernel accepts a cyclic contiguous active interval. It is not a
general arbitrary-pattern SpMSpV implementation. `CyclicSparseRSNNCuda`
launches asynchronously on PyTorch's current stream and reuses output buffers
by default; pass `clone_outputs=True` when retaining multiple results.

This benchmark scans input spike rates from 0.1% to 10% at batch size one. It
measures both 128 independent SpMSpV steps and the existing
`RecurrentNN(LIF, ExponentialPSC)` path. Both loops are captured as CUDA Graphs
before timing. The Triton packed weights and workspace are prepared once
outside graph capture and reused by graph warmup, capture, and every replay.
Connection construction and preparation are reported separately from 10
warmups and 30 steady-state CUDA-event samples. The default dataset path can be
replaced with `--dataset`.

To reproduce the larger recurrent proxy without CUDA Graphs or scan, run each
backend in a separate process:

```bash
micromamba run -n ml-py312 \
  python benchmarks/sparse_rnn/benchmark_compile_triton_proxy.py \
  --backend native \
  --output benchmarks/sparse_rnn/results/rtx5090_compile_proxy_native.json

micromamba run -n ml-py312 \
  python benchmarks/sparse_rnn/benchmark_compile_triton_proxy.py \
  --backend triton \
  --output benchmarks/sparse_rnn/results/rtx5090_compile_proxy_triton.json
```

The defaults use 32,768 neurons, 512 sampled edges per neuron for each of two
recurrent projections, 1,000 time steps, 0.3% activity, and unroll factor 8.
The checked-in JSON files contain the raw RTX 5090 measurements used by the
English and Chinese tutorials.

By default, outputs land in `fig/benchmark/...` with a timestamped folder.

Optional flags:

- `--device cpu|cuda`
- `--backend native|torch_sparse`
- `--grad-checkpoint`
- `--seq-len`, `--batch-size`, `--input-size`, `--hidden-size`
- `--density` (sparsity of the recurrent weight)
- `--wait-steps`, `--warmup-steps`, `--active-steps`, `--repeat`

## Outputs

- `trace_*.json`: Chrome trace for the profiler UI (Chrome tracing or TensorBoard).
- `stacks_*_memory.txt`: collapsed stacks for memory flamegraphs.
- `summary.txt`: time and memory tables from `torch.profiler`.
