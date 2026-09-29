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
