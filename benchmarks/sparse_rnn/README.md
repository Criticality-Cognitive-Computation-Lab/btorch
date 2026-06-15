# Sparse RNN Profiling

This folder contains a simple profiler entry point for sparse recurrent
connections. The script records a Chrome trace, memory stacks for a flamegraph,
and summary tables.

## Usage

```bash
python benchmark/sparse_rnn/profile_sparse_rnn.py
```

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
