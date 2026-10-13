# Direct cell kernel benchmark

Measured on 2026-10-10 with an NVIDIA RTX 5090, Torch 2.12.0,
Triton 3.7.0, TileLang 0.1.13 and CuPy 14.2.0. All tensors use FP32.
These small runs check the benchmark workflow; timings are not a general
performance ranking. Other GPU tests were compiling concurrently, and three
steady-state repetitions are insufficient for publication-quality statistics.

Reproduce with the optional backends installed:

```bash
PYTHONPATH=. python benchmarks/models/bench_cell_backends.py \
  --model alpha --steps 32 --batch 2 --neurons 128 --repeats 3
PYTHONPATH=. python benchmarks/models/bench_cell_backends.py \
  --model lif --steps 32 --batch 2 --neurons 128 --repeats 3
```

The current eager baseline calls the existing btorch Python Modules step by
step with FP32 states and ordinary autograd. The Torch operator is checked
against those Modules as well. The tables below are historical measurements
from the earlier duplicated recurrence baseline; they have not been relabelled
as measurements of the corrected Module baseline.
The three DSL backends execute a serial time loop per neuron. Connection
propagation is excluded; sparse connection and RSNN benchmarks remain separate.

| Model | Backend | Training forward device ms | Backward device ms | Simulation graph device ms | Peak training allocation bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| AlphaPSC | Eager | 1.862 | 4.066 | 0.28315 | 147456 |
| AlphaPSC | Torch operator | 1.620 | 5.494 | 0.27629 | 203264 |
| AlphaPSC | Triton | 0.520 | 0.724 | 0.01135 | 206336 |
| AlphaPSC | TileLang | 0.713 | 1.354 | 0.00843 | 206336 |
| AlphaPSC | CuPy | 0.627 | 0.960 | 0.01198 | 206336 |
| LIF | Eager | 2.916 | 7.418 | 0.43984 | 198656 |
| LIF | Torch operator | 2.552 | 10.673 | 0.43151 | 137728 |
| LIF | Triton | 0.528 | 0.824 | 0.01467 | 143872 |
| LIF | TileLang | 0.799 | 2.123 | 0.01569 | 143872 |
| LIF | CuPy | 0.789 | 1.078 | 0.01599 | 143872 |

Host first-call times include Python startup of lazy backend components,
compilation and device synchronization. Disk caches may be warm, so these
are not cold-cache compilation measurements.

| Model | Backend | First simulation host ms | First training host ms |
| --- | --- | ---: | ---: |
| AlphaPSC | Eager | 28.94 | 25.13 |
| AlphaPSC | Torch operator | 1.98 | 29.27 |
| AlphaPSC | Triton | 534.49 | 123.86 |
| AlphaPSC | TileLang | 1577.37 | 2394.63 |
| AlphaPSC | CuPy | 264.20 | 89.27 |
| LIF | Eager | 38.79 | 23.92 |
| LIF | Torch operator | 2.75 | 35.72 |
| LIF | Triton | 506.48 | 127.86 |
| LIF | TileLang | 1550.62 | 2401.73 |
| LIF | CuPy | 123.11 | 88.44 |

GPU verification compared outputs, all state and parameter gradients, and
record-loss gradients against the earlier recurrence autograd baseline for eight models and six
surrogates. FP16 and BF16 LIF runs of 224 timesteps showed no spike divergence
with FP32 state accumulation. The paired PSC→neuron pipeline was checked for
every supported model pair, separate receptor states and merged receptor drive.

The benchmark now verifies each backend's simulation output and training
input/parameter gradients against the existing btorch Python Modules and their ordinary autograd
before timing. It additionally reports
`training_graph_device_ms` for a captured complete forward-plus-backward call;
this value is separate from the two eager training device timings. Capture
uses dedicated leaves created and warmed on the capture stream, avoiding
retained eager autograd nodes on the default stream. Both AlphaPSC and LIF
completed this workflow on all five implementations with `[T, B, N] =
[16, 2, 64]` and three repetitions. The table above records the earlier run
and does not retroactively include this new measurement.
