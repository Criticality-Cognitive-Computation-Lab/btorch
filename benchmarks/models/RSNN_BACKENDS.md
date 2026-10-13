# Sparse receptor/delay RSNN benchmark

The tables below were measured before replacing the explicit Torch backend's
duplicated VJP with local autograd through existing Python Modules. They remain
historical measurements. The current benchmark was rerun for all five rows at
`[T, B, N] = [4, 1, 8]` with two repetitions; output/state/gradient comparisons and
simulation/training replay passed. Rerun the command below for current timings.

Measured on 2026-10-10 with an NVIDIA RTX 5090, Torch 2.11.0,
Triton 3.6.0, TileLang 0.1.13 and CuPy 14.1.1. This is a small workflow
verification run, not a performance ranking: three repetitions, warm disk
caches, and potentially concurrent GPU review work. The measured device time
includes the complete LIF + AlphaPSC sequence, independent sparse propagation,
two signed receptors and three delay bins (zero, one and two timesteps).

Reproduce after installing the optional backend packages:

```bash
PYTHONPATH=. python benchmarks/models/bench_rsnn_backends.py \
  --steps 8 --batch 2 --neurons 16 --repeats 3
```

The historical `backend="torch"` module is the ordinary-autograd baseline.
The explicit Torch row uses `KernelOptions(cell="torch")` and its handwritten
sequence VJP. All rows use the existing automatic sparse connection routing;
connection algorithms are not ported to TileLang or CuPy. Sparse edge count
is 48. Parameters include sparse weights, neuron tau/threshold and PSC
tau/gain. The benchmark first checks exact spikes, recorded voltage and PSC,
every registered final memory, and input/participating parameter gradients
against the historical module. Simulation and training graph replay pass the
same independent checks before any replay timing is reported.

| Backend | Training forward device ms | Backward device ms | Simulation graph device ms | Training graph device ms | Peak training allocation bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager | 4.6077 | 6.0460 | 0.3669 | 1.1921 | 105984 |
| torch | 2.8194 | 7.9198 | 0.2505 | 1.2118 | 51712 |
| triton | 4.4757 | 10.6480 | 0.1001 | 0.4439 | 48640 |
| tilelang | 11.1555 | 45.0005 | 0.1025 | 0.4480 | 39424 |
| cupy | 6.1762 | 14.4360 | 0.1028 | 0.4503 | 48640 |

Training graph time includes forward, record losses and the complete reverse;
the two uncaptured device times separate forward and backward. CUDA events
can include Python launch gaps in uncaptured execution. Captured trials use
dedicated leaves on the capture stream and the same reset initial state.
Timings omit optimizer updates, encoder/readout and host reset allocations.
Peak allocation is the incremental `torch.cuda.max_memory_allocated()` during
uncaptured training, not allocator-reserved memory or graph-pool memory.

| Backend | First simulation host ms | First training host ms |
| --- | ---: | ---: |
| eager | 4.22 | 11.14 |
| torch | 29.33 | 11.25 |
| triton | 142.15 | 254.94 |
| tilelang | 3960.07 | 4197.30 |
| cupy | 185.39 | 144.15 |

First-call host time includes lazy preparation, compilation and device
synchronization. Existing on-disk compiler caches may be warm, so these
numbers do not establish cold compilation latency. The independent eager
baseline was evaluated before timing. Larger networks, more repetitions and
isolated GPU execution are required for throughput or memory conclusions.
This run verifies sparse/delay integration and captured training functionality;
it does not establish megakernel performance or eliminate Python dispatch
overhead outside capture.
