# Train a sparse RSNN with cell backends

This executable example uses a trainable encoder, independent sparse recurrent
connection, LIF/AlphaPSC dynamics and a trainable readout:

```bash
PYTHONPATH=. python examples/kernel_backend_training.py
PYTHONPATH=. python examples/kernel_backend_training.py --backend triton --device cuda
```

Choose `tilelang` or `cupy` for the corresponding optional cell implementation.
The connection still uses the existing sparse runtime, selected independently.

The example initializes FP32 state for `[T,B,N]` input, runs inference under
inference_mode, resets, then trains two contiguous chunks and calls backward once.
Voltage and PSC records add auxiliary losses to the readout objective. `eval()`
retains encoder, neuron and readout gradients. There is no implicit detach between
chunks; `functional.detach_net` explicitly ends the trajectory after backward.

For checkpointing, use the existing `chunk_size`, `unroll` and `grad_checkpoint`
options. Keep chunk sizes compatible with unroll. Test continuation and all losses
before changing receptor merge mode. Reset when switching merge modes.

Use [KernelOptions](../guides/kernel_backends.md) to choose backends, and validate
external implementations with [the OOT parity tutorial](oot_kernels.md).
