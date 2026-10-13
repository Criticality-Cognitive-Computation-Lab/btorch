# Register and validate a generated cell kernel

Run the executable example from the repository root:

```bash
PYTHONPATH=. python examples/oot_cell_backend.py
PYTHONPATH=. python examples/oot_cell_backend.py --backend triton --device cuda --source /tmp/lif_kernel.py --capture
```

The CPU run needs no optional GPU package. Triton/TileLang/CuPy runs use real
shared-template generated kernels. `--source` writes inspectable code before
validation. `--compile` additionally checks static fullgraph forward and backward.

The example resolves a built-in implementation, registers it under an external
package name, and validates a nonzero initial-state trajectory with voltage and
charged-voltage record losses. The normal `LIF(..., backend="tutorial_cell")` API
then trains an upstream scale in eval mode. A second parity case checks AlphaPSC
state and record gradients, and a sparse RSNN combines both registered models
with the existing connection kernel and an auxiliary state loss. This is also a useful starting point
for a package that changes scheduling while preserving btorch model semantics.

## Diagnose a derivative mistake

A successful forward does not establish training correctness. In your external
package, try replacing the VJP with one that zeroes the initial-state gradient:

```python
def broken_backward(*args, **kwargs):
    grad_x, grad_states, grad_parameters = implementation.backward(*args, **kwargs)
    grad_states = tuple(None if g is None else torch.zeros_like(g) for g in grad_states)
    return grad_x, grad_states, grad_parameters
```

Register that implementation under a separate name and run the same `ParityCase`.
`report.assert_close()` rejects it with `grad_v` and the first failing coordinate,
even when spike outputs are unchanged. The repository's
`tests/models/kernels/test_oot.py` executes this regression.

Inspect skipped capability checks before publishing a package. Test the shapes,
precision, surrogate/reset options and record losses you advertise. See the
[callback and generation contract](../guides/oot_kernels.md).
