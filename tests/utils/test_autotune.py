"""Exercise trial isolation and real prepared execution, not option
plumbing."""

import pytest
import torch

from btorch.utils.autotune import scratch_arguments


def test_trials_preserve_overlapping_typed_strided_views_and_reset():
    # A timestep view must not clone the entire sequence behind it. Two
    # overlapping arguments must still observe each other's writes, including
    # when an integer view shares the same bytes as a floating state buffer.
    sequence = torch.arange(10000, dtype=torch.float32)
    state = sequence[401:417:2]
    overlapping = sequence[405:413]
    bits = sequence.view(torch.int32)[405:413]
    coefficient = torch.tensor([0.5])
    original = sequence.clone()
    trial, reset = scratch_arguments((state, overlapping, bits, coefficient), (0,))
    assert trial[0].stride() == state.stride()
    assert trial[0].untyped_storage().nbytes() == 15 * state.element_size()
    trial[0].add_(10)
    torch.testing.assert_close(trial[1][::2], overlapping[::2] + 10)
    assert torch.equal(trial[2], trial[1].view(torch.int32))
    torch.testing.assert_close(sequence, original, rtol=0, atol=0)
    reset()
    for actual, expected in zip(trial[:3], (state, overlapping, bits)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Immutable coefficient storage is retained, not copied for every trial.
    assert trial[3] is coefficient


def test_trial_reset_preserves_disjoint_output_and_empty_storage():
    # Atomic consumers need identical initial output on every timed repeat.
    # Empty receptor/record destinations are legal and must remain empty.
    output = torch.full((2, 4), 3.0).t()
    empty = torch.empty(0)
    trial, reset = scratch_arguments((output, empty), (0, 1))
    for _ in range(3):
        reset()
        trial[0].add_(2)
        torch.testing.assert_close(trial[0], output + 2)
    torch.testing.assert_close(output, torch.full_like(output, 3))
    assert trial[1].numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_native_triton_trials_do_not_advance_state_and_capture_uses_winner(monkeypatch):
    import gc
    import weakref

    triton = pytest.importorskip("triton")
    import triton.language as tl

    from btorch.utils import autotune
    from btorch.utils.triton import prepare_launch

    @triton.jit
    def update(source, destination, N: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(source + index, index < N, 0)
        previous = tl.load(destination + index, index < N, 0)
        tl.store(destination + index, previous + value, index < N)

    source = torch.linspace(-1, 1, 8193, device="cuda")
    state = torch.full_like(source, 2)
    arguments = [source, state]
    trial_refs = []
    original_scratch = autotune.scratch_arguments

    def observe_scratch(args, writes):
        scratch, reset = original_scratch(args, writes)
        trial_refs.extend(weakref.ref(scratch[i]) for i in writes)
        return scratch, reset

    monkeypatch.setattr(autotune, "scratch_arguments", observe_scratch)
    launch = prepare_launch(
        update,
        arguments,
        (triton.cdiv(source.numel(), 128),),
        (("N", source.numel()), ("BLOCK", 128)),
        writes=(1,),
        work=source.numel(),
    )
    # Preparation may run many candidates; the live state has not advanced.
    torch.testing.assert_close(state, torch.full_like(state, 2), rtol=0, atol=0)
    launch.submit()
    torch.testing.assert_close(state, 2 + source, rtol=0, atol=0)
    gc.collect()
    assert trial_refs and all(ref() is None for ref in trial_refs)

    def no_new_scratch(*_, **__):
        pytest.fail("native tuner cache hit allocated new trial storage")

    monkeypatch.setattr(autotune, "scratch_arguments", no_new_scratch)
    monkeypatch.setattr(autotune, "_ranges", no_new_scratch)
    prepare_launch(
        update,
        arguments,
        (triton.cdiv(source.numel(), 128),),
        (("N", source.numel()), ("BLOCK", 128)),
        writes=(1,),
        work=source.numel(),
    )
    # Repreparing a new chunk reuses the native Config without rerunning a
    # candidate or retaining the earlier chunk's scratch storage.
    torch.testing.assert_close(state, 2 + source, rtol=0, atol=0)

    def forbidden(*_, **__):
        pytest.fail("prepared submission re-entered autotuning")

    monkeypatch.setattr(autotune, "triton_config", forbidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch.submit()
    state.fill_(2)
    graph.replay()
    torch.testing.assert_close(state, 2 + source, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ("triton", "cupy", "tilelang"))
def test_tuned_sequence_keeps_full_state_and_parameter_vjp(backend):
    from btorch.models.kernels import NeuronSpec
    from tests.models.kernels.helpers import (
        assert_module_sequence,
        require_cuda_backend,
    )

    require_cuda_backend(backend)
    # A ragged population exceeds the tuning threshold. Nonuniform voltage
    # cotangents detect a trial that accidentally advanced live state, while
    # trainable broadcast coefficients exercise the reverse kernel's reductions.
    generator = torch.Generator(device="cuda").manual_seed(19)
    x = torch.rand(3, 1, 4099, generator=generator, device="cuda").requires_grad_()
    state = (torch.full((1, 4099), 0.1, device="cuda", requires_grad=True),)
    parameters = tuple(
        torch.tensor(value, device="cuda", requires_grad=True)
        for value in (3.0, 1.0, 0.0, 1.2)
    )
    # Template order, not a second model factory, is authoritative.
    from btorch.models.kernels.templates import get_template

    values = {
        "tau": parameters[0],
        "v_threshold": parameters[1],
        "v_reset": parameters[2],
        "c_m": parameters[3],
    }
    spec = NeuronSpec("lif", dt=1)
    ordered = tuple(values[name] for name in get_template(spec).parameter_names)
    assert_module_sequence(backend, spec, x, state, ordered, ("v",))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cupy_native_profiler_does_not_retain_invocation_storage(monkeypatch):
    import importlib
    import weakref

    cupy = pytest.importorskip("cupy")
    from btorch.models.kernels.backends.renderers import CuPyRuntime
    from btorch.utils import autotune

    # Real strided, accumulating state makes trial isolation necessary.
    # Simulate an optional dependency retaining an import traceback. Native
    # profiler initialization must not attach that traceback to this invocation.
    size = 8193
    kernel = cupy.RawKernel(
        f'extern "C" __global__ void update(const float* x, float* state) {{'
        f"int i = blockIdx.x * blockDim.x + threadIdx.x;"
        f"if (i < {size}) state[2*i] += x[2*i];"
        "}",
        "update",
    )
    kernel.compile()
    original_import = importlib.import_module
    deferred = []

    def retaining_import(name, *args, **kwargs):
        if name == "cupyx.profiler":
            try:
                raise ImportError("optional profiler dependency")
            except ImportError as error:
                deferred.append(error)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", retaining_import)
    autotune._cupy_profiler.cache_clear()
    scratch_refs = []
    original_scratch = autotune.scratch_arguments

    def observe_scratch(args, writes):
        scratch, reset = original_scratch(args, writes)
        scratch_refs.extend(weakref.ref(scratch[i]) for i in writes)
        return scratch, reset

    monkeypatch.setattr(autotune, "scratch_arguments", observe_scratch)
    source = torch.linspace(-1, 1, size * 2, device="cuda")[::2]
    state = torch.full((size * 2,), 2.0, device="cuda")[::2]
    runtime = CuPyRuntime()
    launch = runtime.prepare_launch(
        kernel,
        [source, state],
        size,
        changing=(0, 1),
        enable_fp_fusion=True,
        writes=(1,),
    )
    torch.testing.assert_close(state, torch.full_like(state, 2), rtol=0, atol=0)
    assert scratch_refs and all(ref() is None for ref in scratch_refs)
    launch.submit()
    torch.testing.assert_close(state, 2 + source, rtol=0, atol=0)

    def no_new_trial(*_, **__):
        pytest.fail("cached CuPy preparation repeated its trial allocation")

    monkeypatch.setattr(autotune, "scratch_arguments", no_new_trial)
    other_state = torch.full_like(state, 4).repeat_interleave(2)[::2]
    other = runtime.prepare_launch(
        kernel,
        [source, other_state],
        size,
        changing=(0, 1),
        enable_fp_fusion=True,
        writes=(1,),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        other.submit()
    graph.replay()
    torch.testing.assert_close(other_state, 4 + source, rtol=0, atol=0)
    references = [
        weakref.ref(value)
        for tensor in (source, state, other_state)
        for value in (tensor, tensor._base)
    ]
    del graph, other, launch, source, state, other_state
    # Keep the deferred exception alive deliberately. Neither it nor a native
    # profiler/tuning cache may retain completed invocation storage.
    assert deferred
    assert all(ref() is None for ref in references)


@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_tuned_sparse_product_preserves_duplicate_edge_full_vjp(backend, monkeypatch):
    import gc
    import weakref

    from btorch.sparse.runtime import csr_propagate, use_backend
    from btorch.sparse.runtime.cache import RepresentationCache
    from btorch.utils import autotune
    from tests.models.kernels.helpers import require_cuda_backend

    require_cuda_backend(backend)
    generator = torch.Generator(device="cuda").manual_seed(23)
    # Large enough to tune, with an explicit split hub, duplicate edges and
    # strided sample storage. FP64 exposes reduction/order errors independently
    # of the lower precision accumulation policies covered elsewhere.
    rows = torch.randint(1, 33, (5003,), generator=generator, device="cuda")
    rows[:1000] = 1
    columns = torch.randint(259, rows.shape, generator=generator, device="cuda")
    topology = RepresentationCache()
    topology.build(rows, columns, (33, 259), 0)
    values = torch.randn(
        5003, generator=generator, device="cuda", dtype=torch.float64
    ).requires_grad_()
    inputs = torch.randn(
        3, 518, generator=generator, device="cuda", dtype=torch.float64
    )[:, ::2].requires_grad_()
    trial_refs = []
    original_scratch = autotune.scratch_arguments

    def observe_scratch(args, writes):
        scratch, reset = original_scratch(args, writes)
        trial_refs.extend(weakref.ref(scratch[i]) for i in writes)
        return scratch, reset

    monkeypatch.setattr(autotune, "scratch_arguments", observe_scratch)
    with use_backend(backend):
        actual = csr_propagate(
            topology.crow,
            topology.col,
            values,
            inputs,
            topology.t_crow,
            topology.t_col,
            topology.t_perm,
        )
    canonical_rows = torch.repeat_interleave(
        torch.arange(33, device="cuda"), topology.crow.diff()
    )
    expected = inputs.new_zeros(3, 33).index_add(
        1, canonical_rows, inputs[:, topology.col] * values
    )
    cotangent = torch.randn(
        3, 33, generator=generator, device="cuda", dtype=torch.float64
    )
    actual_vjp = torch.autograd.grad(actual, (values, inputs), cotangent)
    expected_vjp = torch.autograd.grad(expected, (values, inputs), cotangent)
    for result, reference in zip((actual, *actual_vjp), (expected, *expected_vjp)):
        torch.testing.assert_close(result, reference, rtol=1e-11, atol=1e-11)
    gc.collect()
    assert trial_refs and all(ref() is None for ref in trial_refs)

    def no_new_scratch(*_, **__):
        pytest.fail("native tuner cache hit allocated new trial storage")

    monkeypatch.setattr(autotune, "scratch_arguments", no_new_scratch)
    with use_backend(backend):
        # Repeat with new owning tensors and changed values. The same native
        # tuning cache must select the executable without capturing old inputs.
        repeated = csr_propagate(
            topology.crow,
            topology.col,
            values.detach(),
            inputs.detach() * 0.5,
            topology.t_crow,
            topology.t_col,
            topology.t_perm,
        )
    torch.testing.assert_close(repeated, expected * 0.5, rtol=1e-11, atol=1e-11)
