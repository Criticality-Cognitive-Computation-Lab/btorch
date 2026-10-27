"""Verify generated cell source is reused without retaining runtime tensors."""

import gc
import inspect
import linecache
import weakref

import pytest
import torch

from btorch.models.kernels.backends import engine, fused
from btorch.models.kernels.contracts import NeuronSpec
from btorch.models.kernels.templates import get_template


def _scalar_layouts(spec: NeuronSpec, neurons: int):
    return tuple(
        engine._ParameterLayout((), (), (neurons,), 1)
        for _ in get_template(spec).parameter_names
    )


def test_source_cache_uses_semantics_and_does_not_retain_runtime_dt(monkeypatch):
    engine._generate_source.cache_clear()
    spec = NeuronSpec("izhikevich", dt=torch.tensor(0.25), hard_reset=True)
    dt_reference = weakref.ref(spec.dt)
    layouts = _scalar_layouts(spec, 17)

    # The timestep is a runtime device value. Two invocations with different
    # physical timesteps must share the same program while preserving the dt
    # pointer in its ABI, and the cache must not keep the first tensor alive.
    first = engine.generate_source(
        "triton", spec, 17, 23, ("v",), parameter_layouts=layouts
    )
    del spec
    gc.collect()
    assert dt_reference() is None
    second = engine.generate_source(
        "triton",
        NeuronSpec("izhikevich", dt=0.75, hard_reset=True),
        17,
        23,
        ("v",),
        parameter_layouts=layouts,
    )
    assert second == first
    assert engine._generate_source.cache_info().hits == 1

    # Reset policy changes executable equations and therefore must not reuse
    # the hard-reset source. Parameter layout and optional cotangent presence
    # likewise change concrete load expressions and the reverse pointer ABI.
    soft = engine.generate_source(
        "triton",
        NeuronSpec("izhikevich", dt=0.75),
        17,
        23,
        ("v",),
        parameter_layouts=layouts,
    )
    assert soft[0] != first[0]
    dense_layouts = tuple(
        engine._ParameterLayout((17,), (1,), (17,), 17) for _ in layouts
    )
    dense = engine.generate_source(
        "triton",
        NeuronSpec("izhikevich", hard_reset=True),
        17,
        23,
        ("v",),
        parameter_layouts=dense_layouts,
    )
    assert dense[0] != first[0]
    reverse_all = engine.generate_source(
        "triton",
        NeuronSpec("izhikevich", hard_reset=True),
        17,
        23,
        ("v",),
        backward=True,
        parameter_layouts=layouts,
    )
    reverse_partial = engine.generate_source(
        "triton",
        NeuronSpec("izhikevich", hard_reset=True),
        17,
        23,
        ("v",),
        backward=True,
        parameter_layouts=layouts,
        gradient_presence=(False, (True, False), (False,)),
    )
    assert reverse_partial[1] != reverse_all[1]

    # Template replacement is revisioned elsewhere in the registry. Including
    # that revision in this cache key guarantees a replacement cannot execute
    # source rendered from the previous declaration.
    misses = engine._generate_source.cache_info().misses
    monkeypatch.setattr(engine, "model_template_revision", lambda _model: 2)
    assert (
        engine.generate_source(
            "triton",
            NeuronSpec("izhikevich", hard_reset=True),
            17,
            23,
            ("v",),
            parameter_layouts=layouts,
        )
        == first
    )
    assert engine._generate_source.cache_info().misses == misses + 1


def test_fused_source_caches_final_abis_without_retaining_runtime_dt(monkeypatch):
    fused._cached_fused_source.cache_clear()
    fused._cached_fused_backward_source.cache_clear()
    neuron_spec = NeuronSpec("lif", dt=torch.tensor(0.25), hard_reset=True)
    psc_spec = NeuronSpec("alpha", dt=torch.tensor(0.25))
    neuron_dt = weakref.ref(neuron_spec.dt)
    psc_dt = weakref.ref(psc_spec.dt)

    # Both physical timesteps are runtime buffers. The finalized source and
    # pointer ABI can be reused without retaining either caller-owned tensor.
    forward = fused.generate_fused_source(
        "triton",
        neuron_spec,
        psc_spec,
        17,
        2,
        2,
        ("v_charged",),
        ("psc",),
    )
    del neuron_spec, psc_spec
    gc.collect()
    assert neuron_dt() is None and psc_dt() is None
    assert (
        fused.generate_fused_source(
            "triton",
            NeuronSpec("lif", dt=0.75, hard_reset=True),
            NeuronSpec("alpha", dt=0.75),
            17,
            2,
            2,
            ("v_charged",),
            ("psc",),
        )
        == forward
    )
    assert fused._cached_fused_source.cache_info().hits == 1

    # History stores and merged receptor topology change executable stores and
    # declaration sizes, so neither may collide with the cached dense stage.
    no_histories = fused.generate_fused_source(
        "triton", NeuronSpec("lif", hard_reset=True), NeuronSpec("alpha"), 17, 2, 2
    )
    merged = fused.generate_fused_source(
        "triton", NeuronSpec("lif", hard_reset=True), NeuronSpec("alpha"), 17, 1, 3
    )
    assert no_histories[1] != forward[1]
    assert merged[0] != no_histories[0]
    misses = fused._cached_fused_source.cache_info().misses
    monkeypatch.setattr(fused, "model_template_revision", lambda _model: 2)
    assert (
        fused.generate_fused_source(
            "triton",
            NeuronSpec("lif", hard_reset=True),
            NeuronSpec("alpha"),
            17,
            2,
            2,
            ("v_charged",),
            ("psc",),
        )
        == forward
    )
    assert fused._cached_fused_source.cache_info().misses == misses + 1

    # Reverse source is cached only after optional cotangents and trainable
    # parameter leaves have changed its ABI. A partial loss must remove those
    # pointers rather than reuse the dense reverse declaration.
    neuron = get_template(NeuronSpec("lif"))
    psc = get_template(NeuronSpec("alpha"))
    reverse_neuron_spec = NeuronSpec("lif", dt=torch.tensor(0.5), hard_reset=True)
    reverse_psc_spec = NeuronSpec("alpha", dt=torch.tensor(0.5))
    reverse_neuron_dt = weakref.ref(reverse_neuron_spec.dt)
    reverse_psc_dt = weakref.ref(reverse_psc_spec.dt)
    full_reverse = fused._generate_fused_backward_source(
        "triton",
        reverse_neuron_spec,
        reverse_psc_spec,
        17,
        2,
        2,
        0,
        neuron.parameter_names,
        psc.parameter_names,
        True,
        (True,),
        (True, True),
        (True, True),
        (True, True),
    )
    del reverse_neuron_spec, reverse_psc_spec
    gc.collect()
    assert reverse_neuron_dt() is None and reverse_psc_dt() is None
    reverse_hits = fused._cached_fused_backward_source.cache_info().hits
    assert (
        fused._generate_fused_backward_source(
            "triton",
            NeuronSpec("lif", dt=0.75, hard_reset=True),
            NeuronSpec("alpha", dt=0.75),
            17,
            2,
            2,
            0,
            neuron.parameter_names,
            psc.parameter_names,
            True,
            (True,),
            (True, True),
            (True, True),
            (True, True),
        )
        == full_reverse
    )
    assert fused._cached_fused_backward_source.cache_info().hits == reverse_hits + 1
    partial_reverse = fused._generate_fused_backward_source(
        "triton",
        NeuronSpec("lif", dt=0.75, hard_reset=True),
        NeuronSpec("alpha", dt=0.75),
        17,
        2,
        2,
        0,
        ("c_m",),
        (),
        False,
        (True,),
        (False, False),
        (False, False),
        (False, False),
    )
    full_names = {name for name, _ in full_reverse[1]}
    partial_names = {name for name, _ in partial_reverse[1]}
    assert {
        "grad_spike",
        "psc_grad_psc",
        "psc_grad_h",
        "neuron_record_grad_v_charged",
        "neuron_grad_tau",
    } <= full_names
    assert {
        "grad_spike",
        "psc_grad_psc",
        "psc_grad_h",
        "neuron_record_grad_v_charged",
        "neuron_grad_tau",
    }.isdisjoint(partial_names)
    assert "neuron_grad_c_m" in partial_names


def test_compiled_source_lives_until_the_kernel_is_released():
    """Keep source inspectable after eviction, without leaking dead kernels."""
    pytest.importorskip("triton")
    source = "import triton\n@triton.jit\ndef cell_kernel():\n    pass\n"
    engine._compile_cached.cache_clear()
    kernel = engine._compile("triton", source)
    filename = kernel.fn.__code__.co_filename
    reference = weakref.ref(kernel)
    # Triton also owns its latest JIT function by qualified name. Compile a
    # replacement normally so that ownership moves away from the first one;
    # do not manipulate the backend's private function registry.
    replacement = engine._compile("triton", source + "# replacement\n")
    assert replacement is not kernel
    # A caller can retain a JIT function beyond the compilation LRU. Its source
    # remains available for inspection and backend diagnostics in that period.
    engine._compile_cached.cache_clear()
    gc.collect()
    assert "def cell_kernel" in inspect.getsource(kernel.fn)
    assert filename in linecache.cache
    del kernel
    gc.collect()
    assert reference() is None
    assert filename not in linecache.cache


def test_failed_compilation_does_not_leave_generated_source():
    """Release source even when execution fails before producing a kernel."""
    before = set(linecache.cache)
    with pytest.raises(SyntaxError):
        engine._compile("triton", "def cell_kernel(:\n")
    assert not any(
        name.startswith("<btorch-") for name in set(linecache.cache) - before
    )
