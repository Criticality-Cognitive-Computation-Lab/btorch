"""CPU checks for backend source rendering and token-level transforms."""

import ast

import pytest

from btorch.models.kernels.backends.engine import _TensorStorage
from btorch.models.kernels.backends.renderers import (
    get_renderer,
    local_statement,
    transform_statement,
)
from btorch.models.kernels.program import namespace_statements


def test_math_and_name_transforms_use_python_tokens():
    """Function names and identifiers must not be changed by substrings."""
    line = "tau_syn = where(abs(tau_syn) >= 0.5, exp(tau_syn), 0.0)"
    lowered = transform_statement(
        line,
        {"where": "tl.where", "abs": "tl.abs", "exp": "tl.exp"},
    )
    assert lowered == "tau_syn = tl.where(tl.abs(tau_syn) >= 0.5, tl.exp(tau_syn), 0.0)"

    prefixed = namespace_statements((line,), "psc_")[0]
    assert prefixed == (
        "psc_tau_syn = where(abs(psc_tau_syn) >= 0.5, exp(psc_tau_syn), 0.0)"
    )

    local = local_statement("x = x + exp(v)", {"x", "v"})
    assert local == "x[0] = x[0] + exp(v[0])"


@pytest.mark.parametrize("backend", ("triton", "cupy", "tilelang"))
def test_renderer_keeps_shared_math_and_emits_backend_syntax(backend):
    """Each renderer lowers the same scalar statement without equations
    forks."""
    renderer = get_renderer(backend, 17, 3, {}, precise_exp=False)
    source = renderer.render(
        ("x = fma(a, b, c)", "END"),
        (("x", 17),),
        False,
        False,
    )
    ast.parse("x = fma(a, b, c)")
    assert "fma(a" not in source.replace("tl.fma(a", "")
    assert "x =" in source or "x[0] =" in source
    if backend == "triton":
        assert "tl.fma(a, b, c)" in source
        assert "END" not in source
    elif backend == "cupy":
        assert "fmaf(a, b, c)" in source
        assert "for (int" not in source
    else:
        assert "T.call_pure_extern('float32', 'fmaf', a, b, c)" in source


def test_renderer_storage_index_preserves_strided_views():
    """The renderer owns integer division syntax while preserving view
    strides."""
    layout = _TensorStorage((2, 3), (4, 1), "float32", 7)
    triton = get_renderer("triton", 6, 1, {}, False)
    cupy = get_renderer("cupy", 6, 1, {}, False)
    assert "//" in triton.storage_index(layout, "index")
    assert "/" in cupy.storage_index(layout, "index")


def test_cupy_storage_index_keeps_integer_bounds_integer():
    """Time-history bounds must use CUDA integer min/max, never fminf/fmaxf."""
    cupy = get_renderer("cupy", 6, 3, {}, False)
    expression = cupy.storage_index(None, "maximum(time - 1, 0)")
    assert "max(time - 1, 0)" in expression
    assert "fmaxf" not in expression
