"""Regression sentinels: training (backward) through a ``scan``-based RNN.

``scan`` is the structurally correct fix for keeping ``torch.compile`` fast on the
long RNN loop in ``btorch/models/rnn.py`` (the step is traced once, T stays a
runtime dim -> O(1) compile). Forward/inference works today. Training does not,
for **two independent reasons** -- each gets its own sentinel below so that a
torch upgrade fixing one of them produces a signal rather than being masked by
the other:

1. ``test_scan_compiled_backward_alias_blocker`` -- output-output aliasing.
2. ``test_scan_eager_grad_flows_through_carry`` -- silently wrong gradients.

Verified against torch 2.11.0. ``scan`` is a prototype op whose own docstring
warns "You may run into miscompiles"; (2) is one.

--- (1) the aliasing blocker -------------------------------------------------

``scan``'s documented restriction is:

    The combine_fn shouldn't have any aliasing between input-input,
    input-output, and output-output. [...] As a workaround, can clone the
    output to avoid aliasing.

The failure is **output-output** (``out_out_alias_map == {0: 3}``), and it is
*not* about the lifted parameters (``inp_out_alias_map`` and ``inp_mutation``
are both empty). Mechanism: ``tanh_backward`` needs tanh's *result*, so the
partitioner saves that node for backward; ``scan``'s autograd forward must emit
saved values as extra combine_fn outputs; but that tensor is already output 0,
the carry. The fw graph returns ``tanh`` twice -> raise.

This is why the ``h.clone()`` in ``rnn_utils.ScanRNN`` is not enough: it removes
the carry<->y alias (the workaround the docs describe), but the carry<->saved-
activation alias is manufactured inside the partitioner and is unreachable from
user code. It reproduces with a combine_fn that closes over *nothing at all*,
and it is specific to activations whose backward saves their output:

    tanh (saves output) -> alias error   |   sin / identity (saves input / -) -> OK

That makes it a real blocker for RNN cells generally (tanh, sigmoid, ...), not
an artifact of this cell.

--- (2) the silent gradient bug ----------------------------------------------

Independent of compile: when the scan ``init`` does not require grad, ``scan``
drops the gradient flowing back through the carry, so parameters receive only
their direct single-step contribution. No error is raised and the forward is
bit-exact, which makes this the more dangerous of the two.

Upstream refs (not yet confirmed to describe (1)/(2) specifically):
    https://github.com/pytorch/pytorch/issues/156337  (mutations/aliasing in training)
    https://github.com/pytorch/pytorch/issues/153437  (graph breaks + backward errors)

The compile-time benchmark lives in ``benchmarks/rnn/test_compile_strategies.py``.
"""

import platform

import pytest
import torch

from btorch.models.functional import reset_net_state
from btorch.models.rnn import make_rnn

from .rnn_utils import DTYPE, ScanRNN, SimpleRNNCell, requires_scan


try:  # resolved at import time by the raises= below, so it must not hard-fail
    from torch._dynamo.exc import BackendCompilerFailed
except ImportError:  # pragma: no cover - depends on the installed torch
    BackendCompilerFailed = RuntimeError  # BackendCompilerFailed subclasses it

pytestmark = [
    pytest.mark.skipif(
        platform.system() != "Linux",
        reason="torch.compile/scan only supported on Linux",
    ),
    requires_scan,  # torch predating the scan HOP: nothing here is meaningful
]

T, B, N_IN, H = 16, 2, 4, 8


def _fixture():
    """A cell, btorch's own loop over it (shared weights), and an input."""
    torch.manual_seed(0)
    cell = SimpleRNNCell(N_IN, H)
    return cell, make_rnn(cell), torch.randn(T, B, N_IN, dtype=DTYPE)


def _grads_of(cell, fn):
    """Run ``fn``, backward through ``.sum()``, return this cell's grads."""
    for p in cell.parameters():
        p.grad = None
    fn().sum().backward()
    return {n: p.grad.clone() for n, p in cell.named_parameters()}


def _ref_grads(cell, ref, x):
    """Gradients of btorch's unrolled loop -- the ground truth for parity.

    Cross-checked against a hand-written loop: they agree exactly, so a
    disagreement below indicts ``scan``, not ``make_rnn``.
    """

    def run():
        reset_net_state(ref, batch_size=B)
        return ref(x)[0]

    return _grads_of(cell, run)


def _assert_parity(ref_grads, cell):
    for n, p in cell.named_parameters():
        assert torch.allclose(ref_grads[n], p.grad, atol=1e-5), (
            f"{n}: scan grad != btorch loop grad "
            f"(max|diff|={(ref_grads[n] - p.grad).abs().max():.3e})"
        )


@pytest.mark.xfail(
    strict=True,
    raises=BackendCompilerFailed,
    reason=(
        "scan combine_fn output-output aliasing: the saved-for-backward tanh "
        "activation is also the carry, so the fw graph returns it twice "
        "(out_out_alias_map={0: 3}). Flips to a failure when torch fixes it."
    ),
)
def test_scan_compiled_backward_alias_blocker():
    """The *only* thing blocking compiled scan training, once (2) is ruled out.

    ``h0`` requires grad here -- deliberately -- so this cannot fail for reason
    (2). It is built outside the compiled region because dynamo refuses
    ``requires_grad=True`` tensor creation inside one.

    ``raises=`` pins the expectation: an unrelated regression fails loudly
    instead of masquerading as the known bug. When torch fixes the aliasing this
    XPASSes -> strict -> failure -> scan is ready to replace the loop in rnn.py.
    """
    cell, ref, x = _fixture()
    ref_grads = _ref_grads(cell, ref, x)

    h0 = torch.zeros(B, H, dtype=DTYPE, requires_grad=True)
    compiled = torch.compile(ScanRNN(cell), fullgraph=True)
    _grads_of(cell, lambda: compiled(x, h0))

    _assert_parity(ref_grads, cell)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "scan silently drops the carry's gradient when init doesn't require "
        "grad: params get only their direct single-step contribution. No error, "
        "forward is bit-exact. Eager -- no compile involved."
    ),
)
def test_scan_eager_grad_flows_through_carry():
    """Sentinel for the silent miscompile. Deliberately eager: no compile.

    ``h0`` does *not* require grad (what a natural zero-init looks like), which
    is the whole trigger. ``h0.requires_grad = True`` alone makes these grads
    exact -- see ``test_scan_eager_grad_parity_requires_grad_init``.

    Nothing raises here; the assertion is the signal.
    """
    cell, ref, x = _fixture()
    ref_grads = _ref_grads(cell, ref, x)

    h0 = torch.zeros(B, H, dtype=DTYPE)  # no requires_grad -> triggers the bug
    _grads_of(cell, lambda: ScanRNN(cell)(x, h0))

    _assert_parity(ref_grads, cell)


def test_scan_eager_grad_parity_requires_grad_init():
    """The known-good configuration -- passes today; guards against regression.

    Pins the counterfactual that makes the two xfails above precise: eager scan
    is *numerically correct* when the init requires grad. So (1) is purely a
    compile-stack aliasing issue and (2) is purely about ``init.requires_grad``
    -- neither is scan's recurrence math being wrong.
    """
    cell, ref, x = _fixture()
    ref_grads = _ref_grads(cell, ref, x)

    h0 = torch.zeros(B, H, dtype=DTYPE, requires_grad=True)
    _grads_of(cell, lambda: ScanRNN(cell)(x, h0))

    _assert_parity(ref_grads, cell)


def test_scan_inference_forward_parity():
    """Inference works today: compiled scan forward matches btorch's loop.

    Bounds the blast radius of (1) and (2) -- both are training-only.
    """
    cell, ref, x = _fixture()
    reset_net_state(ref, batch_size=B)
    with torch.no_grad():
        expected = ref(x)[0]
        actual = torch.compile(ScanRNN(cell), fullgraph=True)(x)
    torch.testing.assert_close(actual, expected)
