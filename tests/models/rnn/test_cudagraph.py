"""Tests for the ``cudagraph=True`` inference path and the reduce-overhead
training path (``btorch/models/cudagraph.py``).

The runner wraps torch's CUDA graph API, so these cover typical use,
edge cases, and the API guards -- not graph capture itself.
"""

import pytest
import torch

from btorch.models.functional import (
    named_hidden_states,
    reset_net_state,
    set_hidden_states,
)
from btorch.models.rnn import make_rnn
from tests.models.rnn.rnn_utils import SimpleRNNCell


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cudagraph capture requires CUDA"
)

T, B, INPUT_DIM, H = 32, 2, 4, 8


def _build(**kwargs):
    torch.manual_seed(42)
    cell = make_rnn(SimpleRNNCell, unroll=8, **kwargs)
    return cell(input_size=INPUT_DIM, hidden_size=H).to("cuda")


def _pair(**kwargs):
    """An eager reference and a cudagraph model sharing the same weights."""
    eager = _build(**kwargs)
    graphed = _build(cudagraph=True, **kwargs)
    graphed.rnn_cell.load_state_dict(eager.rnn_cell.state_dict())
    return eager, graphed


def _run(model, x):
    reset_net_state(model, batch_size=x.shape[1])
    return model(x)


# --- typical use --------------------------------------------------------------


@torch.no_grad()
def test_cudagraph_matches_eager():
    eager, graphed = _pair()
    x = torch.randn(T, B, INPUT_DIM, device="cuda")

    ref, ref_states = _run(eager, x)
    out, states = _run(graphed, x)

    # Same kernels on the same inputs -> replay is bit-exact, not merely close.
    assert torch.equal(out, ref)
    assert states.keys() == ref_states.keys()
    for name, value in states.items():
        assert torch.equal(value, ref_states[name]), name


@torch.no_grad()
def test_cudagraph_carries_state_across_calls():
    """Fresh inputs take effect and state resumes from the previous call.

    Also the load-bearing correctness check: the second call only matches eager if
    replay fed x2's data into the static buffers *and* left x1's final state live
    on the module rather than the zeros ``reset_net_state`` bound.
    """
    eager, graphed = _pair()
    x1 = torch.randn(T, B, INPUT_DIM, device="cuda")
    x2 = torch.randn(T, B, INPUT_DIM, device="cuda")

    reset_net_state(eager, batch_size=B)
    eager(x1)
    ref, _ = eager(x2)

    reset_net_state(graphed, batch_size=B)
    graphed(x1)
    out, _ = graphed(x2)  # no reset -> must continue from x1's final state

    assert torch.equal(out, ref)


@torch.no_grad()
def test_cudagraph_composes_with_compile():
    """The recommended fast path: Inductor fuses the unroll block, capture wraps
    the compiled kernels."""
    torch._dynamo.reset()
    eager, graphed = _pair()
    x = torch.randn(T, B, INPUT_DIM, device="cuda")

    ref, _ = _run(eager, x)
    compiled = torch.compile(graphed)
    reset_net_state(graphed, batch_size=B)
    out, _ = compiled(x)

    assert torch.allclose(out, ref, atol=1e-5)


@torch.no_grad()
@pytest.mark.parametrize("seq_len", [64, 72], ids=["exact-chunks", "ragged-remainder"])
def test_cudagraph_with_cpu_offload(seq_len):
    """Offload composes with capture: device work replays, the D2H moves stay eager.

    ``seq_len=72`` with ``chunk_size=16`` leaves a short final chunk that keys to
    its own capture against an already-dirty pool -- the case that catches a
    capture reading uninitialised pool memory, so it is worth parametrizing.
    """
    eager, graphed = _pair(chunk_size=16, cpu_offload=True)
    x = torch.randn(seq_len, B, INPUT_DIM, device="cuda")

    ref, ref_states = _run(eager, x)
    out, states = _run(graphed, x)

    assert out.device.type == "cpu", "offload should still land the result on CPU"
    assert out.shape == (seq_len, B, H)
    assert torch.equal(out, ref)
    for name, value in states.items():
        assert torch.equal(value, ref_states[name]), name


# --- edge cases ---------------------------------------------------------------


@torch.no_grad()
def test_cudagraph_recaptures_per_shape():
    """A graph is valid only for its captured shape, so a new T re-captures."""
    eager, graphed = _pair()
    long_x = torch.randn(T, B, INPUT_DIM, device="cuda")
    short_x = torch.randn(T // 2, B, INPUT_DIM, device="cuda")

    assert torch.equal(_run(graphed, long_x)[0], _run(eager, long_x)[0])
    assert torch.equal(_run(graphed, short_x)[0], _run(eager, short_x)[0])
    assert len(graphed._cudagraph_runner.entries) == 2


@pytest.mark.parametrize(
    "x_needs_grad", [True, False], ids=["grad-input", "plain-input"]
)
def test_cudagraph_is_inference_only(x_needs_grad):
    """Refused before the forward whenever autograd would record.

    ``plain-input`` is the case that matters: real training feeds plain data
    through parameters that require grad, which an args-only check would miss.
    """
    graphed = _build(cudagraph=True)
    x = torch.randn(T, B, INPUT_DIM, device="cuda", requires_grad=x_needs_grad)
    assert any(p.requires_grad for p in graphed.parameters())

    with pytest.raises(RuntimeError, match="inference-only"):
        _run(graphed, x)


@torch.no_grad()
@pytest.mark.parametrize(
    "kwargs, x_device, match",
    [
        (dict(grad_checkpoint=True), "cuda", "incompatible with grad_checkpoint"),
        (dict(), "cpu", "CUDA device"),
    ],
    ids=["grad_checkpoint", "cpu-input"],
)
def test_cudagraph_rejects_unsupported(kwargs, x_device, match):
    graphed = _build(cudagraph=True, **kwargs)
    with pytest.raises(RuntimeError, match=match):
        _run(graphed, torch.randn(T, B, INPUT_DIM, device=x_device))


# --- training path ------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [dict(), dict(grad_checkpoint=True), dict(cpu_offload=True, chunk_size=8)],
    ids=["plain", "grad_checkpoint", "cpu_offload"],
)
def test_reduce_overhead_training_matches_eager(kwargs):
    """Training with CUDA graphs goes through torch.compile(mode="reduce-
    overhead"), whose Trees capture forward and backward separately. It
    composes with what cudagraph=True cannot (grad_checkpoint, cpu_offload);
    check grads vs eager.

    mark_step_begin() each step: Trees reuse buffers and need the
    iteration boundary to stay sound.
    """
    torch._dynamo.reset()
    eager = _build(**kwargs)
    model = _build(**kwargs)
    model.rnn_cell.load_state_dict(eager.rnn_cell.state_dict())
    x = torch.randn(T, B, INPUT_DIM, device="cuda")

    def grad_of(m, fn, steps):
        for _ in range(steps):
            if fn is not m:
                torch.compiler.cudagraph_mark_step_begin()
            m.rnn_cell.W_x.grad = None
            reset_net_state(m, batch_size=B)
            fn(x)[0].sum().backward()
        torch.cuda.synchronize()
        return m.rnn_cell.W_x.grad

    g_ref = grad_of(eager, eager, steps=1)
    compiled = torch.compile(model, mode="reduce-overhead")
    g = grad_of(model, compiled, steps=3)  # reach steady state

    assert torch.allclose(g.cpu(), g_ref.cpu(), atol=1e-4)


def _capture_graph(step_fn, *, warmup=3, before_capture=None):
    """Warm up on a side stream (capture requires it), then capture one
    ``step_fn``

    call into a CUDA graph. ``before_capture`` runs after warmup and before capture
    -- used to undo any state the warmup mutated.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            step_fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    if before_capture is not None:
        before_capture()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step_fn()
    return graph


def test_whole_step_training_capture_matches_eager():
    """A whole training step (fwd+loss+backward+optim) captured as one CUDA
    graph trains identically to eager -- tensordict.nn.CudaGraphModule's
    approach.

    The pattern's requirements: static input and grad buffers (``set_to_none=False``),
    and a persistent state to start each step from instead of ``reset_net_state``
    (which does a host copy and rebinds the buffers). The cell reads that state at
    step 0 and rebinds away, so it is never written and needs no per-step reset.
    Forward and backward share one graph, so activations saved in the forward live
    and die within each replay -- unlike ``cudagraph=True``, which captures the
    forward only.
    """
    torch.manual_seed(0)
    xs = [torch.randn(T, B, INPUT_DIM, device="cuda") for _ in range(5)]
    targets = [torch.randn(T, B, H, device="cuda") for _ in range(5)]

    def make():
        model = _build()  # deterministic init -> eager and captured start identical
        reset_net_state(model, batch_size=B)
        start_state = {k: v.clone() for k, v in named_hidden_states(model).items()}
        return model, start_state, torch.optim.SGD(model.parameters(), lr=0.1)

    def step(model, start_state, opt, x, target):
        opt.zero_grad(set_to_none=False)  # keep .grad buffers static
        set_hidden_states(model, start_state)  # start each step from the same state
        ((model(x)[0] - target) ** 2).mean().backward()
        opt.step()

    # Eager reference.
    ref, ref_state, ref_opt = make()
    for x, target in zip(xs, targets):
        step(ref, ref_state, ref_opt, x, target)

    # Same init and data, but the step is captured and replayed.
    model, start_state, opt = make()
    init_weights = {k: v.clone() for k, v in model.state_dict().items()}
    static_x, static_target = xs[0].clone(), targets[0].clone()

    graph = _capture_graph(
        lambda: step(model, start_state, opt, static_x, static_target),
        before_capture=lambda: model.load_state_dict(init_weights),  # undo warmup
    )
    for x, target in zip(xs, targets):
        static_x.copy_(x)
        static_target.copy_(target)
        graph.replay()  # one launch = whole fwd+backward+optim step
    torch.cuda.synchronize()

    assert torch.allclose(model.rnn_cell.W_x, ref.rnn_cell.W_x, atol=1e-4)
