"""Gradients and ``torch.compile`` of ``SparseConnection`` (spec sections 35,
36, 47 and 50).

Every gradient is compared with a *dense twin*: the same edge list written as
an ordinary differentiable dense matrix, so autograd's own dense rules are the
reference. Every compiled result is compared with the eager module and with
the dense twin, and the compiled artefact must be one graph without breaks.

The connection kinds under test:

- ``edge``: one trainable weight per edge;
- ``constrained``: ``w[e] = base[e] * scale[group[e]]``, only ``scale`` trains;
- ``batched``: ``G`` networks sharing one pattern, values ``[G, E]``;
- ``constrained_batched``: an unbatched matrix with scales ``[G, n_group]``.
"""

import io

import pytest
import torch
from torch.func import functional_call

from btorch.models.connection import ConstrainedWeight, SparseConnection, Synapse
from tests.sparse.helpers import DEVICES


N_POST, N_PRE, G = 4, 6, 2
# Edge list in a deliberately non-canonical order; neuron 2 has no input and
# neuron 3 no output. The connection sorts edges internally, so comparisons of
# per-edge gradients go through dense matrices, never through edge order.
PRE = torch.tensor([5, 0, 2, 1, 4, 0, 5, 2, 1])
POST = torch.tensor([3, 3, 0, 0, 0, 1, 1, 3, 1])
# A second pattern with the same number of edges (for checkpoint loading).
PRE_B = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2])
POST_B = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 3])
N_EDGE = PRE.shape[0]
GROUP = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
KINDS = ["edge", "constrained", "batched", "constrained_batched"]

_gen = torch.Generator().manual_seed(0)
VALUES = torch.randn(N_EDGE, generator=_gen, dtype=torch.float64)  # weights / base
VALUES_G = torch.randn(G, N_EDGE, generator=_gen, dtype=torch.float64)
SCALE = torch.tensor([1.5, -0.5, 2.0], dtype=torch.float64)
SCALE_G = torch.stack([SCALE, SCALE.flip(0)])


def build(kind, pre=PRE, post=POST, dtype=torch.float64, device=None):
    """The connection of one kind on the edge list ``(pre, post)``."""
    values, weight = VALUES, None
    if kind == "batched":
        values = VALUES_G
    elif kind == "constrained":
        weight = ConstrainedWeight(GROUP, scale=SCALE)
    elif kind == "constrained_batched":
        weight = ConstrainedWeight(GROUP, scale=SCALE_G)
    conn = SparseConnection.from_edges(
        pre, post, N_PRE, N_POST, Synapse(weight=weight), values=values.to(dtype)
    )
    return conn.to(device=device, dtype=dtype)


def leaf_of(kind):
    """The trainable tensor of the dense twin (input-edge order)."""
    leaf = {
        "edge": VALUES,
        "batched": VALUES_G,
        "constrained": SCALE,
        "constrained_batched": SCALE_G,
    }[kind]
    return leaf.clone().requires_grad_()


def dense_twin(kind, leaf, pre=PRE, post=POST):
    """Dense operator ``[*batch, n_post, n_pre]`` as a differentiable function
    of the trainable tensor, written with plain indexing."""
    weight = leaf
    if kind.startswith("constrained"):
        weight = VALUES.to(leaf) * leaf[..., GROUP.to(leaf.device)]
    dense = torch.zeros(*weight.shape[:-1], N_POST, N_PRE).to(leaf)
    dense[..., post, pre] = weight
    return dense


def dense_forward(dense, x):
    """``y = A @ x``; a network batch pairs ``A[g]`` with ``x[g]``."""
    if dense.ndim == 3:
        return torch.einsum("gmn,g...n->g...m", dense, x)
    return x @ dense.T


def make_input(kind, n_sample=3, dtype=torch.float64, device=None, seed=1):
    shape = (G, n_sample, N_PRE) if "batched" in kind else (n_sample, N_PRE)
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=gen, dtype=torch.float64)
    return x.to(device=device, dtype=dtype).requires_grad_()


def loss_and_grads(module, conn, x):
    """Output and gradients ``(d/dparam, d/dx)`` of a fixed scalar loss when
    the forward pass runs through ``module`` (eager or compiled ``conn``)."""
    (param,) = conn.parameters()
    out = module(x)
    grads = torch.autograd.grad(out.square().sum(), (param, x))
    return out.detach(), grads


def param_grad_as_dense(conn, grad):
    """Scatter a per-slot gradient to ``[*batch, n_post, n_pre]`` using the
    connection's semantic edge list, to compare it independent of slot
    order."""
    dense = torch.zeros(*grad.shape[:-1], N_POST, N_PRE).to(grad)
    dense[..., conn.indices[0], conn.indices[1]] = grad
    return dense


def assert_matches_dense_twin(kind, conn, out, grads, x, tol=1e-9):
    """Forward and both gradients equal those of the dense twin."""
    leaf = leaf_of(kind).to(x)
    leaf.retain_grad()
    x_ref = x.detach().clone().requires_grad_()
    dense = dense_twin(kind, leaf)
    dense.retain_grad()
    expected = dense_forward(dense, x_ref)
    expected.square().sum().backward()
    kw = {"atol": tol, "rtol": tol}
    torch.testing.assert_close(out, expected.detach(), **kw)
    torch.testing.assert_close(grads[1], x_ref.grad, **kw)
    if kind.startswith("constrained"):
        # d/dscale[g] = sum over the edges of group g of base[e] * dL/dw[e].
        torch.testing.assert_close(grads[0], leaf.grad, **kw)
    else:
        # dL/dA is a full matrix; the edge weights receive its entries at the
        # stored coordinates (slot k is the edge post[k] <- pre[k]).
        post, pre = conn.indices
        torch.testing.assert_close(grads[0], dense.grad[..., post, pre], **kw)


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    """Each test compiles from scratch (no recompile limits, no reuse)."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# ---------------------------------------------------------------- gradients
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_gradients_match_dense_reference(kind, device):
    """Gradients w.r.t.

    the trainable tensor and the dense input equal the dense twin's.
    Topology tensors receive no gradient.
    """
    conn = build(kind, device=device)
    x = make_input(kind, device=device)
    out, grads = loss_and_grads(conn, conn, x)
    assert_matches_dense_twin(kind, conn, out, grads, x)
    assert [n for n, p in conn.named_parameters() if p.requires_grad] == [
        "weight.scale" if kind.startswith("constrained") else "weight.value"
    ]
    assert not any(b.requires_grad for b in conn.buffers())


@pytest.mark.parametrize("kind", KINDS)
def test_gradcheck(kind):
    """Finite-difference check (float64) of the module as a function of its
    input and of its trainable tensor."""
    conn = build(kind)
    x = make_input(kind)
    ((name, param),) = conn.named_parameters()
    assert torch.autograd.gradcheck(conn, (x,))
    assert torch.autograd.gradcheck(
        lambda p: functional_call(conn, {name: p}, (x.detach(),)), (param,)
    )


def test_constrained_scale_gradient_formula():
    """For ``w[e] = base[e] * scale[group[e]]`` the scale gradient is ``sum_{e
    in group} base[e] * dL/dw[e]``, checked numerically against the per-edge
    gradient of an unconstrained connection with the same weights."""
    constrained = build("constrained")
    x = make_input("constrained")
    constrained(x).square().sum().backward()

    free = SparseConnection.from_edges(
        PRE, POST, N_PRE, N_POST, values=VALUES * SCALE[GROUP]
    )
    free(x.detach()).square().sum().backward()
    # Per-edge gradient back in input-edge order, via the dense matrix.
    dL_dw = param_grad_as_dense(free, free.weight.value.grad)[POST, PRE]
    expected = torch.zeros(3, dtype=torch.float64).index_add(0, GROUP, VALUES * dL_dw)
    torch.testing.assert_close(constrained.weight.scale.grad, expected)
    # The fixed base and the group ids are buffers: no gradient, no training.
    assert constrained.weight.base.grad is None
    assert not constrained.weight.group.is_floating_point()


def test_batched_networks_have_independent_gradients():
    """With shared-pattern values ``[G, E]`` the loss of network ``g`` only
    produces gradient in row ``g`` of the values."""
    conn = build("batched")
    x = make_input("batched")
    conn(x)[1].square().sum().backward()
    grad = conn.weight.value.grad
    assert torch.equal(grad[0], torch.zeros(N_EDGE, dtype=torch.float64))
    assert grad[1].abs().sum() > 0


# ------------------------------------------------------------ torch.compile
@pytest.mark.parametrize("kind", KINDS)
def test_compile_is_one_graph_without_breaks(kind):
    """Dynamo traces the whole forward as a single graph: the registered
    operator is the only sparse node and no Python planner runs inside."""
    conn = build(kind)
    report = torch._dynamo.explain(conn)(make_input(kind).detach())
    assert report.graph_count == 1, report
    assert report.graph_break_count == 0, report.break_reasons


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_compiled_forward_backward_equal_eager(kind, device):
    """``torch.compile(conn, fullgraph=True)`` over repeated calls and a
    changed batch size: outputs and gradients equal eager and the dense twin.
    An optimizer step between calls must be visible to the compiled module
    (parameters are graph inputs, not baked-in constants)."""
    conn = build(kind, device=device)
    compiled = torch.compile(conn, fullgraph=True)
    (param,) = conn.parameters()
    for call, n_sample in enumerate([3, 3, 5, 1]):
        x = make_input(kind, n_sample, device=device, seed=call)
        out_c, grads_c = loss_and_grads(compiled, conn, x)
        out_e, grads_e = loss_and_grads(conn, conn, x)
        torch.testing.assert_close(out_c, out_e)
        for g_c, g_e in zip(grads_c, grads_e):
            torch.testing.assert_close(g_c, g_e)
        with torch.no_grad():  # an SGD step
            param.sub_(0.05 * grads_c[0])
    # After the updates the compiled module still agrees with eager, and the
    # weights really changed (so the agreement above was not trivial).
    x = make_input(kind, device=device, seed=9)
    torch.testing.assert_close(compiled(x), conn(x))
    assert not torch.allclose(param.detach().cpu(), leaf_of(kind).detach())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["edge", "constrained", "batched"])
def test_to_before_compile(kind, device, dtype):
    """Build in float64 on CPU, move with ``.to(device, dtype)``, then
    compile: the compiled module matches the dense twin in that dtype."""
    conn = build(kind).to(device=device, dtype=dtype)
    compiled = torch.compile(conn, fullgraph=True)
    x = make_input(kind, dtype=dtype, device=device)
    out, grads = loss_and_grads(compiled, conn, x)
    assert out.dtype == dtype and out.device.type == device
    tol = 1e-9 if dtype == torch.float64 else 1e-4
    assert_matches_dense_twin(kind, conn, out, grads, x, tol=tol)


@pytest.mark.parametrize("kind", ["edge", "constrained", "batched"])
def test_checkpoint_then_compile(kind):
    """Save a trained connection, load it into a *fresh* module that was built
    with a different pattern (same edge count), compile: the result is the
    saved network.

    Nothing of the fresh module's own pattern survives.
    """
    trained = build(kind)
    (param,) = trained.parameters()
    with torch.no_grad():
        param.mul_(1.7)
    buffer = io.BytesIO()
    torch.save(trained.state_dict(), buffer)
    buffer.seek(0)

    fresh = build(kind, PRE_B, POST_B)
    x = make_input(kind)
    assert not torch.allclose(fresh(x), trained(x))
    fresh.load_state_dict(torch.load(buffer))
    compiled = torch.compile(fresh, fullgraph=True)

    leaf = (leaf_of(kind) * 1.7).detach()
    expected = dense_forward(dense_twin(kind, leaf), x)
    for _ in range(2):
        out, grads = loss_and_grads(compiled, fresh, x)
        torch.testing.assert_close(out, expected.detach())
        _, grads_trained = loss_and_grads(trained, trained, x)
        torch.testing.assert_close(grads[1], grads_trained[1])
        if kind == "constrained":  # scales are ordered by group, not by slot
            torch.testing.assert_close(grads[0], grads_trained[0])
        else:
            torch.testing.assert_close(
                param_grad_as_dense(fresh, grads[0]),
                param_grad_as_dense(trained, grads_trained[0]),
            )


@pytest.mark.parametrize("kind", ["edge", "constrained"])
def test_checkpoint_loaded_after_compile_is_used(kind):
    """Loading a checkpoint with another pattern into an *already compiled*
    module changes what the compiled module computes (the derived layouts are
    rebuilt in place, never left stale inside the compiled graph)."""
    conn = build(kind)
    compiled = torch.compile(conn, fullgraph=True)
    x = make_input(kind).detach()
    torch.testing.assert_close(
        compiled(x), dense_forward(dense_twin(kind, leaf_of(kind)), x)
    )

    other = build(kind, PRE_B, POST_B)
    conn.load_state_dict(other.state_dict())
    expected = dense_forward(dense_twin(kind, leaf_of(kind), PRE_B, POST_B), x)
    torch.testing.assert_close(conn(x), expected.detach())
    torch.testing.assert_close(compiled(x), expected.detach())
