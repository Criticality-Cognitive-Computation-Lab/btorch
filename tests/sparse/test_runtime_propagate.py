"""The eager entry point ``ops.propagate`` and operators in ``sparse.matmul``.

Two things are tested here.

**1. One product, two routes.** A connection computes ``y = A @ x`` by calling
:func:`btorch.sparse.runtime.ops.propagate`. That function has two routes:

- under ``torch.compile`` it emits one *registered operator*
  (``csr_propagate``, or ``spike_propagate`` when a density limit is given),
  which is what Dynamo can trace;
- in eager mode it runs the same kernels through a lighter autograd node
  (``_Propagate``), and with no gradient required it calls the kernels
  directly, without any autograd node.

The routes must be indistinguishable: same output, same gradient with respect
to the input, same gradient with respect to the edge values. Every test
compares them with each other *and* with a dense matrix that is built by a
NumPy scatter, independently of ``btorch.sparse``.

The *algorithm* is chosen by ``max_density``: ``None`` is the
destination-driven product ("pull"); a number selects the source-driven
product ("push") for inputs whose non-zero fraction is at most that number,
with a fallback to pull above it. ``1.0`` therefore always pushes and ``0.0``
always falls back, which lets small tests reach both branches.

**2. Operators that are not sparse arrays** (``ConstantOperator``,
``ImplicitOperator``, lazy sums and products) are accepted by
``sparse.matmul`` / ``sparse.matvec`` / ``sparse.rmatvec``.
"""

import numpy as np
import pytest
import torch
from torch import nn

from btorch import sparse
from btorch.models.connection import SparseConnection, Synapse
from btorch.sparse import Hints
from btorch.sparse.operator import (
    ConstantOperator,
    DiagonalOperator,
    ImplicitOperator,
    is_linear_operator,
)
from btorch.sparse.runtime import ops, registry
from tests.sparse.helpers import DEVICES


M, N = 5, 7  # non-square: a transposed product has the wrong shape
TOL = {"atol": 1e-10, "rtol": 1e-10}  # everything below runs in float64

# name -> max_density handed to ``ops.propagate`` (see the module docstring).
ROUTES = {"pull": None, "push": 1.0, "push_fallback": 0.0}


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    """Each test compiles from scratch (no recompile limits, no reuse)."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# ------------------------------------------------------------------ builders
def make_layout(variant="plain", batch=(), device="cpu", seed=0):
    """Random operator ``A [M, N]`` as raw buffers plus its dense tensor.

    The buffers are what every propagation operator takes: the
    destination-major CSR ``crow [M + 1]``, ``col [E]`` with ``values
    [*batch, E]`` in that order, and the same edges sorted by source,
    ``t_crow [N + 1]``, ``t_col [E]`` (destination of every source-major
    entry) and ``t_perm [E]`` (its position in the CSR order).

    Everything is derived with NumPy sorts; nothing of btorch is involved.

    Args:
        variant: ``"plain"``, ``"empty_rows"`` (rows 1 and 4 and column 0 hold
            no entry) or ``"nnz0"`` (no entry at all).
        batch: Shape of a shared-pattern batch of values (``G`` networks).
        device: Device of all returned tensors.
        seed: Seed of the NumPy generator.

    Returns:
        ``(crow, col, values, t_crow, t_col, t_perm)`` and the dense
        ``[*batch, M, N]`` reference, all float64 / int64.
    """
    rng = np.random.default_rng(seed)
    flat = np.sort(rng.choice(M * N, size=M * N // 3, replace=False))
    row, col = flat // N, flat % N  # row-major sorted == CSR order
    if variant == "empty_rows":
        keep = (row != 1) & (row != 4) & (col != 0)
        row, col = row[keep], col[keep]
    elif variant == "nnz0":
        row, col = row[:0], col[:0]
    values = rng.normal(size=(*batch, len(row)))
    dense = np.zeros((*batch, M, N))
    dense[..., row, col] = values
    crow = np.concatenate([[0], np.cumsum(np.bincount(row, minlength=M))])
    t_perm = np.lexsort((row, col))  # sort by source, then by destination
    t_crow = np.concatenate([[0], np.cumsum(np.bincount(col, minlength=N))])
    crow, col, t_crow, t_col, t_perm = (
        torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)
        for a in (crow, col, t_crow, row[t_perm], t_perm)
    )
    values = torch.as_tensor(values, dtype=torch.float64, device=device)
    dense = torch.as_tensor(dense, dtype=torch.float64, device=device)
    return (crow, col, values, t_crow, t_col, t_perm), dense


def eager(buffers, x, values, max_density):
    """The route connections take outside ``torch.compile``."""
    crow, col, _, t_crow, t_col, t_perm = buffers
    return ops.propagate(crow, col, values, x, t_crow, t_col, t_perm, max_density)


def registered(buffers, x, values, max_density):
    """The registered operator that compiled code calls for the same
    ``max_density``."""
    crow, col, _, t_crow, t_col, t_perm = buffers
    if max_density is None:
        return ops.csr_propagate(crow, col, values, x, t_crow, t_col, t_perm)
    return ops.spike_propagate(crow, col, values, x, t_crow, t_col, t_perm, max_density)


def dense_apply(dense, x):
    """Reference ``y = A @ x`` along the last axis.

    With a value batch, network ``g`` of ``A`` is paired with ``x[g]`` (a
    leading dimension of size 1 broadcasts).
    """
    lead = "ab"[: dense.ndim - 2]
    return torch.einsum(f"{lead}mn,{lead}...n->{lead}...m", dense, x)


def loss_grads(out, *leaves):
    """Gradients of a fixed scalar loss; ``None`` for a leaf without
    ``requires_grad``."""
    wanted = [leaf for leaf in leaves if leaf.requires_grad]
    grads = iter(torch.autograd.grad(out.square().sum(), wanted) if wanted else ())
    return [next(grads) if leaf.requires_grad else None for leaf in leaves]


# One entry per situation: (variant, value batch, shape of x).
SAMPLES = {
    "vector": ("plain", (), (N,)),
    "batch": ("plain", (), (3, N)),
    "time_batch": ("plain", (), (2, 3, N)),
    "empty_rows": ("empty_rows", (), (3, N)),
    "nnz0": ("nnz0", (), (3, N)),
    "value_batch": ("plain", (2,), (2, 3, N)),
    # A size-1 network dimension of x broadcasts against the value batch, so
    # the gradient w.r.t. x has to be summed back over the networks.
    "value_batch_broadcast": ("plain", (2,), (1, 3, N)),
}


def make_sample(name, device="cpu", seed=1):
    variant, batch, x_shape = SAMPLES[name]
    buffers, dense = make_layout(variant, batch, device)
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(x_shape, generator=gen, dtype=torch.float64).to(device)
    return buffers, dense, x


# ------------------------------------------------- eager == registered == dense
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("route", list(ROUTES))
@pytest.mark.parametrize("name", list(SAMPLES))
def test_eager_route_equals_registered_operator_and_dense(name, route, device):
    """Output and both gradients are the same on the eager route, through the
    registered operator and for the dense reference.

    The dense reference is differentiated by autograd's own dense rules:
    ``dL/dA`` is a full matrix and the edge values receive its entries at the
    stored coordinates.
    """
    buffers, dense, x = make_sample(name, device)
    crow, col = buffers[0], buffers[1]
    max_density = ROUTES[route]

    results = {}
    for label, fn in (("eager", eager), ("registered", registered)):
        values = buffers[2].clone().requires_grad_()
        xi = x.clone().requires_grad_()
        out = fn(buffers, xi, values, max_density)
        results[label] = (out, *loss_grads(out, values, xi))

    # The eager route is the light autograd node, not the operator dispatch.
    assert type(results["eager"][0].grad_fn).__name__ == "_PropagateBackward"
    assert "propagate" in type(results["registered"][0].grad_fn).__name__
    assert type(results["registered"][0].grad_fn).__name__ != "_PropagateBackward"

    # Dense reference.
    A = dense.clone().requires_grad_()
    x_ref = x.clone().requires_grad_()
    expected = dense_apply(A, x_ref)
    grad_A, grad_x = torch.autograd.grad(expected.square().sum(), (A, x_ref))
    # CSR row of every entry, to read dL/dA at the stored coordinates.
    row = torch.repeat_interleave(torch.arange(M, device=device), crow[1:] - crow[:-1])
    grad_values = grad_A[..., row, col]

    for label, (out, g_values, g_x) in results.items():
        assert out.shape == expected.shape and out.dtype == torch.float64, label
        torch.testing.assert_close(out, expected.detach(), **TOL, msg=label)
        assert g_values.shape == buffers[2].shape and g_x.shape == x.shape, label
        torch.testing.assert_close(g_values, grad_values, **TOL, msg=label)
        torch.testing.assert_close(g_x, grad_x, **TOL, msg=label)
    # ... and therefore with each other.
    for a, b in zip(results["eager"], results["registered"]):
        torch.testing.assert_close(a, b, **TOL)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("route", list(ROUTES))
@pytest.mark.parametrize(
    "x_grad, values_grad",
    [(True, True), (True, False), (False, True), (False, False)],
    ids=["both", "only_input", "only_values", "neither"],
)
def test_eager_route_for_every_requires_grad_combination(
    route, device, x_grad, values_grad
):
    """Only the tensors that ask for a gradient get one, and with no gradient
    required at all the kernels run without an autograd node."""
    buffers, dense, x = make_sample("batch", device)
    values = buffers[2].clone().requires_grad_(values_grad)
    x = x.clone().requires_grad_(x_grad)

    out = eager(buffers, x, values, ROUTES[route])
    torch.testing.assert_close(out.detach(), dense_apply(dense, x.detach()), **TOL)
    assert out.requires_grad == (x_grad or values_grad)
    if not (x_grad or values_grad):
        assert out.grad_fn is None
        return

    g_values, g_x = loss_grads(out, values, x)
    # Same loss through the registered operator.
    ref = registered(buffers, x, values, ROUTES[route])
    r_values, r_x = loss_grads(ref, values, x)
    assert (g_values is None) == (not values_grad) and (g_x is None) == (not x_grad)
    if values_grad:
        torch.testing.assert_close(g_values, r_values, **TOL)
    if x_grad:
        # dL/dx = A^T (2 y), written out by hand.
        torch.testing.assert_close(g_x, 2 * out.detach() @ dense, **TOL)
        torch.testing.assert_close(g_x, r_x, **TOL)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("route", list(ROUTES))
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_eager_route_without_grad_mode(context, route, device):
    """Under ``torch.no_grad()`` / ``torch.inference_mode()`` the kernels are
    called directly even when the leaves require a gradient: the result has no
    graph and equals the registered operator and the dense reference."""
    buffers, dense, x = make_sample("time_batch", device)
    values = buffers[2].clone().requires_grad_()
    x = x.clone().requires_grad_()
    with context():
        out = eager(buffers, x, values, ROUTES[route])
        ref = registered(buffers, x, values, ROUTES[route])
        assert out.grad_fn is None and not out.requires_grad
        torch.testing.assert_close(out, ref, **TOL)
        torch.testing.assert_close(out, dense_apply(dense, x), **TOL)


def test_max_density_selects_the_kernel(monkeypatch):
    """``max_density`` is what decides between pull and push in eager mode.

    The kernel lookups of one call are recorded. On the CPU the reference
    push packs the active inputs on the host and is used only while their
    fraction is at most ``max_density``; denser input falls back to pull.
    """
    requested = []
    resolve = registry.resolve

    def recording_resolve(kernel, device):
        requested.append(kernel)
        return resolve(kernel, device)

    # Every kernel call goes through the registry (the public lookup).
    monkeypatch.setattr(registry, "resolve", recording_resolve)
    buffers, dense, _ = make_sample("batch")
    values = buffers[2]
    # 3 x 7 = 21 inputs, 2 of them active: density 2 / 21 = 0.095.
    x = torch.zeros(3, N, dtype=torch.float64)
    x[0, 2], x[2, 5] = 1.0, -2.0

    for max_density, kernel in (
        (None, "csr_matvec"),
        (0.1, "spike_push"),
        (0.09, "csr_matvec"),
    ):
        requested.clear()
        out = eager(buffers, x, values, max_density)
        assert requested == [kernel], max_density
        torch.testing.assert_close(out, dense_apply(dense, x), **TOL)


# ------------------------------------------------------------------ gradcheck
@pytest.mark.parametrize("route", list(ROUTES))
@pytest.mark.parametrize("name", ["vector", "batch", "empty_rows", "value_batch"])
def test_gradcheck_through_the_eager_route(name, route):
    """Finite differences agree with the hand-written backward of the eager
    autograd node (float64, CPU), for the edge values and for the input."""
    buffers, _, x = make_sample(name)
    values = buffers[2].clone().requires_grad_()
    x = x.clone().requires_grad_()

    def fn(values, x):
        out = eager(buffers, x, values, ROUTES[route])
        assert type(out.grad_fn).__name__ == "_PropagateBackward"
        return out

    assert torch.autograd.gradcheck(fn, (values, x), atol=1e-6, rtol=1e-4)


# ------------------------------------------------------------ double backward
# The backward of the propagation is written by hand and is not itself
# differentiable. A second-order request must therefore fail; it must never
# produce a number, because that number would silently miss the term that
# goes through the connection.
def _mixed_second_order(apply, values, x, wrt):
    """``d/d(wrt) sum(dL/d(wrt))`` for ``L = sum(y^2) + sum(wrt^3)``.

    The cubic term gives the first-order gradient a second, ordinary path to
    ``wrt``: this is the shape of a gradient penalty or of a meta-learning
    loss, where the connection is only one of several contributions.
    """
    leaf = {"x": x, "values": values}[wrt]
    loss = apply(values, x).square().sum() + leaf.pow(3).sum()
    (first,) = torch.autograd.grad(loss, leaf, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), leaf)
    return second


def _dense_from_values(buffers, values):
    """Differentiable dense twin ``A [M, N]`` of the CSR ``values``."""
    crow, col = buffers[0], buffers[1]
    row = torch.repeat_interleave(torch.arange(M), crow[1:] - crow[:-1])
    return torch.zeros(M, N, dtype=values.dtype).index_put((row, col), values)


def _second_order_case():
    buffers, _, x = make_sample("batch")
    return buffers, buffers[2].clone().requires_grad_(), x.clone().requires_grad_()


@pytest.mark.parametrize("route", list(ROUTES))
def test_double_backward_through_the_eager_route_raises(route):
    """Asking for a differentiable gradient of the eager route is refused.

    The kernels are not differentiable, so a second derivative cannot be
    formed. The request is rejected as soon as the first backward is run with
    ``create_graph=True``, with an error that names the reason, rather than
    later (or never) when the missing branch would silently be pruned.
    """
    buffers, values, x = _second_order_case()
    out = eager(buffers, x, values, ROUTES[route])
    with pytest.raises(RuntimeError, match="double backward"):
        torch.autograd.grad(out.square().sum(), (values, x), create_graph=True)
    # An ordinary first-order backward of the same graph is unaffected.
    out = eager(buffers, x, values, ROUTES[route])
    grads = torch.autograd.grad(out.square().sum(), (values, x))
    assert all(g is not None and not g.requires_grad for g in grads)


@pytest.mark.parametrize("wrt", ["x", "values"])
@pytest.mark.parametrize("route", list(ROUTES))
def test_double_backward_through_the_registered_operator_raises(route, wrt):
    """The registered operators refuse a second derivative even when the
    quantity being differentiated has another path to the leaf."""
    buffers, values, x = _second_order_case()

    def apply(values, x):
        return registered(buffers, x, values, ROUTES[route])

    with pytest.raises(RuntimeError, match="no autograd formula"):
        _mixed_second_order(apply, values, x, wrt)


@pytest.mark.parametrize("wrt", ["x", "values"])
@pytest.mark.parametrize("route", list(ROUTES))
def test_double_backward_never_returns_wrong_numbers(route, wrt):
    """A second derivative through the eager route either raises or is right.

    The reference is the same expression through a dense matrix, which
    autograd differentiates twice without restrictions.
    """
    buffers, values, x = _second_order_case()

    def apply(values, x):
        return eager(buffers, x, values, ROUTES[route])

    def apply_dense(values, x):
        return x @ _dense_from_values(buffers, values).T

    expected = _mixed_second_order(apply_dense, values, x, wrt)
    try:
        got = _mixed_second_order(apply, values, x, wrt)
    except RuntimeError:
        return  # a refusal is the designed behaviour
    torch.testing.assert_close(got, expected, **TOL)


# ------------------------------------------------- connections: eager == compiled
# Large enough that a few spikes are below the CPU push limit of 0.2 %:
# 4 samples x 500 inputs = 2000 entries, of which 3 are active (0.15 %).
N_PRE, N_POST, N_EDGE, N_SAMPLE = 500, 30, 400, 4
_gen = torch.Generator().manual_seed(0)
PRE = torch.randint(0, N_PRE, (N_EDGE,), generator=_gen)
POST = torch.randint(0, N_POST, (N_EDGE,), generator=_gen)
WEIGHT = torch.randn(N_EDGE, generator=_gen, dtype=torch.float64)
# The expected density of the hint; the planner turns it into a source-driven
# plan: "push" on a backend that compacts the input on the device (Triton on
# CUDA), "adaptive-push" on the reference backend (CPU, or CUDA under
# ``use_backend("aten")``).
KINDS = {"pull": None, "push": Hints(expected_density=0.001)}


def build_connection(kind, device):
    """A trainable connection whose plan is pull or (hinted) adaptive push."""
    conn = SparseConnection.from_edges(
        PRE,
        POST,
        N_PRE,
        N_POST,
        Synapse(weight=nn.Parameter(WEIGHT.clone())),
        hints=KINDS[kind],
        device=device,
    )
    # Guard against a planner change that would silently turn the push case
    # into a second pull case.
    # The name depends on the selected backend, not on the device name.
    device_type = torch.device(device).type
    source_driven = "adaptive-push"
    if registry.has("spike_push_dense", device_type) and WEIGHT.dtype == torch.float32:
        source_driven = "push"
    elif device_type == "cuda":
        source_driven = "pull"
    algorithm = source_driven if kind == "push" else "pull"
    assert f"algorithm = {algorithm} (" in conn.explain()
    return conn


def dense_twin(weight):
    """``A [N_POST, N_PRE]`` as a differentiable function of the per-edge
    weights in input order; parallel edges add up."""
    A = torch.zeros(N_POST, N_PRE, dtype=weight.dtype, device=weight.device)
    return A.index_put((POST.to(weight.device), PRE.to(weight.device)), weight, True)


def make_spikes(density, device):
    """``[N_SAMPLE, N_PRE]`` input: 3 isolated spikes or dense activity."""
    x = torch.zeros(N_SAMPLE, N_PRE, dtype=torch.float64)
    if density == "sparse":
        x[0, 17], x[2, 17], x[3, 311] = 1.0, 1.0, 1.0
    else:
        x = torch.rand(N_SAMPLE, N_PRE, generator=torch.Generator().manual_seed(3))
        x = x.double()
    return x.to(device)


def test_compiled_connection_calls_the_registered_operator():
    """Dynamo sees exactly one call per forward: the registered operator of
    the planned algorithm. The eager module uses the light autograd node.

    A pass-through compile backend records the calls of the captured graph.
    """
    for kind in ("pull", "push"):
        conn = build_connection(kind, "cpu")
        calls = []

        def record(gm, example_inputs, calls=calls):
            calls.extend(
                str(n.target) for n in gm.graph.nodes if n.op == "call_function"
            )
            return gm.forward

        x = make_spikes("sparse", "cpu").requires_grad_()
        compiled = torch.compile(conn, backend=record, fullgraph=True)
        out = compiled(x)
        assert calls == ["btorch.route_propagate.default"]
        assert type(conn(x).grad_fn).__name__ == "_BoundPropagateBackward"
        torch.testing.assert_close(out, conn(x), **TOL)
        torch._dynamo.reset()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize(
    "x_grad, weight_grad",
    [(True, True), (True, False), (False, True), (False, False)],
    ids=["both", "only_input", "only_weights", "neither"],
)
def test_connection_eager_equals_compiled(kind, device, x_grad, weight_grad):
    """``torch.compile(conn)`` and ``conn`` give the same output and the same
    gradients, for sparse and for dense activity, whichever of input and
    weights asks for a gradient.

    Both equal the dense twin.
    """
    conn = build_connection(kind, device)
    conn.weight.value.requires_grad_(weight_grad)
    compiled = torch.compile(conn, fullgraph=True)
    weight = conn.weight.value
    post, pre = conn.indices

    for density in ("sparse", "dense"):
        x = make_spikes(density, device).requires_grad_(x_grad)
        out_e = conn(x)
        out_c = compiled(x)

        # Dense twin, differentiated by autograd's dense rules.
        A = dense_twin(WEIGHT.to(device)).requires_grad_()
        x_ref = x.detach().clone().requires_grad_()
        expected = x_ref @ A.T
        grad_A, grad_x = torch.autograd.grad(expected.square().sum(), (A, x_ref))

        for label, out in (("eager", out_e), ("compiled", out_c)):
            msg = f"{label}, {density} input"
            torch.testing.assert_close(out.detach(), expected.detach(), **TOL, msg=msg)
            assert out.requires_grad == (x_grad or weight_grad), msg
            g_weight, g_x = loss_grads(out, weight, x)
            if weight_grad:
                # Slot k is the edge post[k] <- pre[k]; merged parallel edges
                # share one slot and therefore one entry of dL/dA.
                torch.testing.assert_close(g_weight, grad_A[post, pre], **TOL, msg=msg)
            if x_grad:
                torch.testing.assert_close(g_x, grad_x, **TOL, msg=msg)
        if not (x_grad or weight_grad):
            assert out_e.grad_fn is None  # kernels were called directly
        torch.testing.assert_close(out_c, out_e, **TOL)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_connection_eager_equals_compiled_without_grad_mode(context, kind, device):
    """Inference: under ``no_grad`` / ``inference_mode`` the eager and the
    compiled connection agree and neither builds a graph, although the
    weights are trainable."""
    conn = build_connection(kind, device)
    compiled = torch.compile(conn, fullgraph=True)
    A = dense_twin(WEIGHT.to(device))
    for density in ("sparse", "dense"):
        x = make_spikes(density, device)
        with context():
            out_e, out_c = conn(x), compiled(x)
        assert out_e.grad_fn is None and not out_e.requires_grad
        assert not out_c.requires_grad
        torch.testing.assert_close(out_e, x @ A.T, **TOL)
        torch.testing.assert_close(out_c, out_e, **TOL)


# ------------------------------------- sparse.matmul / matvec / rmatvec: operators
# Operators of shape (M, N) that are not sparse arrays, each with the dense
# matrix it stands for (written with ordinary dense algebra).
def _constant_case():
    return ConstantOperator((M, N), 0.7, dtype=torch.float64), torch.full(
        (M, N), 0.7, dtype=torch.float64
    )


def _implicit_case():
    # The operator only knows two callables; they happen to apply ``W``.
    W = torch.randn(
        M, N, dtype=torch.float64, generator=torch.Generator().manual_seed(4)
    )
    op = ImplicitOperator(
        (M, N), matvec=lambda x: x @ W.T, rmatvec=lambda x: x @ W, dtype=torch.float64
    )
    return op, W


def _sum_case():
    # constant + explicit sparse: a lazy sum that never adds the matrices.
    (crow, col, values, *_), dense = make_layout()
    row = torch.repeat_interleave(torch.arange(M), crow[1:] - crow[:-1])
    A = sparse.from_edges(row, col, values, (M, N))
    return ConstantOperator((M, N), -0.3, dtype=torch.float64) + A, dense - 0.3


def _product_case():
    # (M, N) = diagonal (M, M) @ implicit (M, N).
    op, W = _implicit_case()
    diag = torch.arange(1.0, M + 1, dtype=torch.float64)
    return DiagonalOperator(diag) @ op, torch.diag(diag) @ W


OPERATORS = {
    "constant": _constant_case,
    "implicit": _implicit_case,
    "sum": _sum_case,
    "product": _product_case,
}


def _randn(*shape, seed=5):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=torch.float64, generator=gen)


@pytest.mark.parametrize("lead", [(), (3,), (2, 3)], ids=["vector", "batch", "time"])
@pytest.mark.parametrize("name", list(OPERATORS))
def test_matvec_and_rmatvec_accept_operators(name, lead):
    """``sparse.matvec(A, x)`` is ``A @ x`` and ``sparse.rmatvec(A, x)`` is
    ``A^T @ x`` along the last axis, for operators that store no matrix."""
    op, dense = OPERATORS[name]()
    assert is_linear_operator(op) and not isinstance(op, sparse.Sparse)
    x, y = _randn(*lead, N), _randn(*lead, M, seed=6)
    torch.testing.assert_close(sparse.matvec(op, x), x @ dense.T, **TOL)
    torch.testing.assert_close(sparse.rmatvec(op, y), y @ dense, **TOL)


@pytest.mark.parametrize("name", list(OPERATORS))
def test_matmul_accepts_operators_on_either_side(name):
    """``sparse.matmul`` follows ``torch.matmul`` with the operator as either
    operand, for matrix and for vector operands."""
    op, dense = OPERATORS[name]()
    for other in (_randn(N, 2), _randn(N)):  # A @ B, A @ v
        torch.testing.assert_close(sparse.matmul(op, other), dense @ other, **TOL)
    for other in (_randn(2, M), _randn(M)):  # B @ A, v @ A
        torch.testing.assert_close(sparse.matmul(other, op), other @ dense, **TOL)


def test_matmul_of_two_operators_stays_lazy():
    """Operator times operator (or times a sparse array) is not evaluated: the
    result is again an operator, and applying it gives the dense product."""
    op, dense = _constant_case()
    diag = torch.arange(1.0, N + 1, dtype=torch.float64)
    (crow, col, values, *_), sparse_dense = make_layout()
    row = torch.repeat_interleave(torch.arange(M), crow[1:] - crow[:-1])
    A = sparse.from_edges(row, col, values, (M, N))
    x = _randn(3, N)
    cases = [
        (sparse.matmul(op, DiagonalOperator(diag)), dense @ torch.diag(diag)),
        # (N, M) @ (M, N) -> (N, N), with a sparse array on either side.
        (sparse.matmul(op.T, A), dense.T @ sparse_dense),
        (sparse.matmul(A.T, op), sparse_dense.T @ dense),
    ]
    for product, expected in cases:
        assert is_linear_operator(product) and not isinstance(product, torch.Tensor)
        torch.testing.assert_close(sparse.matvec(product, x), x @ expected.T, **TOL)


def test_sparse_arrays_still_take_the_sparse_path():
    """The operator branch does not change the result for a sparse array."""
    (crow, col, values, *_), dense = make_layout()
    row = torch.repeat_interleave(torch.arange(M), crow[1:] - crow[:-1])
    A = sparse.from_edges(row, col, values, (M, N))
    x, y, B = _randn(3, N), _randn(3, M, seed=6), _randn(N, 2, seed=7)
    torch.testing.assert_close(sparse.matvec(A, x), x @ dense.T, **TOL)
    torch.testing.assert_close(sparse.rmatvec(A, y), y @ dense, **TOL)
    torch.testing.assert_close(sparse.matmul(A, B), dense @ B, **TOL)
