"""Integration tests for the recording engine on the RNN loop.

``update_state_names`` is the single recording spec: plain dotted strings populate
the classic ``stacked_states`` return; :class:`~btorch.monitor.Expr` / ``grad(...)``
/ ``{name: expr}`` specs (see :mod:`btorch.monitor`) are read via ``get_records()``.
These tests double as usage examples and pin the compatibility surface: torch.compile,
grad_checkpoint, cpu_offload, and cudagraph.

The cell is ``SimpleRNNCell`` whose single hidden state is the dotted name ``"h"``;
``net(x)`` returns ``(outputs, states)`` where ``outputs`` is ``h`` stacked over T.
"""

import platform

import pytest
import torch

from btorch.models.functional import reset_net_state
from btorch.models.rnn import make_rnn
from btorch.monitor import Reducer, col, grad

from .rnn_utils import DTYPE, SimpleRNNCell


T, B, DIN, H = 12, 2, 4, 8


class EMA(Reducer):
    """Exponential moving average -- a stateful custom reduction (functional
    carry)."""

    def __init__(self, alpha):
        self.alpha = alpha

    def init(self, example):
        return torch.zeros_like(example)

    def update(self, carry, value, valid):
        return self.alpha * value + (1 - self.alpha) * carry

    def finalize(self, carry):
        return carry


def recording_rnn(update_state_names=None, seed=0, **kwargs):
    """A ``make_rnn`` over ``SimpleRNNCell`` with ``update_state_names=``,
    reset."""
    torch.manual_seed(seed)
    cell = SimpleRNNCell(DIN, H)
    net = make_rnn(cell, update_state_names=update_state_names, **kwargs)
    reset_net_state(net, batch_size=B)
    return net


def random_inputs(seed=1):
    """A ``[T, B, DIN]`` input sequence (fixed seed -> reproducible)."""
    torch.manual_seed(seed)
    return torch.randn(T, B, DIN, dtype=DTYPE)


# ===========================================================================
# Usage examples
# ===========================================================================
def test_bare_string_records_the_hidden_state_into_stacked_states():
    net = recording_rnn(update_state_names=["h"])
    outputs, states = net(random_inputs())
    # a dotted string records the hidden state each step into the states return
    torch.testing.assert_close(states["h"], outputs)


def test_col_expr_records_the_raw_trace_into_records():
    # col("h") (an Expr, not a bare string) routes to get_records() instead;
    # strings and Exprs mix freely in one spec -- each goes to its own channel
    net = recording_rnn(
        update_state_names=[
            col("h").alias("h_trace"),
            "h",
            col("h").mean().alias("h_mean"),
        ]
    )
    outputs, states = net(random_inputs())
    torch.testing.assert_close(net.get_records()["h_trace"], outputs)
    torch.testing.assert_close(states["h"], outputs)  # string -> stacked_states
    torch.testing.assert_close(net.get_records()["h_mean"], outputs.mean(0))


def test_reductions_and_derived_quantities():
    net = recording_rnn(
        update_state_names={
            "h_mean": col("h").mean(),
            "h_max": col("h").max(),
            "mean_speed": col("h").diff().mean(),  # mean |Δh| over time
        }
    )
    outputs, _ = net(random_inputs())
    records = net.get_records()
    torch.testing.assert_close(records["h_mean"], outputs.mean(0))
    torch.testing.assert_close(records["h_max"], outputs.max(0).values)
    speed = (outputs[1:] - outputs[:-1]).mean(0)
    torch.testing.assert_close(records["mean_speed"], speed)


def test_no_expr_specs_builds_no_recorder():
    net = recording_rnn(update_state_names=["h"])  # plain string: legacy path only
    assert net._recorder is None
    net(random_inputs())
    assert net.get_records() == {}


def test_plain_buffers_need_allow_buffer_like_the_legacy_path():
    # policy parity: by default the recorder accepts exactly what the legacy
    # stacked_states collection accepts -- memories of MemoryModules.  A plain
    # (non-memory) buffer target is REFUSED instead of being silently skipped;
    # allow_buffer=True explicitly unblocks it on the recorder channel.
    torch.manual_seed(0)
    cell = SimpleRNNCell(DIN, H)
    cell.register_buffer("aux", torch.zeros(B, H, dtype=DTYPE))  # NOT a memory

    refused = make_rnn(cell, update_state_names=[col("aux")])
    reset_net_state(refused, batch_size=B)
    with pytest.raises(KeyError, match="allow_buffer"):
        refused(random_inputs())

    recorded = make_rnn(
        cell, update_state_names={"aux_sq": col("aux") ** 2}, allow_buffer=True
    )
    reset_net_state(recorded, batch_size=B)
    recorded(random_inputs())
    assert recorded.get_records()["aux_sq"].shape == (T, B, H)


def test_records_follow_a_batch_size_change_after_reset():
    # reset with a new batch size -> the recorder re-infers its carry layout;
    # a stale carry would broadcast-corrupt reductions (regression guard for
    # the engine-level fix, driven end-to-end through the loop).
    net = recording_rnn(update_state_names={"mean": col("h").mean()})
    net(random_inputs())
    assert net.get_records()["mean"].shape == (B, H)

    reset_net_state(net, batch_size=B + 3)
    net(torch.randn(T, B + 3, DIN, dtype=DTYPE))
    assert net.get_records()["mean"].shape == (B + 3, H)


def test_raw_records_keep_grad_but_reductions_are_detached():
    # raw records stay grad-connected (train through them, like stacked_states);
    # reductions are observations and are detached.
    net = recording_rnn(update_state_names={"raw": col("h"), "mean": col("h").mean()})
    net(random_inputs())
    records = net.get_records()
    assert records["raw"].requires_grad
    assert not records["mean"].requires_grad


# ===========================================================================
# Execution-mode compatibility: records must be invariant to how the loop runs
# ===========================================================================
# The default reference (unroll=8 over T=12) already exercises multi-block
# eager with a partial final block; unroll=1/4/full repeat the same code
# paths, so only modes with DISTINCT machinery are parametrised.
_EXECUTION_MODES = [
    pytest.param({"unroll": 4, "chunk_size": 8}, id="chunked"),
    pytest.param({"unroll": 4, "grad_checkpoint": True}, id="grad_checkpoint"),
    pytest.param({"unroll": 4, "cpu_offload": True}, id="cpu_offload"),
]


@pytest.mark.parametrize("mode", _EXECUTION_MODES)
def test_records_are_invariant_across_execution_modes(mode):
    # value records AND a stateful custom Reducer (EMA) written as a functional
    # tensor carry inherit correctness across every execution mode.
    specs = {
        "raw": col("h"),
        "mean": col("h").mean(),
        "speed": col("h").diff().mean(),
        "ema": col("h").fold(EMA(0.3)),
    }

    reference = recording_rnn(update_state_names=dict(specs))
    reference(random_inputs())
    expected = {k: v.clone() for k, v in reference.get_records().items()}

    net = recording_rnn(update_state_names=dict(specs), **mode)
    net(random_inputs())
    for key, got in net.get_records().items():
        want = expected[key]
        got = got.to(want.device)  # cpu_offload moves records to CPU
        torch.testing.assert_close(got, want, msg=f"{key} under {mode}")


@pytest.mark.skipif(platform.system() != "Linux", reason="torch.compile: Linux only")
def test_records_match_eager_under_torch_compile():
    # includes a custom fold(reducer) -- its pytree carry must trace cleanly too
    specs = {
        "raw": col("h"),
        "mean": col("h").mean(),
        "speed": col("h").diff().mean(),
        "ema": col("h").fold(EMA(0.3)),
    }

    eager = recording_rnn(update_state_names=dict(specs))
    eager(random_inputs())
    expected = {k: v.clone() for k, v in eager.get_records().items()}

    torch._dynamo.reset()
    net = recording_rnn(update_state_names=dict(specs))
    torch.compile(net)(random_inputs())
    for key, want in expected.items():
        torch.testing.assert_close(net.get_records()[key], want, msg=key)


# ===========================================================================
# Gradient monitors (eager, backward-time)
# ===========================================================================
def test_grad_monitor_captures_per_step_hidden_state_gradients():
    # grad("h") records d(loss)/d(h_t) for every step via backward hooks.
    net = recording_rnn(update_state_names={"h_grad": grad("h")})
    outputs, _ = net(random_inputs())
    outputs.sum().backward()
    grads = net.get_records()["h_grad"]
    assert len(grads) == T
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    # the last step feeds no future step, so d(sum outputs)/d(h_{T-1}) == 1
    torch.testing.assert_close(grads[-1], torch.ones_like(grads[-1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_grad_monitor_captures_under_cpu_offload_on_cuda():
    # grad sources must stay on-device under cpu_offload, else the backward hooks
    # fire on offloaded copies that are off the backward path and capture nothing.
    net = recording_rnn(
        update_state_names={"h_grad": grad("h")}, cpu_offload=True
    ).cuda()
    reset_net_state(net, batch_size=B)
    net(random_inputs().cuda())[0].sum().backward()
    grads = net.get_records()["h_grad"]
    assert len(grads) == T and all(g is not None for g in grads)


# ===========================================================================
# cudagraph: value records replay; grad / multi-chunk are refused
# ===========================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cudagraph needs CUDA")
def test_cudagraph_replays_value_records():
    # value/reduction records are folded inside the captured chunk graph, so replay
    # recomputes them like the states -- including a custom fold(reducer).
    specs = {
        "raw": col("h"),
        "mean": col("h").mean(),
        "speed": col("h").diff().mean(),
        "ema": col("h").fold(EMA(0.3)),
    }

    eager = recording_rnn(update_state_names=dict(specs)).cuda()
    reset_net_state(eager, batch_size=B)
    with torch.no_grad():
        eager(random_inputs().cuda())
    expected = {k: v.clone() for k, v in eager.get_records().items()}

    graphed = recording_rnn(update_state_names=dict(specs), cudagraph=True).cuda()
    reset_net_state(graphed, batch_size=B)
    with torch.no_grad():
        graphed(random_inputs().cuda())
    for key, want in expected.items():
        torch.testing.assert_close(graphed.get_records()[key], want, msg=key)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cudagraph needs CUDA")
def test_cudagraph_refuses_grad_monitors_and_multichunk_records():
    # two recording refusals under replay: (a) grad hooks only fire in backward,
    # which replay never runs; (b) a streaming reduction cannot be finalised
    # across separate per-chunk replays.
    grad_net = recording_rnn(
        update_state_names={"h_grad": grad("h")}, cudagraph=True
    ).cuda()
    reset_net_state(grad_net, batch_size=B)
    with torch.no_grad():  # reach the grad-monitor refusal, not the requires-grad guard
        with pytest.raises(RuntimeError, match=r"grad\(\.\.\.\) monitors"):
            grad_net(random_inputs().cuda())

    chunked = recording_rnn(
        update_state_names={"mean": col("h").mean()},
        cudagraph=True,
        unroll=4,
        chunk_size=8,
    ).cuda()
    reset_net_state(chunked, batch_size=B)
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="single chunk"):
            chunked(random_inputs().cuda())
