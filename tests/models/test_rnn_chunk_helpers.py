"""Focused tests for the small helpers shared by the RNN loop paths and for the
explicit state-setter functions in ``btorch.models.functional``."""

import numpy as np
import pytest
import torch
from torch import nn

from btorch.models import functional
from btorch.models.base import ResetValue
from btorch.models.cudagraph import CudaGraphRunner
from btorch.models.neurons import LIF
from btorch.models.rnn import _append_chunk, _chunk_to_cpu, make_rnn


def test_append_chunk_handles_lists_and_stacked_tensors():
    # The eager path yields per-step *lists* of tensors; the cudagraph path yields
    # one stacked tensor per chunk. The same accumulator must serve both.
    z_list, states = [], {}
    _append_chunk(
        z_list, states, [torch.zeros(2), torch.ones(2)], {"v": [torch.ones(2)]}
    )
    assert len(z_list) == 2 and len(states["v"]) == 1

    z_list, states = [], {}
    _append_chunk(z_list, states, torch.zeros(3, 2), {"v": torch.ones(3, 2)})
    _append_chunk(z_list, states, torch.zeros(3, 2), {"v": torch.ones(3, 2)})
    assert len(z_list) == 2 and len(states["v"]) == 2  # one entry per chunk


def test_chunk_to_cpu_preserves_structure():
    z, states = _chunk_to_cpu([torch.zeros(2)], {"v": [torch.ones(2)]})
    assert isinstance(z, list) and isinstance(states["v"], list)
    z, states = _chunk_to_cpu(torch.zeros(2), {"v": torch.ones(2)})
    assert z.device.type == "cpu" and states["v"].device.type == "cpu"


def test_iter_large_chunks_splits_only_loop_args():
    # Loop arg (position 0) is split along time; the scalar passes through.
    rnn = make_rnn(LIF, chunk_size=None)(n_neuron=2)
    x = torch.arange(10.0).reshape(10, 1)
    chunks = list(rnn._iter_large_chunks((x, 7), (0,), 4))
    assert [c[0].shape[0] for c in chunks] == [4, 4, 2]
    assert all(c[1] == 7 for c in chunks)
    assert torch.equal(torch.cat([c[0] for c in chunks]), x)


def test_detect_loop_args_shape_heuristic_documented():
    """Pin the loop-arg heuristic (documentation of current behaviour).

    ``_detect_loop_args`` guesses which positional args are time sequences:
    every tensor whose leading dim equals the FIRST arg's leading dim (``T``)
    is a loop arg; non-tensors and scalar tensors never are.  Consequence: a
    static tensor whose first dim happens to equal ``T`` is silently treated
    as a time sequence, so pass such tensors as keyword args or reshape them.
    """
    rnn = make_rnn(LIF, chunk_size=None)(n_neuron=2)
    x = torch.zeros(5, 3, 2)

    # Single arg: always the loop arg.
    assert rnn._detect_loop_args(x) == (5, (0,))

    # Same leading dim -> loop arg; different leading dim / scalar / non-tensor
    # -> passed through unchanged.
    T, loop = rnn._detect_loop_args(x, torch.zeros(5, 2), torch.zeros(4, 2), 7)
    assert (T, loop) == (5, (0, 1))
    T, loop = rnn._detect_loop_args(x, torch.tensor(1.0), torch.zeros(2, 2))
    assert (T, loop) == (5, (0,))

    # The pitfall: a static (n_neuron=5)-shaped tensor is mistaken for a
    # sequence when n_neuron == T.
    _, loop = rnn._detect_loop_args(x, torch.zeros(5))
    assert loop == (0, 1)


def test_cudagraph_incompatibilities_reports_enabled_options():
    rnn = make_rnn(LIF, grad_checkpoint=True)(n_neuron=2)
    assert list(rnn._cudagraph_incompatibilities()) == ["grad_checkpoint"]
    assert make_rnn(LIF)(n_neuron=2)._cudagraph_incompatibilities() == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_runner_refuses_caller_supplied_incompatibility():
    # The runner is module-agnostic: it refuses whatever reason it is handed.
    m = LIF(n_neuron=2).cuda()
    x = torch.zeros(1, 2, device="cuda")
    with torch.no_grad(), pytest.raises(RuntimeError, match="foo=True"):
        CudaGraphRunner()(m, lambda a: a, (x,), incompatible={"foo": "because"})


def test_set_hidden_states_and_reset_values_roundtrip():
    # Exercises both explicit setters (rebinding and inplace) and the reset
    # value setter on a dotted name.
    m = LIF(n_neuron=3)
    functional.init_net_state(m, batch_size=2)
    new_v = torch.full((2, 3), 0.5)
    functional.set_hidden_states(m, {"v": new_v})
    assert torch.equal(m.v, new_v)
    addr = m.v.data_ptr()
    functional.set_hidden_states(m, {"v": torch.zeros(2, 3)}, inplace=True)
    assert m.v.data_ptr() == addr and m.v.abs().sum() == 0
    rv = functional.named_memory_reset_values(m)
    functional.set_memory_reset_values(m, rv)


def test_detect_loop_args_rejects_non_tensor_first_arg():
    """No time dimension can be inferred from a non-tensor / 0-dim first arg.

    This used to be a bare ``assert`` (stripped under ``python -O``); it is now
    a ``ValueError`` that tells the caller to pass ``loop_args`` explicitly.
    """
    rnn = make_rnn(LIF, chunk_size=None)(n_neuron=2)
    with pytest.raises(ValueError, match="loop_args"):
        rnn._detect_loop_args(7, torch.zeros(5, 2))
    with pytest.raises(ValueError, match="loop_args"):
        rnn._detect_loop_args(torch.tensor(1.0), torch.zeros(5, 2))


def test_abstract_single_step_forward_is_a_bound_method_stub():
    """``RecurrentNNAbstract.single_step_forward`` takes ``self`` like the
    base.

    It used to be declared without ``self``. A subclass that forgets to
    override it must now fail with a clear ``NotImplementedError`` rather than a
    confusing argument-binding error or a silent ``None``.
    """
    from btorch.models.rnn import RecurrentNNAbstract

    stub = RecurrentNNAbstract()
    with pytest.raises(NotImplementedError):
        stub.single_step_forward(torch.zeros(2))


def test_chunk_step_helpers_match_a_manual_loop():
    """``_run_chunk_steps`` (unroll blocks of ``_run_unroll_block``) equals a
    plain per-step loop, whatever the unroll size, including a remainder."""
    torch.manual_seed(0)
    cell = make_rnn(LIF)(n_neuron=3, step_mode="s")
    functional.init_net_state(cell, batch_size=2)
    x = torch.rand(7, 2, 3) * 3
    start = functional.named_hidden_states(cell, clone=True)

    from btorch.models import environ

    with environ.context(dt=1.0):
        z_ref, _ = cell._run_unroll_block(x, loop_args=(0,))
        functional.set_hidden_states(cell, start)
        z_chunked, _ = cell._run_chunk_steps(x, loop_args=(0,), unroll_size=3)
    torch.testing.assert_close(torch.stack(z_chunked), torch.stack(z_ref))


def test_functional_set_memory_reset_values_dotted_and_nested():
    """The network-wide setter takes ``reset_values`` addressed by dotted name,
    or per-module nested dicts (``{"neuron": {"v": ...}}``)."""
    net = nn.ModuleDict({"neuron": LIF(n_neuron=3)})
    assert set(functional.named_memory_reset_values(net)) == {"neuron.v"}

    functional.set_memory_reset_values(net, {"neuron.v": 0.7})
    assert float(net["neuron"].memory_reset_values["v"].value) == 0.7

    # Whole-module entry: used to fail because the walker looked the name up in
    # the (nonexistent) reset-value registry of the container.
    functional.set_memory_reset_values(net, {"neuron": {"v": 0.2}})
    assert float(net["neuron"].memory_reset_values["v"].value) == 0.2

    # The keyword is ``reset_values`` (it used to be called ``hidden_states``).
    functional.set_memory_reset_values(net, reset_values={"neuron.v": 0.1})
    functional.init_net_state(net)
    assert torch.allclose(net["neuron"].v.double(), torch.full((3,), 0.1).double())

    with pytest.raises(KeyError, match="nope"):
        functional.set_memory_reset_values(net, {"neuron.nope": 1.0})


def test_set_memory_reset_values_strict_default_is_consistent():
    """Function and method both default to ``strict=True``: a ``ResetValue``
    with different ``sizes`` than the registered memory is rejected by both,
    and accepted by both when ``strict=False``."""
    bad = ResetValue(value=np.zeros(()), sizes=(5,), has_batch=False)

    m = LIF(n_neuron=3)
    with pytest.raises(ValueError, match="sizes mismatch"):
        m.set_memory_reset_values({"v": bad})
    with pytest.raises(ValueError, match="sizes mismatch"):
        functional.set_memory_reset_values(m, {"v": bad})

    m.set_memory_reset_values({"v": bad}, strict=False)
    assert m.memory_reset_values["v"].sizes == (5,)
    functional.set_memory_reset_values(m, {"v": bad}, strict=False)
