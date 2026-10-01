"""Focused tests for the small helpers shared by the RNN loop paths and for the
explicit state-setter functions in ``btorch.models.functional``."""

import pytest
import torch

from btorch.models import functional
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
