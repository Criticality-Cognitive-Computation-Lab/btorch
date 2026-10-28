"""Focused checks for recurrent training-history ownership.

These tests cover the memory contract behind ``cpu_offload`` without expanding
the recurrent option matrix. Public trajectories remain differentiable, while
private autograd intermediates can move to CPU and model-owned storage remains
resident.
"""

import pytest
import torch

from btorch.models.functional import reset_net_state, set_hidden_states
from btorch.models.rnn import make_rnn
from tests.models.rnn.rnn_utils import SimpleRNNCell


def _build(*, cpu_offload: bool):
    return make_rnn(
        SimpleRNNCell,
        unroll=2,
        chunk_size=3,
        cpu_offload=cpu_offload,
    )(input_size=3, hidden_size=4)


def _run_training(model, x, h0):
    set_hidden_states(model, {"rnn_cell.h": h0})
    output, states = model(x)
    # Include the returned trajectory in the loss. This verifies that moving
    # public chunk tensors to CPU does not detach their backward path.
    loss = output.square().sum() + states["h"].square().sum()
    loss.backward()
    parameter_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    return (
        output.detach().cpu(),
        states["h"].detach().cpu(),
        x.grad.detach().clone(),
        h0.grad.detach().clone(),
        parameter_grads,
    )


def test_cpu_offload_preserves_chunked_full_bptt_without_checkpointing():
    """Offloading private saves keeps output, record, and input gradients
    equal."""
    torch.manual_seed(1827)
    reference = _build(cpu_offload=False)
    offloaded = _build(cpu_offload=True)
    offloaded.load_state_dict(reference.state_dict())

    x_reference = torch.randn(7, 2, 3, requires_grad=True)
    x_offloaded = x_reference.detach().clone().requires_grad_()
    h_reference = torch.randn(2, 4, requires_grad=True)
    h_offloaded = h_reference.detach().clone().requires_grad_()
    expected = _run_training(reference, x_reference, h_reference)
    actual = _run_training(offloaded, x_offloaded, h_offloaded)

    for value, target in zip(actual[:4], expected[:4]):
        torch.testing.assert_close(value, target, atol=1e-6, rtol=1e-6)
    assert actual[4].keys() == expected[4].keys()
    for name in actual[4]:
        torch.testing.assert_close(
            actual[4][name], expected[4][name], atol=1e-6, rtol=1e-6
        )


def test_cpu_offload_preserves_saved_tensor_inplace_version_check():
    """Resident saved views still reject parameter mutation before backward."""
    model = _build(cpu_offload=True)
    x = torch.randn(7, 2, 3, requires_grad=True)
    h0 = torch.randn(2, 4, requires_grad=True)
    set_hidden_states(model, {"rnn_cell.h": h0})
    output, states = model(x)
    with torch.no_grad():
        model.rnn_cell.W_x.add_(1)

    with pytest.raises(RuntimeError, match="modified in-place"):
        (output.square().sum() + states["h"].square().sum()).backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_offload_packs_private_saves_and_keeps_parameters_resident(monkeypatch):
    """Private CUDA intermediates are packed while weights stay on the GPU.

    ``save_on_cpu`` cannot be applied blindly: a sparse recurrent operator also
    saves its topology and parameters, which must not be copied for every
    chunk. The hook records the actual packing decisions made by the RNN path.
    """
    captured = []
    original_hooks = torch.autograd.graph.saved_tensors_hooks

    def recording_hooks(pack, unpack):
        def record_pack(value):
            packed = pack(value)
            captured.append((value.device, value.data_ptr(), value.numel(), packed))
            return packed

        return original_hooks(record_pack, unpack)

    monkeypatch.setattr(torch.autograd.graph, "saved_tensors_hooks", recording_hooks)
    model = _build(cpu_offload=True).to("cuda")
    x = torch.randn(7, 2, 3, device="cuda", requires_grad=True)
    reset_net_state(model, batch_size=2)
    output, states = model(x)
    (output.square().sum() + states["h"].square().sum()).backward()

    packed_kinds = {
        packed[3][0]
        for packed in captured
        if isinstance(packed[3], tuple) and len(packed[3]) == 4
    }
    assert "resident" in packed_kinds
    assert "cpu" in packed_kinds
    weight_ptr = model.rnn_cell.W_x.data_ptr()
    assert any(
        source_ptr == weight_ptr and packed[0] == "resident"
        for _, source_ptr, _, packed in captured
    )
    assert not any(
        source_ptr == weight_ptr and packed[0] == "cpu"
        for _, source_ptr, _, packed in captured
    )
    assert model.rnn_cell.W_x.grad is not None
