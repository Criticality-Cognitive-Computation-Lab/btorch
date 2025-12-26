import platform

import pytest
import torch

from btorch.models import environ
from btorch.models.functional import init_net_state, reset_net_state
from btorch.models.neurons.lif import LIF
from tests.utils.compile import compile_or_skip


def _run_lif(
    neuron: LIF, x_seq: torch.Tensor, dt: float, *, init: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    with environ.context(dt=dt):
        if init:
            init_net_state(neuron, dtype=x_seq.dtype, device=x_seq.device)
        else:
            reset_net_state(neuron)
        out = neuron.multi_step_forward(x_seq)
        return out, neuron.v.detach().clone()


@pytest.mark.skipif(
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
def test_compiled_lif_matches_eager_after_dt_change():
    dt_first = 1.0
    dt_second = 0.5

    n_neuron = 3
    steps = 6
    x_seq = torch.full((steps, n_neuron), 0.2)

    eager = LIF(
        n_neuron=n_neuron,
        v_threshold=10.0,
        v_reset=0.0,
        tau=5.0,
        step_mode="s",
    )
    compiled = LIF(
        n_neuron=n_neuron,
        v_threshold=10.0,
        v_reset=0.0,
        tau=5.0,
        step_mode="s",
    )
    compiled = compile_or_skip(compiled)

    out_eager_first, v_eager_first = _run_lif(eager, x_seq, dt_first, init=True)
    out_compiled_first, v_compiled_first = _run_lif(
        compiled, x_seq, dt_first, init=True
    )

    torch.testing.assert_close(out_eager_first, out_compiled_first, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(v_eager_first, v_compiled_first, atol=1e-6, rtol=0.0)

    out_eager_second, v_eager_second = _run_lif(eager, x_seq, dt_second, init=False)

    # Recompile after changing dt so the compiled graph uses the new timestep.
    compiled_second = LIF(
        n_neuron=n_neuron,
        v_threshold=10.0,
        v_reset=0.0,
        tau=5.0,
        step_mode="s",
    )
    compiled_second = compile_or_skip(compiled_second)
    out_compiled_second, v_compiled_second = _run_lif(
        compiled_second, x_seq, dt_second, init=True
    )

    torch.testing.assert_close(
        out_eager_second, out_compiled_second, atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(v_eager_second, v_compiled_second, atol=1e-6, rtol=0.0)

    assert not torch.allclose(v_eager_first, v_eager_second)
