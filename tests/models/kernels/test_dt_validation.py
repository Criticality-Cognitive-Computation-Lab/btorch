"""Verify timestep validation without standalone native CUDA launches."""

import os
import subprocess
import sys

import pytest
import torch

from btorch.models.kernels.backends.engine import generate_source
from btorch.models.kernels.contracts import NeuronSpec
from btorch.models.kernels.operators import run_multi_step
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


def _assert_device_asserted(result):
    assert result.returncode != 0, result.stdout
    diagnostic = (result.stdout + result.stderr).lower()
    assert (
        "device-side assert" in diagnostic or "cudaerrorassert" in diagnostic
    ), diagnostic


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_primitive_source_contains_backend_device_timestep_assertions(backend):
    source, _ = generate_source(backend, NeuronSpec("if"), 4, 3, ())
    if backend == "triton":
        assert "tl.device_assert" in source
        assert 'dt < float("inf")' in source
    elif backend == "cupy":
        assert "#include <assert.h>" not in source
        assert "index == 0 && !(isfinite(dt) && dt > 0.0f)" in source
        assert '__assertfail("dt must be finite and positive"' in source
    else:
        assert "index != 0 or (T.isfinite(dt[0]) and dt[0] > 0.0)" in source
        compile(source, "<tilelang-dt-validation>", "exec")


@pytest.mark.parametrize("dt", (0.0, -0.5, float("inf"), float("nan")))
def test_torch_fallback_rejects_invalid_timestep(dt):
    x = torch.ones(2, 3)
    state = (torch.zeros(3),)
    parameters = tuple(torch.ones(()) for _ in range(3))
    with pytest.raises(RuntimeError, match="dt must be finite and positive"):
        run_multi_step(
            x,
            state,
            parameters,
            NeuronSpec("if", dt=torch.tensor(dt)),
            backend="torch",
        )


_DEVICE_ASSERT_SCRIPT = r"""
import sys
import torch

from btorch.models.kernels.contracts import NeuronSpec
from btorch.models.kernels.operators import run_multi_step

backend, value, model, mode = sys.argv[1:]
valid_dt = 1.0 if model == "alpha_billeh" else 0.5
dt = torch.tensor(valid_dt, device="cuda")
x = torch.ones(2, 4, device="cuda")
if mode == "empty":
    x = x[:0]
elif mode == "zero_neuron":
    x = x[:, :0]
if model == "alpha_billeh":
    states = (torch.zeros(4, device="cuda"), torch.zeros(4, device="cuda"))
    parameters = (
        torch.full((), 2.0, device="cuda"),
        torch.full((), 0.7, device="cuda"),
    )
else:
    states = (torch.zeros(4, device="cuda"),)
    parameters = tuple(torch.ones((), device="cuda") for _ in range(3))
if mode == "zero_neuron":
    states = tuple(state[:0] for state in states)

spec = NeuronSpec(model, dt=dt)
if mode == "replaced":
    from dataclasses import replace
    from btorch.models.kernels import registry

    entry = registry.resolve_cell_backend(backend, spec, x.device)
    def replacement(x_seq, states, parameters, spec, history_names, **kwargs):
        return torch.zeros_like(x_seq), tuple(state.clone() for state in states), ()
    registry._entries[backend] = replace(entry, forward=replacement)
if mode in ("normal", "graph"):
    # Compile and execute with valid invocation data first. Mutating this exact
    # device tensor then proves validation reads current data on every call.
    run_multi_step(x, states, parameters, spec, backend=backend)
    torch.cuda.synchronize()
dt.fill_(float(value))
if mode == "graph":
    graph = torch.cuda.CUDAGraph()
    dt.fill_(valid_dt)
    with torch.cuda.graph(graph):
        run_multi_step(x, states, parameters, spec, backend=backend)
    graph.replay()
    torch.cuda.synchronize()
    dt.fill_(float(value))
    graph.replay()
else:
    run_multi_step(x, states, parameters, spec, backend=backend)
torch.cuda.synchronize()
"""


@pytest.mark.parametrize(
    "backend,value,model",
    (
        ("triton", "0", "if"),
        ("triton", "nan", "if"),
        ("cupy", "-0.5", "if"),
        ("tilelang", "inf", "if"),
        ("triton", "0.5", "alpha_billeh"),
        ("cupy", "0.5", "alpha_billeh"),
        ("tilelang", "0.5", "alpha_billeh"),
    ),
)
def test_native_device_assert_rejects_mutated_timestep_in_isolated_context(
    backend, value, model
):
    require_cuda_backend(backend)
    environment = {**os.environ, "PYTHONPATH": "."}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _DEVICE_ASSERT_SCRIPT,
            backend,
            value,
            model,
            "normal",
        ],
        cwd=os.getcwd(),
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    _assert_device_asserted(result)


@pytest.mark.parametrize("backend,mode", (("triton", "empty"), ("cupy", "zero_neuron")))
def test_no_work_native_calls_keep_host_validation_in_isolated_context(backend, mode):
    require_cuda_backend(backend)
    # Host asynchronous assertions can also poison a CUDA context. Isolation
    # keeps the remainder of the suite usable while proving no-work calls do
    # not rely on a scan kernel that cannot launch.
    result = subprocess.run(
        [sys.executable, "-c", _DEVICE_ASSERT_SCRIPT, backend, "0", "if", mode],
        cwd=os.getcwd(),
        env={**os.environ, "PYTHONPATH": "."},
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    _assert_device_asserted(result)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_captured_replay_validates_mutated_timestep_in_isolated_context(backend):
    require_cuda_backend(backend)
    # Capture stores the dt pointer, not its original value. The second replay
    # must therefore observe the in-place mutation and trigger the same device
    # assertion as eager execution.
    result = subprocess.run(
        [sys.executable, "-c", _DEVICE_ASSERT_SCRIPT, backend, "0", "if", "graph"],
        cwd=os.getcwd(),
        env={**os.environ, "PYTHONPATH": "."},
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    _assert_device_asserted(result)


def test_replaced_native_callback_uses_host_validation_in_isolated_context():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # The replacement deliberately launches no primitive cell kernel. A
    # backend-name check would let invalid dt pass; callback identity must keep
    # the host fallback validation active.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _DEVICE_ASSERT_SCRIPT,
            "triton",
            "0",
            "if",
            "replaced",
        ],
        cwd=os.getcwd(),
        env={**os.environ, "PYTHONPATH": "."},
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    _assert_device_asserted(result)
