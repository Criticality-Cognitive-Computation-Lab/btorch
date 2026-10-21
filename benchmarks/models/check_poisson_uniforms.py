"""Diagnose Torch-owned uniforms for native Poisson neuron lowering.

This harness does not implement neuron equations or a native stochastic
kernel. It observes draws made by the existing :class:`PoissonRandomSpike`,
then checks whether explicit Torch draw schedules reproduce those values and
the global CUDA generator continuation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import torch

from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.neurons import ALIF
from btorch.models.surrogate import PoissonRandomSpike


DT = 0.5
SOURCE_PATHS = (
    "benchmarks/models/check_poisson_uniforms.py",
    "btorch/models/surrogate/poisson_random.py",
    "btorch/models/neurons/alif.py",
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=13)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--neurons", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7391)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.neurons) < 1:
        parser.error("steps, batch and neurons must be positive")
    return args


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(root):
    def git(*arguments):
        return subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "sources": {name: _sha256(root / name) for name in SOURCE_PATHS},
    }


def _rng_state():
    return torch.cuda.get_rng_state().clone()


def _set_rng_state(state):
    torch.cuda.set_rng_state(state)


def _draw_sequence(reference, steps):
    """Use the proposed schedule: one ordinary Torch draw per logical step."""
    return torch.stack([torch.rand_like(reference) for _ in range(steps)])


def _spike_sequence(function, values):
    """Call the actual stochastic primitive once per logical timestep."""
    return torch.stack([function(step) for step in values])


def _input(layout, steps, batch, neurons):
    values = torch.linspace(
        -1.2, 0.8, steps * batch * neurons, device="cuda", dtype=torch.float32
    ).reshape(steps, batch, neurons)
    if layout == "contiguous":
        return values
    backing = torch.empty(steps, batch, neurons * 2, device="cuda")
    backing[..., ::2].copy_(values)
    backing[..., 1::2].fill_(-17.0)
    return backing[..., ::2]


def _layout(value):
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "contiguous": value.is_contiguous(),
        "dtype": str(value.dtype),
    }


def _uniform_case(layout, args):
    values = _input(layout, args.steps, args.batch, args.neurons)
    function = PoissonRandomSpike(tau=2.5, rho=0.7, leak=0.2, k=1.1)
    original_rand_like = torch.rand_like
    observed_draws = []
    observed_references = []

    def record(reference, *positional, **keywords):
        result = original_rand_like(reference, *positional, **keywords)
        observed_references.append(_layout(reference))
        observed_draws.append(result.clone())
        return result

    torch.cuda.manual_seed(args.seed)
    initial_rng = _rng_state()
    with torch.no_grad(), patch.object(torch, "rand_like", record):
        spikes = _spike_sequence(function, values)
    actual_rng = _rng_state()
    actual_uniforms = torch.stack(observed_draws)
    if len(observed_draws) != args.steps:
        raise AssertionError(
            f"Expected {args.steps} actual rand_like calls, got {len(observed_draws)}"
        )

    candidates = {
        "contiguous": torch.empty(
            args.batch, args.neurons, device="cuda", dtype=torch.float32
        ),
        "input_step": values[0],
        "observed_stride": torch.empty_strided(
            observed_draws[0].shape,
            observed_draws[0].stride(),
            device="cuda",
            dtype=observed_draws[0].dtype,
        ),
    }
    results = {}
    for name, reference in candidates.items():
        _set_rng_state(initial_rng)
        uniforms = _draw_sequence(reference, args.steps)
        final_rng = _rng_state()
        results[name] = {
            "reference": _layout(reference),
            "uniforms_exact": torch.equal(uniforms, actual_uniforms),
            "final_rng_exact": torch.equal(final_rng, actual_rng),
        }

    _set_rng_state(initial_rng)
    split = max(1, args.steps // 3)
    chunked = torch.cat(
        (
            _draw_sequence(candidates["observed_stride"], split),
            _draw_sequence(candidates["observed_stride"], args.steps - split),
        )
    )
    chunked_rng = _rng_state()
    if not torch.equal(chunked, actual_uniforms) or not torch.equal(
        chunked_rng, actual_rng
    ):
        raise AssertionError(
            "Chunked Torch draws changed values or generator continuation"
        )
    if not results["observed_stride"]["uniforms_exact"]:
        raise AssertionError(
            "Recorded rand_like layout did not reproduce actual uniforms"
        )
    if not spikes.eq(0).any() or not spikes.eq(1).any():
        raise AssertionError("Diagnostic needs both sampled zeros and ones")

    return {
        "input": _layout(values),
        "actual_draw_references": observed_references,
        "actual_uniform": _layout(observed_draws[0]),
        "draw_calls": len(observed_draws),
        "spike_zeros": int(spikes.eq(0).sum()),
        "spike_ones": int(spikes.eq(1).sum()),
        "candidates": results,
        "whole_chunk_exact": True,
    }


def _refractory_case(args):
    """Prove fully masked refractory lanes still consume one draw per step."""
    neuron = ALIF(
        args.neurons,
        v_threshold=1.0,
        v_reset=0.0,
        c_m=1.0,
        g_leak=0.2,
        E_leak=0.0,
        E_k=-1.0,
        tau_adapt=8.0,
        dg_k=0.12,
        tau_ref=100.0,
        surrogate_function=PoissonRandomSpike(tau=2.5, rho=0.7),
        hard_reset=True,
        step_mode="m",
        device="cuda",
        dtype=torch.float32,
    )
    init_net_state(neuron, batch_size=args.batch, device="cuda", dtype=torch.float32)
    neuron.refractory = torch.full_like(neuron.refractory, 100.0)
    drive = 0.45 + 0.15 * torch.sin(
        torch.arange(args.steps, device="cuda")[:, None, None]
        + torch.arange(args.neurons, device="cuda")[None, None] * 0.31
    ).expand(-1, args.batch, -1)
    original_rand_like = torch.rand_like
    draw_shapes = []

    def record(reference, *positional, **keywords):
        draw_shapes.append(list(reference.shape))
        return original_rand_like(reference, *positional, **keywords)

    torch.cuda.manual_seed(args.seed + 1)
    initial_rng = _rng_state()
    with (
        torch.no_grad(),
        environ.context(dt=DT),
        patch.object(torch, "rand_like", record),
    ):
        spikes = neuron(drive)
    actual_rng = _rng_state()

    _set_rng_state(initial_rng)
    reference = torch.empty(args.batch, args.neurons, device="cuda")
    _draw_sequence(reference, args.steps)
    expected_rng = _rng_state()
    if len(draw_shapes) != args.steps or any(
        shape != [args.batch, args.neurons] for shape in draw_shapes
    ):
        raise AssertionError(f"Unexpected refractory draw schedule: {draw_shapes}")
    if not torch.equal(actual_rng, expected_rng):
        raise AssertionError("Refractory masking changed global RNG advancement")
    if bool(spikes.any()):
        raise AssertionError(
            "The refractory diagnostic expected every event to be masked"
        )
    return {
        "draw_calls": len(draw_shapes),
        "draw_shapes": draw_shapes,
        "all_emitted_spikes_masked": True,
        "final_rng_exact": True,
    }


def _graph_case(args):
    """Capture the unchanged primitive loop and compare replay to eager."""
    function = PoissonRandomSpike(tau=2.5, rho=0.7, leak=0.2, k=1.1)
    static_values = _input("contiguous", args.steps, args.batch, args.neurons).clone()
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream), torch.no_grad():
        for _ in range(2):
            _spike_sequence(function, static_values)
    torch.cuda.current_stream().wait_stream(warm_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        captured = _spike_sequence(function, static_values)

    base = _input("contiguous", args.steps, args.batch, args.neurons)
    trials = (base, base.flip(0) * 0.73 + 0.11, base.roll(2, 0) * 1.17 - 0.08)
    torch.cuda.manual_seed(args.seed + 2)
    replay_counts = []
    previous = None
    for values in trials:
        static_values.copy_(values)
        initial_rng = _rng_state()
        graph.replay()
        replay_rng = _rng_state()
        replay = captured.clone()

        _set_rng_state(initial_rng)
        with torch.no_grad():
            eager = _spike_sequence(function, values)
        eager_rng = _rng_state()
        if not torch.equal(replay, eager) or not torch.equal(replay_rng, eager_rng):
            raise AssertionError("CUDA Graph replay changed events or RNG continuation")
        replay_counts.append(int(replay.sum()))
        if previous is not None and torch.equal(replay, previous):
            raise AssertionError(
                "Changed graph input did not change the sampled raster"
            )
        previous = replay
    return {
        "trials": len(trials),
        "spike_counts": replay_counts,
        "changed_input_outputs": True,
        "events_and_final_rng_exact": True,
    }


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = Path(__file__).resolve().parents[2]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        report = {
            "qualification": "Torch-owned Poisson uniform schedule diagnostic",
            "current": _identity(root),
            "environment": {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
            },
            "contract": {
                "dtype": "float32",
                "shape_TBN": [args.steps, args.batch, args.neurons],
                "draw_schedule": "T calls to torch.rand_like over one [B,N] step",
                "scientific_oracle": "actual PoissonRandomSpike and ALIF Modules",
                "native_kernel": False,
            },
            "uniform_layouts": {
                layout: _uniform_case(layout, args)
                for layout in ("contiguous", "strided")
            },
            "refractory": _refractory_case(args),
            "cuda_graph": _graph_case(args),
        }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
