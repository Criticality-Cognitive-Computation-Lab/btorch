#!/usr/bin/env python3
"""Compare native and Triton sparse backends with plain ``torch.compile``."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import scipy.sparse
import torch

from btorch.models.base import MemoryModule
from btorch.models.functional import reset_net_state
from btorch.models.linear import SparseConn
from btorch.models.rnn import make_rnn


class SparseProjectionCell(MemoryModule):
    """Run two sparse projections from a controlled spike vector."""

    def __init__(
        self,
        n_neurons: int,
        degree: int,
        backend: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        rng = np.random.default_rng(20260929)
        rows = np.repeat(np.arange(n_neurons, dtype=np.int64), degree)
        cols = rng.integers(0, n_neurons, size=n_neurons * degree, dtype=np.int64)

        def make_conn() -> scipy.sparse.coo_array:
            values = rng.normal(0.0, 0.01, size=n_neurons * degree).astype(np.float32)
            return scipy.sparse.coo_array(
                (values, (rows, cols)), shape=(n_neurons, n_neurons)
            )

        kwargs = {
            "enforce_dale": False,
            "sparse_backend": backend,
            "dtype": dtype,
        }
        self.excitatory = SparseConn(make_conn(), **kwargs)
        self.inhibitory = SparseConn(make_conn(), **kwargs)
        initial = torch.zeros(1, dtype=dtype)
        self.register_memory("ie", initial, n_neurons)
        self.register_memory("ii", initial, n_neurons)
        self.init_state()

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        self.ie = self.excitatory(spikes)
        self.ii = self.inhibitory(spikes)
        return spikes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "triton"), required=True)
    parser.add_argument("--n-neurons", type=int, default=32768)
    parser.add_argument("--degree", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--rate", type=float, default=0.003)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def median_cuda_ms(fn, model, repeats: int, device: torch.device):
    times = []
    output = None
    for _ in range(repeats):
        reset_net_state(model, batch_size=1)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), output


def git_revision() -> str | None:
    """Return the checkout revision when the benchmark runs from a Git tree."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float32
    torch._inductor.config.triton.cudagraphs = False
    torch.manual_seed(20260929)

    build_start = time.perf_counter()
    cell = SparseProjectionCell(args.n_neurons, args.degree, args.backend, dtype).to(
        device
    )
    model = make_rnn(
        cell,
        update_state_names=("ie", "ii"),
        unroll=args.unroll,
        cudagraph=False,
    )
    torch.cuda.synchronize(device)
    build_s = time.perf_counter() - build_start

    generator = torch.Generator(device=device).manual_seed(20260929)
    inputs = (
        torch.rand(
            args.steps,
            1,
            args.n_neurons,
            generator=generator,
            device=device,
        )
        < args.rate
    ).to(dtype)

    def eager_call():
        return model(inputs)

    with torch.inference_mode():
        reset_net_state(model, batch_size=1)
        eager_call()
        eager_ms, eager_output = median_cuda_ms(eager_call, model, args.repeats, device)

    torch._dynamo.reset()
    compiled = torch.compile(model, mode="default")

    def compiled_call():
        return compiled(inputs)

    with torch.inference_mode():
        reset_net_state(model, batch_size=1)
        torch.cuda.synchronize(device)
        first_start = time.perf_counter()
        compiled_call()
        torch.cuda.synchronize(device)
        first_compile_s = time.perf_counter() - first_start
        compiled_ms, compiled_output = median_cuda_ms(
            compiled_call, model, args.repeats, device
        )

    eager_spikes, eager_states = eager_output
    compiled_spikes, compiled_states = compiled_output
    max_abs_error = max(
        float((eager_spikes - compiled_spikes).abs().max()),
        *(
            float((eager_states[name] - compiled_states[name]).abs().max())
            for name in eager_states
        ),
    )
    result = {
        "date": "2026-09-29",
        "btorch_commit": git_revision(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "backend": args.backend,
        "n_neurons": args.n_neurons,
        "degree": args.degree,
        "density": args.degree / args.n_neurons,
        "edges_per_projection": int(cell.excitatory.indices.shape[1]),
        "projections": 2,
        "steps": args.steps,
        "requested_rate": args.rate,
        "measured_rate": float(inputs.mean()),
        "unroll": args.unroll,
        "dtype": str(dtype),
        "recorded_histories": ["spike", "ie", "ii"],
        "cudagraph": False,
        "inductor_cudagraphs": False,
        "scan": False,
        "build_s": build_s,
        "eager_ms": eager_ms,
        "compiled_ms": compiled_ms,
        "compile_speedup": eager_ms / compiled_ms,
        "first_compile_s": first_compile_s,
        "max_abs_error": max_abs_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
