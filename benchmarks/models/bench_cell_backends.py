"""Measure cell sequence compilation, forward/reverse and graph replay.

Run on CUDA after installing any optional backends to be compared::

    PYTHONPATH=. python benchmarks/models/bench_cell_backends.py --model alpha

The JSON rows report device time and allocated tensor memory separately from
first-call host compilation time. Connections are deliberately excluded from
this cell benchmark; use the sparse benchmarks for their independent kernels.
"""

import argparse
import json
import time

import torch

from btorch.models.kernels import CellSpec, run_sequence
from btorch.models.kernels.golden import python_module_sequence


def main():
    """Compare the existing Python Modules, Torch operator and installed GPU
    DSLs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("lif", "alpha"), default="alpha")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--neurons", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=("eager", "torch", "triton", "tilelang", "cupy"),
    )
    args = parser.parse_args()
    if min(args.steps, args.batch, args.neurons, args.repeats) < 1:
        parser.error("steps, batch, neurons and repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    torch.manual_seed(241)
    x = torch.rand(args.steps, args.batch, args.neurons, device="cuda") * 0.4
    spec = CellSpec(args.model)
    states = tuple(
        torch.zeros(args.batch, args.neurons, device="cuda")
        for _ in range(1 if args.model == "lif" else 2)
    )
    values = (1.0, 0.0, 1.0, 20.0) if args.model == "lif" else (3.0, 1.0)
    parameters = tuple(
        torch.full((args.neurons,), value, device="cuda") for value in values
    )
    train_x = x.detach().requires_grad_()
    train_parameters = tuple(value.detach().requires_grad_() for value in parameters)

    def eager(value, params):
        return python_module_sequence(value, states, params, spec)[0]

    # Validate against ordinary autograd independently of the backend being
    # timed. Comparing a replay to that backend's own eager call would hide a
    # consistently incorrect kernel or handwritten derivative.
    with torch.no_grad():
        reference = eager(x, parameters)
    reference_gradients = torch.autograd.grad(
        eager(train_x, train_parameters).sum(), (train_x, *train_parameters)
    )
    for backend in args.backends:

        def simulate(value, params):
            if backend != "eager":
                return run_sequence(value, states, params, spec, backend=backend)[0]
            return eager(value, params)

        started = time.perf_counter()
        with torch.no_grad():
            simulation = simulate(x, parameters)
        torch.cuda.synchronize()
        first_forward_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        loss = simulate(train_x, train_parameters).sum()
        gradients = torch.autograd.grad(loss, (train_x, *train_parameters))
        torch.cuda.synchronize()
        first_training_ms = (time.perf_counter() - started) * 1000
        torch.testing.assert_close(simulation, reference, atol=5e-6, rtol=5e-5)
        for gradient, expected in zip(gradients, reference_gradients):
            torch.testing.assert_close(gradient, expected, atol=5e-5, rtol=5e-4)
        torch.cuda.reset_peak_memory_stats()
        initial_bytes = torch.cuda.memory_allocated()
        forward_ms, backward_ms = [], []
        for _ in range(args.repeats):
            events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            events[0].record()
            loss = simulate(train_x, train_parameters).sum()
            events[1].record()
            torch.autograd.grad(loss, (train_x, *train_parameters))
            events[2].record()
            events[2].synchronize()
            forward_ms.append(events[0].elapsed_time(events[1]))
            backward_ms.append(events[1].elapsed_time(events[2]))
        training_peak = torch.cuda.max_memory_allocated() - initial_bytes
        # Forward simulation capture uses no backward histories. All kernels
        # were compiled by the two warmup calls above.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            simulate(x, parameters)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            captured = simulate(x, parameters)
        graph.replay()
        torch.testing.assert_close(captured, reference)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(args.repeats):
            graph.replay()
        end.record()
        end.synchronize()
        simulation_graph_ms = start.elapsed_time(end) / args.repeats
        # Capture the complete training call, including the registered reverse
        # operator. Replay has no Python autograd work or new allocator calls;
        # its timing is reported separately from the eager device timings.
        with torch.cuda.stream(stream):
            # Separate leaves prevent eager loss references from pinning their
            # AccumulateGrad nodes to the default stream during capture.
            capture_x = train_x.detach().clone().requires_grad_()
            capture_parameters = tuple(
                value.detach().clone().requires_grad_() for value in train_parameters
            )
            for _ in range(3):
                warm_loss = simulate(capture_x, capture_parameters).sum()
                torch.autograd.grad(warm_loss, (capture_x, *capture_parameters))
        torch.cuda.current_stream().wait_stream(stream)
        training_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(training_graph, stream=stream):
            captured_loss = simulate(capture_x, capture_parameters).sum()
            captured_gradients = torch.autograd.grad(
                captured_loss, (capture_x, *capture_parameters)
            )
        training_graph.replay()
        for gradient, expected in zip(captured_gradients, reference_gradients):
            torch.testing.assert_close(gradient, expected, atol=5e-5, rtol=5e-4)
        start.record()
        for _ in range(args.repeats):
            training_graph.replay()
        end.record()
        end.synchronize()
        training_graph_ms = start.elapsed_time(end) / args.repeats
        print(
            json.dumps(
                {
                    "backend": backend,
                    "model": args.model,
                    "shape_TBN": [args.steps, args.batch, args.neurons],
                    "first_forward_host_ms": first_forward_ms,
                    "first_training_host_ms": first_training_ms,
                    "training_forward_device_ms": sum(forward_ms) / len(forward_ms),
                    "backward_device_ms": sum(backward_ms) / len(backward_ms),
                    "training_peak_allocated_bytes": training_peak,
                    "simulation_graph_device_ms": simulation_graph_ms,
                    "training_graph_device_ms": training_graph_ms,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
