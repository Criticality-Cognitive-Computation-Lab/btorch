"""Measure the complete sparse receptor/delay RSNN forward and reverse.

Run after installing the optional cell backends::

    PYTHONPATH=. python benchmarks/models/bench_rsnn_backends.py --steps 16 \
        --batch 2 --neurons 64 --repeats 3

The historical string-Torch module is the independent baseline. Every timed
implementation first passes output, final-state, record-loss and participating
parameter-gradient parity. GPU events include cell and sparse launches. Host
first-call timings include lazy preparation and may use warm disk caches.
"""

import argparse
import copy
import json
import time

import torch

from btorch.models import KernelOptions, environ, functional
from btorch.models.connection import SparseConnection, Synapse
from btorch.models.neurons import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import AlphaPSC, HeterSynapsePSC


def _prototype(n_neurons, batch):
    """Build signed receptor edges with zero, one and two-step delays."""
    pre = torch.arange(n_neurons, device="cuda").repeat_interleave(3)
    post = (pre + torch.arange(1, 4, device="cuda").repeat(n_neurons)) % n_neurons
    receptor = torch.arange(pre.numel(), device="cuda") % 2
    connection = SparseConnection.from_edges(
        pre,
        post,
        n_neurons,
        n_neurons,
        Synapse(
            delay=torch.arange(pre.numel(), device="cuda") % 3,
            receptor=receptor,
            n_delay=3,
            n_receptor=2,
        ),
        values=torch.where(receptor == 0, 0.15, -0.10).float(),
    )
    synapse = HeterSynapsePSC(
        n_neurons,
        2,
        None,
        connection,
        base_psc=AlphaPSC,
        tau_syn=3.0,
        use_circular_buffer=False,
    )
    model = RecurrentNN(
        LIF(
            n_neurons,
            tau=4.0,
            v_threshold=0.7,
            trainable_param={"tau", "v_threshold"},
        ),
        synapse,
        backend="torch",
        update_state_names=("neuron.v", "synapse.base_psc.psc"),
    ).to(device="cuda", dtype=torch.float32)
    functional.init_net_state(
        model, batch_size=batch, device="cuda", dtype=torch.float32
    )
    # PSC tensors are buffers in the existing public module; they still may
    # participate in scientific parameter fitting and need explicit VJPs.
    synapse.base_psc.tau_syn.requires_grad_()
    synapse.base_psc.g_max.requires_grad_()
    return model


def _runner(model, backend):
    """Restore fixed initial tensors without altering continuation
    semantics."""
    model.backend = "torch" if backend == "eager" else KernelOptions(neuron=backend)
    initial_states = tuple(
        (module, name, getattr(module, name))
        for module in model.modules()
        if hasattr(module, "_memories_rv")
        for name in module._memories_rv
    )
    parameters = (
        *tuple(model.parameters()),
        model.synapse.base_psc.tau_syn,
        model.synapse.base_psc.g_max,
    )

    def run(x):
        # Independent trials start from the identical reset state. Rebinding
        # immutable zero states avoids allocating a fresh reset inside timing;
        # no state is detached within an actual sequence or chunk boundary.
        for module, name, value in initial_states:
            setattr(module, name, value)
        spikes, records = model(x)
        final = tuple(getattr(module, name) for module, name, _ in initial_states)
        return spikes, records, final

    def loss(x):
        spikes, records, _ = run(x)
        return spikes.mean() + sum(value.square().mean() for value in records.values())

    return run, loss, parameters


def _check(actual, expected):
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    for name, value in actual[1].items():
        torch.testing.assert_close(value, expected[1][name], atol=1e-5, rtol=1e-4)
    for value, reference in zip(actual[2], expected[2]):
        torch.testing.assert_close(value, reference, atol=1e-5, rtol=1e-4)


def _check_gradients(actual, expected):
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, atol=1e-4, rtol=1e-3)


def main():
    """Compare module and explicit sequence operators with independent
    parity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--neurons", type=int, default=128)
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
    torch.manual_seed(734)
    prototype = _prototype(args.neurons, args.batch)
    x = torch.rand(args.steps, args.batch, args.neurons, device="cuda") * 0.8
    reference_model = copy.deepcopy(prototype)
    reference_run, reference_loss, reference_parameters = _runner(
        reference_model, "eager"
    )
    reference_x = x.detach().clone().requires_grad_()
    with environ.context(dt=1.0):
        with torch.no_grad():
            expected = reference_run(x)
        reference_gradients = torch.autograd.grad(
            reference_loss(reference_x), (reference_x, *reference_parameters)
        )
        for backend in args.backends:
            model = copy.deepcopy(prototype)
            run, loss, parameters = _runner(model, backend)
            train_x = x.detach().clone().requires_grad_()
            started = time.perf_counter()
            with torch.no_grad():
                actual = run(x)
            torch.cuda.synchronize()
            first_forward_ms = (time.perf_counter() - started) * 1000
            _check(actual, expected)
            started = time.perf_counter()
            gradients = torch.autograd.grad(loss(train_x), (train_x, *parameters))
            torch.cuda.synchronize()
            first_training_ms = (time.perf_counter() - started) * 1000
            _check_gradients(gradients, reference_gradients)
            torch.cuda.reset_peak_memory_stats()
            initial_bytes = torch.cuda.memory_allocated()
            forward_ms, backward_ms = [], []
            for _ in range(args.repeats):
                events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                events[0].record()
                value = loss(train_x)
                events[1].record()
                torch.autograd.grad(value, (train_x, *parameters))
                events[2].record()
                events[2].synchronize()
                forward_ms.append(events[0].elapsed_time(events[1]))
                backward_ms.append(events[1].elapsed_time(events[2]))
            peak_bytes = torch.cuda.max_memory_allocated() - initial_bytes
            # Fresh leaves and model ensure all AccumulateGrad nodes are born
            # on the explicit capture stream, independent of eager graph refs.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                capture_model = copy.deepcopy(prototype)
                capture_run, capture_loss, capture_parameters = _runner(
                    capture_model, backend
                )
                capture_x = x.detach().clone().requires_grad_()
                for _ in range(3):
                    with torch.no_grad():
                        capture_run(x)
                    torch.autograd.grad(
                        capture_loss(capture_x), (capture_x, *capture_parameters)
                    )
            torch.cuda.current_stream().wait_stream(stream)
            simulation_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(simulation_graph, stream=stream), torch.no_grad():
                captured = capture_run(x)
            simulation_graph.replay()
            _check(captured, expected)
            training_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(training_graph, stream=stream):
                captured_gradients = torch.autograd.grad(
                    capture_loss(capture_x), (capture_x, *capture_parameters)
                )
            training_graph.replay()
            _check_gradients(captured_gradients, reference_gradients)
            graph_ms = []
            for graph in (simulation_graph, training_graph):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(args.repeats):
                    graph.replay()
                end.record()
                end.synchronize()
                graph_ms.append(start.elapsed_time(end) / args.repeats)
            print(
                json.dumps(
                    {
                        "backend": backend,
                        "shape_TBN": [args.steps, args.batch, args.neurons],
                        "edges": args.neurons * 3,
                        "receptors": 2,
                        "delay_bins": 3,
                        "first_forward_host_ms": first_forward_ms,
                        "first_training_host_ms": first_training_ms,
                        "training_forward_device_ms": sum(forward_ms) / len(forward_ms),
                        "backward_device_ms": sum(backward_ms) / len(backward_ms),
                        "training_peak_allocated_bytes": peak_bytes,
                        "simulation_graph_device_ms": graph_ms[0],
                        "training_graph_device_ms": graph_ms[1],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
