"""Measure cell sequence compilation, forward/reverse and graph replay.

Run on CUDA after installing any optional backends to be compared::

    PYTHONPATH=. python benchmarks/models/bench_cell_backends.py --model alpha

The JSON rows report device time and allocated tensor memory separately from
first-call host compilation time. Connections are deliberately excluded from
this cell benchmark; use the sparse benchmarks for their independent kernels.
The dual-exponential case therefore receives already-projected drive and does
not measure the legacy projected-drive delay, pending state, or external CUDA.
"""

import argparse
import json
import statistics
import time
from dataclasses import replace

import torch

from btorch.models.kernels import CellSpec, HistoryPlan, get_template, run_multi_step
from btorch.models.kernels.golden import python_module_sequence
from btorch.models.kernels.options import resolve_model_options


def main():
    """Compare the existing Python Modules, Torch operator and installed GPU
    DSLs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("lif", "alif", "elif", "alpha", "dual_exponential", "izhikevich"),
        default="alpha",
    )
    parser.add_argument(
        "--parameter-layout", choices=("scalar", "neuron"), default="neuron"
    )
    parser.add_argument(
        "--gradient-scope",
        choices=("input", "parameters", "all"),
        default="all",
        help=(
            "input matches 002HIP's common input-only backward; parameters "
            "freezes input/state and differentiates all parameters; all "
            "differentiates input, eligible initial states, and parameters"
        ),
    )
    parser.add_argument(
        "--precise-exp",
        action="store_true",
        help="Request accurate exponentials without changing other math policies",
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--neurons", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=("eager", "torch", "triton", "tilelang", "cupy"),
    )
    args = parser.parse_args()
    if min(args.steps, args.batch, args.neurons, args.warmup, args.repeats) < 1:
        parser.error("steps, batch, neurons, warmup and repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    torch.manual_seed(241)
    x = torch.rand(args.steps, args.batch, args.neurons, device="cuda") * 0.4
    spec = CellSpec(args.model, hard_reset=args.model == "izhikevich")
    template = get_template(spec)
    state_count = {
        "lif": 1,
        "alif": 2,
        "elif": 2,
        "alpha": 2,
        "dual_exponential": 3,
        "izhikevich": 2,
    }
    states = tuple(
        torch.zeros(args.batch, args.neurons, device="cuda")
        for _ in range(state_count[args.model])
    )
    values = (1.0, 0.0, 1.0, 20.0) if args.model == "lif" else (3.0, 1.0)
    state_gradient_mask = (True,) * len(states)
    if args.model == "dual_exponential":
        # This is the validated finite mapping for q_rise=0.5, q_decay=0.8,
        # D=3 and g_max=0.7. The cell receives projected drive directly; the
        # complete Wz+b delay and legacy state transforms are out of scope.
        values = (1.4426950408889634, 4.481420117724551, 7.0)
        states = (
            states[0],
            torch.full_like(states[1], 0.05),
            torch.full_like(states[2], -0.03),
        )
        # DualExponentialPSC overwrites psc from its two physical traces on
        # every step. Keep psc_init in the state contract, but do not present
        # its intentionally absent derivative as an autograd root.
        state_gradient_mask = (False, True, True)
    if args.model in ("alif", "elif"):
        # Use a finite hard-reset operating envelope from the actual-Module
        # validation. This benchmark measures btorch cells, not an asserted
        # equation match to a historical 002HIP adaptive-neuron variant.
        spec = CellSpec(args.model, dt=0.25, hard_reset=True)
        x = x * 4.0
        states = (
            torch.full_like(states[0], 0.05),
            torch.full_like(states[1], 0.04),
        )
        values = (0.62, -0.12, 1.15, 0.72, -0.08, -0.75, 3.8, 0.09)
        if args.model == "elif":
            values += (0.16, 0.31)
    if args.model == "izhikevich":
        # Match the physical parameters of 002HIP's neuron benchmark. Its
        # quadratic threshold v_t maps to v_threshold, not the spike cutoff.
        x = x * 100 + 200
        states = (torch.full_like(states[0], -65.0), torch.zeros_like(states[0]))
        values = (-40.0, -65.0, 100.0, 2.0, -60.0, 30.0, 0.02, 0.2, 8.0)
    parameter_shape = () if args.parameter_layout == "scalar" else (args.neurons,)
    parameters = tuple(
        torch.full(parameter_shape, value, device="cuda") for value in values
    )
    trajectory_history = HistoryPlan(requested=template.state_names)
    differentiate_input = args.gradient_scope in ("input", "all")
    differentiate_parameters = args.gradient_scope in ("parameters", "all")
    train_x = x.detach().requires_grad_(differentiate_input)
    train_states = tuple(
        value.detach().requires_grad_(
            args.gradient_scope == "all" and state_gradient_mask[index]
        )
        for index, value in enumerate(states)
    )
    train_parameters = tuple(
        value.detach().requires_grad_(differentiate_parameters) for value in parameters
    )
    differentiable_roots = ["input"] if differentiate_input else []
    differentiable_roots.extend(
        f"initial_state.{name}"
        for name, enabled in zip(template.state_names, state_gradient_mask, strict=True)
        if args.gradient_scope == "all" and enabled
    )
    differentiable_roots.extend(
        f"parameter.{name}"
        for name in template.parameter_names
        if differentiate_parameters
    )

    def differentiable_inputs(value, initial, params):
        return tuple(
            tensor for tensor in (value, *initial, *params) if tensor.requires_grad
        )

    def eager(value, initial, params, history=HistoryPlan()):
        output, final, records = python_module_sequence(
            value, initial, params, spec, history.requested
        )
        return (output, *final, *records)

    def parity_loss(outputs):
        return sum(
            index * value.square().mean() for index, value in enumerate(outputs, 1)
        )

    def check_values(actual, expected):
        if len(actual) != len(expected):
            raise AssertionError(
                f"Result count mismatch: {len(actual)} != {len(expected)}"
            )
        for value, reference_value in zip(actual, expected):
            torch.testing.assert_close(value, reference_value, atol=5e-6, rtol=5e-5)

    # Validate against ordinary autograd independently of the backend being
    # timed. Comparing a replay to that backend's own eager call would hide a
    # consistently incorrect kernel or handwritten derivative.
    with torch.no_grad():
        reference = eager(x, states, parameters)
        trajectory_reference = eager(x, states, parameters, trajectory_history)
    reference_gradients = torch.autograd.grad(
        eager(train_x, train_states, train_parameters)[0].sum(),
        differentiable_inputs(train_x, train_states, train_parameters),
    )
    trajectory_reference_gradients = torch.autograd.grad(
        parity_loss(
            eager(
                train_x,
                train_states,
                train_parameters,
                trajectory_history,
            )
        ),
        differentiable_inputs(train_x, train_states, train_parameters),
    )
    for backend in args.backends:

        def simulate(value, initial, params, history=HistoryPlan()):
            if backend != "eager":
                options = (
                    replace(
                        resolve_model_options(backend, spec.model), precise_exp=True
                    )
                    if args.precise_exp
                    else backend
                )
                return run_multi_step(
                    value, initial, params, spec, history, backend=options
                )
            return eager(value, initial, params, history)

        started = time.perf_counter()
        simulation = simulate(x, states, parameters)
        torch.cuda.synchronize()
        first_forward_ms = (time.perf_counter() - started) * 1000
        if any(
            value.grad_fn is not None or value.requires_grad for value in simulation
        ):
            raise AssertionError(
                "Frozen inference unexpectedly created an autograd graph"
            )
        started = time.perf_counter()
        loss = simulate(train_x, train_states, train_parameters)[0].sum()
        gradients = torch.autograd.grad(
            loss, differentiable_inputs(train_x, train_states, train_parameters)
        )
        torch.cuda.synchronize()
        first_training_ms = (time.perf_counter() - started) * 1000
        check_values(simulation, reference)
        for gradient, expected in zip(gradients, reference_gradients):
            torch.testing.assert_close(gradient, expected, atol=5e-5, rtol=5e-4)
        del gradients, loss, simulation

        # Full state trajectories and continuation states are correctness
        # checks only. Timings below request no public records.
        with torch.no_grad():
            trajectory = simulate(x, states, parameters, trajectory_history)
        check_values(trajectory, trajectory_reference)
        trajectory_gradients = torch.autograd.grad(
            parity_loss(
                simulate(
                    train_x,
                    train_states,
                    train_parameters,
                    trajectory_history,
                )
            ),
            differentiable_inputs(train_x, train_states, train_parameters),
        )
        for gradient, expected in zip(
            trajectory_gradients, trajectory_reference_gradients
        ):
            torch.testing.assert_close(gradient, expected, atol=5e-5, rtol=5e-4)
        del trajectory, trajectory_gradients

        for _ in range(args.warmup):
            simulate(x, states, parameters)
            warm_loss = simulate(train_x, train_states, train_parameters)[0].sum()
            torch.autograd.grad(
                warm_loss,
                differentiable_inputs(train_x, train_states, train_parameters),
            )
        torch.cuda.synchronize()
        del warm_loss

        inference_ms = []
        for _ in range(args.repeats):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            inference_result = simulate(x, states, parameters)
            end.record()
            end.synchronize()
            inference_ms.append(start.elapsed_time(end))
        del inference_result
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        initial_bytes = torch.cuda.memory_allocated()
        inference_result = simulate(x, states, parameters)
        torch.cuda.synchronize()
        inference_peak = torch.cuda.max_memory_allocated() - initial_bytes
        inference_return_bytes = sum(
            value.numel() * value.element_size() for value in inference_result
        )
        del inference_result

        forward_ms, backward_ms = [], []
        for _ in range(args.repeats):
            events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            events[0].record()
            loss = simulate(train_x, train_states, train_parameters)[0].sum()
            events[1].record()
            torch.autograd.grad(
                loss,
                differentiable_inputs(train_x, train_states, train_parameters),
            )
            events[2].record()
            events[2].synchronize()
            forward_ms.append(events[0].elapsed_time(events[1]))
            backward_ms.append(events[1].elapsed_time(events[2]))
        del loss
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        initial_bytes = torch.cuda.memory_allocated()
        memory_loss = simulate(train_x, train_states, train_parameters)[0].sum()
        memory_gradients = torch.autograd.grad(
            memory_loss,
            differentiable_inputs(train_x, train_states, train_parameters),
        )
        torch.cuda.synchronize()
        training_peak = torch.cuda.max_memory_allocated() - initial_bytes
        del memory_gradients, memory_loss
        # Forward simulation capture uses no backward histories. All kernels
        # were compiled above. Grad mode deliberately remains enabled while
        # every participating tensor is frozen, exercising inference dispatch.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            simulate(x, states, parameters)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = simulate(x, states, parameters)
        graph.replay()
        check_values(captured, reference)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(args.repeats):
            graph.replay()
        end.record()
        end.synchronize()
        simulation_graph_ms = start.elapsed_time(end) / args.repeats
        captured_before_change = tuple(value.detach().clone() for value in captured)
        original_x = x.detach().clone()
        with torch.no_grad():
            x.mul_(-0.37).add_(0.11)
        graph.replay()
        changed_reference = eager(x, states, parameters)
        check_values(captured, changed_reference)
        if all(
            torch.equal(before, after)
            for before, after in zip(captured_before_change, captured, strict=True)
        ):
            raise AssertionError(
                "Inference CUDA Graph outputs did not respond to changed input"
            )
        with torch.no_grad():
            x.copy_(original_x)
        # Capture the complete training call, including the registered reverse
        # operator. Replay has no Python autograd work or new allocator calls;
        # its timing is reported separately from the eager device timings.
        with torch.cuda.stream(stream):
            # Separate leaves prevent eager loss references from pinning their
            # AccumulateGrad nodes to the default stream during capture.
            capture_x = train_x.detach().clone().requires_grad_(train_x.requires_grad)
            capture_states = tuple(
                value.detach().clone().requires_grad_(value.requires_grad)
                for value in train_states
            )
            capture_parameters = tuple(
                value.detach().clone().requires_grad_(value.requires_grad)
                for value in train_parameters
            )
            for _ in range(args.warmup):
                warm_loss = simulate(capture_x, capture_states, capture_parameters)[
                    0
                ].sum()
                torch.autograd.grad(
                    warm_loss,
                    differentiable_inputs(
                        capture_x, capture_states, capture_parameters
                    ),
                )
        torch.cuda.current_stream().wait_stream(stream)
        training_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(training_graph, stream=stream):
            captured_loss = simulate(capture_x, capture_states, capture_parameters)[
                0
            ].sum()
            captured_gradients = torch.autograd.grad(
                captured_loss,
                differentiable_inputs(capture_x, capture_states, capture_parameters),
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
        gradients_before_change = tuple(
            value.detach().clone() for value in captured_gradients
        )
        with torch.no_grad():
            capture_x.mul_(0.43).sub_(0.07)
        training_graph.replay()
        changed_x = capture_x.detach().clone().requires_grad_(capture_x.requires_grad)
        changed_states = tuple(
            value.detach().clone().requires_grad_(value.requires_grad)
            for value in capture_states
        )
        changed_parameters = tuple(
            value.detach().clone().requires_grad_(value.requires_grad)
            for value in capture_parameters
        )
        changed_loss = eager(changed_x, changed_states, changed_parameters)[0].sum()
        changed_reference_gradients = torch.autograd.grad(
            changed_loss,
            differentiable_inputs(changed_x, changed_states, changed_parameters),
        )
        for gradient, expected in zip(
            captured_gradients, changed_reference_gradients, strict=True
        ):
            torch.testing.assert_close(gradient, expected, atol=5e-5, rtol=5e-4)
        reference_vjp_changed = any(
            not torch.equal(before, after)
            for before, after in zip(
                reference_gradients, changed_reference_gradients, strict=True
            )
        )
        captured_vjp_changed = any(
            not torch.equal(before, after)
            for before, after in zip(
                gradients_before_change, captured_gradients, strict=True
            )
        )
        if reference_vjp_changed and not captured_vjp_changed:
            raise AssertionError(
                "Training CUDA Graph VJPs did not respond to changed input"
            )
        print(
            json.dumps(
                {
                    "backend": backend,
                    "model": args.model,
                    "precise_exp": args.precise_exp,
                    "measurement_scope": (
                        "already_projected_drive_cell_only"
                        if args.model == "dual_exponential"
                        else "cell_only"
                    ),
                    "differentiated_initial_states": [
                        name
                        for name, enabled in zip(
                            template.state_names,
                            state_gradient_mask,
                            strict=True,
                        )
                        if args.gradient_scope == "all" and enabled
                    ],
                    "differentiable_roots": differentiable_roots,
                    "parameter_layout": args.parameter_layout,
                    "gradient_scope": args.gradient_scope,
                    "torch_version": torch.__version__,
                    "gpu": torch.cuda.get_device_name(),
                    "shape_TBN": [args.steps, args.batch, args.neurons],
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "first_forward_host_ms": first_forward_ms,
                    "first_training_host_ms": first_training_ms,
                    "inference_forward_device_ms": sum(inference_ms)
                    / len(inference_ms),
                    "inference_forward_device_median_ms": statistics.median(
                        inference_ms
                    ),
                    "inference_forward_device_range_ms": [
                        min(inference_ms),
                        max(inference_ms),
                    ],
                    "inference_peak_allocated_bytes": inference_peak,
                    "inference_return_bytes": inference_return_bytes,
                    "training_forward_device_ms": sum(forward_ms) / len(forward_ms),
                    "backward_device_ms": sum(backward_ms) / len(backward_ms),
                    "training_forward_device_median_ms": statistics.median(forward_ms),
                    "backward_device_median_ms": statistics.median(backward_ms),
                    "training_forward_device_range_ms": [
                        min(forward_ms),
                        max(forward_ms),
                    ],
                    "backward_device_range_ms": [min(backward_ms), max(backward_ms)],
                    "training_peak_allocated_bytes": training_peak,
                    "inference_graph_device_ms": simulation_graph_ms,
                    "training_graph_device_ms": training_graph_ms,
                    "cuda_graph_changed_input_output_and_vjp": "pass",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
