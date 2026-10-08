import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import btorch.sparse as btorch_sparse
from btorch.models import environ, functional, rnn, synapse
from btorch.models.connection import SparseConnection
from btorch.models.linear import Linear
from btorch.models.neurons import GLIF3
from btorch.sparse import Hints
from btorch.utils.bench import do_bench


@dataclass(frozen=True)
class RSNNWeights:
    input: torch.Tensor
    recurrent_pre: torch.Tensor
    recurrent_post: torch.Tensor
    recurrent_value: torch.Tensor
    output: torch.Tensor


class MinimalRSNN(nn.Module):
    """A minimal Recurrent Spiking Neural Network (RSNN) demonstrating the use
    of btorch's core Spiking Neuron models (GLIF) and AlphaPSC synapses wrapped
    in a multi-step RecurrentNN layer."""

    def __init__(
        self,
        num_input: int,
        num_hidden: int,
        num_output: int,
        recurrent_mode: str = "sparse",
        recurrent_density: float = 0.1,
        expected_spike_density: float = 0.01,
        weights: RSNNWeights | None = None,
        record_states: bool = True,
        seed: int = 0,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()

        if recurrent_mode not in ("dense", "sparse"):
            raise ValueError(
                f"recurrent_mode must be 'dense' or 'sparse', got {recurrent_mode!r}"
            )
        if weights is None:
            weights = _make_shared_weights(
                num_input,
                num_hidden,
                num_output,
                recurrent_density,
                torch.device(device or "cpu"),
                dtype,
                seed,
            )

        # 1. Input projection
        self.fc_in = nn.Linear(
            num_input, num_hidden, bias=False, device=device, dtype=dtype
        )
        with torch.no_grad():
            self.fc_in.weight.copy_(weights.input)

        # 2. Recurrent Spiking Layer
        # Define the biological spiking neuron.
        # Note: parameters (c_m, tau, tau_ref, k, asc_amps, etc.) can also be
        # a numpy array or tensor with an N-neuron dimension
        # (e.g., shape `(num_hidden,)`) to support heterogeneous parameters.
        neuron_module = GLIF3(
            n_neuron=num_hidden,
            v_threshold=-45.0,
            v_reset=-60.0,
            c_m=2.0,
            tau=20.0,
            tau_ref=2.0,
            k=[0.1, 0.2],
            asc_amps=[1.0, -2.0],
            step_mode="s",  # single step definition
            backend="torch",
            device=device,
            dtype=dtype,
        )

        if recurrent_mode == "dense":
            recurrent = torch.zeros(num_hidden, num_hidden, device=device, dtype=dtype)
            recurrent[weights.recurrent_pre, weights.recurrent_post] = (
                weights.recurrent_value
            )
            conn = Linear(
                num_hidden,
                num_hidden,
                weight=recurrent,
                bias=None,
                device=device,
                dtype=dtype,
            )
        else:
            conn = SparseConnection.from_edges(
                weights.recurrent_pre,
                weights.recurrent_post,
                num_hidden,
                num_hidden,
                values=weights.recurrent_value,
                hints=Hints(expected_density=expected_spike_density),
                device=device,
                dtype=dtype,
            )

        # Define the synaptic dynamics (Alpha Post-Synaptic Current)
        psc_module = synapse.AlphaPSC(
            n_neuron=num_hidden,
            tau_syn=5.0,
            linear=conn,
            step_mode="s",
        )

        # Wrap into a RecurrentNN multi-step layer
        self.brain = rnn.RecurrentNN(
            neuron=neuron_module,
            synapse=psc_module,
            step_mode="m",  # process multiple time steps (T, B, ...)
            # `update_state_names` takes a tuple of dot-separated strings
            # to select which internal state variables to record and return.
            # E.g., "neuron.v" fetches the membrane potential `v` of `neuron_module`
            # at every timestep. "synapse.psc" fetches the post-synaptic current.
            # Depending on register_memory, some state vars like Iasc in glif
            # can have an extra dimension, yielding a shape like
            # (T, Batch, num_hidden, num_Iasc).
            update_state_names=("neuron.v", "neuron.Iasc", "synapse.psc")
            if record_states
            else (),
        )

        # 3. Output readout
        self.fc_out = nn.Linear(
            num_hidden, num_output, bias=False, device=device, dtype=dtype
        )
        with torch.no_grad():
            self.fc_out.weight.copy_(weights.output)

    def forward(self, x: torch.Tensor, return_states: bool = False):
        """Forward pass.

        Args:
            x (torch.Tensor): Input sequence of shape (T, Batch, num_input)
            return_states (bool): If True, returns (out, spike, states)
        Returns:
            torch.Tensor or tuple: Output of shape (Batch, num_output),
                                   or (out, spike, states) if requested.
        """
        # Linear projection along the time dimension
        x = self.fc_in(x)  # -> (T, Batch, num_hidden)

        # Process dynamically through recurrent brain layer
        # `spike` will have shape (T, Batch, num_hidden)
        #
        # `states` is a dictionary containing the recorded state variables at each
        # timestep according to `update_state_names` (keys like "neuron.v").
        # You can use `btorch.utils.dict_utils.unflatten_dict(states, dot=True)`
        # to convert this flat dotted-dict into a nested dict:
        # `{"neuron": {"v": Tensor}, "synapse": {"psc": Tensor}}`
        spike, states = self.brain(
            x
        )  # spike: (T, Batch, num_hidden), states values: (T, Batch, num_hidden, ...)

        # Decode using rate-based output (mean spike rate over time)
        rate = spike.mean(dim=0)  # -> (Batch, num_hidden)
        out = self.fc_out(rate)  # -> (Batch, num_output)

        if return_states:
            return out, spike, states
        return out


def sim(
    net: nn.Module,
    num_input: int,
    timesteps: int = 100,
    dt: float = 1.0,
    batch_size: int = 4,
    device="cpu",
):
    """Run a simple forward simulation without training."""
    import matplotlib.pyplot as plt

    from btorch.utils.dict_utils import unflatten_dict
    from btorch.utils.file import save_fig
    from btorch.visualisation.timeseries import (
        SimulationStates,
        plot_neuron_traces,
        plot_raster,
    )

    # Global environment config required for ODE solvers inside neurons
    environ.set(dt=dt)
    print(f"\n--- Running Simulation (dt={dt}, T={timesteps}) ---")

    print("Initializing network internal memory states...")
    functional.init_net_state(
        net, batch_size=batch_size, device=device, dtype=torch.float32
    )

    net.eval()

    # -> (T, Batch, num_input)
    inputs = 10 * torch.rand((timesteps, batch_size, num_input), device=device)

    with torch.no_grad():
        # Reset Network State Before Simulation
        functional.reset_net(
            net, batch_size=batch_size, device=device, dtype=torch.float32
        )
        out, spike, states = net(inputs, return_states=True)
        # out: (Batch, num_output), spike: (T, Batch, num_hidden)

    print(f"Simulation complete. Output shape: {out.shape}")

    # Plotting first batch sample
    print("Generating and saving figures...")
    spike_b0 = spike[:, 0, :]  # -> (T, num_hidden)
    states_nested = unflatten_dict(states, dot=True)
    v_b0 = states_nested["neuron"]["v"][:, 0, :]  # -> (T, num_hidden)
    Iasc_b0 = states_nested["neuron"]["Iasc"][:, 0, ...]  # -> (T, num_hidden, num_Iasc)
    psc_b0 = states_nested["synapse"]["psc"][:, 0, ...]  # -> (T, num_hidden, num_psc)

    # Raster plot
    ax_raster = plot_raster(spike_b0, dt=dt, title="Raster Plot (Batch 0)")
    fig_raster = (
        ax_raster[0].figure if isinstance(ax_raster, tuple) else ax_raster.figure
    )
    save_fig(fig_raster, "rsnn_raster")
    plt.close(fig_raster)

    # Neuron Traces (Plotting first 5 neurons for clarity)
    ax_traces = plot_neuron_traces(
        SimulationStates(
            voltage=v_b0[:, :5],
            dt=dt,
            spikes=spike_b0[:, :5],
            asc=Iasc_b0[:, :5, ...],
            psc=psc_b0[:, :5, ...],
        )
    )
    fig_traces = (
        ax_traces[0].figure if isinstance(ax_traces, tuple) else ax_traces.figure
    )
    save_fig(fig_traces, "rsnn_traces")
    plt.close(fig_traces)
    print("Figures saved successfully.")


def train(
    net: nn.Module,
    num_input: int,
    num_output: int,
    timesteps: int = 15,
    dt: float = 1.0,
    batch_size: int = 4,
    epochs: int = 10,
    device="cpu",
):
    """Run a simple dummy training loop."""
    # Global environment config required for ODE solvers inside neurons
    environ.set(dt=dt)
    print(f"\n--- Running Training (dt={dt}, T={timesteps}, epochs={epochs}) ---")

    print("Initializing network internal memory states...")
    functional.init_net_state(
        net, batch_size=batch_size, device=device, dtype=torch.float32
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    net.train()
    for epoch in range(epochs):
        # Create Dummy Data
        # inputs -> (T, Batch, num_input)
        inputs = torch.rand((timesteps, batch_size, num_input), device=device)
        # targets -> (Batch, num_output)
        targets = torch.rand((batch_size, num_output), device=device)

        # Reset Network State Before Simulation step
        # batch_size can be skipped if it is not changed
        functional.reset_net(
            net, batch_size=batch_size, device=device, dtype=torch.float32
        )

        optimizer.zero_grad()
        out = net(inputs)  # -> (Batch, num_output)
        loss = F.mse_loss(out, targets)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Step Loss: {loss.item():.4f}")

    print("Training complete!")


def _make_shared_weights(
    num_input: int,
    num_hidden: int,
    num_output: int,
    density: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> RSNNWeights:
    if not 0.0 < density <= 1.0:
        raise ValueError(f"recurrent density must be in (0, 1], got {density}")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_weight = (
        torch.randn(num_hidden, num_input, generator=generator, dtype=dtype)
        / num_input**0.5
    )
    n_random = max(0, round(num_hidden * num_hidden * density) - num_hidden)
    random_edge = torch.randint(
        num_hidden * num_hidden, (n_random,), generator=generator
    )
    diagonal = torch.arange(num_hidden) * (num_hidden + 1)
    edge = torch.unique(torch.cat([diagonal, random_edge]), sorted=True)
    recurrent_pre = edge // num_hidden
    recurrent_post = edge % num_hidden
    recurrent_value = (
        torch.randn(recurrent_pre.shape[0], generator=generator, dtype=dtype)
        / num_hidden**0.5
    )
    output_weight = (
        torch.randn(num_output, num_hidden, generator=generator, dtype=dtype)
        / num_hidden**0.5
    )
    return RSNNWeights(
        input=input_weight.to(device=device),
        recurrent_pre=recurrent_pre.to(device=device),
        recurrent_post=recurrent_post.to(device=device),
        recurrent_value=recurrent_value.to(device=device),
        output=output_weight.to(device=device),
    )


def _build_net(
    *,
    mode: str,
    num_input: int,
    num_hidden: int,
    num_output: int,
    recurrent_density: float,
    expected_spike_density: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    shared_weights: RSNNWeights | None = None,
    record_states: bool = True,
) -> MinimalRSNN:
    weights = shared_weights or _make_shared_weights(
        num_input,
        num_hidden,
        num_output,
        recurrent_density,
        device,
        dtype,
        seed,
    )
    return MinimalRSNN(
        num_input=num_input,
        num_hidden=num_hidden,
        num_output=num_output,
        recurrent_mode=mode,
        recurrent_density=recurrent_density,
        expected_spike_density=expected_spike_density,
        weights=weights,
        record_states=record_states,
        seed=seed,
        device=device,
        dtype=dtype,
    )


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def compare_paths(
    *,
    num_input: int,
    num_hidden: int,
    num_output: int,
    timesteps: int,
    batch_size: int,
    recurrent_density: float,
    expected_spike_density: float,
    dt: float,
    device: torch.device,
    seed: int,
    benchmark: bool,
    warmup: int,
    repeats: int,
    input_scale: float,
    json_path: Path | None,
) -> None:
    dtype = torch.float32
    environ.set(dt=dt)
    weights = _make_shared_weights(
        num_input,
        num_hidden,
        num_output,
        recurrent_density,
        device,
        dtype,
        seed,
    )
    dense = _build_net(
        mode="dense",
        num_input=num_input,
        num_hidden=num_hidden,
        num_output=num_output,
        recurrent_density=recurrent_density,
        expected_spike_density=expected_spike_density,
        device=device,
        dtype=dtype,
        seed=seed,
        shared_weights=weights,
        record_states=False,
    )
    sparse = _build_net(
        mode="sparse",
        num_input=num_input,
        num_hidden=num_hidden,
        num_output=num_output,
        recurrent_density=recurrent_density,
        expected_spike_density=expected_spike_density,
        device=device,
        dtype=dtype,
        seed=seed,
        shared_weights=weights,
        record_states=False,
    )
    inputs = input_scale * torch.rand(
        timesteps, batch_size, num_input, device=device, dtype=dtype
    )
    functional.init_net_state(dense, batch_size=batch_size, device=device, dtype=dtype)
    functional.init_net_state(sparse, batch_size=batch_size, device=device, dtype=dtype)

    with torch.no_grad():
        functional.reset_net(dense, batch_size=batch_size)
        dense_out, dense_spike, dense_states = dense(inputs, return_states=True)
        functional.reset_net(sparse, batch_size=batch_size)
        sparse_out, sparse_spike, sparse_states = sparse(inputs, return_states=True)

    differences = {
        "output": _max_abs_diff(dense_out, sparse_out),
        "spike_mismatch_rate": float((dense_spike != sparse_spike).float().mean()),
    }
    for name in dense_states:
        differences[f"state:{name}"] = _max_abs_diff(
            dense_states[name], sparse_states[name]
        )
    output_tolerance = 1e-4 if device.type == "cuda" else 1e-6
    spike_tolerance = 1e-5 if device.type == "cuda" else 0.0
    if (
        differences["output"] > output_tolerance
        or differences["spike_mismatch_rate"] > spike_tolerance
    ):
        raise AssertionError(
            "dense/sparse outputs differ beyond tolerances "
            f"(output={output_tolerance}, spikes={spike_tolerance}): {differences}"
        )

    sparse_conn = sparse.brain.synapse.linear
    measured_spike_density = float((dense_spike != 0).float().mean())
    print("Dense/sparse equivalence: PASS")
    print(f"Sparse runtime: {btorch_sparse.explain(sparse_conn, dense_spike)}")
    print(f"Maximum absolute differences: {differences}")

    results = {
        "differences": differences,
        "measured_spike_density": measured_spike_density,
    }
    if benchmark:

        def timed(net: nn.Module) -> None:
            functional.reset_net(net, batch_size=batch_size)
            with torch.no_grad():
                net(inputs)

        dense_ms = do_bench(
            lambda: timed(dense),
            warmup=warmup,
            rep=repeats,
            return_mode="median",
            timing_method="cpu",
            sync_cuda=True,
        )
        sparse_ms = do_bench(
            lambda: timed(sparse),
            warmup=warmup,
            rep=repeats,
            return_mode="median",
            timing_method="cpu",
            sync_cuda=True,
        )
        results.update(
            dense_ms=float(dense_ms),
            sparse_ms=float(sparse_ms),
            speedup=float(dense_ms / sparse_ms),
        )
        print(
            f"End-to-end median: dense={dense_ms:.3f} ms, "
            f"sparse={sparse_ms:.3f} ms, speedup={dense_ms / sparse_ms:.2f}x"
        )
    if json_path is not None:
        json_path.write_text(json.dumps(results, indent=2) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and compare a GLIF RSNN.")
    parser.add_argument(
        "--mode", choices=("dense", "sparse", "compare"), default="sparse"
    )
    parser.add_argument("--device", default=None, choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-input", type=int, default=20)
    parser.add_argument("--num-hidden", type=int, default=64)
    parser.add_argument("--num-output", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--recurrent-density", type=float, default=0.01)
    parser.add_argument("--expected-spike-density", type=float, default=0.01)
    parser.add_argument("--input-scale", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Simple Hyperparameters
    num_input = args.num_input
    num_hidden = args.num_hidden
    num_output = args.num_output

    if args.mode == "compare":
        compare_paths(
            num_input=num_input,
            num_hidden=num_hidden,
            num_output=num_output,
            timesteps=args.timesteps,
            batch_size=args.batch_size,
            recurrent_density=args.recurrent_density,
            expected_spike_density=args.expected_spike_density,
            dt=args.dt,
            device=device,
            seed=args.seed,
            benchmark=args.benchmark,
            warmup=args.warmup,
            repeats=args.repeats,
            input_scale=args.input_scale,
            json_path=args.json,
        )
        return

    net = _build_net(
        mode=args.mode,
        num_input=num_input,
        num_hidden=num_hidden,
        num_output=num_output,
        recurrent_density=args.recurrent_density,
        expected_spike_density=args.expected_spike_density,
        device=device,
        dtype=torch.float32,
        seed=args.seed,
    )
    print("\nNetwork Architecture:")
    print(net)

    if args.benchmark:
        compare_paths(
            num_input=num_input,
            num_hidden=num_hidden,
            num_output=num_output,
            timesteps=args.timesteps,
            batch_size=args.batch_size,
            recurrent_density=args.recurrent_density,
            expected_spike_density=args.expected_spike_density,
            dt=args.dt,
            device=device,
            seed=args.seed,
            benchmark=True,
            warmup=args.warmup,
            repeats=args.repeats,
            input_scale=args.input_scale,
            json_path=args.json,
        )
        return

    if not args.skip_simulation:
        sim(
            net,
            num_input=num_input,
            timesteps=args.timesteps,
            dt=args.dt,
            batch_size=args.batch_size,
            device=device,
        )

    if not args.skip_training:
        train(
            net,
            num_input=num_input,
            num_output=num_output,
            timesteps=min(args.timesteps, 15),
            dt=args.dt,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=device,
        )


if __name__ == "__main__":
    main()
