import time
from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn


try:
    import torch_scatter
except ImportError:
    torch_scatter = None


def expand_leading_dims(tensor, shape):
    return tensor.view(*(1,) * len(shape), *tensor.shape).expand(*shape, *tensor.shape)


class MockEmbedLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class ScatterBenchmarkModule(nn.Module):
    def __init__(self, n_neuron, observations_spec, strategy="merged_native"):
        super().__init__()
        self.n_neuron = n_neuron
        self.strategy = strategy
        self.embeds = nn.ModuleDict()
        self.neuron_ids = {}

        for key, (feat_dim, count) in observations_spec.items():
            ids = torch.randint(0, n_neuron, (count,))
            self.neuron_ids[key] = ids
            self.embeds[key] = MockEmbedLayer(feat_dim, count)

    def forward(self, observ: Dict[str, torch.Tensor]):
        if self.strategy == "seq_native":
            return self.forward_sequential_native(observ)
        elif self.strategy == "seq_scatter":
            return self.forward_sequential_scatter(observ)
        elif self.strategy == "merged_native":
            return self.forward_merged_native(observ)
        elif self.strategy == "merged_scatter":
            return self.forward_merged_scatter(observ)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def forward_sequential_native(self, observ):
        ret = None
        for k, v in observ.items():
            src = self.embeds[k](v)
            ids = self.neuron_ids[k].to(src.device)
            index = expand_leading_dims(ids, src.shape[:-1])
            if ret is None:
                ret = src.new_zeros(src.shape[:-1] + (self.n_neuron,))
            ret = ret.scatter_add(dim=-1, index=index, src=src)
        return ret

    def forward_sequential_scatter(self, observ):
        if torch_scatter is None:
            raise ImportError("torch_scatter missing")
        ret = None
        for k, v in observ.items():
            src = self.embeds[k](v)
            ids = self.neuron_ids[k].to(src.device)
            index = expand_leading_dims(ids, src.shape[:-1])
            # torch_scatter.scatter handles output creation if dim_size is given
            # We need to accumulate results manually or rely on scatter
            # But seq_scatter usually implies we scatter into a buffer iteratively
            if ret is None:
                # Initialize or just scatter first?
                # torch_scatter returns a new tensor. To accumulate, we'd need out=ret
                # But scatter(reduce='sum') doesn't exactly support 'add to existing'
                # effortlessly without 'out' arg?
                # Actually it does support 'out'.
                ret = src.new_zeros(src.shape[:-1] + (self.n_neuron,))

            # torch_scatter.scatter(src, index, dim, dim_size, reduce, out)
            # supports out argument.
            torch_scatter.scatter(
                src, index, dim=-1, dim_size=self.n_neuron, reduce="sum", out=ret
            )
        return ret

    def forward_merged_native(self, observ):
        all_srcs = []
        all_indices = []
        batch_shape = None
        for k, v in observ.items():
            src = self.embeds[k](v)
            if batch_shape is None:
                batch_shape = src.shape[:-1]
            ids = self.neuron_ids[k].to(src.device)
            expanded_ids = expand_leading_dims(ids, batch_shape)
            all_srcs.append(src)
            all_indices.append(expanded_ids)
        if not all_srcs:
            return None
        merged_src = torch.cat(all_srcs, dim=-1)
        merged_index = torch.cat(all_indices, dim=-1)
        ret = merged_src.new_zeros(batch_shape + (self.n_neuron,))
        ret = ret.scatter_add(dim=-1, index=merged_index, src=merged_src)
        return ret

    def forward_merged_scatter(self, observ):
        if torch_scatter is None:
            raise ImportError("torch_scatter missing")
        all_srcs = []
        all_indices = []
        batch_shape = None
        for k, v in observ.items():
            src = self.embeds[k](v)
            if batch_shape is None:
                batch_shape = src.shape[:-1]
            ids = self.neuron_ids[k].to(src.device)
            expanded_ids = expand_leading_dims(ids, batch_shape)
            all_srcs.append(src)
            all_indices.append(expanded_ids)
        if not all_srcs:
            return None
        merged_src = torch.cat(all_srcs, dim=-1)
        merged_index = torch.cat(all_indices, dim=-1)
        return torch_scatter.scatter(
            merged_src, merged_index, dim=-1, dim_size=self.n_neuron, reduce="sum"
        )


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def observations(device):
    batch_size = 128
    return {
        "vision": torch.randn(batch_size, 1, device=device),
        "wind": torch.randn(batch_size, 1, device=device),
        "proprio": torch.randn(batch_size, 1, device=device),
    }


@pytest.fixture
def obs_spec():
    return {
        "vision": (1, 500),
        "wind": (1, 100),
        "proprio": (1, 200),
    }


@pytest.mark.parametrize("n_neuron", [1024 * (2**i) for i in range(8)])
@pytest.mark.parametrize(
    "strategy", ["seq_native", "seq_scatter", "merged_native", "merged_scatter"]
)
@pytest.mark.parametrize("compile_mode", [None, "torch.compile"])
def test_scatter_benchmark(
    benchmark, device, observations, obs_spec, n_neuron, strategy, compile_mode
):
    if "scatter" in strategy and torch_scatter is None:
        pytest.skip("torch_scatter not installed")

    if compile_mode:
        torch._dynamo.reset()

    model = ScatterBenchmarkModule(n_neuron, obs_spec, strategy=strategy).to(device)
    if compile_mode:
        try:
            model = torch.compile(model)
        except Exception as e:
            pytest.skip(f"torch.compile failed: {e}")

    for _ in range(5):
        with torch.no_grad():
            model(observations)

    def run_forward():
        with torch.no_grad():
            output = model(observations)
            torch.cuda.synchronize() if device.type == "cuda" else None
            return output

    benchmark(run_forward)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Ensure btorch is importable
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

    from btorch.utils.file import save_fig

    print("Running 2x2x2 benchmark...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_spec = {
        "vision": (1, 2000),
        "wind": (1, 500),
        "proprio": (1, 500),
    }
    batch_size = 128
    observations = {k: torch.randn(batch_size, 1, device=device) for k in obs_spec}

    neuron_sizes = [1024 * (2**i) for i in range(8)]
    strategies = ["seq_native", "seq_scatter", "merged_native", "merged_scatter"]
    if torch_scatter is None:
        print("WARNING: torch_scatter not found, skipping related strategies.")
        strategies = [s for s in strategies if "scatter" not in s]

    compile_modes = [None, "reduce-overhead"]

    plot_data = {(s, c): [] for s in strategies for c in compile_modes}

    for n in neuron_sizes:
        print(f"\nN_Neuron: {n}")
        for strat in strategies:
            for comp in compile_modes:
                key = (strat, comp)

                # Clear compile cache
                if comp:
                    torch._dynamo.reset()

                try:
                    model = ScatterBenchmarkModule(n, obs_spec, strategy=strat).to(
                        device
                    )
                    if comp:
                        model = torch.compile(model, mode=comp)

                    # Warmup
                    for _ in range(10):
                        with torch.no_grad():
                            model(observations)

                    # Time
                    start = time.perf_counter()
                    iters = 50
                    for _ in range(iters):
                        with torch.no_grad():
                            model(observations)
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                    end = time.perf_counter()
                    avg_time = (end - start) / iters * 1000  # ms

                    print(f"  {strat:15} | compile={str(comp):15} | {avg_time:.3f} ms")
                    plot_data[key].append(avg_time)
                except Exception as e:
                    print(f"  {strat:15} | compile={str(comp):15} | Failed: {e}")
                    plot_data[key].append(None)

    # Plotting
    try:
        sns.set_theme(style="whitegrid")
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(12, 7))

    input_size = sum(count for _, count in obs_spec.values())

    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    strat_colors = {s: c for s, c in zip(strategies, colors)}

    for (strat, comp), times in plot_data.items():
        valid_idxs = [i for i, t in enumerate(times) if t is not None]
        if not valid_idxs:
            continue

        valid_ns = [neuron_sizes[i] for i in valid_idxs]
        valid_times = [times[i] for i in valid_idxs]

        # Color by strategy, Style by compile
        # Strat mapping:
        # seq_native -> 0
        # seq_scatter -> 1 etc

        color = strat_colors[strat]
        marker = "o" if comp else "x"
        linestyle = "-" if comp else "--"
        line_label = f"{strat}{' (compiled)' if comp else ''}"

        ax.plot(
            valid_ns,
            valid_times,
            label=line_label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            alpha=0.8,
        )

    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("Number of Neurons (N)")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title(f"Scatter Strategy Benchmark (Total Input Size: {input_size})")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Use save_fig
    output_path = save_fig(fig, name="scatter_benchmark_comparison_2x2x2")
    print(f"Figure saved to: {output_path}")
