import time

import numpy as np
import pytest
import torch
import torch.nn as nn


def expand_leading_dims(tensor, shape):
    return tensor.view(*(1,) * len(shape), *tensor.shape).expand(*shape, *tensor.shape)


class MockEmbedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Identity()

    def forward(self, x):
        return self.net(x)


class GatherBenchmarkModule(nn.Module):
    def __init__(self, n_neuron, output_specs, strategy="merged"):
        super().__init__()
        self.n_neuron = n_neuron
        self.strategy = strategy
        self.output_specs = output_specs  # Dict[key, count]

        self.embeds = nn.ModuleDict()
        self.neuron_ids = {}

        self.all_ids_list = []
        self.splits = []

        for key, count in output_specs.items():
            ids = torch.randint(0, n_neuron, (count,))
            self.neuron_ids[key] = ids
            self.embeds[key] = MockEmbedLayer()
            self.all_ids_list.append(ids)
            self.splits.append(count)

        self.cat_ids = torch.cat(self.all_ids_list)

    def forward(self, x: torch.Tensor):
        # x shape: [..., n_neuron]
        if self.strategy == "sequential":
            return self.forward_sequential(x)
        elif self.strategy == "merged":
            return self.forward_merged(x)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def forward_sequential(self, x):
        ret = {}
        # Iterate and gather
        for key, ids in self.neuron_ids.items():
            # ids need to be moved to device if not already? assumed on device
            # gather: x[..., ids]
            gathered = x[..., ids]
            ret[key] = self.embeds[key](gathered)
        return ret

    def forward_merged(self, x):
        # Gather once
        # Assuming ids are on same device
        all_gathered = x[..., self.cat_ids]

        # Split
        # torch.split returns a tuple of views
        split_gathered = torch.split(all_gathered, self.splits, dim=-1)

        ret = {}
        for i, (key, _) in enumerate(self.output_specs.items()):
            ret[key] = self.embeds[key](split_gathered[i])
        return ret


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def input_tensor(device):
    # Large state tensor
    # e.g. [B, n_neuron]
    batch_size = 128
    n_neuron = 131072  # Fixed large N for this test? Or sweep?
    return torch.randn(batch_size, n_neuron, device=device)


@pytest.fixture
def output_specs():
    # Simulate reading out multiple populations
    return {
        "dn": 500,
        "mbon": 200,
        "kc": 2000,
        "visual": 1000,
        "motor": 300,
    }


@pytest.mark.parametrize("n_neuron", [1024 * (2**i) for i in range(8)])
@pytest.mark.parametrize("strategy", ["sequential", "merged"])
@pytest.mark.parametrize("compile_mode", [None, "reduce-overhead"])
def test_gather_benchmark(
    benchmark, device, output_specs, n_neuron, strategy, compile_mode
):
    # Adjust input size based on n_neuron param
    batch_size = 128
    x = torch.randn(batch_size, n_neuron, device=device)

    if compile_mode:
        torch._dynamo.reset()

    # Ensure ids are in range
    model = GatherBenchmarkModule(n_neuron, output_specs, strategy=strategy).to(device)
    # Move ids to device
    for k in model.neuron_ids:
        model.neuron_ids[k] = model.neuron_ids[k].to(device)
    model.cat_ids = model.cat_ids.to(device)

    if compile_mode:
        try:
            model = torch.compile(model, mode=compile_mode)
        except Exception as e:
            pytest.skip(f"torch.compile failed: {e}")

    for _ in range(5):
        with torch.no_grad():
            model(x)

    def run_forward():
        with torch.no_grad():
            output = model(x)
            torch.cuda.synchronize() if device.type == "cuda" else None
            return output

    benchmark(run_forward)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import seaborn as sns

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

    from btorch.utils.file import save_fig

    print("Running Gather Benchmark...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_specs = {
        "dn": 500,
        "mbon": 200,
        "kc": 2000,
        "visual": 1000,
        "motor": 300,
    }
    batch_size = 128

    strategies = ["sequential", "merged"]
    compile_modes = [None, "torch.compile"]
    neuron_sizes = [1024 * (2**i) for i in range(8)]

    plot_data = {(s, c): [] for s in strategies for c in compile_modes}

    for n in neuron_sizes:
        print(f"\nN_Neuron: {n}")
        x = torch.randn(batch_size, n, device=device)

        for strat in strategies:
            for comp in compile_modes:
                key = (strat, comp)

                if comp:
                    torch._dynamo.reset()

                try:
                    model = GatherBenchmarkModule(n, output_specs, strategy=strat).to(
                        device
                    )
                    # Move ids
                    for k in model.neuron_ids:
                        model.neuron_ids[k] = model.neuron_ids[k].to(device)
                    model.cat_ids = model.cat_ids.to(device)

                    if comp:
                        model = torch.compile(model)

                    # Warmup
                    for _ in range(10):
                        with torch.no_grad():
                            model(x)

                    # Time
                    start = time.perf_counter()
                    iters = 100
                    for _ in range(iters):
                        with torch.no_grad():
                            model(x)
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

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    strat_colors = {s: c for s, c in zip(strategies, colors)}

    for (strat, comp), times in plot_data.items():
        valid_idxs = [i for i, t in enumerate(times) if t is not None]
        if not valid_idxs:
            continue

        valid_ns = [neuron_sizes[i] for i in valid_idxs]
        valid_times = [times[i] for i in valid_idxs]

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
        )

    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("Total Neurons (N)")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title("Gather Strategy Benchmark (EnvOutputLayer)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    output_path = save_fig(fig, name="gather_benchmark_comparison")
    print(f"Figure saved to: {output_path}")
