import time

import matplotlib.pyplot as plt
import numpy as np
import torch

# Import specific backends for comparison
from btorch.backend.triton.sparse import coo_spmm as coo_spmm_triton
from btorch.backend.warp.sparse import coo_spmm_warp
from btorch.utils.file import fig_path


try:
    import torch_sparse

    HAS_TORCH_SPARSE = True
except ImportError:
    HAS_TORCH_SPARSE = False
    print("torch_sparse not installed, skipping comparison")


def benchmark_op():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA not available, skipping benchmark")
        return

    sizes = np.logspace(10, 17, 20, base=2, dtype=int)
    density = 1e-5
    max_torch_size = 2**14
    max_torch_sparse_size = 1.5 * 2**16

    results = {
        "triton_fwd": [],
        "triton_bwd": [],
        "warp_fwd": [],
        "warp_bwd": [],
        "warp_bool_fwd": [],
        "warp_bool_bwd": [],
        "torch_fwd": [],
        "torch_bwd": [],
        "torch_sparse_fwd": [],
        "torch_sparse_bwd": [],
        "triton_bool_fwd": [],
        "triton_bool_bwd": [],
    }

    valid_sizes = []

    for N in sizes:
        print(f"Benchmarking size {N}...")
        try:
            # Create Sparse Matrix A (N x N)
            nnz = int(N * N * density)
            indices = torch.randint(0, N, (2, nnz), device=device)
            values = torch.randn(nnz, device=device, requires_grad=True)

            # Sort indices for torch.sparse.mm and torch_sparse compatibility
            # (Strictly speaking torch_sparse likes sorted, our triton op doesn't care
            # but okay)
            # coalescing for torch sparse
            A_torch = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
            indices_sorted = A_torch.indices()
            values_sorted = A_torch.values().detach().requires_grad_(True)

            # Dense Matrix B (N x 64) - typical hidden size
            K = 64
            B = torch.randn(N, K, device=device, requires_grad=True)
            # B = (B > 0.5).float()

            # 1. Triton
            # Warmup
            out = coo_spmm_triton(indices_sorted, values_sorted, B)
            out.sum().backward()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                out = coo_spmm_triton(indices_sorted, values_sorted, B)
            torch.cuda.synchronize()
            results["triton_fwd"].append((time.time() - start) / 10)

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                out = coo_spmm_triton(indices_sorted, values_sorted, B)
                out.sum().backward()
            torch.cuda.synchronize()
            results["triton_bwd"].append(
                (time.time() - start) / 10
            )  # Measures fwd+bwd approx

            # 1b. Triton Bool
            # Warmup
            out = coo_spmm_triton(indices_sorted, values_sorted, B, is_bool_float=True)
            out.sum().backward()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                out = coo_spmm_triton(
                    indices_sorted, values_sorted, B, is_bool_float=True
                )
            torch.cuda.synchronize()
            results["triton_bool_fwd"].append((time.time() - start) / 10)

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                out = coo_spmm_triton(
                    indices_sorted, values_sorted, B, is_bool_float=True
                )
                out.sum().backward()
            torch.cuda.synchronize()
            results["triton_bool_bwd"].append((time.time() - start) / 10)

            # 2. Warp
            # Warmup
            out = coo_spmm_warp(indices_sorted, values_sorted, B)
            out.sum().backward()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                out = coo_spmm_warp(indices_sorted, values_sorted, B)
            torch.cuda.synchronize()
            results["warp_fwd"].append((time.time() - start) / 10)

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                out = coo_spmm_warp(indices_sorted, values_sorted, B)
                out.sum().backward()
            torch.cuda.synchronize()
            results["warp_bwd"].append((time.time() - start) / 10)

            # 2b. Warp Bool
            # Warmup
            out = coo_spmm_warp(indices_sorted, values_sorted, B, is_bool_float=True)
            out.sum().backward()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                out = coo_spmm_warp(
                    indices_sorted, values_sorted, B, is_bool_float=True
                )
            torch.cuda.synchronize()
            results["warp_bool_fwd"].append((time.time() - start) / 10)

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                out = coo_spmm_warp(
                    indices_sorted, values_sorted, B, is_bool_float=True
                )
                out.sum().backward()
            torch.cuda.synchronize()
            results["warp_bool_bwd"].append((time.time() - start) / 10)

            # 3. Torch native
            if N <= max_torch_size:
                # Torch sparse only supports spmm (Sparse x Dense -> Dense)
                # Warmup
                out_t = torch.sparse.mm(A_torch, B)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(10):
                    out_t = torch.sparse.mm(A_torch, B)
                torch.cuda.synchronize()
                results["torch_fwd"].append((time.time() - start) / 10)

                # Benchmarking full fwd+bwd
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(10):
                    vals = values_sorted.clone().detach().requires_grad_(True)
                    A_t = torch.sparse_coo_tensor(
                        indices_sorted, vals, (N, N)
                    ).coalesce()

                    out_t = torch.sparse.mm(A_t, B)
                    out_t.sum().backward()
                torch.cuda.synchronize()
                results["torch_bwd"].append((time.time() - start) / 10)
            else:
                results["torch_fwd"].append(float("nan"))
                results["torch_bwd"].append(float("nan"))

            # 4. TorchSparse
            if HAS_TORCH_SPARSE and N <= max_torch_sparse_size:
                # torch_sparse.spmm(index, value, m, n, matrix)
                row, col = indices_sorted

                # Warmup
                out_ts = torch_sparse.spmm(indices_sorted, values_sorted, N, N, B)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(10):
                    out_ts = torch_sparse.spmm(indices_sorted, values_sorted, N, N, B)
                torch.cuda.synchronize()
                results["torch_sparse_fwd"].append((time.time() - start) / 10)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(10):
                    out_ts = torch_sparse.spmm(indices_sorted, values_sorted, N, N, B)
                    out_ts.sum().backward()
                torch.cuda.synchronize()
                results["torch_sparse_bwd"].append((time.time() - start) / 10)
            else:
                results["torch_sparse_fwd"].append(float("nan"))
                results["torch_sparse_bwd"].append(float("nan"))

            valid_sizes.append(N)

        except Exception as e:
            print(f"OOM or Error at size {N}: {e}")
            break

    # -- Professional Plotting --
    # Use a clean style
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        pass  # Fallback to default if style not available

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Define styles for consistency
    styles = {
        "Triton": {"color": "#1f77b4", "marker": "o", "ls": "-"},
        "Triton Bool": {"color": "#1f77b4", "marker": "o", "ls": "--", "mfc": "white"},
        "Warp": {"color": "#d62728", "marker": "s", "ls": "-"},
        "Warp Bool": {"color": "#d62728", "marker": "s", "ls": "--", "mfc": "white"},
        "Torch Native": {"color": "#2ca02c", "marker": "^", "ls": "-"},
        "TorchSparse": {"color": "#9467bd", "marker": "x", "ls": "-"},
    }

    def plot_line(ax, data_key, label_key):
        if not results[data_key]:
            return
        style = styles.get(label_key, {})
        ax.plot(
            valid_sizes,
            [t * 1000 for t in results[data_key]],
            label=label_key,
            linewidth=2,
            **style,
        )

    # 1. Forward Plot
    ax = axes[0]
    plot_line(ax, "triton_fwd", "Triton")
    plot_line(ax, "triton_bool_fwd", "Triton Bool")
    plot_line(ax, "warp_fwd", "Warp")
    plot_line(ax, "warp_bool_fwd", "Warp Bool")
    plot_line(ax, "torch_fwd", "Torch Native")
    if HAS_TORCH_SPARSE:
        plot_line(ax, "torch_sparse_fwd", "TorchSparse")

    ax.set_title(f"SpMM Forward (Density={density})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Matrix Size N", fontsize=10)
    ax.set_ylabel("Time (ms)", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="major", ls="-", alpha=0.5)

    # 2. Backward Plot
    ax = axes[1]
    plot_line(ax, "triton_bwd", "Triton")
    plot_line(ax, "triton_bool_bwd", "Triton Bool")
    plot_line(ax, "warp_bwd", "Warp")
    plot_line(ax, "warp_bool_bwd", "Warp Bool")
    plot_line(ax, "torch_bwd", "Torch Native")
    if HAS_TORCH_SPARSE:
        plot_line(ax, "torch_sparse_bwd", "TorchSparse")

    ax.set_title("SpMM Backward (Fwd+Bwd)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Matrix Size N", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")  # Log scale for y is usually better for benchmarks
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="major", ls="-", alpha=0.5)

    plt.tight_layout()
    fig_path_v = fig_path()
    out_file = fig_path_v / "sparse_benchmark.png"
    plt.savefig(out_file, dpi=300)
    print(f"Benchmark plot saved to {out_file}")


if __name__ == "__main__":
    benchmark_op()
