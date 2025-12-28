import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from btorch.backend import coo_spmm
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
            out = coo_spmm(indices_sorted, values_sorted, B)
            out.sum().backward()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                out = coo_spmm(indices_sorted, values_sorted, B)
            torch.cuda.synchronize()
            results["triton_fwd"].append((time.time() - start) / 10)

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                out = coo_spmm(indices_sorted, values_sorted, B)
                out.sum().backward()
            torch.cuda.synchronize()
            results["triton_bwd"].append(
                (time.time() - start) / 10
            )  # Measures fwd+bwd approx

            # 1b. Triton Bool
            # Warmup
            out = coo_spmm(indices_sorted, values_sorted, B, is_bool_float=True)
            out.sum().backward()

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                out = coo_spmm(indices_sorted, values_sorted, B, is_bool_float=True)
            torch.cuda.synchronize()
            results["triton_bool_fwd"].append((time.time() - start) / 10)

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                out = coo_spmm(indices_sorted, values_sorted, B, is_bool_float=True)
                out.sum().backward()
            torch.cuda.synchronize()
            results["triton_bool_bwd"].append((time.time() - start) / 10)

            # 2. Torch
            if N <= max_torch_size:
                # Torch sparse only supports spmm (Sparse x Dense -> Dense)
                # Warmup
                out_t = torch.sparse.mm(A_torch, B)
                # A_torch typically doesn't support grad on values nicely for all
                # versions?
                # Actually standard torch.sparse.mm supports bp to values?
                # Let's check.

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(10):
                    out_t = torch.sparse.mm(A_torch, B)
                torch.cuda.synchronize()
                results["torch_fwd"].append((time.time() - start) / 10)

                # Backward
                # Re-create A_torch with grad enabled values
                # Typically `torch.sparse.mm` backward might be slow or dense-ish?
                # Requires coalesced.

                # Note: We can't simply reuse A_torch if we didn't retain graph or if
                # values don't track grad well.
                # SparseTensor construct might break graph.
                # Let's use functional if possible, but torch.sparse is mostly Tensor
                # method.

                # Benchmarking full fwd+bwd
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(10):
                    # Need to reconstruct to ensure grad flow if strictly testing
                    # autograd
                    # But here we assume A_torch carries grad?
                    # Actually A_torch.values() requires_grad=True?
                    # torch.sparse_coo_tensor docs say values must require grad.
                    # Let's ensure it does.
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

            # 3. TorchSparse
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

    # Plotting
    plt.figure(figsize=(10, 5))

    # Forward
    plt.subplot(1, 2, 1)
    plt.plot(
        valid_sizes,
        [t * 1000 for t in results["triton_fwd"]],
        label="Triton",
        marker="o",
    )
    plt.plot(
        valid_sizes,
        [t * 1000 for t in results["triton_bool_fwd"]],
        label="Triton Bool",
        marker="^",
    )
    plt.plot(
        valid_sizes,
        [t * 1000 for t in results["torch_fwd"]],
        label="Torch Native",
        marker="x",
    )
    if HAS_TORCH_SPARSE:
        plt.plot(
            valid_sizes,
            [t * 1000 for t in results["torch_sparse_fwd"]],
            label="TorchSparse",
            marker="s",
        )

    plt.title(f"SpMM Forward (Density={density})")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True)

    # Backward
    plt.subplot(1, 2, 2)
    plt.plot(
        valid_sizes,
        [t * 1000 for t in results["triton_bwd"]],
        label="Triton (Fwd+Bwd)",
        marker="o",
    )
    plt.plot(
        valid_sizes,
        [t * 1000 for t in results["triton_bool_bwd"]],
        label="Triton Bool (Fwd+Bwd)",
        marker="^",
    )
    plt.plot(
        valid_sizes,
        [t * 1000 for t in results["torch_bwd"]],
        label="Torch Native (Fwd+Bwd)",
        marker="x",
    )
    if HAS_TORCH_SPARSE:
        plt.plot(
            valid_sizes,
            [t * 1000 for t in results["torch_sparse_bwd"]],
            label="TorchSparse (Fwd+Bwd)",
            marker="s",
        )

    plt.title("SpMM Backward")
    plt.xlabel("Matrix Size N")
    plt.grid(True)

    fig_path_v = fig_path()
    plt.savefig(fig_path_v / "sparse_benchmark.png")
    print(f"Benchmark plot saved to {fig_path_v / 'sparse_benchmark.png'}")


if __name__ == "__main__":
    benchmark_op()
