import os
from functools import partial


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import torch
from jax.experimental import sparse as jsparse
from torch_sparse import SparseTensor
from triton.testing import Benchmark, perf_report

from tests.utils.file import fig_path

from ..utils.bench import do_bench


providers_forward = ["torch.sparse", "torch_sparse", "jax", "cupy"]
line_names_forward = providers_forward
providers_backward = ["torch.sparse", "torch_sparse", "jax"]
line_names_backward = providers_backward
styles = [("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-")]


@jax.jit
def spmv(a, b):
    return a @ b


@partial(jax.jit, static_argnames=["shape"])
@jax.value_and_grad
def spmv_grad(data, indices, b, shape):
    jax_coo = jsparse.BCOO((data, indices), shape=shape)
    return (jax_coo @ b).sum()


@perf_report(
    [
        Benchmark(
            x_names=["N"],
            x_vals=np.logspace(4, 5, 20, dtype=int),
            line_arg="provider",
            line_vals=providers_forward,
            line_names=line_names_forward,
            styles=styles,
            ylabel="GB/s",
            plot_name="spmv_bandwidth_vs_size",
            args={"density": 10**-4},
            x_log=True,
        ),
        Benchmark(
            x_names=["density"],
            x_vals=np.logspace(-5, -3, 20).tolist(),
            line_arg="provider",
            line_vals=providers_forward,
            line_names=line_names_forward,
            styles=styles,
            ylabel="GB/s",
            plot_name="spmv_bandwidth_vs_density",
            # args={"N": 1_000_000},
            args={"N": 60_000},
            x_log=True,
        ),
    ]
)
def bench_spmv_forward(N, density, provider):
    size = int(N)

    # TODO: sp.random_array is extremely slow, optimisation ideas?
    coo = sp.random_array(shape=(size, size), density=density, dtype=np.float32)
    x_np = np.random.rand(size).astype(np.float32)

    if provider == "torch.sparse":
        # torch.sparse only supports COO on CPU/GPU with sparse_coo_tensor
        indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.int64)
        values = torch.tensor(coo.data, dtype=torch.float32)
        torch_coo = torch.sparse_coo_tensor(
            indices=indices, values=values, size=coo.shape, device="cuda"
        )
        x = torch.tensor(x_np, device="cuda")
        fn = lambda: torch.sparse.mm(torch_coo, x.unsqueeze(1))

    elif provider == "torch_sparse":
        coo_tensor = SparseTensor(
            row=torch.tensor(coo.row, device="cuda", dtype=torch.long),
            col=torch.tensor(coo.col, device="cuda", dtype=torch.long),
            value=torch.tensor(coo.data, device="cuda"),
            sparse_sizes=coo.shape,
        )
        x = torch.tensor(x_np, device="cuda")
        fn = lambda: coo_tensor.matmul(x.unsqueeze(1))

    elif provider == "cupy":
        cupy_coo = cp.sparse.coo_matrix(coo)
        x = cp.asarray(x_np)
        fn = lambda: cupy_coo @ x

    elif provider == "jax":
        # Create BCOO from COO data
        data = jnp.array(coo.data)
        indices = jnp.stack([jnp.array(coo.row), jnp.array(coo.col)], axis=1)
        jax_coo = jsparse.BCOO((data, indices), shape=coo.shape).block_until_ready()

        x = jnp.array(x_np).block_until_ready()

        fn = lambda: spmv(jax_coo, x).block_until_ready()

    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms = do_bench(fn, sync_cuda=provider != "jax", quantiles=[0.5, 0.2, 0.8])

    # Compute GB/s
    if provider in ["torch.sparse", "torch_sparse"]:
        numel = x.numel()
        elem_size = x.element_size()
    elif provider == "cupy":
        numel = int(x.size)
        elem_size = x.itemsize
    else:
        numel = x.size
        elem_size = x.itemsize

    gbps = lambda ms: (
        (
            2 * numel * elem_size
            + size * size * density * (2 * elem_size + coo.data.itemsize)
        )
        / (ms * 1e-3)
        / 1e9
    )
    return tuple(gbps(t) for t in ms)


@perf_report(
    [
        Benchmark(
            x_names=["N"],
            x_vals=np.logspace(3, 4.5, 20, dtype=int),
            line_arg="provider",
            line_vals=providers_backward,
            line_names=line_names_backward,
            styles=styles[: len(providers_backward)],
            ylabel="GB/s",
            plot_name="spmv_bandwidth_vs_size_with_grad",
            args={"density": 10**-4},
            x_log=True,
        ),
        Benchmark(
            x_names=["density"],
            x_vals=np.logspace(-5, -3, 20).tolist(),
            line_arg="provider",
            line_vals=providers_backward,
            line_names=line_names_backward,
            styles=styles[: len(providers_backward)],
            ylabel="GB/s",
            plot_name="spmv_bandwidth_vs_density_with_grad",
            args={"N": 10_000},
            x_log=True,
        ),
    ]
)
def bench_spmv_forward_backward(N, density, provider):
    size = int(N)
    coo = sp.random_array(shape=(size, size), density=density, dtype=np.float32)
    x_np = np.random.rand(size).astype(np.float32)

    if provider == "torch.sparse":
        # torch.sparse only supports COO on CPU/GPU with sparse_coo_tensor
        indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        x = torch.tensor(x_np, device="cuda")

        def fn():
            torch_coo = torch.sparse_coo_tensor(
                indices=indices,
                values=values,
                size=coo.shape,
                device="cuda",
                requires_grad=True,
            )
            torch.sparse.mm(torch_coo, x.unsqueeze(1)).sum().backward()
            torch_coo.grad.zero_()

    elif provider == "torch_sparse":
        values = torch.tensor(coo.data, device="cuda", requires_grad=True)
        x = torch.tensor(x_np, device="cuda")

        def fn():
            coo_tensor = SparseTensor(
                row=torch.tensor(coo.row, device="cuda", dtype=torch.long),
                col=torch.tensor(coo.col, device="cuda", dtype=torch.long),
                value=values,
                sparse_sizes=coo.shape,
            )
            coo_tensor.matmul(x.unsqueeze(1)).sum().backward()
            values.grad.zero_()

    elif provider == "jax":
        # Create BCOO from COO data
        data = jnp.array(coo.data).block_until_ready()
        indices = jnp.stack(
            [jnp.array(coo.row), jnp.array(coo.col)], axis=1
        ).block_until_ready()

        x = jnp.array(x_np).block_until_ready()

        def fn():
            v, grad = spmv_grad(data, indices, x, coo.shape)
            grad.block_until_ready()

    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms = do_bench(
        fn,
        sync_cuda=provider != "jax",
        # grad_to_none=[] if provider == "jax" else [values],
        quantiles=[0.5, 0.2, 0.8],
    )

    # Compute GB/s
    if provider in ["torch.sparse", "torch_sparse"]:
        numel = x.numel()
        elem_size = x.element_size()
    else:
        numel = x.size
        elem_size = x.itemsize

    gbps = lambda ms: (
        (
            2 * numel * elem_size
            + size * size * density * (2 * elem_size + coo.data.itemsize)
        )
        / (ms * 1e-3)
        / 1e9
    )
    return tuple(gbps(t) for t in ms)


def test_compile_sparse(size=100, density=0.01):
    coo = sp.random_array(shape=(size, size), density=density, dtype=np.float32)
    x_np = np.random.rand(size).astype(np.float32)
    values = torch.tensor(coo.data, device="cuda")
    x = torch.tensor(x_np, device="cuda")

    @torch.compile
    def fn(values, x):
        coo_tensor = SparseTensor(
            row=torch.tensor(coo.row, device="cuda", dtype=torch.long),
            col=torch.tensor(coo.col, device="cuda", dtype=torch.long),
            value=values,
            sparse_sizes=coo.shape,
        )
        coo_tensor.matmul(x.unsqueeze(1)).sum()

    fn(values, x)


if __name__ == "__main__":
    # jax.config.update("jax_enable_x64", True)
    # jax.config.update("jax_platform_name", "cpu")

    # print(do_bench(f, quantiles=[0.2, 0.5, 0.8]))

    # X-fail
    # https://github.com/rusty1s/pytorch_sparse/issues/400
    # test_compile_sparse()

    save_path = fig_path(__file__)
    bench_spmv_forward.run(save_path=save_path, show_plots=False)
    bench_spmv_forward_backward.run(save_path=save_path, show_plots=False)
