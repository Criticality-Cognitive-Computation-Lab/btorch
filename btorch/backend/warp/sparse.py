import torch
import torch.library
import warp as wp


# Initialize Warp
wp.init()


@wp.kernel
def coo_spmm_kernel(
    A_indices_rows: wp.array(dtype=wp.int64),
    A_indices_cols: wp.array(dtype=wp.int64),
    A_values: wp.array(dtype=wp.float32),
    B: wp.array(dtype=wp.float32, ndim=2),
    C: wp.array(dtype=wp.float32, ndim=2),
    is_bool_float: int,  # 0 or 1
):
    tid = wp.tid()
    N = B.shape[1]

    idx_nnz = tid // N
    idx_n = tid % N

    if idx_nnz < A_values.shape[0]:
        row = A_indices_rows[idx_nnz]
        col = A_indices_cols[idx_nnz]
        val = A_values[idx_nnz]

        b_val = B[col, idx_n]

        res = float(0.0)
        if is_bool_float != 0:
            if b_val > 0.5:
                res = val
            else:
                res = 0.0
        else:
            res = val * b_val

        wp.atomic_add(C, row, idx_n, res)


@wp.kernel
def sddmm_kernel(
    A_indices_rows: wp.array(dtype=wp.int64),
    A_indices_cols: wp.array(dtype=wp.int64),
    A_grad_values: wp.array(dtype=wp.float32),
    Grad: wp.array(dtype=wp.float32, ndim=2),
    B: wp.array(dtype=wp.float32, ndim=2),
    is_bool_float: int,
):
    tid = wp.tid()

    if tid < A_indices_rows.shape[0]:
        row = A_indices_rows[tid]
        col = A_indices_cols[tid]
        N = B.shape[1]

        acc = float(0.0)

        for n in range(N):
            g_val = Grad[row, n]
            b_val = B[col, n]

            if is_bool_float != 0:
                if b_val > 0.5:
                    acc += g_val
            else:
                acc += g_val * b_val

        A_grad_values[tid] = acc


# --- Custom Ops Definitions ---


def _warp_coo_spmm_impl(indices, values, dense, output, is_bool_float):
    # Warp launch wrapper
    # Note: 'output' is passed in here to be filled, but custom op typical style is
    # return new tensor.
    # However, to be safe with allocs, let's alloc inside or pass in.
    # Standard torch ops usually return new tensor.
    # But here we want to modify 'output' in-place if we passed it?
    # Actually, for custom_op integration, let's stick to functional style: return
    # output.

    # We will allocate output inside this impl to be clean, or pass it?
    # If we pass it, it must be an input.
    pass


# We define the custom library for pure functional calls
# Namespace: btorch_warp


@torch.library.custom_op("btorch_warp::coo_spmm_driver", mutates_args=())
def coo_spmm_driver(
    indices: torch.Tensor,
    values: torch.Tensor,
    dense: torch.Tensor,
    M: int,
    is_bool_float: bool,
) -> torch.Tensor:
    K, N = dense.shape
    device = dense.device
    out = torch.zeros((M, N), device=device, dtype=dense.dtype)

    nnz = values.numel()
    if nnz == 0:
        return out

    w_indices_rows = wp.from_torch(indices[0], dtype=wp.int64)
    w_indices_cols = wp.from_torch(indices[1], dtype=wp.int64)
    w_values = wp.from_torch(values, dtype=wp.float32)
    w_dense = wp.from_torch(dense, dtype=wp.float32)
    w_out = wp.from_torch(out, dtype=wp.float32)

    wp.launch(
        kernel=coo_spmm_kernel,
        dim=nnz * N,
        inputs=[
            w_indices_rows,
            w_indices_cols,
            w_values,
            w_dense,
            w_out,
            1 if is_bool_float else 0,
        ],
        device=f"cuda:{device.index}" if device.type == "cuda" else "cpu",
    )
    return out


@coo_spmm_driver.register_fake
def _(indices, values, dense, M, is_bool_float):
    K, N = dense.shape
    return torch.empty((M, N), device=dense.device, dtype=dense.dtype)


@torch.library.custom_op("btorch_warp::sddmm_driver", mutates_args=())
def sddmm_driver(
    indices: torch.Tensor, dense: torch.Tensor, grad: torch.Tensor, is_bool_float: bool
) -> torch.Tensor:
    # returns grad_values
    nnz = indices.shape[1]
    device = dense.device
    grad_values = torch.zeros(nnz, device=device, dtype=dense.dtype)

    if nnz == 0:
        return grad_values

    # Convert to warp
    w_indices_rows = wp.from_torch(indices[0], dtype=wp.int64)
    w_indices_cols = wp.from_torch(indices[1], dtype=wp.int64)
    w_grad_values = wp.from_torch(grad_values, dtype=wp.float32)
    w_grad_out = wp.from_torch(grad, dtype=wp.float32)
    w_dense = wp.from_torch(dense, dtype=wp.float32)

    wp.launch(
        kernel=sddmm_kernel,
        dim=nnz,
        inputs=[
            w_indices_rows,
            w_indices_cols,
            w_grad_values,
            w_grad_out,
            w_dense,
            1 if is_bool_float else 0,
        ],
        device=f"cuda:{device.index}" if device.type == "cuda" else "cpu",
    )
    return grad_values


@sddmm_driver.register_fake
def _(indices, dense, grad, is_bool_float):
    nnz = indices.shape[1]
    return torch.empty(nnz, device=dense.device, dtype=dense.dtype)


class SparseCOOMatMulWarp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, dense, is_bool_float=False, size_m=None):
        indices = indices.contiguous()
        values = values.contiguous()
        dense = dense.contiguous()

        is_vec = dense.dim() == 1
        if is_vec:
            dense = dense.unsqueeze(1)

        K, N = dense.shape
        if size_m is not None:
            M = size_m
        elif indices.numel() > 0:
            M = int(indices[0].max().item()) + 1
        else:
            M = 0

        # Use custom op
        out = torch.ops.btorch_warp.coo_spmm_driver(
            indices, values, dense, M, is_bool_float
        )

        ctx.save_for_backward(indices, values, dense)
        ctx.is_bool_float = is_bool_float
        ctx.is_vec = is_vec
        ctx.M = M

        return out.squeeze(1) if is_vec else out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.contiguous()
        indices, values, dense = ctx.saved_tensors
        is_bool_float = ctx.is_bool_float
        is_vec = ctx.is_vec

        if is_vec:
            grad_out = grad_out.unsqueeze(1)

        K, N = dense.shape

        # 1. grad_dense (grad_B) = A.T @ grad_out
        # Transpose indices: (indices[1], indices[0])
        # We need to construct transposed indices tensor for the op?
        # The op takes indices as (2, nnz).
        # We can just stack them flipped.
        indices_T = torch.stack([indices[1], indices[0]], dim=0).contiguous()

        grad_dense = torch.ops.btorch_warp.coo_spmm_driver(
            indices_T, values, grad_out, K, False
        )

        # 2. grad_values
        grad_values = torch.ops.btorch_warp.sddmm_driver(
            indices, dense, grad_out, is_bool_float
        )

        if is_vec:
            grad_dense = grad_dense.squeeze(1)

        return None, grad_values, grad_dense, None, None


def coo_spmm_warp(indices, values, mat, is_bool_float=False, size_m=None):
    return SparseCOOMatMulWarp.apply(indices, values, mat, is_bool_float, size_m)


def coo_spmv_warp(indices, values, vec, is_bool_float=False, size_m=None):
    return SparseCOOMatMulWarp.apply(indices, values, vec, is_bool_float, size_m)
