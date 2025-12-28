import torch
import triton

from .triton_utils import coo_spmm_kernel, sddmm_kernel


class SparseCOOMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, dense, is_bool_float=False, size_m=None):
        """
        indices: (2, nnz) LongTensor
        values: (nnz,) FloatTensor
        dense: (K, N) FloatTensor (or K if vec)
        is_bool_float: bool, if True, treat dense as 0/1 (threshold 0.5)
        size_m: optional int, output rows, mandatory if using torch.compile.
        """
        # Ensure contiguous
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

        # We need to allocate output.
        out = torch.zeros((M, N), device=dense.device, dtype=dense.dtype)

        nnz = values.numel()
        if nnz == 0:
            return out.squeeze(1) if is_vec else out

        # Kernel Config
        BLOCK_NNZ = 32  # Tuning param
        BLOCK_N = 32

        # Grid
        grid = (triton.cdiv(nnz, BLOCK_NNZ), triton.cdiv(N, BLOCK_N))

        coo_spmm_kernel[grid](
            indices[0],
            indices[1],
            values,
            dense,
            out,
            dense.stride(0),
            dense.stride(1),
            out.stride(0),
            out.stride(1),
            nnz,
            N,  # Passed N
            BLOCK_NNZ=BLOCK_NNZ,
            BLOCK_N=BLOCK_N,
            IS_BOOL_FLOAT=is_bool_float,
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

        # Gradients:
        # 1. grad_dense (grad_B): A.T @ grad_out
        #    This is SpMM with A transposed.
        #    A.T indices are (indices[1], indices[0]).
        #    Values are same.
        #    Shape: (K, M) x (M, N) -> (K, N) matches dense.

        # We can reuse coo_spmm_kernel for grad_dense
        # But we need to be careful: the output size for grad_dense is dense.shape.
        # K, N = dense.shape
        # grad_dense = zeros(K, N)

        # Infer K from dense shape (saved)
        K, N = dense.shape
        grad_dense = torch.zeros_like(dense)
        nnz = values.numel()

        if nnz > 0:
            BLOCK_NNZ = 32
            BLOCK_N = 32
            grid_b = (triton.cdiv(nnz, BLOCK_NNZ), triton.cdiv(N, BLOCK_N))

            # Note: Transpose A means passing indices[1] as row_ptr and indices[0] as
            # col_ptr
            # And target is grad_dense (K, N)

            # Wait, SpMM: C = A @ B.
            # Here: Grad_B = A.T @ Grad_C
            # A_T indices: row=indices[1] (which are cols of A), col=indices[0]
            # Matrix dims: A.T is (K, M). Grad_C is (M, N). Result (K, N).

            # One catch: coo_spmm_kernel assumes atomic add.
            # IS_BOOL_FLOAT applies to B?
            # In backward, B is Grad_C. Grad_C is float.
            # A is values (float).
            # So is_bool_float should be False for grad_B calculation.
            # Because we are backpropagating through linear float mult (or surrogate).
            # Surrogate usually implies "pass through" meaning we treat forward as mul
            # by 1 or similar?
            # User said: "for backward still keep the float grad".
            # Standard backprop through matmul is float.

            coo_spmm_kernel[grid_b](
                indices[1],
                indices[0],
                values,  # Transposed indices
                grad_out,
                grad_dense,
                grad_out.shape[1],  # Inferred stride(0)
                1,  # Inferred stride(1)
                grad_dense.shape[1],
                1,
                nnz,
                N,  # Passed N
                BLOCK_NNZ=BLOCK_NNZ,
                BLOCK_N=BLOCK_N,
                IS_BOOL_FLOAT=False,  # Grad flow is fully float
            )

        # 2. grad_values (grad_A_values):
        #    We need dot products of rows of grad_out and rows of dense
        #    (appropriately indexed).
        #    grad_a_v[k] = dot(grad_out[row[k]], dense[col[k]]) (if B was bool_float,
        #    do we use bool?)
        #    User: "backward still keep the float grad".
        #    Usually: d(w * mask)/dw = mask.
        #    So if dense was treated as bool, we should use that bool value here?
        #    Yes, if Y = W * (X>0.5), dY/dW = (X>0.5).
        #    So we SHOULD use is_bool_float for the dense operand here.

        grad_values = torch.zeros_like(values)
        if nnz > 0:
            BLOCK_NNZ = 32
            BLOCK_N = 32  # For reduction loop
            grid_val = (triton.cdiv(nnz, BLOCK_NNZ),)

            sddmm_kernel[grid_val](
                indices[0],
                indices[1],
                grad_values,
                grad_out,
                dense,
                grad_out.shape[1],  # Inferred stride(0)
                1,  # Inferred stride(1)
                dense.shape[1],
                1,
                nnz,
                N,
                BLOCK_NNZ=BLOCK_NNZ,
                BLOCK_N=BLOCK_N,
                IS_BOOL_FLOAT=is_bool_float,
            )

        if is_vec:
            grad_dense = grad_dense.squeeze(1)

        return None, grad_values, grad_dense, None, None


def coo_spmv(indices, values, vec, is_bool_float=False, size_m=None):
    return SparseCOOMatMul.apply(indices, values, vec, is_bool_float, size_m)


def coo_spmm(indices, values, mat, is_bool_float=False, size_m=None):
    return SparseCOOMatMul.apply(indices, values, mat, is_bool_float, size_m)
