import triton
import triton.language as tl


@triton.jit
def coo_spmm_kernel(
    A_indices0_ptr,
    A_indices1_ptr,
    A_values_ptr,
    B_ptr,
    C_ptr,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    nnz,
    N,
    BLOCK_NNZ: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_BOOL_FLOAT: tl.constexpr,
):
    """Computes C += A @ B A is sparse COO (indices, values). B is dense (K,
    N). C is dense (M, N).

    Parallelization:
    - pid_0: covers range of NNZ
    - pid_1: covers range of N (dense columns)
    """
    pid_nnz = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program
    nnz_start = pid_nnz * BLOCK_NNZ
    n_start = pid_n * BLOCK_N

    # Range of N handled by this program
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Iterate over the chunk of NNZ assigned to this pid
    # Unrolling might be hard if BLOCK_NNZ is large, but Triton handles loops well.
    # We process BLOCK_NNZ elements.
    # To use vectorization efficiently, we might want to load a vector of
    # rows/cols/vals.

    # Loop over the BLOCK_NNZ
    # Note: Loading A_indices and A_values can be vectorized if we want,
    # but the scatter to C is random (dependent on row indices).
    # So we probably iterate one by one or in small vectors if possible.
    # But different NNZs have different rows, so we can't coalesce the write to C easily
    # unless we sort or shared-memory buffer (which is complex for general COO).
    # Simple approach: atomic add per NNZ.

    for k in range(0, BLOCK_NNZ):
        idx = nnz_start + k
        if idx < nnz:
            # Load row, col, val for this non-zero
            row = tl.load(A_indices0_ptr + idx)
            col = tl.load(A_indices1_ptr + idx)
            val = tl.load(A_values_ptr + idx)

            # Pointers to B and C
            # B: shape (K, N). Address: B_ptr + col * stride_bk + offs_n * stride_bn
            # C: shape (M, N). Address: C_ptr + row * stride_cm + offs_n * stride_bn

            b_ptrs = B_ptr + col * stride_bk + offs_n * stride_bn
            c_ptrs = C_ptr + row * stride_cm + offs_n * stride_bn

            # Load B row
            b_vals = tl.load(b_ptrs, mask=mask_n, other=0.0)

            if IS_BOOL_FLOAT:
                # Optimized: if B > 0.5, result is val, else 0.
                res = tl.where(b_vals > 0.5, val, 0.0)
            else:
                res = val * b_vals

            # Atomic add to C
            tl.atomic_add(c_ptrs, res, mask=mask_n)


@triton.jit
def sddmm_kernel(
    A_indices0_ptr,
    A_indices1_ptr,
    A_grad_values_ptr,
    Grad_ptr,
    B_ptr,  # Grad is dY (M, N), B is (K, N)
    stride_grad_m,
    stride_grad_n,
    stride_bk,
    stride_bn,
    nnz,
    N,
    BLOCK_NNZ: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_BOOL_FLOAT: tl.constexpr,
):
    """Computes Gradient with respect to values of A (Sparse). A_grad_values[k]
    = dot(Grad[row[k], :], B[col[k], :])

    Parallelization:
    - pid_0: covers range of NNZ (one thread/block per NNZ chunk?)

    Actually, for dot product over N, we might want to loop over N inside the kernel
    or have a 2D grid if N is huge.
    If N is small (e.g. 64-128), one block can handle the dot product loop.
    If N is large, we need reduction.

    Simplification: Assume N fits in block or we loop over N with a single PID per NNZ?
    Better: One PID handles a chunk of NNZs. Inner loop over N.
    """
    pid = tl.program_id(0)

    # We process a block of NNZs
    start_nnz = pid * BLOCK_NNZ

    # Range of N to loop over
    # We can iterate over N in chunks of BLOCK_N

    for k in range(0, BLOCK_NNZ):
        idx = start_nnz + k
        if idx < nnz:
            row = tl.load(A_indices0_ptr + idx)
            col = tl.load(A_indices1_ptr + idx)

            acc = 0.0

            # Loop over N dimension
            for n_offset in range(0, N, BLOCK_N):
                # Handle masking for the last block
                cols_n = n_offset + tl.arange(0, BLOCK_N)
                mask = cols_n < N

                grad_ptrs = Grad_ptr + row * stride_grad_m + cols_n * stride_grad_n
                b_ptrs = B_ptr + col * stride_bk + cols_n * stride_bn

                g_val = tl.load(grad_ptrs, mask=mask, other=0.0)
                b_val = tl.load(b_ptrs, mask=mask, other=0.0)

                if IS_BOOL_FLOAT:
                    # Optimized: sum(g_val) where b_val > 0.5
                    term = tl.where(b_val > 0.5, g_val, 0.0)
                    acc += tl.sum(term)
                else:
                    acc += tl.sum(g_val * b_val)

            # Store result
            tl.store(A_grad_values_ptr + idx, acc)
