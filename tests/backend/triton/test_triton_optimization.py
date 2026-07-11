import pytest
import torch

from btorch.backend.triton.kernels import coo_spmm_kernel


def get_ptx(kernel, *args, **kwargs):
    """Helper to extract PTX from a Triton kernel for given args.

    Handles nested cache structure of recent Triton versions.
    """
    # Force compilation by calling the kernel (or dry run)
    # Note: This executes the kernel, but we just want to ensure it's in cache.
    # To avoid execution overhead, we could use JIT warmup if available,
    # but execution is robust for tests.
    kernel[*args](**kwargs)

    # Traverse cache
    # cache[device_id][signature] = kernel_object
    # We assume only one device active for simplicity or grab the first.
    if not kernel.cache:
        raise RuntimeError("Kernel cache empty after execution")

    # Recent Triton: cache maps device key to signatures
    # We just grab the most recently added or the only one if we cleared it.
    device_cache = list(kernel.cache.values())[0]

    # Get the compiled kernel
    compiled_kernel = list(device_cache.values())[0]

    if hasattr(compiled_kernel, "asm"):
        return compiled_kernel.asm["ptx"]
    else:
        return compiled_kernel["ptx"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bool_float_optimization_ptx():
    """Verifies that IS_BOOL_FLOAT=True uses specialized instructions (selp)
    instead of floating point multiplication (mul.f32)."""
    device = torch.device("cuda")

    # Dummy inputs
    N = 128
    nnz = 128
    indices0 = torch.zeros(nnz, device=device, dtype=torch.long)
    indices1 = torch.zeros(nnz, device=device, dtype=torch.long)
    values = torch.randn(nnz, device=device, dtype=torch.float32)
    B = torch.randn(128, N, device=device, dtype=torch.float32)
    C = torch.randn(128, N, device=device, dtype=torch.float32)

    BLOCK_NNZ = 32
    BLOCK_N = 32
    grid = (1, 1)

    # 1. Capture PTX for IS_BOOL_FLOAT=False
    coo_spmm_kernel.cache.clear()
    args_false = (
        indices0,
        indices1,
        values,
        B,
        C,
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        nnz,
        N,
    )
    kwargs_false = {"BLOCK_NNZ": BLOCK_NNZ, "BLOCK_N": BLOCK_N, "IS_BOOL_FLOAT": False}

    # Call kernel grid
    coo_spmm_kernel[grid](*args_false, **kwargs_false)

    # Extract PTX
    ptx_false = get_ptx_from_cache(coo_spmm_kernel)

    # 2. Capture PTX for IS_BOOL_FLOAT=True
    coo_spmm_kernel.cache.clear()
    args_true = (
        indices0,
        indices1,
        values,
        B,
        C,
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        nnz,
        N,
    )
    kwargs_true = {"BLOCK_NNZ": BLOCK_NNZ, "BLOCK_N": BLOCK_N, "IS_BOOL_FLOAT": True}

    coo_spmm_kernel[grid](*args_true, **kwargs_true)

    ptx_true = get_ptx_from_cache(coo_spmm_kernel)

    # assertions
    muls_false = ptx_false.count("mul.f32")
    selps_false = ptx_false.count("selp")

    muls_true = ptx_true.count("mul.f32")
    selps_true = ptx_true.count("selp")

    print(f"False: mul={muls_false}, selp={selps_false}")
    print(f"True:  mul={muls_true}, selp={selps_true}")

    # Expect multiplication in standard mode
    assert muls_false > 0, "IS_BOOL_FLOAT=False should use multiplication"

    # Expect NO multiplication in optimized mode (replaced by select)
    assert muls_true == 0, "IS_BOOL_FLOAT=True should NOT use mul.f32"
    assert selps_true > 0, "IS_BOOL_FLOAT=True should use selp"


def get_ptx_from_cache(kernel):
    # Re-implement extraction logic cleanly
    # Assumes cache was just cleared and populated with one entry
    device_cache = list(kernel.cache.values())[0]
    compiled_kernel = list(device_cache.values())[0]

    if hasattr(compiled_kernel, "asm"):
        return compiled_kernel.asm["ptx"]
    else:
        return compiled_kernel["ptx"]
