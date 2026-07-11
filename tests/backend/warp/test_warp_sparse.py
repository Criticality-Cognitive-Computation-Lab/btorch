import torch

from btorch.backend.warp.sparse import coo_spmm_warp, coo_spmv_warp


# Check availability
HAS_CUDA = torch.cuda.is_available()
# Triton requires CUDA
HAS_TRITON = HAS_CUDA
if HAS_TRITON:
    from btorch.backend.triton.sparse import coo_spmm, coo_spmv

DEVICE = "cuda" if HAS_CUDA else "cpu"


def sparse_coo_to_dense(indices, values, size_m, size_n):
    return torch.sparse_coo_tensor(indices, values, (size_m, size_n)).to_dense()


def test_warp_vs_triton_or_dense_spmv():
    device = DEVICE
    indices = torch.tensor([[0, 1, 1], [0, 1, 2]], dtype=torch.long, device=device)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    vec = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)

    out_warp = coo_spmv_warp(indices, values, vec)

    if HAS_TRITON and device == "cuda":
        out_ref = coo_spmv(indices, values, vec)
    else:
        # Fallback to dense verification
        M = int(indices[0].max().item()) + 1
        dense_A = sparse_coo_to_dense(indices, values, M, vec.shape[0])
        out_ref = torch.mv(dense_A, vec)

    assert torch.allclose(out_warp, out_ref), "Warp SpMV output mismatch"


def test_warp_vs_triton_or_dense_spmm():
    device = DEVICE
    indices = torch.tensor([[0, 1, 1], [0, 1, 2]], dtype=torch.long, device=device)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    B = torch.randn(3, 4, device=device)

    out_warp = coo_spmm_warp(indices, values, B)

    if HAS_TRITON and device == "cuda":
        out_ref = coo_spmm(indices, values, B)
    else:
        # Fallback to dense
        M = int(indices[0].max().item()) + 1
        dense_A = sparse_coo_to_dense(indices, values, M, B.shape[0])
        out_ref = torch.matmul(dense_A, B)

    assert torch.allclose(out_warp, out_ref, atol=1e-5), "Warp SpMM output mismatch"


def test_warp_spmv_bool_float():
    device = DEVICE
    indices = torch.tensor([[0, 0], [0, 1]], dtype=torch.long, device=device)
    values = torch.tensor([10.0, 10.0], dtype=torch.float32, device=device)
    vec = torch.tensor([0.6, 0.4], dtype=torch.float32, device=device)

    # Triton result (from test_sparse_ops): 10*1 + 10*0 = 10.
    out_warp = coo_spmv_warp(indices, values, vec, is_bool_float=True)
    expected = torch.tensor([10.0], device=device)

    assert torch.allclose(out_warp, expected), "Warp bool-float logic mismatch"


def test_warp_backward():
    device = DEVICE
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    values = torch.tensor(
        [2.0, 3.0], dtype=torch.float32, device=device, requires_grad=True
    )
    B = torch.tensor(
        [[1.0], [2.0]], dtype=torch.float32, device=device, requires_grad=True
    )

    # Forward with warp
    out = coo_spmv_warp(indices, values, B)
    loss = out.sum()
    loss.backward()

    # Expected gradients (calculated in test_sparse_ops)
    # dL/dv: [2.0, 1.0]
    expected_grad_v = torch.tensor([2.0, 1.0], device=device)

    # dL/dB: [3.0, 2.0]
    expected_grad_B = torch.tensor([3.0, 2.0], device=device).unsqueeze(1)

    assert torch.allclose(
        values.grad, expected_grad_v
    ), f"Values Grad mismatch: {values.grad}"
    assert torch.allclose(B.grad, expected_grad_B), f"B Grad mismatch: {B.grad}"


def test_warp_torch_compile():
    device = DEVICE
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    values = torch.tensor([2.0, 3.0], dtype=torch.float32, device=device)
    B = torch.randn(2, 2, device=device)

    # Compile the warp function
    @torch.compile(fullgraph=True)
    def f(i, v, b, size_m):
        return coo_spmm_warp(i, v, b, size_m=size_m)

    out_ref = coo_spmm_warp(indices, values, B, size_m=2)
    # First run (compile)
    out_compiled = f(indices, values, B, size_m=2)

    assert torch.allclose(out_compiled, out_ref), "Warp torch.compile output mismatch"

    # Second run (cached)
    out_compiled_2 = f(indices, values, B, size_m=2)
    assert torch.allclose(
        out_compiled_2, out_ref
    ), "Warp torch.compile cached output mismatch"


if __name__ == "__main__":
    test_warp_vs_triton_or_dense_spmv()
    test_warp_vs_triton_or_dense_spmm()
    test_warp_spmv_bool_float()
    test_warp_backward()
    test_warp_torch_compile()
    print("All Warp tests passed!")
