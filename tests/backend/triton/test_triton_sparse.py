import pytest
import torch

from btorch.backend.triton.sparse import coo_spmm, coo_spmv


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_spmv_correctness():
    device = "cuda"
    indices = torch.tensor([[0, 1, 1], [0, 1, 2]], dtype=torch.long, device=device)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    # A = [[1, 0, 0], [0, 2, 3]] (2x3)

    vec = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    # Target:
    # row 0: 1*1 = 1
    # row 1: 0*1 + 2*2 + 3*3 = 0 + 4 + 9 = 13.
    # Out: [1, 13]

    out = coo_spmv(indices, values, vec)
    expected = torch.tensor([1.0, 13.0], device=device)

    assert torch.allclose(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_spmv_bool_float():
    device = "cuda"
    indices = torch.tensor([[0, 0], [0, 1]], dtype=torch.long, device=device)
    values = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)
    # A = [[0.5, 0.5]]

    # Vec with "soft" values
    vec = torch.tensor([0.6, 0.4], dtype=torch.float32, device=device)
    # threshold(vec) -> [1.0, 0.0]
    # Expected: 0.5 * 1.0 + 0.5 * 0.0 = 0.5

    out = coo_spmv(indices, values, vec, is_bool_float=True)
    expected = torch.tensor([0.5], device=device)

    assert torch.allclose(out, expected)

    # Check without flag -> 0.5*0.6 + 0.5*0.4 = 0.3 + 0.2 = 0.5
    # Wait, my example yields same result for vanilla float too strict?

    # Better example:
    # val=10.0. vec=0.6 -> thresh=1.0. res=10.0. Linear=6.0.
    values = torch.tensor([10.0, 10.0], dtype=torch.float32, device=device)
    indices = torch.tensor([[0, 0], [0, 1]], dtype=torch.long, device=device)

    out_bool = coo_spmv(indices, values, vec, is_bool_float=True)
    # 0.6->1, 0.4->0. Res: 10*1 + 10*0 = 10.
    assert torch.allclose(out_bool, torch.tensor([10.0], device=device))

    out_float = coo_spmv(indices, values, vec, is_bool_float=False)  # noqa: F841
    # 10*0.6 + 10*0.4 = 6+4=10. Accidental match again!

    # vec=[0.6, 0.6]. Thresh->[1, 1]. Res=20. Linear=12.
    vec2 = torch.tensor([0.6, 0.6], dtype=torch.float32, device=device)
    out_bool2 = coo_spmv(indices, values, vec2, is_bool_float=True)
    assert torch.allclose(out_bool2, torch.tensor([20.0], device=device))

    out_float2 = coo_spmv(indices, values, vec2, is_bool_float=False)
    assert torch.allclose(out_float2, torch.tensor([12.0], device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_spmm_correctness():
    device = "cuda"
    # Identity matrix (2x2)
    indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=device)
    values = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)

    B = torch.randn(2, 4, device=device)
    out = coo_spmm(indices, values, B)

    assert torch.allclose(out, B)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_backward_gradients():
    device = "cuda"
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    # A = [[0, v1], [v2, 0]]
    values = torch.tensor(
        [2.0, 3.0], dtype=torch.float32, device=device, requires_grad=True
    )

    B = torch.tensor(
        [[1.0], [2.0]], dtype=torch.float32, device=device, requires_grad=True
    )

    # Forward:
    # Row 0: 0*1 + 2*2 = 4
    # Row 1: 3*1 + 0*2 = 3
    # Y = [4, 3]

    out = coo_spmv(indices, values, B)
    loss = out.sum()
    loss.backward()

    # dL/dOut = [1, 1]

    # Gradients w.r.t values:
    # dL/dv1 (idx 0 -> row 0, col 1): dL/dY0 * B[1] = 1 * 2 = 2.
    # dL/dv2 (idx 1 -> row 1, col 0): dL/dY1 * B[0] = 1 * 1 = 1.
    assert torch.allclose(values.grad, torch.tensor([2.0, 1.0], device=device))

    # Gradients w.r.t B:
    # dL/dB0: A00*dY0 + A10*dY1 = 0 + 3*1 = 3.
    # dL/dB1: A01*dY0 + A11*dY1 = 2*1 + 0 = 2.
    assert torch.allclose(B.grad, torch.tensor([3.0, 2.0], device=device).unsqueeze(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_backward_gradients_bool_float():
    device = "cuda"
    indices = torch.tensor([[0], [0]], dtype=torch.long, device=device)
    values = torch.tensor([2.0], dtype=torch.float32, device=device, requires_grad=True)

    # B = [0.6]. Thresh -> 1.0.
    B = torch.tensor([0.6], dtype=torch.float32, device=device, requires_grad=True)

    # Y = 2.0 * 1.0 = 2.0.
    out = coo_spmv(indices, values, B, is_bool_float=True)
    loss = out.sum()
    loss.backward()

    # dL/dValues: dL/dY * Threshold(B).
    # 1.0 * 1.0 = 1.0.
    # If we used linear B, it would be 0.6.
    assert torch.allclose(values.grad, torch.tensor([1.0], device=device))

    # dL/dB: A.T * dY.
    # Standard float backprop: 2.0 * 1.0 = 2.0.
    assert torch.allclose(B.grad, torch.tensor([2.0], device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_torch_compile():
    device = "cuda"
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    values = torch.tensor([2.0, 3.0], dtype=torch.float32, device=device)
    B = torch.randn(2, 2, device=device)

    @torch.compile(fullgraph=True)
    def f(i, v, b, size_m):
        return coo_spmm(i, v, b, size_m=size_m)

    out_ref = coo_spmm(indices, values, B, size_m=2)
    out_compiled = f(indices, values, B, size_m=2)

    assert torch.allclose(out_compiled, out_ref)
