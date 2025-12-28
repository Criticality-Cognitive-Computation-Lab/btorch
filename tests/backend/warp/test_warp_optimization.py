import glob
import os

import pytest
import torch
import warp as wp


# Initialize warp
wp.init()


# Define a simple kernel to test lowering of bool-float logic
@wp.kernel
def kernel_bool_float_lowering_kernel(
    data: wp.array(dtype=float), out: wp.array(dtype=float), threshold: float
):
    tid = wp.tid()
    val = data[tid]
    # This is the logic we use in sparse.py
    # if val > 0.5: val = 1.0 else: val = 0.0
    if val > threshold:
        out[tid] = 1.0
    else:
        out[tid] = 0.0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Warp PTX check requires CUDA"
)
def test_warp_bool_float_optimization():
    device = "cuda"
    n = 32
    data = torch.randn(n, device=device)
    out = torch.zeros(n, device=device)

    data_wp = wp.from_torch(data)
    out_wp = wp.from_torch(out)

    # Trigger compilation
    wp.launch(
        kernel=kernel_bool_float_lowering_kernel,
        dim=n,
        inputs=[data_wp, out_wp, 0.5],
        device=device,
    )

    # Introspect to find PTX
    module = kernel_bool_float_lowering_kernel.module
    ptx = None

    if hasattr(module, "ptx"):
        ptx = module.ptx
    else:
        # Check verify cache dir
        cache_dir = wp.config.kernel_cache_dir
        ptx_files = glob.glob(os.path.join(cache_dir, "**", "*.ptx"), recursive=True)

        if ptx_files:
            # Sort by mtime
            ptx_files.sort(key=os.path.getmtime, reverse=True)
            latest_ptx = ptx_files[0]
            with open(latest_ptx, "r") as f:
                ptx = f.read()

    assert ptx is not None, "Could not locate generated PTX code"

    # Check for selp (predicated select)
    count_selp = ptx.count("selp")
    # print(ptx)

    # We expect selp instructions for efficient branching
    assert count_selp > 0, (
        "Logic did not lower to 'selp' (predicated select), "
        "usage of expensive branching suspected."
    )
