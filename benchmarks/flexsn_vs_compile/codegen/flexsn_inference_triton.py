# ruff: noqa
import triton
import triton.language as tl


@triton.jit
def convert_and_store(pointer, value, boundary_check):
    # For block pointers created by tl.make_block_pointer(),
    # implicit type casting is not supported when calling tl.store().
    # This function manually converts dtype and then stores the data.
    value = value.to(pointer.dtype.element_ty.element_ty)
    tl.store(pointer, value, boundary_check=boundary_check)


import triton


@triton.jit
def flexsn_core_inductor_scan_0fdb4046(x_1, y_1, v_1, rho_1):
    mul = v_1 * 0.9
    add = mul + x_1
    add_1 = rho_1 + 1.0
    sub = add - add_1
    ge = sub >= 0
    _to_copy = ge.to(tl.float32)
    sub_1 = add - 1.0
    ge_1 = sub_1 >= 0
    _to_copy_1 = ge_1.to(tl.float32)
    mul_1 = rho_1 * 0.8
    add_2 = mul_1 + _to_copy
    rsub = 1.0 - _to_copy
    mul_2 = add * rsub
    sub_2 = add - _to_copy_1
    sigmoid = tl.sigmoid(y_1.to(tl.float32)).to(y_1.dtype)
    mul_3 = mul_2 * sigmoid
    rsub_1 = 1.0 - sigmoid
    mul_4 = sub_2 * rsub_1
    add_3 = mul_3 + mul_4
    return _to_copy, _to_copy_1, add_3, add_2


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [2, 4]
    ],
    key=["T", "dtype"],
    restore_value=["s0_seq_ptr", "s1_seq_ptr", "v0_seq_ptr", "v1_seq_ptr"],
)
@triton.jit
def flexsn_inference_kernel_0fdb4046(
    x0_seq_ptr,
    x1_seq_ptr,
    v0_init_ptr,
    v1_init_ptr,  # inputs (including init states)
    s0_seq_ptr,
    s1_seq_ptr,
    v0_seq_ptr,
    v1_seq_ptr,  # outputs
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    v0_init_ptrs = tl.make_block_ptr(
        v0_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    v0 = tl.load(v0_init_ptrs, boundary_check=(1,), padding_option="zero")

    v1_init_ptrs = tl.make_block_ptr(
        v1_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    v1 = tl.load(v1_init_ptrs, boundary_check=(1,), padding_option="zero")

    for t in tl.static_range(0, T, 1):
        x0_ptrs = tl.make_block_ptr(
            x0_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        x0 = tl.load(x0_ptrs, boundary_check=(1,), padding_option="zero")

        x1_ptrs = tl.make_block_ptr(
            x1_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        x1 = tl.load(x1_ptrs, boundary_check=(1,), padding_option="zero")

        s0, s1, v0, v1 = flexsn_core_inductor_scan_0fdb4046(x0, x1, v0, v1)

        s0_ptrs = tl.make_block_ptr(
            s0_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(s0_ptrs, s0, boundary_check=(1,))
        # tl.store(s0_ptrs, s0, boundary_check=(1,))

        s1_ptrs = tl.make_block_ptr(
            s1_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(s1_ptrs, s1, boundary_check=(1,))
        # tl.store(s1_ptrs, s1, boundary_check=(1,))

        v0_ptrs = tl.make_block_ptr(
            v0_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(v0_ptrs, v0, boundary_check=(1,))
        # tl.store(v0_ptrs, v0, boundary_check=(1,))

        v1_ptrs = tl.make_block_ptr(
            v1_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(v1_ptrs, v1, boundary_check=(1,))
        # tl.store(v1_ptrs, v1, boundary_check=(1,))
