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
def flexsn_core_inductor_train_fwd_d0ab698c(primals_1, primals_2, primals_3, primals_4):
    mul_tensor = primals_3 * 0.9
    add_tensor = mul_tensor + primals_1
    add_tensor_1 = primals_4 + 1.0
    sub_tensor = add_tensor - add_tensor_1
    ge_scalar = sub_tensor >= 0
    _to_copy_default = ge_scalar.to(tl.float32)
    sub_tensor_1 = add_tensor - 1.0
    ge_scalar_1 = sub_tensor_1 >= 0
    _to_copy_default_1 = ge_scalar_1.to(tl.float32)
    mul_tensor_1 = primals_4 * 0.8
    add_tensor_2 = mul_tensor_1 + _to_copy_default
    rsub_scalar = 1.0 - _to_copy_default
    mul_tensor_2 = add_tensor * rsub_scalar
    sub_tensor_2 = add_tensor - _to_copy_default_1
    sigmoid_default = tl.sigmoid(primals_2.to(tl.float32)).to(primals_2.dtype)
    mul_tensor_3 = mul_tensor_2 * sigmoid_default
    rsub_scalar_1 = 1.0 - sigmoid_default
    mul_tensor_4 = sub_tensor_2 * rsub_scalar_1
    add_tensor_3 = mul_tensor_3 + mul_tensor_4
    return (
        _to_copy_default,
        _to_copy_default_1,
        add_tensor_3,
        add_tensor_2,
        primals_4,
        add_tensor,
        _to_copy_default,
        _to_copy_default_1,
        sigmoid_default,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [2, 4]
    ],
    key=["T", "dtype"],
    restore_value=[
        "s0_seq_ptr",
        "s1_seq_ptr",
        "v0_seq_ptr",
        "v1_seq_ptr",
        "res0_f_seq_ptr",
        "res1_f_seq_ptr",
        "res2_f_seq_ptr",
    ],
)
@triton.jit
def flexsn_forward_kernel_d0ab698c(
    x0_seq_ptr,
    x1_seq_ptr,
    v0_init_ptr,
    v1_init_ptr,  # inputs (including init states)
    s0_seq_ptr,
    s1_seq_ptr,
    v0_seq_ptr,
    v1_seq_ptr,
    res0_f_seq_ptr,
    res1_f_seq_ptr,
    res2_f_seq_ptr,  # outputs
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

        s0, s1, v0, v1, res0_f, res1_f, _, _, res2_f = (
            flexsn_core_inductor_train_fwd_d0ab698c(x0, x1, v0, v1)
        )

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

        res0_f_ptrs = tl.make_block_ptr(
            res0_f_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(res0_f_ptrs, res0_f, boundary_check=(1,))
        # tl.store(res0_f_ptrs, res0_f, boundary_check=(1,))

        res1_f_ptrs = tl.make_block_ptr(
            res1_f_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(res1_f_ptrs, res1_f, boundary_check=(1,))
        # tl.store(res1_f_ptrs, res1_f, boundary_check=(1,))

        res2_f_ptrs = tl.make_block_ptr(
            res2_f_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(res2_f_ptrs, res2_f, boundary_check=(1,))
        # tl.store(res2_f_ptrs, res2_f, boundary_check=(1,))


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
import triton.language as tl


@triton.jit
def flexsn_core_inductor_train_bwd_78a9d3c0(
    primals_4,
    add,
    _to_copy,
    _to_copy_1,
    sigmoid,
    tangents_1,
    tangents_2,
    tangents_3,
    tangents_4,
):
    sub_tensor = add - _to_copy_1
    mul_tensor = tangents_3 * sub_tensor
    rsub_scalar = 1.0 - sigmoid
    mul_tensor_1 = tangents_3 * rsub_scalar
    neg_default = -mul_tensor
    rsub_scalar_1 = 1.0 - _to_copy
    mul_tensor_2 = add * rsub_scalar_1
    mul_tensor_3 = tangents_3 * mul_tensor_2
    mul_tensor_4 = tangents_3 * sigmoid
    add_tensor = neg_default + mul_tensor_3
    sigmoid_backward_default = add_tensor * sigmoid * (1 - sigmoid)
    neg_default_1 = -mul_tensor_1
    add_tensor_1 = tangents_2 + neg_default_1
    mul_tensor_5 = mul_tensor_4 * add
    mul_tensor_6 = mul_tensor_4 * rsub_scalar_1
    add_tensor_2 = mul_tensor_1 + mul_tensor_6
    neg_default_2 = -mul_tensor_5
    add_tensor_3 = tangents_1 + neg_default_2
    add_tensor_4 = add_tensor_3 + tangents_4
    mul_tensor_7 = tangents_4 * 0.8
    sub_tensor_1 = add - 1.0
    mul_tensor_8 = sub_tensor_1 * 3.141592653589793
    mul_tensor_9 = mul_tensor_8 * mul_tensor_8
    add_tensor_5 = mul_tensor_9 + 1
    reciprocal_default = (1.0 / add_tensor_5).to(add_tensor_5.dtype)
    mul_tensor_10 = reciprocal_default * 1.0
    mul_tensor_11 = mul_tensor_10 * add_tensor_1
    add_tensor_6 = add_tensor_2 + mul_tensor_11
    add_tensor_7 = primals_4 + 1.0
    sub_tensor_2 = add - add_tensor_7
    mul_tensor_12 = sub_tensor_2 * 3.141592653589793
    mul_tensor_13 = mul_tensor_12 * mul_tensor_12
    add_tensor_8 = mul_tensor_13 + 1
    reciprocal_default_1 = (1.0 / add_tensor_8).to(add_tensor_8.dtype)
    mul_tensor_14 = reciprocal_default_1 * 1.0
    mul_tensor_15 = mul_tensor_14 * add_tensor_4
    neg_default_3 = -mul_tensor_15
    add_tensor_9 = add_tensor_6 + mul_tensor_15
    add_tensor_10 = mul_tensor_7 + neg_default_3
    mul_tensor_16 = add_tensor_9 * 0.9
    return add_tensor_9, sigmoid_backward_default, mul_tensor_16, add_tensor_10


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NCL": f * w * 32}, num_warps=w)
        for f in [1, 2]
        for w in [2, 4]
    ],
    key=["T", "dtype"],
    restore_value=[
        "grad_x0_seq_ptr",
        "grad_x1_seq_ptr",
        "grad_v0_init_ptr",
        "grad_v1_init_ptr",
    ],
)
@triton.jit
def flexsn_backward_kernel_78a9d3c0(
    grad_s0_seq_ptr,
    grad_s1_seq_ptr,
    grad_v0_seq_ptr,
    grad_v1_seq_ptr,
    res0_b_seq_ptr,
    res1_b_seq_ptr,
    res2_b_seq_ptr,
    res3_b_seq_ptr,
    res4_b_seq_ptr,  # inputs (including init states)
    grad_x0_seq_ptr,
    grad_x1_seq_ptr,
    grad_v0_init_ptr,
    grad_v1_init_ptr,  # outputs
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    grad_v0_accumulate = tl.zeros([1, BLOCK_NCL], dtype=dtype)
    grad_v1_accumulate = tl.zeros([1, BLOCK_NCL], dtype=dtype)

    for t in tl.static_range(T - 1, -1, -1):
        grad_s0_ptrs = tl.make_block_ptr(
            grad_s0_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_s0 = tl.load(grad_s0_ptrs, boundary_check=(1,), padding_option="zero")

        grad_s1_ptrs = tl.make_block_ptr(
            grad_s1_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_s1 = tl.load(grad_s1_ptrs, boundary_check=(1,), padding_option="zero")

        grad_v0_ptrs = tl.make_block_ptr(
            grad_v0_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_v0 = tl.load(grad_v0_ptrs, boundary_check=(1,), padding_option="zero")

        grad_v1_ptrs = tl.make_block_ptr(
            grad_v1_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        grad_v1 = tl.load(grad_v1_ptrs, boundary_check=(1,), padding_option="zero")

        res0_b_ptrs = tl.make_block_ptr(
            res0_b_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        res0_b = tl.load(res0_b_ptrs, boundary_check=(1,), padding_option="zero")

        res1_b_ptrs = tl.make_block_ptr(
            res1_b_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        res1_b = tl.load(res1_b_ptrs, boundary_check=(1,), padding_option="zero")

        res2_b_ptrs = tl.make_block_ptr(
            res2_b_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        res2_b = tl.load(res2_b_ptrs, boundary_check=(1,), padding_option="zero")

        res3_b_ptrs = tl.make_block_ptr(
            res3_b_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        res3_b = tl.load(res3_b_ptrs, boundary_check=(1,), padding_option="zero")

        res4_b_ptrs = tl.make_block_ptr(
            res4_b_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        res4_b = tl.load(res4_b_ptrs, boundary_check=(1,), padding_option="zero")

        grad_v0_accumulate = grad_v0_accumulate + grad_v0
        grad_v1_accumulate = grad_v1_accumulate + grad_v1
        grad_x0, grad_x1, grad_v0_accumulate, grad_v1_accumulate = (
            flexsn_core_inductor_train_bwd_78a9d3c0(
                res0_b,
                res1_b,
                res2_b,
                res3_b,
                res4_b,
                grad_s0,
                grad_s1,
                grad_v0_accumulate,
                grad_v1_accumulate,
            )
        )

        grad_x0_ptrs = tl.make_block_ptr(
            grad_x0_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(grad_x0_ptrs, grad_x0, boundary_check=(1,))
        # tl.store(grad_x0_ptrs, grad_x0, boundary_check=(1,))

        grad_x1_ptrs = tl.make_block_ptr(
            grad_x1_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(grad_x1_ptrs, grad_x1, boundary_check=(1,))
        # tl.store(grad_x1_ptrs, grad_x1, boundary_check=(1,))

    grad_v0_init_ptrs = tl.make_block_ptr(
        grad_v0_init_ptr,
        shape=(T, NCL),
        strides=(NCL, 1),
        offsets=(t, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    convert_and_store(grad_v0_init_ptrs, grad_v0_accumulate, boundary_check=(1,))
    # tl.store(grad_v0_ptrs, grad_v0, boundary_check=(1,))

    grad_v1_init_ptrs = tl.make_block_ptr(
        grad_v1_init_ptr,
        shape=(T, NCL),
        strides=(NCL, 1),
        offsets=(t, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    convert_and_store(grad_v1_init_ptrs, grad_v1_accumulate, boundary_check=(1,))
    # tl.store(grad_v1_ptrs, grad_v1, boundary_check=(1,))
