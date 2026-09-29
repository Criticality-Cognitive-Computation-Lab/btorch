# ruff: noqa
# AOT ID: ['0_inference']

import torch
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.async_compile import AsyncCompile


aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_fanqixuan/6r/c6rlg2b37tkrl4pf4vfjsnoqlntzzdvhucfs2dhaxxuyocnr24az.py
# Topologically Sorted Source Nodes: [mul, getitem, h, add_1, sub, ge, spike, mul_1, atan, truediv, soft, sub_1, s1, sub_4, v1, getitem_1, yy, mul_5, sub_2, ge_1, spike_1, mul_2, atan_1, truediv_1, soft_1, sub_3, s2, v2, sub_6, mul_6, v, mul_7, getitem_2, h_1, mul_3, rho, add_9, sub_7, ge_2, spike_2, mul_8, atan_2, truediv_2, soft_2, sub_8, s1_1, sub_11, v1_1, getitem_3, yy_1, mul_12, sub_9, ge_3, spike_3, mul_9, atan_3, truediv_3, soft_3, sub_10, s2_1, v2_1, sub_13, mul_13, v_1, mul_14, getitem_4, h_2, mul_10, rho_1, add_17, sub_14, ge_4, spike_4, mul_15, atan_4, truediv_4, soft_4, sub_15, s1_2, sub_18, v1_2, getitem_5, yy_2, mul_19, sub_16, ge_5, spike_5, mul_16, atan_5, truediv_5, soft_5, sub_17, s2_2, v2_2, sub_20, mul_20, v_2, mul_21, getitem_6, h_3, mul_17, rho_2, add_25, sub_21, ge_6, spike_6, mul_22, atan_6, truediv_6, soft_6, sub_22, s1_3, sub_23, ge_7, spike_7, mul_23, atan_7, truediv_7, soft_7, sub_24, s2_3, sub_25, v1_3, getitem_7, yy_3, mul_26, v2_3, sub_27, mul_27, v_3, mul_24, rho_3], Original ATen: [aten.mul, aten.select, aten.add, aten.sub, aten.ge, aten._to_copy, aten.atan, aten.div, aten.rsub, aten.sigmoid]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_17 => add_17
#   add_25 => add_25
#   add_9 => add_9
#   atan => atan
#   atan_1 => atan_1
#   atan_2 => atan_2
#   atan_3 => atan_3
#   atan_4 => atan_4
#   atan_5 => atan_5
#   atan_6 => atan_6
#   atan_7 => atan_7
#   ge => ge
#   ge_1 => ge_1
#   ge_2 => ge_2
#   ge_3 => ge_3
#   ge_4 => ge_4
#   ge_5 => ge_5
#   ge_6 => ge_6
#   ge_7 => ge_7
#   getitem => select
#   getitem_1 => select_1
#   getitem_2 => select_2
#   getitem_3 => select_3
#   getitem_4 => select_4
#   getitem_5 => select_5
#   getitem_6 => select_6
#   getitem_7 => select_7
#   h => add
#   h_1 => add_8
#   h_2 => add_16
#   h_3 => add_24
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_10
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_17 => mul_17
#   mul_19 => mul_19
#   mul_2 => mul_2
#   mul_20 => mul_20
#   mul_21 => mul_21
#   mul_22 => mul_22
#   mul_23 => mul_23
#   mul_24 => mul_24
#   mul_26 => mul_26
#   mul_27 => mul_27
#   mul_3 => mul_3
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   rho => add_6
#   rho_1 => add_14
#   rho_2 => add_22
#   rho_3 => add_30
#   s1 => add_3
#   s1_1 => add_11
#   s1_2 => add_19
#   s1_3 => add_27
#   s2 => add_5
#   s2_1 => add_13
#   s2_2 => add_21
#   s2_3 => add_29
#   soft => add_2
#   soft_1 => add_4
#   soft_2 => add_10
#   soft_3 => add_12
#   soft_4 => add_18
#   soft_5 => add_20
#   soft_6 => add_26
#   soft_7 => add_28
#   spike => convert_element_type
#   spike_1 => convert_element_type_1
#   spike_2 => convert_element_type_2
#   spike_3 => convert_element_type_3
#   spike_4 => convert_element_type_4
#   spike_5 => convert_element_type_5
#   spike_6 => convert_element_type_6
#   spike_7 => convert_element_type_7
#   sub => sub
#   sub_1 => sub_1
#   sub_10 => sub_10
#   sub_11 => sub_11
#   sub_13 => sub_13
#   sub_14 => sub_14
#   sub_15 => sub_15
#   sub_16 => sub_16
#   sub_17 => sub_17
#   sub_18 => sub_18
#   sub_2 => sub_2
#   sub_20 => sub_20
#   sub_21 => sub_21
#   sub_22 => sub_22
#   sub_23 => sub_23
#   sub_24 => sub_24
#   sub_25 => sub_25
#   sub_27 => sub_27
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_6 => sub_6
#   sub_7 => sub_7
#   sub_8 => sub_8
#   sub_9 => sub_9
#   truediv => div
#   truediv_1 => div_1
#   truediv_2 => div_2
#   truediv_3 => div_3
#   truediv_4 => div_4
#   truediv_5 => div_5
#   truediv_6 => div_6
#   truediv_7 => div_7
#   v => add_7
#   v1 => mul_4
#   v1_1 => mul_11
#   v1_2 => mul_18
#   v1_3 => mul_25
#   v2 => sub_5
#   v2_1 => sub_12
#   v2_2 => sub_19
#   v2_3 => sub_26
#   v_1 => add_15
#   v_2 => add_23
#   v_3 => add_31
#   yy => sigmoid
#   yy_1 => sigmoid_1
#   yy_2 => sigmoid_2
#   yy_3 => sigmoid_3
# Graph fragment:
#   %arg2_1 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg0_1 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg1_1 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %add_7 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=add_7]
#   %sub_7 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=sub_7]
#   %add_15 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=add_15]
#   %add_14 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=add_14]
#   %add_23 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=add_23]
#   %sub_21 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=sub_21]
#   %mul : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, 0.9), kwargs = {})
#   %select : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 0), kwargs = {})
#   %add : Tensor "f32[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %select), kwargs = {})
#   %add_1 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1.0), kwargs = {})
#   %sub : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %add_1), kwargs = {})
#   %ge : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub, 0.0), kwargs = {})
#   %convert_element_type : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge, torch.float32), kwargs = {})
#   %mul_1 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, 3.141592653589793), kwargs = {})
#   %atan : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_1,), kwargs = {})
#   %div : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan, 3.141592653589793), kwargs = {})
#   %add_2 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, 0.5), kwargs = {})
#   %sub_1 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %add_2), kwargs = {})
#   %add_3 : Tensor "f32[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %sub_1), kwargs = {})
#   %sub_4 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %add_3), kwargs = {})
#   %mul_4 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %sub_4), kwargs = {})
#   %select_1 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg1_1, 0, 0), kwargs = {})
#   %sigmoid : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%select_1,), kwargs = {})
#   %mul_5 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %sigmoid), kwargs = {})
#   %sub_2 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, 1.0), kwargs = {})
#   %ge_1 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_2, 0.0), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge_1, torch.float32), kwargs = {})
#   %mul_2 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, 3.141592653589793), kwargs = {})
#   %atan_1 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_2,), kwargs = {})
#   %div_1 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan_1, 3.141592653589793), kwargs = {})
#   %add_4 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_1, 0.5), kwargs = {})
#   %sub_3 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_1, %add_4), kwargs = {})
#   %add_5 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %sub_3), kwargs = {})
#   %sub_5 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %add_5), kwargs = {})
#   %sub_6 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sigmoid), kwargs = {})
#   %mul_6 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %sub_6), kwargs = {})
#   %add_7 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %mul_7 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.9), kwargs = {})
#   %select_2 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 1), kwargs = {})
#   %add_8 : Tensor "f32[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %select_2), kwargs = {})
#   %mul_3 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, 0.8), kwargs = {})
#   %add_6 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %add_3), kwargs = {})
#   %add_9 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, 1.0), kwargs = {})
#   %sub_7 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %add_9), kwargs = {})
#   %ge_2 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_7, 0.0), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge_2, torch.float32), kwargs = {})
#   %mul_8 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, 3.141592653589793), kwargs = {})
#   %atan_2 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_8,), kwargs = {})
#   %div_2 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan_2, 3.141592653589793), kwargs = {})
#   %add_10 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_2, 0.5), kwargs = {})
#   %sub_8 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %add_10), kwargs = {})
#   %add_11 : Tensor "f32[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %sub_8), kwargs = {})
#   %sub_11 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %add_11), kwargs = {})
#   %mul_11 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %sub_11), kwargs = {})
#   %select_3 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg1_1, 0, 1), kwargs = {})
#   %sigmoid_1 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%select_3,), kwargs = {})
#   %mul_12 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %sigmoid_1), kwargs = {})
#   %sub_9 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, 1.0), kwargs = {})
#   %ge_3 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_9, 0.0), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge_3, torch.float32), kwargs = {})
#   %mul_9 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, 3.141592653589793), kwargs = {})
#   %atan_3 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_9,), kwargs = {})
#   %div_3 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan_3, 3.141592653589793), kwargs = {})
#   %add_12 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, 0.5), kwargs = {})
#   %sub_10 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_3, %add_12), kwargs = {})
#   %add_13 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %sub_10), kwargs = {})
#   %sub_12 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %add_13), kwargs = {})
#   %sub_13 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sigmoid_1), kwargs = {})
#   %mul_13 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %sub_13), kwargs = {})
#   %add_15 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %mul_13), kwargs = {})
#   %mul_14 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, 0.9), kwargs = {})
#   %select_4 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 2), kwargs = {})
#   %add_16 : Tensor "f32[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %select_4), kwargs = {})
#   %mul_10 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 0.8), kwargs = {})
#   %add_14 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %add_11), kwargs = {})
#   %add_17 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_14, 1.0), kwargs = {})
#   %sub_14 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_16, %add_17), kwargs = {})
#   %ge_4 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_14, 0.0), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge_4, torch.float32), kwargs = {})
#   %mul_15 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, 3.141592653589793), kwargs = {})
#   %atan_4 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_15,), kwargs = {})
#   %div_4 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan_4, 3.141592653589793), kwargs = {})
#   %add_18 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, 0.5), kwargs = {})
#   %sub_15 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_4, %add_18), kwargs = {})
#   %add_19 : Tensor "f32[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %sub_15), kwargs = {})
#   %sub_18 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %add_19), kwargs = {})
#   %mul_18 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, %sub_18), kwargs = {})
#   %select_5 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg1_1, 0, 2), kwargs = {})
#   %sigmoid_2 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%select_5,), kwargs = {})
#   %mul_19 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %sigmoid_2), kwargs = {})
#   %sub_16 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_16, 1.0), kwargs = {})
#   %ge_5 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_16, 0.0), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge_5, torch.float32), kwargs = {})
#   %mul_16 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, 3.141592653589793), kwargs = {})
#   %atan_5 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_16,), kwargs = {})
#   %div_5 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan_5, 3.141592653589793), kwargs = {})
#   %add_20 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_5, 0.5), kwargs = {})
#   %sub_17 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_5, %add_20), kwargs = {})
#   %add_21 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %sub_17), kwargs = {})
#   %sub_19 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_16, %add_21), kwargs = {})
#   %sub_20 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sigmoid_2), kwargs = {})
#   %mul_20 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %sub_20), kwargs = {})
#   %add_23 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %mul_20), kwargs = {})
#   %mul_21 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, 0.9), kwargs = {})
#   %select_6 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 3), kwargs = {})
#   %add_24 : Tensor "f32[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %select_6), kwargs = {})
#   %mul_17 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, 0.8), kwargs = {})
#   %add_22 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %add_19), kwargs = {})
#   %add_25 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, 1.0), kwargs = {})
#   %sub_21 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %add_25), kwargs = {})
#   %ge_6 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_21, 0.0), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge_6, torch.float32), kwargs = {})
#   %mul_22 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, 3.141592653589793), kwargs = {})
#   %atan_6 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_22,), kwargs = {})
#   %div_6 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan_6, 3.141592653589793), kwargs = {})
#   %add_26 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_6, 0.5), kwargs = {})
#   %sub_22 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_6, %add_26), kwargs = {})
#   %add_27 : Tensor "f32[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %sub_22), kwargs = {})
#   %sub_23 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, 1.0), kwargs = {})
#   %ge_7 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_23, 0.0), kwargs = {})
#   %convert_element_type_7 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge_7, torch.float32), kwargs = {})
#   %mul_23 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, 3.141592653589793), kwargs = {})
#   %atan_7 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_23,), kwargs = {})
#   %div_7 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan_7, 3.141592653589793), kwargs = {})
#   %add_28 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, 0.5), kwargs = {})
#   %sub_24 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_7, %add_28), kwargs = {})
#   %add_29 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_28, %sub_24), kwargs = {})
#   %sub_25 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %add_27), kwargs = {})
#   %mul_25 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %sub_25), kwargs = {})
#   %select_7 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg1_1, 0, 3), kwargs = {})
#   %sigmoid_3 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%select_7,), kwargs = {})
#   %mul_26 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %sigmoid_3), kwargs = {})
#   %sub_26 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %add_29), kwargs = {})
#   %sub_27 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sigmoid_3), kwargs = {})
#   %mul_27 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %sub_27), kwargs = {})
#   %add_31 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %mul_27), kwargs = {})
#   %mul_24 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 0.8), kwargs = {})
#   %add_30 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %add_27), kwargs = {})
#   return %add_7,%sub_7,%add_14,%add_15,%add_23,%sub_21,%add_30,%add_31
triton_poi_fused__to_copy_add_atan_div_ge_mul_rsub_select_sigmoid_sub_0 = (
    async_compile.triton(
        "triton_poi_fused__to_copy_add_atan_div_ge_mul_rsub_select_sigmoid_sub_0",
        """
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_atan_div_ge_mul_rsub_select_sigmoid_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 8, 'num_reduction': 0, 'backend_hash': '2FF95D04095FCCDF8D543A741630862F9E2E47FF012BC5A02BD474C2C7B615C2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3407872}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_atan_div_ge_mul_rsub_select_sigmoid_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp23 = tl.load(in_ptr3 + (x0), None)
    tmp40 = tl.load(in_ptr1 + (32768 + x0), None)
    tmp59 = tl.load(in_ptr3 + (32768 + x0), None)
    tmp76 = tl.load(in_ptr1 + (65536 + x0), None)
    tmp90 = tl.load(in_ptr3 + (65536 + x0), None)
    tmp107 = tl.load(in_ptr1 + (98304 + x0), None)
    tmp125 = tl.load(in_ptr3 + (98304 + x0), None)
    tmp1 = tl.full([1], 0.9, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 - tmp7
    tmp9 = tl.full([1], 3.141592653589793, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.atan(tmp10)
    tmp12 = tl.full([1], 0.3183098861837907, tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full([1], 0.5, tl.float32)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tmp8 >= tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 - tmp15
    tmp20 = tmp15 + tmp19
    tmp21 = tmp6 - tmp20
    tmp22 = tmp4 * tmp21
    tmp24 = tl.sigmoid(tmp23)
    tmp25 = tmp22 * tmp24
    tmp26 = tmp4 - tmp6
    tmp27 = tmp26 * tmp9
    tmp28 = libdevice.atan(tmp27)
    tmp29 = tmp28 * tmp12
    tmp30 = tmp29 + tmp14
    tmp31 = tmp26 >= tmp16
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32 - tmp30
    tmp34 = tmp30 + tmp33
    tmp35 = tmp4 - tmp34
    tmp36 = tmp6 - tmp24
    tmp37 = tmp35 * tmp36
    tmp38 = tmp25 + tmp37
    tmp39 = tmp38 * tmp1
    tmp41 = tmp39 + tmp40
    tmp42 = tl.full([1], 0.8, tl.float32)
    tmp43 = tmp5 * tmp42
    tmp44 = tmp43 + tmp20
    tmp45 = tmp44 + tmp6
    tmp46 = tmp41 - tmp45
    tmp47 = tmp44 * tmp42
    tmp48 = tmp46 * tmp9
    tmp49 = libdevice.atan(tmp48)
    tmp50 = tmp49 * tmp12
    tmp51 = tmp50 + tmp14
    tmp52 = tmp46 >= tmp16
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp53 - tmp51
    tmp55 = tmp51 + tmp54
    tmp56 = tmp47 + tmp55
    tmp57 = tmp6 - tmp55
    tmp58 = tmp41 * tmp57
    tmp60 = tl.sigmoid(tmp59)
    tmp61 = tmp58 * tmp60
    tmp62 = tmp41 - tmp6
    tmp63 = tmp62 * tmp9
    tmp64 = libdevice.atan(tmp63)
    tmp65 = tmp64 * tmp12
    tmp66 = tmp65 + tmp14
    tmp67 = tmp62 >= tmp16
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tmp68 - tmp66
    tmp70 = tmp66 + tmp69
    tmp71 = tmp41 - tmp70
    tmp72 = tmp6 - tmp60
    tmp73 = tmp71 * tmp72
    tmp74 = tmp61 + tmp73
    tmp75 = tmp74 * tmp1
    tmp77 = tmp75 + tmp76
    tmp78 = tmp56 + tmp6
    tmp79 = tmp77 - tmp78
    tmp80 = tmp79 * tmp9
    tmp81 = libdevice.atan(tmp80)
    tmp82 = tmp81 * tmp12
    tmp83 = tmp82 + tmp14
    tmp84 = tmp79 >= tmp16
    tmp85 = tmp84.to(tl.float32)
    tmp86 = tmp85 - tmp83
    tmp87 = tmp83 + tmp86
    tmp88 = tmp6 - tmp87
    tmp89 = tmp77 * tmp88
    tmp91 = tl.sigmoid(tmp90)
    tmp92 = tmp89 * tmp91
    tmp93 = tmp77 - tmp6
    tmp94 = tmp93 * tmp9
    tmp95 = libdevice.atan(tmp94)
    tmp96 = tmp95 * tmp12
    tmp97 = tmp96 + tmp14
    tmp98 = tmp93 >= tmp16
    tmp99 = tmp98.to(tl.float32)
    tmp100 = tmp99 - tmp97
    tmp101 = tmp97 + tmp100
    tmp102 = tmp77 - tmp101
    tmp103 = tmp6 - tmp91
    tmp104 = tmp102 * tmp103
    tmp105 = tmp92 + tmp104
    tmp106 = tmp105 * tmp1
    tmp108 = tmp106 + tmp107
    tmp109 = tmp56 * tmp42
    tmp110 = tmp109 + tmp87
    tmp111 = tmp110 + tmp6
    tmp112 = tmp108 - tmp111
    tmp113 = tmp110 * tmp42
    tmp114 = tmp112 * tmp9
    tmp115 = libdevice.atan(tmp114)
    tmp116 = tmp115 * tmp12
    tmp117 = tmp116 + tmp14
    tmp118 = tmp112 >= tmp16
    tmp119 = tmp118.to(tl.float32)
    tmp120 = tmp119 - tmp117
    tmp121 = tmp117 + tmp120
    tmp122 = tmp113 + tmp121
    tmp123 = tmp6 - tmp121
    tmp124 = tmp108 * tmp123
    tmp126 = tl.sigmoid(tmp125)
    tmp127 = tmp124 * tmp126
    tmp128 = tmp108 - tmp6
    tmp129 = tmp128 * tmp9
    tmp130 = libdevice.atan(tmp129)
    tmp131 = tmp130 * tmp12
    tmp132 = tmp131 + tmp14
    tmp133 = tmp128 >= tmp16
    tmp134 = tmp133.to(tl.float32)
    tmp135 = tmp134 - tmp132
    tmp136 = tmp132 + tmp135
    tmp137 = tmp108 - tmp136
    tmp138 = tmp6 - tmp126
    tmp139 = tmp137 * tmp138
    tmp140 = tmp127 + tmp139
    tl.store(out_ptr0 + (x0), tmp38, None)
    tl.store(out_ptr1 + (x0), tmp46, None)
    tl.store(out_ptr2 + (x0), tmp56, None)
    tl.store(out_ptr3 + (x0), tmp74, None)
    tl.store(out_ptr4 + (x0), tmp105, None)
    tl.store(out_ptr5 + (x0), tmp112, None)
    tl.store(out_ptr6 + (x0), tmp122, None)
    tl.store(out_ptr7 + (x0), tmp140, None)
""",
        device_str="cuda",
    )
)


# kernel path: /tmp/torchinductor_fanqixuan/zv/czvwasnip3y3k2sw2hazxapi352mksdrovlrpsbjrvf6ppcf53eq.py
# Topologically Sorted Source Nodes: [mul, getitem, h, add_1, sub, ge, spike, mul_1, atan, truediv, soft, sub_1, s1, sub_2, ge_1, spike_1, mul_2, atan_1, truediv_1, soft_1, sub_3, s2, mul_7, getitem_2, h_1, ge_2, spike_2, mul_8, atan_2, truediv_2, soft_2, sub_8, s1_1, sub_9, ge_3, spike_3, mul_9, atan_3, truediv_3, soft_3, sub_10, s2_1, mul_14, getitem_4, h_2, add_17, sub_14, ge_4, spike_4, mul_15, atan_4, truediv_4, soft_4, sub_15, s1_2, sub_16, ge_5, spike_5, mul_16, atan_5, truediv_5, soft_5, sub_17, s2_2, mul_21, getitem_6, h_3, ge_6, spike_6, mul_22, atan_6, truediv_6, soft_6, sub_22, s1_3, stack, sub_23, ge_7, spike_7, mul_23, atan_7, truediv_7, soft_7, sub_24, s2_3, stack_1], Original ATen: [aten.mul, aten.select, aten.add, aten.sub, aten.ge, aten._to_copy, aten.atan, aten.div, aten.stack]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_17 => add_17
#   atan => atan
#   atan_1 => atan_1
#   atan_2 => atan_2
#   atan_3 => atan_3
#   atan_4 => atan_4
#   atan_5 => atan_5
#   atan_6 => atan_6
#   atan_7 => atan_7
#   ge => ge
#   ge_1 => ge_1
#   ge_2 => ge_2
#   ge_3 => ge_3
#   ge_4 => ge_4
#   ge_5 => ge_5
#   ge_6 => ge_6
#   ge_7 => ge_7
#   getitem => select
#   getitem_2 => select_2
#   getitem_4 => select_4
#   getitem_6 => select_6
#   h => add
#   h_1 => add_8
#   h_2 => add_16
#   h_3 => add_24
#   mul => mul
#   mul_1 => mul_1
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_2 => mul_2
#   mul_21 => mul_21
#   mul_22 => mul_22
#   mul_23 => mul_23
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   s1 => add_3
#   s1_1 => add_11
#   s1_2 => add_19
#   s1_3 => add_27
#   s2 => add_5
#   s2_1 => add_13
#   s2_2 => add_21
#   s2_3 => add_29
#   soft => add_2
#   soft_1 => add_4
#   soft_2 => add_10
#   soft_3 => add_12
#   soft_4 => add_18
#   soft_5 => add_20
#   soft_6 => add_26
#   soft_7 => add_28
#   spike => convert_element_type
#   spike_1 => convert_element_type_1
#   spike_2 => convert_element_type_2
#   spike_3 => convert_element_type_3
#   spike_4 => convert_element_type_4
#   spike_5 => convert_element_type_5
#   spike_6 => convert_element_type_6
#   spike_7 => convert_element_type_7
#   stack => cat
#   stack_1 => cat_1
#   sub => sub
#   sub_1 => sub_1
#   sub_10 => sub_10
#   sub_14 => sub_14
#   sub_15 => sub_15
#   sub_16 => sub_16
#   sub_17 => sub_17
#   sub_2 => sub_2
#   sub_22 => sub_22
#   sub_23 => sub_23
#   sub_24 => sub_24
#   sub_3 => sub_3
#   sub_8 => sub_8
#   sub_9 => sub_9
#   truediv => div
#   truediv_1 => div_1
#   truediv_2 => div_2
#   truediv_3 => div_3
#   truediv_4 => div_4
#   truediv_5 => div_5
#   truediv_6 => div_6
#   truediv_7 => div_7
# Graph fragment:
#   %arg2_1 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg0_1 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %sub_7 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=sub_7]
#   %add_15 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=add_15]
#   %add_14 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=add_14]
#   %sub_21 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=sub_21]
#   %add_7 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=add_7]
#   %add_23 : Tensor "f32[32768][1]cuda:0" = PlaceHolder[target=add_23]
#   %mul : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, 0.9), kwargs = {})
#   %select : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 0), kwargs = {})
#   %add : Tensor "f32[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %select), kwargs = {})
#   %add_1 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1.0), kwargs = {})
#   %sub : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %add_1), kwargs = {})
#   %ge : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub, 0.0), kwargs = {})
#   %convert_element_type : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge, torch.float32), kwargs = {})
#   %mul_1 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, 3.141592653589793), kwargs = {})
#   %atan : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_1,), kwargs = {})
#   %div : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan, 3.141592653589793), kwargs = {})
#   %add_2 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, 0.5), kwargs = {})
#   %sub_1 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %add_2), kwargs = {})
#   %add_3 : Tensor "f32[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %sub_1), kwargs = {})
#   %sub_2 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, 1.0), kwargs = {})
#   %ge_1 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_2, 0.0), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge_1, torch.float32), kwargs = {})
#   %mul_2 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, 3.141592653589793), kwargs = {})
#   %atan_1 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_2,), kwargs = {})
#   %div_1 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan_1, 3.141592653589793), kwargs = {})
#   %add_4 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_1, 0.5), kwargs = {})
#   %sub_3 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_1, %add_4), kwargs = {})
#   %add_5 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %sub_3), kwargs = {})
#   %mul_7 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.9), kwargs = {})
#   %select_2 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 1), kwargs = {})
#   %add_8 : Tensor "f32[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %select_2), kwargs = {})
#   %ge_2 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_7, 0.0), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge_2, torch.float32), kwargs = {})
#   %mul_8 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, 3.141592653589793), kwargs = {})
#   %atan_2 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_8,), kwargs = {})
#   %div_2 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan_2, 3.141592653589793), kwargs = {})
#   %add_10 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_2, 0.5), kwargs = {})
#   %sub_8 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %add_10), kwargs = {})
#   %add_11 : Tensor "f32[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %sub_8), kwargs = {})
#   %sub_9 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, 1.0), kwargs = {})
#   %ge_3 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_9, 0.0), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge_3, torch.float32), kwargs = {})
#   %mul_9 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, 3.141592653589793), kwargs = {})
#   %atan_3 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_9,), kwargs = {})
#   %div_3 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan_3, 3.141592653589793), kwargs = {})
#   %add_12 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, 0.5), kwargs = {})
#   %sub_10 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_3, %add_12), kwargs = {})
#   %add_13 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %sub_10), kwargs = {})
#   %mul_14 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, 0.9), kwargs = {})
#   %select_4 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 2), kwargs = {})
#   %add_16 : Tensor "f32[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %select_4), kwargs = {})
#   %add_17 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_14, 1.0), kwargs = {})
#   %sub_14 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_16, %add_17), kwargs = {})
#   %ge_4 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_14, 0.0), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge_4, torch.float32), kwargs = {})
#   %mul_15 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, 3.141592653589793), kwargs = {})
#   %atan_4 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_15,), kwargs = {})
#   %div_4 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan_4, 3.141592653589793), kwargs = {})
#   %add_18 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, 0.5), kwargs = {})
#   %sub_15 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_4, %add_18), kwargs = {})
#   %add_19 : Tensor "f32[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %sub_15), kwargs = {})
#   %sub_16 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_16, 1.0), kwargs = {})
#   %ge_5 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_16, 0.0), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge_5, torch.float32), kwargs = {})
#   %mul_16 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, 3.141592653589793), kwargs = {})
#   %atan_5 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_16,), kwargs = {})
#   %div_5 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan_5, 3.141592653589793), kwargs = {})
#   %add_20 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_5, 0.5), kwargs = {})
#   %sub_17 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_5, %add_20), kwargs = {})
#   %add_21 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %sub_17), kwargs = {})
#   %mul_21 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, 0.9), kwargs = {})
#   %select_6 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 3), kwargs = {})
#   %add_24 : Tensor "f32[32768][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %select_6), kwargs = {})
#   %ge_6 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_21, 0.0), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge_6, torch.float32), kwargs = {})
#   %mul_22 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, 3.141592653589793), kwargs = {})
#   %atan_6 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_22,), kwargs = {})
#   %div_6 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan_6, 3.141592653589793), kwargs = {})
#   %add_26 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_6, 0.5), kwargs = {})
#   %sub_22 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_6, %add_26), kwargs = {})
#   %add_27 : Tensor "f32[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %sub_22), kwargs = {})
#   %cat : Tensor "f32[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_3, %add_11, %add_19, %add_27],), kwargs = {})
#   %sub_23 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, 1.0), kwargs = {})
#   %ge_7 : Tensor "b8[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_23, 0.0), kwargs = {})
#   %convert_element_type_7 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%ge_7, torch.float32), kwargs = {})
#   %mul_23 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, 3.141592653589793), kwargs = {})
#   %atan_7 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_23,), kwargs = {})
#   %div_7 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%atan_7, 3.141592653589793), kwargs = {})
#   %add_28 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, 0.5), kwargs = {})
#   %sub_24 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_7, %add_28), kwargs = {})
#   %add_29 : Tensor "f32[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_28, %sub_24), kwargs = {})
#   %cat_1 : Tensor "f32[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_5, %add_13, %add_21, %add_29],), kwargs = {})
#   return %cat,%cat_1
triton_poi_fused__to_copy_add_atan_div_ge_mul_select_stack_sub_1 = async_compile.triton(
    "triton_poi_fused__to_copy_add_atan_div_ge_mul_select_stack_sub_1",
    """
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_atan_div_ge_mul_select_stack_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 12, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '2FF95D04095FCCDF8D543A741630862F9E2E47FF012BC5A02BD474C2C7B615C2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3407872}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_atan_div_ge_mul_select_stack_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full([1], 0.9, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full([1], 1.0, tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 - tmp12
    tmp14 = tl.full([1], 3.141592653589793, tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = libdevice.atan(tmp15)
    tmp17 = tl.full([1], 0.3183098861837907, tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full([1], 0.5, tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0.0, tl.float32)
    tmp22 = tmp13 >= tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23 - tmp20
    tmp25 = tmp20 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp4, tmp25, tmp26)
    tmp28 = tmp0 >= tmp3
    tmp29 = tl.full([1], 65536, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tmp28 & tmp30
    tmp32 = tl.load(in_ptr3 + ((-32768) + x0), tmp31, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.full([1], 3.141592653589793, tl.float32)
    tmp34 = tmp32 * tmp33
    tmp35 = libdevice.atan(tmp34)
    tmp36 = tl.full([1], 0.3183098861837907, tl.float32)
    tmp37 = tmp35 * tmp36
    tmp38 = tl.full([1], 0.5, tl.float32)
    tmp39 = tmp37 + tmp38
    tmp40 = tl.full([1], 0.0, tl.float32)
    tmp41 = tmp32 >= tmp40
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42 - tmp39
    tmp44 = tmp39 + tmp43
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp31, tmp44, tmp45)
    tmp47 = tmp0 >= tmp29
    tmp48 = tl.full([1], 98304, tl.int64)
    tmp49 = tmp0 < tmp48
    tmp50 = tmp47 & tmp49
    tmp51 = tl.load(in_ptr4 + ((-65536) + x0), tmp50, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.full([1], 0.9, tl.float32)
    tmp53 = tmp51 * tmp52
    tmp54 = tl.load(in_ptr1 + (65536 + ((-65536) + x0)), tmp50, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp53 + tmp54
    tmp56 = tl.load(in_ptr5 + ((-65536) + x0), tmp50, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full([1], 1.0, tl.float32)
    tmp58 = tmp56 + tmp57
    tmp59 = tmp55 - tmp58
    tmp60 = tl.full([1], 3.141592653589793, tl.float32)
    tmp61 = tmp59 * tmp60
    tmp62 = libdevice.atan(tmp61)
    tmp63 = tl.full([1], 0.3183098861837907, tl.float32)
    tmp64 = tmp62 * tmp63
    tmp65 = tl.full([1], 0.5, tl.float32)
    tmp66 = tmp64 + tmp65
    tmp67 = tl.full([1], 0.0, tl.float32)
    tmp68 = tmp59 >= tmp67
    tmp69 = tmp68.to(tl.float32)
    tmp70 = tmp69 - tmp66
    tmp71 = tmp66 + tmp70
    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
    tmp73 = tl.where(tmp50, tmp71, tmp72)
    tmp74 = tmp0 >= tmp48
    tmp75 = tl.full([1], 131072, tl.int64)
    tmp76 = tmp0 < tmp75
    tmp77 = tl.load(in_ptr6 + ((-98304) + x0), tmp74, eviction_policy='evict_last', other=0.0)
    tmp78 = tl.full([1], 3.141592653589793, tl.float32)
    tmp79 = tmp77 * tmp78
    tmp80 = libdevice.atan(tmp79)
    tmp81 = tl.full([1], 0.3183098861837907, tl.float32)
    tmp82 = tmp80 * tmp81
    tmp83 = tl.full([1], 0.5, tl.float32)
    tmp84 = tmp82 + tmp83
    tmp85 = tl.full([1], 0.0, tl.float32)
    tmp86 = tmp77 >= tmp85
    tmp87 = tmp86.to(tl.float32)
    tmp88 = tmp87 - tmp84
    tmp89 = tmp84 + tmp88
    tmp90 = tl.full(tmp89.shape, 0.0, tmp89.dtype)
    tmp91 = tl.where(tmp74, tmp89, tmp90)
    tmp92 = tl.where(tmp50, tmp73, tmp91)
    tmp93 = tl.where(tmp31, tmp46, tmp92)
    tmp94 = tl.where(tmp4, tmp27, tmp93)
    tmp95 = tmp9 - tmp11
    tmp96 = tmp95 * tmp14
    tmp97 = libdevice.atan(tmp96)
    tmp98 = tmp97 * tmp17
    tmp99 = tmp98 + tmp19
    tmp100 = tmp95 >= tmp21
    tmp101 = tmp100.to(tl.float32)
    tmp102 = tmp101 - tmp99
    tmp103 = tmp99 + tmp102
    tmp104 = tl.full(tmp103.shape, 0.0, tmp103.dtype)
    tmp105 = tl.where(tmp4, tmp103, tmp104)
    tmp106 = tl.load(in_ptr7 + ((-32768) + x0), tmp31, eviction_policy='evict_last', other=0.0)
    tmp107 = tl.full([1], 0.9, tl.float32)
    tmp108 = tmp106 * tmp107
    tmp109 = tl.load(in_ptr1 + (32768 + ((-32768) + x0)), tmp31, eviction_policy='evict_last', other=0.0)
    tmp110 = tmp108 + tmp109
    tmp111 = tl.full([1], 1.0, tl.float32)
    tmp112 = tmp110 - tmp111
    tmp113 = tmp112 * tmp33
    tmp114 = libdevice.atan(tmp113)
    tmp115 = tmp114 * tmp36
    tmp116 = tmp115 + tmp38
    tmp117 = tmp112 >= tmp40
    tmp118 = tmp117.to(tl.float32)
    tmp119 = tmp118 - tmp116
    tmp120 = tmp116 + tmp119
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp31, tmp120, tmp121)
    tmp123 = tmp55 - tmp57
    tmp124 = tmp123 * tmp60
    tmp125 = libdevice.atan(tmp124)
    tmp126 = tmp125 * tmp63
    tmp127 = tmp126 + tmp65
    tmp128 = tmp123 >= tmp67
    tmp129 = tmp128.to(tl.float32)
    tmp130 = tmp129 - tmp127
    tmp131 = tmp127 + tmp130
    tmp132 = tl.full(tmp131.shape, 0.0, tmp131.dtype)
    tmp133 = tl.where(tmp50, tmp131, tmp132)
    tmp134 = tl.load(in_ptr8 + ((-98304) + x0), tmp74, eviction_policy='evict_last', other=0.0)
    tmp135 = tl.full([1], 0.9, tl.float32)
    tmp136 = tmp134 * tmp135
    tmp137 = tl.load(in_ptr1 + (98304 + ((-98304) + x0)), tmp74, eviction_policy='evict_last', other=0.0)
    tmp138 = tmp136 + tmp137
    tmp139 = tl.full([1], 1.0, tl.float32)
    tmp140 = tmp138 - tmp139
    tmp141 = tmp140 * tmp78
    tmp142 = libdevice.atan(tmp141)
    tmp143 = tmp142 * tmp81
    tmp144 = tmp143 + tmp83
    tmp145 = tmp140 >= tmp85
    tmp146 = tmp145.to(tl.float32)
    tmp147 = tmp146 - tmp144
    tmp148 = tmp144 + tmp147
    tmp149 = tl.full(tmp148.shape, 0.0, tmp148.dtype)
    tmp150 = tl.where(tmp74, tmp148, tmp149)
    tmp151 = tl.where(tmp50, tmp133, tmp150)
    tmp152 = tl.where(tmp31, tmp122, tmp151)
    tmp153 = tl.where(tmp4, tmp105, tmp152)
    tl.store(out_ptr0 + (x0), tmp94, None)
    tl.store(out_ptr1 + (x0), tmp153, None)
""",
    device_str="cuda",
)


async_compile.wait(globals())
del async_compile


class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        assert_size_stride(arg0_1, (4, 32768), (32768, 1))
        assert_size_stride(arg1_1, (4, 32768), (32768, 1))
        assert_size_stride(arg2_1, (32768,), (1,))
        assert_size_stride(arg3_1, (32768,), (1,))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((32768,), (1,), torch.float32)
            buf1 = empty_strided_cuda((32768,), (1,), torch.float32)
            buf3 = empty_strided_cuda((32768,), (1,), torch.float32)
            buf2 = empty_strided_cuda((32768,), (1,), torch.float32)
            buf4 = empty_strided_cuda((32768,), (1,), torch.float32)
            buf5 = empty_strided_cuda((32768,), (1,), torch.float32)
            buf9 = empty_strided_cuda((32768,), (1,), torch.float32)
            buf8 = empty_strided_cuda((32768,), (1,), torch.float32)
            # Topologically Sorted Source Nodes: [mul, getitem, h, add_1, sub, ge, spike, mul_1, atan, truediv, soft, sub_1, s1, sub_4, v1, getitem_1, yy, mul_5, sub_2, ge_1, spike_1, mul_2, atan_1, truediv_1, soft_1, sub_3, s2, v2, sub_6, mul_6, v, mul_7, getitem_2, h_1, mul_3, rho, add_9, sub_7, ge_2, spike_2, mul_8, atan_2, truediv_2, soft_2, sub_8, s1_1, sub_11, v1_1, getitem_3, yy_1, mul_12, sub_9, ge_3, spike_3, mul_9, atan_3, truediv_3, soft_3, sub_10, s2_1, v2_1, sub_13, mul_13, v_1, mul_14, getitem_4, h_2, mul_10, rho_1, add_17, sub_14, ge_4, spike_4, mul_15, atan_4, truediv_4, soft_4, sub_15, s1_2, sub_18, v1_2, getitem_5, yy_2, mul_19, sub_16, ge_5, spike_5, mul_16, atan_5, truediv_5, soft_5, sub_17, s2_2, v2_2, sub_20, mul_20, v_2, mul_21, getitem_6, h_3, mul_17, rho_2, add_25, sub_21, ge_6, spike_6, mul_22, atan_6, truediv_6, soft_6, sub_22, s1_3, sub_23, ge_7, spike_7, mul_23, atan_7, truediv_7, soft_7, sub_24, s2_3, sub_25, v1_3, getitem_7, yy_3, mul_26, v2_3, sub_27, mul_27, v_3, mul_24, rho_3], Original ATen: [aten.mul, aten.select, aten.add, aten.sub, aten.ge, aten._to_copy, aten.atan, aten.div, aten.rsub, aten.sigmoid]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_atan_div_ge_mul_rsub_select_sigmoid_sub_0.run(
                arg2_1,
                arg0_1,
                arg3_1,
                arg1_1,
                buf0,
                buf1,
                buf3,
                buf2,
                buf4,
                buf5,
                buf9,
                buf8,
                32768,
                stream=stream0,
            )
            del arg1_1
            buf6 = empty_strided_cuda((131072,), (1,), torch.float32)
            buf7 = empty_strided_cuda((131072,), (1,), torch.float32)
            # Topologically Sorted Source Nodes: [mul, getitem, h, add_1, sub, ge, spike, mul_1, atan, truediv, soft, sub_1, s1, sub_2, ge_1, spike_1, mul_2, atan_1, truediv_1, soft_1, sub_3, s2, mul_7, getitem_2, h_1, ge_2, spike_2, mul_8, atan_2, truediv_2, soft_2, sub_8, s1_1, sub_9, ge_3, spike_3, mul_9, atan_3, truediv_3, soft_3, sub_10, s2_1, mul_14, getitem_4, h_2, add_17, sub_14, ge_4, spike_4, mul_15, atan_4, truediv_4, soft_4, sub_15, s1_2, sub_16, ge_5, spike_5, mul_16, atan_5, truediv_5, soft_5, sub_17, s2_2, mul_21, getitem_6, h_3, ge_6, spike_6, mul_22, atan_6, truediv_6, soft_6, sub_22, s1_3, stack, sub_23, ge_7, spike_7, mul_23, atan_7, truediv_7, soft_7, sub_24, s2_3, stack_1], Original ATen: [aten.mul, aten.select, aten.add, aten.sub, aten.ge, aten._to_copy, aten.atan, aten.div, aten.stack]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_atan_div_ge_mul_select_stack_sub_1.run(
                arg2_1,
                arg0_1,
                arg3_1,
                buf1,
                buf2,
                buf3,
                buf5,
                buf0,
                buf4,
                buf6,
                buf7,
                131072,
                stream=stream0,
            )
            del arg0_1
            del arg2_1
            del arg3_1
            del buf0
            del buf1
            del buf2
            del buf3
            del buf4
            del buf5
        return (
            reinterpret_tensor(buf6, (4, 32768), (32768, 1), 0),
            reinterpret_tensor(buf7, (4, 32768), (32768, 1), 0),
            buf8,
            buf9,
        )


runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided

    arg0_1 = rand_strided((4, 32768), (32768, 1), device="cuda:0", dtype=torch.float32)
    arg1_1 = rand_strided((4, 32768), (32768, 1), device="cuda:0", dtype=torch.float32)
    arg2_1 = rand_strided((32768,), (1,), device="cuda:0", dtype=torch.float32)
    arg3_1 = rand_strided((32768,), (1,), device="cuda:0", dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance

    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main

    args = get_args()
    compiled_module_main(
        "None",
        lambda times, repeat: benchmark_compiled_module(
            args, times=times, repeat=repeat
        ),
    )
