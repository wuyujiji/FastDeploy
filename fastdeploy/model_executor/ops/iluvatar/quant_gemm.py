from typing import Optional

import paddle

try:
    from fastdeploy.model_executor.ops.iluvatar import weight_only_gemm, gemm
except ImportError:
    weight_only_gemm = None
    gemm = None

def weight_only_linear(
    input: paddle.Tensor,
    weight: paddle.Tensor,
    bias: Optional[paddle.Tensor],
    weight_scale: paddle.Tensor,
    weight_dtype="int8",
    act_type="none",
    group_size=-1,
    arch=-1,
):
    input_shape= input.shape
    assert input.ndim < 4 # TODO: remove
    if input.ndim == 3:
        input = input.view([-1, input_shape[-1]])
    if input_shape[-1] == weight.shape[-1]:
        # TN
        transpose_format = "TN"
    elif input_shape[-1] == weight.shape[0]:
        # NN
        transpose_format = "NN"
    else:
        raise NotImplementedError(f"Unspport transpose format: {input.shape} vs {weight.shape}")
    output =  weight_only_gemm(
        input, weight, weight_scale, bias,
        weight_dtype, transpose_format, group_size,
        act_type=act_type
    )
    if len(input_shape) == 3:
        output = output.view([input_shape[0], input_shape[1], output.shape[-1]])
    return output

def siglip_mlp(
    input: paddle.Tensor,
    fc1_weight: paddle.Tensor,
    fc1_bias: Optional[paddle.Tensor],
    fc2_weight: paddle.Tensor,
    fc2_bias: Optional[paddle.Tensor],
    act_type: str,
):
    fc1_output = gemm(input, fc1_weight, fc1_bias, act_type=act_type) 
    output = gemm(fc1_output, fc2_weight, fc2_bias, act_type="none") 
    return output

def ernie_mlp(
    input: paddle.Tensor,
    fc1_weight: paddle.Tensor,
    fc1_weight_scale: paddle.Tensor,
    fc1_bias: Optional[paddle.Tensor],
    fc2_weight: paddle.Tensor,
    fc2_weight_scale: paddle.Tensor,
    fc2_bias: Optional[paddle.Tensor],
    act_type="none",
    weight_dtype="int8",
    transpose_format="TN",
    group_size=-1,
):
    # cuinfer gemm cannot fuse act when bias is None
    if act_type != "none" and fc1_bias is None:
        fc1_bias = paddle.zeros([fc1_weight.shape[0]], dtype=input.dtype)
    fc1_output = weight_only_gemm(input, fc1_weight, fc1_weight_scale, fc1_bias, weight_dtype, transpose_format, group_size, act_type=act_type) 
    output = weight_only_gemm(fc1_output, fc2_weight, fc2_weight_scale, fc2_bias, weight_dtype, transpose_format, group_size, act_type="none")
    return output
